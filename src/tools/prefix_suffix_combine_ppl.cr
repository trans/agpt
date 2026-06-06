require "../agpt/radix_trie_reader"
require "option_parser"

# Joint-distribution probe for the prefix/suffix masked-LM combiner hypothesis.
#
# For each held-out corpus position p with target t_p, compute CE/PPL using:
#   - P-only:   smoothed distribution from forward trie at corpus[p-D..p-1]
#   - S-only:   smoothed distribution from reverse trie at reverse(corpus[p+1..p+D])
#   - Mixture:  λ·P_P + (1-λ)·P_S, swept over λ ∈ {0.1, 0.3, 0.5, 0.7, 0.9}
#   - Product:  P_P(t)·P_S(t), renormalized (log-linear at α=1)
#   - GeoMix:   P_P(t)^α · P_S(t)^(1-α), renormalized, α=0.5
#
# The probe answers the necessary condition for a masked-LM auxiliary loss to
# help: does the joint (prefix-context, suffix-context) distribution carry signal
# beyond what either side alone provides at the n-gram level?
#
# All variants use the same smoothing scheme: target-aware backoff per side
# (drop oldest char until prob(target) > eps or context empty), then read the
# full smoothed distribution at the deepest valid match. P and S back off
# independently — each side gets its deepest match where the trie has any
# signal on the target. Combined variants are computed from those two
# distributions. The first-cut numbers without backoff were dominated by
# mid-edge divergence (~47% of held-out positions hit a radix edge that
# forces the wrong char), making relative comparison meaningless.
#
# Usage:
#   bin/prefix_suffix_combine_ppl \
#     --prefix-trie /tmp/probe_P_radix \
#     --suffix-trie /tmp/probe_S_radix \
#     --file data/.splits/<hash>/heldout_corpus.txt \
#     --vocab-file data/input.txt \
#     --seq-len 16 \
#     --max-positions 8192

include MicroGPT::AGPT

prefix_trie_dir = ""
suffix_trie_dir = ""
corpus_path = ""
vocab_path = ""
seq_len = 16
max_positions = 8192
laplace_eps = 1e-6
quiet = false

OptionParser.parse do |p|
  p.banner = "Usage: prefix_suffix_combine_ppl --prefix-trie DIR --suffix-trie DIR --file PATH --vocab-file PATH [options]"
  p.on("--prefix-trie DIR", "Forward radix-trie directory") { |v| prefix_trie_dir = v }
  p.on("--suffix-trie DIR", "Reverse (suffix) radix-trie directory") { |v| suffix_trie_dir = v }
  p.on("--file PATH", "Held-out text file") { |v| corpus_path = v }
  p.on("--vocab-file PATH", "Build vocab from this file (defaults to --file)") { |v| vocab_path = v }
  p.on("--seq-len N", "Max context length per side (default 16)") { |v| seq_len = v.to_i }
  p.on("--max-positions N", "Limit positions scored (default 8192; 0 = all)") { |v| max_positions = v.to_i }
  p.on("--laplace-eps F", "Laplace smoothing epsilon (default 1e-6)") { |v| laplace_eps = v.to_f }
  p.on("--quiet", "Suppress progress output") { quiet = true }
  p.on("-h", "--help", "") { puts p; exit 0 }
end

abort "missing --prefix-trie" if prefix_trie_dir.empty?
abort "missing --suffix-trie" if suffix_trie_dir.empty?
abort "missing --file" if corpus_path.empty?

vocab_path = corpus_path if vocab_path.empty?

prefix_reader = RadixTrieReader.new(prefix_trie_dir, max_cached: 64)
suffix_reader = RadixTrieReader.new(suffix_trie_dir, max_cached: 64)
vocab_size = prefix_reader.vocab_size
if suffix_reader.vocab_size != vocab_size
  abort "vocab size mismatch: prefix=#{vocab_size}, suffix=#{suffix_reader.vocab_size}"
end

chars = File.read(vocab_path).chars.to_set.to_a.sort
char_to_id = {} of Char => Int32
chars.each_with_index { |c, i| char_to_id[c] = i }
if chars.size != vocab_size
  STDERR.puts "WARN: vocab-file derived #{chars.size} chars but tries have vocab_size=#{vocab_size}"
end

# Index children by (parent_id, first_token) for fast walk
def build_child_index(reader : RadixTrieReader) : Hash(Int32, Hash(Int32, RadixTrieReader::LoadedRecord))
  idx = Hash(Int32, Hash(Int32, RadixTrieReader::LoadedRecord)).new do |h, k|
    h[k] = {} of Int32 => RadixTrieReader::LoadedRecord
  end
  reader.each do |r|
    idx[r.parent_id][r.edge_tokens[0]] = r
  end
  idx
end

prefix_idx = build_child_index(prefix_reader)
suffix_idx = build_child_index(suffix_reader)

# Walk from root following `context` tokens.
def walk(context : Array(Int32),
         child_by_token : Hash(Int32, Hash(Int32, RadixTrieReader::LoadedRecord))) : {RadixTrieReader::LoadedRecord?, Int32, Int32}
  parent_id = 0
  pos = 0
  last : RadixTrieReader::LoadedRecord? = nil
  edge_pos = 0
  while pos < context.size
    kids = child_by_token[parent_id]?
    break if kids.nil?
    kid = kids[context[pos]]?
    break if kid.nil?
    edge = kid.edge_tokens
    consume = 0
    while consume < edge.size && pos + consume < context.size && edge[consume] == context[pos + consume]
      consume += 1
    end
    last = kid
    edge_pos = consume
    pos += consume
    break unless consume == edge.size && pos < context.size
    parent_id = kid.id
  end
  {last, edge_pos, pos}
end

# Per-target probability at the walk's final node (used for backoff decisions).
def prob_target(record : RadixTrieReader::LoadedRecord?,
                edge_pos : Int32,
                target : Int32,
                vocab_size : Int32,
                eps : Float64) : Float64
  if record.nil?
    return 1.0 / vocab_size
  end
  if edge_pos < record.edge_tokens.size
    forced = record.edge_tokens[edge_pos]
    return forced == target ? 1.0 - eps * (vocab_size - 1) : eps
  end
  total = 0_i64
  record.counts.each { |e| total += e[1].to_i64 }
  return 1.0 / vocab_size if total == 0
  tcount = 0_i32
  record.counts.each do |e|
    if e[0] == target
      tcount = e[1]
      break
    end
  end
  (tcount.to_f64 + eps) / (total.to_f64 + eps * vocab_size)
end

# Target-aware backoff. Walk full context; if prob(target) <= eps, drop the
# OLDEST element (front for prefix, back for reversed-suffix — caller controls
# direction by what it passes in) and retry. Returns final (record, edge_pos)
# and the final context length actually used.
def walk_with_backoff(context : Array(Int32),
                      target : Int32,
                      child_by_token : Hash(Int32, Hash(Int32, RadixTrieReader::LoadedRecord)),
                      vocab_size : Int32,
                      eps : Float64) : {RadixTrieReader::LoadedRecord?, Int32, Int32}
  ctx = context
  rec, ep, _matched = walk(ctx, child_by_token)
  prob = prob_target(rec, ep, target, vocab_size, eps)
  while prob <= eps && ctx.size > 0
    ctx = ctx[1..]
    rec, ep, _matched = walk(ctx, child_by_token)
    prob = prob_target(rec, ep, target, vocab_size, eps)
  end
  {rec, ep, ctx.size}
end

# Smoothed distribution over vocab from the walk's final node.
# - nil record           → uniform
# - mid-edge             → degenerate on forced token
# - boundary             → Laplace-smoothed empirical counts
def dist_from_walk(record : RadixTrieReader::LoadedRecord?,
                   edge_pos : Int32,
                   vocab_size : Int32,
                   eps : Float64) : Array(Float64)
  dist = Array(Float64).new(vocab_size, 0.0)
  if record.nil?
    inv = 1.0 / vocab_size
    vocab_size.times { |i| dist[i] = inv }
    return dist
  end
  if edge_pos < record.edge_tokens.size
    forced = record.edge_tokens[edge_pos]
    eps_off = eps
    main = 1.0 - eps_off * (vocab_size - 1)
    vocab_size.times { |i| dist[i] = (i == forced) ? main : eps_off }
    return dist
  end
  total = 0_i64
  record.counts.each { |e| total += e[1].to_i64 }
  if total == 0
    inv = 1.0 / vocab_size
    vocab_size.times { |i| dist[i] = inv }
    return dist
  end
  denom = total.to_f64 + eps * vocab_size
  vocab_size.times { |i| dist[i] = eps / denom }
  record.counts.each do |e|
    tid = e[0]
    cnt = e[1]
    dist[tid] = (cnt.to_f64 + eps) / denom
  end
  dist
end

text = File.read(corpus_path)
tokens = Array(Int32).new(text.size)
text.each_char do |c|
  tid = char_to_id[c]?
  tokens << (tid ? tid : 0)
end

# Score positions p in [seq_len, tokens.size - seq_len - 1] so that both sides
# have a full window. Sample evenly via stride if max_positions < available.
start_pos = seq_len
end_pos = tokens.size - seq_len - 1
n_avail = end_pos - start_pos
n_score = (max_positions > 0 && max_positions < n_avail) ? max_positions : n_avail
stride = (n_avail.to_f64 / n_score.to_f64).clamp(1.0, Float64::MAX)

STDERR.puts "Prefix trie: #{prefix_trie_dir}" unless quiet
STDERR.puts "Suffix trie: #{suffix_trie_dir}" unless quiet
STDERR.puts "Corpus: #{corpus_path} (#{tokens.size} tokens)" unless quiet
STDERR.puts "Vocab: #{vocab_size}, seq-len: #{seq_len}, scoring #{n_score} positions (stride #{stride.round(2)})" unless quiet

# Accumulators
nll_p_only = 0.0
nll_s_only = 0.0
nll_product = 0.0
nll_geo = 0.0
mix_lambdas = [0.1, 0.3, 0.5, 0.7, 0.9]
nll_mix = Array(Float64).new(mix_lambdas.size, 0.0)
n_scored = 0
n_p_root = 0
n_s_root = 0
n_p_midedge = 0
n_s_midedge = 0
sum_p_ctx_used = 0
sum_s_ctx_used = 0
t0 = Time.monotonic

# Reusable scratch buffer for combined distribution
combined = Array(Float64).new(vocab_size, 0.0)

n_score.times do |i|
  p = start_pos + (i.to_f64 * stride).to_i
  break if p >= end_pos
  target = tokens[p]

  left_ctx = tokens[Math.max(0, p - seq_len)...p]
  right_raw = tokens[(p + 1)..Math.min(tokens.size - 1, p + seq_len)]
  right_ctx_reversed = right_raw.reverse

  rec_p, ep_p, p_ctx_used = walk_with_backoff(left_ctx, target, prefix_idx, vocab_size, laplace_eps)
  rec_s, ep_s, s_ctx_used = walk_with_backoff(right_ctx_reversed, target, suffix_idx, vocab_size, laplace_eps)
  sum_p_ctx_used += p_ctx_used
  sum_s_ctx_used += s_ctx_used

  dist_p = dist_from_walk(rec_p, ep_p, vocab_size, laplace_eps)
  dist_s = dist_from_walk(rec_s, ep_s, vocab_size, laplace_eps)

  prob_p = dist_p[target]
  prob_s = dist_s[target]
  nll_p_only -= Math.log(prob_p)
  nll_s_only -= Math.log(prob_s)

  # Product (renormalized)
  total_prod = 0.0
  vocab_size.times { |t| total_prod += dist_p[t] * dist_s[t] }
  if total_prod > 0
    prob_prod = (dist_p[target] * dist_s[target]) / total_prod
  else
    prob_prod = 1.0 / vocab_size
  end
  nll_product -= Math.log(prob_prod)

  # Geometric mean at α=0.5 (renormalized)
  total_geo = 0.0
  vocab_size.times { |t| total_geo += Math.sqrt(dist_p[t] * dist_s[t]) }
  if total_geo > 0
    prob_geo = Math.sqrt(dist_p[target] * dist_s[target]) / total_geo
  else
    prob_geo = 1.0 / vocab_size
  end
  nll_geo -= Math.log(prob_geo)

  # Linear mixtures
  mix_lambdas.each_with_index do |lam, idx|
    prob_mix = lam * prob_p + (1.0 - lam) * prob_s
    nll_mix[idx] -= Math.log(prob_mix)
  end

  n_p_root += 1 if rec_p.nil?
  n_s_root += 1 if rec_s.nil?
  if !rec_p.nil? && ep_p < rec_p.edge_tokens.size
    n_p_midedge += 1
  end
  if !rec_s.nil? && ep_s < rec_s.edge_tokens.size
    n_s_midedge += 1
  end

  n_scored += 1
end

elapsed = (Time.monotonic - t0).total_seconds
log2 = Math.log(2.0)

def report(name : String, total_nll : Float64, n : Int32)
  mean = total_nll / n
  printf "  %-22s  mean_NLL=%.4f  PPL=%.4f  BPC=%.4f\n", name, mean, Math.exp(mean), mean / Math.log(2.0)
end

puts ""
puts "Positions scored: #{n_scored}  (elapsed #{elapsed.round(2)}s, #{(n_scored / elapsed).round(0)} pos/sec)"
puts ""
puts "Individual sides:"
report("P-only (left ctx)", nll_p_only, n_scored)
report("S-only (right ctx)", nll_s_only, n_scored)
puts ""
puts "Combinations:"
report("Product (renorm)", nll_product, n_scored)
report("GeoMix α=0.5", nll_geo, n_scored)
mix_lambdas.each_with_index do |lam, idx|
  report("Mix λ=#{lam}", nll_mix[idx], n_scored)
end
puts ""
puts "Diagnostics:"
puts "  P walk: root-only=#{n_p_root}  mid-edge=#{n_p_midedge}  mean_ctx_used=#{(sum_p_ctx_used.to_f64 / n_scored).round(2)}"
puts "  S walk: root-only=#{n_s_root}  mid-edge=#{n_s_midedge}  mean_ctx_used=#{(sum_s_ctx_used.to_f64 / n_scored).round(2)}"
