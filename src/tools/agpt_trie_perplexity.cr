require "../agpt/radix_trie_reader"
require "option_parser"

# Trie-only perplexity evaluator.
#
# Walks the prefix radix trie with held-out context (no model). At each
# position, finds the deepest valid match, reads the empirical count
# distribution over next chars, computes -log P(actual_next_char).
# Backs off to shorter context when the deepest match has zero count
# for the target. Unigram fallback if everything backs off to root.
#
# Output format mirrors lib/microgpt's perplexity tool so the numbers
# are directly comparable to your trained-model PPLs.
#
# Usage:
#   bin/agpt_trie_perplexity \
#     --trie /tmp/gutenberg_5m_baseline_d16_radix \
#     --file data/gutenberg_5m.txt \
#     --seq-len 16 \
#     --max-positions 4096

include MicroGPT::AGPT

trie_dir = ""
corpus_path = ""
seq_len = 16
max_positions = 4096
laplace_eps = 1e-6
quiet = false

OptionParser.parse do |p|
  p.banner = "Usage: agpt_trie_perplexity --trie DIR --file PATH [options]"
  p.on("--trie DIR", "Prefix radix-trie directory") { |v| trie_dir = v }
  p.on("--file PATH", "Held-out text file") { |v| corpus_path = v }
  p.on("--seq-len N", "Max context length (default 16)") { |v| seq_len = v.to_i }
  p.on("--max-positions N", "Limit positions scored (default 4096; 0 = all)") { |v| max_positions = v.to_i }
  p.on("--laplace-eps F", "Laplace smoothing epsilon (default 1e-6)") { |v| laplace_eps = v.to_f }
  p.on("--quiet", "Suppress progress output") { quiet = true }
  p.on("-h", "--help", "") { puts p; exit 0 }
end

abort "missing --trie" if trie_dir.empty?
abort "missing --file" if corpus_path.empty?

reader = RadixTrieReader.new(trie_dir, max_cached: 64)
vocab_size = reader.vocab_size

# Build vocab the same way parrot does — chars from the corpus, sorted
chars = File.read(corpus_path).chars.to_set.to_a.sort
char_to_id = {} of Char => Int32
chars.each_with_index { |c, i| char_to_id[c] = i }

# Index children by (parent_id, first_token) for fast walk
child_by_token = Hash(Int32, Hash(Int32, RadixTrieReader::LoadedRecord)).new do |h, k|
  h[k] = {} of Int32 => RadixTrieReader::LoadedRecord
end
reader.each do |r|
  child_by_token[r.parent_id][r.edge_tokens[0]] = r
end

# Walk from root following `context` tokens. Returns (record, edge_pos, depth_matched).
#   record: the trie node we ended at (nil if no walk possible — context[0] not in vocab/no child)
#   edge_pos: position within record's edge (0..edge_tokens.size). If = edge_tokens.size, we
#             are at the END of the edge (boundary), and record.counts is the distribution over
#             next chars (or root counts if record is nil at root).
#             If < edge_tokens.size, we are MID-EDGE; next char is forced to be edge_tokens[edge_pos].
#   depth_matched: how many context tokens we matched in total.
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
    # How many of edge match context starting at pos?
    consume = 0
    while consume < edge.size && pos + consume < context.size && edge[consume] == context[pos + consume]
      consume += 1
    end
    last = kid
    edge_pos = consume
    pos += consume
    # If we consumed entire edge AND there's more context, descend; otherwise stop.
    break unless consume == edge.size && pos < context.size
    parent_id = kid.id
  end
  {last, edge_pos, pos}
end

# Compute P(target | distribution at deepest match) with Laplace smoothing.
#   - If walk ended mid-edge: distribution is degenerate on edge_tokens[edge_pos].
#     P(target) = 1 - eps*(V-1) if target == that, else eps  (effectively).
#   - If walk ended at edge boundary: use record.counts as distribution.
#   - If walk returned nil (couldn't even start): use uniform.
def prob_target(record : RadixTrieReader::LoadedRecord?,
                edge_pos : Int32,
                target : Int32,
                vocab_size : Int32,
                eps : Float64) : Float64
  if record.nil?
    return 1.0 / vocab_size
  end
  if edge_pos < record.edge_tokens.size
    # Mid-edge: forced next-char is edge_tokens[edge_pos]
    forced = record.edge_tokens[edge_pos]
    if forced == target
      return 1.0 - eps * (vocab_size - 1)
    else
      return eps
    end
  end
  # At edge boundary — use record.counts
  total = 0_i64
  record.counts.each { |e| total += e[1].to_i64 }
  if total == 0
    return 1.0 / vocab_size
  end
  target_count = 0_i32
  record.counts.each do |e|
    if e[0] == target
      target_count = e[1]
      break
    end
  end
  smoothed = (target_count.to_f64 + eps) / (total.to_f64 + eps * vocab_size)
  smoothed
end

# Read corpus, tokenize
text = File.read(corpus_path)
tokens = Array(Int32).new(text.size)
text.each_char do |c|
  tid = char_to_id[c]?
  tokens << (tid ? tid : 0)
end

# Score positions p in [seq_len, tokens.size - 1]. Sample evenly if max_positions > 0.
start_pos = seq_len
end_pos = tokens.size - 1
n_avail = end_pos - start_pos
n_score = (max_positions > 0 && max_positions < n_avail) ? max_positions : n_avail
stride = (n_avail.to_f64 / n_score.to_f64).clamp(1.0, Float64::MAX)

STDERR.puts "Trie: #{trie_dir}" unless quiet
STDERR.puts "Corpus: #{corpus_path} (#{tokens.size} tokens)" unless quiet
STDERR.puts "Vocab: #{vocab_size}, seq-len: #{seq_len}, scoring #{n_score} positions (stride #{stride.round(2)})" unless quiet

total_nll = 0.0
n_scored = 0
n_backoffs = 0
n_midedge = 0
n_root_fallback = 0
t0 = Time.monotonic

n_score.times do |i|
  p = start_pos + (i.to_f64 * stride).to_i
  break if p >= end_pos
  context = tokens[Math.max(0, p - seq_len)...p]
  target = tokens[p]

  # Initial walk with full context
  record, edge_pos, _matched = walk(context, child_by_token)
  prob = prob_target(record, edge_pos, target, vocab_size, laplace_eps)

  # Backoff: if prob is below uniform AND we have context to drop, try shorter context.
  # We only back off when the forced-edge / count-distribution truly excludes target.
  ctx = context
  while prob <= laplace_eps && ctx.size > 0
    ctx = ctx[1..]
    record, edge_pos, _matched = walk(ctx, child_by_token)
    prob = prob_target(record, edge_pos, target, vocab_size, laplace_eps)
    n_backoffs += 1
  end

  if record.nil?
    n_root_fallback += 1
  elsif edge_pos < (record.edge_tokens.size)
    n_midedge += 1
  end

  total_nll -= Math.log(prob)
  n_scored += 1
end

elapsed = (Time.monotonic - t0).total_seconds
mean_nll = total_nll / n_scored
ppl = Math.exp(mean_nll)
bpc = mean_nll / Math.log(2.0)

puts "Positions scored:   #{n_scored}"
puts "Mean per-token NLL: #{mean_nll.round(6)} nats"
puts "Perplexity:         #{ppl.round(4)}"
puts "Bits per character: #{bpc.round(4)} bpc"
puts "Elapsed:            #{elapsed.round(2)}s (#{(n_scored / elapsed).round(0)} pos/sec)"
puts ""
puts "Diagnostics:"
puts "  Backoff invocations: #{n_backoffs}"
puts "  Mid-edge positions:  #{n_midedge}  (forced-next-char regime)"
puts "  Root-only fallback:  #{n_root_fallback}  (no walk possible)"
