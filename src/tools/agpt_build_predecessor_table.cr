require "../agpt"
require "../agpt/radix_trie_reader"
require "../agpt/corpus_trie_walker"
require "option_parser"

# Build the predecessor table for cap-recurrence.
#
# For each radix node K, record the set of (K_prev, count) pairs where
# K_prev is the trie node for the d-character window ending at the start
# of K's occurrence in the corpus. K_prev is the recurrent "incoming
# state" anchor — at training time the model will fetch h_cap[K_prev]
# and aggregate as h_in for K's fire.
#
# Algorithm:
#   Pass 1: walk corpus, record `k_at_pos[s] = radix_id` for the trie
#           node whose path matches corpus[s..s+d-1] in full (depth==d).
#           Positions where no full d-window matches get -1.
#   Pass 2: walk again. For each yielded (kid, s, terminal_pos), if
#           s >= d, look up k_at_pos[s - d] and increment
#           counts[kid][k_at_pos[s - d]] by 1.
#
# Output format (binary, little-endian):
#   uint32  magic = 'PRED' (0x44455250)
#   uint32  version = 1
#   uint32  radix_count
#   uint32  d_window
#   uint64  total_entries
#   uint64  offsets[radix_count + 1]   — CSR start index per K
#   uint32  predecessor_ids[total_entries]
#   uint32  counts[total_entries]
#
# See: notes/seq-len-extension/cap-recurrence-design.md (Q2 aggregation).

include MicroGPT::AGPT

trie_dir = ""
corpus_path = ""
out_path = ""
d_window = 16
max_cached = 64

OptionParser.parse do |p|
  p.banner = "Usage: agpt_build_predecessor_table --trie DIR --corpus PATH --out PATH [options]"
  p.on("--trie DIR", "Radix-trie directory") { |v| trie_dir = v }
  p.on("--corpus PATH", "Character-level corpus file") { |v| corpus_path = v }
  p.on("--out PATH", "Output table path") { |v| out_path = v }
  p.on("--d-window N", "Recurrent window size in chars (default 16)") { |v| d_window = v.to_i }
  p.on("--max-cached N", "Reader LRU cache size (default 64)") { |v| max_cached = v.to_i }
  p.on("-h", "--help", "") { puts p; exit 0 }
end

if trie_dir.empty? || corpus_path.empty? || out_path.empty?
  STDERR.puts "missing required args; see --help"
  exit 1
end

unless File.exists?(corpus_path)
  STDERR.puts "corpus not found: #{corpus_path}"
  exit 1
end

# Tokenize the corpus with the SAME char→id mapping used by the radix
# builder. CharDataset sorts unique chars alphabetically and assigns
# IDs 0..vocab_size-1.
t0 = Time.monotonic
text = File.read(corpus_path)
dataset = MicroGPT::CharDataset.new(text)
corpus = dataset.data
n_corpus = corpus.size
puts "[pred] corpus: #{n_corpus} tokens, vocab=#{dataset.vocab_size}  (#{(Time.monotonic - t0).total_seconds.round(2)}s)"

# Load trie.
t1 = Time.monotonic
reader = RadixTrieReader.new(trie_dir, max_cached: max_cached)
walker = CorpusTrieWalker.new(reader, corpus)
radix_count = walker.radix_count
d_max = walker.d_max
puts "[pred] trie: #{radix_count} radix nodes, d_max=#{d_max}  " \
     "(#{(Time.monotonic - t1).total_seconds.round(2)}s)"

if d_window > d_max
  STDERR.puts "ERROR: --d-window #{d_window} exceeds trie d_max #{d_max}"
  exit 1
end

# --------------------------------------------------------------------
# Pass 1: build k_at_pos[s] = radix_id of the depth-d_window node
# whose path matches corpus[s..s+d_window-1] exactly.
# --------------------------------------------------------------------
puts "[pred] pass 1: building k_at_pos..."
t2 = Time.monotonic
k_at_pos = Slice(Int32).new(n_corpus, -1)
n_filled_p1 = 0_i64

walker.walk do |kid, s, terminal_pos|
  depth = terminal_pos - s + 1
  if depth == d_window
    k_at_pos[s] = kid
    n_filled_p1 += 1
  end
end

dt2 = (Time.monotonic - t2).total_seconds
filled_pct = 100.0 * n_filled_p1.to_f / n_corpus.to_f
puts "[pred] pass 1 done: #{n_filled_p1}/#{n_corpus} (#{filled_pct.round(1)}%) " \
     "positions have full d=#{d_window} predecessor  (#{dt2.round(2)}s)"

# --------------------------------------------------------------------
# Pass 2: accumulate (K, K_prev) pair counts.
# --------------------------------------------------------------------
puts "[pred] pass 2: accumulating predecessor pairs..."
t3 = Time.monotonic

# Array of Hash, indexed by K (radix_id). Inner hash: K_prev → count.
# Slot 0 (virtual root) stays empty; trie nodes start at 1.
pred_counts = Array(Hash(Int32, Int32)?).new(radix_count) { nil }

n_pairs_total = 0_i64
walker.walk do |kid, s, terminal_pos|
  next if s < d_window
  k_prev = k_at_pos[s - d_window]
  next if k_prev < 0
  h = pred_counts[kid]
  if h.nil?
    h = Hash(Int32, Int32).new
    pred_counts[kid] = h
  end
  h[k_prev] = (h[k_prev]? || 0) + 1
  n_pairs_total += 1
end

dt3 = (Time.monotonic - t3).total_seconds
n_unique_pairs = 0_i64
n_filled_k = 0_i64
pred_counts.each do |h|
  next if h.nil?
  n_filled_k += 1
  n_unique_pairs += h.size
end
avg_pred_per_k = n_filled_k > 0 ? n_unique_pairs.to_f / n_filled_k.to_f : 0.0
puts "[pred] pass 2 done: #{n_pairs_total} total pair instances, " \
     "#{n_unique_pairs} unique (K, K_prev) pairs across #{n_filled_k} K's " \
     "(avg #{avg_pred_per_k.round(2)} preds/K)  (#{dt3.round(2)}s)"

# --------------------------------------------------------------------
# Emit CSR-format binary side-table.
# --------------------------------------------------------------------
puts "[pred] writing #{out_path}..."
t4 = Time.monotonic
File.open(out_path, "wb") do |f|
  # Header
  f.write_bytes(0x44455250_u32, IO::ByteFormat::LittleEndian)  # 'PRED'
  f.write_bytes(1_u32, IO::ByteFormat::LittleEndian)            # version
  f.write_bytes(radix_count.to_u32, IO::ByteFormat::LittleEndian)
  f.write_bytes(d_window.to_u32, IO::ByteFormat::LittleEndian)
  f.write_bytes(n_unique_pairs.to_u64, IO::ByteFormat::LittleEndian)

  # Offsets pass — we need to compute offsets[K+1] = offsets[K] + len(K)
  offsets = Slice(UInt64).new(radix_count + 1, 0_u64)
  cum = 0_u64
  (0...radix_count).each do |k|
    offsets[k] = cum
    h = pred_counts[k]
    cum += h.nil? ? 0_u64 : h.size.to_u64
  end
  offsets[radix_count] = cum

  (0..radix_count).each { |k| f.write_bytes(offsets[k], IO::ByteFormat::LittleEndian) }

  # Body: per-K, emit (K_prev, count) pairs in K_prev-sorted order
  # for determinism.
  pred_ids = Slice(UInt32).new(n_unique_pairs.to_i32, 0_u32)
  pred_cnt = Slice(UInt32).new(n_unique_pairs.to_i32, 0_u32)
  write_pos = 0
  (0...radix_count).each do |k|
    h = pred_counts[k]
    next if h.nil?
    keys = h.keys.sort
    keys.each do |kp|
      pred_ids[write_pos] = kp.to_u32
      pred_cnt[write_pos] = h[kp].to_u32
      write_pos += 1
    end
  end
  (0...n_unique_pairs.to_i32).each { |i| f.write_bytes(pred_ids[i], IO::ByteFormat::LittleEndian) }
  (0...n_unique_pairs.to_i32).each { |i| f.write_bytes(pred_cnt[i], IO::ByteFormat::LittleEndian) }
end

dt4 = (Time.monotonic - t4).total_seconds
file_size = File.size(out_path)
puts "[pred] wrote #{file_size} bytes (#{(file_size / 1024.0 / 1024.0).round(1)} MB) " \
     "in #{dt4.round(2)}s"
puts "[pred] done."
