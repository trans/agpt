# Build a per-radix-node precondition sidecar for the AGPT precondition
# strand experiment (notes/seq-len-extension/precondition.md).
#
# For each prefix-trie radix node K, the tool records — across all corpus
# positions where K's prefix ends — the d_pre tokens immediately preceding
# K's start position. At training time the trainer samples one of K's
# instances per fire, reads its d_pre tokens, and feeds them into a small
# GRU encoder whose output is residual-added to K's layer-0 input.
#
# This sidecar is the data substrate that lets the v1 trainer access
# pre-K context without loading the raw corpus itself. Build once per
# (corpus, prefix-trie, d_pre) triple.
#
# Algorithm: walk the forward corpus once via CorpusTrieWalker. For each
# (radix_id, start_pos) the walker emits, capture the d_pre tokens at
# positions [start_pos - d_pre .. start_pos - 1]. Instances with
# start_pos < d_pre are skipped (insufficient preceding context).
#
# Output side-table format (binary, little-endian):
#   magic        u32 = 'PREC' (0x43455250)
#   version      u32 = 1
#   n_radix      u32   number of radix nodes (must match the trie)
#   d_pre        u32   precondition prefix length in tokens
#   n_instances  u64   total instances stored across all nodes
#   skipped      u64   instances skipped (start_pos < d_pre); diagnostic
#   offsets      u32[n_radix + 1]
#                       offsets[k]..offsets[k+1] is K's slice into inst_tokens
#                       (in units of *instances*, not raw tokens)
#   inst_tokens  i32[n_instances * d_pre]
#                       per-instance d_pre tokens in forward (corpus) order
#                       (oldest first, latest immediately before K's start)
#
# Storage on Shakespeare d=16, d_pre=16:
#   ~1M instances * 16 tokens * 4 bytes  ≈ 64 MB inst_tokens
#   ~1.5M nodes * 4 bytes (offsets)      ≈ 6 MB
#   total ≈ 70 MB

require "../agpt"
require "option_parser"

include MicroGPT::AGPT

MAGIC = 0x43455250_u32   # 'PREC'
VERSION = 1_u32

trie_dir = ""
corpus_path = ""
out_path = ""
d_pre = 16
progress_every = 250_000

OptionParser.parse do |p|
  p.banner = "Usage: agpt_build_precondition_sidecar --trie DIR --corpus FILE --out PATH [--d-pre N]"
  p.on("--trie DIR", "Prefix radix-trie directory") { |v| trie_dir = v }
  p.on("--corpus FILE", "Forward-order corpus text file") { |v| corpus_path = v }
  p.on("--out PATH", "Output side-table path") { |v| out_path = v }
  p.on("--d-pre N", "Precondition prefix length [default 16]") { |v| d_pre = v.to_i }
  p.on("-h", "--help") { puts p; exit 0 }
end

raise "--trie required" if trie_dir.empty?
raise "--corpus required" if corpus_path.empty?
raise "--out required" if out_path.empty?
raise "--d-pre must be > 0" if d_pre <= 0

STDERR.puts "[agpt precondition sidecar] d_pre=#{d_pre}"
STDERR.puts "  trie:   #{trie_dir}"
STDERR.puts "  corpus: #{corpus_path}"
STDERR.puts "  out:    #{out_path}"

# Load corpus tokens via CharDataset (matches what the trie was built from).
text = File.read(corpus_path)
dataset = MicroGPT::CharDataset.new(text)
corpus_tokens = dataset.data
STDERR.puts "  loaded #{corpus_tokens.size} tokens (vocab=#{dataset.vocab_size})"

# Load prefix trie + build walker over forward corpus.
reader = RadixTrieReader.new(trie_dir, max_cached: 2)
walker = CorpusTrieWalker.new(reader, corpus_tokens)
n_radix = walker.radix_count
STDERR.puts "  walker: #{n_radix} radix nodes"

# Pass 1: count instances per node (skipping those with start_pos < d_pre).
STDERR.puts ""
STDERR.puts "Pass 1: count instances per node"
instance_counts = Slice(UInt32).new(n_radix, 0_u32)
skipped = 0_i64
total_emitted = 0_i64
t0 = Time.monotonic
walker.walk do |radix_id, start_pos, _terminal_pos|
  if start_pos < d_pre
    skipped += 1
    next
  end
  instance_counts[radix_id] += 1_u32
  total_emitted += 1
  if (total_emitted % progress_every) == 0
    STDERR.puts "  counted #{total_emitted} instances (#{(total_emitted / (Time.monotonic - t0).total_seconds).round(0)} inst/s)"
  end
end
STDERR.puts "  pass 1 done: #{total_emitted} instances counted, #{skipped} skipped (#{(Time.monotonic - t0).total_seconds.round(2)}s)"

# Build offsets via prefix sum.
offsets = Slice(UInt32).new(n_radix + 1, 0_u32)
acc = 0_u32
(0...n_radix).each do |i|
  offsets[i] = acc
  acc += instance_counts[i]
end
offsets[n_radix] = acc
total_instances = acc.to_i64
STDERR.puts "  total instances to write: #{total_instances}"

# Pass 2: scatter d_pre tokens per instance into the flat buffer.
STDERR.puts ""
STDERR.puts "Pass 2: scatter d_pre tokens per instance"
inst_tokens = Slice(Int32).new(total_instances.to_i32 * d_pre, 0)
cursors = Slice(UInt32).new(n_radix, 0_u32)
t0 = Time.monotonic
written = 0_i64
walker.walk do |radix_id, start_pos, _terminal_pos|
  next if start_pos < d_pre
  slot = offsets[radix_id] + cursors[radix_id]
  cursors[radix_id] += 1_u32
  base = slot.to_i64 * d_pre
  d_pre.times do |j|
    src_pos = start_pos - d_pre + j
    inst_tokens[(base + j).to_i32] = corpus_tokens[src_pos]
  end
  written += 1
  if (written % progress_every) == 0
    STDERR.puts "  scattered #{written}/#{total_instances} (#{(written / (Time.monotonic - t0).total_seconds).round(0)} inst/s)"
  end
end
STDERR.puts "  pass 2 done (#{(Time.monotonic - t0).total_seconds.round(2)}s)"

# Sanity: cursors should match instance_counts.
(0...n_radix).each do |i|
  if cursors[i] != instance_counts[i]
    STDERR.puts "WARN: cursor mismatch at radix_id=#{i}: cursor=#{cursors[i]} count=#{instance_counts[i]}"
  end
end

# Write sidecar.
STDERR.puts ""
STDERR.puts "Writing sidecar to #{out_path}..."
t0 = Time.monotonic
File.open(out_path, "wb") do |f|
  f.write_bytes(MAGIC, IO::ByteFormat::LittleEndian)
  f.write_bytes(VERSION, IO::ByteFormat::LittleEndian)
  f.write_bytes(n_radix.to_u32, IO::ByteFormat::LittleEndian)
  f.write_bytes(d_pre.to_u32, IO::ByteFormat::LittleEndian)
  f.write_bytes(total_instances.to_u64, IO::ByteFormat::LittleEndian)
  f.write_bytes(skipped.to_u64, IO::ByteFormat::LittleEndian)
  (0..n_radix).each { |i| f.write_bytes(offsets[i], IO::ByteFormat::LittleEndian) }
  inst_tokens.each { |t| f.write_bytes(t, IO::ByteFormat::LittleEndian) }
end
size_mb = (File.size(out_path) / 1024.0 / 1024.0).round(2)
STDERR.puts "  wrote #{File.size(out_path)} bytes (#{size_mb} MB) in #{(Time.monotonic - t0).total_seconds.round(2)}s"

# Diagnostic summary.
nonzero = 0
max_per_node = 0_u32
(0...n_radix).each do |i|
  c = instance_counts[i]
  if c > 0
    nonzero += 1
    max_per_node = c if c > max_per_node
  end
end
STDERR.puts ""
STDERR.puts "Summary:"
STDERR.puts "  total instances:  #{total_instances}"
STDERR.puts "  skipped:          #{skipped} (start_pos < d_pre)"
STDERR.puts "  nodes with >=1:   #{nonzero} / #{n_radix} (#{(100.0 * nonzero / n_radix).round(1)}%)"
STDERR.puts "  max per node:     #{max_per_node}"
STDERR.puts "  d_pre:            #{d_pre}"
STDERR.puts "[agpt precondition sidecar] done."
