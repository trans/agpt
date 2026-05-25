# Dump every radix-trie node's context string (the chars from root to that node)
# as one line per node, sorted by radix_id. Used to feed downstream tools that
# need to compute per-node distributions (e.g. KN-smoothed soft targets).
#
# Format: one context per line, raw chars (no separator). Empty line = root.
# Real corpus spaces are written verbatim; downstream tools may substitute.
#
# Usage:
#   bin/dump_trie_contexts <radix_dir> [--vocab data/input.txt] > contexts.txt

require "../agpt"

if ARGV.size < 1
  STDERR.puts "Usage: dump_trie_contexts <radix_dir> [--vocab <file>]"
  exit 1
end

radix_dir = ARGV[0]
vocab_path = "data/input.txt"
i = 1
while i < ARGV.size
  case ARGV[i]
  when "--vocab"
    vocab_path = ARGV[i + 1]
    i += 2
  else
    i += 1
  end
end

# Build id → char mapping by reading the corpus and computing the same sorted-unique
# tokenization the trie used.
text = File.read(vocab_path)
chars = text.chars.to_set.to_a
chars.sort!
id_to_char = chars

STDERR.puts "Loading radix trie: #{radix_dir}"
reader = MicroGPT::AGPT::RadixTrieReader.new(radix_dir, max_cached: 256)
STDERR.puts "  radix_count = #{reader.radix_count}"
STDERR.puts "  vocab_size  = #{reader.vocab_size}"
STDERR.puts "  depth_file_count = #{reader.depth_file_count}"

# We need (parent_id, edge_tokens) for every node, then reconstruct each context
# by walking parent pointers. Build the lookup table by streaming through every node.

# id → (parent_id, edge_tokens)
parent_map = Hash(Int32, Int32).new
edge_map = Hash(Int32, Array(Int32)).new

STDERR.puts "Pass 1: collect parent/edge info..."
reader.each do |rec|
  parent_map[rec.id] = rec.parent_id
  edge_map[rec.id] = rec.edge_tokens
end
STDERR.puts "  collected #{parent_map.size} nodes"

# Sort by id for stable output
ids = parent_map.keys.sort

STDERR.puts "Pass 2: reconstruct contexts and emit..."
emitted = 0
ids.each do |id|
  # Walk back from id to root, collecting edge_tokens at each level
  segments = [] of Array(Int32)
  cur = id
  while cur >= 0 && parent_map.has_key?(cur)
    segments << edge_map[cur]
    cur = parent_map[cur]
    break if cur == -1 || cur == 0  # 0 is root in our trie convention
  end
  # Reverse to get root-to-node order
  segments.reverse!

  # Convert tokens to chars. Skip newlines (the KenLM tokenization treats them
  # as sentence boundaries and they break the line-per-context output format).
  io = String::Builder.new
  segments.each do |seg|
    seg.each do |tok|
      ch = id_to_char[tok]
      next if ch == '\n'
      io << ch
    end
  end
  puts io.to_s
  emitted += 1
  if emitted % 200_000 == 0
    STDERR.puts "  emitted #{emitted}/#{ids.size}"
  end
end
STDERR.puts "Done. Emitted #{emitted} contexts."
