require "../agpt/radix_trie_reader"
require "option_parser"

# Inspect a VTRE side-table — spot-check that per-cap composite distributions
# are sane (sum to 1, non-degenerate entropy, top tokens make sense for the
# cap's corpus context).
#
# Loads the VTRE file, the source trie (for cap path reconstruction), and a
# vocab file (for token→char decoding). For N sampled caps, prints:
#   - radix_id, decoded cap path, cap edge_tokens
#   - per tunnel position: top-K (char, prob), entropy, sum-of-probs
#
# Usage:
#   bin/agpt_inspect_virtual_tree --trie /tmp/agpt_input_d32_radix \
#     --table /tmp/agpt_vtree_d32_e3.bin \
#     --vocab data/input.txt --n 8 --seed 7

include MicroGPT::AGPT

trie_dir = ""
table_path = ""
vocab_path = "data/input.txt"
n_inspect = 8
seed = 0_u64
specific_id = -1

OptionParser.parse do |p|
  p.banner = "Usage: agpt_inspect_virtual_tree --trie DIR --table PATH [options]"
  p.on("--trie DIR", "Source radix-trie directory") { |v| trie_dir = v }
  p.on("--table PATH", "VTRE side-table to inspect") { |v| table_path = v }
  p.on("--vocab PATH", "Vocab text (default data/input.txt)") { |v| vocab_path = v }
  p.on("--n N", "Number of caps to sample (default 8)") { |v| n_inspect = v.to_i }
  p.on("--seed N", "Sampling seed (default 0)") { |v| seed = v.to_u64 }
  p.on("--id ID", "Inspect a specific cap radix_id instead of sampling") { |v| specific_id = v.to_i }
  p.on("-h", "--help", "") { puts p; exit 0 }
end

abort "missing --trie" if trie_dir.empty?
abort "missing --table" if table_path.empty?

reader = RadixTrieReader.new(trie_dir, max_cached: 64)
n_radix = reader.radix_count
STDERR.puts "Loaded trie: #{trie_dir} (#{n_radix} nodes)"

# Build path-reconstruction index.
record_by_id = {} of Int32 => RadixTrieReader::LoadedRecord
reader.each { |r| record_by_id[r.id] = r }

def full_path(rec : RadixTrieReader::LoadedRecord,
              record_by_id : Hash(Int32, RadixTrieReader::LoadedRecord)) : Array(Int32)
  segs = [] of Array(Int32)
  cur = rec
  loop do
    segs << cur.edge_tokens
    break if cur.parent_id == 0
    parent = record_by_id[cur.parent_id]?
    raise "missing parent" if parent.nil?
    cur = parent
  end
  out = [] of Int32
  segs.reverse_each { |s| s.each { |t| out << t } }
  out
end

# Vocab.
chars = File.read(vocab_path).chars.to_set.to_a.sort
char_of = chars
STDERR.puts "Vocab: #{chars.size} chars from #{vocab_path}"

# Parse VTRE.
io = File.open(table_path, "rb")
magic = io.read_bytes(UInt32, IO::ByteFormat::LittleEndian)
abort "bad magic 0x#{magic.to_s(16)}" unless magic == 0x45525456_u32  # 'VTRE'
version = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
abort "unsupported version #{version}" unless version == 1
table_n_radix = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
table_vocab = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
table_top_k = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
expansion_depth = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)

abort "n_radix mismatch" unless table_n_radix == n_radix
STDERR.puts "VTRE: n_radix=#{table_n_radix} vocab=#{table_vocab} top_k=#{table_top_k} expansion_depth=#{expansion_depth}"

n_slots = n_radix * expansion_depth
offsets = Array(Int32).new(n_slots) { io.read_bytes(Int32, IO::ByteFormat::LittleEndian) }
lengths = Array(Int32).new(n_slots) { io.read_bytes(Int32, IO::ByteFormat::LittleEndian) }

# Read entries lazily — compute total then read them all into a flat array.
total_entries = 0
n_slots.times { |i| total_entries += lengths[i] }
all_tokens = Array(Int32).new(total_entries)
all_probs = Array(Float32).new(total_entries)
total_entries.times do
  all_tokens << io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
  all_probs << io.read_bytes(Float32, IO::ByteFormat::LittleEndian)
end
io.close
STDERR.puts "Loaded #{total_entries} entries (#{(total_entries * 8.0 / 1024 / 1024).round(2)} MB)"

# Helper: format an integer-token sequence as text.
def decode(tokens : Array(Int32), chars : Array(Char)) : String
  String.build do |io|
    tokens.each do |t|
      ch = chars[t]?
      io << (ch ? ch.inspect[1...-1] : "<#{t}>")
    end
  end
end

# Print a slot.
def print_slot(slot : Int32, off : Int32, len : Int32,
               all_tokens : Array(Int32), all_probs : Array(Float32),
               chars : Array(Char))
  if len == 0
    puts "    (empty)"
    return
  end
  sum = 0.0
  h = 0.0
  len.times do |i|
    p = all_probs[off + i].to_f64
    sum += p
    h -= p * Math.log(p + 1e-30) if p > 0
  end
  puts "    sum=#{sum.round(6)}  H=#{h.round(4)} nats  top-#{len}:"
  len.times do |i|
    tok = all_tokens[off + i]
    prob = all_probs[off + i]
    ch = chars[tok]?
    label = ch ? ch.inspect : "<#{tok}>"
    puts "      #{label.ljust(8)} #{prob.to_s}"
  end
end

# Pick caps to inspect.
selected = [] of RadixTrieReader::LoadedRecord
if specific_id >= 0
  rec = record_by_id[specific_id]?
  abort "cap id #{specific_id} not in trie" if rec.nil?
  abort "id #{specific_id} is not a cap (counts.size=#{rec.counts.size})" if rec.counts.size != 1
  selected << rec
else
  rng = Random.new(seed)
  caps_filled_per_n = [] of RadixTrieReader::LoadedRecord
  reader.each do |r|
    next unless r.counts.size == 1
    slot0 = r.id * expansion_depth
    # Only consider caps where ALL expansion_depth positions are filled
    fully_filled = true
    expansion_depth.times do |p|
      if lengths[slot0 + p] == 0
        fully_filled = false
        break
      end
    end
    caps_filled_per_n << r if fully_filled
  end
  STDERR.puts "Caps with all #{expansion_depth} positions filled: #{caps_filled_per_n.size}"
  caps_filled_per_n.sample(Math.min(n_inspect, caps_filled_per_n.size), rng).each { |r| selected << r }
end

selected.each_with_index do |cap, idx|
  path = full_path(cap, record_by_id)
  edge_start_0idx = cap.first_char_depth - 1
  prefix_chars = path[0...edge_start_0idx]
  edge_chars = cap.edge_tokens

  puts ""
  puts "=== cap ##{idx + 1}/#{selected.size} (radix_id=#{cap.id}) ==="
  puts "  full path (#{path.size} chars): #{decode(path, char_of).inspect}"
  puts "  branching prefix (depth 1..#{edge_start_0idx}): #{decode(prefix_chars, char_of).inspect}"
  puts "  cap edge (depth #{cap.first_char_depth}..#{cap.first_char_depth + edge_chars.size - 1}, len=#{edge_chars.size}): #{decode(edge_chars, char_of).inspect}"
  puts "  cap.counts (single): token=#{cap.counts[0][0]} (#{char_of[cap.counts[0][0]]?.try(&.inspect) || "?"}), count=#{cap.counts[0][1]}"

  expansion_depth.times do |p|
    slot = cap.id * expansion_depth + p
    off = offsets[slot]
    len = lengths[slot]
    if p < edge_chars.size
      cap_char = char_of[edge_chars[p]]?
      cap_char_lbl = cap_char ? cap_char.inspect : "<#{edge_chars[p]}>"
      puts "  position p=#{p} (cap edge char = #{cap_char_lbl}, depth #{cap.first_char_depth + p}):"
    else
      puts "  position p=#{p} (past cap edge, no expansion needed):"
    end
    print_slot(slot, off, len, all_tokens, all_probs, char_of)
  end
end
