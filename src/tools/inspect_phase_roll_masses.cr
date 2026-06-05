# Print per-roll phase masses for sampled radix nodes.
#
# Usage:
#   crystal run src/tools/inspect_phase_roll_masses.cr -- \
#     --trie /tmp/<prefix_radix> \
#     --position-data /tmp/<position_data> \
#     --samples 12

require "../agpt"
require "option_parser"

include MicroGPT::AGPT

trie_dir = ""
position_data_dir = ""
vocab_source = "data/input.txt"
samples = 12
min_mass = 2
max_rolls = 16
sample_mode = "top"

OptionParser.parse do |p|
  p.banner = "Usage: inspect_phase_roll_masses --trie DIR --position-data DIR [options]"
  p.on("--trie DIR", "Prefix radix trie directory") { |v| trie_dir = v }
  p.on("--position-data DIR", "Position data directory") { |v| position_data_dir = v }
  p.on("--vocab-source PATH", "Text file used for tokenizer display [default data/input.txt]") { |v| vocab_source = v }
  p.on("--samples N", "Number of nodes to print [default 12]") { |v| samples = v.to_i }
  p.on("--min-mass N", "Minimum global edge mass [default 2]") { |v| min_mass = v.to_i }
  p.on("--max-rolls N", "Maximum legal presentation rolls to print [default 16]") { |v| max_rolls = v.to_i }
  p.on("--mode MODE", "Sample mode: top|spread [default top]") { |v| sample_mode = v }
  p.on("-h", "--help", "") { puts p; exit 0 }
end

raise "--trie required" if trie_dir.empty?
raise "--position-data required" if position_data_dir.empty?

prt_path = File.join(position_data_dir, "prefix_radix_to_substring.bin")
ppt_path = File.join(position_data_dir, "prefix_position_table.bin")
[prt_path, ppt_path].each { |p| raise "Missing: #{p}" unless File.exists?(p) }

reader = RadixTrieReader.new(trie_dir, max_cached: 4)
radix_to_substring = File.open(prt_path, "rb") { |f| RadixToSubstring.read_from(f) }
position_table = File.open(ppt_path, "rb") { |f| PositionTable.read_from(f) }

if radix_to_substring.radix_count != reader.radix_count
  raise "radix count mismatch: trie=#{reader.radix_count} position-map=#{radix_to_substring.radix_count}"
end

id_to_char = nil
if File.exists?(vocab_source)
  ds = MicroGPT::CharDataset.new(File.read(vocab_source))
  id_to_char = ds.id_to_char
end

def token_text(tokens, id_to_char)
  return tokens.join(" ") unless id_to_char
  tokens.map do |t|
    if t >= 0 && t < id_to_char.size
      id_to_char[t]
    else
      "<?>"
    end
  end.join("")
end

def prefix_tokens(walker, radix_id : Int32) : Array(Int32)
  ids = [] of Int32
  id = radix_id
  while id > 0
    ids << id
    id = walker.parent_id_of(id)
  end
  tokens = [] of Int32
  ids.reverse_each do |rid|
    walker.edge_tokens_of(rid).each { |t| tokens << t }
  end
  tokens
end

def bin_count(position_table, sid : Int32, phase : Int32) : UInt32
  pos = phase % position_table.window_size
  pos += position_table.window_size if pos < 0
  position_table.bins(sid).each do |bin|
    return bin.count if bin.pos.to_i == pos
    break if bin.pos.to_i > pos
  end
  0_u32
end

records = [] of RadixTrieReader::LoadedRecord
reader.each do |rec|
  next if rec.id == 0
  next if rec.edge_mass < min_mass
  sid = radix_to_substring.substring_id_for(rec.id)
  next if sid < 0 || sid >= position_table.substring_count
  records << rec
end

selected = [] of RadixTrieReader::LoadedRecord
case sample_mode
when "top"
  selected = records.sort_by { |r| {-r.edge_mass, r.endpoint_depth, r.id} }.first(samples)
when "spread"
  by_depth = records.group_by(&.endpoint_depth)
  by_depth.keys.sort.each do |depth|
    break if selected.size >= samples
    if rec = by_depth[depth].max_by?(&.edge_mass)
      selected << rec
    end
  end
else
  raise "--mode must be top or spread"
end

walker = CorpusTrieWalker.new(reader, [] of Int32)

window = position_table.window_size
tree_depth = reader.depth_file_count - 1
phase_span = window - tree_depth + 1
phase_span = 1 if phase_span < 1
roll_count = {phase_span, max_rolls}.min

puts "Phase Roll Mass Samples"
puts "  trie: #{trie_dir}"
puts "  position_data: #{position_data_dir}"
puts "  window: #{window}"
puts "  tree_depth: #{tree_depth}"
puts "  legal_presentation_rolls: 0..#{phase_span - 1} (span=#{phase_span})"
puts "  printed_rolls: 0..#{roll_count - 1}"
puts "  sample_mode: #{sample_mode}"
puts ""

selected.each do |rec|
  sid = radix_to_substring.substring_id_for(rec.id)
  total = position_table.total_count(sid)
  endpoint_zero_based = rec.endpoint_depth - 1
  text = token_text(rec.edge_tokens, id_to_char)
  prefix = token_text(prefix_tokens(walker, rec.id), id_to_char)
  puts "radix_id=#{rec.id} depth=#{rec.endpoint_depth} edge_mass=#{rec.edge_mass} position_sum=#{total} sid=#{sid}"
  puts "  prefix=#{prefix.inspect}"
  puts "  edge=#{text.inspect}"
  rolls = [] of String
  roll_count.times do |roll|
    endpoint_phase = (roll + endpoint_zero_based) % window
    mass = bin_count(position_table, sid, roll)
    rolls << "r#{roll}->q#{endpoint_phase}:#{mass}"
  end
  puts "  #{rolls.join(" ")}"
  puts ""
end
