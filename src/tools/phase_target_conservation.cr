# Check phase-conditioned target mass conservation.
#
# For active non-cap prefixes P at presentation start phase q:
#
#   mass(P @ q) ?= sum_t mass((P+t) @ q+1)
#
# Usage:
#   crystal run src/tools/phase_target_conservation.cr -- \
#     --trie /tmp/<prefix_radix> \
#     --position-data /tmp/<position_data> \
#     --phase 0

require "../agpt"
require "option_parser"

include MicroGPT::AGPT

trie_dir = ""
position_data_dir = ""
phase = 0
sample_limit = 16

OptionParser.parse do |p|
  p.banner = "Usage: phase_target_conservation --trie DIR --position-data DIR [--phase N] [--sample-limit N]"
  p.on("--trie DIR", "Prefix radix trie directory") { |v| trie_dir = v }
  p.on("--position-data DIR", "Position data directory") { |v| position_data_dir = v }
  p.on("--phase N", "Presentation start phase [default 0]") { |v| phase = v.to_i }
  p.on("--sample-limit N", "Mismatch examples [default 16]") { |v| sample_limit = v.to_i }
  p.on("-h", "--help", "") { puts p; exit 0 }
end

raise "--trie required" if trie_dir.empty?
raise "--position-data required" if position_data_dir.empty?

catalog_path = File.join(position_data_dir, "substrings.bin")
prt_path = File.join(position_data_dir, "prefix_radix_to_substring.bin")
ppt_path = File.join(position_data_dir, "prefix_position_table.bin")
ptg_path = File.join(position_data_dir, "prefix_phase_targets.bin")
[catalog_path, prt_path, ppt_path].each { |path| raise "Missing: #{path}" unless File.exists?(path) }

reader = RadixTrieReader.new(trie_dir, max_cached: 4)
catalog = File.open(catalog_path, "rb") { |f| SubstringCatalog.read_from(f) }
radix_to_substring = File.open(prt_path, "rb") { |f| RadixToSubstring.read_from(f) }
position_table = File.open(ppt_path, "rb") { |f| PositionTable.read_from(f) }
direct_target_offsets = Slice(Int32).new(0)
direct_target_entries = [] of Tuple(Int32, Int32, Int64)
direct_target_window = 0
if File.exists?(ptg_path)
  File.open(ptg_path, "rb") do |io|
    magic = Bytes.new(4)
    io.read_fully(magic)
    raise "Bad magic in phase target table: #{String.new(magic)}" unless String.new(magic) == "APTG"
    direct_target_window = io.read_bytes(UInt16, IO::ByteFormat::LittleEndian).to_i
    io.read_bytes(UInt16, IO::ByteFormat::LittleEndian)
    radix_count = io.read_bytes(UInt32, IO::ByteFormat::LittleEndian).to_i
    total_entries = io.read_bytes(UInt64, IO::ByteFormat::LittleEndian).to_i
    direct_target_offsets = Slice(Int32).new(radix_count + 1) do
      io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
    end
    direct_target_entries = Array(Tuple(Int32, Int32, Int64)).new(total_entries)
    total_entries.times do
      ph = io.read_bytes(UInt16, IO::ByteFormat::LittleEndian).to_i
      tok = io.read_bytes(UInt16, IO::ByteFormat::LittleEndian).to_i
      count = io.read_bytes(UInt32, IO::ByteFormat::LittleEndian).to_i64
      direct_target_entries << {ph, tok, count}
    end
  end
end

if radix_to_substring.radix_count != reader.radix_count
  raise "radix count mismatch: trie=#{reader.radix_count} position-map=#{radix_to_substring.radix_count}"
end

window = position_table.window_size
phase %= window
phase += window if phase < 0
tree_depth = reader.depth_file_count - 1

def bin_count(position_table : PositionTable, sid : Int32, start_phase : Int32) : Int64
  pos = start_phase % position_table.window_size
  pos += position_table.window_size if pos < 0
  position_table.bins(sid).each do |bin|
    bpos = bin.pos.to_i
    return bin.count.to_i64 if bpos == pos
    break if bpos > pos
  end
  0_i64
end

def reconstruct_prefix(records_by_id : Hash(Int32, RadixTrieReader::LoadedRecord), rec : RadixTrieReader::LoadedRecord) : Array(Int32)
  segments = [] of Array(Int32)
  cur = rec
  while cur.id > 0
    segments << cur.edge_tokens
    parent = records_by_id[cur.parent_id]?
    break if parent.nil? || parent.id == cur.id
    cur = parent
  end
  out = [] of Int32
  segments.reverse_each { |seg| seg.each { |tok| out << tok } }
  out
end

records_by_id = Hash(Int32, RadixTrieReader::LoadedRecord).new
reader.each { |rec| records_by_id[rec.id] = rec }

def direct_target_count(offsets : Slice(Int32), entries : Array(Tuple(Int32, Int32, Int64)), rid : Int32, phase : Int32, token : Int32) : Int64
  return -1_i64 if offsets.empty?
  start = offsets[rid]
  stop = offsets[rid + 1]
  i = start
  while i < stop
    ph, tok, count = entries[i]
    return count if ph == phase && tok == token
    break if ph > phase || (ph == phase && tok > token)
    i += 1
  end
  0_i64
end

checked = 0_i64
matched = 0_i64
mismatched = 0_i64
inactive = 0_i64
cap_skipped = 0_i64
missing_prefix_sid = 0_i64
missing_child_sid_entries = 0_i64
zero_child_bin_entries = 0_i64

prefix_mass_sum = 0_i64
target_mass_sum = 0_i64
delta_abs_sum = 0_i64

by_depth = Hash(Int32, Tuple(Int64, Int64, Int64, Int64, Int64)).new { |h, k| h[k] = {0_i64, 0_i64, 0_i64, 0_i64, 0_i64} }
examples = [] of String

reader.each do |rec|
  next if rec.id == 0 || rec.counts.empty?
  if rec.endpoint_depth >= tree_depth
    cap_skipped += 1
    next
  end

  sid = radix_to_substring.substring_id_for(rec.id)
  if sid < 0 || sid >= position_table.substring_count
    missing_prefix_sid += 1
    next
  end

  prefix_mass = bin_count(position_table, sid, phase)
  if prefix_mass <= 0
    inactive += 1
    next
  end

  prefix_tokens = reconstruct_prefix(records_by_id, rec)
  child_sum = 0_i64
  missing_children = 0
  zero_child_bins = 0
  rec.counts.each do |tok, _cnt|
    child_count = direct_target_count(direct_target_offsets, direct_target_entries, rec.id, phase, tok)
    if child_count >= 0
      if child_count <= 0
        zero_child_bins += 1
        zero_child_bin_entries += 1
      end
      child_sum += child_count
    else
      prefix_tokens << tok
      child_sid = catalog.lookup(prefix_tokens)
      prefix_tokens.pop
      if child_sid
        child_count = bin_count(position_table, child_sid, phase)
        if child_count <= 0
          zero_child_bins += 1
          zero_child_bin_entries += 1
        end
        child_sum += child_count
      else
        missing_children += 1
        missing_child_sid_entries += 1
      end
    end
  end

  checked += 1
  prefix_mass_sum += prefix_mass
  target_mass_sum += child_sum
  delta = prefix_mass - child_sum
  delta_abs_sum += delta.abs

  depth_checked, depth_mismatch, depth_prefix, depth_target, depth_delta_abs = by_depth[rec.endpoint_depth]
  depth_checked += 1
  depth_prefix += prefix_mass
  depth_target += child_sum
  depth_delta_abs += delta.abs

  if delta == 0
    matched += 1
  else
    mismatched += 1
    depth_mismatch += 1
    if examples.size < sample_limit
      examples << "radix_id=#{rec.id} depth=#{rec.endpoint_depth} prefix_mass=#{prefix_mass} child_sum=#{child_sum} delta=#{delta} counts=#{rec.counts.size} missing_child_sid=#{missing_children} zero_child_bins=#{zero_child_bins}"
    end
  end
  by_depth[rec.endpoint_depth] = {depth_checked, depth_mismatch, depth_prefix, depth_target, depth_delta_abs}
end

def pct(num, den)
  return 0.0 if den == 0
  100.0 * num.to_f / den.to_f
end

puts "Phase Target Conservation"
puts "  trie: #{trie_dir}"
puts "  position_data: #{position_data_dir}"
puts "  window: #{window}"
puts "  tree_depth: #{tree_depth}"
puts "  presentation_start_phase: #{phase}"
puts "  direct_phase_targets: #{direct_target_offsets.empty? ? "absent" : "present entries=#{direct_target_entries.size} window=#{direct_target_window}"}"
puts ""
puts "Scope"
puts "  checked active non-cap nodes: #{checked}"
puts "  inactive skipped: #{inactive}"
puts "  cap skipped: #{cap_skipped}"
puts "  missing prefix substring ids: #{missing_prefix_sid}"
puts ""
puts "Conservation"
puts "  matched nodes: #{matched} (#{sprintf("%.2f", pct(matched, checked))}%)"
puts "  mismatched nodes: #{mismatched} (#{sprintf("%.2f", pct(mismatched, checked))}%)"
puts "  prefix mass sum: #{prefix_mass_sum}"
puts "  child target mass sum: #{target_mass_sum}"
puts "  delta: #{prefix_mass_sum - target_mass_sum}"
puts "  abs delta sum: #{delta_abs_sum}"
puts "  retained vs prefix: #{sprintf("%.2f", pct(target_mass_sum, prefix_mass_sum))}%"
puts ""
puts "Child Lookup"
puts "  missing child substring entries: #{missing_child_sid_entries}"
puts "  zero child-bin entries: #{zero_child_bin_entries}"
puts ""
puts "By Endpoint Depth"
by_depth.keys.sort.each do |depth|
  depth_checked, depth_mismatch, depth_prefix, depth_target, depth_delta_abs = by_depth[depth]
  printf "  depth=%2d checked=%8d mismatch=%8d (%6.2f%%) prefix=%8d target=%8d retained=%6.2f%% abs_delta=%8d\n",
    depth, depth_checked, depth_mismatch, pct(depth_mismatch, depth_checked),
    depth_prefix, depth_target, pct(depth_target, depth_prefix), depth_delta_abs
end

unless examples.empty?
  puts ""
  puts "Examples"
  examples.each { |ex| puts "  #{ex}" }
end
