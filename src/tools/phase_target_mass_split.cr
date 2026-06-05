# Report where global endpoint target mass lands under phase-conditioned
# target lookup for one presentation start phase.
#
# Usage:
#   crystal run src/tools/phase_target_mass_split.cr -- \
#     --trie /tmp/<prefix_radix> \
#     --position-data /tmp/<position_data> \
#     --phase 0

require "../agpt"
require "option_parser"

include MicroGPT::AGPT

trie_dir = ""
position_data_dir = ""
phase = 0

OptionParser.parse do |p|
  p.banner = "Usage: phase_target_mass_split --trie DIR --position-data DIR [--phase N]"
  p.on("--trie DIR", "Prefix radix trie directory") { |v| trie_dir = v }
  p.on("--position-data DIR", "Position data directory") { |v| position_data_dir = v }
  p.on("--phase N", "Presentation start phase [default 0]") { |v| phase = v.to_i }
  p.on("-h", "--help", "") { puts p; exit 0 }
end

raise "--trie required" if trie_dir.empty?
raise "--position-data required" if position_data_dir.empty?

catalog_path = File.join(position_data_dir, "substrings.bin")
prt_path = File.join(position_data_dir, "prefix_radix_to_substring.bin")
ppt_path = File.join(position_data_dir, "prefix_position_table.bin")
[catalog_path, prt_path, ppt_path].each { |path| raise "Missing: #{path}" unless File.exists?(path) }

reader = RadixTrieReader.new(trie_dir, max_cached: 4)
catalog = File.open(catalog_path, "rb") { |f| SubstringCatalog.read_from(f) }
radix_to_substring = File.open(prt_path, "rb") { |f| RadixToSubstring.read_from(f) }
position_table = File.open(ppt_path, "rb") { |f| PositionTable.read_from(f) }

if radix_to_substring.radix_count != reader.radix_count
  raise "radix count mismatch: trie=#{reader.radix_count} position-map=#{radix_to_substring.radix_count}"
end

window = position_table.window_size
phase %= window
phase += window if phase < 0

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

nodes = 0_i64
zero_nodes = 0_i64
nonzero_nodes = 0_i64
singleton_nodes = 0_i64
inactive_nodes = 0_i64
active_prefix_mass_total = 0_i64
zero_cap_nodes = 0_i64
zero_noncap_nodes = 0_i64

global_mass_total = 0_i64
global_mass_on_zero_nodes = 0_i64
global_mass_on_nonzero_nodes = 0_i64
phase_mass_total = 0_i64
global_mass_on_zero_cap_nodes = 0_i64
global_mass_on_zero_noncap_nodes = 0_i64

matched_token_global_mass = 0_i64
missing_token_global_mass = 0_i64
matched_token_count = 0_i64
missing_token_count = 0_i64

by_depth = Hash(Int32, Tuple(Int64, Int64, Int64, Int64, Int64, Int64)).new { |h, k| h[k] = {0_i64, 0_i64, 0_i64, 0_i64, 0_i64, 0_i64} }
tree_depth = reader.depth_file_count - 1

reader.each do |rec|
  next if rec.id == 0 || rec.counts.empty?
  sid = radix_to_substring.substring_id_for(rec.id)
  next if sid < 0 || sid >= position_table.substring_count

  endpoint_depth_zero_based = rec.endpoint_depth - 1
  active_prefix_mass = bin_count(position_table, sid, phase)
  if active_prefix_mass <= 0
    inactive_nodes += 1
    next
  end
  active_prefix_mass_total += active_prefix_mass

  prefix_tokens = reconstruct_prefix(records_by_id, rec)
  global_total = 0_i64
  local_total = 0_i64
  local_tokens = 0

  rec.counts.each do |tok, cnt|
    global_total += cnt.to_i64
    prefix_tokens << tok
    child_sid = catalog.lookup(prefix_tokens)
    prefix_tokens.pop
    if child_sid
      start_phase = (phase + endpoint_depth_zero_based + 1 - (endpoint_depth_zero_based + 1)) % window
      child_count = bin_count(position_table, child_sid, start_phase)
      if child_count > 0
        local_total += child_count
        local_tokens += 1
        matched_token_count += 1
        matched_token_global_mass += cnt.to_i64
      else
        missing_token_count += 1
        missing_token_global_mass += cnt.to_i64
      end
    else
      missing_token_count += 1
      missing_token_global_mass += cnt.to_i64
    end
  end

  nodes += 1
  global_mass_total += global_total
  phase_mass_total += local_total
  depth_nodes, depth_zero, depth_global, depth_zero_global, depth_phase, depth_prefix_mass = by_depth[rec.endpoint_depth]
  depth_nodes += 1
  depth_global += global_total
  depth_phase += local_total
  depth_prefix_mass += active_prefix_mass
  if local_total <= 0
    zero_nodes += 1
    global_mass_on_zero_nodes += global_total
    depth_zero += 1
    depth_zero_global += global_total
    if rec.endpoint_depth >= tree_depth
      zero_cap_nodes += 1
      global_mass_on_zero_cap_nodes += global_total
    else
      zero_noncap_nodes += 1
      global_mass_on_zero_noncap_nodes += global_total
    end
  else
    nonzero_nodes += 1
    global_mass_on_nonzero_nodes += global_total
    singleton_nodes += 1 if local_tokens == 1
  end
  by_depth[rec.endpoint_depth] = {depth_nodes, depth_zero, depth_global, depth_zero_global, depth_phase, depth_prefix_mass}
end

def pct(num, den)
  return 0.0 if den == 0
  100.0 * num.to_f / den.to_f
end

puts "Phase Target Mass Split"
puts "  trie: #{trie_dir}"
puts "  position_data: #{position_data_dir}"
puts "  window: #{window}"
puts "  tree_depth: #{tree_depth}"
puts "  presentation_start_phase: #{phase}"
puts ""
puts "Active Endpoint Nodes"
puts "  inactive nodes skipped: #{inactive_nodes}"
puts "  active nodes: #{nodes}"
puts "  active prefix mass: #{active_prefix_mass_total}"
puts "  zero phase-target nodes: #{zero_nodes} (#{sprintf("%.2f", pct(zero_nodes, nodes))}%)"
puts "    at cap depth: #{zero_cap_nodes}"
puts "    below cap depth: #{zero_noncap_nodes}"
puts "  nonzero phase-target nodes: #{nonzero_nodes} (#{sprintf("%.2f", pct(nonzero_nodes, nodes))}%)"
puts "  singleton nonzero nodes: #{singleton_nodes} (#{sprintf("%.2f", pct(singleton_nodes, nodes))}%)"
puts ""
puts "Global Target Mass"
puts "  total global outgoing mass: #{global_mass_total}"
puts "  on zero phase-target nodes: #{global_mass_on_zero_nodes} (#{sprintf("%.2f", pct(global_mass_on_zero_nodes, global_mass_total))}%)"
puts "    at cap depth: #{global_mass_on_zero_cap_nodes} (#{sprintf("%.2f", pct(global_mass_on_zero_cap_nodes, global_mass_total))}%)"
puts "    below cap depth: #{global_mass_on_zero_noncap_nodes} (#{sprintf("%.2f", pct(global_mass_on_zero_noncap_nodes, global_mass_total))}%)"
puts "  on nonzero phase-target nodes: #{global_mass_on_nonzero_nodes} (#{sprintf("%.2f", pct(global_mass_on_nonzero_nodes, global_mass_total))}%)"
puts "  retained phase target mass: #{phase_mass_total} (#{sprintf("%.2f", pct(phase_mass_total, global_mass_total))}% of global)"
puts ""
puts "Target Tokens"
puts "  matched token entries: #{matched_token_count}"
puts "  missing/zero token entries: #{missing_token_count}"
puts "  global mass on matched token entries: #{matched_token_global_mass} (#{sprintf("%.2f", pct(matched_token_global_mass, global_mass_total))}%)"
puts "  global mass on missing/zero token entries: #{missing_token_global_mass} (#{sprintf("%.2f", pct(missing_token_global_mass, global_mass_total))}%)"
puts ""
puts "By Endpoint Depth"
by_depth.keys.sort.each do |depth|
  depth_nodes, depth_zero, depth_global, depth_zero_global, depth_phase, depth_prefix_mass = by_depth[depth]
  printf "  depth=%2d active_nodes=%9d zero=%9d (%6.2f%%) prefix_mass=%10d global=%10d zero_global=%10d (%6.2f%%) phase=%10d (%6.2f%%)\n",
    depth, depth_nodes, depth_zero, pct(depth_zero, depth_nodes),
    depth_prefix_mass, depth_global, depth_zero_global, pct(depth_zero_global, depth_global),
    depth_phase, pct(depth_phase, depth_global)
end
