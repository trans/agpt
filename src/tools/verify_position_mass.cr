# Verify that prefix position-table mass agrees with the prefix radix trie.
#
# Usage:
#   crystal run src/tools/verify_position_mass.cr -- \
#     --trie /tmp/<prefix_radix> \
#     --position-data /tmp/<position_data>

require "../agpt"
require "option_parser"

include MicroGPT::AGPT

trie_dir = ""
position_data_dir = ""
sample_limit = 8

OptionParser.parse do |p|
  p.banner = "Usage: verify_position_mass --trie DIR --position-data DIR [--sample-limit N]"
  p.on("--trie DIR", "Prefix radix trie directory") { |v| trie_dir = v }
  p.on("--position-data DIR", "Position data directory") { |v| position_data_dir = v }
  p.on("--sample-limit N", "Number of mismatch examples to print [default 8]") { |v| sample_limit = v.to_i }
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

puts "Position Mass Verification"
puts "  trie: #{trie_dir}"
puts "  position_data: #{position_data_dir}"
puts "  radix_count: #{reader.radix_count}"
puts "  window: #{position_table.window_size}"
puts "  substring_count: #{position_table.substring_count}"
puts ""

node_count = 0_i64
node_match = 0_i64
node_mismatch = 0_i64
node_missing_sid = 0_i64
node_mass_sum = 0_i64
node_pos_sum = 0_i64
node_examples = [] of String

endpoint_count = 0_i64
endpoint_match = 0_i64
endpoint_mismatch = 0_i64
endpoint_missing_child = 0_i64
endpoint_global_sum = 0_i64
endpoint_phase_sum = 0_i64
endpoint_examples = [] of String
endpoint_by_depth = Hash(Int32, Tuple(Int64, Int64, Int64, Int64)).new { |h, k| h[k] = {0_i64, 0_i64, 0_i64, 0_i64} }

reader.each do |rec|
  rid = rec.id
  next if rid == 0
  node_count += 1
  node_mass_sum += rec.edge_mass
  sid = radix_to_substring.substring_id_for(rid)
  if sid < 0 || sid >= position_table.substring_count
    node_missing_sid += 1
    if node_examples.size < sample_limit
      node_examples << "radix_id=#{rid} depth=#{rec.endpoint_depth} mass=#{rec.edge_mass} missing substring_id=#{sid}"
    end
  else
    pos_total = position_table.total_count(sid).to_i64
    node_pos_sum += pos_total
    if pos_total == rec.edge_mass
      node_match += 1
    else
      node_mismatch += 1
      if node_examples.size < sample_limit
        node_examples << "radix_id=#{rid} depth=#{rec.endpoint_depth} edge_mass=#{rec.edge_mass} position_sum=#{pos_total} sid=#{sid}"
      end
    end
  end
end

# Second pass: build the parent/token -> child index needed for endpoint checks.
child_index = Hash(UInt64, Int32).new(initial_capacity: reader.radix_count)
reader.each do |rec|
  next if rec.id == 0 || rec.edge_tokens.empty?
  key = (rec.parent_id.to_u64 << 32) | rec.edge_tokens[0].to_u32.to_u64
  child_index[key] = rec.id
end

endpoint_count = 0_i64
endpoint_match = 0_i64
endpoint_mismatch = 0_i64
endpoint_missing_child = 0_i64
endpoint_global_sum = 0_i64
endpoint_phase_sum = 0_i64
endpoint_examples.clear

reader.each do |rec|
  next if rec.id == 0 || rec.counts.empty?
  endpoint_count += 1
  global_total = rec.counts.sum { |pair| pair[1] }.to_i64
  phase_total = 0_i64
  missing_children = [] of Int32

  rec.counts.each do |token, _count|
    child_id = child_index[(rec.id.to_u64 << 32) | token.to_u32.to_u64]?
    if child_id.nil?
      missing_children << token
      next
    end
    child_sid = radix_to_substring.substring_id_for(child_id)
    if child_sid < 0 || child_sid >= position_table.substring_count
      missing_children << token
      next
    end
    phase_total += position_table.total_count(child_sid).to_i64
  end

  endpoint_global_sum += global_total
  endpoint_phase_sum += phase_total
  depth_checked, depth_mismatch, depth_global, depth_phase = endpoint_by_depth[rec.endpoint_depth]
  depth_checked += 1
  depth_global += global_total
  depth_phase += phase_total

  if missing_children.empty? && phase_total == global_total
    endpoint_match += 1
  else
    endpoint_mismatch += 1
    depth_mismatch += 1
    endpoint_missing_child += 1 unless missing_children.empty?
    if endpoint_examples.size < sample_limit
      endpoint_examples << "radix_id=#{rec.id} depth=#{rec.endpoint_depth} global_out=#{global_total} child_position_sum=#{phase_total} missing_child_tokens=#{missing_children.size}"
    end
  end
  endpoint_by_depth[rec.endpoint_depth] = {depth_checked, depth_mismatch, depth_global, depth_phase}
end

def pct(num, den)
  return 0.0 if den == 0
  100.0 * num.to_f / den.to_f
end

puts "Node Mass"
puts "  checked: #{node_count}"
puts "  matches: #{node_match} (#{sprintf("%.2f", pct(node_match, node_count))}%)"
puts "  mismatches: #{node_mismatch}"
puts "  missing substring ids: #{node_missing_sid}"
puts "  global edge_mass sum: #{node_mass_sum}"
puts "  position mass sum: #{node_pos_sum}"
unless node_examples.empty?
  puts "  examples:"
  node_examples.each { |ex| puts "    #{ex}" }
end

puts ""
puts "Endpoint Child Target Mass"
puts "  checked: #{endpoint_count}"
puts "  matches: #{endpoint_match} (#{sprintf("%.2f", pct(endpoint_match, endpoint_count))}%)"
puts "  mismatches: #{endpoint_mismatch}"
puts "  endpoints with missing child mapping: #{endpoint_missing_child}"
puts "  global outgoing sum: #{endpoint_global_sum}"
puts "  child position mass sum: #{endpoint_phase_sum}"
puts "  by depth:"
endpoint_by_depth.keys.sort.each do |depth|
  checked, mismatches, global_sum, phase_sum = endpoint_by_depth[depth]
  printf "    depth=%2d checked=%9d mismatches=%9d global=%10d child_position=%10d delta=%10d\n",
    depth, checked, mismatches, global_sum, phase_sum, global_sum - phase_sum
end
unless endpoint_examples.empty?
  puts "  examples:"
  endpoint_examples.each { |ex| puts "    #{ex}" }
end

ok = node_mismatch == 0 && node_missing_sid == 0 && endpoint_mismatch == 0
puts ""
puts "status: #{ok ? "ok" : "mismatch"}"
exit(ok ? 0 : 1)
