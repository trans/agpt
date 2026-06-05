# Report node/query coverage for phase-weighted presentation over legal rolls.
#
# Usage:
#   crystal run src/tools/phase_coverage.cr -- \
#     --trie /tmp/<prefix_radix> \
#     --position-data /tmp/<position_data>

require "../agpt"
require "option_parser"

include MicroGPT::AGPT

trie_dir = ""
position_data_dir = ""
sample_limit = 12

OptionParser.parse do |p|
  p.banner = "Usage: phase_coverage --trie DIR --position-data DIR [--sample-limit N]"
  p.on("--trie DIR", "Prefix radix trie directory") { |v| trie_dir = v }
  p.on("--position-data DIR", "Position data directory") { |v| position_data_dir = v }
  p.on("--sample-limit N", "Number of never-active examples to print [default 12]") { |v| sample_limit = v.to_i }
  p.on("-h", "--help", "") { puts p; exit 0 }
end

raise "--trie required" if trie_dir.empty?
raise "--position-data required" if position_data_dir.empty?

prt_path = File.join(position_data_dir, "prefix_radix_to_substring.bin")
ppt_path = File.join(position_data_dir, "prefix_position_table.bin")
[prt_path, ppt_path].each { |path| raise "Missing: #{path}" unless File.exists?(path) }

reader = RadixTrieReader.new(trie_dir, max_cached: 4)
radix_to_substring = File.open(prt_path, "rb") { |f| RadixToSubstring.read_from(f) }
position_table = File.open(ppt_path, "rb") { |f| PositionTable.read_from(f) }

if radix_to_substring.radix_count != reader.radix_count
  raise "radix count mismatch: trie=#{reader.radix_count} position-map=#{radix_to_substring.radix_count}"
end

window = position_table.window_size
tree_depth = reader.depth_file_count - 1
phase_span = window
phase_span = 1 if phase_span < 1

def pct(num, den)
  return 0.0 if den == 0
  100.0 * num.to_f / den.to_f
end

def has_bin_at_phase?(position_table, sid : Int32, phase : Int32) : Bool
  pos = phase % position_table.window_size
  pos += position_table.window_size if pos < 0
  position_table.bins(sid).each do |bin|
    bpos = bin.pos.to_i
    return true if bpos == pos && bin.count > 0
    break if bpos > pos
  end
  false
end

total_nodes = 0_i64
total_queries = 0_i64
total_mass_events = 0_i64
missing_sid_nodes = 0_i64

active_nodes_any = 0_i64
active_queries_any = 0_i64
never_active_nodes = 0_i64
never_active_queries = 0_i64

active_nodes_per_roll = Array(Int64).new(phase_span, 0_i64)
active_queries_per_roll = Array(Int64).new(phase_span, 0_i64)
active_events_per_roll = Array(Int64).new(phase_span, 0_i64)

roll_hits_hist = Hash(Int32, Int64).new(0_i64)
by_depth = Hash(Int32, Tuple(Int64, Int64, Int64, Int64)).new { |h, k| h[k] = {0_i64, 0_i64, 0_i64, 0_i64} }
never_examples = [] of String

reader.each do |rec|
  next if rec.id == 0
  total_nodes += 1
  total_queries += rec.edge_len
  total_mass_events += rec.edge_len.to_i64 * rec.edge_mass.to_i64

  sid = radix_to_substring.substring_id_for(rec.id)
  if sid < 0 || sid >= position_table.substring_count
    missing_sid_nodes += 1
    never_active_nodes += 1
    never_active_queries += rec.edge_len
    checked, active, queries, active_queries = by_depth[rec.endpoint_depth]
    by_depth[rec.endpoint_depth] = {checked + 1, active, queries + rec.edge_len, active_queries}
    if never_examples.size < sample_limit
      never_examples << "radix_id=#{rec.id} depth=#{rec.endpoint_depth} edge_len=#{rec.edge_len} mass=#{rec.edge_mass} missing_sid=#{sid}"
    end
    next
  end

  endpoint_zero_based = rec.endpoint_depth - 1
  hit_count = 0
  phase_span.times do |roll|
    endpoint_phase = roll + endpoint_zero_based
    if has_bin_at_phase?(position_table, sid, endpoint_phase)
      hit_count += 1
      active_nodes_per_roll[roll] += 1
      active_queries_per_roll[roll] += rec.edge_len
      active_events_per_roll[roll] += rec.edge_len.to_i64 * position_table.bins(sid).find { |bin| bin.pos.to_i == endpoint_phase % window }.not_nil!.count.to_i64
    end
  end

  roll_hits_hist[hit_count] += 1
  checked, active, queries, active_queries = by_depth[rec.endpoint_depth]
  checked += 1
  queries += rec.edge_len
  if hit_count > 0
    active_nodes_any += 1
    active_queries_any += rec.edge_len
    active += 1
    active_queries += rec.edge_len
  else
    never_active_nodes += 1
    never_active_queries += rec.edge_len
    if never_examples.size < sample_limit
      never_examples << "radix_id=#{rec.id} depth=#{rec.endpoint_depth} edge_len=#{rec.edge_len} mass=#{rec.edge_mass} sid=#{sid} position_sum=#{position_table.total_count(sid)}"
    end
  end
  by_depth[rec.endpoint_depth] = {checked, active, queries, active_queries}
end

def stats(xs : Array(Int64))
  return {0_i64, 0_i64, 0.0} if xs.empty?
  min = xs.min
  max = xs.max
  mean = xs.sum.to_f / xs.size
  {min, max, mean}
end

node_min, node_max, node_mean = stats(active_nodes_per_roll)
query_min, query_max, query_mean = stats(active_queries_per_roll)
event_min, event_max, event_mean = stats(active_events_per_roll)

puts "Phase Coverage"
puts "  trie: #{trie_dir}"
puts "  position_data: #{position_data_dir}"
puts "  window: #{window}"
puts "  tree_depth: #{tree_depth}"
puts "  legal_presentation_rolls: 0..#{phase_span - 1} (span=#{phase_span})"
puts ""
puts "Full-Cycle Coverage"
puts "  total nodes: #{total_nodes}"
puts "  active at least once: #{active_nodes_any} (#{sprintf("%.2f", pct(active_nodes_any, total_nodes))}%)"
puts "  never active: #{never_active_nodes} (#{sprintf("%.2f", pct(never_active_nodes, total_nodes))}%)"
puts "  missing substring ids: #{missing_sid_nodes}"
puts "  total query positions: #{total_queries}"
puts "  active query positions at least once: #{active_queries_any} (#{sprintf("%.2f", pct(active_queries_any, total_queries))}%)"
puts "  never-active query positions: #{never_active_queries} (#{sprintf("%.2f", pct(never_active_queries, total_queries))}%)"
puts "  global linear event mass: #{total_mass_events}"
puts ""
puts "Per-Roll Active Signal"
puts "  active nodes min/max/mean: #{node_min} / #{node_max} / #{sprintf("%.1f", node_mean)}"
puts "  active query positions min/max/mean: #{query_min} / #{query_max} / #{sprintf("%.1f", query_mean)}"
puts "  active linear event mass min/max/mean: #{event_min} / #{event_max} / #{sprintf("%.1f", event_mean)}"
puts ""
puts "Roll-Hit Histogram"
roll_hits_hist.keys.sort.each do |hits|
  count = roll_hits_hist[hits]
  puts "  hit_rolls=#{hits}: nodes=#{count} (#{sprintf("%.2f", pct(count, total_nodes))}%)"
end
puts ""
puts "By Endpoint Depth"
by_depth.keys.sort.each do |depth|
  checked, active, queries, active_queries = by_depth[depth]
  printf "  depth=%2d nodes=%9d active=%9d (%6.2f%%) queries=%9d active_queries=%9d (%6.2f%%)\n",
    depth, checked, active, pct(active, checked), queries, active_queries, pct(active_queries, queries)
end

unless never_examples.empty?
  puts ""
  puts "Never-Active Examples"
  never_examples.each { |ex| puts "  #{ex}" }
end
