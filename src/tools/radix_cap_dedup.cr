require "../agpt/radix_trie_reader"

# Count duplicate edge-token sequences across radix nodes.
# Tests the K=decision/V=identity framing: at deep nodes, do many radix
# nodes share the same compressed edge text (suggesting redundant
# memorization), or are they all unique?

trie_dir = ARGV[0]? || abort "usage: radix_cap_dedup <trie-dir>"

reader = MicroGPT::AGPT::RadixTrieReader.new(trie_dir, max_cached: 32)
puts "Trie: #{reader.radix_count} nodes, #{reader.depth_file_count} depth files"

# Per-depth statistics
struct DepthStats
  property total : Int32 = 0
  property unique : Int32 = 0
  property max_dup : Int32 = 0
  property max_dup_text : String = ""
end

stats = Hash(Int32, DepthStats).new { |h, k| h[k] = DepthStats.new }

# Aggregate over all depths
all_seqs = Hash(Array(Int32), Int32).new(0)
total_nodes = 0
nodes_per_dup = Hash(Int32, Int32).new(0)  # how many caps appear N times → count

# Per-depth dedup
per_depth_seqs = Hash(Int32, Hash(Array(Int32), Int32)).new { |h, k| h[k] = Hash(Array(Int32), Int32).new(0) }

reader.each do |rec|
  next if rec.id == 0  # skip virtual root
  ed = rec.endpoint_depth
  per_depth_seqs[ed][rec.edge_tokens] += 1
  all_seqs[rec.edge_tokens] += 1
  stats[ed].total += 1
  total_nodes += 1
end

# Compute per-depth stats
puts ""
puts "PER-DEPTH EDGE-SEQUENCE DEDUP"
puts "depth  n_nodes   unique_seqs  dup_rate  max_dup_count  example_dup_count"
per_depth_seqs.keys.sort.each do |d|
  seqs = per_depth_seqs[d]
  total = stats[d].total
  unique = seqs.size
  dup_rate = total > 0 ? (1.0 - unique.to_f / total) * 100 : 0.0
  max_dup = seqs.values.max? || 0

  printf("  %3d   %8d  %10d   %5.2f%%  %8d\n", d, total, unique, dup_rate, max_dup)
end

# Aggregate
total_unique = all_seqs.size
agg_dup_rate = total_nodes > 0 ? (1.0 - total_unique.to_f / total_nodes) * 100 : 0.0

puts ""
puts "OVERALL"
puts "  Total radix nodes (excl root): #{total_nodes}"
puts "  Unique edge-token sequences:    #{total_unique}"
puts "  Duplicate rate:                  #{sprintf("%.2f%%", agg_dup_rate)}"
puts "  Average occurrences per unique: #{sprintf("%.4f", total_nodes.to_f / total_unique)}"

# Distribution of duplicate counts
puts ""
puts "DUPLICATION DISTRIBUTION (across all depths)"
dup_dist = Hash(Int32, Int32).new(0)
all_seqs.each_value { |count| dup_dist[count] += 1 }

puts "  occurs N times | how many unique seqs | total nodes"
dup_dist.keys.sort.first(15).each do |n|
  count = dup_dist[n]
  printf("  %8d       | %12d         | %12d\n", n, count, n * count)
end
if dup_dist.keys.size > 15
  high_n = dup_dist.keys.sort[15..]
  remaining_unique = high_n.sum { |n| dup_dist[n] }
  remaining_nodes = high_n.sum { |n| n * dup_dist[n] }
  puts "  (#{remaining_unique} more unique seqs occurring 16+ times = #{remaining_nodes} nodes)"
  printf("  TOP duplicate count: %d times\n", dup_dist.keys.max)
end

# Show some highly-duplicated examples (excluding tiny edges)
puts ""
puts "MOST-DUPLICATED EDGE SEQUENCES (edge_len ≥ 3)"
top_dups = all_seqs.to_a.select { |(seq, _)| seq.size >= 3 }.sort_by { |(_, count)| -count }
top_dups.first(10).each do |(seq, count)|
  text = seq.map { |t| t.chr.inspect[1..-2] }.join
  printf("  %4d × \"%s\" (len=%d)\n", count, text, seq.size)
end
