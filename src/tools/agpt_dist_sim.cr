require "../agpt/radix_trie_reader"
require "option_parser"

# Diagnostic: how many trie nodes share the same (or near-identical) next-char
# distribution? Reports:
#   - exact-match groups: nodes whose counts arrays are byte-identical
#   - near-match (TV-distance ≤ threshold): pairwise sampling
#
# Comparing probability vectors:
#   - Total variation distance: TV(P, Q) = ½ Σ |P(x) - Q(x)|, bounded [0, 1].
#     Interpretable as "maximum probability mass moved between distributions."
#   - JSD: symmetric KL, bounded [0, ln 2]. More info-theoretic but slower.
# This tool uses TV for speed and interpretability.

include MicroGPT::AGPT

trie_dir = ""
sample_pairs = 100_000
near_threshold = 0.1

OptionParser.parse do |p|
  p.banner = "Usage: agpt_dist_sim --trie DIR [options]"
  p.on("--trie DIR", "Radix trie directory") { |v| trie_dir = v }
  p.on("--sample-pairs N", "Random pairs to sample for near-match (default 100000)") { |v| sample_pairs = v.to_i }
  p.on("--threshold F", "TV-distance threshold for 'near-match' (default 0.1)") { |v| near_threshold = v.to_f }
  p.on("-h", "--help", "") { puts p; exit 0 }
end

abort "missing --trie" if trie_dir.empty?

reader = RadixTrieReader.new(trie_dir, max_cached: 64)
n_radix = reader.radix_count
vocab_size = reader.vocab_size
STDERR.puts "Loaded #{n_radix} nodes, vocab=#{vocab_size}"

# Group nodes by exact-match counts signature.
# Signature = sorted (token, prob) tuples (hashable).
sigs = Hash(String, Int32).new(0)
node_counts = {} of Int32 => Array({Int32, Float64})  # id → normalized counts

reader.each do |r|
  next if r.counts.empty?
  total = 0
  r.counts.each { |entry| total += entry[1] }
  next if total <= 0
  inv = 1.0 / total.to_f64
  normalized = r.counts.map { |entry| {entry[0], entry[1].to_f64 * inv} }
  # Signature: probabilities rounded to 4 decimals
  sig = String.build do |io|
    normalized.sort_by { |e| -e[1] }.each do |entry|
      io << entry[0] << ":" << (entry[1] * 10000).to_i << ","
    end
  end
  sigs[sig] += 1
  node_counts[r.id] = normalized
end

# Stats on exact-match groups.
group_sizes = sigs.values.sort.reverse
total_in_groups = group_sizes.sum
unique_sigs = sigs.size
nodes_with_dist = node_counts.size
collision_groups = group_sizes.count { |c| c > 1 }
nodes_in_collisions = group_sizes.select { |c| c > 1 }.sum

STDERR.puts "\n=== Exact-match groups (counts identical to 4 decimal places) ==="
STDERR.puts "Nodes with non-empty counts: #{nodes_with_dist}"
STDERR.puts "Unique distribution signatures: #{unique_sigs}"
STDERR.puts "Groups of size > 1 (collisions): #{collision_groups}"
STDERR.puts "Nodes inside a collision group: #{nodes_in_collisions} (#{(100.0 * nodes_in_collisions / nodes_with_dist).round(2)}%)"
STDERR.puts "Top group sizes:"
group_sizes.first(10).each_with_index do |sz, i|
  STDERR.puts "  rank #{i + 1}: #{sz} nodes share one distribution"
end

# Random pairs for near-match TV-distance histogram.
STDERR.puts "\n=== Pairwise TV/JSD sample (#{sample_pairs} pairs) ==="
ids = node_counts.keys
rng = Random.new(42)
buckets = Hash(Int32, Int32).new(0)  # bucket = floor(TV * 20), so 0..20 = 0.05 bins
near_count = 0
jsd_buckets = Hash(Int32, Int32).new(0)  # bucket = floor(JSD/ln2 * 20), 0..20 = 0.05 bins
jsd_near_count = 0
jsd_threshold_nats = 0.05  # near-identical in nats; ln 2 ≈ 0.693 = max JSD
sample_pairs.times do
  i = rng.rand(ids.size)
  j = rng.rand(ids.size)
  next if i == j
  a = node_counts[ids[i]]
  b = node_counts[ids[j]]
  # TV: ½ Σ |P(x) - Q(x)|, bounded [0, 1].
  # JSD: ½ KL(P || M) + ½ KL(Q || M) where M = ½(P+Q), bounded [0, ln 2].
  # Build merged token set; compute both metrics in one pass.
  m_diff = Hash(Int32, Float64).new(0.0)
  a.each { |e| m_diff[e[0]] += e[1] }
  b.each { |e| m_diff[e[0]] -= e[1] }
  tv = 0.0
  m_diff.each_value { |v| tv += v.abs }
  tv *= 0.5
  m_avg = Hash(Int32, Float64).new(0.0)
  a.each { |e| m_avg[e[0]] += 0.5 * e[1] }
  b.each { |e| m_avg[e[0]] += 0.5 * e[1] }
  jsd = 0.0
  a.each do |e|
    p = e[1]
    mv = m_avg[e[0]]
    jsd += 0.5 * p * Math.log(p / mv) if p > 0 && mv > 0
  end
  b.each do |e|
    q = e[1]
    mv = m_avg[e[0]]
    jsd += 0.5 * q * Math.log(q / mv) if q > 0 && mv > 0
  end
  bucket = (tv * 20).to_i
  bucket = 20 if bucket > 20
  buckets[bucket] += 1
  near_count += 1 if tv <= near_threshold
  # JSD bucket (normalized to [0,1] by dividing by ln 2)
  jsd_norm = jsd / Math.log(2.0)
  jbucket = (jsd_norm * 20).to_i
  jbucket = 20 if jbucket > 20
  jsd_buckets[jbucket] += 1
  jsd_near_count += 1 if jsd <= jsd_threshold_nats
end
STDERR.puts "Pairs with TV ≤ #{near_threshold}: #{near_count} / #{sample_pairs} (#{(100.0 * near_count / sample_pairs).round(3)}%)"
STDERR.puts "TV distribution (bin = 0.05 wide):"
21.times do |b|
  count = buckets[b]
  next if count == 0
  bar_len = ((count * 60) / sample_pairs).to_i.clamp(0, 60)
  bar = "#" * bar_len
  STDERR.puts "  #{(b * 0.05).round(2).to_s.ljust(5)}-#{((b + 1) * 0.05).round(2).to_s.ljust(5)} #{count.to_s.rjust(8)}  #{bar}"
end
STDERR.puts ""
STDERR.puts "Pairs with JSD ≤ #{jsd_threshold_nats} nats (≈ very-similar): #{jsd_near_count} / #{sample_pairs} (#{(100.0 * jsd_near_count / sample_pairs).round(3)}%)"
STDERR.puts "JSD distribution (bin = 0.05·ln2 ≈ 0.0347 nats wide; max JSD = ln 2 ≈ 0.693):"
21.times do |b|
  count = jsd_buckets[b]
  next if count == 0
  bar_len = ((count * 60) / sample_pairs).to_i.clamp(0, 60)
  bar = "#" * bar_len
  lo = (b * 0.05 * Math.log(2.0)).round(3)
  hi = ((b + 1) * 0.05 * Math.log(2.0)).round(3)
  STDERR.puts "  #{lo.to_s.ljust(5)}-#{hi.to_s.ljust(5)} #{count.to_s.rjust(8)}  #{bar}"
end
