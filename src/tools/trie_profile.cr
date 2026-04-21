# Trie sparsity profile.
#
# Loads a radix trie (global-radix format: radix_depth_NNN.bin + meta.bin) and
# prints a depth-by-depth profile of:
#   - number of radix endpoint nodes at that depth,
#   - branching factor distribution (how many are singletons / low / high),
#   - count distribution (min/median/max of total count per endpoint),
#   - fraction of endpoints that are singletons (= deterministic next-token).
#
# Usage: bin/trie-profile <radix_dir>
#
# The tool only loads from a global-radix format (NOT per-subtree). For
# per-subtree formats, concatenate the files or write a thin multiplexer.

require "../agpt"

if ARGV.size < 1
  STDERR.puts "Usage: trie-profile <radix_dir>"
  exit 1
end

dir = ARGV[0]

# Detect per-subtree vs global radix format.
if File.exists?(File.join(dir, "manifest.bin"))
  STDERR.puts "Per-subtree format detected. This tool currently only handles global-radix (radix_depth_*.bin at top level)."
  STDERR.puts "Point it at a dir like /tmp/agpt_input_d16_radix instead of /tmp/agpt_input_d16_radix_pst."
  exit 1
end

puts "Loading radix trie: #{dir}"
reader = MicroGPT::AGPT::RadixTrieReader.new(dir, max_cached: 8)
puts ""

puts "=============================================================================="
puts "TRIE SIZE"
puts "=============================================================================="
puts "  radix_count        = #{reader.radix_count}"
puts "    Number of nodes AFTER radix compression (storage units in this file)."
puts "    Each radix node has an 'edge' of 1+ tokens that represents a unary"
puts "    chain collapsed from the uncompressed trie."
puts "    (Includes the virtual root at id=0, edge_len=0.)"
puts ""
puts "  total_edge_chars   = #{reader.total_edge_chars}"
puts "    Sum of edge lengths across all radix nodes. Equivalent to the node"
puts "    count of the UNCOMPRESSED (leveled) D-trie — i.e., the number of"
puts "    distinct (path-from-root, token) character positions the corpus"
puts "    produced."
puts ""
puts "  compression        = #{sprintf("%.2f", reader.total_edge_chars.to_f / reader.radix_count)}x"
puts "    = total_edge_chars / radix_count. How many leveled-trie nodes, on"
puts "    average, each radix storage unit represents."
puts ""
puts "  depth_file_count   = #{reader.depth_file_count} (= D+1, files are indexed 0..D where D = cap depth)"
puts "  vocab_size         = #{reader.vocab_size}"
puts "  corpus_token_count = #{reader.corpus_token_count}"
puts ""
puts "=============================================================================="
puts "PER-DEPTH PROFILE"
puts "=============================================================================="
puts "  depth          Endpoint depth of radix nodes (where their edges end)."
puts "                 d=1..D-1 = interior branching points."
puts "                 d=D     = CAP — the max depth, where long unary tails terminate."
puts ""
puts "  n_nodes        Number of radix endpoints at this depth."
puts ""
puts "  total_count    Σ counts_val across those endpoints = total corpus mass"
puts "                 that hits a branching decision at this depth."
puts "                 (A corpus D-gram branches multiple times on its way to"
puts "                 the cap; it contributes to several depths' total_count.)"
puts ""
puts "  median_cnt     Median / mean / max of the per-endpoint total count"
puts "  mean_cnt       at this depth (how many corpus positions passed through"
puts "  max_cnt        each endpoint at this depth)."
puts ""
puts "  avg_branch     Mean branching factor = mean counts.size per endpoint."
puts "                 At interior depths ≥ 2 by construction (radix collapses"
puts "                 unary chains, so only real branching points survive as"
puts "                 interior endpoints). At the cap, can be 1 (singleton)."
puts ""
puts "  avg_edge_len   Mean length of the unary-compressed edge ending at this"
puts "                 endpoint. 1 = no compression; larger = longer unary"
puts "                 chain got absorbed into this edge."
puts ""
puts "  chars_absorbed Σ (edge_len - 1) at this depth = # of leveled-trie nodes"
puts "                 that got collapsed INTO edges ending at this depth."
puts ""
puts "  %singleton     At the cap only: fraction of endpoints with counts.size==1"
puts "                 (i.e., unique corpus D-grams with a single observed"
puts "                 continuation). Interior depths are tautologically 0% and"
puts "                 shown as (n/a)."
puts ""
puts "  depth  n_nodes    total_count  median_cnt  mean_cnt   max_cnt  avg_branch  avg_edge_len  chars_absorbed  %singleton"

(1..reader.depth_file_count - 1).each do |d|
  records = reader.nodes_at_endpoint_depth(d)
  next if records.empty?

  totals = records.map { |r| r.counts.sum { |pair| pair[1] } }
  branches = records.map { |r| r.counts.size }
  edge_lens = records.map { |r| r.edge_len }

  n = records.size
  total_count_all = totals.sum
  sorted = totals.sort
  median = sorted[n // 2]
  mean = total_count_all.to_f / n
  max = sorted.last
  avg_branch = branches.sum.to_f / n
  avg_edge_len = edge_lens.sum.to_f / n
  chars_absorbed = edge_lens.sum - n  # (edge_len - 1) summed across endpoints

  is_cap = (d == reader.depth_file_count - 1)
  singletons = records.count { |r| r.counts.size <= 1 }
  singleton_str = is_cap ? sprintf("%.2f%%", 100.0 * singletons / n) : "   (n/a)"

  printf "  %5d  %9d  %13d  %10d  %9.2f  %9d  %9.2f  %12.2f  %14d  %s\n",
    d, n, total_count_all, median, mean, max,
    avg_branch, avg_edge_len, chars_absorbed, singleton_str
end

# Summary totals
puts ""
puts "=============================================================================="
puts "COMPRESSION SUMMARY"
puts "=============================================================================="
total_radix_endpoints = (0..reader.depth_file_count - 1).sum { |d| reader.nodes_at_endpoint_depth(d).size }
total_edge_chars = reader.total_edge_chars
absorbed_chars_total = total_edge_chars - total_radix_endpoints
compression_ratio = total_edge_chars.to_f / total_radix_endpoints
printf "  leveled trie node count  = %d  (= total_edge_chars)\n", total_edge_chars
printf "  radix endpoints          = %d  (= storage units with an edge)\n", total_radix_endpoints
printf "  absorbed into edges      = %d  (= leveled nodes hidden inside radix edges)\n", absorbed_chars_total
printf "                             %.2f%% of leveled nodes got compressed\n", 100.0 * absorbed_chars_total / total_edge_chars
printf "  compression ratio        = %.2fx  (avg leveled nodes per radix storage unit)\n", compression_ratio
