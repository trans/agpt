# Build a prefix-to-suffix match index for p2s-attention exploration.
#
# For each prefix-tree leaf π, find the suffix-tree leaves σ whose REVERSED
# leaf edge has maximum prefix overlap with π's leaf edge. The "reversed
# leaf edge" undoes the suffix tree's reversed-corpus build, putting σ's
# edge content in forward-corpus order.
#
# Algorithm:
#   1. Walk suffix tree, identify leaves (records that are no other record's parent).
#   2. For each leaf σ, reverse its edge tokens and insert as a path into a
#      compact "head trie" (struct-of-arrays). The terminal at the inserted
#      path's leaf records σ.id.
#   3. Walk prefix tree, identify leaves.
#   4. For each prefix leaf π with edge length L:
#        try k = L, L-1, ..., 1 (largest first):
#          walk head trie from root following π.edge[L-k], ..., π.edge[L-1]
#          if reach depth k → max overlap = k, stop
#      Then collect σ ids in the subtree under the matched head-trie node.
#
# Output: distribution stats (always) + optional binary match-index file.
# Memory: ~800MB-1GB peak on Gutenberg 5M (head trie dominates).

require "option_parser"
require "../agpt"

prefix_dir = ""
suffix_dir = ""
output_path = ""
min_overlap = 1
max_candidates = 64

OptionParser.parse do |parser|
  parser.banner = "Usage: bin/agpt_p2s_match --prefix DIR --suffix DIR [--out PATH]"
  parser.on("--prefix DIR", "Prefix radix tree dir") { |v| prefix_dir = v }
  parser.on("--suffix DIR", "Suffix radix tree dir (built with --reverse)") { |v| suffix_dir = v }
  parser.on("--out PATH", "Output match index path (optional; if absent only stats are printed)") { |v| output_path = v }
  parser.on("--min-overlap N", "Skip matches with k < N (default 1; the index will only enumerate σs for matches ≥ this threshold)") { |v| min_overlap = v.to_i }
  parser.on("--max-candidates N", "Cap on σ list size per π in output (default 64; prevents blowup at low k)") { |v| max_candidates = v.to_i }
  parser.on("-h", "--help", "Help") { puts parser; exit 0 }
end

if prefix_dir.empty? || suffix_dir.empty?
  STDERR.puts "Error: --prefix and --suffix required"
  exit 1
end

# ---------------- Phase 2a: identify suffix tree leaves ----------------
STDERR.puts "[p2s] Loading suffix tree from #{suffix_dir}"
suffix_reader = MicroGPT::AGPT::RadixTrieReader.new(suffix_dir, max_cached: 4)
STDERR.puts "[p2s]   #{suffix_reader.radix_count} radix nodes, #{suffix_reader.total_edge_chars} total edge chars"

STDERR.puts "[p2s] Pass 1: collecting suffix-tree parent ids"
suffix_is_parent = Bytes.new(suffix_reader.radix_count, 0_u8)
n_records = 0
suffix_reader.each do |r|
  if r.parent_id >= 0 && r.parent_id < suffix_is_parent.size
    suffix_is_parent[r.parent_id] = 1_u8
  end
  n_records += 1
end
STDERR.puts "[p2s]   scanned #{n_records} records"

# ---------------- Phase 2b: build head trie ----------------
STDERR.puts "[p2s] Pass 2: inserting reversed leaf edges into head trie"
node_char        = [] of Int32
node_first_child = [] of Int32
node_next_sib    = [] of Int32
node_first_term  = [] of Int32
term_sigma = [] of Int32
term_next  = [] of Int32

# Root
node_char << -1
node_first_child << -1
node_next_sib << -1
node_first_term << -1

n_suffix_leaves = 0
total_inserted_chars = 0_i64
suffix_reader.each do |r|
  next if r.id < suffix_is_parent.size && suffix_is_parent[r.id] == 1_u8
  next if r.id == 0
  # leaf — reverse edge tokens and insert
  edge = r.edge_tokens
  cur = 0
  i = edge.size - 1
  while i >= 0
    ch = edge[i]
    # find child of cur with char ch (linear scan; root has up to vocab children, deeper has few)
    child = node_first_child[cur]
    while child != -1
      break if node_char[child] == ch
      child = node_next_sib[child]
    end
    if child == -1
      new_idx = node_char.size
      node_char << ch
      node_first_child << -1
      node_next_sib << node_first_child[cur]
      node_first_term << -1
      node_first_child[cur] = new_idx
      cur = new_idx
    else
      cur = child
    end
    i -= 1
  end
  # add r.id as terminal at cur
  term_idx = term_sigma.size
  term_sigma << r.id
  term_next << node_first_term[cur]
  node_first_term[cur] = term_idx
  n_suffix_leaves += 1
  total_inserted_chars += edge.size
end

mem_bytes = (node_char.size.to_i64 * 16 + term_sigma.size.to_i64 * 8)
STDERR.puts "[p2s]   inserted #{n_suffix_leaves} suffix leaves (#{total_inserted_chars} chars)"
STDERR.puts "[p2s]   head trie: #{node_char.size} nodes, ~#{(mem_bytes / (1024 * 1024))} MB"

# ---------------- Phase 3a: identify prefix tree leaves ----------------
STDERR.puts "[p2s] Loading prefix tree from #{prefix_dir}"
prefix_reader = MicroGPT::AGPT::RadixTrieReader.new(prefix_dir, max_cached: 4)
STDERR.puts "[p2s]   #{prefix_reader.radix_count} radix nodes"

STDERR.puts "[p2s] Pass 3: collecting prefix-tree parent ids"
prefix_is_parent = Bytes.new(prefix_reader.radix_count, 0_u8)
prefix_reader.each do |r|
  if r.parent_id >= 0 && r.parent_id < prefix_is_parent.size
    prefix_is_parent[r.parent_id] = 1_u8
  end
end

# ---------------- Phase 3b: match prefix leaves against head trie ----------------
STDERR.puts "[p2s] Pass 4: matching prefix leaves..."

out : File? = nil
if !output_path.empty?
  out = File.open(output_path, "wb")
  out.write_bytes(0x50325343_u32, IO::ByteFormat::LittleEndian)  # 'P2SC' magic
  out.write_bytes(prefix_reader.radix_count, IO::ByteFormat::LittleEndian)
  out.write_bytes(suffix_reader.radix_count, IO::ByteFormat::LittleEndian)
  out.write_bytes(n_suffix_leaves, IO::ByteFormat::LittleEndian)
  STDERR.puts "[p2s]   writing match index to #{output_path} (min-overlap=#{min_overlap}, max-candidates=#{max_candidates})"
end

n_prefix_leaves = 0_i64
n_with_match    = 0_i64       # count where max_k > 0
n_at_min        = 0_i64       # count where max_k >= min_overlap
total_subtree_size = 0_i64
total_truncated_size = 0_i64
overlap_hist = Array(Int64).new(64, 0_i64)
size_hist_buckets = [0, 1, 2, 5, 20, 100, 1000, 10_000, 100_000, 1_000_000_000]
size_hist = Array(Int64).new(size_hist_buckets.size, 0_i64)

# Reusable buffers
sigma_buf = [] of Int32
stack = [] of Int32

prefix_reader.each do |r|
  next if r.id < prefix_is_parent.size && prefix_is_parent[r.id] == 1_u8
  next if r.id == 0
  edge = r.edge_tokens
  l = edge.size
  next if l == 0

  # Find max k by trying k from l down to 1 (longest-first; first hit wins)
  max_k = 0
  matched_node = -1
  k = l
  while k >= 1
    cur = 0
    success = true
    i = 0
    while i < k
      ch = edge[l - k + i]
      child = node_first_child[cur]
      while child != -1
        break if node_char[child] == ch
        child = node_next_sib[child]
      end
      if child == -1
        success = false
        break
      end
      cur = child
      i += 1
    end
    if success
      max_k = k
      matched_node = cur
      break
    end
    k -= 1
  end

  # Collect σs in subtree under matched_node (DFS), capped at max_candidates
  sigma_buf.clear
  truncated = false
  if matched_node != -1 && max_k >= min_overlap
    stack.clear
    stack << matched_node
    while !stack.empty? && sigma_buf.size < max_candidates
      n = stack.pop
      t = node_first_term[n]
      while t != -1 && sigma_buf.size < max_candidates
        sigma_buf << term_sigma[t]
        t = term_next[t]
      end
      c = node_first_child[n]
      while c != -1
        stack << c
        c = node_next_sib[c]
      end
    end
    # Note: if we hit max_candidates, the actual subtree may have more leaves;
    # we report num_sigmas as the truncated size and flag truncation in stats.
    truncated = true if !stack.empty? || (matched_node != -1 && sigma_buf.size == max_candidates)
  end

  # Stats
  overlap_hist[max_k.clamp(0, 63)] += 1
  n_prefix_leaves += 1
  n_with_match += 1 if max_k > 0
  n_at_min += 1 if max_k >= min_overlap
  if max_k >= min_overlap
    total_subtree_size += sigma_buf.size
    total_truncated_size += sigma_buf.size if truncated
  end
  # bucket histogram on subtree size
  s = sigma_buf.size
  bi = 0
  while bi < size_hist_buckets.size - 1
    break if s < size_hist_buckets[bi + 1]
    bi += 1
  end
  size_hist[bi] += 1

  # Output record
  if out
    out.write_bytes(r.id, IO::ByteFormat::LittleEndian)
    out.write_bytes(max_k, IO::ByteFormat::LittleEndian)
    out.write_bytes(sigma_buf.size, IO::ByteFormat::LittleEndian)
    sigma_buf.each { |s_id| out.write_bytes(s_id, IO::ByteFormat::LittleEndian) }
  end

  if n_prefix_leaves % 500_000 == 0
    STDERR.puts "[p2s]   processed #{n_prefix_leaves} prefix leaves..."
  end
end

out.try &.close

STDERR.puts ""
STDERR.puts "[p2s] === Match index summary ==="
STDERR.puts "[p2s]   prefix leaves total:       #{n_prefix_leaves}"
STDERR.puts "[p2s]   with any match (k≥1):      #{n_with_match} (#{(100.0 * n_with_match / n_prefix_leaves).round(2)}%)"
STDERR.puts "[p2s]   with k ≥ min_overlap=#{min_overlap}: #{n_at_min} (#{(100.0 * n_at_min / n_prefix_leaves).round(2)}%)"
mean_set = n_at_min == 0 ? 0.0 : (total_subtree_size.to_f / n_at_min)
STDERR.puts "[p2s]   mean candidate-set size (where k≥#{min_overlap}): #{mean_set.round(2)} (capped at #{max_candidates})"
STDERR.puts "[p2s]   truncated count (subtree exceeded cap): #{total_truncated_size}"
STDERR.puts ""
STDERR.puts "[p2s] Overlap k histogram:"
overlap_hist.each_with_index do |c, k|
  next if c == 0
  pct = (100.0 * c / n_prefix_leaves).round(2)
  STDERR.puts "  k=#{k.to_s.rjust(2)}: #{c.to_s.rjust(10)} (#{pct.to_s.rjust(5)}%)"
end
STDERR.puts ""
STDERR.puts "[p2s] Candidate-set size histogram (over ALL prefix leaves):"
size_labels = ["0", "1", "2-4", "5-19", "20-99", "100-999", "1k-10k", "10k-100k", "100k+"]
size_hist.each_with_index do |c, b|
  next if c == 0
  pct = (100.0 * c / n_prefix_leaves).round(2)
  STDERR.puts "  #{size_labels[b].rjust(8)}: #{c.to_s.rjust(10)} (#{pct.to_s.rjust(5)}%)"
end
