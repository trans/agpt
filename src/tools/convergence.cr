# Trie Path Probability Convergence Experiment
# Standalone tool — do not wire into main.cr
#
# Usage:
#   crystal run src/tools/convergence.cr -- --depth 4
#   crystal run src/tools/convergence.cr -- --depth 8
#   crystal build src/tools/convergence.cr -o bin/convergence

require "option_parser"

# ---------------------------------------------------------------------------
# Trie node — simple struct-based trie stored as flat array
# ---------------------------------------------------------------------------

class TrieNode
  property count : Int32
  property next_counts : Hash(Int32, Int32)
  property children : Hash(Int32, Int32)  # token -> node index

  def initialize
    @count = 0
    @next_counts = Hash(Int32, Int32).new(0)
    @children = Hash(Int32, Int32).new
  end
end

class Trie
  property nodes : Array(TrieNode)

  def initialize
    @nodes = [TrieNode.new]  # index 0 = root
  end

  def root
    @nodes[0]
  end

  # Insert one sliding window starting at position `pos` in tokens, up to max_depth
  def insert_at(tokens : Array(Int32), pos : Int32, max_depth : Int32)
    node_idx = 0
    @nodes[node_idx].count += 1

    d = 0
    while d < max_depth && pos + d < tokens.size
      token = tokens[pos + d]
      # record next token in next_counts
      if pos + d + 1 < tokens.size
        next_tok = tokens[pos + d + 1]
        @nodes[node_idx].next_counts[next_tok] = @nodes[node_idx].next_counts.fetch(next_tok, 0) + 1
      end

      # descend / create child
      child_idx = @nodes[node_idx].children.fetch(token, -1)
      if child_idx == -1
        child_idx = @nodes.size
        @nodes << TrieNode.new
        @nodes[node_idx].children[token] = child_idx
      end
      node_idx = child_idx
      @nodes[node_idx].count += 1
      d += 1
    end
  end

  def build(tokens : Array(Int32), max_depth : Int32)
    tokens.each_index do |pos|
      insert_at(tokens, pos, max_depth)
    end
  end
end

# ---------------------------------------------------------------------------
# Log-space probability with Laplace smoothing
# ---------------------------------------------------------------------------

VOCAB_SIZE = 256
EPSILON    = 1e-6

def log_prob(next_counts : Hash(Int32, Int32), count : Int32, token : Int32) : Float64
  raw = next_counts.fetch(token, 0).to_f64
  denom = count.to_f64 + VOCAB_SIZE * EPSILON
  Math.log((raw + EPSILON) / denom)
end

# ---------------------------------------------------------------------------
# CSV output record
# ---------------------------------------------------------------------------

record PathRecord,
  depth : Int32,
  log_pi_a : Float64,
  log_pi_b : Float64,
  log_pi_mean : Float64,
  log_abs_err : Float64,
  count_a : Int32,
  count_b : Int32,
  log_pi_min : Float64,
  path : Array(Int32)

# ---------------------------------------------------------------------------
# DFS shared path walk
# ---------------------------------------------------------------------------

def walk_shared(
  nodes_a : Array(TrieNode),
  nodes_b : Array(TrieNode),
  idx_a : Int32,
  idx_b : Int32,
  depth : Int32,
  max_depth : Int32,
  log_pi_a : Float64,
  log_pi_b : Float64,
  log_pi_min : Float64,
  results : Array(PathRecord),
  min_count : Int32,
  path : Array(Int32)
)
  node_a = nodes_a[idx_a]
  node_b = nodes_b[idx_b]

  return if node_a.count < min_count || node_b.count < min_count
  return if depth > max_depth

  # Record this node as a path (depth >= 1 means at least one token consumed)
  if depth >= 1
    lmax = Math.max(log_pi_a, log_pi_b)
    log_pi_mean = lmax + Math.log(Math.exp(log_pi_a - lmax) + Math.exp(log_pi_b - lmax)) - Math.log(2.0)
    log_abs_err = (log_pi_a - log_pi_b).abs

    results << PathRecord.new(
      depth: depth,
      log_pi_a: log_pi_a,
      log_pi_b: log_pi_b,
      log_pi_mean: log_pi_mean,
      log_abs_err: log_abs_err,
      count_a: node_a.count,
      count_b: node_b.count,
      log_pi_min: log_pi_min,
      path: path.dup
    )
  end

  return if depth >= max_depth

  # Recurse into shared children
  node_a.children.each do |token, child_a_idx|
    child_b_idx = node_b.children.fetch(token, -1)
    next if child_b_idx == -1

    child_a = nodes_a[child_a_idx]
    child_b = nodes_b[child_b_idx]
    next if child_a.count < min_count || child_b.count < min_count

    lp_a = log_prob(node_a.next_counts, node_a.count, token)
    lp_b = log_prob(node_b.next_counts, node_b.count, token)

    new_log_pi_a = log_pi_a + lp_a
    new_log_pi_b = log_pi_b + lp_b

    edge_min = Math.min(lp_a, lp_b)
    new_log_pi_min = Math.min(log_pi_min, edge_min)

    path << token
    walk_shared(
      nodes_a, nodes_b,
      child_a_idx, child_b_idx,
      depth + 1, max_depth,
      new_log_pi_a, new_log_pi_b,
      new_log_pi_min,
      results,
      min_count,
      path
    )
    path.pop
  end
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

max_depth = 4
corpus_path = "data/input.txt"
output_dir = "rnd"
min_count = 5
n_blocks = 20

OptionParser.parse do |parser|
  parser.banner = "Usage: convergence [options]"
  parser.on("--depth D", "Max trie depth (default: 4)") { |d| max_depth = d.to_i }
  parser.on("--corpus PATH", "Corpus file (default: data/input.txt)") { |p| corpus_path = p }
  parser.on("--output-dir DIR", "Output directory (default: rnd)") { |d| output_dir = d }
  parser.on("--min-count N", "Min count threshold (default: 5)") { |n| min_count = n.to_i }
  parser.on("--blocks N", "Number of interleaved blocks for split (default: 20)") { |n| n_blocks = n.to_i }
  parser.on("-h", "--help", "Show help") { puts parser; exit 0 }
end

STDERR.puts "[convergence] depth=#{max_depth}, corpus=#{corpus_path}, min_count=#{min_count}, blocks=#{n_blocks}"

# --- Read and tokenize corpus ---
STDERR.puts "[convergence] Reading corpus..."
raw = File.read(corpus_path)
tokens = raw.bytes.map { |b| b.to_i32 }
STDERR.puts "[convergence] Corpus size: #{tokens.size} tokens, vocab up to 256"

# --- Interleaved block split ---
block_size = tokens.size // n_blocks
STDERR.puts "[convergence] Block size: #{block_size}, n_blocks: #{n_blocks}"

tokens_a = Array(Int32).new
tokens_b = Array(Int32).new

n_blocks.times do |i|
  start_idx = i * block_size
  end_idx = (i == n_blocks - 1) ? tokens.size : (i + 1) * block_size
  block = tokens[start_idx...end_idx]
  if i.even?
    tokens_a.concat(block)
  else
    tokens_b.concat(block)
  end
end

STDERR.puts "[convergence] Corpus A: #{tokens_a.size} tokens, Corpus B: #{tokens_b.size} tokens"

# --- Build tries ---
STDERR.puts "[convergence] Building trie A (depth=#{max_depth})..."
t0 = Time.instant
trie_a = Trie.new
trie_a.build(tokens_a, max_depth)
STDERR.puts "[convergence] Trie A built in #{(Time.instant - t0).total_seconds.round(2)}s, #{trie_a.nodes.size} nodes"

STDERR.puts "[convergence] Building trie B (depth=#{max_depth})..."
t0 = Time.instant
trie_b = Trie.new
trie_b.build(tokens_b, max_depth)
STDERR.puts "[convergence] Trie B built in #{(Time.instant - t0).total_seconds.round(2)}s, #{trie_b.nodes.size} nodes"

# --- Walk shared paths ---
STDERR.puts "[convergence] Walking shared paths..."
t0 = Time.instant
results = Array(PathRecord).new

walk_shared(
  trie_a.nodes, trie_b.nodes,
  0, 0,
  0, max_depth,
  0.0, 0.0,
  Float64::INFINITY,
  results,
  min_count,
  [] of Int32
)

STDERR.puts "[convergence] Walk complete in #{(Time.instant - t0).total_seconds.round(2)}s, #{results.size} shared paths"

# --- Write CSV ---
Dir.mkdir_p(output_dir)
csv_path = "#{output_dir}/results_D#{max_depth}.csv"
STDERR.puts "[convergence] Writing #{csv_path}..."

File.open(csv_path, "w") do |f|
  f.puts "depth,log_pi_a,log_pi_b,log_pi_mean,log_abs_err,count_a,count_b,log_pi_min,path"
  results.each do |r|
    lpm = r.log_pi_min.infinite? ? 0.0 : r.log_pi_min
    path_str = r.path.map { |b| b.chr.inspect[1..-2] }.join
    f.puts "#{r.depth},#{r.log_pi_a},#{r.log_pi_b},#{r.log_pi_mean},#{r.log_abs_err},#{r.count_a},#{r.count_b},#{lpm},\"#{path_str}\""
  end
end

STDERR.puts "[convergence] Done. #{results.size} records written to #{csv_path}"

# Quick summary to stdout
by_depth = Hash(Int32, Array(Float64)).new { |h, k| h[k] = Array(Float64).new }
results.each { |r| by_depth[r.depth] << r.log_abs_err }

puts "\nSummary for depth=#{max_depth}:"
puts "%-8s %-12s %-12s %-12s" % ["depth", "n_paths", "median_err", "p90_err"]
by_depth.keys.sort.each do |d|
  errs = by_depth[d].sort
  n = errs.size
  median = n > 0 ? errs[n // 2] : 0.0
  p90 = n > 0 ? errs[(n * 9) // 10] : 0.0
  puts "%-8d %-12d %-12.4f %-12.4f" % [d, n, median, p90]
end

# Print the deepest surviving paths
deepest = results.max_by { |r| r.depth }
max_d = deepest.depth
puts "\nDeepest surviving paths (depth=#{max_d}, count_a>=#{min_count} and count_b>=#{min_count}):"
puts "%-6s %-8s %-8s  %s" % ["depth", "count_a", "count_b", "path"]
results.select { |r| r.depth == max_d }
       .sort_by { |r| -r.count_a }
       .each do |r|
  path_str = String.new(Bytes.new(r.path.size) { |i| r.path[i].to_u8 })
  puts "%-6d %-8d %-8d  %s" % [r.depth, r.count_a, r.count_b, path_str.inspect]
end
