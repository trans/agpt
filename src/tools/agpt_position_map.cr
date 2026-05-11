require "../agpt"
require "../agpt/radix_trie_reader"
require "option_parser"

# Phase 0 of the seq_len decoupling work: build the
# `corpus_position → radix_node` map and report what we find.
#
# For each position p in [0, N - d) (where d is the trie's max endpoint
# depth), walks the d-window from root through the radix trie and
# records the landing node and depth reached. Reports per-depth and
# per-mass distributions so we can see how corpus positions cluster
# across the tree.
#
# Optional: dumps a binary array `pos_to_node[p] = radix_id` for use by
# downstream decoupled-attention experiments.

include MicroGPT::AGPT

trie_dir = ""
file = ""
out_path = ""
out_inverse = ""
sample_n = 0  # 0 = scan whole corpus
verbose_samples = 0
emit_full_contributions = true  # record every node a position contributes to, not just leaf

OptionParser.parse do |p|
  p.banner = "Usage: agpt_position_map --trie DIR --file PATH [--out PATH] [--sample N]"
  p.on("--trie DIR", "Prefix radix trie directory") { |v| trie_dir = v }
  p.on("--file PATH", "Corpus text file (same one the trie was built on)") { |v| file = v }
  p.on("--out PATH", "Optional: per-position contributions binary (count + radix_ids)") { |v| out_path = v }
  p.on("--out-inverse PATH", "Optional: per-node positions binary (node_id → [positions])") { |v| out_inverse = v }
  p.on("--leaf-only", "Record only the deepest landing node per position (old behavior)") { emit_full_contributions = false }
  p.on("--sample N", "Walk only first N positions instead of full corpus") { |v| sample_n = v.to_i }
  p.on("--verbose-samples N", "Print N example position→node landings to stderr") { |v| verbose_samples = v.to_i }
  p.on("-h", "--help", "") { puts p; exit 0 }
end

raise "--trie required" if trie_dir.empty?
raise "--file required" if file.empty?

# Read corpus and tokenize using the same CharDataset the radix builder used.
# Otherwise our walk steps won't match the trie's token-id edges.
text = File.read(file)
dataset = MicroGPT::CharDataset.new(text)
corpus_tokens = dataset.data  # Array(Int32) of length |text|
STDERR.puts "Corpus: #{file} (#{corpus_tokens.size} tokens, vocab=#{dataset.vocab_size})"

# Load radix trie.
reader = RadixTrieReader.new(trie_dir, max_cached: 256)
n_radix = reader.radix_count
vocab_size = reader.vocab_size
STDERR.puts "Trie: #{trie_dir} (#{n_radix} nodes, vocab=#{vocab_size}, depth_files=#{reader.depth_file_count})"

# Build (parent_id → first_token → record) index for O(1) child lookup.
child_by_token = Hash(Int32, Hash(Int32, RadixTrieReader::LoadedRecord)).new do |h, k|
  h[k] = {} of Int32 => RadixTrieReader::LoadedRecord
end
record_by_id = {} of Int32 => RadixTrieReader::LoadedRecord

t_index_start = Time.monotonic
reader.each do |r|
  child_by_token[r.parent_id][r.edge_tokens[0]] = r
  record_by_id[r.id] = r
end
t_index = (Time.monotonic - t_index_start).total_seconds
STDERR.puts "Indexed in #{t_index.round(2)}s"

# Determine d (= maximum endpoint depth across all radix records).
max_endpoint_depth = 0
record_by_id.each_value { |r| max_endpoint_depth = r.endpoint_depth if r.endpoint_depth > max_endpoint_depth }
d_max = max_endpoint_depth
STDERR.puts "Max endpoint depth in trie: d=#{d_max}"

# How many positions to walk?
n_positions = corpus_tokens.size - d_max
n_positions = sample_n if sample_n > 0 && sample_n < n_positions
STDERR.puts "Walking #{n_positions} corpus positions (window=#{d_max})"

# Open optional output.
out_io : IO::FileDescriptor? = nil
if !out_path.empty?
  out_io = File.open(out_path, "wb")
  STDERR.puts "Writing position→node_id binary to: #{out_path}"
end

# Statistics.
node_position_count = Hash(Int32, Int32).new(0)   # node → distinct terminal-position contributions
nodes_per_position_hist = Hash(Int32, Int64).new(0_i64)
nodes_per_position_sum = 0_i64
fall_off_count = 0_i64
no_root_child_count = 0_i64

t_start = Time.monotonic

# Two-pass approach.
# Pass 1: outer loop over starting positions s. For each, walk forward up to
# d_max chars and emit (terminal_corpus_position, radix_node_id) contributions.
# Aggregate per terminal corpus position.
n_corpus = corpus_tokens.size
n_terminal = sample_n > 0 && sample_n < n_corpus ? sample_n : n_corpus
per_position_nodes = Array(Array(Int32)).new(n_terminal) { [] of Int32 }
STDERR.puts "Pass 1: walking from each of #{n_corpus} starting positions, emitting (terminal_position, node) contributions"

s = 0
while s < n_corpus
  parent_id = 0
  pos_off = 0
  while pos_off < d_max && (s + pos_off) < n_corpus
    next_char = corpus_tokens[s + pos_off]
    kids = child_by_token[parent_id]?
    if kids.nil?
      no_root_child_count += 1 if parent_id == 0
      break
    end
    kid = kids[next_char]?
    break if kid.nil?
    edge_len = kid.edge_tokens.size
    max_can = Math.min(edge_len, Math.min(d_max - pos_off, n_corpus - (s + pos_off)))
    match_len = 0
    max_can.times do |i|
      if kid.edge_tokens[i] == corpus_tokens[s + pos_off + i]
        match_len += 1
      else
        break
      end
    end
    if match_len > 0
      terminal_pos = s + pos_off + match_len - 1
      # Attribute this radix node to its TERMINAL corpus position
      # (where its path ends). Only record if within sample limit.
      if terminal_pos < n_terminal
        per_position_nodes[terminal_pos] << kid.id
      end
    end
    pos_off += match_len
    if match_len < edge_len
      fall_off_count += 1
      break
    end
    parent_id = kid.id
  end
  s += 1
  if s % 500000 == 0
    elapsed = (Time.monotonic - t_start).total_seconds
    rate = s / elapsed
    STDERR.puts "  walked from #{s}/#{n_corpus} starting positions (#{rate.round(0)} pos/s)"
  end
end

t_walk = (Time.monotonic - t_start).total_seconds
STDERR.puts "Pass 1 complete in #{t_walk.round(2)}s"

# Pass 2: emit stats and optional binary output.
STDERR.puts "Pass 2: aggregating stats and writing output"
processed = 0_i64
verbose_left = verbose_samples

n_terminal.times do |p|
  contribs = per_position_nodes[p]
  k = contribs.size
  nodes_per_position_hist[k] += 1
  nodes_per_position_sum += k
  contribs.each { |nid| node_position_count[nid] += 1 }

  if verbose_left > 0
    snippet_len = Math.min(d_max, 24)
    start_show = Math.max(0, p - snippet_len + 1)
    snippet = String.build do |sb|
      (start_show..p).each do |i|
        c = dataset.id_to_char[corpus_tokens[i]]
        sb << ((c.ord >= 32 && c.ord < 127) ? c : '?')
      end
    end
    path_str = contribs.map { |nid|
      r = record_by_id[nid]
      "#{nid}(d#{r.endpoint_depth},m#{r.edge_mass})"
    }.join(", ")
    STDERR.puts "  p=#{p} context_ending_at_p=\"#{snippet}\" k=#{k} reps: #{path_str}"
    verbose_left -= 1
  end

  if oi = out_io
    # Per-position record: [k:int32] [nid_1, ..., nid_k : int32]
    oi.write_bytes(k.to_i32, IO::ByteFormat::LittleEndian)
    contribs.each { |nid| oi.write_bytes(nid, IO::ByteFormat::LittleEndian) }
  end

  processed += 1
end

if oi = out_io
  oi.close
end

t_total = (Time.monotonic - t_start).total_seconds
STDERR.puts "Walked #{processed} positions in #{t_total.round(2)}s"

# === Reports ===

puts ""
puts "## Contributions-per-corpus-position distribution (radix-node count where path terminates at p)"
puts "  k     count       pct"
total = processed.to_f
nodes_per_position_hist.to_a.sort_by(&.[0]).each do |k, count|
  pct = (count.to_f / total * 100)
  bar = "*" * (pct * 0.5).to_i
  puts "%5d  %10d  %5.2f%%  %s" % [k, count, pct, bar]
end
puts "  mean: %.2f radix nodes representing each corpus position" % (nodes_per_position_sum.to_f / total)
puts "  upper bound (d_max): #{d_max}"

puts ""
puts "## Aggregates"
puts "  total terminal positions counted: #{processed}"
puts "  walks falling off mid-edge:       #{fall_off_count}"
puts "  walks with no root child:         #{no_root_child_count}"
puts "  unique nodes attributed:          #{node_position_count.size}"
puts "  total (position, node) contributions: #{nodes_per_position_sum}"

# Per-node: count of position-contributions must match trie's stored edge_mass.
# This verifies our walk records the same contributions the builder did.
mismatches = 0
mismatch_examples = [] of {Int32, Int32, Int32}
node_position_count.each do |nid, contribs|
  m = record_by_id[nid].edge_mass
  if contribs != m
    mismatches += 1
    mismatch_examples << {nid, contribs, m} if mismatch_examples.size < 5
  end
end
total_pos = node_position_count.values.sum
total_mass = 0_i64
node_position_count.each_key { |id| total_mass += record_by_id[id].edge_mass }
puts ""
puts "## Mass consistency check"
puts "  sum of position-contributions: #{total_pos}"
puts "  sum of trie edge_mass for those nodes: #{total_mass}"
puts "  ratio (contributions / mass):             #{(total_pos.to_f / total_mass * 100).round(2)}%"
puts "  per-node count mismatches (contribution != edge_mass): #{mismatches}"
if !mismatch_examples.empty?
  puts "  examples (node_id, contribs, trie_mass):"
  mismatch_examples.each { |t| puts "    #{t}" }
end

puts ""
puts "## Top 10 nodes by position count"
puts "node_id    positions  mass     depth  edge_len"
node_position_count.to_a.sort_by { |_, c| -c }.first(10).each do |id, c|
  rec = record_by_id[id]
  puts "%-10d %-10d %-8d %-6d %d" % [id, c, rec.edge_mass, rec.endpoint_depth, rec.edge_len]
end
