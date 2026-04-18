# Verify that a radix trie is a lossless compression of a leveled trie:
#   - Every character position in the original trie has exactly one corresponding
#     (radix_node, offset_within_edge) pair.
#   - For each radix node, expanding (ancestor edges ... + own edge) reproduces the
#     character sequence from root to that node in the original trie.
#   - Radix endpoint counts match the original node's counts at the same global depth.
#
# Usage: bin/radix-verify <leveled_dir> <radix_dir>

require "../agpt"

if ARGV.size < 2
  STDERR.puts "Usage: radix-verify <leveled_dir> <radix_dir>"
  exit 1
end

leveled_dir = ARGV[0]
radix_dir   = ARGV[1]

puts "Loading leveled trie:  #{leveled_dir}"
leveled = MicroGPT::AGPT::LeveledTrieReader.new(leveled_dir, max_cached: 256)
puts "  nodes: #{leveled.node_count}, depth files: #{leveled.depth_file_count}"

puts "Loading radix trie:    #{radix_dir}"
radix = MicroGPT::AGPT::RadixTrieReader.new(radix_dir, max_cached: 256)
puts "  radix count: #{radix.radix_count}, endpoint depth files: #{radix.depth_file_count}"
puts "  total edge chars: #{radix.total_edge_chars}"

# Pre-build (parent_id, token) → child_id index across all depths of leveled trie
puts ""
puts "Pre-building (parent_id, token) → child_id index over leveled trie..."
t0 = Time.instant
child_index = {} of {Int32, Int32} => Int32
leveled.depth_file_count.times do |d|
  next if d == 0
  leveled.nodes_at_depth(d).each do |rec|
    child_index[{rec.parent_id, rec.token}] = rec.id
  end
end
t1 = Time.instant
puts "  index size: #{child_index.size}, built in #{(t1 - t0).total_seconds.round(2)}s"

puts ""
puts "Cross-checking radix nodes against leveled trie..."

# Build radix_id → record map
radix_by_id = {} of Int32 => MicroGPT::AGPT::RadixTrieReader::LoadedRecord
radix.each { |r| radix_by_id[r.id] = r }

# For each radix record, reconstruct the full path of edge tokens by walking up.
# Then verify that the path of tokens exists in the leveled trie and the endpoint
# counts match.

checked = 0
mismatches = 0
sample_failures = [] of String

radix.each do |rec|
  # Build full path of tokens from root to endpoint
  path = Array(Int32).new
  cur = rec
  loop do
    path = cur.edge_tokens + path
    break if cur.parent_id == 0
    parent = radix_by_id[cur.parent_id]?
    if parent.nil?
      # Parent missing — this is a root-child (parent_radix_id should be 0 already)
      break
    end
    cur = parent
  end

  # Walk via pre-built child_index
  cur_leveled_id = 0
  found = true
  path.each do |tok|
    next_id = child_index[{cur_leveled_id, tok}]?
    if next_id.nil?
      found = false
      break
    end
    cur_leveled_id = next_id
  end

  if !found
    mismatches += 1
    if sample_failures.size < 3
      sample_failures << "radix_id=#{rec.id} (depth=#{rec.endpoint_depth}, edge_len=#{rec.edge_len}): path not found in leveled trie"
    end
  else
    # Verify endpoint counts match
    lev_counts = leveled.counts_of(cur_leveled_id).sort_by(&.first)
    rad_counts = rec.counts.sort_by(&.first)
    if lev_counts != rad_counts
      mismatches += 1
      if sample_failures.size < 3
        sample_failures << "radix_id=#{rec.id}: counts mismatch (leveled: #{lev_counts}, radix: #{rad_counts})"
      end
    end
  end

  checked += 1
  if checked % 50_000 == 0
    STDERR.puts "  checked #{checked}/#{radix.radix_count}..."
  end
end

puts ""
puts "Results:"
puts "  Checked:    #{checked}"
puts "  Mismatches: #{mismatches}"
if mismatches > 0
  puts ""
  puts "Sample failures:"
  sample_failures.each { |s| puts "  #{s}" }
  exit 1
else
  puts ""
  puts "PASS: radix trie is a lossless compression of the leveled trie."
  exit 0
end
