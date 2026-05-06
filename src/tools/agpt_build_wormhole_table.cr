require "../agpt/radix_trie_reader"
require "option_parser"

# Build a wormhole side-table for the dumb-baseline experiment.
#
# For each cap (radix endpoint at the trie's max_depth with counts.size == 1),
# this tool emits a wormhole edge to a re-entry target — another node in the
# prefix trie. The trainer consumes this side-table to extend cap events with
# one wormhole hop.
#
# Variants (selectable via --variant):
#   v1  cap → depth-1 root child for the cap's FIRST character (zero suffix info)
#   v2  cap → depth-1 root child for the cap's BOUNDARY character (the first
#       position in the cap path where suffix entropy crosses a threshold;
#       single-bit suffix info)
#
# Output format:
#   magic (u32 = 'WMHL')
#   version (u32)
#   n_radix (u32)                # total radix nodes in the source trie
#   reserved (u32)               # for future flags
#   targets[n_radix] (i32)       # per radix_id, the wormhole target's radix_id;
#                                # -1 = no wormhole (non-cap or no valid target)
#
# Usage:
#   agpt_build_wormhole_table --trie DIR --out PATH [--variant v1|v2]
#                              [--suffix-trie DIR]    # required for v2

include MicroGPT::AGPT

trie_dir = ""
suffix_trie_dir = ""
out_path = ""
variant = "v1"
entropy_threshold = 0.1   # used by v2 to mark "suffix entropy crosses 0"
max_cached = 64

OptionParser.parse do |p|
  p.banner = "Usage: agpt_build_wormhole_table --trie DIR --out PATH [options]"
  p.on("--trie DIR", "Prefix radix-trie directory") { |v| trie_dir = v }
  p.on("--suffix-trie DIR", "Suffix radix-trie (required for --variant v2)") { |v| suffix_trie_dir = v }
  p.on("--out PATH", "Output side-table path") { |v| out_path = v }
  p.on("--variant V", "v1 (first-char) | v2 (boundary-char) (default v1)") { |v| variant = v }
  p.on("--entropy-threshold F", "v2: minimum entropy in nats to consider suffix-dense (default 0.1)") { |v| entropy_threshold = v.to_f }
  p.on("--max-cached N", "Reader LRU cache size (default 64)") { |v| max_cached = v.to_i }
  p.on("-h", "--help", "") { puts p; exit 0 }
end

abort "missing --trie" if trie_dir.empty?
abort "missing --out" if out_path.empty?
abort "v2 requires --suffix-trie" if variant == "v2" && suffix_trie_dir.empty?
abort "unknown variant #{variant}" unless variant == "v1" || variant == "v2"

reader = RadixTrieReader.new(trie_dir, max_cached: max_cached)
n_radix = reader.radix_count
STDERR.puts "Loaded prefix trie: #{trie_dir} (#{n_radix} radix nodes, vocab=#{reader.vocab_size}, depths=#{reader.depth_file_count})"

# Build child-by-parent index for walking. Same pattern as fold-table builder.
child_by_token = Hash(Int32, Hash(Int32, RadixTrieReader::LoadedRecord)).new do |h, k|
  h[k] = {} of Int32 => RadixTrieReader::LoadedRecord
end
record_by_id = {} of Int32 => RadixTrieReader::LoadedRecord
reader.each do |r|
  child_by_token[r.parent_id][r.edge_tokens[0]] = r
  record_by_id[r.id] = r
end
STDERR.puts "Indexed records."

# Build a token → depth-1 record_id lookup. Children of root (parent_id=0) are
# the depth-1 records.
depth1_for_token = {} of Int32 => Int32
if root_children = child_by_token[0]?
  root_children.each do |token, rec|
    depth1_for_token[token] = rec.id
  end
end
STDERR.puts "Found #{depth1_for_token.size} depth-1 root children"

# Reconstruct full path from root to a record by walking parents.
def full_path(rec : RadixTrieReader::LoadedRecord,
              record_by_id : Hash(Int32, RadixTrieReader::LoadedRecord)) : Array(Int32)
  segments = [] of Array(Int32)
  cur = rec
  loop do
    segments << cur.edge_tokens
    break if cur.parent_id == 0
    parent = record_by_id[cur.parent_id]?
    raise "missing parent #{cur.parent_id}" if parent.nil?
    cur = parent
  end
  out = [] of Int32
  segments.reverse_each { |s| s.each { |t| out << t } }
  out
end

# Optional suffix-trie loader for v2.
suffix_reader : RadixTrieReader? = nil
suffix_child_by_token = Hash(Int32, Hash(Int32, RadixTrieReader::LoadedRecord)).new do |h, k|
  h[k] = {} of Int32 => RadixTrieReader::LoadedRecord
end
if variant == "v2"
  suffix_reader = RadixTrieReader.new(suffix_trie_dir, max_cached: max_cached)
  STDERR.puts "Loaded suffix trie: #{suffix_trie_dir} (#{suffix_reader.not_nil!.radix_count} nodes)"
  suffix_reader.not_nil!.each do |r|
    suffix_child_by_token[r.parent_id][r.edge_tokens[0]] = r
  end
end

# v2 helper: walk a substring through the suffix trie from root, return the
# node's children-distribution entropy (in nats). Returns 0.0 if the walk
# fails (substring absent) or if the node is a singleton (mass=1).
def suffix_entropy(substr : Array(Int32),
                   suffix_child_by_token : Hash(Int32, Hash(Int32, RadixTrieReader::LoadedRecord))) : Float64
  return 0.0 if substr.empty?
  parent_id = 0
  pos = 0
  last_record : RadixTrieReader::LoadedRecord? = nil
  while pos < substr.size
    children = suffix_child_by_token[parent_id]?
    return 0.0 if children.nil?
    kid = children[substr[pos]]?
    return 0.0 unless kid
    edge_len = kid.edge_tokens.size
    max_consume = Math.min(edge_len, substr.size - pos)
    max_consume.times do |i|
      return 0.0 if kid.edge_tokens[i] != substr[pos + i]
    end
    pos += max_consume
    last_record = kid
    break if max_consume < edge_len
    parent_id = kid.id
  end
  rec = last_record
  return 0.0 if rec.nil?
  return 0.0 if rec.counts.size <= 1
  total = 0
  rec.counts.each { |entry| total += entry[1] }
  return 0.0 if total <= 0
  h = 0.0
  rec.counts.each do |entry|
    p = entry[1].to_f64 / total.to_f64
    h -= p * Math.log(p) if p > 0
  end
  h
end

# Walk over caps and build the wormhole table.
targets = Array(Int32).new(n_radix, -1)
caps_total = 0
caps_with_target = 0
caps_no_target = 0
boundary_depths = Hash(Int32, Int32).new(0) if variant == "v2"

t_start = Time.instant
processed = 0
reader.each do |r|
  processed += 1
  if processed % 50_000 == 0
    elapsed = (Time.instant - t_start).total_seconds
    STDERR.puts "  scanned #{processed}/#{n_radix} (#{(processed / elapsed).round(0)} rec/s)"
  end

  next unless r.counts.size == 1
  caps_total += 1

  path = full_path(r, record_by_id)
  if path.size == 0
    caps_no_target += 1
    next
  end

  case variant
  when "v1"
    # Re-entry target = depth-1 root child for the CAP HEAD token, i.e.,
    # the first char of the cap's unary edge (= edge_tokens[0]). This is
    # the char at depth first_char_depth, where the prefix-tree's branching
    # has died and the identity tunnel begins. Route by *that* boundary
    # character, NOT the depth-1 char of the whole sequence (path[0]).
    head_char = r.edge_tokens[0]
    target_id = depth1_for_token[head_char]?
    if target_id
      targets[r.id] = target_id
      caps_with_target += 1
    else
      caps_no_target += 1
    end
  when "v2"
    # Walk suffix-tree with progressively-LONGER tails of the cap path
    # (k=1, 2, 3, ...). Suffix entropy is high at small k (where many
    # predecessors are possible) and drops to 0 once k extends past the
    # suffix's "decision depth" into the unique-substring zone. We track
    # the LARGEST k where entropy is still ≥ threshold — that's the
    # deepest tail still in the suffix-dense zone.
    #
    # The "boundary char" is the char at position (path_size - last_valid_k)
    # in the cap path — i.e., the char at the boundary between the cap's
    # suffix-dense and suffix-identity zones. We route to depth-1 of root
    # for that char.
    path_size = path.size
    last_valid_k = 0
    (1..path_size).each do |k|
      tail = path[(path_size - k)..]
      tail_reversed = tail.reverse
      h = suffix_entropy(tail_reversed, suffix_child_by_token)
      if h >= entropy_threshold
        last_valid_k = k
      else
        # entropy dropped below threshold; we've crossed into identity zone.
        # last_valid_k holds the deepest still-dense tail length.
        break
      end
    end
    if last_valid_k > 0 && last_valid_k < path_size
      # Boundary depth in cap (1-indexed from cap start) = path_size - last_valid_k + 1.
      # Boundary char = path[path_size - last_valid_k] (0-indexed).
      boundary_pos = path_size - last_valid_k
      boundary_char = path[boundary_pos]
      target_id = depth1_for_token[boundary_char]?
      if target_id
        targets[r.id] = target_id
        caps_with_target += 1
        boundary_depths.not_nil![boundary_pos + 1] += 1
      else
        caps_no_target += 1
      end
    else
      caps_no_target += 1
    end
  end
end

t_total = (Time.instant - t_start).total_seconds
STDERR.puts ""
STDERR.puts "Wormhole side-table stats (variant=#{variant}):"
STDERR.puts "  caps total:       #{caps_total}"
STDERR.puts "  caps with target: #{caps_with_target} (#{(100.0 * caps_with_target / Math.max(1, caps_total)).round(2)}%)"
STDERR.puts "  caps no target:   #{caps_no_target}"
if variant == "v2" && boundary_depths
  STDERR.puts "  boundary depth histogram (depth-from-start of cap path):"
  boundary_depths.keys.sort.each do |d|
    STDERR.puts "    d=#{d}: #{boundary_depths[d]}"
  end
end
STDERR.puts "  Scan time: #{t_total.round(2)}s"

# Write side-table.
magic = 0x4C484D57_u32  # 'WMHL'
File.open(out_path, "wb") do |io|
  io.write_bytes(magic, IO::ByteFormat::LittleEndian)
  io.write_bytes(1_i32, IO::ByteFormat::LittleEndian)         # version
  io.write_bytes(n_radix.to_i32, IO::ByteFormat::LittleEndian)
  io.write_bytes(0_i32, IO::ByteFormat::LittleEndian)         # reserved
  targets.each { |t| io.write_bytes(t.to_i32, IO::ByteFormat::LittleEndian) }
end

bytes = File.size(out_path)
STDERR.puts "Wrote #{out_path} (#{bytes} bytes, #{(bytes / 1024.0 / 1024.0).round(2)} MB)"
