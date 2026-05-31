require "../agpt/radix_trie_reader"
require "option_parser"

# Build a backoff-target side-table for the AGPT slot-selection (Step 0)
# experiment (notes/seq-len-extension/slot-selection.md).
#
# For each prefix-trie radix node K at endpoint depth d, the tool computes
# B backoff target IDs:
#
#   K_back_i = the radix node whose path equals K's-path-with-first-i-chars-dropped
#              (a length-(d-i) suffix of K's full path), or SENTINEL if no
#              such node exists in the trie.
#
# For Step 0's pd=0 stash-and-gather scheme, the trainer reads this sidecar
# at startup, builds an inverse `rev_lookup[M] -> list of (K, i)`, and uses
# it to scatter h_p[K_back_i] into K's stash slots during the fire.
#
# Algorithm: direct descent from root per (K, i) pair. For each backoff level
# i, take K's path suffix c_{i+1}..c_d and descend the radix trie matching
# chars against compressed edges. Three outcomes:
#
#   - Suffix lands exactly on a radix node endpoint -> store that node's ID.
#   - Suffix lands mid-edge of a compressed node (case 2) -> SENTINEL.
#     Mass=1 intra-edge positions aren't in v1's K/V cache, so there's
#     no hidden state to gather. We measure the case-2 rate here.
#   - Suffix is empty (K's depth < i+1, "shallow K") -> SENTINEL. K simply
#     doesn't have enough path to back off this far.
#
# (For Step 0, this single-tree descent approach was chosen over the
# dual-tree substring-catalog route — see slot-selection.md for the
# rationale: smaller blast radius, no dependency on suffix-tree +
# SubstringCatalog artifacts.)
#
# Output side-table format (binary, little-endian):
#   magic     u32 = 'BKOF' (0x464F4B42)
#   version   u32 = 1
#   n_radix   u32   number of radix nodes (must match the trie)
#   b         u32   backoff levels per node (B)
#   case2     u64   total case-2 sentinels recorded (diagnostic)
#   shallow   u64   total shallow-K sentinels (i >= K.depth) (diagnostic)
#   sidecar   u32[n_radix * b]
#                   sidecar[k * b + i] = K_back_(i+1).id for radix node k,
#                   or UINT32_MAX (0xFFFFFFFF) if no such node.
#
# Storage on Shakespeare d=16, B=4: 1.6M * 4 * 4 bytes = ~25 MB.

include MicroGPT::AGPT

SENTINEL_ID = 0xFFFFFFFF_u32

trie_dir = ""
out_path = ""
b = 4
max_cached = 64
progress_every = 100_000

OptionParser.parse do |p|
  p.banner = "Usage: agpt_build_backoff_table --trie DIR --out PATH [--b N]"
  p.on("--trie DIR", "Prefix radix-trie directory") { |v| trie_dir = v }
  p.on("--out PATH", "Output side-table path") { |v| out_path = v }
  p.on("--b N", "Backoff levels per node (default 4)") { |v| b = v.to_i }
  p.on("--max-cached N", "Reader LRU depth-file cache size (default 64)") { |v| max_cached = v.to_i }
  p.on("-h", "--help", "") { puts p; exit 0 }
end

abort "missing --trie" if trie_dir.empty?
abort "missing --out" if out_path.empty?
abort "--b must be >= 1" if b < 1

reader = RadixTrieReader.new(trie_dir, max_cached: max_cached)
n_radix = reader.radix_count
STDERR.puts "Loaded prefix trie: #{trie_dir} (#{n_radix} nodes, vocab=#{reader.vocab_size}, depth_files=#{reader.depth_file_count})"

# Build (parent_id -> first_token -> record) index for O(1) child lookup,
# and id-to-record for parent-chain path reconstruction. One pass over
# all records. Same pattern as agpt_build_fold_table.
child_by_token = Hash(Int32, Hash(Int32, RadixTrieReader::LoadedRecord)).new do |h, k|
  h[k] = {} of Int32 => RadixTrieReader::LoadedRecord
end
record_by_id = {} of Int32 => RadixTrieReader::LoadedRecord

t_index_start = Time.instant
indexed = 0
reader.each do |r|
  child_by_token[r.parent_id][r.edge_tokens[0]] = r
  record_by_id[r.id] = r
  indexed += 1
end
t_index = (Time.instant - t_index_start).total_seconds
STDERR.puts "Indexed #{indexed} records in #{t_index.round(2)}s"

# Walk a token sequence w starting from root.
# Returns {record, at_endpoint} or nil. at_endpoint=true means w consumed
# exactly to the end of a radix node's edge -> that's the node for this path.
# at_endpoint=false means we exhausted w mid-edge of some node -> case 2.
# nil means the path doesn't exist (a char mismatch or missing child).
def walk_from_root(
  w : Array(Int32),
  child_by_token : Hash(Int32, Hash(Int32, RadixTrieReader::LoadedRecord))
) : Tuple(RadixTrieReader::LoadedRecord, Bool)?
  return nil if w.empty?
  parent_id = 0
  pos = 0
  last_record : RadixTrieReader::LoadedRecord? = nil
  while pos < w.size
    kids = child_by_token[parent_id]?
    return nil if kids.nil?
    kid = kids[w[pos]]?
    return nil unless kid
    edge_len = kid.edge_tokens.size
    max_consume = Math.min(edge_len, w.size - pos)
    max_consume.times do |i|
      return nil if kid.edge_tokens[i] != w[pos + i]
    end
    pos += max_consume
    last_record = kid
    if max_consume < edge_len
      # w ran out mid-edge: case 2
      return {kid, false}
    end
    parent_id = kid.id
  end
  rec = last_record
  return nil if rec.nil?
  {rec, true}
end

# Reconstruct root-to-node token path by walking the parent chain.
# Returns chars c_1..c_d for a node at endpoint depth d.
def full_path(
  node : RadixTrieReader::LoadedRecord,
  record_by_id : Hash(Int32, RadixTrieReader::LoadedRecord)
) : Array(Int32)
  segments = [] of Array(Int32)
  cur = node
  loop do
    segments << cur.edge_tokens
    break if cur.parent_id == 0
    parent = record_by_id[cur.parent_id]?
    raise "parent #{cur.parent_id} missing for record #{cur.id}" if parent.nil?
    cur = parent
  end
  out = [] of Int32
  segments.reverse_each { |s| s.each { |t| out << t } }
  out
end

# Sidecar: u32[n_radix * b]. Allocate as Slice for memcpy-friendly write.
sidecar = Slice(UInt32).new(n_radix.to_i64 * b.to_i64, SENTINEL_ID)

case2_count = 0_u64
shallow_count = 0_u64
not_found_count = 0_u64  # shouldn't happen by sliding-window construction; instrument to verify
endpoint_count = 0_u64

t_scan_start = Time.instant
processed = 0

reader.each do |k|
  processed += 1
  if progress_every > 0 && processed % progress_every == 0
    elapsed = (Time.instant - t_scan_start).total_seconds
    rate = processed / Math.max(elapsed, 1e-6)
    STDERR.puts "  scanned #{processed}/#{n_radix} (#{rate.round(0)} nodes/s, #{elapsed.round(1)}s)"
  end

  # K's full path (length = d = K.endpoint_depth + 1).
  path = full_path(k, record_by_id)
  d = path.size

  b.times do |i|
    level = i + 1  # backoff level 1..B
    if level >= d
      # Shallow K: not enough path to back off this far. Suffix would be
      # empty or root (zero context); not useful as a K/V slot for K.
      shallow_count += 1
      next
    end

    suffix = path[level, d - level]  # path[i+1..d-1] -> length d-i
    walk = walk_from_root(suffix, child_by_token)
    if walk.nil?
      # Path doesn't exist in trie. Should be impossible by sliding-window
      # construction (every length-(d-i) substring of the corpus is a path).
      # Instrument to verify.
      not_found_count += 1
      next
    end
    record, at_endpoint = walk
    if at_endpoint
      sidecar[k.id.to_i64 * b.to_i64 + i] = record.id.to_u32
      endpoint_count += 1
    else
      # Case 2: suffix lands mid-edge of a compressed node.
      case2_count += 1
    end
  end
end

t_scan = (Time.instant - t_scan_start).total_seconds
total_slots = n_radix.to_u64 * b.to_u64
filled = endpoint_count
sentinels = total_slots - filled
STDERR.puts ""
STDERR.puts "Backoff scan complete in #{t_scan.round(2)}s:"
STDERR.puts "  total slots:    #{total_slots} (#{n_radix} nodes x B=#{b})"
STDERR.puts "  filled:         #{filled}  (#{(100.0 * filled / total_slots).round(2)}%)"
STDERR.puts "  case-2 mid-edge: #{case2_count}  (#{(100.0 * case2_count / total_slots).round(2)}%)"
STDERR.puts "  shallow K:      #{shallow_count}  (#{(100.0 * shallow_count / total_slots).round(2)}%)"
STDERR.puts "  not-found:      #{not_found_count}  (sliding-window guarantee; should be 0)"

# Write sidecar.
magic = 0x464F4B42_u32  # 'BKOF'
File.open(out_path, "wb") do |io|
  io.write_bytes(magic, IO::ByteFormat::LittleEndian)
  io.write_bytes(1_u32, IO::ByteFormat::LittleEndian)            # version
  io.write_bytes(n_radix.to_u32, IO::ByteFormat::LittleEndian)
  io.write_bytes(b.to_u32, IO::ByteFormat::LittleEndian)
  io.write_bytes(case2_count, IO::ByteFormat::LittleEndian)
  io.write_bytes(shallow_count, IO::ByteFormat::LittleEndian)
  # Sidecar payload — slice of u32 LE. Write as raw bytes (host is LE).
  io.write(Bytes.new(sidecar.to_unsafe.as(Pointer(UInt8)), sidecar.bytesize))
end

bytes = File.size(out_path)
STDERR.puts "Wrote #{out_path}: #{bytes} bytes (#{(bytes / 1024.0 / 1024.0).round(2)} MB)"
