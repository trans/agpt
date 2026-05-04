require "../agpt/radix_trie_reader"
require "option_parser"

# Build a fold-target side-table for AGPT cap folding.
#
# For each cap (radix record with counts.size == 1 — degenerate H=0 target),
# find the longest tail W (w_max..w_min) such that walking W from prefix-tree
# root lands at a record endpoint with counts.size >= 2 and total mass
# >= mass_min. Store top-K (token, prob) of that record's counts as the cap's
# fold target.
#
# Output side-table format:
#   magic (u32 = 'FOLD')
#   version (u32)
#   n_radix (u32)
#   vocab_size (u32)
#   top_k (u32)
#   offsets[n_radix] (u32)   — start index into entries[]
#   lengths[n_radix] (u32)   — count (0 = no fold target)
#   entries[total]           — each: i32 token + f32 prob

include MicroGPT::AGPT

trie_dir = ""
out_path = ""
w_min = 4
w_max = 16
mass_min = 10
h_min = 0.0
top_k = 16
max_cached = 64
progress_every = 50_000
empty_only = false

OptionParser.parse do |p|
  p.banner = "Usage: agpt_build_fold_table --trie DIR --out PATH [options]"
  p.on("--trie DIR", "Prefix radix-trie directory") { |v| trie_dir = v }
  p.on("--out PATH", "Output side-table path") { |v| out_path = v }
  p.on("--w-min N", "Min tail length to attempt (default 4)") { |v| w_min = v.to_i }
  p.on("--w-max N", "Max tail length to attempt (default 16)") { |v| w_max = v.to_i }
  p.on("--mass-min N", "Min mass at W-endpoint (default 10)") { |v| mass_min = v.to_i }
  p.on("--h-min F", "Min entropy of fold-target distribution in nats (default 0; rejects near-degenerate matches)") { |v| h_min = v.to_f }
  p.on("--top-k N", "Top-K next chars per fold target (default 16)") { |v| top_k = v.to_i }
  p.on("--max-cached N", "Reader LRU depth-file cache size (default 64)") { |v| max_cached = v.to_i }
  p.on("--empty", "Skip cap scan; emit all-dead-end side-table for parity testing") { empty_only = true }
  p.on("-h", "--help", "") { puts p; exit 0 }
end

abort "missing --trie" if trie_dir.empty?
abort "missing --out" if out_path.empty?
abort "w-min must be >= 1" if w_min < 1
abort "w-max must be >= w-min" if w_max < w_min
abort "top-k must be >= 1" if top_k < 1

reader = RadixTrieReader.new(trie_dir, max_cached: max_cached)
n_radix = reader.radix_count
vocab_size = reader.vocab_size
STDERR.puts "Loaded prefix trie: #{trie_dir} (#{n_radix} nodes, vocab=#{vocab_size}, depth_files=#{reader.depth_file_count})"

# Build (parent_id → first_token → record) index for O(1) child lookup,
# and id-to-record for parent-chain path reconstruction. One pass over
# all records.
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

# Walk W from root. Returns {record, at_endpoint} or nil.
# at_endpoint=true means W consumed exactly to the end of the record's edge;
# at_endpoint=false means W exhausted mid-edge (degenerate next-char dist).
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
      return {kid, false}
    end
    parent_id = kid.id
  end
  rec = last_record
  return nil if rec.nil?
  {rec, true}
end

# Reconstruct the full root-to-cap token path by walking parent chain.
def full_path(
  cap : RadixTrieReader::LoadedRecord,
  record_by_id : Hash(Int32, RadixTrieReader::LoadedRecord)
) : Array(Int32)
  segments = [] of Array(Int32)
  cur = cap
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

# Per-cap fold targets, indexed by radix_id.
fold_lengths = Array(Int32).new(n_radix, 0)
fold_entries_by_id = {} of Int32 => Array({Int32, Float32})

caps_total = 0
caps_with_fold = 0
caps_dead_end = 0
fold_w_histogram = Hash(Int32, Int32).new(0)
sum_fold_h = 0.0
processed = 0

t_start = Time.instant

if empty_only
  STDERR.puts "Empty mode: skipping scan, all radix slots get length=0 (dead-end)."
else
reader.each do |r|
  processed += 1
  if processed % progress_every == 0
    elapsed = (Time.instant - t_start).total_seconds
    rate = processed / elapsed
    STDERR.puts "  scanned #{processed}/#{n_radix} (#{rate.round(0)} rec/s)"
  end

  next unless r.counts.size == 1

  caps_total += 1

  path = full_path(r, record_by_id)
  found = false

  w_max.downto(w_min) do |w_len|
    next if w_len > path.size
    w = path[(path.size - w_len)..]

    walk_result = walk_from_root(w, child_by_token)
    next if walk_result.nil?

    record = walk_result[0]
    at_endpoint = walk_result[1]
    next unless at_endpoint
    next if record.counts.size < 2

    total = 0
    record.counts.each { |entry| total += entry[1] }
    next if total < mass_min

    sorted = record.counts.sort_by { |entry| -entry[1] }
    take = Math.min(top_k, sorted.size)
    # Renormalize top-K to sum to 1.0 (consistent with the existing kernel's
    # target convention where target probabilities sum to 1). Truncated tail
    # mass is redistributed proportionally among the top-K.
    topk_total = 0
    take.times { |i| topk_total += sorted[i][1] }
    inv_topk = 1.0_f32 / topk_total.to_f32
    entries = Array({Int32, Float32}).new(take)
    h = 0.0
    take.times do |i|
      tok = sorted[i][0]
      prob = sorted[i][1].to_f32 * inv_topk
      entries << {tok, prob}
      h -= prob * Math.log(prob.to_f64 + 1e-30)
    end
    # Reject near-degenerate fold targets: if entropy of the post-truncation
    # top-K distribution is below the threshold, this match doesn't add
    # meaningful information beyond the cap's own one-hot. Step shorter.
    next if h < h_min

    fold_lengths[r.id] = take
    fold_entries_by_id[r.id] = entries
    fold_w_histogram[w_len] += 1
    sum_fold_h += h
    caps_with_fold += 1
    found = true
    break
  end

  caps_dead_end += 1 unless found
end
end  # if !empty_only

t_total = (Time.instant - t_start).total_seconds

STDERR.puts ""
STDERR.puts "Cap statistics:"
STDERR.puts "  caps total:    #{caps_total}"
STDERR.puts "  caps w/ fold:  #{caps_with_fold} (#{(100.0 * caps_with_fold / Math.max(1, caps_total)).round(2)}%)"
STDERR.puts "  caps dead-end: #{caps_dead_end}"
if caps_with_fold > 0
  mean_h = sum_fold_h / caps_with_fold
  STDERR.puts "  mean fold-target entropy: #{mean_h.round(4)} nats (max possible #{Math.log(vocab_size).round(4)})"
end
STDERR.puts "  W-length histogram (caps with fold):"
fold_w_histogram.keys.sort.reverse.each do |w|
  STDERR.puts "    W=#{w}: #{fold_w_histogram[w]}"
end
STDERR.puts "Scan time: #{t_total.round(2)}s"

offsets = Array(Int32).new(n_radix, 0)
total_entries = 0
n_radix.times do |i|
  offsets[i] = total_entries
  total_entries += fold_lengths[i]
end

magic = 0x444C4F46_u32  # 'FOLD'
File.open(out_path, "wb") do |io|
  io.write_bytes(magic, IO::ByteFormat::LittleEndian)
  io.write_bytes(1_i32, IO::ByteFormat::LittleEndian)
  io.write_bytes(n_radix.to_i32, IO::ByteFormat::LittleEndian)
  io.write_bytes(vocab_size.to_i32, IO::ByteFormat::LittleEndian)
  io.write_bytes(top_k.to_i32, IO::ByteFormat::LittleEndian)
  offsets.each { |o| io.write_bytes(o.to_i32, IO::ByteFormat::LittleEndian) }
  fold_lengths.each { |l| io.write_bytes(l.to_i32, IO::ByteFormat::LittleEndian) }
  n_radix.times do |i|
    next if fold_lengths[i] == 0
    fold_entries_by_id[i].each do |entry|
      io.write_bytes(entry[0].to_i32, IO::ByteFormat::LittleEndian)
      io.write_bytes(entry[1], IO::ByteFormat::LittleEndian)
    end
  end
end

bytes = File.size(out_path)
STDERR.puts "Wrote #{out_path}: #{total_entries} entries, #{bytes} bytes (#{(bytes / 1024.0 / 1024.0).round(2)} MB)"
