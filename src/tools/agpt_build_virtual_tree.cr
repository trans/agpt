require "../agpt/radix_trie_reader"
require "option_parser"

# Build a virtual-tree side-table for AGPT cap-tunnel expansion.
#
# For each cap (radix record with counts.size == 1) of edge length L,
# emit min(expansion_depth, L) composite distributions — one per tunnel
# position p ∈ [0, expansion_depth). Each composite is a length-weighted
# mixture of shifted-prefix walks where the shift-1 walk goes
# path[1 : cap_edge_start + p + 1] from root, the shift-2 walk drops two
# leading chars, and so on up to a maximum shift (default = h-1, where
# h = cap_edge_start). Weight per shift = α^(shift-1) (geometric decay).
#
# Output format ('VTRE' v1):
#   magic (u32 = 'VTRE')
#   version (u32 = 1)
#   n_radix (u32)
#   vocab_size (u32)
#   top_k (u32)
#   expansion_depth (u32)
#   offsets[n_radix * expansion_depth] (u32)
#   lengths[n_radix * expansion_depth] (u32)
#   entries[total]                          # each: i32 token + f32 prob
#
# Index a cap c at tunnel position p via slot = c * expansion_depth + p.
# Non-cap radix nodes have lengths[*] = 0; caps with edge_len < expansion_depth
# also have lengths[*] = 0 for positions p >= edge_len.
#
# Usage:
#   bin/agpt_build_virtual_tree --trie /tmp/agpt_input_d32_radix \
#     --out /tmp/agpt_vtree_d32.bin \
#     --expansion-depth 3 --shift-max 8 --alpha 0.5

include MicroGPT::AGPT

trie_dir = ""
out_path = ""
expansion_depth = 3
shift_min = 1
shift_max = 0  # 0 = auto: cap.first_char_depth - 1 (walk down to 1-char min)
mass_min = 2
top_k = 16
alpha = 0.5
max_cached = 64
progress_every = 50_000

OptionParser.parse do |p|
  p.banner = "Usage: agpt_build_virtual_tree --trie DIR --out PATH [options]"
  p.on("--trie DIR", "Prefix radix-trie directory") { |v| trie_dir = v }
  p.on("--out PATH", "Output side-table path") { |v| out_path = v }
  p.on("--expansion-depth N", "Tunnel positions per cap to emit (default 3)") { |v| expansion_depth = v.to_i }
  p.on("--shift-min N", "Min shift (default 1; shift=1 keeps cap_edge_start chars of context)") { |v| shift_min = v.to_i }
  p.on("--shift-max N", "Max shift (default 0 = auto: cap_edge_start - 1, walks down to 1-char)") { |v| shift_max = v.to_i }
  p.on("--mass-min N", "Min total mass at shifted-walk endpoint (default 2)") { |v| mass_min = v.to_i }
  p.on("--top-k N", "Top-K next chars per composite (default 16)") { |v| top_k = v.to_i }
  p.on("--alpha F", "Geometric weighting decay; weight per shift k = alpha^(k-1) (default 0.5; smaller = sharper preference for long walks)") { |v| alpha = v.to_f }
  p.on("--max-cached N", "Reader LRU cache size (default 64)") { |v| max_cached = v.to_i }
  p.on("-h", "--help", "") { puts p; exit 0 }
end

abort "missing --trie" if trie_dir.empty?
abort "missing --out" if out_path.empty?
abort "expansion-depth must be >= 1" if expansion_depth < 1
abort "shift-min must be >= 1" if shift_min < 1
abort "top-k must be >= 1" if top_k < 1
abort "alpha must be in (0, 1]" unless 0.0 < alpha && alpha <= 1.0

reader = RadixTrieReader.new(trie_dir, max_cached: max_cached)
n_radix = reader.radix_count
vocab_size = reader.vocab_size
STDERR.puts "Loaded prefix trie: #{trie_dir} (#{n_radix} nodes, vocab=#{vocab_size})"

# Index for O(1) child-by-token lookup and parent-chain reconstruction.
child_by_token = Hash(Int32, Hash(Int32, RadixTrieReader::LoadedRecord)).new do |h, k|
  h[k] = {} of Int32 => RadixTrieReader::LoadedRecord
end
record_by_id = {} of Int32 => RadixTrieReader::LoadedRecord

t_idx_start = Time.instant
reader.each do |r|
  child_by_token[r.parent_id][r.edge_tokens[0]] = r
  record_by_id[r.id] = r
end
STDERR.puts "Indexed #{n_radix} records in #{(Time.instant - t_idx_start).total_seconds.round(2)}s"

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

# Per-(cap, position) composite distributions, keyed by slot = cap_id * expansion_depth + position.
slot_lengths = Array(Int32).new(n_radix * expansion_depth, 0)
slot_entries = {} of Int32 => Array({Int32, Float32})

caps_total = 0
caps_with_any_pos = 0
positions_filled = 0
positions_attempted = 0
shift_contrib_count = Hash(Int32, Int32).new(0)  # how many slots got contributions from shift k
shift_total_weight = 0.0
sum_h = 0.0
processed = 0

t_start = Time.instant

reader.each do |cap|
  processed += 1
  if processed % progress_every == 0
    elapsed = (Time.instant - t_start).total_seconds
    rate = processed / elapsed
    STDERR.puts "  scanned #{processed}/#{n_radix} (#{rate.round(0)} rec/s, #{caps_total} caps so far)"
  end

  next unless cap.counts.size == 1
  caps_total += 1

  path = full_path(cap, record_by_id)
  cap_edge_start_0idx = cap.first_char_depth - 1   # 0-indexed start of cap edge in path
  edge_len = cap.edge_tokens.size

  # Effective shift max for this cap.
  eff_shift_max = shift_max > 0 ? shift_max : (cap.first_char_depth - 1)
  next if eff_shift_max < shift_min

  # Per-position composite. Skip positions outside the actual edge length.
  pos_filled_for_this_cap = 0
  positions_to_emit = Math.min(expansion_depth, edge_len)
  positions_to_emit.times do |p|
    positions_attempted += 1

    # Walks for tunnel position p PREDICT the char at path index
    # (cap_edge_start_0idx + p). The walk thus stops JUST BEFORE that
    # index — it walks from shift to (cap_edge_start_0idx + p) exclusive.
    # Reading the distribution at the walk endpoint then gives
    #   P(char at predicted position | walked-substring),
    # which is the corpus-aggregated story for the cap-edge char at
    # position p instead of the trie's degenerate one-hot.
    walk_end = cap_edge_start_0idx + p
    composite = Hash(Int32, Float64).new(0.0)
    total_weight = 0.0

    (shift_min..eff_shift_max).each do |k|
      walk_start = k
      next if walk_start >= walk_end
      w = path[walk_start...walk_end]
      next if w.empty?

      walk_result = walk_from_root(w, child_by_token)
      next if walk_result.nil?
      rec = walk_result[0]
      at_endpoint = walk_result[1]
      next unless at_endpoint
      next if rec.counts.size < 2

      total = 0
      rec.counts.each { |entry| total += entry[1] }
      next if total < mass_min

      weight = alpha ** (k - 1)
      shift_contrib_count[k] += 1
      shift_total_weight += weight

      inv_total = 1.0 / total.to_f64
      rec.counts.each do |entry|
        tok = entry[0]
        prob = entry[1].to_f64 * inv_total
        composite[tok] += weight * prob
      end
      total_weight += weight
    end

    next if total_weight <= 0 || composite.empty?

    # Normalize composite.
    inv_w = 1.0 / total_weight
    composite.transform_values! { |v| v * inv_w }

    # Top-K with renormalization.
    sorted = composite.to_a.sort_by { |e| -e[1] }
    take = Math.min(top_k, sorted.size)
    topk_total = 0.0
    take.times { |i| topk_total += sorted[i][1] }
    next if topk_total <= 0
    inv_topk = 1.0 / topk_total
    entries = Array({Int32, Float32}).new(take)
    h = 0.0
    take.times do |i|
      tok = sorted[i][0]
      prob = (sorted[i][1] * inv_topk).to_f32
      entries << {tok, prob}
      pf = prob.to_f64
      h -= pf * Math.log(pf + 1e-30) if pf > 0
    end

    slot = cap.id * expansion_depth + p
    slot_lengths[slot] = take
    slot_entries[slot] = entries
    positions_filled += 1
    pos_filled_for_this_cap += 1
    sum_h += h
  end

  caps_with_any_pos += 1 if pos_filled_for_this_cap > 0
end

t_total = (Time.instant - t_start).total_seconds

STDERR.puts ""
STDERR.puts "Virtual-tree statistics:"
STDERR.puts "  expansion_depth:       #{expansion_depth}"
STDERR.puts "  caps total:            #{caps_total}"
STDERR.puts "  caps w/ any position:  #{caps_with_any_pos} (#{(100.0 * caps_with_any_pos / Math.max(1, caps_total)).round(2)}%)"
STDERR.puts "  positions attempted:   #{positions_attempted}"
STDERR.puts "  positions filled:      #{positions_filled} (#{(100.0 * positions_filled / Math.max(1, positions_attempted)).round(2)}%)"
if positions_filled > 0
  STDERR.puts "  mean composite entropy: #{(sum_h / positions_filled).round(4)} nats (max #{Math.log(vocab_size).round(4)})"
end
STDERR.puts "  shift contribution counts (how often each shift k contributed to a slot):"
shift_contrib_count.keys.sort.each do |k|
  STDERR.puts "    shift=#{k}: #{shift_contrib_count[k]}"
end
STDERR.puts "  scan time: #{t_total.round(2)}s"

# Build offsets and write.
offsets = Array(Int32).new(n_radix * expansion_depth, 0)
total_entries = 0
(n_radix * expansion_depth).times do |i|
  offsets[i] = total_entries
  total_entries += slot_lengths[i]
end

magic = 0x45525456_u32  # 'VTRE'
File.open(out_path, "wb") do |io|
  io.write_bytes(magic, IO::ByteFormat::LittleEndian)
  io.write_bytes(1_i32, IO::ByteFormat::LittleEndian)
  io.write_bytes(n_radix.to_i32, IO::ByteFormat::LittleEndian)
  io.write_bytes(vocab_size.to_i32, IO::ByteFormat::LittleEndian)
  io.write_bytes(top_k.to_i32, IO::ByteFormat::LittleEndian)
  io.write_bytes(expansion_depth.to_i32, IO::ByteFormat::LittleEndian)
  offsets.each { |o| io.write_bytes(o.to_i32, IO::ByteFormat::LittleEndian) }
  slot_lengths.each { |l| io.write_bytes(l.to_i32, IO::ByteFormat::LittleEndian) }
  (n_radix * expansion_depth).times do |i|
    next if slot_lengths[i] == 0
    slot_entries[i].each do |entry|
      io.write_bytes(entry[0].to_i32, IO::ByteFormat::LittleEndian)
      io.write_bytes(entry[1], IO::ByteFormat::LittleEndian)
    end
  end
end

bytes = File.size(out_path)
STDERR.puts "Wrote #{out_path}: #{total_entries} entries, #{bytes} bytes (#{(bytes / 1024.0 / 1024.0).round(2)} MB)"
