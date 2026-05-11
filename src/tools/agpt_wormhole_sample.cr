require "../agpt/radix_trie_reader"
require "option_parser"

# Sample wormhole-extended paths from the prefix trie.
#
# Walks the trie from root, sampling next-char at each step from the current
# node's empirical distribution. When a sampled step would enter a cap's
# unary tunnel (target is a record with counts.size==1), use the wormhole
# side-table to jump to a re-entry node (depth-1 of root for some char) and
# continue sampling from there. Stop after `--n-loops` jumps OR when the
# sampled path reaches `--max-len`.
#
# Each sampled path is a stitched-stems sequence: ~9 chars from one corpus
# location, wormhole jump, ~9 chars from another. These can be fed to a
# standard LM trainer (per-position one-hot CE) as training data.
#
# Output: one path per line, tokens as space-separated character indices.
# Optionally: a header line with the vocab.

include MicroGPT::AGPT

trie_dir = ""
wormhole_table = ""
vocab_file = ""
out_path = ""
n_samples = 100_000
max_len = 32
n_loops = 1
seed = 42
emit_text = false
no_separator = false
walk_tunnel = false

OptionParser.parse do |p|
  p.banner = "Usage: agpt_wormhole_sample --trie DIR --wormhole-table PATH --out PATH [options]"
  p.on("--trie DIR", "Prefix radix-trie directory") { |v| trie_dir = v }
  p.on("--wormhole-table PATH", "Wormhole side-table from agpt_build_wormhole_table") { |v| wormhole_table = v }
  p.on("--vocab-file PATH", "Source corpus for vocab (defaults to building from trie meta)") { |v| vocab_file = v }
  p.on("--out PATH", "Output file: one path per line (token IDs space-separated)") { |v| out_path = v }
  p.on("--n-samples N", "Number of paths to sample (default 100000)") { |v| n_samples = v.to_i }
  p.on("--max-len N", "Max path length (default 32)") { |v| max_len = v.to_i }
  p.on("--n-loops N", "Number of wormhole jumps per path (default 1)") { |v| n_loops = v.to_i }
  p.on("--seed N", "Random seed (default 42)") { |v| seed = v.to_i }
  p.on("--text", "Output as decoded text (one sample per line) instead of token IDs") { emit_text = true }
  p.on("--no-separator", "When emitting text, concatenate samples back-to-back with no newline between (so microgpt's stride=seq_len aligns with sample boundaries)") { no_separator = true }
  p.on("--walk-tunnel", "Emit cap.edge_tokens (the unary tunnel) before wormholing. Density-matched control: makes path lengths ~32 chars, isolating the routing-rule effect from the wormhole-density confound.") { walk_tunnel = true }
  p.on("-h", "--help", "") { puts p; exit 0 }
end

abort "missing --trie" if trie_dir.empty?
abort "missing --wormhole-table" if wormhole_table.empty?
abort "missing --out" if out_path.empty?

# Load prefix trie.
reader = RadixTrieReader.new(trie_dir, max_cached: 64)
n_radix = reader.radix_count
STDERR.puts "Loaded prefix trie: #{n_radix} nodes, vocab=#{reader.vocab_size}"

# Build child-by-token index.
child_by_token = Hash(Int32, Hash(Int32, RadixTrieReader::LoadedRecord)).new do |h, k|
  h[k] = {} of Int32 => RadixTrieReader::LoadedRecord
end
record_by_id = {} of Int32 => RadixTrieReader::LoadedRecord
reader.each do |r|
  child_by_token[r.parent_id][r.edge_tokens[0]] = r
  record_by_id[r.id] = r
end

# Build token → depth-1 root child id map (used to wormhole-route tokens
# at the trie's max-depth boundary, where we have counts but no child
# records).
depth1_for_token = {} of Int32 => Int32
if root_children = child_by_token[0]?
  root_children.each do |token, rec|
    depth1_for_token[token] = rec.id
  end
end
STDERR.puts "Found #{depth1_for_token.size} depth-1 root children"

# Load wormhole side-table.
# Format: magic(u32='WMHL') + version(u32) + n_radix(u32) + reserved(u32)
#       + targets[n_radix](i32)
wormhole_targets = Array(Int32).new(n_radix, -1)
File.open(wormhole_table, "rb") do |f|
  magic = f.read_bytes(UInt32, IO::ByteFormat::LittleEndian)
  abort "bad wormhole magic 0x#{magic.to_s(16)}" unless magic == 0x4C484D57_u32
  version = f.read_bytes(Int32, IO::ByteFormat::LittleEndian)
  abort "unsupported wormhole version #{version}" unless version == 1
  n_in_table = f.read_bytes(Int32, IO::ByteFormat::LittleEndian)
  abort "wormhole table n_radix=#{n_in_table} != trie n_radix=#{n_radix}" unless n_in_table == n_radix
  _reserved = f.read_bytes(Int32, IO::ByteFormat::LittleEndian)
  n_radix.times do |i|
    wormhole_targets[i] = f.read_bytes(Int32, IO::ByteFormat::LittleEndian)
  end
end
caps_with_target = wormhole_targets.count { |t| t >= 0 }
STDERR.puts "Loaded wormhole table: #{caps_with_target} caps with targets"

# Optional vocab loading for --text output.
chars_by_id : Array(Char)? = nil
if emit_text
  vf = vocab_file.empty? ? nil : vocab_file
  abort "--text requires --vocab-file" if vf.nil?
  text = File.read(vf.not_nil!)
  chars = text.chars.to_set.to_a.sort
  chars_by_id = chars
  STDERR.puts "Vocab from #{vf}: #{chars.size} chars"
end

# Sampler. From a starting node, walk-and-sample. Returns an Array(Int32) of
# tokens accumulated. Stops when max_len reached OR when n_loops_used >= n_loops
# AND we've hit another cap (graceful stop after the budgeted loops).
def pick_depth1_root_child(child_by_token : Hash(Int32, Hash(Int32, RadixTrieReader::LoadedRecord)),
                           rng : Random) : RadixTrieReader::LoadedRecord?
  rc = child_by_token[0]?
  return nil if rc.nil? || rc.empty?
  keys = rc.keys
  masses = keys.map { |t| rc[t].edge_mass.to_i64 }
  idx = sample_index(masses, rng)
  rc[keys[idx]]
end

def sample_path(start : RadixTrieReader::LoadedRecord,
                max_len : Int32, n_loops : Int32,
                wormhole_targets : Array(Int32),
                depth1_for_token : Hash(Int32, Int32),
                child_by_token : Hash(Int32, Hash(Int32, RadixTrieReader::LoadedRecord)),
                record_by_id : Hash(Int32, RadixTrieReader::LoadedRecord),
                rng : Random,
                walk_tunnel : Bool) : {Array(Int32), Int32}
  seq = [] of Int32
  current = start
  loops_used = 0

  # If we're starting from root (id=0), enter via random sample of root's
  # children (= depth-1 nodes). For other starts, current.edge_tokens are
  # already part of the "current location."
  if current.id == 0
    # Sample first child of root
    root_children = child_by_token[0]?
    return {seq, loops_used} if root_children.nil?
    return {seq, loops_used} if root_children.empty?
    rc = root_children
    # Sample by mass (weighted by edge_mass of each child)
    keys = rc.keys
    masses = keys.map { |t| rc[t].edge_mass.to_i64 }
    idx = sample_index(masses, rng)
    current = rc[keys[idx]]
    seq.concat(current.edge_tokens)
  else
    # Already positioned somewhere; record its edge tokens as part of the path.
    seq.concat(current.edge_tokens)
  end

  while seq.size < max_len
    # current is a branching internal node. Sample a child.
    if current.counts.empty?
      # Boundary with no child distribution at all (trie max-depth leaf
      # with no endpoint counts, or empty record). Wormhole back to root
      # via a default token — pick any depth-1 child by edge_mass.
      break if loops_used >= n_loops
      target = pick_depth1_root_child(child_by_token, rng)
      break if target.nil?
      loops_used += 1
      seq.concat(target.edge_tokens)
      current = target
      next
    end
    tokens = current.counts.map { |entry| entry[0] }
    counts = current.counts.map { |entry| entry[1].to_i64 }
    idx = sample_index(counts, rng)
    next_token = tokens[idx]

    # Find the next radix record for this token.
    children = child_by_token[current.id]?
    next_rec = children.nil? ? nil : children[next_token]?

    if next_rec.nil?
      # Hit the trie's stored-knowledge boundary: current has count for
      # next_token but no child record (depth-32 endpoint counts beyond
      # what the trie materializes). Wormhole-jump to depth-1 root child
      # for next_token, same as the cap-head policy. This is the V1
      # routing rule extended to all trie-boundary events.
      if loops_used < n_loops
        tid = depth1_for_token[next_token]?
        break if tid.nil?
        target = record_by_id[tid]?
        break if target.nil?
        loops_used += 1
        seq.concat(target.edge_tokens)
        current = target
        next
      else
        break
      end
    end

    # If the next record is a CAP (counts.size == 1, unary tunnel ahead),
    # we DO NOT walk into its edge_tokens — that's the identity tunnel we
    # want to skip. Instead, intercept at the cap head: wormhole-jump
    # using the cap's pre-mapped target (which is depth-1 of root for the
    # cap head token). The wormhole target's edge_tokens naturally emit
    # that head char as part of the new path.
    if next_rec.counts.size == 1
      if loops_used < n_loops
        target_id = wormhole_targets[next_rec.id]
        cap_target : RadixTrieReader::LoadedRecord? = nil
        if target_id >= 0
          cap_target = record_by_id[target_id]?
        end
        # Fallback: if the side-table has no target for this cap (e.g., V2
        # caps where suffix-entropy boundary couldn't be determined), use
        # V1-style routing — depth-1 root child for the cap head token.
        if cap_target.nil?
          head_char = next_rec.edge_tokens[0]
          tid = depth1_for_token[head_char]?
          cap_target = tid.nil? ? nil : record_by_id[tid]?
        end
        break if cap_target.nil?
        # Optional control: walk through the cap's unary tunnel before
        # wormholing. Matches synth_wrap path lengths (~32 chars). Pure
        # density control — same routing rule applied at the same boundary
        # node, but with the unary tunnel chars emitted instead of skipped.
        if walk_tunnel
          seq.concat(next_rec.edge_tokens)
        end
        loops_used += 1
        seq.concat(cap_target.edge_tokens)
        current = cap_target
        next
      else
        # Out of loop budget; stop without entering the tunnel.
        break
      end
    end

    # Internal branching child: walk into it normally.
    seq.concat(next_rec.edge_tokens)
    current = next_rec
  end

  # Truncate to max_len.
  seq = seq[0, max_len] if seq.size > max_len
  {seq, loops_used}
end

# Sample an index from a discrete distribution given by weights.
def sample_index(weights : Array(Int64), rng : Random) : Int32
  total = weights.sum
  return 0 if total <= 0
  threshold = rng.rand(total.to_i64)
  cum = 0_i64
  weights.each_with_index do |w, i|
    cum += w
    return i if cum > threshold
  end
  weights.size - 1
end

# Get the root record (virtual id=0). The reader's records start at id=1
# (root is implicit). We synthesize a "virtual root" record for sampling.
# Walk children_by_token[0] to find depth-1 records.
root = RadixTrieReader::LoadedRecord.new(0, -1, 0, [] of Int32, 0, [] of {Int32, Int32})

# Generate samples.
rng = Random.new(seed)
t_start = Time.instant
File.open(out_path, "w") do |out_file|
  total_chars = 0
  total_loops = 0_i64
  n_samples.times do |i|
    path, loops_in_path = sample_path(root, max_len, n_loops, wormhole_targets,
                                      depth1_for_token, child_by_token, record_by_id,
                                      rng, walk_tunnel)
    if emit_text && (cb = chars_by_id)
      text_path = String.build { |io| path.each { |t| io << cb[t] } }
      if no_separator
        out_file.print text_path
      else
        out_file.puts text_path
      end
    else
      out_file.puts path.join(" ")
    end
    total_chars += path.size
    total_loops += loops_in_path

    if (i + 1) % 10_000 == 0
      elapsed = (Time.instant - t_start).total_seconds
      rate = (i + 1) / elapsed
      avg_len = total_chars.to_f / (i + 1).to_f
      STDERR.puts "  #{i + 1}/#{n_samples} samples (#{rate.round(0)}/s, avg_len=#{avg_len.round(1)})"
    end
  end
  STDERR.puts "Total chars: #{total_chars}"
  STDERR.puts "Total wormholes: #{total_loops}"
  if total_loops > 0
    avg_seg = total_chars.to_f / total_loops.to_f
    STDERR.puts "Avg chars per wormhole transition: #{avg_seg.round(2)}"
  end
end

t_total = (Time.instant - t_start).total_seconds
STDERR.puts "Sampled #{n_samples} paths in #{t_total.round(2)}s"
STDERR.puts "Wrote #{out_path} (#{File.size(out_path)} bytes)"
