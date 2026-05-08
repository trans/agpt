require "../agpt/radix_trie_reader"
require "option_parser"

# "Poor man's generator" via prefix-tree walk with cap-following.
#
# Algorithm:
#   1. Walk root → cap by mass-weighted child-pick at each branching node.
#   2. At a cap, EMIT the cap's edge_tokens (the unary tunnel chars).
#   3. Wormhole to depth-1 root for cap.edge_tokens[0] (the cap head char).
#   4. Use cap.edge_tokens[1..] as a DIRECTIVE: walk into the depth-1 subtree
#      following these chars. Each step finds the child whose edge_tokens[0]
#      matches the next directive char, consumes matching chars from the
#      directive, advances. Emit the chars walked over.
#   5. When directive exhausted (or no matching child): resume mass-weighted
#      branching from current position. Loop.
#
# d-as-parrot-knob observation (user): with d ~> first_char_depth, cap edges
# are long and they're corpus substrings, so they find verbatim matches in
# the depth-1 subtree → output gets very corpus-verbatim. Smaller d → more
# branching, more diverse generation.

include MicroGPT::AGPT

trie_dir = ""
vocab_path = "data/input.txt"
n_chars = 1000
seed = 42_u64
trace = false

OptionParser.parse do |p|
  p.banner = "Usage: agpt_parrot_sample --trie DIR [options]"
  p.on("--trie DIR", "Prefix radix-trie directory") { |v| trie_dir = v }
  p.on("--vocab PATH", "Vocab text (default data/input.txt)") { |v| vocab_path = v }
  p.on("--n N", "Number of chars to generate (default 1000)") { |v| n_chars = v.to_i }
  p.on("--seed N", "RNG seed (default 42)") { |v| seed = v.to_u64 }
  p.on("--trace", "Per-step diagnostic to stderr") { trace = true }
  p.on("-h", "--help", "") { puts p; exit 0 }
end

abort "missing --trie" if trie_dir.empty?

reader = RadixTrieReader.new(trie_dir, max_cached: 64)
n_radix = reader.radix_count

# Index: parent_id × first_token → record
child_by_token = Hash(Int32, Hash(Int32, RadixTrieReader::LoadedRecord)).new do |h, k|
  h[k] = {} of Int32 => RadixTrieReader::LoadedRecord
end
record_by_id = {} of Int32 => RadixTrieReader::LoadedRecord
reader.each do |r|
  child_by_token[r.parent_id][r.edge_tokens[0]] = r
  record_by_id[r.id] = r
end

# Vocab
chars = File.read(vocab_path).chars.to_set.to_a.sort

def decode(toks : Array(Int32), chars : Array(Char)) : String
  String.build do |io|
    toks.each do |t|
      ch = chars[t]?
      io << (ch ? ch : '?')
    end
  end
end

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

# Pick a depth-1 child by mass.
def pick_depth1(child_by_token : Hash(Int32, Hash(Int32, RadixTrieReader::LoadedRecord)),
                rng : Random) : RadixTrieReader::LoadedRecord?
  rc = child_by_token[0]?
  return nil if rc.nil?
  keys = rc.keys
  weights = keys.map { |t| rc[t].edge_mass.to_i64 }
  idx = sample_index(weights, rng)
  rc[keys[idx]]
end

# Pick a branching child by mass.
def pick_child(node : RadixTrieReader::LoadedRecord,
               child_by_token : Hash(Int32, Hash(Int32, RadixTrieReader::LoadedRecord)),
               rng : Random) : RadixTrieReader::LoadedRecord?
  return nil if node.counts.empty?
  toks = node.counts.map { |e| e[0] }
  cnts = node.counts.map { |e| e[1].to_i64 }
  idx = sample_index(cnts, rng)
  next_tok = toks[idx]
  kids = child_by_token[node.id]?
  return nil if kids.nil?
  kids[next_tok]?
end

# Walk into a subtree using `directive` chars as path — POSITION UPDATE
# ONLY, no emission. The directive chars correspond to a sequence already
# emitted via the cap-edge walk; this function moves the trie position to
# the corresponding node in the new subtree so subsequent mass-weighted
# branching resumes from there.
#
# If we get stuck (no matching child, or partial mid-edge match), we stop
# at the last valid trie node — that node becomes the resumption point.
def follow_directive(start : RadixTrieReader::LoadedRecord,
                     directive : Array(Int32),
                     child_by_token : Hash(Int32, Hash(Int32, RadixTrieReader::LoadedRecord))
                     ) : Tuple(RadixTrieReader::LoadedRecord, Int32, Array(Int32))
  current = start
  d_idx = 0
  overshoot = [] of Int32  # chars from last kid's edge past where directive ran out
  while d_idx < directive.size
    kids = child_by_token[current.id]?
    break if kids.nil?
    kid = kids[directive[d_idx]]?
    break if kid.nil?
    edge = kid.edge_tokens
    # "If radix compressed, follow." Match directive against kid's edge as
    # far as both have chars; if the matched prefix agrees, advance to kid
    # in full. If directive runs out partway through kid's compressed edge,
    # we capture the unmatched suffix as `overshoot` for the caller to emit
    # — those chars are corpus-grounded (they're on the trie path we walked)
    # and should appear in output to extend naturally without phantom stitch.
    match_len = Math.min(edge.size, directive.size - d_idx)
    ok = true
    match_len.times do |i|
      if edge[i] != directive[d_idx + i]
        ok = false
        break
      end
    end
    break unless ok
    d_idx += match_len
    current = kid
    if match_len < edge.size
      # Directive ran out before kid's edge ended. Capture the unmatched
      # suffix so caller can emit it.
      overshoot = edge[match_len..]
    end
  end
  {current, directive.size - d_idx, overshoot}
end

rng = Random.new(seed)
output_tokens = [] of Int32
current : RadixTrieReader::LoadedRecord? = nil

# Initial walk: depth-1 by mass.
init = pick_depth1(child_by_token, rng)
abort "trie has no depth-1 children" if init.nil?
current = init
init.edge_tokens.each { |t| output_tokens << t }

no_progress_count = 0
last_size = 0
while output_tokens.size < n_chars
  if output_tokens.size == last_size
    no_progress_count += 1
    if no_progress_count > 100
      STDERR.puts "[parrot] aborting: 100 iterations without emission" if trace
      break
    end
  else
    no_progress_count = 0
    last_size = output_tokens.size
  end
  cur = current
  break if cur.nil?
  if trace
    last20 = output_tokens.size < 20 ? output_tokens : output_tokens[(output_tokens.size - 20)..]
    STDERR.puts "[step] cur.id=#{cur.id} counts.size=#{cur.counts.size} edge=#{decode(cur.edge_tokens, chars).inspect} last=#{decode(last20, chars).inspect}"
  end
  if cur.counts.size == 1
    cap_edge = cur.edge_tokens
    head = cap_edge[0]
    target = child_by_token[0]?.try &.[head]?
    if target.nil?
      break
    end
    # Move position to target = depth-1 root child whose edge starts with
    # `head`. Do NOT emit target.edge_tokens — those chars are already in
    # output as the leading chars of cap_edge (we just emitted those via
    # the walk-into-cap above). Re-emitting them would produce phantom
    # duplicates like "CapuleC" instead of the intended "Capule" position
    # change. The directive is the cap_edge tail past whatever target's
    # edge already conceptually "covers" (typically 1 char = cap_head).
    target_edge = target.edge_tokens
    skip = target_edge.size
    if skip <= cap_edge.size
      directive = cap_edge[skip..]
    else
      directive = [] of Int32
    end
    if directive.empty?
      current = target
    else
      ending, leftover, overshoot = follow_directive(target, directive, child_by_token)
      overshoot.each { |t| output_tokens << t }
      current = ending
    end
    # Wormhole-loop detection: if directive walked back to the very cap we
    # came from (happens when cap_edge is a unique-in-corpus substring
    # whose only occurrence IS this cap's path), the wormhole hasn't
    # redirected anywhere. Break the loop by emitting a sample from the
    # cap's counts (the corpus's actual post-cap continuation) and
    # wormhole to depth-1 root for that char — same handling as
    # depth-limit-boundary.
    if current.id == cur.id
      if !cur.counts.empty?
        idx = sample_index(cur.counts.map { |e| e[1].to_i64 }, rng)
        next_tok = cur.counts[idx][0]
        output_tokens << next_tok
        nxt_target = (child_by_token[0]?.try &.[next_tok]?)
        if nxt_target.nil?
          break
        end
        current = nxt_target
      else
        break
      end
    end
  else
    # Branching: pick child by mass.
    # If the sampled token has no child record (we're at a max-depth
    # branching node whose counts point past the trie's depth), wormhole:
    # emit the sampled token, position-update to depth-1 root for it (no
    # extra emission — same overlap rule as cap-wormhole), continue.
    break if cur.counts.empty?
    toks = cur.counts.map { |e| e[0] }
    cnts = cur.counts.map { |e| e[1].to_i64 }
    idx = sample_index(cnts, rng)
    next_tok = toks[idx]
    kids = child_by_token[cur.id]?
    nxt = (kids.nil? ? nil : kids[next_tok]?)
    if nxt.nil?
      # Max-depth branching boundary: emit the sampled char (it IS a corpus
      # continuation per cur's counts), then wormhole to depth-1 root for it
      # without re-emitting — overlapping node counts as one.
      output_tokens << next_tok
      target = (child_by_token[0]?.try &.[next_tok]?)
      break if target.nil?
      current = target
    else
      nxt.edge_tokens.each { |t| output_tokens << t }
      current = nxt
    end
  end
end

# Truncate
output_tokens = output_tokens[0, n_chars] if output_tokens.size > n_chars
puts decode(output_tokens, chars)
