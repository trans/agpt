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

OptionParser.parse do |p|
  p.banner = "Usage: agpt_parrot_sample --trie DIR [options]"
  p.on("--trie DIR", "Prefix radix-trie directory") { |v| trie_dir = v }
  p.on("--vocab PATH", "Vocab text (default data/input.txt)") { |v| vocab_path = v }
  p.on("--n N", "Number of chars to generate (default 1000)") { |v| n_chars = v.to_i }
  p.on("--seed N", "RNG seed (default 42)") { |v| seed = v.to_u64 }
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

# Walk into a subtree using `directive` chars as path. Emit chars walked.
# Returns {emitted_tokens, ending_node, leftover_directive_count}.
#
# If we get stuck (no matching child, or partial mid-edge match), we DO NOT
# emit phantom chars. We stop emission at the last valid trie node — that
# node becomes the resumption point for mass-weighted walk in the caller.
# This avoids producing chars that aren't a continuation of any corpus path.
def follow_directive(start : RadixTrieReader::LoadedRecord,
                     directive : Array(Int32),
                     child_by_token : Hash(Int32, Hash(Int32, RadixTrieReader::LoadedRecord))
                     ) : Tuple(Array(Int32), RadixTrieReader::LoadedRecord, Int32)
  emitted = [] of Int32
  current = start
  d_idx = 0
  while d_idx < directive.size
    kids = child_by_token[current.id]?
    break if kids.nil?
    kid = kids[directive[d_idx]]?
    break if kid.nil?
    edge = kid.edge_tokens
    # Verify the entire kid's edge matches the upcoming directive chars.
    full_match = true
    if d_idx + edge.size > directive.size
      # Directive too short to fully consume kid's edge — would land mid-edge.
      full_match = false
    else
      edge.size.times do |i|
        if edge[i] != directive[d_idx + i]
          full_match = false
          break
        end
      end
    end
    break unless full_match
    # Fully matched kid's edge. Advance.
    edge.each { |t| emitted << t }
    d_idx += edge.size
    current = kid
  end
  {emitted, current, directive.size - d_idx}
end

rng = Random.new(seed)
output_tokens = [] of Int32
current : RadixTrieReader::LoadedRecord? = nil

# Initial walk: depth-1 by mass.
init = pick_depth1(child_by_token, rng)
abort "trie has no depth-1 children" if init.nil?
current = init
init.edge_tokens.each { |t| output_tokens << t }

while output_tokens.size < n_chars
  cur = current
  break if cur.nil?
  if cur.counts.size == 1
    # Cap. Emit the cap edge (we are AT the cap, which is a node whose own
    # edge_tokens[0..] is the unary tunnel — but those have already been
    # emitted via the previous walk-into-cur step). The cap node's PRESENT
    # state is at the end of the unary edge; the "cap-edge sequence" the
    # user wants emitted is cur.edge_tokens. So we already emitted them when
    # we walked into cur. Now: wormhole + directive-follow.
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
      emitted, ending, leftover = follow_directive(target, directive, child_by_token)
      emitted.each { |t| output_tokens << t }
      # `ending` is the last fully-walked valid node. Resume mass-weighted
      # branching from there — no phantom-char fresh-root restart.
      current = ending
    end
  else
    # Branching: pick child by mass.
    nxt = pick_child(cur, child_by_token, rng)
    if nxt.nil?
      # No child for sampled token (depth-32 leaf-like). Fall back: pick fresh
      # depth-1 root.
      nxt2 = pick_depth1(child_by_token, rng)
      break if nxt2.nil?
      current = nxt2
      nxt2.edge_tokens.each { |t| output_tokens << t }
    else
      nxt.edge_tokens.each { |t| output_tokens << t }
      current = nxt
    end
  end
end

# Truncate
output_tokens = output_tokens[0, n_chars] if output_tokens.size > n_chars
puts decode(output_tokens, chars)
