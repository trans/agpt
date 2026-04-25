# Synthesize a wrap-around corpus from a radix-compressed prefix trie.
#
# Walks the trie root → leaf via mass-weighted child picks (same as L4
# path-sampling in the AGPT trainer). When a leaf is reached, samples a
# bridge token from the leaf's endpoint count distribution, then "wraps"
# back to root and continues the walk seeded by that token.
#
# Output is a character-level text file structurally identical to
# data/input.txt — ready to feed into bin/microgpt for SGD-window training
# at any --seq-len. The hypothesis is that a model trained on this
# synthesized corpus at seq_len > D should learn long-context attention
# even though the source trie was only built to depth D.
#
# Usage:
#   bin/synth_wrap_corpus --trie-dir <radix-dir> --vocab-text data/input.txt \
#       --total-tokens 1000000 --seed 42 --output data/synth_wrap_d32.txt

require "option_parser"
require "../agpt"

trie_dir = ""
vocab_text_path = "data/input.txt"
total_tokens = 1_000_000
seed = 42_u64
output_path = "data/synth_wrap.txt"
verbose = false
space_align = false
space_align_topk = 3
space_cut = false

OptionParser.parse do |parser|
  parser.banner = "Usage: synth_wrap_corpus --trie-dir DIR --vocab-text PATH ..."
  parser.on("--trie-dir DIR", "Radix trie directory") { |v| trie_dir = v }
  parser.on("--vocab-text PATH", "Original corpus file (for token-id → char mapping)") { |v| vocab_text_path = v }
  parser.on("--total-tokens N", "Total tokens to generate") { |v| total_tokens = v.to_i }
  parser.on("--seed N", "RNG seed") { |v| seed = v.to_u64 }
  parser.on("--output PATH", "Output corpus file") { |v| output_path = v }
  parser.on("--space-align", "(no-op in this trie at d=32 — leaves are 99.99% mass-1 with single-entry counts; kept for compatibility)") { space_align = true }
  parser.on("--space-align-topk N", "Top-k filter for --space-align (default 3)") { |v| space_align_topk = v.to_i }
  parser.on("--space-cut", "At a leaf, back up within its compressed edge to the LAST space and wrap there (no bridge). Eliminates mid-word glue artifacts at the wrap boundary by ending each walk on a word boundary whenever the leaf's edge contains a space. Falls through to normal bridge if no space is in the edge.") { space_cut = true }
  parser.on("--verbose", "Verbose progress") { verbose = true }
  parser.on("-h", "--help", "Help") { puts parser; exit 0 }
end

if trie_dir.empty?
  STDERR.puts "Error: --trie-dir required"
  exit 1
end

# Build the same char→id mapping that bin/microgpt uses, so synthesized
# text can be tokenized to the SAME ids the trie was built against.
vocab_text = File.read(vocab_text_path)
chars = vocab_text.chars.uniq.sort
id_to_char = {} of Int32 => Char
chars.each_with_index { |c, i| id_to_char[i] = c }

# Reverse lookup for the space token id (used by --space-align).
space_token_id = chars.index(' ')

# Simple xorshift RNG so seed gives reproducible output. Guard against
# the all-zeros state (xorshift fixed point); using `| 1` would have
# conflated adjacent seeds (42 and 43 both → 43).
state = seed == 0 ? 1_u64 : seed

next_u32 = ->{
  s = state
  s ^= s << 13
  s ^= s >> 7
  s ^= s << 17
  state = s
  (s & 0xFFFFFFFF_u64).to_u32
}

next_float = -> { next_u32.call.to_f64 / 4294967296.0 }

# Sample an index in 0..n-1 with given non-negative weights.
weighted_pick = ->(weights : Array(Int32)) {
  total = weights.sum
  if total == 0
    next_u32.call.to_i32 % weights.size
  else
    u = next_float.call * total.to_f64
    acc = 0.0
    pick = weights.size - 1
    weights.each_with_index do |w, idx|
      acc += w.to_f64
      if u <= acc
        pick = idx
        break
      end
    end
    pick
  end
}

reader = MicroGPT::AGPT::RadixTrieReader.new(trie_dir, max_cached: 128)
STDERR.puts "Loaded radix trie: #{reader.radix_count} nodes, max_endpoint_depth=#{reader.depth_file_count - 1}, vocab_size=#{reader.vocab_size}"

if reader.vocab_size != chars.size
  STDERR.puts "WARN: trie vocab_size=#{reader.vocab_size} does not match vocab_text vocab=#{chars.size}. Char mapping may be off."
end

# Pre-build a parent_id → [children] index across ALL depths. A radix node's
# children can live at ANY deeper depth (edge length is variable), so a
# per-depth scan of nodes_at_endpoint_depth(d) misses children whose
# endpoint is deeper than d. Scanning all depths once is cheap (~150 MB
# at d=32 Shakespeare, 1.7M records).
STDERR.puts "Building parent → children index..."
t_idx = Time.instant
children_of = {} of Int32 => Array(MicroGPT::AGPT::RadixTrieReader::LoadedRecord)
(1..reader.depth_file_count - 1).each do |d|
  reader.nodes_at_endpoint_depth(d).each do |rec|
    arr = children_of[rec.parent_id]?
    if arr
      arr << rec
    else
      children_of[rec.parent_id] = [rec]
    end
  end
end
STDERR.puts "  built in #{(Time.instant - t_idx).total_seconds.round(1)}s, #{children_of.size} parent nodes have children"

children_at = ->(parent_id : Int32) {
  children_of[parent_id]? || ([] of MicroGPT::AGPT::RadixTrieReader::LoadedRecord)
}

# Root children are children_of[0].
root_children = children_at.call(0)
STDERR.puts "Root children: #{root_children.size}"

# Map first-edge-token → root_child, for resuming after a wrap.
root_by_first_token = {} of Int32 => Array(MicroGPT::AGPT::RadixTrieReader::LoadedRecord)
root_children.each do |rc|
  arr = root_by_first_token[rc.edge_tokens[0]]?
  if arr
    arr << rc
  else
    root_by_first_token[rc.edge_tokens[0]] = [rc]
  end
end

# The actual walk. Returns a generator-like loop over emitted token IDs.
out_io = File.open(output_path, "w")
emitted = 0
wrap_count = 0
space_aligned_wraps = 0
space_cut_wraps = 0
last_progress = 0

if space_align
  if space_token_id.nil?
    STDERR.puts "WARN: --space-align set but corpus has no space character; flag will be inert"
  else
    STDERR.puts "Space-align: bridge prefers token id=#{space_token_id} (' ') when in leaf's top-#{space_align_topk} counts"
  end
end
if space_cut
  if space_token_id.nil?
    STDERR.puts "WARN: --space-cut set but corpus has no space character; flag will be inert"
  else
    STDERR.puts "Space-cut: at a leaf, emit only up to the LAST space in its edge and wrap (no bridge); falls through to bridge if leaf's edge has no space"
  end
end

# Pick a starting root-child by mass. Returns {root_child, matched_seed?}
# where matched_seed? is true if the picked child's first edge token
# equals the seed (i.e., we matched on the bridge). The caller uses this
# to avoid double-emitting the bridge token, which was already printed
# at the previous iteration's wrap.
pick_root_child = ->(seed_token : Int32?) {
  if seed_token.nil?
    weights = root_children.map { |r| r.edge_mass }
    {root_children[weighted_pick.call(weights)], false}
  else
    candidates = root_by_first_token[seed_token]?
    if candidates.nil? || candidates.empty?
      # Fall back: pick by mass (no seed match) — bridge stays printed.
      weights = root_children.map { |r| r.edge_mass }
      {root_children[weighted_pick.call(weights)], false}
    else
      weights = candidates.map { |r| r.edge_mass }
      {candidates[weighted_pick.call(weights)], true}
    end
  end
}

current_seed : Int32? = nil

while emitted < total_tokens
  current, seed_matched = pick_root_child.call(current_seed)
  current_seed = nil  # consumed

  # Walk down: at each node, decide whether it's a leaf (no children
  # OR depth-cap) BEFORE emitting, so --space-cut can short-circuit at
  # the leaf and stop the emit at the last space in its edge.
  #
  # Without --space-cut: emit the full edge of every node, then bridge
  # at the leaf via the leaf's endpoint counts.
  #
  # If skip_next is true (the picked root child matched the previously
  # emitted bridge token), we skip emitting edge_tokens[0] to avoid
  # duplicating the bridge.
  skip_next = seed_matched
  did_space_cut = false
  loop do
    next_d = current.endpoint_depth + 1
    children = (next_d >= reader.depth_file_count) ? ([] of MicroGPT::AGPT::RadixTrieReader::LoadedRecord) : children_at.call(current.id)
    is_leaf = children.empty?

    if is_leaf && space_cut && (sid = space_token_id)
      edge = current.edge_tokens
      start_idx = skip_next ? 1 : 0
      skip_next = false
      last_space_idx = -1
      i = edge.size - 1
      while i >= start_idx
        if edge[i] == sid
          last_space_idx = i
          break
        end
        i -= 1
      end
      if last_space_idx >= 0
        (start_idx..last_space_idx).each do |j|
          out_io.print id_to_char[edge[j]]
          emitted += 1
          break if emitted >= total_tokens
        end
        space_cut_wraps += 1
        # Seed the next walk with space so pick_root_child finds a
        # space-prefixed root child; that walk's first edge token (==
        # space) will be skipped via seed_matched, avoiding "  " glue.
        current_seed = sid
        did_space_cut = true
        break
      end
      # No space in this leaf's edge — fall through to default emit-all + bridge.
    end

    current.edge_tokens.each do |tok|
      if skip_next
        skip_next = false
      else
        out_io.print id_to_char[tok]
        emitted += 1
        break if emitted >= total_tokens
      end
    end
    break if emitted >= total_tokens
    break if is_leaf

    # Pick child by edge_mass (matches L4 mass-walk semantics).
    weights = children.map { |c| c.edge_mass }
    current = children[weighted_pick.call(weights)]
  end

  break if emitted >= total_tokens

  # Bridge step. Skipped when --space-cut already wrapped at a word boundary.
  unless did_space_cut
    if !current.counts.empty?
      bridge_token = nil.as(Int32?)
      if space_align && (sid = space_token_id)
        # Prefer space if it's among the top-k entries by count.
        top_k = current.counts.to_a.sort_by { |(_t, c)| -c }.first(space_align_topk)
        if top_k.any? { |(t, _c)| t == sid }
          bridge_token = sid
          space_aligned_wraps += 1
        end
      end
      if bridge_token.nil?
        weights = current.counts.map { |c| c[1] }
        pick = weighted_pick.call(weights)
        bridge_token = current.counts[pick][0]
      end
      out_io.print id_to_char[bridge_token]
      emitted += 1
      current_seed = bridge_token
    else
      current_seed = nil
    end
  end
  wrap_count += 1

  if verbose && emitted - last_progress >= 100_000
    STDERR.puts "  emitted #{emitted} tokens, wraps=#{wrap_count}"
    last_progress = emitted
  end
end

out_io.close

STDERR.puts "Done: #{emitted} tokens written to #{output_path}, #{wrap_count} wraps"
STDERR.puts "  Avg path length per wrap: #{(emitted.to_f64 / [wrap_count, 1].max).round(1)} chars"
if space_align
  pct = wrap_count > 0 ? (100.0 * space_aligned_wraps / wrap_count).round(1) : 0.0
  STDERR.puts "  Space-aligned wraps: #{space_aligned_wraps} / #{wrap_count} (#{pct}%)"
end
if space_cut
  pct = wrap_count > 0 ? (100.0 * space_cut_wraps / wrap_count).round(1) : 0.0
  STDERR.puts "  Space-cut wraps: #{space_cut_wraps} / #{wrap_count} (#{pct}%)"
end
