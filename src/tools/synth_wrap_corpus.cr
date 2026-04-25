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

OptionParser.parse do |parser|
  parser.banner = "Usage: synth_wrap_corpus --trie-dir DIR --vocab-text PATH ..."
  parser.on("--trie-dir DIR", "Radix trie directory") { |v| trie_dir = v }
  parser.on("--vocab-text PATH", "Original corpus file (for token-id → char mapping)") { |v| vocab_text_path = v }
  parser.on("--total-tokens N", "Total tokens to generate") { |v| total_tokens = v.to_i }
  parser.on("--seed N", "RNG seed") { |v| seed = v.to_u64 }
  parser.on("--output PATH", "Output corpus file") { |v| output_path = v }
  parser.on("--space-align", "Prefer wrapping at a space-token bridge when one is in the leaf's top-k continuations (eliminates mid-word glue artifacts like 'bishhanged')") { space_align = true }
  parser.on("--space-align-topk N", "How many top-by-count endpoint entries to scan for a space when --space-align is set (default 3)") { |v| space_align_topk = v.to_i }
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
last_progress = 0

if space_align
  if space_token_id.nil?
    STDERR.puts "WARN: --space-align set but corpus has no space character; flag will be inert"
  else
    STDERR.puts "Space-align: bridge prefers token id=#{space_token_id} (' ') when in leaf's top-#{space_align_topk} counts"
  end
end

# Pick a starting root-child by mass.
pick_root_child = ->(seed_token : Int32?) {
  if seed_token.nil?
    weights = root_children.map { |r| r.edge_mass }
    root_children[weighted_pick.call(weights)]
  else
    candidates = root_by_first_token[seed_token]?
    if candidates.nil? || candidates.empty?
      # Fall back: pick by mass (no seed match)
      weights = root_children.map { |r| r.edge_mass }
      root_children[weighted_pick.call(weights)]
    else
      weights = candidates.map { |r| r.edge_mass }
      candidates[weighted_pick.call(weights)]
    end
  end
}

current_seed : Int32? = nil

while emitted < total_tokens
  current = pick_root_child.call(current_seed)
  current_seed = nil  # consumed

  # Walk down: emit current's edge tokens, then descend into a
  # mass-weighted child. Stop at a leaf.
  loop do
    current.edge_tokens.each do |tok|
      out_io.print id_to_char[tok]
      emitted += 1
      break if emitted >= total_tokens
    end
    break if emitted >= total_tokens

    # Find ALL children of current (at any deeper depth — edge length is variable)
    next_d = current.endpoint_depth + 1
    if next_d >= reader.depth_file_count
      # Hit cap. Wrap.
      break
    end
    children = children_at.call(current.id)
    if children.empty?
      # Hit a leaf — wrap with bridge token sampled from current's
      # endpoint counts (real corpus continuation distribution).
      break
    end

    # Pick child by edge_mass (matches L4 mass-walk semantics).
    weights = children.map { |c| c.edge_mass }
    current = children[weighted_pick.call(weights)]
  end

  break if emitted >= total_tokens

  # Sample bridge token from leaf's endpoint counts (the real "what
  # comes next in corpus" distribution at this 32-gram).
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
