# Build per-substring position distribution tables from a corpus + both
# (prefix and suffix) radix tries in one invocation. Produces five artifacts:
#
#   substrings.bin                       (canonical substring catalog,
#                                         unified across prefix + suffix)
#   prefix_radix_to_substring.bin        (prefix_trie radix_id → substring_id)
#   suffix_radix_to_substring.bin        (suffix_trie radix_id → substring_id)
#   prefix_position_table.bin            (sparse pos_counts per substring,
#                                         binned by ORIGINAL-START)
#   prefix_phase_targets.bin             (sparse next-token counts by
#                                         prefix radix_id and ORIGINAL-START)
#   suffix_position_table.bin            (sparse pos_counts per substring,
#                                         binned by ORIGINAL-END, recovered
#                                         from the reverse-corpus walk's start)
#
# Suffix trie must be pre-built via `bin/agpt_build_radix_corpus --reverse`.
#
# Usage:
#   bin/agpt_build_position_table \
#     --prefix-trie /tmp/<corpus>_d<D>_radix \
#     --suffix-trie /tmp/<corpus>_d<D>_suffix_radix \
#     --corpus data/<corpus>.txt \
#     --out /tmp/<corpus>_position_data \
#     --window 64

require "../agpt"
require "option_parser"

include MicroGPT::AGPT

prefix_trie_dir = ""
suffix_trie_dir = ""
corpus_path = ""
out_dir = ""
window_size = 64

OptionParser.parse do |p|
  p.banner = "Usage: agpt_build_position_table --prefix-trie DIR --suffix-trie DIR --corpus FILE --out DIR [--window 64]"
  p.on("--prefix-trie DIR", "Prefix radix trie directory") { |v| prefix_trie_dir = v }
  p.on("--suffix-trie DIR", "Suffix radix trie directory (built with build_radix_corpus --reverse)") { |v| suffix_trie_dir = v }
  p.on("--corpus FILE", "Corpus text file (forward order; tool reverses internally for suffix walk)") { |v| corpus_path = v }
  p.on("--out DIR", "Output directory") { |v| out_dir = v }
  p.on("--window N", "Position window size W [default 64]") { |v| window_size = v.to_i }
  p.on("-h", "--help", "") { puts p; exit 0 }
end

raise "--prefix-trie required" if prefix_trie_dir.empty?
raise "--suffix-trie required" if suffix_trie_dir.empty?
raise "--corpus required" if corpus_path.empty?
raise "--out required" if out_dir.empty?
Dir.mkdir_p(out_dir)

# Load corpus + tokenize (forward).
text = File.read(corpus_path)
dataset = MicroGPT::CharDataset.new(text)
corpus_tokens_fwd = dataset.data
STDERR.puts "Corpus: #{corpus_path} (#{corpus_tokens_fwd.size} tokens, vocab=#{dataset.vocab_size})"

def append_wrap_lookahead(tokens : Array(Int32), max_depth : Int32) : Array(Int32)
  return tokens if max_depth <= 0 || tokens.empty?
  wrap_len = Math.min(max_depth, tokens.size)
  tokens + tokens[0, wrap_len]
end

# Helper: reconstruct the substring (token sequence, root-to-node order)
# for a given radix node by walking parent pointers via the walker's
# compact slice storage. No hashmap allocations.
def reconstruct_tokens(walker : MicroGPT::AGPT::CorpusTrieWalker, id : Int32) : Array(Int32)?
  # Walk parents from `id` up to the root (parent_id == 0), collecting
  # the edge_tokens slice for each. The root itself has parent_id == 0
  # (its own parent is itself by convention; edge_len is 0). Stop when
  # we either hit cur == 0 (root) or cur becomes negative.
  segments = [] of Slice(Int32)
  cur = id
  while cur > 0 && cur < walker.radix_count
    edge = walker.edge_tokens_of(cur)
    segments << edge unless edge.size == 0
    parent = walker.parent_id_of(cur)
    break if parent == cur  # safety against self-loop
    cur = parent
    break if cur == 0
  end
  segments.reverse!
  tokens = [] of Int32
  segments.each { |seg| seg.each { |t| tokens << t } }
  return nil if tokens.empty?
  tokens
end

# ===== Pass A: load both tries, enumerate substrings, build unified catalog =====

catalog = SubstringCatalog.new

STDERR.puts ""
STDERR.puts "Pass A1: load prefix trie + enumerate substrings"
# Small reader cache: walker copies what it needs into compact slices, so
# we don't want the reader to retain every depth file in memory after init.
prefix_reader = RadixTrieReader.new(prefix_trie_dir, max_cached: 2)
prefix_depth = prefix_reader.depth_file_count - 1
prefix_walk_tokens = append_wrap_lookahead(corpus_tokens_fwd, prefix_depth)
prefix_walker = CorpusTrieWalker.new(prefix_reader, prefix_walk_tokens)
prefix_n_radix = prefix_walker.radix_count
prefix_radix_to_substring = Slice(Int32).new(prefix_n_radix, -1)
STDERR.puts "  walker: #{prefix_n_radix} nodes, #{(prefix_walker.approximate_bytes / 1024.0 / 1024.0).round(1)} MB compact storage"

t0 = Time.monotonic
prefix_walker.each_id do |id|
  next if id == 0  # skip the (virtual) root
  tokens = reconstruct_tokens(prefix_walker, id)
  next if tokens.nil?
  sid = catalog.get_or_assign(tokens)
  prefix_radix_to_substring[id] = sid
end
STDERR.puts "  prefix nodes: #{prefix_n_radix}, catalog after: #{catalog.size} (#{(Time.monotonic - t0).total_seconds.round(2)}s)"

# Suffix trie + corpus reversal.
corpus_tokens_rev = corpus_tokens_fwd.reverse

STDERR.puts ""
STDERR.puts "Pass A2: load suffix trie + enumerate substrings (reverse-order tokens, flipped for catalog)"
suffix_reader = RadixTrieReader.new(suffix_trie_dir, max_cached: 2)
suffix_depth = suffix_reader.depth_file_count - 1
suffix_walk_tokens = append_wrap_lookahead(corpus_tokens_rev, suffix_depth)
suffix_walker = CorpusTrieWalker.new(suffix_reader, suffix_walk_tokens)
suffix_n_radix = suffix_walker.radix_count
suffix_radix_to_substring = Slice(Int32).new(suffix_n_radix, -1)
STDERR.puts "  walker: #{suffix_n_radix} nodes, #{(suffix_walker.approximate_bytes / 1024.0 / 1024.0).round(1)} MB compact storage"

t0 = Time.monotonic
n_catalog_before_suffix = catalog.size
suffix_walker.each_id do |id|
  next if id == 0
  tokens = reconstruct_tokens(suffix_walker, id)
  next if tokens.nil?
  # Suffix-trie tokens are in REVERSED-CORPUS order; reverse to get
  # FORWARD (canonical) substring form for the catalog.
  sid = catalog.get_or_assign(tokens.reverse)
  suffix_radix_to_substring[id] = sid
end
n_new_from_suffix = catalog.size - n_catalog_before_suffix
STDERR.puts "  suffix nodes: #{suffix_n_radix}, catalog after: #{catalog.size} (+#{n_new_from_suffix} new) (#{(Time.monotonic - t0).total_seconds.round(2)}s)"

# ===== Pass B: walk forward corpus → prefix_position_table =====

STDERR.puts ""
STDERR.puts "Pass B1: walk forward corpus → prefix position counts (W=#{window_size})"
prefix_builder = PositionTable::Builder.new(window_size, PositionTable::Regime::Sliding, catalog.size)
contributions = 0_i64
t0 = Time.monotonic
prefix_walker.walk(corpus_tokens_fwd.size) do |radix_id, start_pos, _terminal_pos|
  sid = prefix_radix_to_substring[radix_id]
  next if sid < 0
  prefix_builder.increment(sid, start_pos)
  contributions += 1
end
STDERR.puts "  #{contributions} contributions (#{(Time.monotonic - t0).total_seconds.round(2)}s)"

STDERR.puts "  compacting..."
t0 = Time.monotonic
prefix_position_table = prefix_builder.build
STDERR.puts "  total_bins=#{prefix_position_table.total_bins} (#{(Time.monotonic - t0).total_seconds.round(2)}s)"

# ===== Pass B2: walk forward corpus → direct prefix phase target table =====

STDERR.puts ""
STDERR.puts "Pass B2: walk forward corpus → prefix phase target counts (W=#{window_size})"
prefix_phase_target_counts = Array(Hash(UInt64, UInt32)).new(prefix_n_radix) do
  Hash(UInt64, UInt32).new
end
target_contributions = 0_i64
t0 = Time.monotonic
prefix_walker.walk(corpus_tokens_fwd.size) do |radix_id, start_pos, terminal_pos|
  next_token_pos = terminal_pos + 1
  next if next_token_pos >= prefix_walk_tokens.size
  phase = start_pos % window_size
  token = prefix_walk_tokens[next_token_pos]
  key = (phase.to_u64 << 32) | token.to_u32.to_u64
  h = prefix_phase_target_counts[radix_id]
  h[key] = (h[key]? || 0_u32) + 1_u32
  target_contributions += 1
end
STDERR.puts "  #{target_contributions} target contributions (#{(Time.monotonic - t0).total_seconds.round(2)}s)"

def write_phase_target_table(io : IO, counts : Array(Hash(UInt64, UInt32)), window_size : Int32)
  io.write("APTG".to_slice)
  io.write_bytes(window_size.to_u16, IO::ByteFormat::LittleEndian)
  io.write_bytes(0_u16, IO::ByteFormat::LittleEndian)
  io.write_bytes(counts.size.to_u32, IO::ByteFormat::LittleEndian)
  total_entries = counts.sum { |h| h.size.to_u64 }
  io.write_bytes(total_entries, IO::ByteFormat::LittleEndian)

  offset = 0_i32
  counts.each do |h|
    io.write_bytes(offset, IO::ByteFormat::LittleEndian)
    offset += h.size.to_i32
  end
  io.write_bytes(offset, IO::ByteFormat::LittleEndian)

  counts.each do |h|
    h.to_a.sort_by { |(key, _count)| key }.each do |key, count|
      phase = (key >> 32).to_u16
      token = (key & 0xffff_ffff_u64).to_u16
      io.write_bytes(phase, IO::ByteFormat::LittleEndian)
      io.write_bytes(token, IO::ByteFormat::LittleEndian)
      io.write_bytes(count, IO::ByteFormat::LittleEndian)
    end
  end
end

# ===== Pass C: walk reversed corpus → suffix_position_table =====

STDERR.puts ""
STDERR.puts "Pass C1: walk reversed corpus → suffix position counts (W=#{window_size})"
suffix_builder = PositionTable::Builder.new(window_size, PositionTable::Regime::Sliding, catalog.size)
contributions = 0_i64
n_corpus = corpus_tokens_fwd.size
t0 = Time.monotonic
suffix_walker.walk(corpus_tokens_fwd.size) do |radix_id, start_pos_rev, _terminal_pos|
  sid = suffix_radix_to_substring[radix_id]
  next if sid < 0
  # The walker's start_pos_rev is the start in the REVERSED corpus.
  # Convert to "original END position" of this substring's occurrence:
  #   original_end = (n_corpus - 1) - start_pos_rev
  original_end = n_corpus - 1 - start_pos_rev
  suffix_builder.increment(sid, original_end)
  contributions += 1
end
STDERR.puts "  #{contributions} contributions (#{(Time.monotonic - t0).total_seconds.round(2)}s)"

STDERR.puts "  compacting..."
t0 = Time.monotonic
suffix_position_table = suffix_builder.build
STDERR.puts "  total_bins=#{suffix_position_table.total_bins} (#{(Time.monotonic - t0).total_seconds.round(2)}s)"

# ===== Write all outputs =====

prefix_rts = RadixToSubstring.new(RadixToSubstring::Side::Prefix, prefix_radix_to_substring)
suffix_rts = RadixToSubstring.new(RadixToSubstring::Side::Suffix, suffix_radix_to_substring)

STDERR.puts ""
STDERR.puts "Writing outputs to #{out_dir}/"
out_files = {
  "substrings.bin"                  => ->(io : IO) { catalog.write_to(io) },
  "prefix_radix_to_substring.bin"   => ->(io : IO) { prefix_rts.write_to(io) },
  "suffix_radix_to_substring.bin"   => ->(io : IO) { suffix_rts.write_to(io) },
  "prefix_position_table.bin"       => ->(io : IO) { prefix_position_table.write_to(io) },
  "prefix_phase_targets.bin"        => ->(io : IO) { write_phase_target_table(io, prefix_phase_target_counts, window_size) },
  "suffix_position_table.bin"       => ->(io : IO) { suffix_position_table.write_to(io) },
}
out_files.each do |name, writer|
  path = File.join(out_dir, name)
  File.open(path, "wb") { |f| writer.call(f) }
  STDERR.puts "  #{name} (#{File.size(path)} bytes)"
end

STDERR.puts ""
STDERR.puts "Done. catalog_size=#{catalog.size} prefix_radix=#{prefix_n_radix} suffix_radix=#{suffix_n_radix}"
