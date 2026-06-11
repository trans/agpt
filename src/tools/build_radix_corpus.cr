# Build a radix-compressed trie directly from a corpus, processing one
# root-character subtree at a time. Skips the leveled-trie intermediate, so
# peak memory is bounded by ONE subtree's working set instead of the full
# corpus.
#
# Output is the standard radix format (radix_depth_NNN.bin + meta.bin),
# read unchanged by bin/synth_wrap_corpus, bin/agpt_train, etc.
#
# Usage:
#   bin/agpt_build_radix_corpus --corpus data/gutenberg_5m.txt --max-depth 32
#   bin/agpt_build_radix_corpus --corpus data/input.txt --max-depth 16 \
#       --out /tmp/my_radix --per-subtree

require "option_parser"
require "../agpt"

corpus_path = ""
vocab_path = ""
out_dir = ""
max_depth = 0
per_subtree = false
prune_min_mass = 1
prune_min_depth = 4
reverse = false
stride = 1
phase = 0
target_offset = 0

OptionParser.parse do |parser|
  parser.banner = "Usage: bin/agpt_build_radix_corpus --corpus PATH --max-depth N [options]"
  parser.on("--corpus PATH", "Character-level corpus text file") { |v| corpus_path = v }
  parser.on("--vocab-file PATH", "Build vocab from this file (defaults to --corpus). Use the full-corpus file when --corpus is a truncated subset to ensure consistent token IDs across builds.") { |v| vocab_path = v }
  parser.on("--out DIR", "Output radix directory (default: /tmp/agpt_<basename>_d<depth>[_suffix]_radix)") { |v| out_dir = v }
  parser.on("--max-depth N", "Trie max depth (required)") { |v| max_depth = v.to_i }
  parser.on("--per-subtree", "Also emit per-subtree files for memory-scoped training") { per_subtree = true }
  parser.on("--prune-min-mass N", "Drop edges with prefix count < N past --prune-min-depth (default 1)") { |v| prune_min_mass = v.to_i }
  parser.on("--prune-min-depth N", "Never prune at depths shallower than this (default 4)") { |v| prune_min_depth = v.to_i }
  parser.on("--reverse", "Reverse corpus before building → suffix radix tree (for p2s-attention)") { reverse = true }
  parser.on("--stride N", "Build stride-N prefix paths, reading corpus positions i, i+N, i+2N... (default 1)") { |v| stride = v.to_i }
  parser.on("--phase N", "For --stride N, only use starts where corpus index mod N == phase (default 0)") { |v| phase = v.to_i }
  parser.on("--target-offset N", "Predict token N original positions after each prefix endpoint instead of the next path child (default 0 = standard child target)") { |v| target_offset = v.to_i }
  parser.on("-h", "--help", "Help") { puts parser; exit 0 }
end

if corpus_path.empty?
  STDERR.puts "Error: --corpus required"
  exit 1
end
if max_depth <= 0
  STDERR.puts "Error: --max-depth must be > 0"
  exit 1
end
if stride <= 0
  STDERR.puts "Error: --stride must be > 0"
  exit 1
end
if phase < 0 || phase >= stride
  STDERR.puts "Error: --phase must satisfy 0 <= phase < stride"
  exit 1
end
if target_offset < 0
  STDERR.puts "Error: --target-offset must be >= 0"
  exit 1
end
if out_dir.empty?
  basename = File.basename(corpus_path, File.extname(corpus_path))
  suffix_tag = reverse ? "_suffix" : ""
  stride_tag = stride == 1 ? "" : "_s#{stride}_p#{phase}"
  target_tag = target_offset == 0 ? "" : "_to#{target_offset}"
  out_dir = "/tmp/agpt_#{basename}_d#{max_depth}#{suffix_tag}#{stride_tag}#{target_tag}_radix"
end

text = File.read(corpus_path)
# Build vocab from vocab_path if specified (allows truncated corpora to share a
# vocab with the full corpus), otherwise from the corpus text itself.
vocab_text = vocab_path.empty? ? text : File.read(vocab_path)
dataset = MicroGPT::CharDataset.new(vocab_text)
# Encode the (possibly truncated) corpus text using the shared vocab. Any chars
# in the corpus but absent from the vocab raise (shouldn't happen if vocab_path
# is a superset of corpus_path).
tokens = text.chars.map do |c|
  id = dataset.char_to_id[c]?
  raise "Corpus char #{c.inspect} not in vocab from #{vocab_path.empty? ? corpus_path : vocab_path}" if id.nil?
  id
end
# Reverse the token sequence to build a suffix radix tree. The same
# CorpusRadixBuilder produces a prefix-radix-shaped output over reversed
# input; semantically that's the suffix tree (paths read right-to-left
# in the original corpus). Vocab and tokenizer are unchanged.
tokens = tokens.reverse if reverse
corpus_hash = MicroGPT::AGPT::TrieCorpus.token_hash(tokens)

stride_note = stride == 1 ? "" : " [stride=#{stride}, phase=#{phase}]"
target_note = target_offset == 0 ? "" : " [target_offset=#{target_offset}]"
STDERR.puts "[radix-corpus] corpus: #{corpus_path} (#{tokens.size} tokens, vocab=#{dataset.vocab_size})#{reverse ? " [REVERSED → suffix tree]" : ""}#{stride_note}#{target_note}"
STDERR.puts "[radix-corpus] max_depth=#{max_depth}, output=#{out_dir}"
if prune_min_mass > 1
  STDERR.puts "[radix-corpus] pruning: drop paths with mass < #{prune_min_mass} past depth #{prune_min_depth}"
end

builder = MicroGPT::AGPT::CorpusRadixBuilder.new(
  corpus_tokens: tokens,
  vocab_size: dataset.vocab_size,
  max_depth: max_depth,
  out_dir: out_dir,
  corpus_hash: corpus_hash,
  tokenizer_tag: MicroGPT::AGPT::TOKENIZER_TAG,
  per_subtree: per_subtree,
  prune_min_mass: prune_min_mass,
  prune_min_depth: prune_min_depth,
  stride: stride,
  phase: phase,
  target_offset: target_offset,
)
result = builder.build

puts "  radix_count:        #{result[:radix_count]}"
puts "  total_edge_chars:   #{result[:total_edge_chars]}"
puts "  max_endpoint_depth: #{result[:max_endpoint_depth]}"
