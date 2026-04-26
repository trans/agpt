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
out_dir = ""
max_depth = 0
per_subtree = false
prune_min_mass = 1
prune_min_depth = 4

OptionParser.parse do |parser|
  parser.banner = "Usage: bin/agpt_build_radix_corpus --corpus PATH --max-depth N [options]"
  parser.on("--corpus PATH", "Character-level corpus text file") { |v| corpus_path = v }
  parser.on("--out DIR", "Output radix directory (default: /tmp/agpt_<basename>_d<depth>_radix)") { |v| out_dir = v }
  parser.on("--max-depth N", "Trie max depth (required)") { |v| max_depth = v.to_i }
  parser.on("--per-subtree", "Also emit per-subtree files for memory-scoped training") { per_subtree = true }
  parser.on("--prune-min-mass N", "Drop edges with prefix count < N past --prune-min-depth (default 1)") { |v| prune_min_mass = v.to_i }
  parser.on("--prune-min-depth N", "Never prune at depths shallower than this (default 4)") { |v| prune_min_depth = v.to_i }
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
if out_dir.empty?
  basename = File.basename(corpus_path, File.extname(corpus_path))
  out_dir = "/tmp/agpt_#{basename}_d#{max_depth}_radix"
end

text = File.read(corpus_path)
dataset = MicroGPT::CharDataset.new(text)
tokens = dataset.data
corpus_hash = MicroGPT::AGPT::TrieCorpus.token_hash(tokens)

STDERR.puts "[radix-corpus] corpus: #{corpus_path} (#{tokens.size} tokens, vocab=#{dataset.vocab_size})"
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
)
result = builder.build

puts "  radix_count:        #{result[:radix_count]}"
puts "  total_edge_chars:   #{result[:total_edge_chars]}"
puts "  max_endpoint_depth: #{result[:max_endpoint_depth]}"
