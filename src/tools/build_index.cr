# Build a leveled (per-depth) AGPT trie index from a character-level
# corpus. Produces depth_NNN.bin + meta.bin in the output directory,
# ready to be radix-compressed (bin/agpt_build_radix) or loaded by a
# Crystal-side trainer.
#
# Replaces the old `bin/microgpt --agpt-build-index` / `--agpt-save-index-dir`
# flows that lived in µGPT's main.cr.
#
# Usage:
#   bin/agpt_build_index --corpus data/input.txt --max-depth 32
#   bin/agpt_build_index --corpus data/input.txt --max-depth 64 --out /tmp/my_trie

require "option_parser"
require "../agpt"

corpus_path = ""
out_dir = ""
max_depth = 0

OptionParser.parse do |parser|
  parser.banner = "Usage: bin/agpt_build_index --corpus PATH --max-depth N [--out DIR]"
  parser.on("--corpus PATH", "Character-level corpus text file") { |v| corpus_path = v }
  parser.on("--out DIR", "Output directory (default: /tmp/agpt_<basename>_d<depth>)") { |v| out_dir = v }
  parser.on("--max-depth N", "Trie max depth (required)") { |v| max_depth = v.to_i }
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
  out_dir = "/tmp/agpt_#{basename}_d#{max_depth}"
end

text = File.read(corpus_path)
dataset = MicroGPT::CharDataset.new(text)
train_tokens = dataset.data
corpus_hash = MicroGPT::AGPT::TrieCorpus.token_hash(train_tokens)

STDERR.puts "[agpt index] building leveled trie → #{out_dir}"
STDERR.puts "  corpus: #{corpus_path} (#{train_tokens.size} tokens, vocab=#{dataset.vocab_size})"
STDERR.puts "  max_depth: #{max_depth}"

started_at = Time.instant
builder = MicroGPT::AGPT::StreamingLeveledBuilder.new(
  train_tokens,
  out_dir,
  max_depth: max_depth,
  vocab_size: dataset.vocab_size,
  corpus_hash: corpus_hash,
  tokenizer_tag: MicroGPT::AGPT::TOKENIZER_TAG,
)
builder.build
elapsed = Time.instant - started_at

STDERR.puts "[agpt index] saved in #{elapsed.total_seconds.round(1)}s"
