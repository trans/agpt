# Build a radix-compressed trie from an existing leveled (per-depth) trie.
# Collapses unary chains into single multi-character edges so training only
# touches branching points.
#
# Replaces the old `bin/microgpt --agpt-build-radix` flow that lived in
# µGPT's main.cr.
#
# Usage:
#   bin/agpt_build_radix --leveled /tmp/agpt_input_d32
#   bin/agpt_build_radix --leveled /tmp/agpt_input_d32 --out /tmp/my_radix \
#       --per-subtree --subtree-level 2 \
#       --prune-min-mass 2 --prune-min-depth 8

require "option_parser"
require "../agpt"

leveled_dir = ""
out_dir = ""
per_subtree = false
subtree_level = 1
prune_min_mass = 1
prune_min_depth = 4
max_cached = 128

OptionParser.parse do |parser|
  parser.banner = "Usage: bin/agpt_build_radix --leveled DIR [options]"
  parser.on("--leveled DIR", "Input leveled trie directory") { |v| leveled_dir = v }
  parser.on("--out DIR", "Output radix directory (default: /tmp/<basename>_radix)") { |v| out_dir = v }
  parser.on("--per-subtree", "Also emit per-subtree files for scalable training") { per_subtree = true }
  parser.on("--subtree-level N", "Subtree key prefix depth (1=unigram, 2=bigram); default 1") { |v| subtree_level = v.to_i }
  parser.on("--prune-min-mass N", "Drop edges with prefix count < N past --prune-min-depth (default 1)") { |v| prune_min_mass = v.to_i }
  parser.on("--prune-min-depth N", "Never prune at depths shallower than this (default 4)") { |v| prune_min_depth = v.to_i }
  parser.on("--max-cached N", "Per-depth LRU cache size for the leveled reader (default 128)") { |v| max_cached = v.to_i }
  parser.on("-h", "--help", "Help") { puts parser; exit 0 }
end

if leveled_dir.empty?
  STDERR.puts "Error: --leveled required"
  exit 1
end
unless subtree_level == 1 || subtree_level == 2
  STDERR.puts "Error: --subtree-level must be 1 (unigram) or 2 (bigram)"
  exit 1
end
if out_dir.empty?
  basename = File.basename(leveled_dir.rstrip('/'))
  out_dir = "/tmp/#{basename}_radix"
end

puts "Building radix trie from #{leveled_dir} → #{out_dir}#{per_subtree ? " (per-subtree, level=#{subtree_level})" : ""}"
if prune_min_mass > 1
  puts "  Pruning: drop paths with mass < #{prune_min_mass} past depth #{prune_min_depth}"
end

reader = MicroGPT::AGPT::LeveledTrieReader.new(leveled_dir, max_cached: max_cached)
builder = MicroGPT::AGPT::StreamingRadixBuilder.new(reader, out_dir,
  per_subtree: per_subtree, subtree_level: subtree_level,
  prune_min_mass: prune_min_mass, prune_min_depth: prune_min_depth)
result_info = builder.build

puts "  radix_count:        #{result_info[:radix_count]}"
puts "  total_edge_chars:   #{result_info[:total_edge_chars]}"
puts "  max_endpoint_depth: #{result_info[:max_endpoint_depth]}"
puts "Original leveled trie had #{reader.node_count} nodes."
compression = reader.node_count.to_f64 / result_info[:radix_count].to_f64
puts "  compression ratio:  #{compression.round(2)}×"
