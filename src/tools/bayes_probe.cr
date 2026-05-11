require "../agpt/radix_trie_reader"
require "option_parser"

# Probe Bayesian consistency between forward and suffix radix-tries.
#
# For a given prefix p, computes:
#   P_p(t | p) = mass_forward(p++[t]) / mass_forward(p)        — direct lookup
#   P_s(t | p) = mass_suffix([t]++reverse(p)) / mass_suffix(reverse(p))  — joint via suffix
#
# Both should equal the same empirical distribution from the corpus.
# Reports magnitudes, KL divergence, and any disagreements.

forward_dir = ""
suffix_dir = ""
corpus_path = ""
prefix_str = ""
verbose = false

OptionParser.parse do |p|
  p.on("--forward DIR", "Forward radix trie directory") { |v| forward_dir = v }
  p.on("--suffix DIR", "Suffix radix trie directory") { |v| suffix_dir = v }
  p.on("--corpus PATH", "Corpus file (for vocab)") { |v| corpus_path = v }
  p.on("--prefix STR", "Prefix string to probe") { |v| prefix_str = v }
  p.on("--verbose", "Verbose output") { verbose = true }
end

abort "missing --forward" if forward_dir.empty?
abort "missing --suffix" if suffix_dir.empty?
abort "missing --corpus" if corpus_path.empty?
abort "missing --prefix" if prefix_str.empty?

# Build vocab from corpus
text = File.read(corpus_path)
chars = text.chars.to_set.to_a.sort
char_to_id = {} of Char => Int32
chars.each_with_index { |c, i| char_to_id[c] = i }
vocab_size = chars.size
puts "Vocab size: #{vocab_size}"

def encode(s : String, char_to_id : Hash(Char, Int32)) : Array(Int32)
  s.chars.map { |c| char_to_id[c] }
end

# Trie index: parent_id → list of children records, for fast walk
class TrieIndex
  getter reader : MicroGPT::AGPT::RadixTrieReader
  getter children_by_parent : Hash(Int32, Array(MicroGPT::AGPT::RadixTrieReader::LoadedRecord))

  def initialize(@reader)
    @children_by_parent = Hash(Int32, Array(MicroGPT::AGPT::RadixTrieReader::LoadedRecord)).new do |h, k|
      h[k] = [] of MicroGPT::AGPT::RadixTrieReader::LoadedRecord
    end
    @reader.each do |r|
      @children_by_parent[r.parent_id] << r
    end
  end

  # Walk the path of tokens through the radix trie. Returns the radix node
  # we land on (which may be mid-edge or at endpoint). Returns nil if the
  # path doesn't exist in the trie.
  def walk(tokens : Array(Int32)) : MicroGPT::AGPT::RadixTrieReader::LoadedRecord?
    return nil if tokens.empty?
    parent_id = 0
    pos = 0
    last_node = nil
    while pos < tokens.size
      kids = @children_by_parent[parent_id]?
      return nil if kids.nil? || kids.empty?
      kid = kids.find { |k| k.edge_tokens[0] == tokens[pos] }
      return nil unless kid
      edge_len = kid.edge_tokens.size
      max_consume = Math.min(edge_len, tokens.size - pos)
      max_consume.times do |i|
        return nil if kid.edge_tokens[i] != tokens[pos + i]
      end
      pos += max_consume
      last_node = kid
      parent_id = kid.id
      # If we stopped mid-edge, that's still a valid landing point
      break if max_consume < edge_len
    end
    last_node
  end

  # Given a node, get its endpoint mass (count of corpus positions
  # passing through this node).
  def mass_at(node : MicroGPT::AGPT::RadixTrieReader::LoadedRecord) : Int64
    node.edge_mass.to_i64
  end
end

puts ""
puts "Loading forward trie: #{forward_dir}"
forward_reader = MicroGPT::AGPT::RadixTrieReader.new(forward_dir, max_cached: 64)
forward = TrieIndex.new(forward_reader)
puts "  #{forward_reader.radix_count} nodes"

puts "Loading suffix trie: #{suffix_dir}"
suffix_reader = MicroGPT::AGPT::RadixTrieReader.new(suffix_dir, max_cached: 64)
suffix = TrieIndex.new(suffix_reader)
puts "  #{suffix_reader.radix_count} nodes"

# Encode prefix
prefix_tokens = encode(prefix_str, char_to_id)
puts ""
puts "Prefix: #{prefix_str.inspect}  (#{prefix_tokens.size} tokens)"
puts "  encoded: #{prefix_tokens}"

# ---- Forward tree: P_p(t | p) ----
forward_node = forward.walk(prefix_tokens)
if forward_node.nil?
  abort "Prefix not found in forward tree"
end
forward_prefix_mass = forward.mass_at(forward_node)
puts ""
puts "Forward tree:"
puts "  prefix node id=#{forward_node.id}, edge_len=#{forward_node.edge_tokens.size}, mass=#{forward_prefix_mass}"

forward_dist = Hash(Int32, Float64).new(0.0)
vocab_size.times do |t|
  joint_node = forward.walk(prefix_tokens + [t])
  if joint_node.nil?
    forward_dist[t] = 0.0
  else
    forward_dist[t] = forward.mass_at(joint_node).to_f / forward_prefix_mass.to_f
  end
end

# ---- Suffix tree: P_s(t | p) via joint-mass inversion ----
prefix_reversed = prefix_tokens.reverse
suffix_prefix_node = suffix.walk(prefix_reversed)
if suffix_prefix_node.nil?
  puts ""
  puts "WARNING: prefix_reversed not found in suffix tree"
  suffix_prefix_mass = 0_i64
else
  suffix_prefix_mass = suffix.mass_at(suffix_prefix_node)
end
puts ""
puts "Suffix tree:"
puts "  reverse(prefix) mass=#{suffix_prefix_mass}"

suffix_dist = Hash(Int32, Float64).new(0.0)
if suffix_prefix_mass > 0
  vocab_size.times do |t|
    joint_node = suffix.walk([t] + prefix_reversed)
    if joint_node.nil?
      suffix_dist[t] = 0.0
    else
      suffix_dist[t] = suffix.mass_at(joint_node).to_f / suffix_prefix_mass.to_f
    end
  end
end

# ---- Compare ----
puts ""
puts "DISTRIBUTIONS"
puts sprintf("  %-6s  %-12s  %-12s  %-10s  %s", "token", "P_forward", "P_suffix", "diff", "char")
total_kl_pf = 0.0  # KL(P_forward || P_suffix)
total_kl_ps = 0.0  # KL(P_suffix || P_forward)
shown = 0
vocab_size.times do |t|
  pf = forward_dist[t]
  ps = suffix_dist[t]
  next if pf < 1e-9 && ps < 1e-9
  shown += 1
  c = chars[t].inspect[1..-2]
  diff = ps - pf
  highlight = diff.abs > 1e-6 ? " *" : ""
  printf("  %-6d  %-12.6f  %-12.6f  %+10.6f  %s%s\n", t, pf, ps, diff, c, highlight)
  if pf > 1e-12 && ps > 1e-12
    total_kl_pf += pf * Math.log(pf / ps)
    total_kl_ps += ps * Math.log(ps / pf)
  end
end
puts "  (#{vocab_size - shown} tokens with both probs zero, omitted)"

puts ""
puts "DIVERGENCE"
printf("  KL(P_forward || P_suffix) = %.8f nats\n", total_kl_pf)
printf("  KL(P_suffix || P_forward) = %.8f nats\n", total_kl_ps)
printf("  Symmetric (avg)            = %.8f nats\n", (total_kl_pf + total_kl_ps) / 2)

# Mass equality check
fwd_mass_total = vocab_size.times.sum do |t|
  joint = forward.walk(prefix_tokens + [t])
  joint.nil? ? 0_i64 : forward.mass_at(joint)
end
puts ""
puts "MASS CHECK"
puts "  forward Σ mass(p++[t]) over all t  = #{fwd_mass_total}"
puts "  forward mass(p)                      = #{forward_prefix_mass}"
puts "  ratio                                = #{fwd_mass_total.to_f / forward_prefix_mass.to_f}"
puts "  (should be ≤ 1.0; <1 means some forward continuations not measured)"
