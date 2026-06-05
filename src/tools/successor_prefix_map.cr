# Inspect observed successor links between depth-cap prefix occurrences.
#
# This is diagnostic only. It preserves radix node identity and records two
# anchor variants:
#   end  -- cap at corpus[pos..pos+d-1] links to cap at corpus[pos+d..pos+2d-1]
#   head -- same cap links to cap starting at the first token of its compressed
#           edge, corpus[pos+first_char_depth-1..]

require "../agpt"
require "json"
require "option_parser"

include MicroGPT::AGPT

trie_dir = ""
corpus_path = "data/.splits/4fa9aec1db6b3aea/train_corpus.txt"
out_dir = "rnd/successor-prefix-attention/successor-prefix-map"
sample_limit = 24
top_edges_limit = 64
wrap = true
mass_one_only = false

OptionParser.parse do |p|
  p.banner = "Usage: successor_prefix_map --trie DIR [--corpus FILE] [--out DIR]"
  p.on("--trie DIR", "Prefix radix trie directory") { |v| trie_dir = v }
  p.on("--corpus FILE", "Corpus text file [default train split]") { |v| corpus_path = v }
  p.on("--out DIR", "Output directory") { |v| out_dir = v }
  p.on("--sample-limit N", "Number of example rows [default 24]") { |v| sample_limit = v.to_i }
  p.on("--top-edges N", "Number of top edge rows [default 64]") { |v| top_edges_limit = v.to_i }
  p.on("--mass-one-only", "Only use cap nodes with edge_mass == 1") { mass_one_only = true }
  p.on("--no-wrap", "Do not wrap successor starts around corpus end") { wrap = false }
  p.on("-h", "--help", "") { puts p; exit 0 }
end

raise "--trie required" if trie_dir.empty?
raise "missing trie dir: #{trie_dir}" unless Dir.exists?(trie_dir)
raise "missing corpus: #{corpus_path}" unless File.exists?(corpus_path)
Dir.mkdir_p(out_dir)

text = File.read(corpus_path)
dataset = MicroGPT::CharDataset.new(text)
tokens = dataset.data
chars = text.chars.uniq.sort
n_tokens = tokens.size

reader = RadixTrieReader.new(trie_dir, max_cached: 2)
d_max = reader.depth_file_count - 1

def append_wrap_lookahead(tokens : Array(Int32), lookahead : Int32) : Array(Int32)
  return tokens if lookahead <= 0 || tokens.empty?
  tokens + tokens[0, Math.min(lookahead, tokens.size)]
end

walk_tokens = wrap ? append_wrap_lookahead(tokens, d_max * 2 + 2) : tokens
walker = CorpusTrieWalker.new(reader, walk_tokens)
d_max = walker.d_max

STDERR.puts "Corpus: #{corpus_path} (#{n_tokens} tokens, vocab=#{dataset.vocab_size})"
STDERR.puts "Trie: #{trie_dir} (#{walker.radix_count} radix nodes, d_max=#{d_max})"
STDERR.puts "Mode: wrap=#{wrap}, mass_one_only=#{mass_one_only}"

def token_label(token : Int32, chars : Array(Char)) : String
  ch = chars[token]?
  return "?" if ch.nil?
  case ch
  when '\n' then "\\n"
  when '\t' then "\\t"
  when ' '  then "space"
  else ch.to_s
  end
end

def decode_tokens(toks : Array(Int32), chars : Array(Char)) : String
  String.build do |io|
    toks.each do |tok|
      ch = chars[tok]?
      io << (ch || '?')
    end
  end
end

def reconstruct_tokens(walker : CorpusTrieWalker, id : Int32) : Array(Int32)
  segments = [] of Slice(Int32)
  cur = id
  while cur > 0 && cur < walker.radix_count
    edge = walker.edge_tokens_of(cur)
    segments << edge unless edge.empty?
    parent = walker.parent_id_of(cur)
    break if parent == cur
    cur = parent
  end
  tokens = [] of Int32
  segments.reverse_each { |seg| seg.each { |t| tokens << t } }
  tokens
end

def pct(num : Int64, den : Int64) : Float64
  return 0.0 if den == 0
  100.0 * num.to_f / den.to_f
end

def add_edge(edges : Hash(Int32, Hash(Int32, Int64)), from : Int32, to : Int32)
  h = edges[from]?
  unless h
    h = Hash(Int32, Int64).new(0_i64)
    edges[from] = h
  end
  h[to] += 1_i64
end

def edge_total(edges : Hash(Int32, Hash(Int32, Int64))) : Int64
  total = 0_i64
  edges.each_value do |h|
    h.each_value { |count| total += count }
  end
  total
end

def concentration_stats(edges : Hash(Int32, Hash(Int32, Int64)))
  nodes = edges.size.to_i64
  total_edges = 0_i64
  total_occ = 0_i64
  fanouts = [] of Int32
  top1_sum = 0_i64
  top4_sum = 0_i64
  top8_sum = 0_i64
  single_successor_nodes = 0_i64

  edges.each_value do |h|
    fanouts << h.size
    total_edges += h.size
    vals = h.values.sort.reverse
    occ = vals.sum(0_i64)
    total_occ += occ
    top1_sum += vals[0]? || 0_i64
    top4_sum += vals.first(4).sum(0_i64)
    top8_sum += vals.first(8).sum(0_i64)
    single_successor_nodes += 1_i64 if h.size == 1
  end

  fanouts.sort!
  p50 = fanouts.empty? ? 0 : fanouts[fanouts.size // 2]
  p90 = fanouts.empty? ? 0 : fanouts[(fanouts.size * 90 // 100).clamp(0, fanouts.size - 1)]
  p99 = fanouts.empty? ? 0 : fanouts[(fanouts.size * 99 // 100).clamp(0, fanouts.size - 1)]
  max = fanouts.empty? ? 0 : fanouts[-1]

  {
    "source_nodes" => nodes,
    "distinct_edges" => total_edges,
    "occurrences" => total_occ,
    "single_successor_nodes" => single_successor_nodes,
    "single_successor_node_pct" => pct(single_successor_nodes, nodes),
    "fanout_p50" => p50,
    "fanout_p90" => p90,
    "fanout_p99" => p99,
    "fanout_max" => max,
    "top1_occurrence_coverage_pct" => pct(top1_sum, total_occ),
    "top4_occurrence_coverage_pct" => pct(top4_sum, total_occ),
    "top8_occurrence_coverage_pct" => pct(top8_sum, total_occ),
  }
end

cap_at_start = Slice(Int32).new(n_tokens, -1)
cap_occurrences = [] of NamedTuple(id: Int32, start: Int32, terminal: Int32, first_depth: Int32, edge_len: Int32, edge_mass: Int32)

STDERR.puts "Pass 1: walk corpus and collect cap occurrences"
walker.walk(n_tokens) do |radix_id, start_pos, terminal_pos|
  next unless walker.endpoint_depth_of(radix_id) == d_max
  next unless terminal_pos - start_pos + 1 == d_max
  edge_mass = walker.edge_mass_of(radix_id)
  next if mass_one_only && edge_mass != 1

  edge_len = walker.edge_len_of(radix_id)
  first_depth = d_max - edge_len + 1
  cap_at_start[start_pos] = radix_id
  cap_occurrences << {
    id: radix_id,
    start: start_pos,
    terminal: terminal_pos,
    first_depth: first_depth,
    edge_len: edge_len,
    edge_mass: edge_mass,
  }
end

STDERR.puts "  cap occurrences: #{cap_occurrences.size}"

end_edges = Hash(Int32, Hash(Int32, Int64)).new
head_edges = Hash(Int32, Hash(Int32, Int64)).new
end_skipped = 0_i64
head_skipped = 0_i64

STDERR.puts "Pass 2: build end/head successor maps"
cap_occurrences.each do |occ|
  end_start = occ[:terminal] + 1
  head_start = occ[:start] + occ[:first_depth] - 1

  if wrap || end_start < n_tokens
    b = cap_at_start[end_start % n_tokens]
    if b >= 0
      add_edge(end_edges, occ[:id], b)
    else
      end_skipped += 1
    end
  else
    end_skipped += 1
  end

  if wrap || head_start < n_tokens
    b = cap_at_start[head_start % n_tokens]
    if b >= 0
      add_edge(head_edges, occ[:id], b)
    else
      head_skipped += 1
    end
  else
    head_skipped += 1
  end
end

cap_node_ids = cap_occurrences.map(&.[:id]).uniq
mass_one_caps = cap_node_ids.count { |id| walker.edge_mass_of(id) == 1 }
compressed_caps = cap_node_ids.count { |id| walker.edge_len_of(id) > 1 }

def top_edge_rows(edges : Hash(Int32, Hash(Int32, Int64)),
                  walker : CorpusTrieWalker,
                  chars : Array(Char),
                  limit : Int32)
  rows = [] of NamedTuple(from_id: Int32, to_id: Int32, count: Int64, from_text: String, to_text: String)
  flat = [] of Tuple(Int32, Int32, Int64)
  edges.each do |from, h|
    h.each { |to, count| flat << {from, to, count} }
  end
  flat.sort_by! { |(_, _, count)| -count }
  flat.first(limit).each do |from, to, count|
    rows << {
      from_id: from,
      to_id: to,
      count: count,
      from_text: decode_tokens(reconstruct_tokens(walker, from), chars),
      to_text: decode_tokens(reconstruct_tokens(walker, to), chars),
    }
  end
  rows
end

def example_rows(edges : Hash(Int32, Hash(Int32, Int64)),
                 walker : CorpusTrieWalker,
                 chars : Array(Char),
                 limit : Int32)
  rows = [] of NamedTuple(
    from_id: Int32,
    from_text: String,
    first_char_depth: Int32,
    edge_len: Int32,
    edge_mass: Int32,
    fanout: Int32,
    total: Int64,
    top_successors: Array(NamedTuple(to_id: Int32, count: Int64, pct: Float64, text: String))
  )

  items = edges.to_a.sort_by do |(from, h)|
    vals = h.values
    total = vals.sum(0_i64)
    top = vals.max? || 0_i64
    {-h.size, -total, -top}
  end

  items.first(limit).each do |from, h|
    total = h.values.sum(0_i64)
    succ = [] of NamedTuple(to_id: Int32, count: Int64, pct: Float64, text: String)
    h.to_a.sort_by { |(_, count)| -count }.first(8).each do |to, count|
      succ << {
        to_id: to,
        count: count,
        pct: pct(count, total),
        text: decode_tokens(reconstruct_tokens(walker, to), chars),
      }
    end
    edge_len = walker.edge_len_of(from)
    rows << {
      from_id: from,
      from_text: decode_tokens(reconstruct_tokens(walker, from), chars),
      first_char_depth: walker.endpoint_depth_of(from) - edge_len + 1,
      edge_len: edge_len,
      edge_mass: walker.edge_mass_of(from),
      fanout: h.size,
      total: total,
      top_successors: succ,
    }
  end
  rows
end

def write_successor_table(path : String,
                          edges : Hash(Int32, Hash(Int32, Int64)),
                          radix_count : Int32,
                          d_max : Int32,
                          mode : Int32,
                          mass_one_only : Bool)
  successors = Slice(Int32).new(radix_count, -1)
  deterministic = 0_i64
  skipped_fanout = 0_i64
  edges.each do |from, h|
    if h.size == 1
      successors[from] = h.first_key
      deterministic += 1
    else
      skipped_fanout += 1
    end
  end

  File.open(path, "wb") do |io|
    io.write("ASUC".to_slice)
    io.write_bytes(1_u32, IO::ByteFormat::LittleEndian) # version
    io.write_bytes(radix_count.to_u32, IO::ByteFormat::LittleEndian)
    io.write_bytes(d_max.to_u32, IO::ByteFormat::LittleEndian)
    io.write_bytes(mode.to_u32, IO::ByteFormat::LittleEndian) # 1=end, 2=head
    io.write_bytes((mass_one_only ? 1_u32 : 0_u32), IO::ByteFormat::LittleEndian)
    io.write_bytes(deterministic.to_u64, IO::ByteFormat::LittleEndian)
    io.write_bytes(skipped_fanout.to_u64, IO::ByteFormat::LittleEndian)
    successors.each { |succ| io.write_bytes(succ, IO::ByteFormat::LittleEndian) }
  end
end

summary = {
  "trie" => trie_dir,
  "corpus" => corpus_path,
  "out_dir" => out_dir,
  "tokens" => n_tokens,
  "vocab_size" => dataset.vocab_size,
  "d_max" => d_max,
  "wrap" => wrap,
  "mass_one_only" => mass_one_only,
  "cap_occurrences" => cap_occurrences.size,
  "cap_nodes" => cap_node_ids.size,
  "cap_nodes_edge_mass_1" => mass_one_caps,
  "cap_nodes_edge_mass_1_pct" => pct(mass_one_caps.to_i64, cap_node_ids.size.to_i64),
  "cap_nodes_compressed_edge_len_gt_1" => compressed_caps,
  "cap_nodes_compressed_edge_len_gt_1_pct" => pct(compressed_caps.to_i64, cap_node_ids.size.to_i64),
  "end_anchor_skipped" => end_skipped,
  "head_anchor_skipped" => head_skipped,
  "end_anchor" => concentration_stats(end_edges),
  "head_anchor" => concentration_stats(head_edges),
}

File.write(File.join(out_dir, "summary.json"), JSON.parse(summary.to_json).to_pretty_json)
File.write(File.join(out_dir, "top_end_edges.json"), JSON.parse(top_edge_rows(end_edges, walker, chars, top_edges_limit).to_json).to_pretty_json)
File.write(File.join(out_dir, "top_head_edges.json"), JSON.parse(top_edge_rows(head_edges, walker, chars, top_edges_limit).to_json).to_pretty_json)
File.write(File.join(out_dir, "examples_end.json"), JSON.parse(example_rows(end_edges, walker, chars, sample_limit).to_json).to_pretty_json)
File.write(File.join(out_dir, "examples_head.json"), JSON.parse(example_rows(head_edges, walker, chars, sample_limit).to_json).to_pretty_json)
write_successor_table(File.join(out_dir, "successors_end.bin"), end_edges, walker.radix_count, d_max, 1, mass_one_only)
write_successor_table(File.join(out_dir, "successors_head.bin"), head_edges, walker.radix_count, d_max, 2, mass_one_only)

puts JSON.parse(summary.to_json).to_pretty_json
