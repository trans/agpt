# Build an inspection graph for phase-column trie states.
#
# This is deliberately diagnostic, not a trainer input. It preserves the radix
# trie edge structure while accumulating collapsed views such as
# (token, depth, phase) and (token, phase), so we can see how much merging each
# identity choice would introduce.

require "../agpt"
require "json"
require "option_parser"
require "html"

include MicroGPT::AGPT

trie_dir = ""
position_data_dir = ""
vocab_path = "data/input.txt"
out_dir = "rnd/sampled-node-phase-rope/phase-column-graph"
min_edge_mass = 128_i64
top_states = 18
top_edges = 260
phase_window_override = 0

OptionParser.parse do |p|
  p.banner = "Usage: phase_column_graph --trie DIR --position-data DIR [--vocab FILE] [--out DIR]"
  p.on("--trie DIR", "Prefix radix trie directory") { |v| trie_dir = v }
  p.on("--position-data DIR", "Position data directory") { |v| position_data_dir = v }
  p.on("--vocab FILE", "Corpus/vocab source for token labels [default data/input.txt]") { |v| vocab_path = v }
  p.on("--out DIR", "Output directory [default rnd/sampled-node-phase-rope/phase-column-graph]") { |v| out_dir = v }
  p.on("--min-edge-mass N", "Minimum rendered edge mass [default 128]") { |v| min_edge_mass = v.to_i64 }
  p.on("--top-states N", "Rendered states per phase column [default 18]") { |v| top_states = v.to_i }
  p.on("--top-edges N", "Maximum rendered edges [default 260]") { |v| top_edges = v.to_i }
  p.on("--phase-window N", "Render only first N phase columns [default all]") { |v| phase_window_override = v.to_i }
  p.on("-h", "--help", "") { puts p; exit 0 }
end

raise "--trie required" if trie_dir.empty?
raise "--position-data required" if position_data_dir.empty?

prt_path = File.join(position_data_dir, "prefix_radix_to_substring.bin")
ppt_path = File.join(position_data_dir, "prefix_position_table.bin")
[prt_path, ppt_path].each { |path| raise "Missing: #{path}" unless File.exists?(path) }

Dir.mkdir_p(out_dir)

reader = RadixTrieReader.new(trie_dir, max_cached: 4)
radix_to_substring = File.open(prt_path, "rb") { |f| RadixToSubstring.read_from(f) }
position_table = File.open(ppt_path, "rb") { |f| PositionTable.read_from(f) }

if radix_to_substring.radix_count != reader.radix_count
  raise "radix count mismatch: trie=#{reader.radix_count} position-map=#{radix_to_substring.radix_count}"
end

chars = File.read(vocab_path).chars.uniq.sort
window = position_table.window_size
render_window = phase_window_override > 0 ? Math.min(phase_window_override, window) : window

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

def pct(num, den)
  return 0.0 if den == 0
  100.0 * num.to_f / den.to_f
end

struct StateKey
  getter token : Int32
  getter depth : Int32
  getter phase : Int32

  def initialize(@token, @depth, @phase)
  end
end

struct TokenPhaseKey
  getter token : Int32
  getter phase : Int32

  def initialize(@token, @phase)
  end
end

struct EdgeKey
  getter from : StateKey
  getter to : StateKey

  def initialize(@from, @to)
  end
end

endpoint_token = Slice(Int32).new(reader.radix_count, -1)
endpoint_depth = Slice(Int32).new(reader.radix_count, 0)

state_mass = Hash(StateKey, Int64).new(0_i64)
token_phase_mass = Hash(TokenPhaseKey, Int64).new(0_i64)
edge_mass = Hash(EdgeKey, Int64).new(0_i64)

records_seen = 0_i64
records_with_sid = 0_i64
position_bins_seen = 0_i64
implicit_full_states = 0_i64
implicit_state_mass = 0_i64
token_depth_phase_observations = 0_i64
token_phase_observations = 0_i64
skipped_missing_sid = 0_i64
skipped_empty_bins = 0_i64
edge_observations = 0_i64
edge_mass_total = 0_i64

by_depth = Hash(Int32, Tuple(Int64, Int64, Int64)).new { |h, k| h[k] = {0_i64, 0_i64, 0_i64} }

reader.each do |rec|
  next if rec.id == 0
  records_seen += 1
  endpoint_depth[rec.id] = rec.endpoint_depth
  endpoint_token[rec.id] = rec.edge_tokens[-1]? || -1

  sid = radix_to_substring.substring_id_for(rec.id)
  if sid < 0 || sid >= position_table.substring_count
    skipped_missing_sid += 1
    next
  end
  records_with_sid += 1
  bins = position_table.bins(sid)
  if bins.empty?
    skipped_empty_bins += 1
    next
  end

  nodes_at_depth, bins_at_depth, mass_at_depth = by_depth[rec.endpoint_depth]
  nodes_at_depth += 1
  bins_at_depth += bins.size

  bins.each do |bin|
    start_phase = bin.pos.to_i
    count = bin.count.to_i64
    position_bins_seen += 1

    prev_token = -1
    prev_depth = 0
    if rec.parent_id > 0
      prev_token = endpoint_token[rec.parent_id]
      prev_depth = endpoint_depth[rec.parent_id]
    end

    rec.edge_tokens.each_with_index do |token, offset|
      depth = rec.first_char_depth + offset
      phase = (start_phase + depth - 1) % window
      state = StateKey.new(token, depth, phase)
      state_mass[state] += count
      token_phase_mass[TokenPhaseKey.new(token, phase)] += count
      implicit_full_states += 1
      implicit_state_mass += count

      if prev_token >= 0 && prev_depth > 0
        prev_phase = (start_phase + prev_depth - 1) % window
        prev = StateKey.new(prev_token, prev_depth, prev_phase)
        edge_mass[EdgeKey.new(prev, state)] += count
        edge_observations += 1
        edge_mass_total += count
      end

      prev_token = token
      prev_depth = depth
    end
    mass_at_depth += count
  end
  by_depth[rec.endpoint_depth] = {nodes_at_depth, bins_at_depth, mass_at_depth}
end

token_depth_phase_observations = state_mass.size.to_i64
token_phase_observations = token_phase_mass.size.to_i64

states_by_phase = Hash(Int32, Array(Tuple(StateKey, Int64))).new do |h, k|
  h[k] = [] of Tuple(StateKey, Int64)
end
state_mass.each do |key, mass|
  next if key.phase >= render_window
  states_by_phase[key.phase] << {key, mass}
end
states_by_phase.each_value do |items|
  items.sort_by! { |(key, mass)| {-mass, key.depth, key.token} }
end

rendered_states = {} of StateKey => Int32
state_nodes = [] of Tuple(StateKey, Int64)
render_window.times do |phase|
  states_by_phase[phase].first(top_states).each do |item|
    key, mass = item
    rendered_states[key] = state_nodes.size
    state_nodes << item
  end
end

rendered_edges = edge_mass.to_a.select do |(key, mass)|
  mass >= min_edge_mass && rendered_states.has_key?(key.from) && rendered_states.has_key?(key.to)
end
rendered_edges.sort_by! { |(key, mass)| -mass }
rendered_edges = rendered_edges.first(top_edges)

summary = {
  "trie" => trie_dir,
  "position_data" => position_data_dir,
  "window" => window,
  "render_window" => render_window,
  "radix_records_seen" => records_seen,
  "records_with_position_substring" => records_with_sid,
  "skipped_missing_sid" => skipped_missing_sid,
  "skipped_empty_bins" => skipped_empty_bins,
  "position_bins_seen" => position_bins_seen,
  "implicit_full_state_observations" => implicit_full_states,
  "implicit_state_mass" => implicit_state_mass,
  "distinct_token_depth_phase_states" => token_depth_phase_observations,
  "distinct_token_phase_states" => token_phase_observations,
  "distinct_token_depth_phase_edges" => edge_mass.size,
  "edge_observations" => edge_observations,
  "edge_mass_total" => edge_mass_total,
  "merge_ratio_full_to_token_depth_phase" => implicit_full_states.to_f / Math.max(1, token_depth_phase_observations).to_f,
  "merge_ratio_token_depth_phase_to_token_phase" => token_depth_phase_observations.to_f / Math.max(1, token_phase_observations).to_f,
}

summary_path = File.join(out_dir, "summary.json")
File.write(summary_path, summary.to_pretty_json)

depth_rows = by_depth.keys.sort.map do |depth|
  nodes, bins, mass = by_depth[depth]
  {
    "depth" => depth,
    "nodes_with_bins" => nodes,
    "position_bins" => bins,
    "phase_mass" => mass,
  }
end
File.write(File.join(out_dir, "by_depth.json"), depth_rows.to_pretty_json)

top_state_rows = state_mass.to_a.sort_by { |(key, mass)| -mass }.first(400).map do |(key, mass)|
  {
    "token" => key.token,
    "label" => token_label(key.token, chars),
    "depth" => key.depth,
    "phase" => key.phase,
    "mass" => mass,
  }
end
File.write(File.join(out_dir, "top_states.json"), top_state_rows.to_pretty_json)

top_edge_rows = edge_mass.to_a.sort_by { |(key, mass)| -mass }.first(400).map do |(key, mass)|
  {
    "from" => {
      "token" => key.from.token,
      "label" => token_label(key.from.token, chars),
      "depth" => key.from.depth,
      "phase" => key.from.phase,
    },
    "to" => {
      "token" => key.to.token,
      "label" => token_label(key.to.token, chars),
      "depth" => key.to.depth,
      "phase" => key.to.phase,
    },
    "mass" => mass,
  }
end
File.write(File.join(out_dir, "top_edges.json"), top_edge_rows.to_pretty_json)

col_w = 92
row_h = 28
margin_left = 70
margin_top = 70
height = margin_top + (top_states + 3) * row_h
width = margin_left + render_window * col_w + 80

positions = {} of StateKey => Tuple(Float64, Float64)
render_window.times do |phase|
  states_by_phase[phase].first(top_states).each_with_index do |(key, mass), rank|
    x = margin_left + phase * col_w + col_w / 2
    y = margin_top + rank * row_h
    positions[key] = {x.to_f, y.to_f}
  end
end

max_edge_mass = rendered_edges.map { |(_key, mass)| mass }.max? || 1_i64
max_state_mass = state_nodes.map { |(_key, mass)| mass }.max? || 1_i64

svg = String.build do |io|
  io << %(<svg viewBox="0 0 #{width} #{height}" xmlns="http://www.w3.org/2000/svg" role="img">\n)
  io << %(<rect width="#{width}" height="#{height}" fill="#fbfaf8"/>\n)
  render_window.times do |phase|
    x = margin_left + phase * col_w + col_w / 2
    io << %(<line x1="#{x}" y1="42" x2="#{x}" y2="#{height - 20}" stroke="#ddd6cc" stroke-width="1"/>\n)
    io << %(<text x="#{x}" y="28" text-anchor="middle" font-size="12" fill="#5c5348">#{phase}</text>\n)
  end
  rendered_edges.reverse_each do |(edge, mass)|
    from = positions[edge.from]?
    to = positions[edge.to]?
    next if from.nil? || to.nil?
    x1, y1 = from
    x2, y2 = to
    sw = 0.35 + 4.0 * Math.sqrt(mass.to_f / max_edge_mass.to_f)
    opacity = 0.14 + 0.5 * Math.sqrt(mass.to_f / max_edge_mass.to_f)
    io << %(<path d="M #{x1} #{y1} C #{(x1 + x2) / 2} #{y1}, #{(x1 + x2) / 2} #{y2}, #{x2} #{y2}" fill="none" stroke="#4b7891" stroke-width="#{sw.round(2)}" opacity="#{opacity.round(2)}"/>\n)
  end
  state_nodes.each do |(key, mass)|
    pos = positions[key]?
    next if pos.nil?
    x, y = pos
    r = 4.0 + 9.0 * Math.sqrt(mass.to_f / max_state_mass.to_f)
    label = HTML.escape("#{token_label(key.token, chars)} d#{key.depth}")
    io << %(<circle cx="#{x}" cy="#{y}" r="#{r.round(2)}" fill="#f3c45b" stroke="#5f5130" stroke-width="0.8"/>\n)
    io << %(<text x="#{x}" y="#{y + 4}" text-anchor="middle" font-size="10" fill="#211f1b">#{label}</text>\n)
  end
  io << "</svg>\n"
end

html_path = File.join(out_dir, "phase_column_graph.html")
html = String.build do |io|
  io << "<!doctype html><meta charset=\"utf-8\"><title>Phase Column Graph</title>\n"
  io << "<style>"
  io << "body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:24px;background:#fbfaf8;color:#24211d}"
  io << "h1{font-size:22px;margin:0 0 10px} h2{font-size:16px;margin-top:24px}"
  io << ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px;margin:18px 0}"
  io << ".metric{border:1px solid #ddd6cc;background:white;padding:10px;border-radius:6px}"
  io << ".metric b{display:block;font-size:18px} .metric span{color:#6f665b;font-size:12px}"
  io << ".svgwrap{overflow:auto;border:1px solid #ddd6cc;background:white;border-radius:6px;padding:12px}"
  io << "table{border-collapse:collapse;font-size:13px;background:white}td,th{border:1px solid #ddd6cc;padding:4px 7px;text-align:right}th{background:#efeae2}td:first-child,th:first-child{text-align:left}"
  io << "code{background:#efeae2;padding:1px 4px;border-radius:3px}"
  io << "</style>\n"
  io << "<h1>Phase Column Graph</h1>\n"
  io << "<p>Diagnostic graph from <code>#{HTML.escape(trie_dir)}</code> and <code>#{HTML.escape(position_data_dir)}</code>. Nodes rendered below use collapsed <code>(token, depth, phase)</code> identity; edge masses come from preserved radix trie edges before collapse.</p>\n"
  io << "<div class=\"grid\">"
  {
    "window" => window.to_s,
    "radix records" => records_seen.to_s,
    "position bins" => position_bins_seen.to_s,
    "implicit full states" => implicit_full_states.to_s,
    "distinct (token, depth, phase)" => token_depth_phase_observations.to_s,
    "distinct (token, phase)" => token_phase_observations.to_s,
    "distinct collapsed edges" => edge_mass.size.to_s,
    "full -> t/d/p merge ratio" => sprintf("%.1fx", summary["merge_ratio_full_to_token_depth_phase"]),
    "t/d/p -> t/p merge ratio" => sprintf("%.1fx", summary["merge_ratio_token_depth_phase_to_token_phase"]),
  }.each do |name, value|
    io << "<div class=\"metric\"><b>#{HTML.escape(value)}</b><span>#{HTML.escape(name)}</span></div>"
  end
  io << "</div>"
  io << "<h2>Rendered Columns</h2><p>Top #{top_states} states per phase, first #{render_window} phase columns. Edges below mass #{min_edge_mass} are omitted from the drawing.</p>"
  io << "<div class=\"svgwrap\">#{svg}</div>"
  io << "<h2>Top Collapsed Edges</h2><table><tr><th>from</th><th>to</th><th>mass</th></tr>"
  top_edge_rows.first(40).each do |row|
    from = row["from"].as(Hash)
    to = row["to"].as(Hash)
    io << "<tr><td>#{HTML.escape(from["label"].to_s)} d#{from["depth"]} p#{from["phase"]}</td><td>#{HTML.escape(to["label"].to_s)} d#{to["depth"]} p#{to["phase"]}</td><td>#{row["mass"]}</td></tr>"
  end
  io << "</table>"
  io << "<h2>Top States</h2><table><tr><th>state</th><th>mass</th></tr>"
  top_state_rows.first(60).each do |row|
    io << "<tr><td>#{HTML.escape(row["label"].to_s)} d#{row["depth"]} p#{row["phase"]}</td><td>#{row["mass"]}</td></tr>"
  end
  io << "</table>"
end
File.write(html_path, html)

puts "Phase Column Graph"
puts "  summary: #{summary_path}"
puts "  html   : #{html_path}"
puts "  window : #{window}"
puts "  radix records: #{records_seen}"
puts "  position bins: #{position_bins_seen}"
puts "  implicit full states: #{implicit_full_states}"
puts "  distinct (token, depth, phase): #{token_depth_phase_observations}"
puts "  distinct (token, phase): #{token_phase_observations}"
puts "  distinct collapsed edges: #{edge_mass.size}"
puts "  full -> token/depth/phase merge ratio: #{sprintf("%.2fx", summary["merge_ratio_full_to_token_depth_phase"])}"
puts "  token/depth/phase -> token/phase merge ratio: #{sprintf("%.2fx", summary["merge_ratio_token_depth_phase_to_token_phase"])}"
