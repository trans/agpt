# Inspect a built position-table directory: verify round-trip, sample
# distributions, report statistics.
#
# Usage: bin/agpt_inspect_position_table <position_data_dir>

require "../agpt"

include MicroGPT::AGPT

if ARGV.size < 1
  STDERR.puts "Usage: inspect_position_table <position_data_dir>"
  exit 1
end

dir = ARGV[0]
sub_path = File.join(dir, "substrings.bin")
prt_path = File.join(dir, "prefix_radix_to_substring.bin")
ppt_path = File.join(dir, "prefix_position_table.bin")
srt_path = File.join(dir, "suffix_radix_to_substring.bin")
spt_path = File.join(dir, "suffix_position_table.bin")
[sub_path, prt_path, ppt_path].each do |p|
  raise "Missing: #{p}" unless File.exists?(p)
end
have_suffix = File.exists?(srt_path) && File.exists?(spt_path)

# Load + corpus for vocab
text = File.read("data/input.txt")
ds = MicroGPT::CharDataset.new(text)
id_to_char = ds.id_to_char

puts "## Substring Catalog"
catalog = File.open(sub_path, "rb") { |f| SubstringCatalog.read_from(f) }
puts "  size: #{catalog.size}"

puts ""
puts "## Prefix Radix → Substring"
prt = File.open(prt_path, "rb") { |f| RadixToSubstring.read_from(f) }
puts "  side: #{prt.side}"
puts "  radix_count: #{prt.radix_count}"

puts ""
puts "## Prefix Position Table"
ppt = File.open(ppt_path, "rb") { |f| PositionTable.read_from(f) }
puts "  window_size: #{ppt.window_size}"
puts "  regime: #{ppt.regime}"
puts "  substring_count: #{ppt.substring_count}"
puts "  total_bins: #{ppt.total_bins}"

# Spot-check a few substrings
def show_dist(catalog, ppt, id_to_char, sid)
  tokens = catalog.tokens_for(sid)
  text = tokens.map { |t| id_to_char[t] }.join("")
  bins = ppt.bins(sid).to_a
  total = ppt.total_count(sid)
  expected = ppt.expected_pos(sid)
  # Top 5 bins by count
  top = bins.sort_by { |b| -b.count.to_i64 }.first(5)
  top_str = top.map { |b| "(#{b.pos}, #{b.count})" }.join(" ")
  printf "  [%-20s] sid=%d total=%d expected=%.2f top5: %s\n", text.inspect, sid, total, expected, top_str
end

puts ""
puts "## Spot checks (first 10 substrings by id)"
10.times do |sid|
  show_dist(catalog, ppt, id_to_char, sid)
end

# Find some specific substrings to inspect
puts ""
puts "## Spot checks (specific patterns, if found)"
patterns = ["the", "and", "qu", "ing", "tion"]
patterns.each do |label|
  tokens = label.chars.map { |c| ds.char_to_id[c].to_i32 }
  if sid = catalog.lookup(tokens)
    show_dist(catalog, ppt, id_to_char, sid)
  else
    puts "  [#{label.inspect}] not in catalog"
  end
end

# If we have suffix data, verify unification by showing the same substring
# from both sides.
if have_suffix
  puts ""
  puts "## Unification check (suffix-side same substring_id)"
  spt = File.open(spt_path, "rb") { |f| PositionTable.read_from(f) }
  patterns.each do |label|
    tokens = label.chars.map { |c| ds.char_to_id[c].to_i32 }
    if sid = catalog.lookup(tokens)
      prefix_total = ppt.total_count(sid)
      suffix_total = spt.total_count(sid)
      prefix_exp = ppt.expected_pos(sid)
      suffix_exp = spt.expected_pos(sid)
      printf "  [%-6s] sid=%d  prefix:(total=%d, E[pos]=%.2f)  suffix:(total=%d, E[end]=%.2f)\n",
        label, sid, prefix_total, prefix_exp, suffix_total, suffix_exp
    end
  end
end

# Aggregate statistics
puts ""
puts "## Aggregate stats"
total_count_all = 0_u64
bin_count_dist = Hash(Int32, Int32).new(0)
ppt.substring_count.times do |sid|
  bins = ppt.bins(sid)
  total_count_all += ppt.total_count(sid)
  bin_count_dist[bins.size] += 1
end
puts "  sum of all counts: #{total_count_all}"
puts "  bins-per-substring histogram (count: substrings):"
bin_count_dist.to_a.sort_by(&.[0]).first(20).each do |bins, count|
  pct = count.to_f / ppt.substring_count * 100
  bar = "#" * (pct * 0.5).to_i
  printf "    %3d bins: %8d (%5.2f%%) %s\n", bins, count, pct, bar
end
