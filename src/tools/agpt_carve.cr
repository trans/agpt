# bin/agpt_carve
#
# Carve a corpus into train + held-out files per a carve recipe.
# Two invocation modes:
#   1) --config <yaml>           — read corpus.carve.* from a YAML config
#   2) explicit CLI flags        — --source --mode --ratio [--chunks --seed] [--out-dir]
#
# Output layout: <out-dir>/{train_corpus.txt, heldout_corpus.txt,
#                          heldout_chunks/chunk_NN.txt (sample mode only),
#                          manifest.json}
#
# When --out-dir is omitted, it's auto-derived as
# data/.splits/<short-hash>/ where the hash covers (source_sha256,
# mode, ratio, chunks, seed). Identical carve params → identical cache
# dir → reused across runs.

require "option_parser"
require "yaml"
require "json"
require "digest"
require "file_utils"

config_path = ""
source      = ""
mode        = ""
ratio       = 0.0
chunks      = 10
seed        = 42_u64
out_dir     = ""
quiet       = false

OptionParser.parse do |p|
  p.banner = <<-USAGE
    Usage: agpt_carve --config <yaml>
           agpt_carve --source <path> --mode <sample|tail> --ratio <r>
                      [--chunks <c>] [--seed <s>] [--out-dir <dir>]
    USAGE
  p.on("--config PATH", "Read carve spec from a YAML config (corpus.carve.*)") { |v| config_path = v }
  p.on("--source PATH", "Source corpus file") { |v| source = v }
  p.on("--mode MODE", "Carve mode: sample | tail") { |v| mode = v }
  p.on("--ratio R", "Held-out fraction (0 < r < 1)") { |v| ratio = v.to_f }
  p.on("--chunks N", "Number of disjoint chunks (sample mode only)") { |v| chunks = v.to_i }
  p.on("--seed N", "RNG seed (sample mode only)") { |v| seed = v.to_u64 }
  p.on("--out-dir PATH", "Output directory (default: data/.splits/<hash>/)") { |v| out_dir = v }
  p.on("--quiet", "Suppress per-step prints") { quiet = true }
  p.on("-h", "--help", "Show this help") { puts p; exit 0 }
end

# --- If --config given, extract from YAML's corpus.carve block ---
unless config_path.empty?
  yaml_root = YAML.parse(File.read(config_path))
  corpus_node = yaml_root["corpus"]?
  if corpus_node.nil?
    STDERR.puts "Error: --config #{config_path} has no 'corpus' section"
    exit 1
  end
  carve_node = corpus_node["carve"]?
  if carve_node.nil?
    STDERR.puts "Error: --config #{config_path} has no 'corpus.carve' block"
    exit 1
  end

  src_node = carve_node["source"]?
  if src_node.nil?
    STDERR.puts "Error: corpus.carve.source required in #{config_path}"
    exit 1
  end
  source = src_node.as_s

  mode_node = carve_node["mode"]?
  if mode_node.nil?
    STDERR.puts "Error: corpus.carve.mode required in #{config_path}"
    exit 1
  end
  mode = mode_node.as_s

  ratio_node = carve_node["ratio"]?
  if ratio_node.nil?
    STDERR.puts "Error: corpus.carve.ratio required in #{config_path}"
    exit 1
  end
  ratio = ratio_node.as_f

  if mode == "sample"
    chunks_node = carve_node["chunks"]?
    if chunks_node.nil?
      STDERR.puts "Error: corpus.carve.chunks required for mode: sample"
      exit 1
    end
    chunks = chunks_node.as_i

    seed_node = carve_node["seed"]?
    if seed_node.nil?
      STDERR.puts "Error: corpus.carve.seed required for mode: sample"
      exit 1
    end
    seed = seed_node.as_i.to_u64
  end
end

# --- Validate inputs ---
if source.empty?
  STDERR.puts "Error: --source (or corpus.carve.source) is required"
  exit 1
end
unless File.exists?(source)
  STDERR.puts "Error: source corpus not found: #{source}"
  exit 1
end
unless mode == "sample" || mode == "tail"
  STDERR.puts "Error: --mode must be 'sample' or 'tail' (got #{mode.inspect})"
  exit 1
end
unless 0.0 < ratio && ratio < 1.0
  STDERR.puts "Error: --ratio must be in (0, 1) (got #{ratio})"
  exit 1
end

# --- Read source corpus + compute content hash ---
corpus = File.read(source)
n      = corpus.bytesize
source_sha = Digest::SHA256.hexdigest(corpus)

# Hash covers source + carve params for cache-key purposes.
hash_input = "#{source_sha}|#{mode}|#{ratio}|#{chunks}|#{seed}"
short_hash = Digest::SHA256.hexdigest(hash_input)[0, 16]

if out_dir.empty?
  out_dir = "data/.splits/#{short_hash}"
end

# --- Cache hit? ---
manifest_path = "#{out_dir}/manifest.json"
if File.exists?(manifest_path)
  puts "Cache hit: #{out_dir} already exists (manifest.json present). Reusing." unless quiet
  puts ""
  print_snippet(source, out_dir, mode, ratio, chunks, seed)
  exit 0
end

FileUtils.mkdir_p(out_dir)
holdout_total = (n.to_f * ratio).to_i

# --- Carve ---
case mode
when "tail"
  split_at = n - holdout_total
  train    = corpus.byte_slice(0, split_at)
  heldout  = corpus.byte_slice(split_at, holdout_total)
  File.write("#{out_dir}/train_corpus.txt", train)
  File.write("#{out_dir}/heldout_corpus.txt", heldout)
  train_sha   = Digest::SHA256.hexdigest(train)
  heldout_sha = Digest::SHA256.hexdigest(heldout)

  manifest = {
    "source_path"   => source,
    "source_sha256" => source_sha,
    "source_size"   => n,
    "mode"          => "tail",
    "ratio"         => ratio,
    "split_at"      => split_at,
    "train_size"    => train.bytesize,
    "heldout_size"  => heldout.bytesize,
    "train_sha256"  => train_sha,
    "heldout_sha256" => heldout_sha,
  }
  File.write(manifest_path, manifest.to_pretty_json)
when "sample"
  chunk_size = holdout_total // chunks
  if chunk_size < 64
    STDERR.puts "Error: chunk_size=#{chunk_size} is too small; reduce --chunks or raise --ratio"
    exit 1
  end

  # Seeded rejection sampling for non-overlapping chunks.
  rng = Random.new(seed)
  positions = [] of Int32
  max_attempts = 100 * chunks
  attempts = 0
  while positions.size < chunks
    attempts += 1
    if attempts > max_attempts
      STDERR.puts "Error: could not place #{chunks} non-overlapping chunks of size #{chunk_size} after #{max_attempts} attempts"
      exit 1
    end
    pos = rng.rand(n - chunk_size)
    if positions.all? { |p| (pos - p).abs >= chunk_size }
      positions << pos
    end
  end
  positions.sort!

  # Write per-chunk files + record metadata
  chunks_dir = "#{out_dir}/heldout_chunks"
  FileUtils.mkdir_p(chunks_dir)
  chunk_records = [] of Hash(String, JSON::Any)
  positions.each_with_index do |p, i|
    chunk = corpus.byte_slice(p, chunk_size)
    chunk_path = "#{chunks_dir}/chunk_#{i.to_s.rjust(2, '0')}.txt"
    File.write(chunk_path, chunk)
    chunk_records << {
      "index"   => JSON::Any.new(i.to_i64),
      "start"   => JSON::Any.new(p.to_i64),
      "end"     => JSON::Any.new((p + chunk_size).to_i64),
      "size"    => JSON::Any.new(chunk_size.to_i64),
      "sha256"  => JSON::Any.new(Digest::SHA256.hexdigest(chunk)),
    }
  end

  # Build train: source minus the K chunks, concatenated in order.
  train_io = IO::Memory.new
  last_end = 0
  positions.each do |p|
    train_io << corpus.byte_slice(last_end, p - last_end)
    last_end = p + chunk_size
  end
  train_io << corpus.byte_slice(last_end, n - last_end)
  train = train_io.to_s

  # Concatenated heldout = chunks in position order
  heldout_io = IO::Memory.new
  positions.each { |p| heldout_io << corpus.byte_slice(p, chunk_size) }
  heldout = heldout_io.to_s

  File.write("#{out_dir}/train_corpus.txt", train)
  File.write("#{out_dir}/heldout_corpus.txt", heldout)
  train_sha   = Digest::SHA256.hexdigest(train)
  heldout_sha = Digest::SHA256.hexdigest(heldout)

  manifest = {
    "source_path"    => source,
    "source_sha256"  => source_sha,
    "source_size"    => n,
    "mode"           => "sample",
    "ratio"          => ratio,
    "chunks"         => chunks,
    "chunk_size"     => chunk_size,
    "seed"           => seed,
    "chunk_records"  => chunk_records,
    "train_size"     => train.bytesize,
    "heldout_size"   => heldout.bytesize,
    "train_sha256"   => train_sha,
    "heldout_sha256" => heldout_sha,
  }
  File.write(manifest_path, manifest.to_pretty_json)
end

unless quiet
  puts "Carved split written to #{out_dir}/"
  puts ""
end
print_snippet(source, out_dir, mode, ratio, chunks, seed)

# --- Helpers ---
def print_snippet(source : String, out_dir : String, mode : String, ratio : Float64, chunks : Int32, seed : UInt64)
  puts "Paste into your experiment YAML:"
  puts ""
  puts "corpus:"
  puts "  path: #{out_dir}/train_corpus.txt"
  puts "  heldout: #{out_dir}/heldout_corpus.txt"
  puts "  vocab_source: #{source}"
  puts "  carve:"
  puts "    source: #{source}"
  puts "    mode: #{mode}"
  puts "    ratio: #{ratio}"
  if mode == "sample"
    puts "    chunks: #{chunks}"
    puts "    seed: #{seed}"
  end
end
