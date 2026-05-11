# Sliding-window AGPT perplexity — v1 inference prototype.
#
# Tests the simplest variant of the sliding-window-AGPT design from
# notes/agpt/sliding_window_agpt.md: LOGIT POOLING.
#
# For each target position i, compute the predicted log-prob of
# tokens[i] from d different contributing windows. Each window starts
# at a different corpus position w ∈ [i-d, i-1] and walks d chars
# forward; within-window position (i-1-w) predicts tokens[i] with
# (i-1-w) chars of prior context already integrated.
#
# Pool the d log-prob vectors and compute NLL on the pooled target.
#
# Pooling rules:
#   uniform   — uniform mean of log-probs (geometric mean of probs)
#   deep_only — use only the d-1 (deepest-context) contributor.
#               Equivalent to standard PPL@d; sanity check baseline.
#   depth_w   — weighted mean, w_k ∝ k+1 (favor deeper contributors)
#
# Usage:
#   bin/agpt_sliding_window_perplexity \
#     --model /tmp/agpt-gut-d16-pd1.model \
#     --file data/gutenberg_5m.txt \
#     --d 16 \
#     --pool uniform \
#     --max-positions 4096

require "../../lib/microgpt/src/microgpt"
require "option_parser"

include MicroGPT

model_path = ""
test_file  = ""
vocab_file = ""
d_window   = 16          # the trie depth / per-window context length
pool_mode  = "uniform"   # uniform | deep_only | depth_w
backend    = "openblas"
max_positions = -1

OptionParser.parse do |p|
  p.banner = "Usage: agpt_sliding_window_perplexity --model PATH --file PATH [options]"
  p.on("--model PATH", "Model checkpoint path") { |v| model_path = v }
  p.on("--file PATH", "Held-out test text file") { |v| test_file = v }
  p.on("--vocab-file PATH", "Vocab source (defaults to --file)") { |v| vocab_file = v }
  p.on("--d N", "Per-window context length (default 16). Must match model's effective d.") { |v| d_window = v.to_i }
  p.on("--pool MODE", "uniform | deep_only | depth_w (default uniform)") { |v| pool_mode = v }
  p.on("--backend B", "crystal|openblas|cublas (default openblas)") { |v| backend = v }
  p.on("--max-positions N", "Cap positions scored") { |v| max_positions = v.to_i }
  p.on("-h", "--help", "") { puts p; exit 0 }
end

if model_path.empty? || test_file.empty?
  STDERR.puts "--model and --file are required."
  exit 1
end
if !["uniform", "deep_only", "depth_w"].includes?(pool_mode)
  STDERR.puts "--pool must be one of uniform, deep_only, depth_w"
  exit 1
end

case backend
when "openblas" then MicroGPT.use_openblas!
when "cublas"   then MicroGPT.use_cublas!
else                  MicroGPT.use_crystal!
end

# Vocab + load corpus
vf = vocab_file.empty? ? test_file : vocab_file
dataset = CharDataset.new(File.read(vf))
STDERR.puts "Vocab: #{dataset.vocab_size} unique chars from #{vf}"

test_text = File.read(test_file)
filtered = String.build { |io| test_text.each_char { |c| io << c if dataset.char_to_id.has_key?(c) } }
test_text = filtered
tokens = dataset.encode(test_text)
STDERR.puts "Tokens: #{tokens.size}"

# Auto-detect architecture from checkpoint
d_model = 64; n_heads = 4; n_layers = 2; d_ff = 256; ckpt_seq_len = 128
File.open(model_path, "rb") do |f|
  magic = f.read_bytes(UInt32, IO::ByteFormat::LittleEndian)
  if magic == 0x4D475054_u32
    d_model      = f.read_bytes(Int32, IO::ByteFormat::LittleEndian)
    n_heads      = f.read_bytes(Int32, IO::ByteFormat::LittleEndian)
    n_layers     = f.read_bytes(Int32, IO::ByteFormat::LittleEndian)
    d_ff         = f.read_bytes(Int32, IO::ByteFormat::LittleEndian)
    _vocab       = f.read_bytes(Int32, IO::ByteFormat::LittleEndian)
    ckpt_seq_len = f.read_bytes(Int32, IO::ByteFormat::LittleEndian)
  end
end
STDERR.puts "Arch: d_model=#{d_model} n_heads=#{n_heads} n_layers=#{n_layers} d_ff=#{d_ff} ckpt_seq=#{ckpt_seq_len}"

config = Config.new
config.d_model    = d_model
config.n_heads    = n_heads
config.n_layers   = n_layers
config.d_ff       = d_ff
config.seq_len    = d_window
config.vocab_size = dataset.vocab_size

model = MiniGPT.new(config)
model.load(model_path)
STDERR.puts "Model loaded"

# Compute pooling weights once (for depth_w)
pool_weights = Array(Float32).new(d_window, 1.0_f32)
case pool_mode
when "depth_w"
  d_window.times { |j| pool_weights[j] = (j + 1).to_f32 }
  total = pool_weights.sum
  d_window.times { |j| pool_weights[j] = pool_weights[j] / total }
else
  d_window.times { |j| pool_weights[j] = 1.0_f32 / d_window }
end

vocab_size = config.vocab_size
n_targets = tokens.size - d_window  # need d chars before any target
n_targets = max_positions if max_positions > 0 && max_positions < n_targets

# For each target i ∈ [d_window, d_window + n_targets - 1]:
#   contributors j ∈ [0, d_window - 1]
#   for j, window starts at w = i - 1 - j, which means context = tokens[i-1-j .. i-1-j+d-1]
#   model.forward(context), grab logits at row j of the output (predicting token at w+j+1 = i)

nll_sum = 0.0
t0 = Time.instant
report_every = (n_targets // 20).clamp(50, 5000)

n_targets.times do |k|
  i = d_window + k
  target = tokens[i]

  if pool_mode == "deep_only"
    # j = d_window - 1, w = i - 1 - (d_window - 1) = i - d_window
    # context = tokens[i - d_window, d_window], last position predicts target.
    context = tokens[i - d_window, d_window]
    logits = model.forward(context)
    # Logits row d_window-1
    row = Array(Float32).new(vocab_size)
    vocab_size.times { |c| row << logits[d_window - 1, c] }
    max_lg = row.max
    exp_sum = 0.0
    row.each { |lg| exp_sum += Math.exp(lg.to_f64 - max_lg.to_f64) }
    log_p = row[target].to_f64 - (Math.log(exp_sum) + max_lg.to_f64)
    nll_sum -= log_p
  else
    # Pool log-probs from d contributing windows.
    pooled_lp = Array(Float64).new(vocab_size, 0.0)
    d_window.times do |j|
      w = i - 1 - j
      next if w < 0
      context = tokens[w, d_window]
      logits = model.forward(context)
      # Log-softmax of row j → log-probs at within-window-position j (predicting token at w+j+1 = i)
      row = Array(Float32).new(vocab_size)
      vocab_size.times { |c| row << logits[j, c] }
      max_lg = row.max
      exp_sum = 0.0
      row.each { |lg| exp_sum += Math.exp(lg.to_f64 - max_lg.to_f64) }
      log_denom = Math.log(exp_sum) + max_lg.to_f64
      weight = pool_weights[j].to_f64
      vocab_size.times do |c|
        log_p_c = row[c].to_f64 - log_denom
        pooled_lp[c] += weight * log_p_c
      end
    end
    nll_sum -= pooled_lp[target]
  end

  if (k + 1) % report_every == 0
    elapsed = (Time.instant - t0).total_seconds
    rate = (k + 1) / elapsed
    eta = (n_targets - k - 1) / rate
    STDERR.puts "  #{k + 1}/#{n_targets} (%.1f/s, ETA %.0fs)" % [rate, eta]
  end
end

elapsed = (Time.instant - t0).total_seconds
mean_nll = nll_sum / n_targets
ppl = Math.exp(mean_nll)
bpc = mean_nll / Math.log(2.0)
puts ""
puts "Pool mode:          #{pool_mode}"
puts "d_window:           #{d_window}"
puts "Positions scored:   #{n_targets}"
puts "Mean per-token NLL: %.6f nats" % mean_nll
puts "Perplexity:         %.4f" % ppl
puts "Bits per char:      %.4f bpc" % bpc
puts "Elapsed:            #{elapsed.round(1)}s (#{(n_targets / elapsed).round(1)} pos/sec)"
