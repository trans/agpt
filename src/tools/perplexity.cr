# Held-out perplexity evaluator.
#
# Loads a trained MGPT model and scores a held-out text file under fixed
# context length (seq_len). For each position i past the warmup (i >= seq_len),
# slides a window ending just before position i, computes next-token
# probability, and accumulates -log(p(target_i)).
#
# Reports: N evaluated positions, mean per-token NLL, perplexity (= exp NLL),
# and bits-per-character (= NLL / log(2)).
#
# Usage:
#   bin/perplexity --model <path> --file <path> [--seq-len N] [--backend crystal|openblas|cublas]
#                  [--d-model N] [--n-layers N] [--n-heads N] [--d-ff N]
#                  [--max-positions N]   # cap eval positions for speed

require "../microgpt"
require "option_parser"

include MicroGPT

model_path = ""
test_file  = ""
vocab_file = ""   # source file for vocab/tokenization; defaults to test_file
seq_len    = 128
d_model    = 64
n_heads    = 4
n_layers   = 2
d_ff       = 256
backend    = "crystal"
max_positions = -1  # -1 = all positions

OptionParser.parse do |p|
  p.banner = "Usage: perplexity --model <path> --file <test text path> [options]"
  p.on("--model PATH", "Model checkpoint path") { |v| model_path = v }
  p.on("--file PATH", "Held-out test text file") { |v| test_file = v }
  p.on("--vocab-file PATH", "Build vocab from this file (defaults to --file). Use the training-corpus file to avoid encode errors when the test text has fewer unique chars.") { |v| vocab_file = v }
  p.on("--seq-len N", "Context window length (default 128)") { |v| seq_len = v.to_i }
  p.on("--d-model N", "") { |v| d_model = v.to_i }
  p.on("--n-heads N", "") { |v| n_heads = v.to_i }
  p.on("--n-layers N", "") { |v| n_layers = v.to_i }
  p.on("--d-ff N", "") { |v| d_ff = v.to_i }
  p.on("--backend B", "crystal|openblas|cublas (default crystal)") { |v| backend = v }
  p.on("--max-positions N", "Cap the number of positions scored (default: all)") { |v| max_positions = v.to_i }
  p.on("-h", "--help", "") { puts p; exit 0 }
end

if model_path.empty? || test_file.empty?
  STDERR.puts "--model and --file are required."
  exit 1
end

# Set backend (mirrors main.cr)
case backend
when "openblas"
  MicroGPT.use_openblas!
when "cublas"
  MicroGPT.use_cublas!
else
  MicroGPT.use_crystal!
end

# Build vocabulary from --vocab-file (defaults to --file). The right choice is
# the TRAINING corpus, so the tokenizer matches the model's training-time vocab.
vf = vocab_file.empty? ? test_file : vocab_file
vocab_source = File.read(vf)
dataset = CharDataset.new(vocab_source)
STDERR.puts "Vocab source: #{vf} → #{dataset.vocab_size} unique chars"

test_text = File.read(test_file)
STDERR.puts "Test file: #{test_file} (#{test_text.bytesize} bytes)"

# Drop any characters in test_text that aren't in the vocab (e.g., OOD chars).
filtered = String.build do |io|
  test_text.each_char { |c| io << c if dataset.char_to_id.has_key?(c) }
end
if filtered.bytesize < test_text.bytesize
  STDERR.puts "Dropped #{test_text.bytesize - filtered.bytesize} out-of-vocab chars."
end
test_text = filtered

# Build model and load checkpoint
config = Config.new
config.d_model = d_model
config.n_heads = n_heads
config.n_layers = n_layers
config.d_ff = d_ff
config.seq_len = seq_len
config.vocab_size = dataset.vocab_size

model = MiniGPT.new(config)
model.load(model_path)
STDERR.puts "Model loaded: #{model_path} (d_model=#{config.d_model}, n_layers=#{config.n_layers}, vocab=#{config.vocab_size}, seq_len=#{config.seq_len})"

tokens = dataset.encode(test_text)
STDERR.puts "Tokens: #{tokens.size}"

if tokens.size <= seq_len
  STDERR.puts "Test text too short for a single window (need > seq_len = #{seq_len})."
  exit 1
end

# Evaluation loop. For each position i >= seq_len, predict target = tokens[i] from
# context = tokens[i-seq_len..i-1]. Use the model's forward over the whole context
# (length seq_len), take logits at the last position (position seq_len - 1),
# softmax, record -log p(target).

total_positions = tokens.size - seq_len
if max_positions > 0 && max_positions < total_positions
  total_positions = max_positions
end

nll_sum = 0.0
progress_step = (total_positions // 20).clamp(1, Int32::MAX)
progress_every = progress_step.clamp(100, 100_000)

t0 = Time.instant
total_positions.times do |k|
  i = seq_len + k
  context = tokens[i - seq_len, seq_len]
  target = tokens[i]

  logits = model.forward(context)  # [seq_len, V]
  # Softmax on the LAST position's row
  v = config.vocab_size
  last_row = Array(Float32).new(v)
  v.times { |j| last_row << logits[seq_len - 1, j] }
  # Numerically-stable log-softmax
  max_logit = last_row.max
  exp_sum = 0.0
  last_row.each { |lg| exp_sum += Math.exp(lg.to_f64 - max_logit.to_f64) }
  log_denom = Math.log(exp_sum) + max_logit.to_f64
  log_p_target = last_row[target].to_f64 - log_denom
  nll_sum -= log_p_target

  if (k + 1) % progress_every == 0
    elapsed = (Time.instant - t0).total_seconds
    rate = (k + 1) / elapsed
    eta = (total_positions - k - 1) / rate
    STDERR.puts "  #{k + 1}/#{total_positions} (%.1f/s, ETA %.0fs)" % [rate, eta]
  end
end

elapsed = (Time.instant - t0).total_seconds
mean_nll = nll_sum / total_positions
perplexity = Math.exp(mean_nll)
bpc = mean_nll / Math.log(2.0)

puts ""
puts "Positions scored:  #{total_positions}"
puts "Mean per-token NLL: %.6f nats" % mean_nll
puts "Perplexity:         %.4f" % perplexity
puts "Bits per character: %.4f bpc" % bpc
puts "Elapsed:            #{elapsed.round(1)}s (#{(total_positions / elapsed).round(1)} pos/sec)"
