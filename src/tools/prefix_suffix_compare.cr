require "../../lib/microgpt/src/microgpt"
require "option_parser"

# Compare forward (prefix-conditioned) and backward (suffix-conditioned)
# model predictions at held-out positions in a corpus.
#
# Both models predict the same target c at position p, but conditioned on
# opposite sides:
#   forward F(prefix) = P_F(c | tokens[p-seq_len..p-1])
#   backward B(rev_suffix) = P_B(c | reverse of tokens[p+1..p+seq_len])
# where the backward model was trained on the reversed corpus. Its
# "next-token" output corresponds to the previous-in-original-direction
# char, which is c at position p when we feed it the reverse of the
# original-direction suffix.
#
# Reports KL(F||B), KL(B||F), symmetric KL, and JS divergence — averaged
# across positions.

include MicroGPT

forward_path = ""
backward_path = ""
test_file = ""
vocab_file = ""
seq_len = 32
max_positions = 4096
backend = "openblas"

OptionParser.parse do |p|
  p.banner = "Usage: prefix_suffix_compare --forward F.model --backward B.model --file PATH [options]"
  p.on("--forward PATH", "Forward model checkpoint") { |v| forward_path = v }
  p.on("--backward PATH", "Backward model checkpoint (trained on reversed corpus)") { |v| backward_path = v }
  p.on("--file PATH", "Test text file") { |v| test_file = v }
  p.on("--vocab-file PATH", "Build vocab from this file (defaults to --file)") { |v| vocab_file = v }
  p.on("--seq-len N", "Context window length (default 32)") { |v| seq_len = v.to_i }
  p.on("--max-positions N", "Cap evaluated positions (default 4096)") { |v| max_positions = v.to_i }
  p.on("--backend B", "crystal|openblas|cublas (default openblas)") { |v| backend = v }
  p.on("-h", "--help", "") { puts p; exit 0 }
end

abort "missing --forward" if forward_path.empty?
abort "missing --backward" if backward_path.empty?
abort "missing --file" if test_file.empty?

case backend
when "openblas"
  MicroGPT.use_openblas!
when "cublas"
  MicroGPT.use_cublas!
else
  MicroGPT.use_crystal!
end

# Build vocab from --vocab-file (defaults to --file).
vf = vocab_file.empty? ? test_file : vocab_file
vocab_source = File.read(vf)
dataset = CharDataset.new(vocab_source)
v = dataset.vocab_size
STDERR.puts "Vocab: #{v} chars from #{vf}"

test_text = File.read(test_file)
filtered = String.build do |io|
  test_text.each_char { |c| io << c if dataset.char_to_id.has_key?(c) }
end
test_text = filtered
tokens = dataset.encode(test_text)
STDERR.puts "Test corpus: #{tokens.size} tokens"

# Helper: load model from checkpoint with auto-detected architecture.
def load_model(path : String, vocab_size : Int32, seq_len : Int32) : MiniGPT
  d_model = 64
  n_heads = 4
  n_layers = 2
  d_ff = 256
  File.open(path, "rb") do |f|
    magic = f.read_bytes(UInt32, IO::ByteFormat::LittleEndian)
    if magic == 0x4D475054_u32
      d_model  = f.read_bytes(Int32, IO::ByteFormat::LittleEndian)
      n_heads  = f.read_bytes(Int32, IO::ByteFormat::LittleEndian)
      n_layers = f.read_bytes(Int32, IO::ByteFormat::LittleEndian)
      d_ff     = f.read_bytes(Int32, IO::ByteFormat::LittleEndian)
      _vocab   = f.read_bytes(Int32, IO::ByteFormat::LittleEndian)
      _seq     = f.read_bytes(Int32, IO::ByteFormat::LittleEndian)
    end
  end
  config = Config.new
  config.d_model = d_model
  config.n_heads = n_heads
  config.n_layers = n_layers
  config.d_ff = d_ff
  config.seq_len = seq_len
  config.vocab_size = vocab_size
  m = MiniGPT.new(config)
  m.load(path)
  STDERR.puts "Loaded #{path}: d_model=#{d_model} n_heads=#{n_heads} n_layers=#{n_layers}"
  m
end

forward = load_model(forward_path, v, seq_len)
backward = load_model(backward_path, v, seq_len)

# Softmax with log-sum-exp for stability. Returns probability vector.
def softmax(row : Array(Float32)) : Array(Float64)
  max_logit = row.max.to_f64
  exp_vals = row.map { |x| Math.exp(x.to_f64 - max_logit) }
  s = exp_vals.sum
  exp_vals.map { |x| x / s }
end

# KL(p || q) = Σ p_i log(p_i / q_i)
def kl(p : Array(Float64), q : Array(Float64)) : Float64
  acc = 0.0
  p.each_with_index do |pi, i|
    next if pi <= 0.0
    qi = q[i]
    if qi <= 0.0
      qi = 1e-30
    end
    acc += pi * Math.log(pi / qi)
  end
  acc
end

# Walk corpus. For each position p with seq_len chars on each side, compute
# F's prediction (from prefix) and B's prediction (from reversed suffix).
n_total = tokens.size - 2 * seq_len
n_total = max_positions if max_positions > 0 && max_positions < n_total

kl_FB_sum = 0.0
kl_BF_sum = 0.0
js_sum = 0.0
nll_F_sum = 0.0
nll_B_sum = 0.0
agree_top1 = 0
disagree_top1 = 0

t0 = Time.instant
n_total.times do |k|
  pos = seq_len + k
  break if pos + seq_len + 1 > tokens.size

  prefix = tokens[pos - seq_len, seq_len]
  suffix_rev = tokens[pos + 1, seq_len].reverse
  target = tokens[pos]

  logits_F = forward.forward(prefix)
  logits_B = backward.forward(suffix_rev)

  row_F = Array(Float32).new(v)
  row_B = Array(Float32).new(v)
  v.times do |j|
    row_F << logits_F[seq_len - 1, j]
    row_B << logits_B[seq_len - 1, j]
  end
  pF = softmax(row_F)
  pB = softmax(row_B)

  kl_FB_sum += kl(pF, pB)
  kl_BF_sum += kl(pB, pF)
  pm = Array(Float64).new(v) { |i| 0.5 * (pF[i] + pB[i]) }
  js_sum += 0.5 * (kl(pF, pm) + kl(pB, pm))

  nll_F_sum -= Math.log(pF[target] + 1e-30)
  nll_B_sum -= Math.log(pB[target] + 1e-30)

  argmax_F = 0
  argmax_B = 0
  best_F = -1.0
  best_B = -1.0
  v.times do |j|
    if pF[j] > best_F
      best_F = pF[j]; argmax_F = j
    end
    if pB[j] > best_B
      best_B = pB[j]; argmax_B = j
    end
  end
  if argmax_F == argmax_B
    agree_top1 += 1
  else
    disagree_top1 += 1
  end

  if (k + 1) % 500 == 0
    elapsed = (Time.instant - t0).total_seconds
    STDERR.puts "  #{k + 1}/#{n_total} (#{(k + 1).to_f / elapsed.to_f}/s)"
  end
end

elapsed = (Time.instant - t0).total_seconds
n = n_total
puts ""
puts "Compared #{n} positions in #{elapsed.round(1)}s"
puts ""
puts "Per-position divergence (mean across positions):"
puts "  KL(F || B)   = %.4f nats" % (kl_FB_sum / n)
puts "  KL(B || F)   = %.4f nats" % (kl_BF_sum / n)
puts "  Symmetric KL = %.4f nats" % ((kl_FB_sum + kl_BF_sum) / (2 * n))
puts "  JS           = %.4f nats" % (js_sum / n)
puts ""
puts "Per-position NLL of true target:"
puts "  Forward    : %.4f nats (PPL %.3f)" % [nll_F_sum / n, Math.exp(nll_F_sum / n)]
puts "  Backward   : %.4f nats (PPL %.3f)" % [nll_B_sum / n, Math.exp(nll_B_sum / n)]
puts ""
puts "Top-1 agreement:"
puts "  Both predict same top char: #{agree_top1} / #{n} (%.1f%%)" % (100.0 * agree_top1 / n)
puts "  Disagreement              : #{disagree_top1} / #{n} (%.1f%%)" % (100.0 * disagree_top1 / n)
