require "option_parser"
require "../agpt"

# Held-out PPL evaluator for recurrent-AGPT checkpoints.
#
# Loads a `.recur` checkpoint (either ACGR = tanh-Elman or ACGL = linear)
# and scores held-out text per position: starting from a zero state, walk
# the last `--seq-len` tokens through f_θ, project through the readout,
# softmax, and accumulate -log P(true_next_token). Report mean NLL, PPL,
# and BPC.
#
# This is the canonical "real number" for recur trainers — the trainers'
# own in-loop PPL is count-weighted CE over the training trie, which is
# fine as a training signal but not directly comparable to held-out PPL.
#
# Detects the f_θ variant from the magic header; identical file layouts
# (vocab_size, d_model, emb, w_h, w_x, b, w_o, c_o) for both variants.
#
# Usage:
#   bin/agpt_recur_perplexity \
#     --checkpoint /tmp/linear_500ep.recur \
#     --file data/.splits/<HASH>/heldout_corpus.txt \
#     --vocab-file data/input.txt \
#     --seq-len 4 \
#     --max-positions 0

include MicroGPT::AGPT

MAGIC_RECUR_TANH    = 0x52474341_u32 # 'ACGR'  -- Codex's agpt_train_recur
MAGIC_RECUR_LIN     = 0x4C474341_u32 # 'ACGL'  -- agpt_train_recur_linear
MAGIC_RECUR_LIN_RMS = 0x4E474341_u32 # 'ACGN'  -- agpt_train_recur_linear_rms
EPS_NORM            = 1e-6_f64

enum Variant
  TanhElman
  Linear
  LinearRMS
end

class RecurParams
  getter vocab_size : Int32
  getter d_model : Int32
  property variant : Variant = Variant::Linear
  getter emb : Array(Float64)
  getter w_h : Array(Float64)
  getter w_x : Array(Float64)
  getter b : Array(Float64)
  getter g : Array(Float64) # RMSNorm gain; unused for tanh / linear
  getter w_o : Array(Float64)
  getter c_o : Array(Float64)

  def initialize(@vocab_size : Int32, @d_model : Int32)
    v = @vocab_size
    d = @d_model
    @emb = Array(Float64).new(v * d, 0.0)
    @w_h = Array(Float64).new(d * d, 0.0)
    @w_x = Array(Float64).new(d * d, 0.0)
    @b   = Array(Float64).new(d, 0.0)
    @g   = Array(Float64).new(d, 1.0)
    @w_o = Array(Float64).new(v * d, 0.0)
    @c_o = Array(Float64).new(v, 0.0)
  end

  # Iterate in the same order the trainer wrote them.
  def each_array(&block : Array(Float64) ->)
    yield @emb
    yield @w_h
    yield @w_x
    yield @b
    yield @g if @variant == Variant::LinearRMS
    yield @w_o
    yield @c_o
  end
end

def read_f64_array(io : IO, a : Array(Float64))
  a.size.times do |i|
    a[i] = io.read_bytes(Float64, IO::ByteFormat::LittleEndian)
  end
end

def load_checkpoint(path : String) : {Variant, RecurParams}
  File.open(path, "rb") do |io|
    magic = io.read_bytes(UInt32, IO::ByteFormat::LittleEndian)
    variant = case magic
              when MAGIC_RECUR_TANH    then Variant::TanhElman
              when MAGIC_RECUR_LIN     then Variant::Linear
              when MAGIC_RECUR_LIN_RMS then Variant::LinearRMS
              else raise "unknown recur checkpoint magic 0x#{magic.to_s(16)} in #{path} (expected ACGR=0x52474341, ACGL=0x4C474341, or ACGN=0x4E474341)"
              end
    version = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
    raise "unsupported recur checkpoint version #{version}" unless version == 1
    vocab_size = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
    d_model = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
    _epoch = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
    _adam_step = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
    _seed = io.read_bytes(UInt64, IO::ByteFormat::LittleEndian)
    params = RecurParams.new(vocab_size, d_model)
    params.variant = variant
    params.each_array { |a| read_f64_array(io, a) }
    # We don't need Adam state for eval; ignore the rest.
    {variant, params}
  end
end

# One step of f_θ. Mutates `h` in place.
def step!(variant : Variant, params : RecurParams, h : Array(Float64), tok : Int32)
  d = params.d_model
  # Use a scratch z buffer; h is read fully before being overwritten.
  z = Array(Float64).new(d, 0.0)
  emb_base = tok * d
  d.times do |j|
    zj = params.b[j]
    wh_base = j * d
    wx_base = j * d
    d.times do |k|
      zj += params.w_h[wh_base + k] * h[k]
      zj += params.w_x[wx_base + k] * params.emb[emb_base + k]
    end
    z[j] = zj
  end
  case variant
  when Variant::TanhElman
    d.times { |j| h[j] = Math.tanh(z[j]) }
  when Variant::LinearRMS
    sumsq = 0.0
    d.times { |j| sumsq += z[j] * z[j] }
    sigma = Math.sqrt(sumsq / d.to_f + EPS_NORM)
    d.times { |j| h[j] = z[j] * params.g[j] / sigma }
  else
    # Linear
    d.times { |j| h[j] = z[j] }
  end
end

def neg_log_prob(params : RecurParams, h : Array(Float64), target : Int32) : Float64
  d = params.d_model
  v = params.vocab_size
  # logits = c_o + w_o · h  (w_o stored as v×d, row-major: w_o[tok*d + j])
  max_logit = -Float64::INFINITY
  logits = Array(Float64).new(v, 0.0)
  v.times do |tok|
    z = params.c_o[tok]
    wo_base = tok * d
    d.times { |j| z += params.w_o[wo_base + j] * h[j] }
    logits[tok] = z
    max_logit = z if z > max_logit
  end
  sum_exp = 0.0
  v.times do |tok|
    sum_exp += Math.exp(logits[tok] - max_logit)
  end
  log_sum = max_logit + Math.log(sum_exp)
  log_sum - logits[target]  # = -log softmax[target]
end

checkpoint_path = ""
corpus_path = ""
vocab_path = ""
seq_len = 16
max_positions = 8192
quiet = false

OptionParser.parse do |p|
  p.banner = "Usage: agpt_recur_perplexity --checkpoint PATH --file HELDOUT --vocab-file PATH [options]"
  p.on("--checkpoint PATH", "Trained .recur checkpoint (ACGR/ACGL)") { |v| checkpoint_path = v }
  p.on("--file PATH", "Held-out text file") { |v| corpus_path = v }
  p.on("--vocab-file PATH", "Vocab source (defaults to --file)") { |v| vocab_path = v }
  p.on("--seq-len N", "Context length per position (default 16)") { |v| seq_len = v.to_i }
  p.on("--max-positions N", "Limit positions scored (default 8192; 0 = all)") { |v| max_positions = v.to_i }
  p.on("--quiet", "Suppress progress output") { quiet = true }
  p.on("-h", "--help", "") { puts p; exit 0 }
end

abort "missing --checkpoint" if checkpoint_path.empty?
abort "missing --file" if corpus_path.empty?
vocab_path = corpus_path if vocab_path.empty?
abort "--seq-len must be > 0" if seq_len <= 0

variant, params = load_checkpoint(checkpoint_path)
v = params.vocab_size
d = params.d_model

unless quiet
  STDERR.puts "Checkpoint: #{checkpoint_path} (#{variant} d_model=#{d} vocab=#{v})"
  STDERR.puts "Corpus: #{corpus_path}"
end

chars = File.read(vocab_path).chars.to_set.to_a.sort
char_to_id = {} of Char => Int32
chars.each_with_index { |c, i| char_to_id[c] = i }
if chars.size != v
  STDERR.puts "WARN: vocab-file derived #{chars.size} chars but checkpoint vocab_size=#{v}"
end

text = File.read(corpus_path)
tokens = Array(Int32).new(text.size)
text.each_char do |c|
  tid = char_to_id[c]?
  tokens << (tid ? tid : 0)
end

start_pos = seq_len
end_pos = tokens.size - 1
n_avail = end_pos - start_pos
n_score = (max_positions > 0 && max_positions < n_avail) ? max_positions : n_avail
stride = (n_avail.to_f64 / n_score.to_f64).clamp(1.0, Float64::MAX)

STDERR.puts "Vocab: #{v}, seq-len: #{seq_len}, scoring #{n_score} positions (stride #{stride.round(2)})" unless quiet

total_nll = 0.0
n_scored = 0
t0 = Time.instant

h_buf = Array(Float64).new(d, 0.0)

n_score.times do |i|
  p = start_pos + (i.to_f64 * stride).to_i
  break if p >= end_pos
  target = tokens[p]

  # Reset h to zero (root state) and walk the last seq_len tokens.
  d.times { |j| h_buf[j] = 0.0 }
  start_ctx = Math.max(0, p - seq_len)
  (start_ctx...p).each do |q|
    step!(variant, params, h_buf, tokens[q])
  end

  nll = neg_log_prob(params, h_buf, target)
  total_nll += nll
  n_scored += 1
end

elapsed = (Time.instant - t0).total_seconds
mean_nll = total_nll / n_scored
ppl = Math.exp(mean_nll)
bpc = mean_nll / Math.log(2.0)

puts "Variant:            #{variant}"
puts "Positions scored:   #{n_scored}"
puts "Mean per-token NLL: #{mean_nll.round(6)} nats"
puts "Perplexity:         #{ppl.round(4)}"
puts "Bits per character: #{bpc.round(4)} bpc"
puts "Elapsed:            #{elapsed.round(2)}s (#{(n_scored / elapsed).round(0)} pos/sec)"
