require "option_parser"
require "../agpt"

include MicroGPT::AGPT

MAGIC_RECUR_TANH = 0x52474341_u32 # 'ACGR'
MAGIC_RECUR_RMS_POST = 0x54474341_u32 # 'ACGT'
MAGIC_RECUR_RMS_PRE  = 0x50474341_u32 # 'ACGP'
MAGIC_RECUR_PHASE          = 0x51474341_u32 # 'ACGQ'
MAGIC_RECUR_PHASE_RMS_POST = 0x55474341_u32 # 'ACGU'
MAGIC_RECUR_PHASE_RMS_PRE  = 0x56474341_u32 # 'ACGV'
EPS_NORM = 1e-6_f64

enum Variant
  TanhElman
  TanhRMSPost
  TanhRMSPre
  TanhPhase
  TanhPhaseRMSPost
  TanhPhaseRMSPre
end

class RecurEvalParams
  getter vocab_size : Int32
  getter d_model : Int32
  getter phase_window : Int32
  property variant : Variant = Variant::TanhElman
  getter emb : Array(Float64)
  getter phase_emb : Array(Float64)
  getter w_h : Array(Float64)
  getter w_x : Array(Float64)
  getter b : Array(Float64)
  getter g : Array(Float64)
  getter w_o : Array(Float64)
  getter c_o : Array(Float64)

  def initialize(@vocab_size : Int32, @d_model : Int32, @phase_window : Int32 = 0)
    v = @vocab_size
    d = @d_model
    @emb = Array(Float64).new(v * d, 0.0)
    @phase_emb = Array(Float64).new(@phase_window * d, 0.0)
    @w_h = Array(Float64).new(d * d, 0.0)
    @w_x = Array(Float64).new(d * d, 0.0)
    @b = Array(Float64).new(d, 0.0)
    @g = Array(Float64).new(d, 1.0)
    @w_o = Array(Float64).new(v * d, 0.0)
    @c_o = Array(Float64).new(v, 0.0)
  end

  def each_array(&block : Array(Float64) ->)
    yield @emb
    yield @phase_emb if @phase_window > 0
    yield @w_h
    yield @w_x
    yield @b
    yield @g if rms_variant?
    yield @w_o
    yield @c_o
  end

  def rms_variant? : Bool
    @variant.tanh_rms_post? || @variant.tanh_rms_pre? || @variant.tanh_phase_rms_post? || @variant.tanh_phase_rms_pre?
  end
end

def read_f64_array(io : IO, a : Array(Float64))
  a.size.times { |i| a[i] = io.read_bytes(Float64, IO::ByteFormat::LittleEndian) }
end

def load_checkpoint(path : String) : {Variant, RecurEvalParams}
  File.open(path, "rb") do |io|
    magic = io.read_bytes(UInt32, IO::ByteFormat::LittleEndian)
    variant = case magic
              when MAGIC_RECUR_TANH then Variant::TanhElman
              when MAGIC_RECUR_RMS_POST then Variant::TanhRMSPost
              when MAGIC_RECUR_RMS_PRE then Variant::TanhRMSPre
              when MAGIC_RECUR_PHASE then Variant::TanhPhase
              when MAGIC_RECUR_PHASE_RMS_POST then Variant::TanhPhaseRMSPost
              when MAGIC_RECUR_PHASE_RMS_PRE then Variant::TanhPhaseRMSPre
              else raise "unknown recurrent checkpoint magic 0x#{magic.to_s(16)} in #{path}"
              end
    version = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
    raise "unsupported recurrent checkpoint version #{version}" unless version == 1
    vocab_size = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
    d_model = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
    _epoch = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
    _adam_step = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
    _seed = io.read_bytes(UInt64, IO::ByteFormat::LittleEndian)
    phase_checkpoint = variant.tanh_phase? || variant.tanh_phase_rms_post? || variant.tanh_phase_rms_pre?
    phase_window = phase_checkpoint ? io.read_bytes(Int32, IO::ByteFormat::LittleEndian) : 0
    params = RecurEvalParams.new(vocab_size, d_model, phase_window)
    params.variant = variant
    params.each_array { |a| read_f64_array(io, a) }
    {variant, params}
  end
end

def step!(variant : Variant, params : RecurEvalParams, h : Array(Float64), tok : Int32, corpus_pos : Int32)
  d = params.d_model
  emb_base = tok * d
  z = Array(Float64).new(d, 0.0)
  phase_base = -1
  if params.phase_window > 0
    phase = corpus_pos % params.phase_window
    phase += params.phase_window if phase < 0
    phase_base = phase * d
  end

  d.times do |j|
    zj = params.b[j]
    zj += params.phase_emb[phase_base + j] if phase_base >= 0
    row = j * d
    d.times do |k|
      zj += params.w_h[row + k] * h[k]
      zj += params.w_x[row + k] * params.emb[emb_base + k]
    end
    z[j] = zj
  end

  if variant.tanh_rms_pre? || variant.tanh_phase_rms_pre?
    sumsq = 0.0
    d.times { |j| sumsq += z[j] * z[j] }
    inv = 1.0 / Math.sqrt(sumsq / d.to_f + EPS_NORM)
    d.times { |j| h[j] = Math.tanh(z[j] * params.g[j] * inv) }
  elsif variant.tanh_rms_post? || variant.tanh_phase_rms_post?
    d.times { |j| z[j] = Math.tanh(z[j]) }
    sumsq = 0.0
    d.times { |j| sumsq += z[j] * z[j] }
    inv = 1.0 / Math.sqrt(sumsq / d.to_f + EPS_NORM)
    d.times { |j| h[j] = z[j] * params.g[j] * inv }
  else
    d.times { |j| h[j] = Math.tanh(z[j]) }
  end
end

def neg_log_prob(params : RecurEvalParams, h : Array(Float64), target : Int32) : Float64
  d = params.d_model
  v = params.vocab_size
  logits = Array(Float64).new(v, 0.0)
  max_logit = -Float64::INFINITY

  v.times do |tok|
    z = params.c_o[tok]
    base = tok * d
    d.times { |j| z += params.w_o[base + j] * h[j] }
    logits[tok] = z
    max_logit = z if z > max_logit
  end

  sum_exp = 0.0
  v.times { |tok| sum_exp += Math.exp(logits[tok] - max_logit) }
  max_logit + Math.log(sum_exp) - logits[target]
end

checkpoint_path = ""
corpus_path = ""
vocab_path = ""
seq_len = 16
max_positions = 8192
stride_step = 1
phase = 0
target_offset = 0
quiet = false

OptionParser.parse do |p|
  p.banner = "Usage: bin/agpt_recur_perplexity --checkpoint PATH --file HELDOUT --vocab-file PATH [options]"
  p.on("--checkpoint PATH", "Trained .recur checkpoint") { |v| checkpoint_path = v }
  p.on("--file PATH", "Held-out text file") { |v| corpus_path = v }
  p.on("--vocab-file PATH", "Vocab source (defaults to --file)") { |v| vocab_path = v }
  p.on("--seq-len N", "Context length per position (default 16)") { |v| seq_len = v.to_i }
  p.on("--stride N", "Evaluate strided context positions p-N*seq_len...p-N,p (default 1)") { |v| stride_step = v.to_i }
  p.on("--phase N", "Only score target positions where p mod stride == phase (default 0)") { |v| phase = v.to_i }
  p.on("--target-offset N", "Predict endpoint+N instead of endpoint+stride for strided contexts (default 0)") { |v| target_offset = v.to_i }
  p.on("--max-positions N", "Limit positions scored (default 8192; 0 = all)") { |v| max_positions = v.to_i }
  p.on("--quiet", "Suppress progress output") { quiet = true }
  p.on("-h", "--help", "Help") { puts p; exit 0 }
end

abort "missing --checkpoint" if checkpoint_path.empty?
abort "missing --file" if corpus_path.empty?
vocab_path = corpus_path if vocab_path.empty?
abort "--seq-len must be > 0" if seq_len <= 0
abort "--stride must be > 0" if stride_step <= 0
abort "--phase must satisfy 0 <= phase < stride" if phase < 0 || phase >= stride_step
abort "--target-offset must be >= 0" if target_offset < 0

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
STDERR.puts "WARN: vocab-file derived #{chars.size} chars but checkpoint vocab_size=#{v}" if chars.size != v

tokens = [] of Int32
File.read(corpus_path).each_char do |c|
  tokens << (char_to_id[c]? || 0)
end

effective_offset = target_offset > 0 ? target_offset : stride_step
start_pos = effective_offset + ((seq_len - 1) * stride_step)
end_pos = tokens.size - 1
n_avail = ((start_pos...end_pos).count do |p|
  endpoint = p - effective_offset
  (endpoint % stride_step) == phase
end)
abort "no positions available for seq-len=#{seq_len}, stride=#{stride_step}, phase=#{phase}, target-offset=#{target_offset}" if n_avail <= 0
n_score = (max_positions > 0 && max_positions < n_avail) ? max_positions : n_avail
stride = (n_avail.to_f64 / n_score.to_f64).clamp(1.0, Float64::MAX)

STDERR.puts "Vocab: #{v}, seq-len: #{seq_len}, eval-stride: #{stride_step}, phase: #{phase}, target-offset: #{target_offset}, scoring #{n_score} positions (sample stride #{stride.round(2)})" unless quiet

total_nll = 0.0
n_scored = 0
t0 = Time.instant
h = Array(Float64).new(d, 0.0)

n_score.times do |i|
  ordinal = (i.to_f64 * stride).to_i
  p = start_pos
  seen = 0
  while p < end_pos
    endpoint = p - effective_offset
    if (endpoint % stride_step) == phase
      break if seen >= ordinal
      seen += 1
    end
    p += 1
  end
  break if p >= end_pos
  target = tokens[p]

  h.fill(0.0)
  endpoint = p - effective_offset
  q = endpoint - ((seq_len - 1) * stride_step)
  seq_len.times do
    step!(variant, params, h, tokens[q], q)
    q += stride_step
  end

  total_nll += neg_log_prob(params, h, target)
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
