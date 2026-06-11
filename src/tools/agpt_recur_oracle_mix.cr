require "option_parser"
require "../agpt"

include MicroGPT::AGPT

MAGIC_RECUR_TANH = 0x52474341_u32
MAGIC_RECUR_RMS_POST = 0x54474341_u32
MAGIC_RECUR_RMS_PRE  = 0x50474341_u32
MAGIC_RECUR_PHASE          = 0x51474341_u32
MAGIC_RECUR_PHASE_RMS_POST = 0x55474341_u32
MAGIC_RECUR_PHASE_RMS_PRE  = 0x56474341_u32
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

def load_checkpoint(path : String) : RecurEvalParams
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
    params
  end
end

def step!(params : RecurEvalParams, h : Array(Float64), tok : Int32, corpus_pos : Int32)
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

  if params.variant.tanh_rms_pre? || params.variant.tanh_phase_rms_pre?
    sumsq = 0.0
    d.times { |j| sumsq += z[j] * z[j] }
    inv = 1.0 / Math.sqrt(sumsq / d.to_f + EPS_NORM)
    d.times { |j| h[j] = Math.tanh(z[j] * params.g[j] * inv) }
  elsif params.variant.tanh_rms_post? || params.variant.tanh_phase_rms_post?
    d.times { |j| z[j] = Math.tanh(z[j]) }
    sumsq = 0.0
    d.times { |j| sumsq += z[j] * z[j] }
    inv = 1.0 / Math.sqrt(sumsq / d.to_f + EPS_NORM)
    d.times { |j| h[j] = z[j] * params.g[j] * inv }
  else
    d.times { |j| h[j] = Math.tanh(z[j]) }
  end
end

def probs_for(params : RecurEvalParams, tokens : Array(Int32), positions : Array(Int32), probs : Array(Float64), h : Array(Float64))
  d = params.d_model
  v = params.vocab_size
  h.fill(0.0)
  positions.each { |pos| step!(params, h, tokens[pos], pos) }

  max_logit = -Float64::INFINITY
  v.times do |tok|
    z = params.c_o[tok]
    base = tok * d
    d.times { |j| z += params.w_o[base + j] * h[j] }
    probs[tok] = z
    max_logit = z if z > max_logit
  end

  sum_exp = 0.0
  v.times do |tok|
    p = Math.exp(probs[tok] - max_logit)
    probs[tok] = p
    sum_exp += p
  end
  v.times { |tok| probs[tok] /= sum_exp }
end

def nll_from_prob(p : Float64) : Float64
  -Math.log(p > 1e-300 ? p : 1e-300)
end

s1_path = ""
s2_p0_path = ""
s2_p1_path = ""
s2_next_p0_path = ""
s2_next_p1_path = ""
corpus_path = ""
vocab_path = ""
seq_len = 8
max_positions = 8192
grid_step = 0.1

OptionParser.parse do |p|
  p.banner = "Usage: bin/agpt_recur_oracle_mix --s1 CKPT --s2-p0 CKPT --s2-p1 CKPT --s2-next-p0 CKPT --s2-next-p1 CKPT --file HELDOUT --vocab-file PATH [options]"
  p.on("--s1 PATH", "Stride-1 adjacent recurrent checkpoint") { |v| s1_path = v }
  p.on("--s2-p0 PATH", "Stride-2 phase-0 same-phase checkpoint (predicts x[t+2])") { |v| s2_p0_path = v }
  p.on("--s2-p1 PATH", "Stride-2 phase-1 same-phase checkpoint (predicts x[t+2])") { |v| s2_p1_path = v }
  p.on("--s2-next-p0 PATH", "Stride-2 phase-0 target-offset=1 checkpoint (predicts x[t+1])") { |v| s2_next_p0_path = v }
  p.on("--s2-next-p1 PATH", "Stride-2 phase-1 target-offset=1 checkpoint (predicts x[t+1])") { |v| s2_next_p1_path = v }
  p.on("--file PATH", "Held-out text file") { |v| corpus_path = v }
  p.on("--vocab-file PATH", "Vocab source") { |v| vocab_path = v }
  p.on("--seq-len N", "Tokens per oracle context (default 8)") { |v| seq_len = v.to_i }
  p.on("--max-positions N", "Limit positions scored (default 8192; 0 = all)") { |v| max_positions = v.to_i }
  p.on("--grid-step F", "Mixture grid step (default 0.1)") { |v| grid_step = v.to_f }
  p.on("-h", "--help", "Help") { puts p; exit 0 }
end

abort "missing --s1" if s1_path.empty?
abort "missing --s2-p0" if s2_p0_path.empty?
abort "missing --s2-p1" if s2_p1_path.empty?
abort "missing --s2-next-p0" if s2_next_p0_path.empty?
abort "missing --s2-next-p1" if s2_next_p1_path.empty?
abort "missing --file" if corpus_path.empty?
abort "missing --vocab-file" if vocab_path.empty?
abort "--seq-len must be > 0" if seq_len <= 0
abort "--grid-step must be > 0 and <= 1" if grid_step <= 0.0 || grid_step > 1.0

s1 = load_checkpoint(s1_path)
s2_same = {load_checkpoint(s2_p0_path), load_checkpoint(s2_p1_path)}
s2_next = {load_checkpoint(s2_next_p0_path), load_checkpoint(s2_next_p1_path)}
v = s1.vocab_size
raise "vocab mismatch" unless s2_same[0].vocab_size == v && s2_same[1].vocab_size == v && s2_next[0].vocab_size == v && s2_next[1].vocab_size == v

chars = File.read(vocab_path).chars.to_set.to_a.sort
char_to_id = {} of Char => Int32
chars.each_with_index { |c, i| char_to_id[c] = i }
STDERR.puts "WARN: vocab-file derived #{chars.size} chars but checkpoint vocab_size=#{v}" if chars.size != v

tokens = [] of Int32
File.read(corpus_path).each_char { |c| tokens << (char_to_id[c]? || 0) }

start_pos = seq_len * 2
end_pos = tokens.size - 1
positions = [] of Int32
p = start_pos
while p < end_pos
  positions << p
  p += 1
end
if max_positions > 0 && max_positions < positions.size
  sample_stride = positions.size.to_f64 / max_positions.to_f64
  positions = Array.new(max_positions) { |i| positions[(i.to_f64 * sample_stride).to_i] }
end

probs1 = Array(Float64).new(v, 0.0)
probs_same = Array(Float64).new(v, 0.0)
probs_next = Array(Float64).new(v, 0.0)
h1 = Array(Float64).new(s1.d_model, 0.0)
h2_same = Array(Float64).new(s2_same[0].d_model, 0.0)
h2_next = Array(Float64).new(s2_next[0].d_model, 0.0)

base_s1 = 0.0
base_same = 0.0
base_next = 0.0
mix_same = [] of {Float64, Float64, Float64}
mix_both = [] of {Float64, Float64, Float64, Float64}

steps = (1.0 / grid_step).round.to_i
(0..steps).each do |i|
  w_same = i * grid_step
  w1 = 1.0 - w_same
  mix_same << {w1, w_same, 0.0}
end
(0..steps).each do |i|
  (0..steps - i).each do |j|
    w_same = i * grid_step
    w_other = j * grid_step
    w1 = 1.0 - w_same - w_other
    next if w1 < -1e-9
    mix_both << {w1, w_same, w_other, 0.0}
  end
end

t0 = Time.instant
positions.each do |target_pos|
  target = tokens[target_pos]
  s1_positions = Array.new(seq_len) { |i| target_pos - seq_len + i }
  same_phase = target_pos % 2
  prev_phase = 1 - same_phase
  same_positions = Array.new(seq_len) { |i| target_pos - (seq_len * 2) + (i * 2) }
  prev_phase_positions = Array.new(seq_len) { |i| target_pos - (seq_len * 2) + 1 + (i * 2) }

  probs_for(s1, tokens, s1_positions, probs1, h1)
  probs_for(s2_same[same_phase], tokens, same_positions, probs_same, h2_same)
  probs_for(s2_next[prev_phase], tokens, prev_phase_positions, probs_next, h2_next)

  base_s1 += nll_from_prob(probs1[target])
  base_same += nll_from_prob(probs_same[target])
  base_next += nll_from_prob(probs_next[target])

  mix_same.each_with_index do |(w1, ws, loss), idx|
    p_mix = w1 * probs1[target] + ws * probs_same[target]
    mix_same[idx] = {w1, ws, loss + nll_from_prob(p_mix)}
  end
  mix_both.each_with_index do |(w1, ws, wo, loss), idx|
    p_mix = w1 * probs1[target] + ws * probs_same[target] + wo * probs_next[target]
    mix_both[idx] = {w1, ws, wo, loss + nll_from_prob(p_mix)}
  end
end

n = positions.size.to_f64
best_same = mix_same.min_by { |(_, _, loss)| loss }
best_both = mix_both.min_by { |(_, _, _, loss)| loss }
elapsed = (Time.instant - t0).total_seconds

puts "Positions scored: #{positions.size}"
puts "Baselines:"
printf "  s1 adjacent       nll %.6f  ppl %.4f\n", base_s1 / n, Math.exp(base_s1 / n)
printf "  s2 same-phase     nll %.6f  ppl %.4f\n", base_same / n, Math.exp(base_same / n)
printf "  s2 target-next    nll %.6f  ppl %.4f\n", base_next / n, Math.exp(base_next / n)
puts "Best mixtures:"
printf "  s1+same  w1 %.3f wsame %.3f        nll %.6f  ppl %.4f\n",
  best_same[0], best_same[1], best_same[2] / n, Math.exp(best_same[2] / n)
printf "  s1+both  w1 %.3f wsame %.3f wnext %.3f  nll %.6f  ppl %.4f\n",
  best_both[0], best_both[1], best_both[2], best_both[3] / n, Math.exp(best_both[3] / n)
puts "Elapsed: #{elapsed.round(2)}s"
