require "../../lib/microgpt/src/microgpt"
require "option_parser"

# Dual-view consistency trainer.
#
# Trains two AGPT models simultaneously on the same corpus:
#   F sees the prefix [i-seq_len .. i-1] and predicts corpus[i]
#   B sees reverse(corpus[i+1 .. i+seq_len]) and predicts corpus[i]
#
# At each position both models compute CE against the true target, and a
# stop-grad symmetric KL term pulls each toward the other's prediction:
#   loss_F = CE(target, P_F) + β · KL(stop_grad(P_B) || P_F)
#   loss_B = CE(target, P_B) + β · KL(stop_grad(P_F) || P_B)
#
# This is per-corpus-position training (one-hot CE, not trie-distribution
# CE). Adam fires per position in this initial version. Per-partition
# batching is a future optimization.
#
# Usage:
#   bin/agpt_dual_train --init-f F.model --init-b B.model --corpus PATH
#                       --save-f Fo.model --save-b Bo.model
#                       --epochs 6 --kl-beta 0.1 --lr 3e-3 [--shuffle-suffix]

include MicroGPT

init_f_path = ""
init_b_path = ""
corpus_file = ""
save_f_path = ""
save_b_path = ""
epochs = 6
seq_len = 32
beta_max = 0.1
warmup_steps = 0
lr = 3e-3
backend = "openblas"
seed = 42
shuffle_suffix = false
log_every = 5000
max_positions = -1

OptionParser.parse do |p|
  p.banner = "Usage: agpt_dual_train --init-f F --init-b B --corpus PATH --save-f Fo --save-b Bo [options]"
  p.on("--init-f PATH", "Forward model initial weights") { |v| init_f_path = v }
  p.on("--init-b PATH", "Backward model initial weights") { |v| init_b_path = v }
  p.on("--corpus PATH", "Training corpus text file") { |v| corpus_file = v }
  p.on("--save-f PATH", "Save forward model to") { |v| save_f_path = v }
  p.on("--save-b PATH", "Save backward model to") { |v| save_b_path = v }
  p.on("--epochs N", "Training epochs (default 6)") { |v| epochs = v.to_i }
  p.on("--seq-len N", "Context length per side (default 32)") { |v| seq_len = v.to_i }
  p.on("--kl-beta F", "KL coupling weight (default 0.1; 0 = no coupling, β=0 sanity)") { |v| beta_max = v.to_f }
  p.on("--kl-warmup N", "Linear warmup steps for β (default 0; instant on)") { |v| warmup_steps = v.to_i }
  p.on("--lr F", "Learning rate (default 3e-3)") { |v| lr = v.to_f }
  p.on("--backend B", "crystal|openblas|cublas (default openblas)") { |v| backend = v }
  p.on("--seed N", "RNG seed for shuffle (default 42)") { |v| seed = v.to_i }
  p.on("--shuffle-suffix", "Negative control: pair F prefix with B suffix from random position") { shuffle_suffix = true }
  p.on("--log-every N", "Log every N steps (default 5000)") { |v| log_every = v.to_i }
  p.on("--max-positions N", "Cap positions per epoch (default: all)") { |v| max_positions = v.to_i }
  p.on("-h", "--help", "") { puts p; exit 0 }
end

abort "missing --init-f"  if init_f_path.empty?
abort "missing --init-b"  if init_b_path.empty?
abort "missing --corpus"  if corpus_file.empty?
abort "missing --save-f"  if save_f_path.empty?
abort "missing --save-b"  if save_b_path.empty?

case backend
when "openblas"
  MicroGPT.use_openblas!
when "cublas"
  MicroGPT.use_cublas!
else
  MicroGPT.use_crystal!
end

text = File.read(corpus_file)
chars = text.chars.to_set.to_a.sort
char_to_id = {} of Char => Int32
chars.each_with_index { |c, i| char_to_id[c] = i }
v = chars.size

filtered = String.build { |io| text.each_char { |c| io << c if char_to_id.has_key?(c) } }
tokens = filtered.chars.map { |c| char_to_id[c] }
STDERR.puts "Corpus: #{tokens.size} tokens, V=#{v}"

# Auto-detect architecture from checkpoint.
def load_model(path : String, vocab : Int32, seq_len : Int32) : MiniGPT
  d_model = 64; n_heads = 4; n_layers = 2; d_ff = 256
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
  cfg = Config.new
  cfg.d_model = d_model; cfg.n_heads = n_heads; cfg.n_layers = n_layers
  cfg.d_ff = d_ff; cfg.seq_len = seq_len; cfg.vocab_size = vocab
  m = MiniGPT.new(cfg)
  m.load(path)
  STDERR.puts "Loaded #{path}: d_model=#{d_model} n_heads=#{n_heads} n_layers=#{n_layers}"
  m
end

forward = load_model(init_f_path, v, seq_len)
backward = load_model(init_b_path, v, seq_len)

# Per-position dual training step.
#
# Computes:
#   logits_F = forward.forward(prefix)       # [seq_len, V]
#   logits_B = backward.forward(reversed_suffix)
#   d_logits_F[last_pos, t] = (P_F[t] - 1_{t==target}) + β · (P_F[t] - P_B[t])
#   d_logits_B[last_pos, t] = (P_B[t] - 1_{t==target}) + β · (P_B[t] - P_F[t])
#   d_logits_*[other_pos, *] = 0
# Then backprops through both models and fires Adam on each.
#
# Returns (ce_F, ce_B, kl_F_unweighted, kl_B_unweighted).
def dual_step(forward : MiniGPT, backward : MiniGPT,
              prefix : Array(Int32), reversed_suffix : Array(Int32),
              target : Int32, beta : Float64, lr : Float64,
              v : Int32, seq_len : Int32) : Tuple(Float64, Float64, Float64, Float64)
  logits_F = forward.forward(prefix)
  logits_B = backward.forward(reversed_suffix)

  # Softmax on the last row of each (predicting target).
  row_F = Array(Float64).new(v) { |j| logits_F[seq_len - 1, j].to_f64 }
  row_B = Array(Float64).new(v) { |j| logits_B[seq_len - 1, j].to_f64 }

  max_F = row_F.max; max_B = row_B.max
  exp_F = row_F.map { |x| Math.exp(x - max_F) }
  exp_B = row_B.map { |x| Math.exp(x - max_B) }
  s_F = exp_F.sum; s_B = exp_B.sum
  p_F = exp_F.map { |x| x / s_F }
  p_B = exp_B.map { |x| x / s_B }

  ce_F = -Math.log(p_F[target] + 1e-30)
  ce_B = -Math.log(p_B[target] + 1e-30)

  kl_F = 0.0
  kl_B = 0.0
  v.times do |t|
    pf = p_F[t]; pb = p_B[t]
    kl_F += pb * (Math.log(pb + 1e-30) - Math.log(pf + 1e-30)) if pb > 0
    kl_B += pf * (Math.log(pf + 1e-30) - Math.log(pb + 1e-30)) if pf > 0
  end

  # Build d_logits for both models. Only the last row carries gradient.
  d_logits_F = Mat.new(seq_len, v)
  d_logits_B = Mat.new(seq_len, v)
  v.times do |t|
    one_hot = (t == target ? 1.0 : 0.0)
    g_F = (p_F[t] - one_hot) + beta * (p_F[t] - p_B[t])
    g_B = (p_B[t] - one_hot) + beta * (p_B[t] - p_F[t])
    d_logits_F[seq_len - 1, t] = g_F
    d_logits_B[seq_len - 1, t] = g_B
  end

  # Backprop through F: proj -> final_norm -> blocks (reverse) -> embedding.
  d_F = forward.output.proj.backward(d_logits_F)
  d_F = forward.final_norm.backward(d_F)
  forward.blocks.size.times do |i|
    li = forward.blocks.size - 1 - i
    d_F = forward.blocks[li].backward(d_F)
  end
  forward.embedding.backward(d_F)

  # Backprop through B.
  d_B = backward.output.proj.backward(d_logits_B)
  d_B = backward.final_norm.backward(d_B)
  backward.blocks.size.times do |i|
    li = backward.blocks.size - 1 - i
    d_B = backward.blocks[li].backward(d_B)
  end
  backward.embedding.backward(d_B)

  # Per-position Adam fire (could batch per partition; future optimization).
  forward.embedding.update(lr)
  forward.blocks.each &.update(lr)
  forward.final_norm.update(lr)
  forward.output.update(lr)

  backward.embedding.update(lr)
  backward.blocks.each &.update(lr)
  backward.final_norm.update(lr)
  backward.output.update(lr)

  {ce_F, ce_B, kl_F, kl_B}
end

# Training loop: shuffled corpus positions, per-position dual step.
total_positions = tokens.size - 2 * seq_len
total_positions = max_positions if max_positions > 0 && max_positions < total_positions
positions = (seq_len..(seq_len + total_positions - 1)).to_a

step = 0
total_steps = 0
rng = Random.new(seed)

t_total_start = Time.instant
epochs.times do |epoch|
  positions.shuffle!(rng)

  ce_F_sum = 0.0; ce_B_sum = 0.0
  kl_F_sum = 0.0; kl_B_sum = 0.0
  count = 0
  t_epoch = Time.instant

  positions.each do |pos|
    next if pos + seq_len + 1 > tokens.size

    prefix = tokens[pos - seq_len, seq_len]
    if shuffle_suffix
      j = rng.rand(seq_len..(tokens.size - seq_len - 1))
      reversed_suffix = tokens[j + 1, seq_len].reverse
    else
      reversed_suffix = tokens[pos + 1, seq_len].reverse
    end
    target = tokens[pos]

    beta_eff = if warmup_steps > 0
                 beta_max * Math.min(1.0, total_steps.to_f / warmup_steps.to_f)
               else
                 beta_max
               end

    ce_F, ce_B, kl_F, kl_B = dual_step(
      forward, backward, prefix, reversed_suffix, target,
      beta_eff, lr, v, seq_len
    )

    ce_F_sum += ce_F; ce_B_sum += ce_B
    kl_F_sum += kl_F; kl_B_sum += kl_B
    count += 1
    total_steps += 1

    if total_steps % log_every == 0
      elapsed = (Time.instant - t_epoch).total_seconds
      rate = count / elapsed
      STDERR.puts "  e#{epoch + 1} step=#{total_steps} pos=#{count} (#{rate.round(0)}/s) " \
        "ce_F=%.4f ce_B=%.4f kl_F=%.4f kl_B=%.4f β=%.4f" % [
          ce_F_sum / count, ce_B_sum / count,
          kl_F_sum / count, kl_B_sum / count, beta_eff
        ]
    end
  end

  elapsed = (Time.instant - t_epoch).total_seconds
  STDERR.puts "Epoch #{epoch + 1}: ce_F=%.4f ce_B=%.4f kl_F=%.4f kl_B=%.4f (%.1fs, %d positions)" % [
    ce_F_sum / count, ce_B_sum / count, kl_F_sum / count, kl_B_sum / count,
    elapsed, count
  ]
end

t_total = (Time.instant - t_total_start).total_seconds
STDERR.puts "Total: #{epochs} epochs in #{t_total.round(1)}s"

forward.save(save_f_path)
backward.save(save_b_path)
STDERR.puts "Saved #{save_f_path} and #{save_b_path}"
