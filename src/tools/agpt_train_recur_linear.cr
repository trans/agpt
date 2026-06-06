require "option_parser"
require "../agpt"

# Linear-recurrence AGPT trainer. f_θ = W_h · h_p + W_x · emb(x) + b.
# No per-step activation. Tests how much of the framework's leverage holds
# once the per-step nonlinearity is removed; the softmax readout supplies
# the only nonlinearity in the model.
#
# Forked surgically from `agpt_train_recur.cr` (tanh-Elman). Identical
# parameter layout (emb, w_h, w_x, b, w_o, c_o), Adam state, save format
# semantics, and CLI — the only changes are:
#   - forward_edge: drop `Math.tanh(z)`; assign `z` directly
#   - backward: tanh-Jacobian factor `(1 − h²)` becomes 1
#   - magic word distinguishes the checkpoint family
# This keeps the head-to-head comparison with tanh-Elman clean: any PPL
# delta is attributable to the missing activation, not to differences in
# parameter shape, optimizer cadence, or counting.

include MicroGPT::AGPT

MAGIC_RECUR_LIN = 0x4C474341_u32 # 'ACGL'  (Linear cousin of 'ACGR')
VERSION_RECUR   =          1_i32

record RecurRecord,
  id : Int32,
  parent_id : Int32,
  endpoint_depth : Int32,
  edge_tokens : Array(Int32),
  counts : Array({Int32, Int32})

class RecurParams
  getter vocab_size : Int32
  getter d_model : Int32
  getter emb : Array(Float64)
  getter w_h : Array(Float64)
  getter w_x : Array(Float64)
  getter b : Array(Float64)
  getter w_o : Array(Float64)
  getter c_o : Array(Float64)

  def initialize(@vocab_size : Int32, @d_model : Int32)
    v = @vocab_size
    d = @d_model
    @emb = Array(Float64).new(v * d, 0.0)
    @w_h = Array(Float64).new(d * d, 0.0)
    @w_x = Array(Float64).new(d * d, 0.0)
    @b = Array(Float64).new(d, 0.0)
    @w_o = Array(Float64).new(v * d, 0.0)
    @c_o = Array(Float64).new(v, 0.0)
  end

  def each_array(&block : Array(Float64) ->)
    yield @emb
    yield @w_h
    yield @w_x
    yield @b
    yield @w_o
    yield @c_o
  end

  def total_floats : Int32
    @emb.size + @w_h.size + @w_x.size + @b.size + @w_o.size + @c_o.size
  end

  def fill_random!(seed : UInt64)
    rng = Random.new(seed)
    scale_emb = 0.02
    scale_rec = 1.0 / Math.sqrt(@d_model.to_f)
    fill_normal!(@emb, rng, scale_emb)
    fill_normal!(@w_h, rng, scale_rec)
    fill_normal!(@w_x, rng, scale_rec)
    fill_normal!(@w_o, rng, scale_rec)
  end

  private def fill_normal!(a : Array(Float64), rng : Random, scale : Float64)
    i = 0
    while i < a.size
      u1 = rng.rand
      u1 = 1e-12 if u1 <= 0.0
      u2 = rng.rand
      r = Math.sqrt(-2.0 * Math.log(u1))
      theta = 2.0 * Math::PI * u2
      a[i] = scale * r * Math.cos(theta)
      if i + 1 < a.size
        a[i + 1] = scale * r * Math.sin(theta)
      end
      i += 2
    end
  end
end

class AdamState
  getter m : RecurParams
  getter v : RecurParams
  property step : Int32

  def initialize(vocab_size : Int32, d_model : Int32)
    @m = RecurParams.new(vocab_size, d_model)
    @v = RecurParams.new(vocab_size, d_model)
    @step = 0
  end
end

def ensure_parent_dir(path : String)
  parent = File.dirname(path)
  return if parent == "." || parent.empty?
  Dir.mkdir_p(parent)
end

def write_f64_array(io : IO, a : Array(Float64))
  a.each { |x| io.write_bytes(x, IO::ByteFormat::LittleEndian) }
end

def read_f64_array(io : IO, a : Array(Float64))
  a.size.times do |i|
    a[i] = io.read_bytes(Float64, IO::ByteFormat::LittleEndian)
  end
end

def save_checkpoint(path : String, params : RecurParams, adam : AdamState, epoch : Int32, seed : UInt64)
  ensure_parent_dir(path)
  File.open(path, "wb") do |io|
    io.write_bytes(MAGIC_RECUR_LIN, IO::ByteFormat::LittleEndian)
    io.write_bytes(VERSION_RECUR, IO::ByteFormat::LittleEndian)
    io.write_bytes(params.vocab_size, IO::ByteFormat::LittleEndian)
    io.write_bytes(params.d_model, IO::ByteFormat::LittleEndian)
    io.write_bytes(epoch, IO::ByteFormat::LittleEndian)
    io.write_bytes(adam.step, IO::ByteFormat::LittleEndian)
    io.write_bytes(seed, IO::ByteFormat::LittleEndian)
    params.each_array { |a| write_f64_array(io, a) }
    adam.m.each_array { |a| write_f64_array(io, a) }
    adam.v.each_array { |a| write_f64_array(io, a) }
  end
end

def load_checkpoint(path : String, params : RecurParams, adam : AdamState) : {Int32, UInt64}
  File.open(path, "rb") do |io|
    magic = io.read_bytes(UInt32, IO::ByteFormat::LittleEndian)
    raise "bad linear-recurrent checkpoint magic in #{path}" unless magic == MAGIC_RECUR_LIN
    version = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
    raise "unsupported linear-recurrent checkpoint version #{version}" unless version == VERSION_RECUR
    vocab_size = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
    d_model = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
    raise "checkpoint vocab mismatch: #{vocab_size} != #{params.vocab_size}" unless vocab_size == params.vocab_size
    raise "checkpoint d_model mismatch: #{d_model} != #{params.d_model}" unless d_model == params.d_model
    epoch = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
    adam.step = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
    seed = io.read_bytes(UInt64, IO::ByteFormat::LittleEndian)
    params.each_array { |a| read_f64_array(io, a) }
    adam.m.each_array { |a| read_f64_array(io, a) }
    adam.v.each_array { |a| read_f64_array(io, a) }
    {epoch, seed}
  end
end

def epoch_checkpoint_path(save_path : String, epoch : Int32) : String
  ext = File.extname(save_path)
  base = ext.empty? ? save_path : save_path[0, save_path.size - ext.size]
  suffix = ".epoch_#{epoch.to_s.rjust(6, '0')}"
  ext.empty? ? "#{base}#{suffix}.recur" : "#{base}#{suffix}#{ext}"
end

def load_records(reader : RadixTrieReader, max_nodes : Int32) : Array(RecurRecord)
  records = [] of RecurRecord
  reader.each do |r|
    next if r.id == 0
    records << RecurRecord.new(r.id, r.parent_id, r.endpoint_depth, r.edge_tokens, r.counts)
    break if max_nodes > 0 && records.size >= max_nodes
  end
  records
end

def estimate_loss_events(records : Array(RecurRecord)) : Int64
  total = 0_i64
  records.each do |r|
    r.counts.each { |pair| total += pair[1].to_i64 }
  end
  total
end

# Linear recurrence: h_{p·x} = W_h h_p + W_x emb(x) + b. No activation.
def forward_edge(params : RecurParams, parent_state : Array(Float64), edge : Array(Int32), states : Array(Float64))
  d = params.d_model
  states.fill(0.0)
  d.times { |j| states[j] = parent_state[j] }
  edge.each_with_index do |tok, pos|
    prev_off = pos * d
    cur_off = (pos + 1) * d
    d.times do |j|
      z = params.b[j]
      wh_base = j * d
      wx_base = j * d
      emb_base = tok * d
      d.times do |k|
        z += params.w_h[wh_base + k] * states[prev_off + k]
        z += params.w_x[wx_base + k] * params.emb[emb_base + k]
      end
      states[cur_off + j] = z
    end
  end
end

def softmax_logits!(logits : Array(Float64))
  max = logits.max
  sum = 0.0
  logits.size.times do |i|
    x = Math.exp(logits[i] - max)
    logits[i] = x
    sum += x
  end
  logits.size.times { |i| logits[i] /= sum }
end

def add_loss_and_head_grads(
  params : RecurParams,
  grads : RecurParams,
  endpoint_states : Array(Float64),
  dh : Array(Float64),
  records : Array(RecurRecord),
) : {Float64, Int64}
  d = params.d_model
  v = params.vocab_size
  logits = Array(Float64).new(v, 0.0)
  loss = 0.0
  events = 0_i64

  records.each do |r|
    next if r.counts.empty?
    state_off = r.id * d
    v.times do |tok|
      z = params.c_o[tok]
      wo_base = tok * d
      d.times { |j| z += params.w_o[wo_base + j] * endpoint_states[state_off + j] }
      logits[tok] = z
    end
    softmax_logits!(logits)

    count_total = 0_i64
    r.counts.each do |pair|
      tok = pair[0]
      cnt = pair[1].to_i64
      count_total += cnt
      loss -= cnt.to_f * Math.log(logits[tok])
    end
    events += count_total

    v.times do |tok|
      g = logits[tok] * count_total.to_f
      r.counts.each do |pair|
        if pair[0] == tok
          g -= pair[1].to_f
          break
        end
      end
      grads.c_o[tok] += g
      wo_base = tok * d
      d.times do |j|
        h = endpoint_states[state_off + j]
        grads.w_o[wo_base + j] += g * h
        dh[state_off + j] += params.w_o[wo_base + j] * g
      end
    end
  end

  {loss, events}
end

def train_epoch(
  params : RecurParams,
  adam : AdamState,
  records : Array(RecurRecord),
  records_desc : Array(RecurRecord),
  endpoint_states : Array(Float64),
  lr : Float64,
  beta1 : Float64,
  beta2 : Float64,
  eps : Float64,
) : {Float64, Float64, Int64}
  d = params.d_model
  n_state = endpoint_states.size
  zero_state = Array(Float64).new(d, 0.0)
  edge_states = [] of Float64

  records.each do |r|
    parent_off = r.parent_id * d
    parent_state = r.parent_id == 0 ? zero_state : endpoint_states[parent_off, d]
    needed = (r.edge_tokens.size + 1) * d
    edge_states = Array(Float64).new(needed, 0.0) if edge_states.size != needed
    forward_edge(params, parent_state, r.edge_tokens, edge_states)
    end_off = r.edge_tokens.size * d
    state_off = r.id * d
    d.times { |j| endpoint_states[state_off + j] = edge_states[end_off + j] }
  end

  grads = RecurParams.new(params.vocab_size, params.d_model)
  dh = Array(Float64).new(n_state, 0.0)
  loss, events = add_loss_and_head_grads(params, grads, endpoint_states, dh, records)
  raise "no loss events found in trie records" if events == 0

  records_desc.each do |r|
    next if r.edge_tokens.empty?
    parent_off = r.parent_id * d
    parent_state = r.parent_id == 0 ? zero_state : endpoint_states[parent_off, d]
    needed = (r.edge_tokens.size + 1) * d
    edge_states = Array(Float64).new(needed, 0.0) if edge_states.size != needed
    forward_edge(params, parent_state, r.edge_tokens, edge_states)

    dh_cur = Array(Float64).new(d, 0.0)
    state_off = r.id * d
    d.times { |j| dh_cur[j] = dh[state_off + j] }

    pos = r.edge_tokens.size - 1
    while pos >= 0
      tok = r.edge_tokens[pos]
      prev_off_local = pos * d
      _cur_off_local = (pos + 1) * d
      # Linear recurrence: ∂z/∂h_prev = W_h, ∂z/∂emb = W_x, ∂z/∂b = 1.
      # No activation Jacobian; dz = dh_cur straight through.
      dz = Array(Float64).new(d, 0.0)
      d.times do |j|
        dz[j] = dh_cur[j]
        grads.b[j] += dz[j]
      end

      dh_prev = Array(Float64).new(d, 0.0)
      emb_base = tok * d
      d.times do |j|
        wh_base = j * d
        wx_base = j * d
        d.times do |k|
          grads.w_h[wh_base + k] += dz[j] * edge_states[prev_off_local + k]
          grads.w_x[wx_base + k] += dz[j] * params.emb[emb_base + k]
          grads.emb[emb_base + k] += dz[j] * params.w_x[wx_base + k]
          dh_prev[k] += dz[j] * params.w_h[wh_base + k]
        end
      end
      dh_cur = dh_prev
      pos -= 1
    end

    if r.parent_id != 0
      d.times { |j| dh[parent_off + j] += dh_cur[j] }
    end
  end

  scale = 1.0 / events.to_f
  grads.each_array { |a| a.size.times { |i| a[i] *= scale } }
  adam_update!(params, grads, adam, lr, beta1, beta2, eps)
  mean_nll = loss * scale
  {mean_nll, Math.exp(mean_nll), events}
end

def adam_update!(params : RecurParams, grads : RecurParams, adam : AdamState, lr : Float64, beta1 : Float64, beta2 : Float64, eps : Float64)
  adam.step += 1
  t = adam.step
  bc1 = 1.0 - beta1 ** t
  bc2 = 1.0 - beta2 ** t
  p_arrays = [] of Array(Float64)
  g_arrays = [] of Array(Float64)
  m_arrays = [] of Array(Float64)
  v_arrays = [] of Array(Float64)
  params.each_array { |a| p_arrays << a }
  grads.each_array { |a| g_arrays << a }
  adam.m.each_array { |a| m_arrays << a }
  adam.v.each_array { |a| v_arrays << a }

  p_arrays.each_with_index do |p, ai|
    g = g_arrays[ai]
    m = m_arrays[ai]
    vv = v_arrays[ai]
    p.size.times do |i|
      gi = g[i]
      m[i] = beta1 * m[i] + (1.0 - beta1) * gi
      vv[i] = beta2 * vv[i] + (1.0 - beta2) * gi * gi
      m_hat = m[i] / bc1
      v_hat = vv[i] / bc2
      p[i] -= lr * m_hat / (Math.sqrt(v_hat) + eps)
    end
  end
end

trie_dir = ""
save_path = ""
load_path = ""
d_model = 0
epochs = 0
lr = 0.001
seed = 1_u64
checkpoint_every = 0
max_nodes = 0
dry_run = false
beta1 = 0.9
beta2 = 0.999
eps = 1e-8

OptionParser.parse do |p|
  p.banner = "Usage: bin/agpt_train_recur_linear --trie DIR --d-model N --epochs N --lr F --seed N --save PATH [options]"
  p.on("--trie DIR", "Global radix trie directory") { |v| trie_dir = v }
  p.on("--d-model N", "Hidden dimension") { |v| d_model = v.to_i }
  p.on("--epochs N", "Training epochs") { |v| epochs = v.to_i }
  p.on("--lr F", "Adam learning rate (default 0.001)") { |v| lr = v.to_f }
  p.on("--seed N", "Initialization seed (default 1)") { |v| seed = v.to_u64 }
  p.on("--save PATH", "Write recurrent checkpoint") { |v| save_path = v }
  p.on("--load PATH", "Resume recurrent checkpoint") { |v| load_path = v }
  p.on("--checkpoint-every N", "Write epoch checkpoints every N epochs") { |v| checkpoint_every = v.to_i }
  p.on("--max-nodes N", "Diagnostic: use first N radix records only") { |v| max_nodes = v.to_i }
  p.on("--dry-run", "Load trie and report shape without training") { dry_run = true }
  p.on("-h", "--help", "Help") { puts p; exit 0 }
end

raise "--trie required" if trie_dir.empty?
raise "--d-model must be > 0" if d_model <= 0
raise "--epochs must be >= 0" if epochs < 0
raise "--lr must be > 0" if lr <= 0.0
if save_path.empty? && !dry_run
  raise "--save required unless --dry-run"
end

reader = RadixTrieReader.new(trie_dir, max_cached: 256)
records = load_records(reader, max_nodes)
records_desc = records.sort_by { |r| {-r.endpoint_depth, -r.id} }
events = estimate_loss_events(records)
expanded_chars = records.sum(0_i64) { |r| r.edge_tokens.size.to_i64 }
max_endpoint_depth = records.empty? ? 0 : records.max_of(&.endpoint_depth)

puts "AGPT linear-recurrent trainer"
puts "  trie: #{trie_dir}"
puts "  radix_records: #{records.size}#{max_nodes > 0 ? " (truncated)" : ""}"
puts "  expanded_states: #{expanded_chars}"
puts "  loss_events: #{events}"
puts "  vocab_size: #{reader.vocab_size}"
puts "  d_model: #{d_model}"
puts "  max_endpoint_depth: #{max_endpoint_depth}"
puts "  optimizer: adam (lr=#{lr}, beta1=#{beta1}, beta2=#{beta2}, eps=#{eps})"

if dry_run
  param_count = RecurParams.new(reader.vocab_size, d_model).total_floats
  state_mb = reader.radix_count.to_i64 * d_model.to_i64 * 8_i64 / 1024.0 / 1024.0
  puts "  params: #{param_count} float64"
  puts "  endpoint_state_memory: #{state_mb.round(2)} MB"
  exit 0
end

params = RecurParams.new(reader.vocab_size, d_model)
adam = AdamState.new(reader.vocab_size, d_model)
start_epoch = 0
if load_path.empty?
  params.fill_random!(seed)
else
  start_epoch, loaded_seed = load_checkpoint(load_path, params, adam)
  seed = loaded_seed
  puts "  loaded: #{load_path} (epoch=#{start_epoch}, adam_step=#{adam.step}, seed=#{seed})"
end

endpoint_states = Array(Float64).new(reader.radix_count * d_model, 0.0)

(start_epoch + 1).upto(start_epoch + epochs) do |epoch|
  t0 = Time.instant
  nll, ppl, trained_events = train_epoch(params, adam, records, records_desc, endpoint_states, lr, beta1, beta2, eps)
  wall = (Time.instant - t0).total_seconds
  ck_msg = ""
  if checkpoint_every > 0 && epoch % checkpoint_every == 0
    ck = epoch_checkpoint_path(save_path, epoch)
    save_checkpoint(ck, params, adam, epoch, seed)
    ck_msg = " checkpoint=#{ck}"
  end
  printf "epoch %6d  nll %.6f  ppl %.6f  events %d  wall %.3fs  adam_step %d%s\n",
    epoch, nll, ppl, trained_events, wall, adam.step, ck_msg
end

save_checkpoint(save_path, params, adam, start_epoch + epochs, seed)
puts "saved #{save_path} (epoch=#{start_epoch + epochs}, adam_step=#{adam.step})"
