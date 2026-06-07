require "option_parser"
require "../agpt"

# Linear-with-RMSNorm-recurrence AGPT trainer.
#   f_θ(h_p, x) = RMSNorm( W_h · h_p + W_x · emb(x) + b ; γ )
# where RMSNorm(z; γ)_i = z_i · γ_i / sqrt(mean(z²) + eps_norm).
#
# Tests the diagnosis that pure linear plateaus at depth ≥ 6-8 because
# the iterated recurrence A^N h_0 either compounds or attenuates — there's
# no mechanism to keep states in a useful range. RMSNorm provides explicit
# magnitude control without giving up the cheap-linear property (still no
# saturation; only one extra scalar normalize-per-step).
#
# Forked from `agpt_train_recur_linear.cr` (pure linear). Changes:
#   - RecurParams gains γ (gain) of size d, init = 1.0
#   - forward_edge applies RMSNorm after the matmul; saves z (pre-norm)
#     and σ per position so backward can compute gradients without
#     reconstructing z from h (avoids divide-by-γ_i)
#   - backward inserts the RMSNorm gradient between dh and dz
#   - magic word 'ACGN' (Linear+Norm) distinguishes the checkpoint family
#   - pd=1 partition support carries over verbatim
#
# Identical hyperparameter shape and CLI to the linear trainer so head-to-
# head comparison stays clean.

include MicroGPT::AGPT

MAGIC_RECUR_LIN_RMS = 0x4E474341_u32 # 'ACGN'
VERSION_RECUR       =          1_i32
EPS_NORM            =        1e-6_f64

@[Link("openblas_64")]
lib LibCBLAS
  fun dgemm = cblas_dgemm(layout : Int64, trans_a : Int64, trans_b : Int64,
                          m : Int64, n : Int64, k : Int64,
                          alpha : Float64,
                          a : Float64*, lda : Int64,
                          b : Float64*, ldb : Int64,
                          beta : Float64,
                          c : Float64*, ldc : Int64)
  fun dgemv = cblas_dgemv(layout : Int64, trans : Int64,
                          m : Int64, n : Int64,
                          alpha : Float64,
                          a : Float64*, lda : Int64,
                          x : Float64*, incx : Int64,
                          beta : Float64,
                          y : Float64*, incy : Int64)
  fun dger = cblas_dger(layout : Int64,
                        m : Int64, n : Int64,
                        alpha : Float64,
                        x : Float64*, incx : Int64,
                        y : Float64*, incy : Int64,
                        a : Float64*, lda : Int64)
end

CBLAS_ROW_MAJOR = 101_i64
CBLAS_NO_TRANS  = 111_i64
CBLAS_TRANS     = 112_i64

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
  getter g : Array(Float64) # RMSNorm gain (γ), size d
  getter w_o : Array(Float64)
  getter c_o : Array(Float64)

  def initialize(@vocab_size : Int32, @d_model : Int32)
    v = @vocab_size
    d = @d_model
    @emb = Array(Float64).new(v * d, 0.0)
    @w_h = Array(Float64).new(d * d, 0.0)
    @w_x = Array(Float64).new(d * d, 0.0)
    @b   = Array(Float64).new(d, 0.0)
    @g   = Array(Float64).new(d, 0.0)
    @w_o = Array(Float64).new(v * d, 0.0)
    @c_o = Array(Float64).new(v, 0.0)
  end

  def each_array(&block : Array(Float64) ->)
    yield @emb
    yield @w_h
    yield @w_x
    yield @b
    yield @g
    yield @w_o
    yield @c_o
  end

  def total_floats : Int32
    @emb.size + @w_h.size + @w_x.size + @b.size + @g.size + @w_o.size + @c_o.size
  end

  def fill_random!(seed : UInt64)
    rng = Random.new(seed)
    scale_emb = 0.02
    scale_rec = 1.0 / Math.sqrt(@d_model.to_f)
    fill_normal!(@emb, rng, scale_emb)
    fill_normal!(@w_h, rng, scale_rec)
    fill_normal!(@w_x, rng, scale_rec)
    fill_normal!(@w_o, rng, scale_rec)
    # RMSNorm gain initialized to 1.0 (identity scaling).
    @g.size.times { |i| @g[i] = 1.0 }
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
    io.write_bytes(MAGIC_RECUR_LIN_RMS, IO::ByteFormat::LittleEndian)
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
    raise "bad linear-RMSNorm checkpoint magic in #{path}" unless magic == MAGIC_RECUR_LIN_RMS
    version = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
    raise "unsupported linear-RMSNorm checkpoint version #{version}" unless version == VERSION_RECUR
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

def build_partitions(records : Array(RecurRecord), partition_depth : Int32) : Array(Array(RecurRecord))
  case partition_depth
  when 0
    [records]
  when 1
    root_child_by_id = {} of Int32 => Int32
    partitions = {} of Int32 => Array(RecurRecord)
    records.each do |r|
      root_child_id =
        if r.parent_id == 0
          r.id
        else
          parent_root = root_child_by_id[r.parent_id]?
          raise "parent #{r.parent_id} missing before child #{r.id}; radix records must be parent-ordered" unless parent_root
          parent_root
        end
      root_child_by_id[r.id] = root_child_id
      partitions[root_child_id] ||= [] of RecurRecord
      partitions[root_child_id] << r
    end
    partitions.keys.sort.map { |id| partitions[id] }
  else
    raise "--partition-depth currently supports only 0 or 1"
  end
end

def compute_x_proj(params : RecurParams) : Array(Float64)
  v = params.vocab_size
  d = params.d_model
  vi = v.to_i64
  di = d.to_i64
  proj = Array(Float64).new(v * d, 0.0)
  LibCBLAS.dgemm(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_TRANS,
    vi, di, di,
    1.0,
    params.emb.to_unsafe, di,
    params.w_x.to_unsafe, di,
    0.0,
    proj.to_unsafe, di)
  proj
end

# forward_edge — linear matmul + RMSNorm at each step.
# Writes h_post into states (size (edge_len+1)*d, includes parent state at position 0).
# Also writes pre-norm z values into z_states (same size; position 0 unused).
# Also writes σ values into sigma_states (size edge_len+1; position 0 unused).
def forward_edge(
  params : RecurParams,
  x_proj : Array(Float64),
  endpoint_states : Array(Float64),
  parent_id : Int32,
  edge : Array(Int32),
  states : Array(Float64),
  z_states : Array(Float64),
  sigma_states : Array(Float64),
)
  d = params.d_model
  di = d.to_i64
  if parent_id != 0
    parent_off = parent_id * d
    d.times { |j| states[j] = endpoint_states[parent_off + j] }
  else
    d.times { |j| states[j] = 0.0 }
  end
  edge.each_with_index do |tok, pos|
    prev_off = pos * d
    cur_off = (pos + 1) * d
    x_base = tok * d
    # z = b + x_proj[tok] + w_h · h_prev
    d.times { |j| z_states[cur_off + j] = params.b[j] + x_proj[x_base + j] }
    LibCBLAS.dgemv(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS,
      di, di,
      1.0,
      params.w_h.to_unsafe, di,
      states.to_unsafe + prev_off, 1_i64,
      1.0,
      z_states.to_unsafe + cur_off, 1_i64)
    # RMSNorm: σ² = mean(z²) + eps;  h_i = z_i · γ_i / σ
    sumsq = 0.0
    d.times { |j| z = z_states[cur_off + j]; sumsq += z * z }
    sigma = Math.sqrt(sumsq / d.to_f + EPS_NORM)
    sigma_states[pos + 1] = sigma
    d.times { |j| states[cur_off + j] = z_states[cur_off + j] * params.g[j] / sigma }
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
  di = d.to_i64
  vi = v.to_i64
  logits = Array(Float64).new(v, 0.0)
  grad_logits = Array(Float64).new(v, 0.0)
  loss = 0.0
  events = 0_i64

  records.each do |r|
    next if r.counts.empty?
    state_off = r.id * d
    v.times { |tok| logits[tok] = params.c_o[tok] }
    LibCBLAS.dgemv(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS,
      vi, di,
      1.0,
      params.w_o.to_unsafe, di,
      endpoint_states.to_unsafe + state_off, 1_i64,
      1.0,
      logits.to_unsafe, 1_i64)
    softmax_logits!(logits)

    count_total = 0_i64
    r.counts.each do |pair|
      tok = pair[0]
      cnt = pair[1].to_i64
      count_total += cnt
      loss -= cnt.to_f * Math.log(logits[tok])
    end
    events += count_total

    count_total_f = count_total.to_f
    v.times { |tok| grad_logits[tok] = logits[tok] * count_total_f }
    r.counts.each do |pair|
      grad_logits[pair[0]] -= pair[1].to_f
    end

    v.times { |tok| grads.c_o[tok] += grad_logits[tok] }
    LibCBLAS.dger(CBLAS_ROW_MAJOR,
      vi, di,
      1.0,
      grad_logits.to_unsafe, 1_i64,
      endpoint_states.to_unsafe + state_off, 1_i64,
      grads.w_o.to_unsafe, di)
    LibCBLAS.dgemv(CBLAS_ROW_MAJOR, CBLAS_TRANS,
      vi, di,
      1.0,
      params.w_o.to_unsafe, di,
      grad_logits.to_unsafe, 1_i64,
      1.0,
      dh.to_unsafe + state_off, 1_i64)
  end

  {loss, events}
end

def train_batch(
  params : RecurParams,
  adam : AdamState,
  records : Array(RecurRecord),
  records_desc : Array(RecurRecord),
  endpoint_states : Array(Float64),
  x_proj : Array(Float64),
  lr : Float64,
  beta1 : Float64,
  beta2 : Float64,
  eps : Float64,
) : {Float64, Int64}
  d = params.d_model
  di = d.to_i64
  d_f = d.to_f
  n_state = endpoint_states.size
  edge_states = [] of Float64
  z_states = [] of Float64
  sigma_states = [] of Float64

  records.each do |r|
    needed = (r.edge_tokens.size + 1) * d
    if edge_states.size != needed
      edge_states = Array(Float64).new(needed, 0.0)
      z_states = Array(Float64).new(needed, 0.0)
      sigma_states = Array(Float64).new(r.edge_tokens.size + 1, 0.0)
    end
    forward_edge(params, x_proj, endpoint_states, r.parent_id, r.edge_tokens, edge_states, z_states, sigma_states)
    end_off = r.edge_tokens.size * d
    state_off = r.id * d
    d.times { |j| endpoint_states[state_off + j] = edge_states[end_off + j] }
  end

  grads = RecurParams.new(params.vocab_size, params.d_model)
  dh = Array(Float64).new(n_state, 0.0)
  loss, events = add_loss_and_head_grads(params, grads, endpoint_states, dh, records)
  raise "no loss events found in trie records" if events == 0

  dh_cur = Array(Float64).new(d, 0.0)
  dh_prev = Array(Float64).new(d, 0.0)
  dz = Array(Float64).new(d, 0.0)
  records_desc.each do |r|
    next if r.edge_tokens.empty?
    parent_off = r.parent_id * d
    needed = (r.edge_tokens.size + 1) * d
    if edge_states.size != needed
      edge_states = Array(Float64).new(needed, 0.0)
      z_states = Array(Float64).new(needed, 0.0)
      sigma_states = Array(Float64).new(r.edge_tokens.size + 1, 0.0)
    end
    forward_edge(params, x_proj, endpoint_states, r.parent_id, r.edge_tokens, edge_states, z_states, sigma_states)

    state_off = r.id * d
    d.times { |j| dh_cur[j] = dh[state_off + j] }

    pos = r.edge_tokens.size - 1
    while pos >= 0
      tok = r.edge_tokens[pos]
      prev_off_local = pos * d
      cur_off_local = (pos + 1) * d
      sigma = sigma_states[pos + 1]

      # RMSNorm backward
      #   forward: h_i = z_i · γ_i / σ
      #   backward: dz_i = (γ_i · dh_i) / σ - z_i · S / (d · σ²),  S = Σ h_j · dh_j
      #             dγ_i += dh_i · z_i / σ  =  dh_i · h_i / γ_i (equivalent)
      s_sum = 0.0
      d.times { |j| s_sum += edge_states[cur_off_local + j] * dh_cur[j] }
      inv_sigma = 1.0 / sigma
      coeff = s_sum / (d_f * sigma * sigma)
      d.times do |j|
        zj = z_states[cur_off_local + j]
        dz[j] = (params.g[j] * dh_cur[j]) * inv_sigma - zj * coeff
        grads.g[j] += dh_cur[j] * zj * inv_sigma
      end

      # Linear backward (same as agpt_train_recur_linear).
      d.times { |j| grads.b[j] += dz[j] }
      dh_prev.fill(0.0)
      emb_base = tok * d
      LibCBLAS.dger(CBLAS_ROW_MAJOR,
        di, di,
        1.0,
        dz.to_unsafe, 1_i64,
        edge_states.to_unsafe + prev_off_local, 1_i64,
        grads.w_h.to_unsafe, di)
      LibCBLAS.dger(CBLAS_ROW_MAJOR,
        di, di,
        1.0,
        dz.to_unsafe, 1_i64,
        params.emb.to_unsafe + emb_base, 1_i64,
        grads.w_x.to_unsafe, di)
      LibCBLAS.dgemv(CBLAS_ROW_MAJOR, CBLAS_TRANS,
        di, di,
        1.0,
        params.w_x.to_unsafe, di,
        dz.to_unsafe, 1_i64,
        1.0,
        grads.emb.to_unsafe + emb_base, 1_i64)
      LibCBLAS.dgemv(CBLAS_ROW_MAJOR, CBLAS_TRANS,
        di, di,
        1.0,
        params.w_h.to_unsafe, di,
        dz.to_unsafe, 1_i64,
        0.0,
        dh_prev.to_unsafe, 1_i64)
      dh_cur, dh_prev = dh_prev, dh_cur
      pos -= 1
    end

    if r.parent_id != 0
      d.times { |j| dh[parent_off + j] += dh_cur[j] }
    end
  end

  scale = 1.0 / events.to_f
  grads.each_array { |a| a.size.times { |i| a[i] *= scale } }
  adam_update!(params, grads, adam, lr, beta1, beta2, eps)
  {loss, events}
end

def train_epoch(
  params : RecurParams,
  adam : AdamState,
  partitions : Array(Array(RecurRecord)),
  endpoint_states : Array(Float64),
  lr : Float64,
  beta1 : Float64,
  beta2 : Float64,
  eps : Float64,
) : {Float64, Float64, Int64, Int32}
  loss_total = 0.0
  events_total = 0_i64
  updates = 0

  partitions.each do |records|
    next if records.empty?
    records_desc = records.sort_by { |r| {-r.endpoint_depth, -r.id} }
    x_proj = compute_x_proj(params)
    loss, events = train_batch(params, adam, records, records_desc, endpoint_states, x_proj, lr, beta1, beta2, eps)
    loss_total += loss
    events_total += events
    updates += 1
  end

  raise "no loss events found in trie records" if events_total == 0
  mean_nll = loss_total / events_total.to_f
  {mean_nll, Math.exp(mean_nll), events_total, updates}
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
partition_depth = 0
dry_run = false
beta1 = 0.9
beta2 = 0.999
eps = 1e-8

OptionParser.parse do |p|
  p.banner = "Usage: bin/agpt_train_recur_linear_rms --trie DIR --d-model N --epochs N --lr F --seed N --save PATH [options]"
  p.on("--trie DIR", "Global radix trie directory") { |v| trie_dir = v }
  p.on("--d-model N", "Hidden dimension") { |v| d_model = v.to_i }
  p.on("--epochs N", "Training epochs") { |v| epochs = v.to_i }
  p.on("--lr F", "Adam learning rate (default 0.001)") { |v| lr = v.to_f }
  p.on("--seed N", "Initialization seed (default 1)") { |v| seed = v.to_u64 }
  p.on("--save PATH", "Write recurrent checkpoint") { |v| save_path = v }
  p.on("--load PATH", "Resume recurrent checkpoint") { |v| load_path = v }
  p.on("--checkpoint-every N", "Write epoch checkpoints every N epochs") { |v| checkpoint_every = v.to_i }
  p.on("--max-nodes N", "Diagnostic: use first N radix records only") { |v| max_nodes = v.to_i }
  p.on("--partition-depth N", "Subtree optimizer partition depth: 0 full batch, 1 root-child batches (default 0)") { |v| partition_depth = v.to_i }
  p.on("--dry-run", "Load trie and report shape without training") { dry_run = true }
  p.on("-h", "--help", "Help") { puts p; exit 0 }
end

raise "--trie required" if trie_dir.empty?
raise "--d-model must be > 0" if d_model <= 0
raise "--epochs must be >= 0" if epochs < 0
raise "--lr must be > 0" if lr <= 0.0
raise "--partition-depth must be >= 0" if partition_depth < 0
if save_path.empty? && !dry_run
  raise "--save required unless --dry-run"
end

reader = RadixTrieReader.new(trie_dir, max_cached: 256)
records = load_records(reader, max_nodes)
events = estimate_loss_events(records)
expanded_chars = records.sum(0_i64) { |r| r.edge_tokens.size.to_i64 }
max_endpoint_depth = records.empty? ? 0 : records.max_of(&.endpoint_depth)
partitions = build_partitions(records, partition_depth)

puts "AGPT linear+RMSNorm recurrent trainer (openblas hot path)"
puts "  trie: #{trie_dir}"
puts "  radix_records: #{records.size}#{max_nodes > 0 ? " (truncated)" : ""}"
puts "  expanded_states: #{expanded_chars}"
puts "  loss_events: #{events}"
puts "  partition_depth: #{partition_depth}"
puts "  partitions: #{partitions.size}"
puts "  vocab_size: #{reader.vocab_size}"
puts "  d_model: #{d_model}"
puts "  max_endpoint_depth: #{max_endpoint_depth}"
puts "  optimizer: adam (lr=#{lr}, beta1=#{beta1}, beta2=#{beta2}, eps=#{eps})"
puts "  norm_eps: #{EPS_NORM}"

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
  nll, ppl, trained_events, updates = train_epoch(params, adam, partitions, endpoint_states, lr, beta1, beta2, eps)
  wall = (Time.instant - t0).total_seconds
  ck_msg = ""
  if checkpoint_every > 0 && epoch % checkpoint_every == 0
    ck = epoch_checkpoint_path(save_path, epoch)
    save_checkpoint(ck, params, adam, epoch, seed)
    ck_msg = " checkpoint=#{ck}"
  end
  printf "epoch %6d  nll %.6f  ppl %.6f  events %d  updates %d  wall %.3fs  adam_step %d%s\n",
    epoch, nll, ppl, trained_events, updates, wall, adam.step, ck_msg
end

save_checkpoint(save_path, params, adam, start_epoch + epochs, seed)
puts "saved #{save_path} (epoch=#{start_epoch + epochs}, adam_step=#{adam.step})"
