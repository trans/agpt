require "option_parser"
require "../agpt"

# Vanilla GRU language model trainer (no trie, no AGPT framework).
#
# Truncated BPTT over length-T chunks of the corpus, h_init = 0 at each
# chunk start. Baseline control for cheap-f_θ AGPT runs (linear/tanh/
# GRU/GRU+wrap). The architecture/param layout/optimizer/save format
# match the AGPT GRU recur trainer (magic 'ACGU') so the same eval
# harness (agpt_recur_perplexity) loads the checkpoint. Magic 'ACGB'
# distinguishes baseline checkpoints from AGPT-trained ones.
#
# Forward at step t: same GRU as agpt_train_recur_gru.cr.
# Head loss at every position 0..T-1 (predict tokens[start+t+1] from
# h_{t+1}). Backward = standard T-step BPTT with grads accumulated
# into a single buffer over a minibatch, then one Adam step.

include MicroGPT::AGPT

MAGIC_GRU_LM   = 0x42474341_u32 # 'ACGB'
VERSION_GRU_LM =          1_i32

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

@[AlwaysInline]
def sigmoid(x : Float64) : Float64
  1.0 / (1.0 + Math.exp(-x))
end

class RecurParams
  getter vocab_size : Int32
  getter d_model : Int32
  getter emb : Array(Float64)
  getter w_z : Array(Float64); getter w_r : Array(Float64); getter w_n : Array(Float64)
  getter u_z : Array(Float64); getter u_r : Array(Float64); getter u_n : Array(Float64)
  getter b_z : Array(Float64); getter b_r : Array(Float64); getter b_n : Array(Float64)
  getter w_o : Array(Float64)
  getter c_o : Array(Float64)

  def initialize(@vocab_size : Int32, @d_model : Int32)
    v = @vocab_size
    d = @d_model
    @emb = Array(Float64).new(v * d, 0.0)
    @w_z = Array(Float64).new(d * d, 0.0)
    @w_r = Array(Float64).new(d * d, 0.0)
    @w_n = Array(Float64).new(d * d, 0.0)
    @u_z = Array(Float64).new(d * d, 0.0)
    @u_r = Array(Float64).new(d * d, 0.0)
    @u_n = Array(Float64).new(d * d, 0.0)
    @b_z = Array(Float64).new(d, 0.0)
    @b_r = Array(Float64).new(d, 0.0)
    @b_n = Array(Float64).new(d, 0.0)
    @w_o = Array(Float64).new(v * d, 0.0)
    @c_o = Array(Float64).new(v, 0.0)
  end

  def each_array(&block : Array(Float64) ->)
    yield @emb
    yield @w_z; yield @w_r; yield @w_n
    yield @u_z; yield @u_r; yield @u_n
    yield @b_z; yield @b_r; yield @b_n
    yield @w_o
    yield @c_o
  end

  def total_floats : Int32
    @emb.size +
      @w_z.size + @w_r.size + @w_n.size +
      @u_z.size + @u_r.size + @u_n.size +
      @b_z.size + @b_r.size + @b_n.size +
      @w_o.size + @c_o.size
  end

  def zero!
    each_array do |a|
      a.size.times { |i| a[i] = 0.0 }
    end
  end

  def fill_random!(seed : UInt64)
    rng = Random.new(seed)
    scale_emb = 0.02
    scale_rec = 1.0 / Math.sqrt(@d_model.to_f)
    fill_normal!(@emb, rng, scale_emb)
    fill_normal!(@w_z, rng, scale_rec); fill_normal!(@w_r, rng, scale_rec); fill_normal!(@w_n, rng, scale_rec)
    fill_normal!(@u_z, rng, scale_rec); fill_normal!(@u_r, rng, scale_rec); fill_normal!(@u_n, rng, scale_rec)
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
    io.write_bytes(MAGIC_GRU_LM, IO::ByteFormat::LittleEndian)
    io.write_bytes(VERSION_GRU_LM, IO::ByteFormat::LittleEndian)
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
    raise "bad GRU-LM checkpoint magic in #{path}" unless magic == MAGIC_GRU_LM
    version = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
    raise "unsupported GRU-LM checkpoint version #{version}" unless version == 1
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

# Build vocab from a source file as sorted unique chars (matches the
# convention used by the AGPT trie + recur trainers).
def load_corpus_tokens(corpus_path : String, vocab_source : String) : Array(Int32)
  chars = File.read(vocab_source).chars.to_set.to_a.sort
  char_to_id = {} of Char => Int32
  chars.each_with_index { |c, i| char_to_id[c] = i }
  text = File.read(corpus_path)
  tokens = Array(Int32).new(text.size)
  text.each_char do |c|
    tid = char_to_id[c]?
    tokens << (tid ? tid : 0)
  end
  tokens
end

def vocab_size_for(vocab_source : String) : Int32
  File.read(vocab_source).chars.to_set.size
end

# Precompute U_z·emb, U_r·emb, U_n·emb tables (V×d). Recomputed at each
# minibatch since params change after every Adam step.
def compute_x_projs(params : RecurParams) : {Array(Float64), Array(Float64), Array(Float64)}
  v = params.vocab_size
  d = params.d_model
  vi = v.to_i64
  di = d.to_i64
  proj_z = Array(Float64).new(v * d, 0.0)
  proj_r = Array(Float64).new(v * d, 0.0)
  proj_n = Array(Float64).new(v * d, 0.0)
  LibCBLAS.dgemm(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_TRANS,
    vi, di, di, 1.0, params.emb.to_unsafe, di, params.u_z.to_unsafe, di, 0.0, proj_z.to_unsafe, di)
  LibCBLAS.dgemm(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_TRANS,
    vi, di, di, 1.0, params.emb.to_unsafe, di, params.u_r.to_unsafe, di, 0.0, proj_r.to_unsafe, di)
  LibCBLAS.dgemm(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_TRANS,
    vi, di, di, 1.0, params.emb.to_unsafe, di, params.u_n.to_unsafe, di, 0.0, proj_n.to_unsafe, di)
  {proj_z, proj_r, proj_n}
end

@[AlwaysInline]
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

# Forward T steps from h=0. Writes states[(T+1)*d] + gate buffers.
def forward_chunk(
  params : RecurParams,
  x_proj_z : Array(Float64), x_proj_r : Array(Float64), x_proj_n : Array(Float64),
  tokens : Array(Int32), start : Int32, trunc : Int32,
  states : Array(Float64),
  z_buf : Array(Float64), r_buf : Array(Float64), n_buf : Array(Float64), m_buf : Array(Float64),
)
  d = params.d_model
  di = d.to_i64
  d.times { |j| states[j] = 0.0 }
  trunc.times do |t|
    tok = tokens[start + t]
    prev_off = t * d
    cur_off = (t + 1) * d
    x_base = tok * d

    d.times { |j| z_buf[cur_off + j] = params.b_z[j] + x_proj_z[x_base + j] }
    LibCBLAS.dgemv(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS,
      di, di, 1.0, params.w_z.to_unsafe, di,
      states.to_unsafe + prev_off, 1_i64,
      1.0, z_buf.to_unsafe + cur_off, 1_i64)
    d.times { |j| z_buf[cur_off + j] = sigmoid(z_buf[cur_off + j]) }

    d.times { |j| r_buf[cur_off + j] = params.b_r[j] + x_proj_r[x_base + j] }
    LibCBLAS.dgemv(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS,
      di, di, 1.0, params.w_r.to_unsafe, di,
      states.to_unsafe + prev_off, 1_i64,
      1.0, r_buf.to_unsafe + cur_off, 1_i64)
    d.times { |j| r_buf[cur_off + j] = sigmoid(r_buf[cur_off + j]) }

    d.times { |j| m_buf[cur_off + j] = r_buf[cur_off + j] * states[prev_off + j] }

    d.times { |j| n_buf[cur_off + j] = params.b_n[j] + x_proj_n[x_base + j] }
    LibCBLAS.dgemv(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS,
      di, di, 1.0, params.w_n.to_unsafe, di,
      m_buf.to_unsafe + cur_off, 1_i64,
      1.0, n_buf.to_unsafe + cur_off, 1_i64)
    d.times { |j| n_buf[cur_off + j] = Math.tanh(n_buf[cur_off + j]) }

    d.times do |j|
      zj = z_buf[cur_off + j]
      states[cur_off + j] = (1.0 - zj) * states[prev_off + j] + zj * n_buf[cur_off + j]
    end
  end
end

# Head loss at each of T positions. Writes dh_per_step[t*d..(t+1)*d-1]
# with W_o^T · (softmax - onehot(target_t)), accumulates head/bias grads.
def chunk_head_loss(
  params : RecurParams,
  grads : RecurParams,
  states : Array(Float64),
  tokens : Array(Int32), start : Int32, trunc : Int32,
  dh_per_step : Array(Float64),
  logits : Array(Float64), grad_logits : Array(Float64),
) : Float64
  d = params.d_model
  v = params.vocab_size
  di = d.to_i64
  vi = v.to_i64
  loss = 0.0

  trunc.times do |t|
    target = tokens[start + t + 1]
    state_off = (t + 1) * d

    v.times { |tok| logits[tok] = params.c_o[tok] }
    LibCBLAS.dgemv(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS,
      vi, di, 1.0, params.w_o.to_unsafe, di,
      states.to_unsafe + state_off, 1_i64,
      1.0, logits.to_unsafe, 1_i64)
    softmax_logits!(logits)

    p_target = logits[target]
    p_target = 1e-30 if p_target < 1e-30
    loss -= Math.log(p_target)

    v.times { |tok| grad_logits[tok] = logits[tok] }
    grad_logits[target] -= 1.0

    v.times { |tok| grads.c_o[tok] += grad_logits[tok] }
    LibCBLAS.dger(CBLAS_ROW_MAJOR,
      vi, di, 1.0,
      grad_logits.to_unsafe, 1_i64,
      states.to_unsafe + state_off, 1_i64,
      grads.w_o.to_unsafe, di)

    dh_off = t * d
    d.times { |j| dh_per_step[dh_off + j] = 0.0 }
    LibCBLAS.dgemv(CBLAS_ROW_MAJOR, CBLAS_TRANS,
      vi, di, 1.0,
      params.w_o.to_unsafe, di,
      grad_logits.to_unsafe, 1_i64,
      0.0, dh_per_step.to_unsafe + dh_off, 1_i64)
  end

  loss
end

# BPTT through T steps. Accumulates into grads. dh_per_step holds the
# per-step head gradient; absorb it at each step and propagate.
def backward_chunk(
  params : RecurParams,
  grads : RecurParams,
  tokens : Array(Int32), start : Int32, trunc : Int32,
  states : Array(Float64),
  z_buf : Array(Float64), r_buf : Array(Float64), n_buf : Array(Float64), m_buf : Array(Float64),
  dh_per_step : Array(Float64),
  dh_cur : Array(Float64), dh_prev : Array(Float64),
  dz_pre : Array(Float64), dr_pre : Array(Float64), dn_pre : Array(Float64), dm : Array(Float64),
)
  d = params.d_model
  di = d.to_i64
  d.times { |j| dh_cur[j] = 0.0 }

  pos = trunc - 1
  while pos >= 0
    tok = tokens[start + pos]
    prev_off_local = pos * d
    cur_off_local = (pos + 1) * d

    dh_off = pos * d
    d.times { |j| dh_cur[j] += dh_per_step[dh_off + j] }

    d.times do |j|
      zj = z_buf[cur_off_local + j]
      nj = n_buf[cur_off_local + j]
      hp = states[prev_off_local + j]
      dhj = dh_cur[j]
      dzj = dhj * (nj - hp)
      dnj = dhj * zj
      dn_pre[j] = dnj * (1.0 - nj * nj)
      dz_pre[j] = dzj * zj * (1.0 - zj)
      dh_prev[j] = dhj * (1.0 - zj)
    end

    LibCBLAS.dgemv(CBLAS_ROW_MAJOR, CBLAS_TRANS,
      di, di, 1.0, params.w_n.to_unsafe, di,
      dn_pre.to_unsafe, 1_i64, 0.0, dm.to_unsafe, 1_i64)

    d.times do |j|
      rj = r_buf[cur_off_local + j]
      hp = states[prev_off_local + j]
      dmj = dm[j]
      drj = dmj * hp
      dh_prev[j] += dmj * rj
      dr_pre[j] = drj * rj * (1.0 - rj)
    end

    LibCBLAS.dgemv(CBLAS_ROW_MAJOR, CBLAS_TRANS,
      di, di, 1.0, params.w_r.to_unsafe, di,
      dr_pre.to_unsafe, 1_i64, 1.0, dh_prev.to_unsafe, 1_i64)
    LibCBLAS.dgemv(CBLAS_ROW_MAJOR, CBLAS_TRANS,
      di, di, 1.0, params.w_z.to_unsafe, di,
      dz_pre.to_unsafe, 1_i64, 1.0, dh_prev.to_unsafe, 1_i64)

    d.times do |j|
      grads.b_z[j] += dz_pre[j]
      grads.b_r[j] += dr_pre[j]
      grads.b_n[j] += dn_pre[j]
    end

    LibCBLAS.dger(CBLAS_ROW_MAJOR, di, di, 1.0,
      dz_pre.to_unsafe, 1_i64,
      states.to_unsafe + prev_off_local, 1_i64,
      grads.w_z.to_unsafe, di)
    LibCBLAS.dger(CBLAS_ROW_MAJOR, di, di, 1.0,
      dr_pre.to_unsafe, 1_i64,
      states.to_unsafe + prev_off_local, 1_i64,
      grads.w_r.to_unsafe, di)
    LibCBLAS.dger(CBLAS_ROW_MAJOR, di, di, 1.0,
      dn_pre.to_unsafe, 1_i64,
      m_buf.to_unsafe + cur_off_local, 1_i64,
      grads.w_n.to_unsafe, di)

    emb_base = tok * d
    LibCBLAS.dger(CBLAS_ROW_MAJOR, di, di, 1.0,
      dz_pre.to_unsafe, 1_i64,
      params.emb.to_unsafe + emb_base, 1_i64,
      grads.u_z.to_unsafe, di)
    LibCBLAS.dger(CBLAS_ROW_MAJOR, di, di, 1.0,
      dr_pre.to_unsafe, 1_i64,
      params.emb.to_unsafe + emb_base, 1_i64,
      grads.u_r.to_unsafe, di)
    LibCBLAS.dger(CBLAS_ROW_MAJOR, di, di, 1.0,
      dn_pre.to_unsafe, 1_i64,
      params.emb.to_unsafe + emb_base, 1_i64,
      grads.u_n.to_unsafe, di)

    LibCBLAS.dgemv(CBLAS_ROW_MAJOR, CBLAS_TRANS,
      di, di, 1.0, params.u_z.to_unsafe, di,
      dz_pre.to_unsafe, 1_i64,
      1.0, grads.emb.to_unsafe + emb_base, 1_i64)
    LibCBLAS.dgemv(CBLAS_ROW_MAJOR, CBLAS_TRANS,
      di, di, 1.0, params.u_r.to_unsafe, di,
      dr_pre.to_unsafe, 1_i64,
      1.0, grads.emb.to_unsafe + emb_base, 1_i64)
    LibCBLAS.dgemv(CBLAS_ROW_MAJOR, CBLAS_TRANS,
      di, di, 1.0, params.u_n.to_unsafe, di,
      dn_pre.to_unsafe, 1_i64,
      1.0, grads.emb.to_unsafe + emb_base, 1_i64)

    dh_cur, dh_prev = dh_prev, dh_cur
    pos -= 1
  end
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

def train_minibatch(
  params : RecurParams,
  adam : AdamState,
  grads : RecurParams,
  tokens : Array(Int32), starts : Slice(Int32), trunc : Int32,
  x_proj_z : Array(Float64), x_proj_r : Array(Float64), x_proj_n : Array(Float64),
  states : Array(Float64),
  z_buf : Array(Float64), r_buf : Array(Float64), n_buf : Array(Float64), m_buf : Array(Float64),
  dh_per_step : Array(Float64),
  dh_cur : Array(Float64), dh_prev : Array(Float64),
  dz_pre : Array(Float64), dr_pre : Array(Float64), dn_pre : Array(Float64), dm : Array(Float64),
  logits : Array(Float64), grad_logits : Array(Float64),
  lr : Float64, beta1 : Float64, beta2 : Float64, eps : Float64,
) : {Float64, Int64}
  grads.zero!
  loss_total = 0.0
  events = 0_i64

  starts.each do |start|
    forward_chunk(params, x_proj_z, x_proj_r, x_proj_n,
      tokens, start, trunc,
      states, z_buf, r_buf, n_buf, m_buf)
    loss = chunk_head_loss(params, grads,
      states, tokens, start, trunc, dh_per_step,
      logits, grad_logits)
    backward_chunk(params, grads,
      tokens, start, trunc,
      states, z_buf, r_buf, n_buf, m_buf, dh_per_step,
      dh_cur, dh_prev, dz_pre, dr_pre, dn_pre, dm)
    loss_total += loss
    events += trunc.to_i64
  end

  raise "no events in minibatch" if events == 0
  scale = 1.0 / events.to_f
  grads.each_array { |a| a.size.times { |i| a[i] *= scale } }
  adam_update!(params, grads, adam, lr, beta1, beta2, eps)
  {loss_total, events}
end

def train_epoch(
  params : RecurParams,
  adam : AdamState,
  tokens : Array(Int32),
  trunc : Int32, batch_size : Int32,
  rng : Random,
  lr : Float64, beta1 : Float64, beta2 : Float64, eps : Float64,
) : {Float64, Float64, Int64, Int32}
  d = params.d_model
  v = params.vocab_size
  n_starts = tokens.size - trunc - 1
  raise "corpus too short: #{tokens.size} tokens for trunc=#{trunc}" if n_starts <= 0

  starts = Array(Int32).new(n_starts) { |i| i }
  starts.shuffle!(rng)

  grads = RecurParams.new(v, d)
  states = Array(Float64).new((trunc + 1) * d, 0.0)
  z_buf = Array(Float64).new((trunc + 1) * d, 0.0)
  r_buf = Array(Float64).new((trunc + 1) * d, 0.0)
  n_buf = Array(Float64).new((trunc + 1) * d, 0.0)
  m_buf = Array(Float64).new((trunc + 1) * d, 0.0)
  dh_per_step = Array(Float64).new(trunc * d, 0.0)
  dh_cur = Array(Float64).new(d, 0.0)
  dh_prev = Array(Float64).new(d, 0.0)
  dz_pre = Array(Float64).new(d, 0.0)
  dr_pre = Array(Float64).new(d, 0.0)
  dn_pre = Array(Float64).new(d, 0.0)
  dm = Array(Float64).new(d, 0.0)
  logits = Array(Float64).new(v, 0.0)
  grad_logits = Array(Float64).new(v, 0.0)

  loss_total = 0.0
  events_total = 0_i64
  updates = 0

  starts_slice = starts.to_unsafe.to_slice(n_starts)
  i = 0
  while i < n_starts
    batch_end = Math.min(i + batch_size, n_starts)
    batch = starts_slice[i, batch_end - i]
    x_proj_z, x_proj_r, x_proj_n = compute_x_projs(params)
    loss, events = train_minibatch(params, adam, grads,
      tokens, batch, trunc,
      x_proj_z, x_proj_r, x_proj_n,
      states, z_buf, r_buf, n_buf, m_buf,
      dh_per_step, dh_cur, dh_prev,
      dz_pre, dr_pre, dn_pre, dm,
      logits, grad_logits,
      lr, beta1, beta2, eps)
    loss_total += loss
    events_total += events
    updates += 1
    i = batch_end
  end

  mean_nll = loss_total / events_total.to_f
  {mean_nll, Math.exp(mean_nll), events_total, updates}
end

corpus_path = ""
vocab_source = ""
save_path = ""
load_path = ""
d_model = 0
trunc = 8
batch_size = 1024
epochs = 0
lr = 0.001
seed = 1_u64
checkpoint_every = 0
dry_run = false
beta1 = 0.9
beta2 = 0.999
eps = 1e-8

OptionParser.parse do |p|
  p.banner = "Usage: bin/agpt_train_gru_lm --corpus PATH --d-model N --epochs N --lr F --seed N --save PATH [options]"
  p.on("--corpus PATH", "Training corpus (bytes)") { |v| corpus_path = v }
  p.on("--vocab-source PATH", "Vocab source file (defaults to --corpus)") { |v| vocab_source = v }
  p.on("--d-model N", "Hidden dimension") { |v| d_model = v.to_i }
  p.on("--trunc-len N", "BPTT truncation length (default 8)") { |v| trunc = v.to_i }
  p.on("--batch-size N", "Chunks per Adam update (default 1024)") { |v| batch_size = v.to_i }
  p.on("--epochs N", "Training epochs") { |v| epochs = v.to_i }
  p.on("--lr F", "Adam learning rate (default 0.001)") { |v| lr = v.to_f }
  p.on("--seed N", "Initialization / shuffle seed (default 1)") { |v| seed = v.to_u64 }
  p.on("--save PATH", "Write checkpoint") { |v| save_path = v }
  p.on("--load PATH", "Resume checkpoint") { |v| load_path = v }
  p.on("--checkpoint-every N", "Write epoch checkpoints every N epochs") { |v| checkpoint_every = v.to_i }
  p.on("--dry-run", "Load + report shape without training") { dry_run = true }
  p.on("-h", "--help", "Help") { puts p; exit 0 }
end

raise "--corpus required" if corpus_path.empty?
raise "--d-model must be > 0" if d_model <= 0
raise "--epochs must be >= 0" if epochs < 0
raise "--lr must be > 0" if lr <= 0.0
raise "--trunc-len must be > 0" if trunc <= 0
raise "--batch-size must be > 0" if batch_size <= 0
if save_path.empty? && !dry_run
  raise "--save required unless --dry-run"
end
vocab_source = corpus_path if vocab_source.empty?

vocab_size = vocab_size_for(vocab_source)
tokens = load_corpus_tokens(corpus_path, vocab_source)

puts "AGPT vanilla GRU LM trainer (no trie; openblas hot path)"
puts "  corpus: #{corpus_path}"
puts "  vocab_source: #{vocab_source}"
puts "  vocab_size: #{vocab_size}"
puts "  corpus_tokens: #{tokens.size}"
puts "  d_model: #{d_model}"
puts "  trunc_len: #{trunc}"
puts "  batch_size: #{batch_size}"
puts "  starts_per_epoch: #{tokens.size - trunc - 1}"
puts "  updates_per_epoch: #{((tokens.size - trunc - 1) + batch_size - 1) // batch_size}"
puts "  optimizer: adam (lr=#{lr}, beta1=#{beta1}, beta2=#{beta2}, eps=#{eps})"

if dry_run
  param_count = RecurParams.new(vocab_size, d_model).total_floats
  puts "  params: #{param_count} float64"
  exit 0
end

params = RecurParams.new(vocab_size, d_model)
adam = AdamState.new(vocab_size, d_model)
start_epoch = 0
if load_path.empty?
  params.fill_random!(seed)
else
  start_epoch, loaded_seed = load_checkpoint(load_path, params, adam)
  seed = loaded_seed
  puts "  loaded: #{load_path} (epoch=#{start_epoch}, adam_step=#{adam.step}, seed=#{seed})"
end

rng = Random.new(seed ^ 0x6c6d5f73686966_u64)

(start_epoch + 1).upto(start_epoch + epochs) do |epoch|
  t0 = Time.instant
  nll, ppl, trained_events, updates = train_epoch(params, adam, tokens, trunc, batch_size,
    rng, lr, beta1, beta2, eps)
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
