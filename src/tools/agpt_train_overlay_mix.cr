require "option_parser"
require "../agpt"

# Overlay-AGPT GRU trainer with learnable depth mixture (KN-style).
#
# Per sample window covering corpus[s, s + n_targets + d_max):
#   For each target position t in [s + d_max, s + d_max + n_targets - 1]:
#     For each backoff depth j in [1, d_max]:
#       walk corpus[t-j..t-1] from h=0, get h_j
#     mass m_j from trie node for corpus[t-j..t-1]
#     score_j = lambda_proj · h_j + lambda_alpha · log(m_j)
#     lambda_j = softmax_over_j(score_j)
#     p_j = softmax(W_o · h_j + c_o)
#     P = Σ_j lambda_j · p_j(target)
#     loss = -log P
#   Backward: mixture gradient distributes weight = lambda_j p_j(target) / P
#     to each walk's logits + BPTT, plus lambda gradient flowing through
#     softmax-over-j and the score net into lambda_proj/lambda_alpha and
#     each h_j (added to the head-derived dh_j before BPTT).
#
# Mass enters the FORWARD path only (as a feature for the score net).
# No mass factor in the gradient — sampling already supplies it.
#
# Magic 'ACGM'. GRU param layout + 2 new arrays (lambda_proj : d-vec,
# lambda_alpha : scalar). Eval requires a separate Python script that
# can load the trie and compute the mixture; the bin/agpt_recur_perplexity
# sliding-window eval doesn't apply directly.

include MicroGPT::AGPT

MAGIC_OVERLAY_MIX = 0x4D474341_u32 # 'ACGM'
VERSION_OVERLAY   =          1_i32

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
  getter lambda_proj : Array(Float64)   # d-vec, score = lambda_proj · h
  getter lambda_alpha : Array(Float64)  # size 1, mass coefficient

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
    @lambda_proj = Array(Float64).new(d, 0.0)
    @lambda_alpha = Array(Float64).new(1, 0.0)
  end

  def each_array(&block : Array(Float64) ->)
    yield @emb
    yield @w_z; yield @w_r; yield @w_n
    yield @u_z; yield @u_r; yield @u_n
    yield @b_z; yield @b_r; yield @b_n
    yield @w_o
    yield @c_o
    yield @lambda_proj
    yield @lambda_alpha
  end

  def total_floats : Int32
    @emb.size +
      @w_z.size + @w_r.size + @w_n.size +
      @u_z.size + @u_r.size + @u_n.size +
      @b_z.size + @b_r.size + @b_n.size +
      @w_o.size + @c_o.size +
      @lambda_proj.size + @lambda_alpha.size
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
    fill_normal!(@lambda_proj, rng, scale_rec)
    # lambda_alpha stays 0 → initial λ depends only on h, mass enters as the
    # model learns the alpha coefficient.
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
    io.write_bytes(MAGIC_OVERLAY_MIX, IO::ByteFormat::LittleEndian)
    io.write_bytes(VERSION_OVERLAY, IO::ByteFormat::LittleEndian)
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
    raise "bad Overlay-Mix checkpoint magic in #{path}" unless magic == MAGIC_OVERLAY_MIX
    version = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
    raise "unsupported Overlay-Mix checkpoint version #{version}" unless version == 1
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

record OverlayRecord,
  id : Int32,
  parent_id : Int32,
  endpoint_depth : Int32,
  edge_tokens : Array(Int32),
  counts : Array({Int32, Int32})

# Walk the trie root-down, build records list and parallel arrays for
# mass and entropy per record (indexed by record id).
def load_trie_with_stats(trie_dir : String) : {Array(OverlayRecord), Array(Float64), Array(Float64), Int32, Hash(Int32, Hash(Int32, Int32))}
  reader = RadixTrieReader.new(trie_dir, max_cached: 256)
  records = [] of OverlayRecord
  reader.each do |r|
    next if r.id == 0
    records << OverlayRecord.new(r.id, r.parent_id, r.endpoint_depth, r.edge_tokens, r.counts)
  end
  max_id = records.max_of(&.id)
  mass = Array(Float64).new(max_id + 1, 0.0)
  entropy = Array(Float64).new(max_id + 1, 0.0)
  records.each do |r|
    m = 0_i64
    r.counts.each { |pair| m += pair[1].to_i64 }
    mass[r.id] = m.to_f
    if m > 0
      h = 0.0
      r.counts.each do |pair|
        c = pair[1].to_f
        next if c <= 0.0
        p = c / m.to_f
        h -= p * Math.log(p)
      end
      entropy[r.id] = h
    end
  end
  # child lookup: (parent_id, first_token_of_edge) -> child_record_id
  child_by_first = Hash(Int32, Hash(Int32, Int32)).new do |h, k|
    h[k] = {} of Int32 => Int32
  end
  records.each do |r|
    next if r.edge_tokens.empty?
    child_by_first[r.parent_id][r.edge_tokens[0]] = r.id
  end
  {records, mass, entropy, max_id, child_by_first}
end

# Walk the trie following prefix tokens. Returns the mass of the trie
# node containing the prefix of length j ending at the walk endpoint.
# Returns 1.0 (so log(m) = 0, no signal) if the prefix doesn't appear
# in the trie. Mid-edge prefixes return the containing record's mass.
@[AlwaysInline]
def mass_at_prefix(
  tokens : Array(Int32), start : Int32, j : Int32,
  child_by_first : Hash(Int32, Hash(Int32, Int32)),
  records : Array(OverlayRecord), record_index : Hash(Int32, Int32),
  mass : Array(Float64),
) : Float64
  parent_id = 0
  consumed = 0
  while consumed < j
    kids = child_by_first[parent_id]?
    return 1.0 if kids.nil?
    needed = tokens[start + consumed]
    child_id = kids[needed]?
    return 1.0 if child_id.nil?
    rec_idx = record_index[child_id]
    rec = records[rec_idx]
    e_idx = 0
    edge_size = rec.edge_tokens.size
    matched_full = true
    while e_idx < edge_size && consumed + e_idx < j
      if rec.edge_tokens[e_idx] != tokens[start + consumed + e_idx]
        matched_full = false
        break
      end
      e_idx += 1
    end
    return 1.0 unless matched_full
    if consumed + e_idx == j
      m = mass[child_id]
      return m < 1.0 ? 1.0 : m
    end
    consumed += e_idx
    parent_id = child_id
  end
  1.0
end

# Precompute U_z·emb, U_r·emb, U_n·emb tables (V×d).
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

# Forward `trunc` GRU steps from h=0 starting at corpus[start].
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

# Endpoint head loss: only computes loss at h_{trunc} (the final state).
# Zeroes dh_per_step for all earlier positions; writes dh at position
# (trunc - 1) so backward_chunk picks it up at the last step.
#
# `weight` scales both the returned loss and the gradient contributions
# (head, dh_per_step). For trust-weighted training; pass 1.0 for
# unweighted behaviour.
def endpoint_head_loss(
  params : RecurParams,
  grads : RecurParams,
  states : Array(Float64),
  target_token : Int32, trunc : Int32,
  dh_per_step : Array(Float64),
  logits : Array(Float64), grad_logits : Array(Float64),
  weight : Float64 = 1.0,
) : Float64
  d = params.d_model
  v = params.vocab_size
  di = d.to_i64
  vi = v.to_i64

  (trunc * d).times { |i| dh_per_step[i] = 0.0 }

  state_off = trunc * d  # h_trunc

  v.times { |tok| logits[tok] = params.c_o[tok] }
  LibCBLAS.dgemv(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS,
    vi, di, 1.0, params.w_o.to_unsafe, di,
    states.to_unsafe + state_off, 1_i64,
    1.0, logits.to_unsafe, 1_i64)
  softmax_logits!(logits)

  p_target = logits[target_token]
  p_target = 1e-30 if p_target < 1e-30
  raw_loss = -Math.log(p_target)
  loss = weight * raw_loss

  v.times { |tok| grad_logits[tok] = weight * logits[tok] }
  grad_logits[target_token] -= weight

  v.times { |tok| grads.c_o[tok] += grad_logits[tok] }
  LibCBLAS.dger(CBLAS_ROW_MAJOR,
    vi, di, 1.0,
    grad_logits.to_unsafe, 1_i64,
    states.to_unsafe + state_off, 1_i64,
    grads.w_o.to_unsafe, di)

  dh_off = (trunc - 1) * d
  LibCBLAS.dgemv(CBLAS_ROW_MAJOR, CBLAS_TRANS,
    vi, di, 1.0,
    params.w_o.to_unsafe, di,
    grad_logits.to_unsafe, 1_i64,
    0.0, dh_per_step.to_unsafe + dh_off, 1_i64)

  loss
end

# BPTT through `trunc` steps. Accumulates into grads.
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

# Mixture forward only (no backward, no Adam) — for eval. Returns total
# NLL and event count over a window's targets. Same forward math as
# train_window but discards everything after computing loss.
def eval_window(
  params : RecurParams,
  tokens : Array(Int32),
  window_start : Int32, n_targets : Int32, d_max : Int32,
  x_proj_z : Array(Float64), x_proj_r : Array(Float64), x_proj_n : Array(Float64),
  walk_states_arr : Array(Array(Float64)),
  walk_z_arr : Array(Array(Float64)),
  walk_r_arr : Array(Array(Float64)),
  walk_n_arr : Array(Array(Float64)),
  walk_m_arr : Array(Array(Float64)),
  per_walk_logits_arr : Array(Array(Float64)),
  per_walk_p_target : Array(Float64),
  mass_arr : Array(Float64), log_mass_arr : Array(Float64),
  score_arr : Array(Float64), lambda_arr : Array(Float64),
  child_by_first : Hash(Int32, Hash(Int32, Int32)),
  records : Array(OverlayRecord),
  record_index : Hash(Int32, Int32),
  mass : Array(Float64),
) : {Float64, Int64}
  d = params.d_model
  v = params.vocab_size
  di = d.to_i64
  vi = v.to_i64
  loss_total = 0.0
  events = 0_i64

  n_targets.times do |i|
    target = window_start + d_max + i
    target_token = tokens[target]

    j = 1
    while j <= d_max
      walk_start = target - j
      forward_chunk(params, x_proj_z, x_proj_r, x_proj_n,
        tokens, walk_start, j,
        walk_states_arr[j - 1], walk_z_arr[j - 1], walk_r_arr[j - 1], walk_n_arr[j - 1], walk_m_arr[j - 1])
      j += 1
    end

    j = 1
    while j <= d_max
      walk_start = target - j
      m = mass_at_prefix(tokens, walk_start, j, child_by_first, records, record_index, mass)
      mass_arr[j - 1] = m
      log_mass_arr[j - 1] = Math.log(m)
      j += 1
    end

    j = 1
    while j <= d_max
      h_off = j * d
      s = 0.0
      d.times { |k| s += params.lambda_proj[k] * walk_states_arr[j - 1][h_off + k] }
      s += params.lambda_alpha[0] * log_mass_arr[j - 1]
      score_arr[j - 1] = s
      j += 1
    end
    max_score = score_arr[0]
    jj = 1
    while jj < d_max
      max_score = score_arr[jj] if score_arr[jj] > max_score
      jj += 1
    end
    sum_exp = 0.0
    d_max.times do |k|
      e = Math.exp(score_arr[k] - max_score)
      lambda_arr[k] = e
      sum_exp += e
    end
    d_max.times { |k| lambda_arr[k] /= sum_exp }

    j = 1
    while j <= d_max
      h_off = j * d
      v.times { |tok| per_walk_logits_arr[j - 1][tok] = params.c_o[tok] }
      LibCBLAS.dgemv(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS,
        vi, di, 1.0, params.w_o.to_unsafe, di,
        walk_states_arr[j - 1].to_unsafe + h_off, 1_i64,
        1.0, per_walk_logits_arr[j - 1].to_unsafe, 1_i64)
      softmax_logits!(per_walk_logits_arr[j - 1])
      per_walk_p_target[j - 1] = per_walk_logits_arr[j - 1][target_token]
      j += 1
    end

    big_p = 0.0
    d_max.times { |k| big_p += lambda_arr[k] * per_walk_p_target[k] }
    big_p = 1e-30 if big_p < 1e-30
    loss_total += -Math.log(big_p)
    events += 1_i64
  end

  {loss_total, events}
end

def eval_corpus(
  params : RecurParams,
  tokens : Array(Int32),
  n_targets : Int32, d_max : Int32,
  child_by_first : Hash(Int32, Hash(Int32, Int32)),
  records : Array(OverlayRecord),
  record_index : Hash(Int32, Int32),
  mass : Array(Float64),
) : {Float64, Float64, Int64}
  d = params.d_model
  v = params.vocab_size
  window_len = n_targets + d_max
  raise "eval corpus too short" if tokens.size < window_len + 1
  max_start = tokens.size - window_len - 1
  n_windows = max_start // n_targets + 1

  buf_len = (d_max + 1) * d
  walk_states_arr = Array(Array(Float64)).new(d_max) { Array(Float64).new(buf_len, 0.0) }
  walk_z_arr = Array(Array(Float64)).new(d_max) { Array(Float64).new(buf_len, 0.0) }
  walk_r_arr = Array(Array(Float64)).new(d_max) { Array(Float64).new(buf_len, 0.0) }
  walk_n_arr = Array(Array(Float64)).new(d_max) { Array(Float64).new(buf_len, 0.0) }
  walk_m_arr = Array(Array(Float64)).new(d_max) { Array(Float64).new(buf_len, 0.0) }
  per_walk_logits_arr = Array(Array(Float64)).new(d_max) { Array(Float64).new(v, 0.0) }
  per_walk_p_target = Array(Float64).new(d_max, 0.0)
  mass_arr = Array(Float64).new(d_max, 1.0)
  log_mass_arr = Array(Float64).new(d_max, 0.0)
  score_arr = Array(Float64).new(d_max, 0.0)
  lambda_arr = Array(Float64).new(d_max, 0.0)

  x_proj_z, x_proj_r, x_proj_n = compute_x_projs(params)

  loss_total = 0.0
  events_total = 0_i64
  n_windows.times do |i|
    window_start = i * n_targets
    break if window_start + window_len + 1 > tokens.size
    loss, events = eval_window(params, tokens, window_start, n_targets, d_max,
      x_proj_z, x_proj_r, x_proj_n,
      walk_states_arr, walk_z_arr, walk_r_arr, walk_n_arr, walk_m_arr,
      per_walk_logits_arr, per_walk_p_target,
      mass_arr, log_mass_arr, score_arr, lambda_arr,
      child_by_first, records, record_index, mass)
    loss_total += loss
    events_total += events
  end

  raise "no eval events" if events_total == 0
  mean_nll = loss_total / events_total.to_f
  {mean_nll, Math.exp(mean_nll), events_total}
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

# One window: for each target position × each backoff depth, do an
# independent rooted GRU walk and accumulate the per-(target,j) loss.
# Then one Adam step per window.
def train_window(
  params : RecurParams,
  adam : AdamState,
  grads : RecurParams,
  tokens : Array(Int32),
  window_start : Int32, n_targets : Int32, d_max : Int32,
  x_proj_z : Array(Float64), x_proj_r : Array(Float64), x_proj_n : Array(Float64),
  walk_states_arr : Array(Array(Float64)),
  walk_z_arr : Array(Array(Float64)),
  walk_r_arr : Array(Array(Float64)),
  walk_n_arr : Array(Array(Float64)),
  walk_m_arr : Array(Array(Float64)),
  per_walk_logits_arr : Array(Array(Float64)),
  per_walk_p_target : Array(Float64),
  per_walk_dh_seed : Array(Array(Float64)),
  dh_per_step : Array(Float64),
  dh_cur : Array(Float64), dh_prev : Array(Float64),
  dz_pre : Array(Float64), dr_pre : Array(Float64), dn_pre : Array(Float64), dm : Array(Float64),
  grad_logits : Array(Float64),
  mass_arr : Array(Float64), log_mass_arr : Array(Float64),
  score_arr : Array(Float64), lambda_arr : Array(Float64),
  grad_score_arr : Array(Float64),
  child_by_first : Hash(Int32, Hash(Int32, Int32)),
  records : Array(OverlayRecord),
  record_index : Hash(Int32, Int32),
  mass : Array(Float64),
  lr : Float64, beta1 : Float64, beta2 : Float64, eps : Float64,
) : {Float64, Int64}
  grads.zero!
  loss_total = 0.0
  events = 0_i64

  d = params.d_model
  v = params.vocab_size
  di = d.to_i64
  vi = v.to_i64

  n_targets.times do |i|
    target = window_start + d_max + i
    target_token = tokens[target]

    # Forward: d_max GRU walks.
    j = 1
    while j <= d_max
      walk_start = target - j
      forward_chunk(params, x_proj_z, x_proj_r, x_proj_n,
        tokens, walk_start, j,
        walk_states_arr[j - 1], walk_z_arr[j - 1], walk_r_arr[j - 1], walk_n_arr[j - 1], walk_m_arr[j - 1])
      j += 1
    end

    # Look up mass per backoff prefix.
    j = 1
    while j <= d_max
      walk_start = target - j
      m = mass_at_prefix(tokens, walk_start, j, child_by_first, records, record_index, mass)
      mass_arr[j - 1] = m
      log_mass_arr[j - 1] = Math.log(m)
      j += 1
    end

    # Score: score_j = lambda_proj · h_j + lambda_alpha · log(m_j)
    j = 1
    while j <= d_max
      h_off = j * d
      s = 0.0
      d.times { |k| s += params.lambda_proj[k] * walk_states_arr[j - 1][h_off + k] }
      s += params.lambda_alpha[0] * log_mass_arr[j - 1]
      score_arr[j - 1] = s
      j += 1
    end

    # Softmax over j → λ_j
    max_score = score_arr[0]
    jj = 1
    while jj < d_max
      max_score = score_arr[jj] if score_arr[jj] > max_score
      jj += 1
    end
    sum_exp = 0.0
    d_max.times do |k|
      e = Math.exp(score_arr[k] - max_score)
      lambda_arr[k] = e
      sum_exp += e
    end
    d_max.times { |k| lambda_arr[k] /= sum_exp }

    # Per-walk head: logits → softmax → p_j(target)
    j = 1
    while j <= d_max
      h_off = j * d
      v.times { |tok| per_walk_logits_arr[j - 1][tok] = params.c_o[tok] }
      LibCBLAS.dgemv(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS,
        vi, di, 1.0, params.w_o.to_unsafe, di,
        walk_states_arr[j - 1].to_unsafe + h_off, 1_i64,
        1.0, per_walk_logits_arr[j - 1].to_unsafe, 1_i64)
      softmax_logits!(per_walk_logits_arr[j - 1])
      per_walk_p_target[j - 1] = per_walk_logits_arr[j - 1][target_token]
      j += 1
    end

    # Mixture: P = Σ_j λ_j p_j(target)
    big_p = 0.0
    d_max.times { |k| big_p += lambda_arr[k] * per_walk_p_target[k] }
    big_p = 1e-30 if big_p < 1e-30
    loss = -Math.log(big_p)
    loss_total += loss
    events += 1_i64

    # Backward through per-walk logits:
    # grad_logits_j[v] = (λ_j p_j(target) / P) · (p_j[v] - δ_v_target)
    j = 1
    while j <= d_max
      weight = lambda_arr[j - 1] * per_walk_p_target[j - 1] / big_p
      v.times { |tok| grad_logits[tok] = weight * per_walk_logits_arr[j - 1][tok] }
      grad_logits[target_token] -= weight

      h_off = j * d
      v.times { |tok| grads.c_o[tok] += grad_logits[tok] }
      LibCBLAS.dger(CBLAS_ROW_MAJOR,
        vi, di, 1.0,
        grad_logits.to_unsafe, 1_i64,
        walk_states_arr[j - 1].to_unsafe + h_off, 1_i64,
        grads.w_o.to_unsafe, di)

      # dh_j_from_logits = W_o^T · grad_logits
      LibCBLAS.dgemv(CBLAS_ROW_MAJOR, CBLAS_TRANS,
        vi, di, 1.0,
        params.w_o.to_unsafe, di,
        grad_logits.to_unsafe, 1_i64,
        0.0, per_walk_dh_seed[j - 1].to_unsafe, 1_i64)
      j += 1
    end

    # Backward through λ-softmax: dL/dscore_k = λ_k (P - p_k(target)) / P
    d_max.times do |k|
      grad_score_arr[k] = lambda_arr[k] * (big_p - per_walk_p_target[k]) / big_p
    end

    # Score net gradients + dh_j_from_score = lambda_proj · grad_score_j (added).
    d_max.times do |k|
      gs = grad_score_arr[k]
      h_off = (k + 1) * d
      d.times { |kk| grads.lambda_proj[kk] += gs * walk_states_arr[k][h_off + kk] }
      grads.lambda_alpha[0] += gs * log_mass_arr[k]
      d.times { |kk| per_walk_dh_seed[k][kk] += gs * params.lambda_proj[kk] }
    end

    # BPTT through each walk using its dh_seed.
    j = 1
    while j <= d_max
      walk_start = target - j
      (j * d).times { |k| dh_per_step[k] = 0.0 }
      d.times { |k| dh_per_step[(j - 1) * d + k] = per_walk_dh_seed[j - 1][k] }
      backward_chunk(params, grads,
        tokens, walk_start, j,
        walk_states_arr[j - 1], walk_z_arr[j - 1], walk_r_arr[j - 1], walk_n_arr[j - 1], walk_m_arr[j - 1],
        dh_per_step,
        dh_cur, dh_prev, dz_pre, dr_pre, dn_pre, dm)
      j += 1
    end
  end

  return {0.0, 0_i64} if events == 0
  scale = 1.0 / events.to_f
  grads.each_array { |a| a.size.times { |i| a[i] *= scale } }
  adam_update!(params, grads, adam, lr, beta1, beta2, eps)
  {loss_total, events}
end

# One epoch: tile corpus with non-overlapping target ranges of length
# n_targets, random offset shift per epoch in [0, n_targets).
def train_epoch(
  params : RecurParams,
  adam : AdamState,
  tokens : Array(Int32),
  n_targets : Int32, d_max : Int32,
  child_by_first : Hash(Int32, Hash(Int32, Int32)),
  records : Array(OverlayRecord),
  record_index : Hash(Int32, Int32),
  mass : Array(Float64),
  rng : Random,
  lr : Float64, beta1 : Float64, beta2 : Float64, eps : Float64,
) : {Float64, Float64, Int64, Int32}
  d = params.d_model
  v = params.vocab_size
  window_len = n_targets + d_max
  raise "corpus too short: #{tokens.size} tokens for window_len=#{window_len}" if tokens.size < window_len + 1

  offset = rng.rand(n_targets)
  max_start = tokens.size - window_len - 1
  n_windows = (max_start - offset) // n_targets + 1
  n_windows = 0 if n_windows < 0

  grads = RecurParams.new(v, d)
  buf_len = (d_max + 1) * d

  # Per-walk forward buffers (one set per backoff depth 1..d_max).
  walk_states_arr = Array(Array(Float64)).new(d_max) { Array(Float64).new(buf_len, 0.0) }
  walk_z_arr = Array(Array(Float64)).new(d_max) { Array(Float64).new(buf_len, 0.0) }
  walk_r_arr = Array(Array(Float64)).new(d_max) { Array(Float64).new(buf_len, 0.0) }
  walk_n_arr = Array(Array(Float64)).new(d_max) { Array(Float64).new(buf_len, 0.0) }
  walk_m_arr = Array(Array(Float64)).new(d_max) { Array(Float64).new(buf_len, 0.0) }

  # Per-walk head + mixture scratch.
  per_walk_logits_arr = Array(Array(Float64)).new(d_max) { Array(Float64).new(v, 0.0) }
  per_walk_p_target = Array(Float64).new(d_max, 0.0)
  per_walk_dh_seed = Array(Array(Float64)).new(d_max) { Array(Float64).new(d, 0.0) }
  mass_arr = Array(Float64).new(d_max, 1.0)
  log_mass_arr = Array(Float64).new(d_max, 0.0)
  score_arr = Array(Float64).new(d_max, 0.0)
  lambda_arr = Array(Float64).new(d_max, 0.0)
  grad_score_arr = Array(Float64).new(d_max, 0.0)

  dh_per_step = Array(Float64).new(d_max * d, 0.0)
  dh_cur = Array(Float64).new(d, 0.0)
  dh_prev = Array(Float64).new(d, 0.0)
  dz_pre = Array(Float64).new(d, 0.0)
  dr_pre = Array(Float64).new(d, 0.0)
  dn_pre = Array(Float64).new(d, 0.0)
  dm = Array(Float64).new(d, 0.0)
  grad_logits = Array(Float64).new(v, 0.0)

  loss_total = 0.0
  events_total = 0_i64
  updates = 0

  n_windows.times do |i|
    window_start = offset + i * n_targets
    x_proj_z, x_proj_r, x_proj_n = compute_x_projs(params)
    loss, events = train_window(params, adam, grads,
      tokens, window_start, n_targets, d_max,
      x_proj_z, x_proj_r, x_proj_n,
      walk_states_arr, walk_z_arr, walk_r_arr, walk_n_arr, walk_m_arr,
      per_walk_logits_arr, per_walk_p_target, per_walk_dh_seed,
      dh_per_step, dh_cur, dh_prev,
      dz_pre, dr_pre, dn_pre, dm,
      grad_logits,
      mass_arr, log_mass_arr, score_arr, lambda_arr, grad_score_arr,
      child_by_first, records, record_index, mass,
      lr, beta1, beta2, eps)
    loss_total += loss
    events_total += events
    updates += 1
  end

  raise "no events in epoch" if events_total == 0
  mean_nll = loss_total / events_total.to_f
  {mean_nll, Math.exp(mean_nll), events_total, updates}
end

corpus_path = ""
vocab_source = ""
save_path = ""
load_path = ""
trie_dir = ""
eval_corpus_path = ""
d_model = 0
n_targets = 16
d_max = 8
epochs = 0
lr = 0.001
seed = 1_u64
checkpoint_every = 0
dry_run = false
beta1 = 0.9
beta2 = 0.999
eps = 1e-8

OptionParser.parse do |p|
  p.banner = "Usage: bin/agpt_train_overlay_gru --corpus PATH --d-model N --epochs N --lr F --seed N --save PATH [options]"
  p.on("--corpus PATH", "Training corpus (bytes)") { |v| corpus_path = v }
  p.on("--vocab-source PATH", "Vocab source file (defaults to --corpus)") { |v| vocab_source = v }
  p.on("--trie DIR", "Radix-trie dir (REQUIRED). Mass per prefix feeds the score net.") { |v| trie_dir = v }
  p.on("--d-model N", "Hidden dimension") { |v| d_model = v.to_i }
  p.on("--n-targets N", "Targets per window (default 16). Stride between window starts.") { |v| n_targets = v.to_i }
  p.on("--d-max N", "Maximum backoff depth (default 8). Trie depth analog.") { |v| d_max = v.to_i }
  p.on("--epochs N", "Training epochs") { |v| epochs = v.to_i }
  p.on("--lr F", "Adam learning rate (default 0.001)") { |v| lr = v.to_f }
  p.on("--seed N", "Initialization / shuffle seed (default 1)") { |v| seed = v.to_u64 }
  p.on("--save PATH", "Write checkpoint") { |v| save_path = v }
  p.on("--load PATH", "Resume checkpoint") { |v| load_path = v }
  p.on("--checkpoint-every N", "Write epoch checkpoints every N epochs") { |v| checkpoint_every = v.to_i }
  p.on("--eval-corpus PATH", "After training, run mixture eval on this corpus and report PPL") { |v| eval_corpus_path = v }
  p.on("--dry-run", "Load + report shape without training") { dry_run = true }
  p.on("-h", "--help", "Help") { puts p; exit 0 }
end

raise "--corpus required" if corpus_path.empty?
raise "--d-model must be > 0" if d_model <= 0
raise "--epochs must be >= 0" if epochs < 0
raise "--lr must be > 0" if lr <= 0.0
raise "--n-targets must be > 0" if n_targets <= 0
raise "--d-max must be > 0" if d_max <= 0
if save_path.empty? && !dry_run
  raise "--save required unless --dry-run"
end
vocab_source = corpus_path if vocab_source.empty?

vocab_size = vocab_size_for(vocab_source)
tokens = load_corpus_tokens(corpus_path, vocab_source)

raise "--trie required (mixture variant needs mass per prefix)" if trie_dir.empty?

records, mass, _entropy_unused, _max_id, child_by_first = load_trie_with_stats(trie_dir)
record_index = {} of Int32 => Int32
records.each_with_index { |r, idx| record_index[r.id] = idx }

window_len = n_targets + d_max
max_start = tokens.size - window_len - 1
n_windows = max_start // n_targets + 1

puts "AGPT overlay-GRU trainer (sample backoff stacks, openblas hot path)"
puts "  corpus: #{corpus_path}"
puts "  vocab_source: #{vocab_source}"
puts "  vocab_size: #{vocab_size}"
puts "  corpus_tokens: #{tokens.size}"
puts "  d_model: #{d_model}"
puts "  n_targets per window: #{n_targets}"
puts "  d_max (max backoff depth): #{d_max}"
puts "  window_length: #{window_len}"
puts "  windows_per_epoch: ~#{n_windows}"
puts "  walks_per_window: #{n_targets * d_max}  (= n_targets × d_max)"
puts "  updates_per_epoch: ~#{n_windows}  (one Adam step per window)"
puts "  trie: #{trie_dir}"
puts "  trie_records: #{records.size}"
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

rng = Random.new(seed ^ 0x6f76726c61795f_u64)

(start_epoch + 1).upto(start_epoch + epochs) do |epoch|
  t0 = Time.instant
  nll, ppl, trained_events, updates = train_epoch(params, adam, tokens, n_targets, d_max,
    child_by_first, records, record_index, mass,
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

if !eval_corpus_path.empty?
  eval_tokens = load_corpus_tokens(eval_corpus_path, vocab_source)
  puts "--- eval ---"
  puts "  eval_corpus: #{eval_corpus_path} (#{eval_tokens.size} tokens)"
  t0 = Time.instant
  eval_nll, eval_ppl, eval_events = eval_corpus(params, eval_tokens, n_targets, d_max,
    child_by_first, records, record_index, mass)
  wall = (Time.instant - t0).total_seconds
  puts "  eval_events: #{eval_events}"
  printf "  eval_nll: %.6f  eval_ppl: %.6f  wall: %.3fs\n", eval_nll, eval_ppl, wall
end
