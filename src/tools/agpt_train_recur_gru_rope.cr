require "option_parser"
require "../agpt"

# GRU+RoPE recurrence AGPT trainer. Same GRU as `agpt_train_recur_gru` but
# with rotary position encoding applied to the input-projection outputs
# (`R(pos) · U_? · emb(x)`). The hidden-state projections (`W_? · h`) are
# unchanged — RoPE only modulates how the input enters each gate, mirroring
# how transformer RoPE rotates Q/K projections rather than the embedding
# itself.
#
# RoPE convention here: position p ∈ {0, 1, …, D−1} where D is the trie's
# max_endpoint_depth. For an edge ending at depth E with k tokens, the
# edge tokens occupy positions (E−k) through (E−1). Same depth as the
# trie-walk position so the model sees consistent positional structure
# at training and evaluation.
#
# Per partition we precompute:
#   raw_x_proj_? = emb · U_?ᵀ          (V×d, same as plain GRU)
#   rotated_x_proj_?[pos, tok] = R(pos) · raw_x_proj_?[tok]   (D×V×d)
# Forward looks up rotated_x_proj_?[pos, tok]. Backward rotates the
# per-gate pre-activation gradient by R(pos)ᵀ for the dU_? and d_emb
# accumulations; W_? / b_? / dh_prev updates are unchanged.
#
# Magic 'ACGP' (AGPT GRU + Position).

include MicroGPT::AGPT

MAGIC_RECUR_GRU_ROPE = 0x50474341_u32 # 'ACGP'
VERSION_RECUR        =          1_i32
ROPE_BASE            =      10000.0_f64

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
    io.write_bytes(MAGIC_RECUR_GRU_ROPE, IO::ByteFormat::LittleEndian)
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
    raise "bad GRU+RoPE recur checkpoint magic in #{path}" unless magic == MAGIC_RECUR_GRU_ROPE
    version = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
    raise "unsupported GRU+RoPE recur checkpoint version #{version}" unless version == 1
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

# ─── RoPE helpers ─────────────────────────────────────────────────────────────

# cos/sin table sized [max_pos, d/2]. theta_k = base^(-2k/d).
def build_rope_tables(d_model : Int32, max_pos : Int32) : {Array(Float64), Array(Float64)}
  raise "d_model must be even for RoPE" if d_model.odd?
  d2 = d_model // 2
  cos_table = Array(Float64).new(max_pos * d2, 0.0)
  sin_table = Array(Float64).new(max_pos * d2, 0.0)
  max_pos.times do |pos|
    d2.times do |k|
      theta_k = ROPE_BASE ** (-2.0 * k.to_f / d_model.to_f)
      angle = pos.to_f * theta_k
      cos_table[pos * d2 + k] = Math.cos(angle)
      sin_table[pos * d2 + k] = Math.sin(angle)
    end
  end
  {cos_table, sin_table}
end

# Apply R(pos) to vec[off..off+d_model-1] in place (the forward rotation).
# Pairs (2k, 2k+1): (v_i, v_j) → (c·v_i − s·v_j, s·v_i + c·v_j).
@[AlwaysInline]
def rope_rotate!(vec : Array(Float64), off : Int32, pos : Int32,
                 cos_table : Array(Float64), sin_table : Array(Float64), d_model : Int32)
  d2 = d_model // 2
  base = pos * d2
  d2.times do |k|
    i = off + 2 * k
    j = off + 2 * k + 1
    c = cos_table[base + k]
    s = sin_table[base + k]
    vi = vec[i]
    vj = vec[j]
    vec[i] = vi * c - vj * s
    vec[j] = vi * s + vj * c
  end
end

# Apply R(pos)ᵀ to vec[off..off+d_model-1] in place (the inverse rotation).
@[AlwaysInline]
def rope_rotate_inverse!(vec : Array(Float64), off : Int32, pos : Int32,
                         cos_table : Array(Float64), sin_table : Array(Float64), d_model : Int32)
  d2 = d_model // 2
  base = pos * d2
  d2.times do |k|
    i = off + 2 * k
    j = off + 2 * k + 1
    c = cos_table[base + k]
    s = sin_table[base + k]
    vi = vec[i]
    vj = vec[j]
    vec[i] = c * vi + s * vj
    vec[j] = -s * vi + c * vj
  end
end

# Precompute raw_x_proj_? = emb · U_?ᵀ (V×d), then rotate per position into
# rotated_x_proj_?[pos*V*d + tok*d + j]. Returns three (max_depth × V × d)
# tables.
def compute_rotated_x_projs(
  params : RecurParams, max_depth : Int32,
  cos_table : Array(Float64), sin_table : Array(Float64),
) : {Array(Float64), Array(Float64), Array(Float64)}
  v = params.vocab_size
  d = params.d_model
  vi = v.to_i64
  di = d.to_i64
  raw_z = Array(Float64).new(v * d, 0.0)
  raw_r = Array(Float64).new(v * d, 0.0)
  raw_n = Array(Float64).new(v * d, 0.0)
  LibCBLAS.dgemm(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_TRANS,
    vi, di, di, 1.0, params.emb.to_unsafe, di, params.u_z.to_unsafe, di, 0.0, raw_z.to_unsafe, di)
  LibCBLAS.dgemm(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_TRANS,
    vi, di, di, 1.0, params.emb.to_unsafe, di, params.u_r.to_unsafe, di, 0.0, raw_r.to_unsafe, di)
  LibCBLAS.dgemm(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_TRANS,
    vi, di, di, 1.0, params.emb.to_unsafe, di, params.u_n.to_unsafe, di, 0.0, raw_n.to_unsafe, di)

  rot_z = Array(Float64).new(max_depth * v * d, 0.0)
  rot_r = Array(Float64).new(max_depth * v * d, 0.0)
  rot_n = Array(Float64).new(max_depth * v * d, 0.0)
  max_depth.times do |pos|
    pos_off = pos * v * d
    v.times do |tok|
      src = tok * d
      dst = pos_off + tok * d
      # Copy raw rows into rot tables, then rotate in-place.
      d.times do |j|
        rot_z[dst + j] = raw_z[src + j]
        rot_r[dst + j] = raw_r[src + j]
        rot_n[dst + j] = raw_n[src + j]
      end
      rope_rotate!(rot_z, dst, pos, cos_table, sin_table, d)
      rope_rotate!(rot_r, dst, pos, cos_table, sin_table, d)
      rope_rotate!(rot_n, dst, pos, cos_table, sin_table, d)
    end
  end
  {rot_z, rot_r, rot_n}
end

# Forward edge. position_offset = trie depth of parent (= endpoint_depth − edge_len).
# Token at edge index k is at RoPE position (position_offset + k).
def forward_edge(
  params : RecurParams,
  x_proj_z : Array(Float64), x_proj_r : Array(Float64), x_proj_n : Array(Float64),
  endpoint_states : Array(Float64),
  parent_id : Int32,
  edge : Array(Int32),
  position_offset : Int32,
  vocab_size : Int32,
  states : Array(Float64),
  z_buf : Array(Float64), r_buf : Array(Float64), n_buf : Array(Float64), m_buf : Array(Float64),
)
  d = params.d_model
  di = d.to_i64
  if parent_id != 0
    parent_off = parent_id * d
    d.times { |j| states[j] = endpoint_states[parent_off + j] }
  else
    d.times { |j| states[j] = 0.0 }
  end
  edge.each_with_index do |tok, edge_idx|
    pos = position_offset + edge_idx
    prev_off = edge_idx * d
    cur_off = (edge_idx + 1) * d
    x_base = (pos * vocab_size + tok) * d

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
      vi, di, 1.0, params.w_o.to_unsafe, di,
      endpoint_states.to_unsafe + state_off, 1_i64,
      1.0, logits.to_unsafe, 1_i64)
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
      vi, di, 1.0,
      grad_logits.to_unsafe, 1_i64,
      endpoint_states.to_unsafe + state_off, 1_i64,
      grads.w_o.to_unsafe, di)
    LibCBLAS.dgemv(CBLAS_ROW_MAJOR, CBLAS_TRANS,
      vi, di, 1.0,
      params.w_o.to_unsafe, di,
      grad_logits.to_unsafe, 1_i64,
      1.0, dh.to_unsafe + state_off, 1_i64)
  end

  {loss, events}
end

def train_batch(
  params : RecurParams,
  adam : AdamState,
  records : Array(RecurRecord),
  records_desc : Array(RecurRecord),
  endpoint_states : Array(Float64),
  x_proj_z : Array(Float64), x_proj_r : Array(Float64), x_proj_n : Array(Float64),
  cos_table : Array(Float64), sin_table : Array(Float64), max_depth : Int32,
  lr : Float64, beta1 : Float64, beta2 : Float64, eps : Float64,
) : {Float64, Int64}
  d = params.d_model
  di = d.to_i64
  vocab_size = params.vocab_size
  n_state = endpoint_states.size
  edge_states = [] of Float64
  z_buf = [] of Float64
  r_buf = [] of Float64
  n_buf = [] of Float64
  m_buf = [] of Float64

  ensure_buffers = ->(edge_len : Int32) {
    needed = (edge_len + 1) * d
    if edge_states.size != needed
      edge_states = Array(Float64).new(needed, 0.0)
      z_buf = Array(Float64).new(needed, 0.0)
      r_buf = Array(Float64).new(needed, 0.0)
      n_buf = Array(Float64).new(needed, 0.0)
      m_buf = Array(Float64).new(needed, 0.0)
    end
  }

  records.each do |r|
    ensure_buffers.call(r.edge_tokens.size)
    pos_off = r.endpoint_depth - r.edge_tokens.size
    forward_edge(params, x_proj_z, x_proj_r, x_proj_n,
      endpoint_states, r.parent_id, r.edge_tokens, pos_off, vocab_size,
      edge_states, z_buf, r_buf, n_buf, m_buf)
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
  dz_pre = Array(Float64).new(d, 0.0)
  dr_pre = Array(Float64).new(d, 0.0)
  dn_pre = Array(Float64).new(d, 0.0)
  # Rotated copies for U_? and emb gradient flow.
  dz_rot = Array(Float64).new(d, 0.0)
  dr_rot = Array(Float64).new(d, 0.0)
  dn_rot = Array(Float64).new(d, 0.0)
  dm = Array(Float64).new(d, 0.0)
  records_desc.each do |r|
    next if r.edge_tokens.empty?
    parent_off = r.parent_id * d
    ensure_buffers.call(r.edge_tokens.size)
    pos_off = r.endpoint_depth - r.edge_tokens.size
    forward_edge(params, x_proj_z, x_proj_r, x_proj_n,
      endpoint_states, r.parent_id, r.edge_tokens, pos_off, vocab_size,
      edge_states, z_buf, r_buf, n_buf, m_buf)

    state_off = r.id * d
    d.times { |j| dh_cur[j] = dh[state_off + j] }

    edge_idx = r.edge_tokens.size - 1
    while edge_idx >= 0
      tok = r.edge_tokens[edge_idx]
      pos = pos_off + edge_idx
      prev_off_local = edge_idx * d
      cur_off_local = (edge_idx + 1) * d

      d.times do |j|
        zj = z_buf[cur_off_local + j]
        nj = n_buf[cur_off_local + j]
        hp = edge_states[prev_off_local + j]
        dhj = dh_cur[j]
        dzj = dhj * (nj - hp)
        dnj = dhj * zj
        dn_pre[j] = dnj * (1.0 - nj * nj)
        dz_pre[j] = dzj * zj * (1.0 - zj)
        dh_prev[j] = dhj * (1.0 - zj)
      end

      LibCBLAS.dgemv(CBLAS_ROW_MAJOR, CBLAS_TRANS,
        di, di, 1.0,
        params.w_n.to_unsafe, di,
        dn_pre.to_unsafe, 1_i64,
        0.0, dm.to_unsafe, 1_i64)

      d.times do |j|
        rj = r_buf[cur_off_local + j]
        hp = edge_states[prev_off_local + j]
        dmj = dm[j]
        drj = dmj * hp
        dh_prev[j] += dmj * rj
        dr_pre[j] = drj * rj * (1.0 - rj)
      end

      LibCBLAS.dgemv(CBLAS_ROW_MAJOR, CBLAS_TRANS,
        di, di, 1.0,
        params.w_r.to_unsafe, di,
        dr_pre.to_unsafe, 1_i64,
        1.0, dh_prev.to_unsafe, 1_i64)
      LibCBLAS.dgemv(CBLAS_ROW_MAJOR, CBLAS_TRANS,
        di, di, 1.0,
        params.w_z.to_unsafe, di,
        dz_pre.to_unsafe, 1_i64,
        1.0, dh_prev.to_unsafe, 1_i64)

      d.times do |j|
        grads.b_z[j] += dz_pre[j]
        grads.b_r[j] += dr_pre[j]
        grads.b_n[j] += dn_pre[j]
      end

      LibCBLAS.dger(CBLAS_ROW_MAJOR, di, di, 1.0,
        dz_pre.to_unsafe, 1_i64,
        edge_states.to_unsafe + prev_off_local, 1_i64,
        grads.w_z.to_unsafe, di)
      LibCBLAS.dger(CBLAS_ROW_MAJOR, di, di, 1.0,
        dr_pre.to_unsafe, 1_i64,
        edge_states.to_unsafe + prev_off_local, 1_i64,
        grads.w_r.to_unsafe, di)
      LibCBLAS.dger(CBLAS_ROW_MAJOR, di, di, 1.0,
        dn_pre.to_unsafe, 1_i64,
        m_buf.to_unsafe + cur_off_local, 1_i64,
        grads.w_n.to_unsafe, di)

      # Rotate dz_pre, dr_pre, dn_pre by R(pos)ᵀ for U_? and emb updates.
      d.times do |j|
        dz_rot[j] = dz_pre[j]
        dr_rot[j] = dr_pre[j]
        dn_rot[j] = dn_pre[j]
      end
      rope_rotate_inverse!(dz_rot, 0, pos, cos_table, sin_table, d)
      rope_rotate_inverse!(dr_rot, 0, pos, cos_table, sin_table, d)
      rope_rotate_inverse!(dn_rot, 0, pos, cos_table, sin_table, d)

      emb_base = tok * d
      LibCBLAS.dger(CBLAS_ROW_MAJOR, di, di, 1.0,
        dz_rot.to_unsafe, 1_i64,
        params.emb.to_unsafe + emb_base, 1_i64,
        grads.u_z.to_unsafe, di)
      LibCBLAS.dger(CBLAS_ROW_MAJOR, di, di, 1.0,
        dr_rot.to_unsafe, 1_i64,
        params.emb.to_unsafe + emb_base, 1_i64,
        grads.u_r.to_unsafe, di)
      LibCBLAS.dger(CBLAS_ROW_MAJOR, di, di, 1.0,
        dn_rot.to_unsafe, 1_i64,
        params.emb.to_unsafe + emb_base, 1_i64,
        grads.u_n.to_unsafe, di)

      LibCBLAS.dgemv(CBLAS_ROW_MAJOR, CBLAS_TRANS,
        di, di, 1.0,
        params.u_z.to_unsafe, di,
        dz_rot.to_unsafe, 1_i64,
        1.0, grads.emb.to_unsafe + emb_base, 1_i64)
      LibCBLAS.dgemv(CBLAS_ROW_MAJOR, CBLAS_TRANS,
        di, di, 1.0,
        params.u_r.to_unsafe, di,
        dr_rot.to_unsafe, 1_i64,
        1.0, grads.emb.to_unsafe + emb_base, 1_i64)
      LibCBLAS.dgemv(CBLAS_ROW_MAJOR, CBLAS_TRANS,
        di, di, 1.0,
        params.u_n.to_unsafe, di,
        dn_rot.to_unsafe, 1_i64,
        1.0, grads.emb.to_unsafe + emb_base, 1_i64)

      dh_cur, dh_prev = dh_prev, dh_cur
      edge_idx -= 1
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
  cos_table : Array(Float64), sin_table : Array(Float64), max_depth : Int32,
  lr : Float64, beta1 : Float64, beta2 : Float64, eps : Float64,
) : {Float64, Float64, Int64, Int32}
  loss_total = 0.0
  events_total = 0_i64
  updates = 0

  partitions.each do |records|
    next if records.empty?
    records_desc = records.sort_by { |r| {-r.endpoint_depth, -r.id} }
    x_proj_z, x_proj_r, x_proj_n = compute_rotated_x_projs(params, max_depth, cos_table, sin_table)
    loss, events = train_batch(params, adam, records, records_desc, endpoint_states,
      x_proj_z, x_proj_r, x_proj_n, cos_table, sin_table, max_depth,
      lr, beta1, beta2, eps)
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
  p.banner = "Usage: bin/agpt_train_recur_gru_rope --trie DIR --d-model N --epochs N --lr F --seed N --save PATH [options]"
  p.on("--trie DIR", "Global radix trie directory") { |v| trie_dir = v }
  p.on("--d-model N", "Hidden dimension (must be even)") { |v| d_model = v.to_i }
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
raise "--d-model must be even for RoPE" if d_model.odd?
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
rope_positions = max_endpoint_depth  # positions 0 .. max_endpoint_depth-1

puts "AGPT GRU+RoPE recurrent trainer (openblas hot path)"
puts "  trie: #{trie_dir}"
puts "  radix_records: #{records.size}#{max_nodes > 0 ? " (truncated)" : ""}"
puts "  expanded_states: #{expanded_chars}"
puts "  loss_events: #{events}"
puts "  partition_depth: #{partition_depth}"
puts "  partitions: #{partitions.size}"
puts "  vocab_size: #{reader.vocab_size}"
puts "  d_model: #{d_model}"
puts "  max_endpoint_depth: #{max_endpoint_depth}"
puts "  rope_positions: #{rope_positions} (0..#{rope_positions - 1})"
puts "  optimizer: adam (lr=#{lr}, beta1=#{beta1}, beta2=#{beta2}, eps=#{eps})"

if dry_run
  param_count = RecurParams.new(reader.vocab_size, d_model).total_floats
  state_mb = reader.radix_count.to_i64 * d_model.to_i64 * 8_i64 / 1024.0 / 1024.0
  puts "  params: #{param_count} float64"
  puts "  endpoint_state_memory: #{state_mb.round(2)} MB"
  exit 0
end

cos_table, sin_table = build_rope_tables(d_model, rope_positions)

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
  nll, ppl, trained_events, updates = train_epoch(params, adam, partitions, endpoint_states,
    cos_table, sin_table, rope_positions, lr, beta1, beta2, eps)
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
