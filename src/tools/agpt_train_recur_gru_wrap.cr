require "option_parser"
require "../agpt"

# GRU-recurrence AGPT trainer with k=2 wrap-around (detached bridge).
#
# Cycle 1: standard trie walk from root with h_init=0 (same as agpt_train_recur_gru).
# Cycle 2: for each corpus position p with cap1 = d-gram at p, walk the trie
#          path of cap2 (the d-gram at p+d) starting from h_init = stop_gradient(h_cap1).
#          Per-step one-hot CE against the corpus's actual next char.
#
# Same parameters (one set of GRU weights) updated by BOTH cycles' gradients
# accumulated together; single Adam step per partition. Stop-gradient at the
# bridge: cycle 2's loss does NOT propagate gradient through h_cap1 back into
# cycle 1's parameters via that path — but the SAME weights see gradient from
# cycle 1's forward and cycle 2's forward independently.
#
# Magic 'ACGW'. New CLI: --corpus PATH (the training corpus to build the
# corpus-position → cap1 index from).
#
# STAGE 1: infrastructure only. Reads corpus, builds cap1-by-position index,
# reports stats. Cycle 2 forward/backward NOT YET IMPLEMENTED — the trainer
# compiles, runs, and produces identical results to plain GRU. Cycle 2 added
# in subsequent stages.

include MicroGPT::AGPT

MAGIC_RECUR_GRU_WRAP = 0x57474341_u32 # 'ACGW'
VERSION_RECUR        =          1_i32

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
    # Biases stay 0.0; gates start at σ(0) = 0.5 (balanced update/keep).
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
    io.write_bytes(MAGIC_RECUR_GRU_WRAP, IO::ByteFormat::LittleEndian)
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
    raise "bad GRU+Wrap recur checkpoint magic in #{path}" unless magic == MAGIC_RECUR_GRU_WRAP
    version = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
    raise "unsupported GRU+Wrap recur checkpoint version #{version}" unless version == 1
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

# Read training corpus from path and tokenize using the same vocab the trie
# was built with. Returns array of token ids, length = corpus byte count.
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

# For each corpus position p in [0, corpus_length - d], find the trie record
# that represents corpus[p..p+d-1]. Returns array where cap1_by_pos[p] is the
# record id of the cap (depth-d trie node). For positions where the d-gram
# doesn't terminate at a depth-d node (e.g., near corpus end, or unique
# subpaths terminating earlier in radix-compressed trie), returns -1.
#
# The walk: from root, find child whose edge starts with corpus[p], match
# all edge tokens against corpus, advance. Continue until d chars consumed.
def build_cap1_by_pos(records : Array(RecurRecord), tokens : Array(Int32), d : Int32) : Array(Int32)
  # Index children by (parent_id, first_token).
  child_by_token = Hash(Int32, Hash(Int32, RecurRecord)).new do |h, k|
    h[k] = {} of Int32 => RecurRecord
  end
  records.each do |r|
    child_by_token[r.parent_id][r.edge_tokens[0]] = r
  end

  n = tokens.size
  cap1_by_pos = Array(Int32).new(n, -1)
  n.times do |p|
    next if p + d > n
    parent_id = 0
    consumed = 0
    cap_id = -1
    while consumed < d
      kids = child_by_token[parent_id]?
      break if kids.nil?
      kid = kids[tokens[p + consumed]]?
      break if kid.nil?
      edge = kid.edge_tokens
      # All edge tokens must match (radix-compressed edge).
      matched_full = true
      e_idx = 0
      while e_idx < edge.size && consumed + e_idx < d
        if edge[e_idx] != tokens[p + consumed + e_idx]
          matched_full = false
          break
        end
        e_idx += 1
      end
      break unless matched_full
      # Edge fully consumed within remaining d budget?
      if e_idx < edge.size
        # Edge extends beyond d — shouldn't happen if trie was built at this max_depth
        # but defensive: skip this position
        break
      end
      consumed += edge.size
      parent_id = kid.id
      cap_id = kid.id if consumed == d
    end
    cap1_by_pos[p] = cap_id
  end
  cap1_by_pos
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

# Precompute the three V×d tables: x_proj_z[v, d] = emb[v, d] · U_z^T[d, d], etc.
# Recomputed at the start of each partition batch (params change after every Adam step).
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

# forward_edge — GRU step at each position. Writes h_new into states.
# Also writes z, r, ntilde gates and m = r⊙h_prev into parallel buffers
# for use in backward. Size of each gate buffer: (edge_len + 1) * d
# (position 0 unused).
def forward_edge(
  params : RecurParams,
  x_proj_z : Array(Float64), x_proj_r : Array(Float64), x_proj_n : Array(Float64),
  endpoint_states : Array(Float64),
  parent_id : Int32,
  edge : Array(Int32),
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
  edge.each_with_index do |tok, pos|
    prev_off = pos * d
    cur_off = (pos + 1) * d
    x_base = tok * d

    # z_pre = b_z + x_proj_z[tok]
    d.times { |j| z_buf[cur_off + j] = params.b_z[j] + x_proj_z[x_base + j] }
    # z_pre += W_z · h_prev
    LibCBLAS.dgemv(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS,
      di, di, 1.0, params.w_z.to_unsafe, di,
      states.to_unsafe + prev_off, 1_i64,
      1.0, z_buf.to_unsafe + cur_off, 1_i64)
    # z = sigmoid
    d.times { |j| z_buf[cur_off + j] = sigmoid(z_buf[cur_off + j]) }

    # r_pre = b_r + x_proj_r[tok]
    d.times { |j| r_buf[cur_off + j] = params.b_r[j] + x_proj_r[x_base + j] }
    # r_pre += W_r · h_prev
    LibCBLAS.dgemv(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS,
      di, di, 1.0, params.w_r.to_unsafe, di,
      states.to_unsafe + prev_off, 1_i64,
      1.0, r_buf.to_unsafe + cur_off, 1_i64)
    # r = sigmoid
    d.times { |j| r_buf[cur_off + j] = sigmoid(r_buf[cur_off + j]) }

    # m = r ⊙ h_prev
    d.times { |j| m_buf[cur_off + j] = r_buf[cur_off + j] * states[prev_off + j] }

    # n_pre = b_n + x_proj_n[tok]
    d.times { |j| n_buf[cur_off + j] = params.b_n[j] + x_proj_n[x_base + j] }
    # n_pre += W_n · m
    LibCBLAS.dgemv(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS,
      di, di, 1.0, params.w_n.to_unsafe, di,
      m_buf.to_unsafe + cur_off, 1_i64,
      1.0, n_buf.to_unsafe + cur_off, 1_i64)
    # ñ = tanh
    d.times { |j| n_buf[cur_off + j] = Math.tanh(n_buf[cur_off + j]) }

    # h_new = (1 - z) ⊙ h_prev + z ⊙ ñ
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

# Cycle 2 training: per-corpus-position walk of d steps from h_init=h_cap1
# (stop-gradient). Computes one-hot CE loss at each step and accumulates
# gradients into shared `grads`. dh_init is discarded (detached bridge).
#
# `positions` is the list of corpus positions in this partition (positions
# whose cap1 has first char matching this partition's first char). All cap1
# records for these positions have endpoint_states populated by cycle 1.
#
# Returns (loss_sum, events_count) for this partition.
def cycle_2_train(
  params : RecurParams,
  grads : RecurParams,
  endpoint_states : Array(Float64),
  corpus_tokens : Array(Int32),
  cap1_by_pos : Array(Int32),
  positions : Array(Int32),
  max_depth : Int32,
  x_proj_z : Array(Float64), x_proj_r : Array(Float64), x_proj_n : Array(Float64),
) : {Float64, Int64}
  d = params.d_model
  di = d.to_i64
  v = params.vocab_size
  vi = v.to_i64
  d_f = d.to_f
  n = corpus_tokens.size

  loss_sum = 0.0
  events_count = 0_i64

  # Buffers sized for d+1 states (h_0 = h_init through h_d).
  needed = (max_depth + 1) * d
  edge_states = Array(Float64).new(needed, 0.0)
  z_buf = Array(Float64).new(needed, 0.0)
  r_buf = Array(Float64).new(needed, 0.0)
  n_buf = Array(Float64).new(needed, 0.0)
  m_buf = Array(Float64).new(needed, 0.0)
  dh_per_step = Array(Float64).new(needed, 0.0)

  dh_cur = Array(Float64).new(d, 0.0)
  dh_prev = Array(Float64).new(d, 0.0)
  dz_pre = Array(Float64).new(d, 0.0)
  dr_pre = Array(Float64).new(d, 0.0)
  dn_pre = Array(Float64).new(d, 0.0)
  dm = Array(Float64).new(d, 0.0)

  logits = Array(Float64).new(v, 0.0)
  grad_logits = Array(Float64).new(v, 0.0)

  positions.each do |p|
    # Guards.
    next if p + 2 * max_depth >= n              # need corpus[p+2d] for last prediction target
    cap1_id = cap1_by_pos[p]
    next if cap1_id <= 0                        # 0 = root, also catches -1 (invalid)

    # h_0 = stop_gradient(h_cap1)  (just a copy of endpoint_states[cap1_id])
    cap1_off = cap1_id * d
    d.times { |j| edge_states[j] = endpoint_states[cap1_off + j] }

    # FORWARD: d steps consuming corpus[p+d..p+2d-1].
    max_depth.times do |k|
      tok = corpus_tokens[p + max_depth + k]
      prev_off = k * d
      cur_off = (k + 1) * d
      x_base = tok * d

      # z = sigmoid(b_z + x_proj_z[tok] + W_z h_prev)
      d.times { |j| z_buf[cur_off + j] = params.b_z[j] + x_proj_z[x_base + j] }
      LibCBLAS.dgemv(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS,
        di, di, 1.0, params.w_z.to_unsafe, di,
        edge_states.to_unsafe + prev_off, 1_i64,
        1.0, z_buf.to_unsafe + cur_off, 1_i64)
      d.times { |j| z_buf[cur_off + j] = sigmoid(z_buf[cur_off + j]) }

      # r = sigmoid(b_r + x_proj_r[tok] + W_r h_prev)
      d.times { |j| r_buf[cur_off + j] = params.b_r[j] + x_proj_r[x_base + j] }
      LibCBLAS.dgemv(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS,
        di, di, 1.0, params.w_r.to_unsafe, di,
        edge_states.to_unsafe + prev_off, 1_i64,
        1.0, r_buf.to_unsafe + cur_off, 1_i64)
      d.times { |j| r_buf[cur_off + j] = sigmoid(r_buf[cur_off + j]) }

      # m = r ⊙ h_prev
      d.times { |j| m_buf[cur_off + j] = r_buf[cur_off + j] * edge_states[prev_off + j] }

      # n = tanh(b_n + x_proj_n[tok] + W_n m)
      d.times { |j| n_buf[cur_off + j] = params.b_n[j] + x_proj_n[x_base + j] }
      LibCBLAS.dgemv(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS,
        di, di, 1.0, params.w_n.to_unsafe, di,
        m_buf.to_unsafe + cur_off, 1_i64,
        1.0, n_buf.to_unsafe + cur_off, 1_i64)
      d.times { |j| n_buf[cur_off + j] = Math.tanh(n_buf[cur_off + j]) }

      # h_new = (1-z) ⊙ h_prev + z ⊙ n
      d.times do |j|
        zj = z_buf[cur_off + j]
        edge_states[cur_off + j] = (1.0 - zj) * edge_states[prev_off + j] + zj * n_buf[cur_off + j]
      end
    end

    # HEAD LOSS at each step: at h_{k+1} predict corpus[p+d+k+1].
    # dh_per_step reset.
    needed.times { |i| dh_per_step[i] = 0.0 }
    max_depth.times do |k|
      cur_off = (k + 1) * d
      target = corpus_tokens[p + max_depth + k + 1]  # safe due to guard above

      # logits = c_o + W_o · h
      v.times { |tok| logits[tok] = params.c_o[tok] }
      LibCBLAS.dgemv(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS,
        vi, di, 1.0, params.w_o.to_unsafe, di,
        edge_states.to_unsafe + cur_off, 1_i64,
        1.0, logits.to_unsafe, 1_i64)
      softmax_logits!(logits)

      loss_sum -= Math.log(logits[target])
      events_count += 1

      # grad_logits = softmax - one_hot(target)
      v.times { |tok| grad_logits[tok] = logits[tok] }
      grad_logits[target] -= 1.0

      # Head grad accumulation.
      v.times { |tok| grads.c_o[tok] += grad_logits[tok] }
      LibCBLAS.dger(CBLAS_ROW_MAJOR,
        vi, di, 1.0,
        grad_logits.to_unsafe, 1_i64,
        edge_states.to_unsafe + cur_off, 1_i64,
        grads.w_o.to_unsafe, di)
      # dh at h_{k+1} += W_o^T · grad_logits
      LibCBLAS.dgemv(CBLAS_ROW_MAJOR, CBLAS_TRANS,
        vi, di, 1.0,
        params.w_o.to_unsafe, di,
        grad_logits.to_unsafe, 1_i64,
        1.0, dh_per_step.to_unsafe + cur_off, 1_i64)
    end

    # BACKWARD through cycle 2 (steps d-1 down to 0).
    k = max_depth - 1
    while k >= 0
      tok = corpus_tokens[p + max_depth + k]
      prev_off = k * d
      cur_off = (k + 1) * d

      # Pull current step's accumulated dh.
      d.times { |j| dh_cur[j] = dh_per_step[cur_off + j] }

      # GRU backward at this step (same as cycle 1's per-step backward).
      d.times do |j|
        zj = z_buf[cur_off + j]
        nj = n_buf[cur_off + j]
        hp = edge_states[prev_off + j]
        dhj = dh_cur[j]
        dzj = dhj * (nj - hp)
        dnj = dhj * zj
        dn_pre[j] = dnj * (1.0 - nj * nj)
        dz_pre[j] = dzj * zj * (1.0 - zj)
        dh_prev[j] = dhj * (1.0 - zj)
      end

      # dm = W_n^T · dn_pre
      LibCBLAS.dgemv(CBLAS_ROW_MAJOR, CBLAS_TRANS,
        di, di, 1.0,
        params.w_n.to_unsafe, di,
        dn_pre.to_unsafe, 1_i64,
        0.0, dm.to_unsafe, 1_i64)

      d.times do |j|
        rj = r_buf[cur_off + j]
        hp = edge_states[prev_off + j]
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
        edge_states.to_unsafe + prev_off, 1_i64,
        grads.w_z.to_unsafe, di)
      LibCBLAS.dger(CBLAS_ROW_MAJOR, di, di, 1.0,
        dr_pre.to_unsafe, 1_i64,
        edge_states.to_unsafe + prev_off, 1_i64,
        grads.w_r.to_unsafe, di)
      LibCBLAS.dger(CBLAS_ROW_MAJOR, di, di, 1.0,
        dn_pre.to_unsafe, 1_i64,
        m_buf.to_unsafe + cur_off, 1_i64,
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
        di, di, 1.0,
        params.u_z.to_unsafe, di,
        dz_pre.to_unsafe, 1_i64,
        1.0, grads.emb.to_unsafe + emb_base, 1_i64)
      LibCBLAS.dgemv(CBLAS_ROW_MAJOR, CBLAS_TRANS,
        di, di, 1.0,
        params.u_r.to_unsafe, di,
        dr_pre.to_unsafe, 1_i64,
        1.0, grads.emb.to_unsafe + emb_base, 1_i64)
      LibCBLAS.dgemv(CBLAS_ROW_MAJOR, CBLAS_TRANS,
        di, di, 1.0,
        params.u_n.to_unsafe, di,
        dn_pre.to_unsafe, 1_i64,
        1.0, grads.emb.to_unsafe + emb_base, 1_i64)

      # Add dh_prev contribution to previous step's accumulator.
      # At k=0 this would flow into dh_per_step[0..d] = dh w.r.t. h_init,
      # which is what we DISCARD (detached bridge). So only accumulate
      # when k > 0.
      if k > 0
        d.times { |j| dh_per_step[prev_off + j] += dh_prev[j] }
      end

      k -= 1
    end
  end

  {loss_sum, events_count}
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
  cycle2_positions : Array(Int32),
  corpus_tokens : Array(Int32),
  cap1_by_pos : Array(Int32),
  max_endpoint_depth : Int32,
  lr : Float64,
  beta1 : Float64,
  beta2 : Float64,
  eps : Float64,
) : {Float64, Int64, Float64, Int64}
  d = params.d_model
  di = d.to_i64
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
    forward_edge(params, x_proj_z, x_proj_r, x_proj_n,
      endpoint_states, r.parent_id, r.edge_tokens,
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
  dm = Array(Float64).new(d, 0.0)
  records_desc.each do |r|
    next if r.edge_tokens.empty?
    parent_off = r.parent_id * d
    ensure_buffers.call(r.edge_tokens.size)
    forward_edge(params, x_proj_z, x_proj_r, x_proj_n,
      endpoint_states, r.parent_id, r.edge_tokens,
      edge_states, z_buf, r_buf, n_buf, m_buf)

    state_off = r.id * d
    d.times { |j| dh_cur[j] = dh[state_off + j] }

    pos = r.edge_tokens.size - 1
    while pos >= 0
      tok = r.edge_tokens[pos]
      prev_off_local = pos * d
      cur_off_local = (pos + 1) * d

      # GRU backward at this position.
      #   h_new = (1 - z) ⊙ h_prev + z ⊙ ñ
      #   dz = dh_new ⊙ (ñ - h_prev)
      #   dñ = dh_new ⊙ z
      #   dh_prev (path A) = dh_new ⊙ (1 - z)
      #
      #   ñ = tanh(W_n · m + x_proj_n[tok] + b_n)
      #   dn_pre = dñ ⊙ (1 - ñ²)
      #   dm = W_n^T · dn_pre
      #
      #   m = r ⊙ h_prev
      #   dr = dm ⊙ h_prev
      #   dh_prev (path B) = dm ⊙ r
      #
      #   r = σ(W_r · h_prev + x_proj_r[tok] + b_r)
      #   dr_pre = dr ⊙ r ⊙ (1 - r)
      #   dh_prev (path C) = W_r^T · dr_pre
      #
      #   z = σ(W_z · h_prev + x_proj_z[tok] + b_z)
      #   dz_pre = dz ⊙ z ⊙ (1 - z)
      #   dh_prev (path D) = W_z^T · dz_pre

      d.times do |j|
        zj = z_buf[cur_off_local + j]
        nj = n_buf[cur_off_local + j]
        hp = edge_states[prev_off_local + j]
        dhj = dh_cur[j]
        # local grads for z and ñ
        dzj = dhj * (nj - hp)
        dnj = dhj * zj
        # tanh derivative for ñ
        dn_pre[j] = dnj * (1.0 - nj * nj)
        # sigmoid derivative for z
        dz_pre[j] = dzj * zj * (1.0 - zj)
        # path A: dh_prev = dh ⊙ (1 - z) (will be accumulated further below)
        dh_prev[j] = dhj * (1.0 - zj)
      end

      # dm = W_n^T · dn_pre  (scratch buffer)
      LibCBLAS.dgemv(CBLAS_ROW_MAJOR, CBLAS_TRANS,
        di, di, 1.0,
        params.w_n.to_unsafe, di,
        dn_pre.to_unsafe, 1_i64,
        0.0, dm.to_unsafe, 1_i64)

      # m = r ⊙ h_prev; dr = dm ⊙ h_prev; path B: dh_prev += dm ⊙ r
      # then dr_pre = dr ⊙ r ⊙ (1-r)
      d.times do |j|
        rj = r_buf[cur_off_local + j]
        hp = edge_states[prev_off_local + j]
        dmj = dm[j]
        drj = dmj * hp
        dh_prev[j] += dmj * rj
        dr_pre[j] = drj * rj * (1.0 - rj)
      end

      # path C: dh_prev += W_r^T · dr_pre
      LibCBLAS.dgemv(CBLAS_ROW_MAJOR, CBLAS_TRANS,
        di, di, 1.0,
        params.w_r.to_unsafe, di,
        dr_pre.to_unsafe, 1_i64,
        1.0, dh_prev.to_unsafe, 1_i64)
      # path D: dh_prev += W_z^T · dz_pre
      LibCBLAS.dgemv(CBLAS_ROW_MAJOR, CBLAS_TRANS,
        di, di, 1.0,
        params.w_z.to_unsafe, di,
        dz_pre.to_unsafe, 1_i64,
        1.0, dh_prev.to_unsafe, 1_i64)

      # Bias gradients.
      d.times do |j|
        grads.b_z[j] += dz_pre[j]
        grads.b_r[j] += dr_pre[j]
        grads.b_n[j] += dn_pre[j]
      end

      # Weight gradients (rank-1 outer products).
      #   W_z += outer(dz_pre, h_prev),  W_r += outer(dr_pre, h_prev)
      #   W_n += outer(dn_pre, m)        U_z/U_r/U_n += outer(d?_pre, emb[tok])
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

      # Embedding gradient: U_z^T · dz_pre + U_r^T · dr_pre + U_n^T · dn_pre
      LibCBLAS.dgemv(CBLAS_ROW_MAJOR, CBLAS_TRANS,
        di, di, 1.0,
        params.u_z.to_unsafe, di,
        dz_pre.to_unsafe, 1_i64,
        1.0, grads.emb.to_unsafe + emb_base, 1_i64)
      LibCBLAS.dgemv(CBLAS_ROW_MAJOR, CBLAS_TRANS,
        di, di, 1.0,
        params.u_r.to_unsafe, di,
        dr_pre.to_unsafe, 1_i64,
        1.0, grads.emb.to_unsafe + emb_base, 1_i64)
      LibCBLAS.dgemv(CBLAS_ROW_MAJOR, CBLAS_TRANS,
        di, di, 1.0,
        params.u_n.to_unsafe, di,
        dn_pre.to_unsafe, 1_i64,
        1.0, grads.emb.to_unsafe + emb_base, 1_i64)

      dh_cur, dh_prev = dh_prev, dh_cur
      pos -= 1
    end

    if r.parent_id != 0
      d.times { |j| dh[parent_off + j] += dh_cur[j] }
    end
  end

  # CYCLE 2: walk corpus positions in this partition, accumulating gradient
  # into the SAME grads buffer.
  cycle2_loss, cycle2_events = cycle_2_train(
    params, grads, endpoint_states, corpus_tokens, cap1_by_pos,
    cycle2_positions, max_endpoint_depth,
    x_proj_z, x_proj_r, x_proj_n)

  total_events = events + cycle2_events
  scale = 1.0 / total_events.to_f
  grads.each_array { |a| a.size.times { |i| a[i] *= scale } }
  adam_update!(params, grads, adam, lr, beta1, beta2, eps)
  {loss, events, cycle2_loss, cycle2_events}
end

def train_epoch(
  params : RecurParams,
  adam : AdamState,
  partitions : Array(Array(RecurRecord)),
  positions_by_partition : Array(Array(Int32)),
  endpoint_states : Array(Float64),
  corpus_tokens : Array(Int32),
  cap1_by_pos : Array(Int32),
  max_endpoint_depth : Int32,
  lr : Float64,
  beta1 : Float64,
  beta2 : Float64,
  eps : Float64,
) : {Float64, Float64, Int64, Int64, Int32}
  loss_total = 0.0
  events_total = 0_i64
  cycle2_loss_total = 0.0
  cycle2_events_total = 0_i64
  updates = 0

  partitions.each_with_index do |records, pi|
    next if records.empty?
    records_desc = records.sort_by { |r| {-r.endpoint_depth, -r.id} }
    x_proj_z, x_proj_r, x_proj_n = compute_x_projs(params)
    cycle2_positions = positions_by_partition[pi]? || ([] of Int32)
    loss, events, c2_loss, c2_events = train_batch(
      params, adam, records, records_desc, endpoint_states,
      x_proj_z, x_proj_r, x_proj_n,
      cycle2_positions, corpus_tokens, cap1_by_pos, max_endpoint_depth,
      lr, beta1, beta2, eps)
    loss_total += loss
    events_total += events
    cycle2_loss_total += c2_loss
    cycle2_events_total += c2_events
    updates += 1
  end

  total_events = events_total + cycle2_events_total
  raise "no loss events found" if total_events == 0
  mean_nll = (loss_total + cycle2_loss_total) / total_events.to_f
  {mean_nll, Math.exp(mean_nll), total_events, cycle2_events_total, updates}
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
corpus_path = ""
vocab_source = ""
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
  p.banner = "Usage: bin/agpt_train_recur_gru_wrap --trie DIR --corpus PATH --vocab-source PATH --d-model N --epochs N --lr F --seed N --save PATH [options]"
  p.on("--trie DIR", "Global radix trie directory") { |v| trie_dir = v }
  p.on("--corpus PATH", "Training corpus file (same one the trie was built from)") { |v| corpus_path = v }
  p.on("--vocab-source PATH", "Vocab source file (typically full data/input.txt; defaults to --corpus)") { |v| vocab_source = v }
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
raise "--corpus required" if corpus_path.empty?
vocab_source = corpus_path if vocab_source.empty?
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

# Stage 1: build the corpus-position → cap1-record-id index.
corpus_tokens = load_corpus_tokens(corpus_path, vocab_source)
cap1_by_pos = build_cap1_by_pos(records, corpus_tokens, max_endpoint_depth)
cap1_pos_total = cap1_by_pos.count { |id| id >= 0 }
cap2_pairs_available = 0
cap1_by_pos.each_with_index do |id, p|
  next unless id >= 0
  if p + 2 * max_endpoint_depth <= corpus_tokens.size
    cap2_id = cap1_by_pos[p + max_endpoint_depth]?
    cap2_pairs_available += 1 if cap2_id && cap2_id >= 0
  end
end

# Build positions_by_partition: which corpus positions go to which partition.
# At pd=1, partition by first char of cap1 (= corpus_tokens[p]). Map first-char
# token → partition index via the partition's root child's edge_tokens[0].
positions_by_partition = Array(Array(Int32)).new(partitions.size) { [] of Int32 }
first_char_to_partition_idx = Hash(Int32, Int32).new
partitions.each_with_index do |part, pi|
  next if part.empty?
  root_child = part.first  # root children are the first record in each partition (parent-ordered)
  next unless root_child.parent_id == 0
  first_char_to_partition_idx[root_child.edge_tokens[0]] = pi
end
corpus_tokens.each_with_index do |tok, p|
  # Need both cap1 and cap2 (and one more char for the final target).
  next if p + 2 * max_endpoint_depth >= corpus_tokens.size
  next if cap1_by_pos[p] <= 0
  next if cap1_by_pos[p + max_endpoint_depth] <= 0
  pi = first_char_to_partition_idx[tok]?
  next unless pi
  positions_by_partition[pi] << p
end
cycle2_total_positions = positions_by_partition.sum(&.size)

puts "AGPT GRU+Wrap recurrent trainer (openblas hot path) — STAGE 1 stub"
puts "  trie: #{trie_dir}"
puts "  corpus: #{corpus_path} (#{corpus_tokens.size} tokens)"
puts "  vocab_source: #{vocab_source}"
puts "  radix_records: #{records.size}#{max_nodes > 0 ? " (truncated)" : ""}"
puts "  expanded_states: #{expanded_chars}"
puts "  loss_events: #{events}"
puts "  partition_depth: #{partition_depth}"
puts "  partitions: #{partitions.size}"
puts "  vocab_size: #{reader.vocab_size}"
puts "  d_model: #{d_model}"
puts "  max_endpoint_depth: #{max_endpoint_depth}"
puts "  cap1_positions_indexed: #{cap1_pos_total} / #{corpus_tokens.size}"
puts "  cap2_pair_positions: #{cap2_pairs_available} (corpus positions with both cap1 and cap2 available)"
puts "  cycle2_positions: #{cycle2_total_positions} (assigned to a partition)"
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
  nll, ppl, total_events, c2_events, updates = train_epoch(
    params, adam, partitions, positions_by_partition, endpoint_states,
    corpus_tokens, cap1_by_pos, max_endpoint_depth,
    lr, beta1, beta2, eps)
  wall = (Time.instant - t0).total_seconds
  ck_msg = ""
  if checkpoint_every > 0 && epoch % checkpoint_every == 0
    ck = epoch_checkpoint_path(save_path, epoch)
    save_checkpoint(ck, params, adam, epoch, seed)
    ck_msg = " checkpoint=#{ck}"
  end
  printf "epoch %6d  nll %.6f  ppl %.6f  events %d (c2 %d)  updates %d  wall %.3fs  adam_step %d%s\n",
    epoch, nll, ppl, total_events, c2_events, updates, wall, adam.step, ck_msg
end

save_checkpoint(save_path, params, adam, start_epoch + epochs, seed)
puts "saved #{save_path} (epoch=#{start_epoch + epochs}, adam_step=#{adam.step})"
