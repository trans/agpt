require "option_parser"
require "../agpt"

include MicroGPT::AGPT

MAGIC_RECUR          = 0x52474341_u32 # 'ACGR'
MAGIC_RECUR_RMS_POST = 0x54474341_u32 # 'ACGT'
MAGIC_RECUR_RMS_PRE  = 0x50474341_u32 # 'ACGP'
MAGIC_RECUR_PHASE          = 0x51474341_u32 # 'ACGQ'
MAGIC_RECUR_PHASE_RMS_POST = 0x55474341_u32 # 'ACGU'
MAGIC_RECUR_PHASE_RMS_PRE  = 0x56474341_u32 # 'ACGV'
VERSION_RECUR =          1_i32
VERSION_RECUR_RMS =      1_i32
VERSION_RECUR_PHASE =    1_i32
EPS_NORM = 1e-6_f64

enum NormMode
  None
  RMSPost
  RMSPre
end

def rms_norm?(mode : NormMode) : Bool
  mode.rms_post? || mode.rms_pre?
end

class PhaseWeightTable
  @radix_to_substring : RadixToSubstring
  @position_table : PositionTable

  getter window_size : Int32

  def initialize(position_data_dir : String, radix_count : Int32)
    rts_path = File.join(position_data_dir, "prefix_radix_to_substring.bin")
    pos_path = File.join(position_data_dir, "prefix_position_table.bin")
    @radix_to_substring = File.open(rts_path, "rb") { |io| RadixToSubstring.read_from(io) }
    raise "position radix_count mismatch: #{@radix_to_substring.radix_count} != #{radix_count}" unless @radix_to_substring.radix_count == radix_count
    @position_table = File.open(pos_path, "rb") { |io| PositionTable.read_from(io) }
    @window_size = @position_table.window_size
  end

  def mass(radix_id : Int32, phase : Int32) : Int64
    sid = @radix_to_substring.substring_id_for(radix_id)
    return 0_i64 if sid < 0 || sid >= @position_table.substring_count
    pos = phase % @window_size
    pos += @window_size if pos < 0
    @position_table.bins(sid).each do |bin|
      bpos = bin.pos.to_i
      return bin.count.to_i64 if bpos == pos
      break if bpos > pos
    end
    0_i64
  end

  def epoch_phase(epoch_index : Int32, fixed_offset : Int32, shuffle : Bool, seed : UInt32) : Int32
    if fixed_offset >= 0
      phase = fixed_offset % @window_size
      phase += @window_size if phase < 0
      return phase
    end
    return epoch_index % @window_size unless shuffle

    cycle = epoch_index // @window_size
    within_cycle = epoch_index % @window_size
    h = mix_u32(seed ^ (cycle.to_u32 &* 0x9e3779b9_u32))
    offset = (h % @window_size.to_u32).to_i
    h = mix_u32(h ^ 0x85ebca6b_u32)
    stride = (h % @window_size.to_u32).to_i
    stride = 1 if stride <= 0
    while gcd(stride, @window_size) != 1
      stride += 1
      stride = 1 if stride >= @window_size
    end
    (offset + within_cycle * stride) % @window_size
  end

  private def mix_u32(x : UInt32) : UInt32
    y = x
    y ^= y >> 16
    y = y &* 0x7feb352d_u32
    y ^= y >> 15
    y = y &* 0x846ca68b_u32
    y ^ (y >> 16)
  end

  private def gcd(a : Int32, b : Int32) : Int32
    x = a.abs
    y = b.abs
    while y != 0
      x, y = y, x % y
    end
    x
  end
end

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

def record_mass(r : RecurRecord) : Int64
  total = 0_i64
  r.counts.each { |pair| total += pair[1].to_i64 }
  total
end

class SingletonBackoffTable
  @targets : Hash(UInt64, Int32)

  getter route_count : Int32
  getter unresolved_count : Int32

  def initialize(records : Array(RecurRecord), radix_count : Int32)
    @targets = {} of UInt64 => Int32
    @route_count = 0
    @unresolved_count = 0

    paths = Array(Array(Int32)?).new(radix_count) { nil }
    paths[0] = [] of Int32
    endpoint_by_path = {} of String => RecurRecord

    records.each do |r|
      parent_path = paths[r.parent_id] || [] of Int32
      path = parent_path + r.edge_tokens
      paths[r.id] = path
      endpoint_by_path[path_key(path)] = r
    end

    records.each do |r|
      next unless r.counts.size == 1 && record_mass(r) == 1
      full = paths[r.id] || next
      parent_len = full.size - r.edge_tokens.size

      r.edge_tokens.each_index do |pos|
        prefix_len = parent_len + pos + 1
        next if prefix_len <= 1
        prefix = full[0, prefix_len]
        target_id = find_backoff_target(prefix, endpoint_by_path)
        if target_id >= 0
          @targets[key(r.id, pos)] = target_id
          @route_count += 1
        else
          @unresolved_count += 1
        end
      end
    end
  end

  def target(record_id : Int32, edge_pos : Int32) : Int32
    @targets[key(record_id, edge_pos)]? || -1
  end

  private def find_backoff_target(prefix : Array(Int32), endpoint_by_path : Hash(String, RecurRecord)) : Int32
    drop = 1
    while drop < prefix.size
      suffix = prefix[drop, prefix.size - drop]
      if rec = endpoint_by_path[path_key(suffix)]?
        return rec.id if record_mass(rec) > 1
      end
      drop += 1
    end
    -1
  end

  private def key(record_id : Int32, edge_pos : Int32) : UInt64
    (record_id.to_u64 << 32) | edge_pos.to_u32.to_u64
  end

  private def path_key(path : Array(Int32)) : String
    path.join(",")
  end
end

class RecurParams
  getter vocab_size : Int32
  getter d_model : Int32
  getter phase_window : Int32
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

  def each_array(norm_mode : NormMode = NormMode::RMSPre, &block : Array(Float64) ->)
    yield @emb
    yield @phase_emb if @phase_window > 0
    yield @w_h
    yield @w_x
    yield @b
    yield @g if rms_norm?(norm_mode)
    yield @w_o
    yield @c_o
  end

  def total_floats(norm_mode : NormMode = NormMode::RMSPre) : Int32
    total = @emb.size + @phase_emb.size + @w_h.size + @w_x.size + @b.size + @w_o.size + @c_o.size
    total += @g.size if rms_norm?(norm_mode)
    total
  end

  def fill_random!(seed : UInt64)
    rng = Random.new(seed)
    scale_emb = 0.02
    scale_rec = 1.0 / Math.sqrt(@d_model.to_f)
    fill_normal!(@emb, rng, scale_emb)
    fill_normal!(@phase_emb, rng, scale_emb) if @phase_window > 0
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

  def initialize(vocab_size : Int32, d_model : Int32, phase_window : Int32 = 0)
    @m = RecurParams.new(vocab_size, d_model, phase_window)
    @v = RecurParams.new(vocab_size, d_model, phase_window)
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

def save_checkpoint(path : String, params : RecurParams, adam : AdamState, epoch : Int32, seed : UInt64, norm_mode : NormMode)
  ensure_parent_dir(path)
  File.open(path, "wb") do |io|
    magic =
      if params.phase_window > 0
        case norm_mode
        when NormMode::None then MAGIC_RECUR_PHASE
        when NormMode::RMSPost then MAGIC_RECUR_PHASE_RMS_POST
        when NormMode::RMSPre then MAGIC_RECUR_PHASE_RMS_PRE
        else raise "unsupported norm mode #{norm_mode}"
        end
      else
        case norm_mode
        when NormMode::None then MAGIC_RECUR
        when NormMode::RMSPost then MAGIC_RECUR_RMS_POST
        when NormMode::RMSPre then MAGIC_RECUR_RMS_PRE
        else raise "unsupported norm mode #{norm_mode}"
        end
      end
    io.write_bytes(magic, IO::ByteFormat::LittleEndian)
    version =
      if params.phase_window > 0
        VERSION_RECUR_PHASE
      else
        rms_norm?(norm_mode) ? VERSION_RECUR_RMS : VERSION_RECUR
      end
    io.write_bytes(version, IO::ByteFormat::LittleEndian)
    io.write_bytes(params.vocab_size, IO::ByteFormat::LittleEndian)
    io.write_bytes(params.d_model, IO::ByteFormat::LittleEndian)
    io.write_bytes(epoch, IO::ByteFormat::LittleEndian)
    io.write_bytes(adam.step, IO::ByteFormat::LittleEndian)
    io.write_bytes(seed, IO::ByteFormat::LittleEndian)
    io.write_bytes(params.phase_window, IO::ByteFormat::LittleEndian) if params.phase_window > 0
    params.each_array(norm_mode) { |a| write_f64_array(io, a) }
    adam.m.each_array(norm_mode) { |a| write_f64_array(io, a) }
    adam.v.each_array(norm_mode) { |a| write_f64_array(io, a) }
  end
end

def load_checkpoint(path : String, params : RecurParams, adam : AdamState, norm_mode : NormMode) : {Int32, UInt64}
  File.open(path, "rb") do |io|
    magic = io.read_bytes(UInt32, IO::ByteFormat::LittleEndian)
    checkpoint_norm = case magic
                      when MAGIC_RECUR then NormMode::None
                      when MAGIC_RECUR_RMS_POST then NormMode::RMSPost
                      when MAGIC_RECUR_RMS_PRE then NormMode::RMSPre
                      when MAGIC_RECUR_PHASE then NormMode::None
                      when MAGIC_RECUR_PHASE_RMS_POST then NormMode::RMSPost
                      when MAGIC_RECUR_PHASE_RMS_PRE then NormMode::RMSPre
                      else raise "bad recurrent checkpoint magic in #{path}"
                      end
    raise "checkpoint norm mismatch: #{checkpoint_norm} != #{norm_mode}" unless checkpoint_norm == norm_mode
    version = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
    checkpoint_phase = magic == MAGIC_RECUR_PHASE || magic == MAGIC_RECUR_PHASE_RMS_POST || magic == MAGIC_RECUR_PHASE_RMS_PRE
    expected_version = checkpoint_phase ? VERSION_RECUR_PHASE : (rms_norm?(checkpoint_norm) ? VERSION_RECUR_RMS : VERSION_RECUR)
    raise "unsupported recurrent checkpoint version #{version}" unless version == expected_version
    vocab_size = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
    d_model = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
    raise "checkpoint vocab mismatch: #{vocab_size} != #{params.vocab_size}" unless vocab_size == params.vocab_size
    raise "checkpoint d_model mismatch: #{d_model} != #{params.d_model}" unless d_model == params.d_model
    epoch = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
    adam.step = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
    seed = io.read_bytes(UInt64, IO::ByteFormat::LittleEndian)
    checkpoint_phase_window = checkpoint_phase ? io.read_bytes(Int32, IO::ByteFormat::LittleEndian) : 0
    raise "checkpoint phase_window mismatch: #{checkpoint_phase_window} != #{params.phase_window}" unless checkpoint_phase_window == params.phase_window
    params.each_array(norm_mode) { |a| read_f64_array(io, a) }
    adam.m.each_array(norm_mode) { |a| read_f64_array(io, a) }
    adam.v.each_array(norm_mode) { |a| read_f64_array(io, a) }
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
  # proj[v, d] = emb[v, d] * transpose(w_x[d, d])
  LibCBLAS.dgemm(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_TRANS,
    vi, di, di,
    1.0,
    params.emb.to_unsafe, di,
    params.w_x.to_unsafe, di,
    0.0,
    proj.to_unsafe, di)
  proj
end

def forward_edge(
  params : RecurParams,
  norm_mode : NormMode,
  x_proj : Array(Float64),
  endpoint_states : Array(Float64),
  backoff_table : SingletonBackoffTable?,
  record_id : Int32,
  parent_id : Int32,
  parent_depth : Int32,
  edge : Array(Int32),
  start_phase : Int32,
  states : Array(Float64),
  norm_inputs : Array(Float64)? = nil,
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
    d.times { |j| states[cur_off + j] = params.b[j] + x_proj[x_base + j] }
    if params.phase_window > 0
      phase = (start_phase + parent_depth + pos) % params.phase_window
      phase += params.phase_window if phase < 0
      phase_base = phase * d
      d.times { |j| states[cur_off + j] += params.phase_emb[phase_base + j] }
    end
    LibCBLAS.dgemv(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS,
      di, di,
      1.0,
      params.w_h.to_unsafe, di,
      states.to_unsafe + prev_off, 1_i64,
      1.0,
      states.to_unsafe + cur_off, 1_i64)
    if norm_mode.rms_pre?
      ni = norm_inputs || raise "norm_inputs required for RMSNorm"
      sumsq = 0.0
      d.times do |j|
        z = states[cur_off + j]
        ni[cur_off + j] = z
        sumsq += z * z
      end
      inv = 1.0 / Math.sqrt(sumsq / d.to_f + EPS_NORM)
      d.times { |j| states[cur_off + j] = Math.tanh(states[cur_off + j] * params.g[j] * inv) }
    else
      d.times { |j| states[cur_off + j] = Math.tanh(states[cur_off + j]) }
    end

    if norm_mode.rms_post?
      ni = norm_inputs || raise "norm_inputs required for RMSNorm"
      sumsq = 0.0
      d.times do |j|
        h = states[cur_off + j]
        ni[cur_off + j] = h
        sumsq += h * h
      end
      inv = 1.0 / Math.sqrt(sumsq / d.to_f + EPS_NORM)
      d.times { |j| states[cur_off + j] = states[cur_off + j] * params.g[j] * inv }
    end

    if table = backoff_table
      target_id = table.target(record_id, pos)
      if target_id >= 0
        target_off = target_id * d
        d.times { |j| states[cur_off + j] = endpoint_states[target_off + j] }
      end
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
  phase_weights : PhaseWeightTable?,
  phase : Int32,
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
    count_total = 0_i64
    r.counts.each { |pair| count_total += pair[1].to_i64 }
    next if count_total <= 0
    event_weight = phase_weights ? phase_weights.mass(r.id, phase) : count_total
    next if event_weight <= 0

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

    loss_scale = event_weight.to_f / count_total.to_f
    r.counts.each do |pair|
      tok = pair[0]
      cnt = pair[1].to_i64
      loss -= cnt.to_f * loss_scale * Math.log(logits[tok])
    end
    events += event_weight

    count_total_f = event_weight.to_f
    v.times do |tok|
      grad_logits[tok] = logits[tok] * count_total_f
    end
    r.counts.each do |pair|
      grad_logits[pair[0]] -= pair[1].to_f * loss_scale
    end

    v.times do |tok|
      g = grad_logits[tok]
      grads.c_o[tok] += g
    end
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
  norm_mode : NormMode,
  adam : AdamState,
  records : Array(RecurRecord),
  records_desc : Array(RecurRecord),
  endpoint_states : Array(Float64),
  x_proj : Array(Float64),
  lr : Float64,
  beta1 : Float64,
  beta2 : Float64,
  eps : Float64,
  phase_weights : PhaseWeightTable?,
  phase : Int32,
  backoff_table : SingletonBackoffTable?,
  backoff_stopgrad : Bool,
) : {Float64, Int64}
  d = params.d_model
  di = d.to_i64
  n_state = endpoint_states.size
  edge_states = [] of Float64
  edge_norm_inputs = [] of Float64

  records.each do |r|
    needed = (r.edge_tokens.size + 1) * d
    edge_states = Array(Float64).new(needed, 0.0) if edge_states.size != needed
    if rms_norm?(norm_mode)
      edge_norm_inputs = Array(Float64).new(needed, 0.0) if edge_norm_inputs.size != needed
      parent_depth = r.endpoint_depth - r.edge_tokens.size
      forward_edge(params, norm_mode, x_proj, endpoint_states, backoff_table, r.id, r.parent_id, parent_depth, r.edge_tokens, phase, edge_states, edge_norm_inputs)
    else
      parent_depth = r.endpoint_depth - r.edge_tokens.size
      forward_edge(params, norm_mode, x_proj, endpoint_states, backoff_table, r.id, r.parent_id, parent_depth, r.edge_tokens, phase, edge_states)
    end
    end_off = r.edge_tokens.size * d
    state_off = r.id * d
    d.times { |j| endpoint_states[state_off + j] = edge_states[end_off + j] }
  end

  grads = RecurParams.new(params.vocab_size, params.d_model, params.phase_window)
  dh = Array(Float64).new(n_state, 0.0)
  loss, events = add_loss_and_head_grads(params, grads, endpoint_states, dh, records, phase_weights, phase)
  return {0.0, 0_i64} if events == 0

  dh_cur = Array(Float64).new(d, 0.0)
  dh_prev = Array(Float64).new(d, 0.0)
  dz = Array(Float64).new(d, 0.0)
  records_desc.each do |r|
    next if r.edge_tokens.empty?
    parent_off = r.parent_id * d
    needed = (r.edge_tokens.size + 1) * d
    edge_states = Array(Float64).new(needed, 0.0) if edge_states.size != needed
    if rms_norm?(norm_mode)
      edge_norm_inputs = Array(Float64).new(needed, 0.0) if edge_norm_inputs.size != needed
      parent_depth = r.endpoint_depth - r.edge_tokens.size
      forward_edge(params, norm_mode, x_proj, endpoint_states, backoff_table, r.id, r.parent_id, parent_depth, r.edge_tokens, phase, edge_states, edge_norm_inputs)
    else
      parent_depth = r.endpoint_depth - r.edge_tokens.size
      forward_edge(params, norm_mode, x_proj, endpoint_states, backoff_table, r.id, r.parent_id, parent_depth, r.edge_tokens, phase, edge_states)
    end

    state_off = r.id * d
    d.times { |j| dh_cur[j] = dh[state_off + j] }

    pos = r.edge_tokens.size - 1
    while pos >= 0
      if table = backoff_table
        target_id = table.target(r.id, pos)
        if target_id >= 0
          unless backoff_stopgrad
            target_off = target_id * d
            d.times { |j| dh[target_off + j] += dh_cur[j] }
          end
          break
        end
      end

      tok = r.edge_tokens[pos]
      prev_off_local = pos * d
      cur_off_local = (pos + 1) * d
      if norm_mode.rms_pre?
        sum_dy_g_z = 0.0
        sumsq = 0.0
        d.times do |j|
          z = edge_norm_inputs[cur_off_local + j]
          h = edge_states[cur_off_local + j]
          dy = dh_cur[j] * (1.0 - h * h)
          sumsq += z * z
          sum_dy_g_z += dy * params.g[j] * z
        end
        inv = 1.0 / Math.sqrt(sumsq / d.to_f + EPS_NORM)
        inv3_over_d = inv * inv * inv / d.to_f
        d.times do |j|
          z = edge_norm_inputs[cur_off_local + j]
          h = edge_states[cur_off_local + j]
          dy = dh_cur[j] * (1.0 - h * h)
          grads.g[j] += dy * z * inv
          dz[j] = dy * params.g[j] * inv - z * inv3_over_d * sum_dy_g_z
          grads.b[j] += dz[j]
        end
      elsif norm_mode.rms_post?
        sum_dy_g_h = 0.0
        sumsq = 0.0
        d.times do |j|
          h_raw = edge_norm_inputs[cur_off_local + j]
          sumsq += h_raw * h_raw
          sum_dy_g_h += dh_cur[j] * params.g[j] * h_raw
        end
        inv = 1.0 / Math.sqrt(sumsq / d.to_f + EPS_NORM)
        inv3_over_d = inv * inv * inv / d.to_f
        d.times do |j|
          h_raw = edge_norm_inputs[cur_off_local + j]
          grads.g[j] += dh_cur[j] * h_raw * inv
          dh_raw = dh_cur[j] * params.g[j] * inv - h_raw * inv3_over_d * sum_dy_g_h
          dz[j] = dh_raw * (1.0 - h_raw * h_raw)
          grads.b[j] += dz[j]
        end
      else
        d.times do |j|
          h = edge_states[cur_off_local + j]
          dz[j] = dh_cur[j] * (1.0 - h * h)
          grads.b[j] += dz[j]
        end
      end

      if params.phase_window > 0
        phase_pos = (phase + (r.endpoint_depth - r.edge_tokens.size) + pos) % params.phase_window
        phase_pos += params.phase_window if phase_pos < 0
        phase_base = phase_pos * d
        d.times { |j| grads.phase_emb[phase_base + j] += dz[j] }
      end

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
  grads.each_array(norm_mode) { |a| a.size.times { |i| a[i] *= scale } }
  adam_update!(params, grads, adam, lr, beta1, beta2, eps, norm_mode)
  {loss, events}
end

def train_epoch(
  params : RecurParams,
  norm_mode : NormMode,
  adam : AdamState,
  partitions : Array(Array(RecurRecord)),
  endpoint_states : Array(Float64),
  lr : Float64,
  beta1 : Float64,
  beta2 : Float64,
  eps : Float64,
  phase_weights : PhaseWeightTable?,
  phase : Int32,
  backoff_table : SingletonBackoffTable?,
  backoff_stopgrad : Bool,
) : {Float64, Float64, Int64, Int32}
  loss_total = 0.0
  events_total = 0_i64
  updates = 0

  partitions.each do |records|
    next if records.empty?
    records_desc = records.sort_by { |r| {-r.endpoint_depth, -r.id} }
    x_proj = compute_x_proj(params)
    loss, events = train_batch(params, norm_mode, adam, records, records_desc, endpoint_states, x_proj, lr, beta1, beta2, eps, phase_weights, phase, backoff_table, backoff_stopgrad)
    next if events == 0
    loss_total += loss
    events_total += events
    updates += 1
  end

  raise "no loss events found in trie records" if events_total == 0
  mean_nll = loss_total / events_total.to_f
  {mean_nll, Math.exp(mean_nll), events_total, updates}
end

def adam_update!(params : RecurParams, grads : RecurParams, adam : AdamState, lr : Float64, beta1 : Float64, beta2 : Float64, eps : Float64, norm_mode : NormMode)
  adam.step += 1
  t = adam.step
  bc1 = 1.0 - beta1 ** t
  bc2 = 1.0 - beta2 ** t
  p_arrays = [] of Array(Float64)
  g_arrays = [] of Array(Float64)
  m_arrays = [] of Array(Float64)
  v_arrays = [] of Array(Float64)
  params.each_array(norm_mode) { |a| p_arrays << a }
  grads.each_array(norm_mode) { |a| g_arrays << a }
  adam.m.each_array(norm_mode) { |a| m_arrays << a }
  adam.v.each_array(norm_mode) { |a| v_arrays << a }

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
norm_mode = NormMode::None
phase_position_data_dir = ""
phase_conditioned_position_data_dir = ""
phase_fixed_offset = -1
phase_shuffle = false
phase_shuffle_seed = 1_u32
singleton_backoff = false
singleton_backoff_stopgrad = false
dry_run = false
beta1 = 0.9
beta2 = 0.999
eps = 1e-8

OptionParser.parse do |p|
  p.banner = "Usage: bin/agpt_train_recur --trie DIR --d-model N --epochs N --lr F --seed N --save PATH [options]"
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
  p.on("--phase-weighted-position-data DIR", "Use phase-position masses from position-data directory; targets remain global") { |v| phase_position_data_dir = v }
  p.on("--phase-conditioned-position-data DIR", "Use phase-position masses plus learned phase embeddings; targets remain global") { |v| phase_conditioned_position_data_dir = v }
  p.on("--phase-offset N", "Fixed phase offset for phase-weighted mode (default sweep by epoch)") { |v| phase_fixed_offset = v.to_i }
  p.on("--phase-shuffle", "Shuffle phase order each window cycle") { phase_shuffle = true }
  p.on("--phase-shuffle-seed N", "Seed for --phase-shuffle (default 1)") { |v| phase_shuffle_seed = v.to_u32 }
  p.on("--singleton-backoff", "Hard-route singleton cap positions to recursive suffix contexts with mass > 1 (requires --partition-depth 0)") { singleton_backoff = true }
  p.on("--singleton-backoff-stopgrad", "Hard-route singleton cap positions but stop gradient at routes; supports partitioned training") { singleton_backoff_stopgrad = true }
  p.on("--norm MODE", "Hidden-state norm: none, rms-pre, or rms-post (default none)") do |v|
    norm_mode = case v
                when "none" then NormMode::None
                when "rms", "rms-pre" then NormMode::RMSPre
                when "rms-post" then NormMode::RMSPost
                else raise "--norm must be none, rms-pre, or rms-post"
                end
  end
  p.on("--dry-run", "Load trie and report shape without training") { dry_run = true }
  p.on("-h", "--help", "Help") { puts p; exit 0 }
end

raise "--trie required" if trie_dir.empty?
raise "--d-model must be > 0" if d_model <= 0
raise "--epochs must be >= 0" if epochs < 0
raise "--lr must be > 0" if lr <= 0.0
raise "--partition-depth must be >= 0" if partition_depth < 0
raise "use only one of --singleton-backoff or --singleton-backoff-stopgrad" if singleton_backoff && singleton_backoff_stopgrad
raise "--singleton-backoff currently requires --partition-depth 0 for correct cross-suffix gradient flow" if singleton_backoff && partition_depth != 0
if save_path.empty? && !dry_run
  raise "--save required unless --dry-run"
end

reader = RadixTrieReader.new(trie_dir, max_cached: 256)
records = load_records(reader, max_nodes)
events = estimate_loss_events(records)
expanded_chars = records.sum(0_i64) { |r| r.edge_tokens.size.to_i64 }
max_endpoint_depth = records.empty? ? 0 : records.max_of(&.endpoint_depth)
partitions = build_partitions(records, partition_depth)
raise "use only one of --phase-weighted-position-data or --phase-conditioned-position-data" unless phase_position_data_dir.empty? || phase_conditioned_position_data_dir.empty?
phase_data_dir = phase_conditioned_position_data_dir.empty? ? phase_position_data_dir : phase_conditioned_position_data_dir
phase_conditioned = !phase_conditioned_position_data_dir.empty?
phase_weights = phase_data_dir.empty? ? nil : PhaseWeightTable.new(phase_data_dir, reader.radix_count)
phase_window = phase_conditioned && phase_weights ? phase_weights.window_size : 0
backoff_enabled = singleton_backoff || singleton_backoff_stopgrad
backoff_table = backoff_enabled ? SingletonBackoffTable.new(records, reader.radix_count) : nil

puts "AGPT tanh-recurrent trainer"
puts "  trie: #{trie_dir}"
puts "  radix_records: #{records.size}#{max_nodes > 0 ? " (truncated)" : ""}"
puts "  expanded_states: #{expanded_chars}"
puts "  loss_events: #{events}"
puts "  partition_depth: #{partition_depth}"
puts "  partitions: #{partitions.size}"
puts "  vocab_size: #{reader.vocab_size}"
puts "  d_model: #{d_model}"
puts "  max_endpoint_depth: #{max_endpoint_depth}"
puts "  norm: #{norm_mode}"
if backoff_table
  mode = singleton_backoff_stopgrad ? "stopgrad" : "full-gradient"
  puts "  singleton_backoff: #{mode} (routes=#{backoff_table.route_count}, unresolved=#{backoff_table.unresolved_count})"
else
  puts "  singleton_backoff: off"
end
if phase_weights
  phase_mode = phase_fixed_offset >= 0 ? "fixed=#{phase_fixed_offset}" : (phase_shuffle ? "shuffle" : "sequential")
  phase_label = phase_conditioned ? "phase_conditioned" : "phase_weighted"
  puts "  #{phase_label}: #{phase_data_dir} (window=#{phase_weights.window_size}, order=#{phase_mode})"
else
  puts "  phase_weighted: off"
end
puts "  optimizer: adam (lr=#{lr}, beta1=#{beta1}, beta2=#{beta2}, eps=#{eps})"

if dry_run
  param_count = RecurParams.new(reader.vocab_size, d_model, phase_window).total_floats(norm_mode)
  state_mb = reader.radix_count.to_i64 * d_model.to_i64 * 8_i64 / 1024.0 / 1024.0
  puts "  params: #{param_count} float64"
  puts "  endpoint_state_memory: #{state_mb.round(2)} MB"
  exit 0
end

params = RecurParams.new(reader.vocab_size, d_model, phase_window)
adam = AdamState.new(reader.vocab_size, d_model, phase_window)
start_epoch = 0
if load_path.empty?
  params.fill_random!(seed)
else
  start_epoch, loaded_seed = load_checkpoint(load_path, params, adam, norm_mode)
  seed = loaded_seed
  puts "  loaded: #{load_path} (epoch=#{start_epoch}, adam_step=#{adam.step}, seed=#{seed})"
end

endpoint_states = Array(Float64).new(reader.radix_count * d_model, 0.0)

(start_epoch + 1).upto(start_epoch + epochs) do |epoch|
  t0 = Time.instant
  phase = phase_weights ? phase_weights.epoch_phase(epoch - 1, phase_fixed_offset, phase_shuffle, phase_shuffle_seed) : -1
  nll, ppl, trained_events, updates = train_epoch(params, norm_mode, adam, partitions, endpoint_states, lr, beta1, beta2, eps, phase_weights, phase, backoff_table, singleton_backoff_stopgrad)
  wall = (Time.instant - t0).total_seconds
  ck_msg = ""
  if checkpoint_every > 0 && epoch % checkpoint_every == 0
    ck = epoch_checkpoint_path(save_path, epoch)
    save_checkpoint(ck, params, adam, epoch, seed, norm_mode)
    ck_msg = " checkpoint=#{ck}"
  end
  phase_msg = phase_weights ? " phase #{phase}" : ""
  printf "epoch %6d  nll %.6f  ppl %.6f  events %d  updates %d%s  wall %.3fs  adam_step %d%s\n",
    epoch, nll, ppl, trained_events, updates, phase_msg, wall, adam.step, ck_msg
end

save_checkpoint(save_path, params, adam, start_epoch + epochs, seed, norm_mode)
puts "saved #{save_path} (epoch=#{start_epoch + epochs}, adam_step=#{adam.step})"
