require "option_parser"
require "../agpt"

include MicroGPT::AGPT

MAGIC_RECUR = 0x52474341_u32 # 'ACGR'
VERSION_RECUR = 1_i32

CBLAS_ROW_MAJOR = MicroGPT::LibCBLAS::Order::RowMajor
CBLAS_NO_TRANS  = MicroGPT::LibCBLAS::Transpose::NoTrans
CBLAS_TRANS     = MicroGPT::LibCBLAS::Transpose::Trans

record RecurRecord,
  id : Int32,
  parent_id : Int32,
  endpoint_depth : Int32,
  edge_tokens : Array(Int32),
  counts : Array({Int32, Int32})

record Transition,
  src_row : Int32,
  dst_row : Int32,
  token : Int32

record EndpointRef,
  row : Int32,
  radix_id : Int32,
  counts : Array({Int32, Int32})

record TargetSidecarEntry,
  token : Int32,
  count : Int32

class TargetSidecar
  getter scale : Int32
  getter substring_count : Int32
  getter total_entries : UInt64

  @offsets : Array(Int32)
  @entries : Array(TargetSidecarEntry)

  def initialize(path : String)
    @scale = 0
    @substring_count = 0
    @total_entries = 0_u64
    @offsets = [] of Int32
    @entries = [] of TargetSidecarEntry
    File.open(path, "rb") do |io|
      magic = Bytes.new(4)
      io.read_fully(magic)
      raise "bad target sidecar magic in #{path}: #{String.new(magic)}" unless String.new(magic) == "AGTS"
      version = io.read_bytes(UInt16, IO::ByteFormat::LittleEndian)
      raise "unsupported target sidecar version #{version}" unless version == 1
      @scale = io.read_bytes(UInt32, IO::ByteFormat::LittleEndian).to_i32
      @substring_count = io.read_bytes(UInt32, IO::ByteFormat::LittleEndian).to_i32
      @total_entries = io.read_bytes(UInt64, IO::ByteFormat::LittleEndian)
      @offsets = Array(Int32).new(@substring_count + 1, 0)
      (@substring_count + 1).times do |i|
        @offsets[i] = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
      end
      @entries = Array(TargetSidecarEntry).new(@total_entries.to_i)
      @total_entries.times do
        tok = io.read_bytes(UInt16, IO::ByteFormat::LittleEndian).to_i32
        cnt = io.read_bytes(UInt32, IO::ByteFormat::LittleEndian).to_i32
        @entries << TargetSidecarEntry.new(tok, cnt)
      end
    end
  end

  def each_entry(substring_id : Int32, &block : TargetSidecarEntry ->)
    return if substring_id < 0 || substring_id >= @substring_count
    start = @offsets[substring_id]
    finish = @offsets[substring_id + 1]
    i = start
    while i < finish
      yield @entries[i]
      i += 1
    end
  end
end

class RecurParams32
  getter vocab_size : Int32
  getter d_model : Int32
  getter emb : Array(Float32)
  getter w_h : Array(Float32)
  getter w_x : Array(Float32)
  getter b : Array(Float32)
  getter w_o : Array(Float32)
  getter c_o : Array(Float32)

  def initialize(@vocab_size : Int32, @d_model : Int32)
    v = @vocab_size
    d = @d_model
    @emb = Array(Float32).new(v * d, 0.0_f32)
    @w_h = Array(Float32).new(d * d, 0.0_f32)
    @w_x = Array(Float32).new(d * d, 0.0_f32)
    @b = Array(Float32).new(d, 0.0_f32)
    @w_o = Array(Float32).new(v * d, 0.0_f32)
    @c_o = Array(Float32).new(v, 0.0_f32)
  end

  def each_array(&block : Array(Float32) ->)
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

  def fill_random!(seed : UInt64, output_scale : Float64? = nil)
    rng = Random.new(seed)
    scale_emb = 0.02_f64
    scale_rec = 1.0 / Math.sqrt(@d_model.to_f64)
    fill_normal!(@emb, rng, scale_emb)
    fill_normal!(@w_h, rng, scale_rec)
    fill_normal!(@w_x, rng, scale_rec)
    fill_normal!(@w_o, rng, output_scale || scale_rec)
  end

  private def fill_normal!(a : Array(Float32), rng : Random, scale : Float64)
    i = 0
    while i < a.size
      u1 = rng.rand
      u1 = 1e-12 if u1 <= 0.0
      u2 = rng.rand
      r = Math.sqrt(-2.0 * Math.log(u1))
      theta = 2.0 * Math::PI * u2
      a[i] = (scale * r * Math.cos(theta)).to_f32
      if i + 1 < a.size
        a[i + 1] = (scale * r * Math.sin(theta)).to_f32
      end
      i += 2
    end
  end
end

class AdamState32
  getter m : RecurParams32
  getter v : RecurParams32
  property step : Int32

  def initialize(vocab_size : Int32, d_model : Int32)
    @m = RecurParams32.new(vocab_size, d_model)
    @v = RecurParams32.new(vocab_size, d_model)
    @step = 0
  end
end

class LevelPlan
  getter max_depth : Int32
  getter row_counts : Array(Int32)
  getter transitions_by_depth : Array(Array(Transition))
  getter endpoints_by_depth : Array(Array(EndpointRef))
  getter transition_count : Int64
  getter endpoint_count : Int32
  getter loss_events : Int64

  def initialize(records : Array(RecurRecord), radix_count : Int32)
    max_depth = records.empty? ? 0 : records.max_of(&.endpoint_depth)
    @max_depth = max_depth
    @row_counts = Array(Int32).new(max_depth + 1, 0)
    @row_counts[0] = 1
    @transitions_by_depth = Array(Array(Transition)).new(max_depth + 1) { [] of Transition }
    @endpoints_by_depth = Array(Array(EndpointRef)).new(max_depth + 1) { [] of EndpointRef }
    @transition_count = 0_i64
    @endpoint_count = 0
    @loss_events = 0_i64

    row_by_radix = Array(Int32).new(radix_count, -1)
    depth_by_radix = Array(Int32).new(radix_count, -1)
    row_by_radix[0] = 0
    depth_by_radix[0] = 0

    records.each do |r|
      parent_depth = r.endpoint_depth - r.edge_tokens.size
      actual_parent_depth = r.parent_id == 0 ? 0 : depth_by_radix[r.parent_id]
      raise "parent #{r.parent_id} missing before record #{r.id}" if actual_parent_depth < 0
      raise "parent depth mismatch for record #{r.id}: #{actual_parent_depth} != #{parent_depth}" unless actual_parent_depth == parent_depth
      src_row = r.parent_id == 0 ? 0 : row_by_radix[r.parent_id]
      src_depth = parent_depth

      r.edge_tokens.each_with_index do |tok, pos|
        dst_depth = src_depth + 1
        dst_row = @row_counts[dst_depth]
        @row_counts[dst_depth] += 1
        @transitions_by_depth[dst_depth] << Transition.new(src_row, dst_row, tok)
        @transition_count += 1

        if pos == r.edge_tokens.size - 1
          raise "endpoint depth mismatch for record #{r.id}" unless dst_depth == r.endpoint_depth
          row_by_radix[r.id] = dst_row
          depth_by_radix[r.id] = dst_depth
          @endpoints_by_depth[dst_depth] << EndpointRef.new(dst_row, r.id, r.counts)
          @endpoint_count += 1
          r.counts.each { |pair| @loss_events += pair[1].to_i64 }
        end

        src_row = dst_row
        src_depth = dst_depth
      end
    end
  end

  def state_floats : Int64
    @row_counts.sum(0_i64) { |n| n.to_i64 }
  end
end

def ensure_parent_dir(path : String)
  parent = File.dirname(path)
  return if parent == "." || parent.empty?
  Dir.mkdir_p(parent)
end

def write_f64_array(io : IO, a : Array(Float32))
  a.each { |x| io.write_bytes(x.to_f64, IO::ByteFormat::LittleEndian) }
end

def read_f64_array(io : IO, a : Array(Float32))
  a.size.times { |i| a[i] = io.read_bytes(Float64, IO::ByteFormat::LittleEndian).to_f32 }
end

def save_checkpoint(path : String, params : RecurParams32, adam : AdamState32, epoch : Int32, seed : UInt64)
  ensure_parent_dir(path)
  File.open(path, "wb") do |io|
    io.write_bytes(MAGIC_RECUR, IO::ByteFormat::LittleEndian)
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

def load_checkpoint(path : String, params : RecurParams32, adam : AdamState32) : {Int32, UInt64}
  File.open(path, "rb") do |io|
    magic = io.read_bytes(UInt32, IO::ByteFormat::LittleEndian)
    raise "unsupported recurrent checkpoint magic 0x#{magic.to_s(16)} in #{path}; level trainer currently supports plain tanh only" unless magic == MAGIC_RECUR
    version = io.read_bytes(Int32, IO::ByteFormat::LittleEndian)
    raise "unsupported recurrent checkpoint version #{version}" unless version == VERSION_RECUR
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

def compute_x_proj(params : RecurParams32) : Array(Float32)
  v = params.vocab_size
  d = params.d_model
  proj = Array(Float32).new(v * d, 0.0_f32)
  MicroGPT::LibCBLAS.cblas_sgemm(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_TRANS,
    v.to_i64, d.to_i64, d.to_i64,
    1.0_f32,
    params.emb.to_unsafe, d.to_i64,
    params.w_x.to_unsafe, d.to_i64,
    0.0_f32,
    proj.to_unsafe, d.to_i64)
  proj
end

def forward_plan(params : RecurParams32, plan : LevelPlan, x_proj : Array(Float32), batch_size : Int32) : Array(Array(Float32))
  d = params.d_model
  states = Array(Array(Float32)).new(plan.max_depth + 1) do |depth|
    Array(Float32).new(plan.row_counts[depth] * d, 0.0_f32)
  end
  parent_mat = Array(Float32).new(batch_size * d, 0.0_f32)
  out_mat = Array(Float32).new(batch_size * d, 0.0_f32)

  1.upto(plan.max_depth) do |depth|
    trans = plan.transitions_by_depth[depth]
    prev_states = states[depth - 1]
    cur_states = states[depth]
    offset = 0
    while offset < trans.size
      n = Math.min(batch_size, trans.size - offset)
      n.times do |i|
        tr = trans[offset + i]
        src_base = tr.src_row * d
        dst_base = i * d
        d.times { |j| parent_mat[dst_base + j] = prev_states[src_base + j] }
      end
      MicroGPT::LibCBLAS.cblas_sgemm(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_TRANS,
        n.to_i64, d.to_i64, d.to_i64,
        1.0_f32,
        parent_mat.to_unsafe, d.to_i64,
        params.w_h.to_unsafe, d.to_i64,
        0.0_f32,
        out_mat.to_unsafe, d.to_i64)
      n.times do |i|
        tr = trans[offset + i]
        x_base = tr.token * d
        out_base = i * d
        state_base = tr.dst_row * d
        d.times do |j|
          z = out_mat[out_base + j] + x_proj[x_base + j] + params.b[j]
          cur_states[state_base + j] = Math.tanh(z.to_f64).to_f32
        end
      end
      offset += n
    end
  end

  states
end

def add_output_loss_and_grads(
  params : RecurParams32,
  grads : RecurParams32,
  plan : LevelPlan,
  states : Array(Array(Float32)),
  dh : Array(Array(Float32)),
  batch_size : Int32,
  prior_sidecar : TargetSidecar?,
  radix_to_substring : RadixToSubstring?,
  prior_scale : Float64,
  prior_floor : Float64,
  residual_scale : Float64,
  residual_l2 : Float64
) : {Float64, Int64}
  d = params.d_model
  v = params.vocab_size
  h_mat = Array(Float32).new(batch_size * d, 0.0_f32)
  logits = Array(Float32).new(batch_size * v, 0.0_f32)
  residuals = Array(Float32).new(batch_size * v, 0.0_f32)
  dh_mat = Array(Float32).new(batch_size * d, 0.0_f32)
  loss = 0.0_f64
  events = 0_i64

  1.upto(plan.max_depth) do |depth|
    endpoints = plan.endpoints_by_depth[depth]
    next if endpoints.empty?
    depth_states = states[depth]
    depth_dh = dh[depth]
    offset = 0
    while offset < endpoints.size
      n = Math.min(batch_size, endpoints.size - offset)
      n.times do |i|
        ep = endpoints[offset + i]
        h_src = ep.row * d
        h_dst = i * d
        d.times { |j| h_mat[h_dst + j] = depth_states[h_src + j] }
        logit_base = i * v
        v.times { |tok| residuals[logit_base + tok] = params.c_o[tok] }
        if sidecar = prior_sidecar
          rts = radix_to_substring || raise "radix_to_substring required with prior sidecar"
          floor = prior_floor
          floor = 1.0e-12 if floor <= 0.0
          v.times do |tok|
            logits[logit_base + tok] = (prior_scale * Math.log(floor)).to_f32
          end
          sid = rts.substring_id_for(ep.radix_id)
          inv_scale = 1.0 / sidecar.scale.to_f64
          sidecar.each_entry(sid) do |entry|
            next if entry.token < 0 || entry.token >= v
            p = entry.count.to_f64 * inv_scale
            p = floor if p < floor
            logits[logit_base + entry.token] = (prior_scale * Math.log(p)).to_f32
          end
        else
          v.times { |tok| logits[logit_base + tok] = 0.0_f32 }
        end
      end
      MicroGPT::LibCBLAS.cblas_sgemm(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_TRANS,
        n.to_i64, v.to_i64, d.to_i64,
        1.0_f32,
        h_mat.to_unsafe, d.to_i64,
        params.w_o.to_unsafe, d.to_i64,
        1.0_f32,
        residuals.to_unsafe, v.to_i64)

      n.times do |i|
        logit_base = i * v
        v.times do |tok|
          logits[logit_base + tok] += (residual_scale * residuals[logit_base + tok].to_f64).to_f32
        end
      end

      n.times do |i|
        ep = endpoints[offset + i]
        logit_base = i * v
        max_logit = -Float32::INFINITY
        v.times do |tok|
          x = logits[logit_base + tok]
          max_logit = x if x > max_logit
        end
        sum = 0.0_f64
        v.times do |tok|
          p = Math.exp((logits[logit_base + tok] - max_logit).to_f64)
          logits[logit_base + tok] = p.to_f32
          sum += p
        end
        inv_sum = 1.0 / sum
        count_total = 0_i64
        ep.counts.each { |pair| count_total += pair[1].to_i64 }
        next if count_total <= 0
        events += count_total
        v.times { |tok| logits[logit_base + tok] = (logits[logit_base + tok].to_f64 * inv_sum * count_total.to_f64).to_f32 }
        ep.counts.each do |pair|
          tok = pair[0]
          cnt = pair[1].to_i64
          prob = logits[logit_base + tok].to_f64 / count_total.to_f64
          prob = 1e-30 if prob <= 0.0
          loss -= cnt.to_f64 * Math.log(prob)
          logits[logit_base + tok] -= cnt.to_f32
        end

        if residual_l2 > 0.0
          reg_scale = count_total.to_f64 * residual_l2 / v.to_f64
          v.times do |tok|
            r = residuals[logit_base + tok].to_f64
            logits[logit_base + tok] = (residual_scale * logits[logit_base + tok].to_f64 + reg_scale * r).to_f32
          end
        elsif residual_scale != 1.0
          v.times { |tok| logits[logit_base + tok] = (residual_scale * logits[logit_base + tok].to_f64).to_f32 }
        end
      end

      v.times do |tok|
        sum = 0.0_f32
        n.times { |i| sum += logits[i * v + tok] }
        grads.c_o[tok] += sum
      end

      MicroGPT::LibCBLAS.cblas_sgemm(CBLAS_ROW_MAJOR, CBLAS_TRANS, CBLAS_NO_TRANS,
        v.to_i64, d.to_i64, n.to_i64,
        1.0_f32,
        logits.to_unsafe, v.to_i64,
        h_mat.to_unsafe, d.to_i64,
        1.0_f32,
        grads.w_o.to_unsafe, d.to_i64)

      MicroGPT::LibCBLAS.cblas_sgemm(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS,
        n.to_i64, d.to_i64, v.to_i64,
        1.0_f32,
        logits.to_unsafe, v.to_i64,
        params.w_o.to_unsafe, d.to_i64,
        0.0_f32,
        dh_mat.to_unsafe, d.to_i64)

      n.times do |i|
        ep = endpoints[offset + i]
        dst = ep.row * d
        src = i * d
        d.times { |j| depth_dh[dst + j] += dh_mat[src + j] }
      end
      offset += n
    end
  end

  {loss, events}
end

def backward_transitions(
  params : RecurParams32,
  grads : RecurParams32,
  plan : LevelPlan,
  states : Array(Array(Float32)),
  dh : Array(Array(Float32)),
  batch_size : Int32
)
  d = params.d_model
  v = params.vocab_size
  parent_mat = Array(Float32).new(batch_size * d, 0.0_f32)
  dz_mat = Array(Float32).new(batch_size * d, 0.0_f32)
  dh_prev_mat = Array(Float32).new(batch_size * d, 0.0_f32)
  token_grad = Array(Float32).new(v * d, 0.0_f32)

  plan.max_depth.downto(1) do |depth|
    trans = plan.transitions_by_depth[depth]
    prev_states = states[depth - 1]
    cur_states = states[depth]
    prev_dh = dh[depth - 1]
    cur_dh = dh[depth]
    offset = 0
    while offset < trans.size
      n = Math.min(batch_size, trans.size - offset)
      n.times do |i|
        tr = trans[offset + i]
        src_base = tr.src_row * d
        dst_base = tr.dst_row * d
        local_base = i * d
        d.times do |j|
          parent_mat[local_base + j] = prev_states[src_base + j]
          h = cur_states[dst_base + j]
          dz = cur_dh[dst_base + j] * (1.0_f32 - h * h)
          dz_mat[local_base + j] = dz
          grads.b[j] += dz
          token_grad[tr.token * d + j] += dz
        end
      end

      MicroGPT::LibCBLAS.cblas_sgemm(CBLAS_ROW_MAJOR, CBLAS_TRANS, CBLAS_NO_TRANS,
        d.to_i64, d.to_i64, n.to_i64,
        1.0_f32,
        dz_mat.to_unsafe, d.to_i64,
        parent_mat.to_unsafe, d.to_i64,
        1.0_f32,
        grads.w_h.to_unsafe, d.to_i64)

      MicroGPT::LibCBLAS.cblas_sgemm(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS,
        n.to_i64, d.to_i64, d.to_i64,
        1.0_f32,
        dz_mat.to_unsafe, d.to_i64,
        params.w_h.to_unsafe, d.to_i64,
        0.0_f32,
        dh_prev_mat.to_unsafe, d.to_i64)

      n.times do |i|
        tr = trans[offset + i]
        dst = tr.src_row * d
        src = i * d
        d.times { |j| prev_dh[dst + j] += dh_prev_mat[src + j] }
      end
      offset += n
    end
  end

  MicroGPT::LibCBLAS.cblas_sgemm(CBLAS_ROW_MAJOR, CBLAS_TRANS, CBLAS_NO_TRANS,
    d.to_i64, d.to_i64, v.to_i64,
    1.0_f32,
    token_grad.to_unsafe, d.to_i64,
    params.emb.to_unsafe, d.to_i64,
    1.0_f32,
    grads.w_x.to_unsafe, d.to_i64)

  MicroGPT::LibCBLAS.cblas_sgemm(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS,
    v.to_i64, d.to_i64, d.to_i64,
    1.0_f32,
    token_grad.to_unsafe, d.to_i64,
    params.w_x.to_unsafe, d.to_i64,
    1.0_f32,
    grads.emb.to_unsafe, d.to_i64)
end

def scale_grads!(grads : RecurParams32, scale : Float32)
  grads.each_array { |a| a.size.times { |i| a[i] *= scale } }
end

def train_plan(
  params : RecurParams32,
  adam : AdamState32,
  plan : LevelPlan,
  lr : Float64,
  beta1 : Float64,
  beta2 : Float64,
  eps : Float64,
  batch_size : Int32,
  prior_sidecar : TargetSidecar?,
  radix_to_substring : RadixToSubstring?,
  prior_scale : Float64,
  prior_floor : Float64,
  residual_scale : Float64,
  residual_l2 : Float64
) : {Float64, Int64}
  x_proj = compute_x_proj(params)
  states = forward_plan(params, plan, x_proj, batch_size)
  dh = Array(Array(Float32)).new(plan.max_depth + 1) do |depth|
    Array(Float32).new(plan.row_counts[depth] * params.d_model, 0.0_f32)
  end
  grads = RecurParams32.new(params.vocab_size, params.d_model)
  loss, events = add_output_loss_and_grads(params, grads, plan, states, dh, batch_size, prior_sidecar, radix_to_substring, prior_scale, prior_floor, residual_scale, residual_l2)
  return {0.0_f64, 0_i64} if events == 0
  backward_transitions(params, grads, plan, states, dh, batch_size)
  scale_grads!(grads, (1.0_f64 / events.to_f64).to_f32)
  adam_update!(params, grads, adam, lr, beta1, beta2, eps)
  {loss, events}
end

def adam_update!(params : RecurParams32, grads : RecurParams32, adam : AdamState32, lr : Float64, beta1 : Float64, beta2 : Float64, eps : Float64)
  adam.step += 1
  t = adam.step
  bc1 = 1.0 - beta1 ** t
  bc2 = 1.0 - beta2 ** t
  p_arrays = [] of Array(Float32)
  g_arrays = [] of Array(Float32)
  m_arrays = [] of Array(Float32)
  v_arrays = [] of Array(Float32)
  params.each_array { |a| p_arrays << a }
  grads.each_array { |a| g_arrays << a }
  adam.m.each_array { |a| m_arrays << a }
  adam.v.each_array { |a| v_arrays << a }

  p_arrays.each_with_index do |p, ai|
    g = g_arrays[ai]
    m = m_arrays[ai]
    vv = v_arrays[ai]
    p.size.times do |i|
      gi = g[i].to_f64
      mi = beta1 * m[i].to_f64 + (1.0 - beta1) * gi
      vi = beta2 * vv[i].to_f64 + (1.0 - beta2) * gi * gi
      m[i] = mi.to_f32
      vv[i] = vi.to_f32
      m_hat = mi / bc1
      v_hat = vi / bc2
      p[i] = (p[i].to_f64 - lr * m_hat / (Math.sqrt(v_hat) + eps)).to_f32
    end
  end
end

trie_dir = ""
save_path = ""
load_path = ""
position_data_dir = ""
prior_sidecar_path = ""
d_model = 0
epochs = 0
lr = 0.001
seed = 1_u64
checkpoint_every = 0
max_nodes = 0
partition_depth = 0
batch_size = 65536
prior_scale = 1.0
prior_floor = 1.0e-12
residual_scale = 1.0
residual_l2 = 0.0
output_init_scale = nil.as(Float64?)
dry_run = false
beta1 = 0.9
beta2 = 0.999
eps = 1e-8

OptionParser.parse do |p|
  p.banner = "Usage: bin/agpt_train_recur_level --trie DIR --d-model N --epochs N --lr F --seed N --save PATH [options]"
  p.on("--trie DIR", "Global radix trie directory") { |v| trie_dir = v }
  p.on("--d-model N", "Hidden dimension") { |v| d_model = v.to_i }
  p.on("--epochs N", "Training epochs") { |v| epochs = v.to_i }
  p.on("--lr F", "Adam learning rate (default 0.001)") { |v| lr = v.to_f }
  p.on("--seed N", "Initialization seed (default 1)") { |v| seed = v.to_u64 }
  p.on("--save PATH", "Write recurrent checkpoint") { |v| save_path = v }
  p.on("--load PATH", "Resume recurrent checkpoint") { |v| load_path = v }
  p.on("--position-data DIR", "Position-data directory containing prefix_radix_to_substring.bin") { |v| position_data_dir = v }
  p.on("--prior-sidecar PATH", "Frozen AGTS target/prior sidecar keyed by substring id") { |v| prior_sidecar_path = v }
  p.on("--prior-scale F", "Scale for log prior added to residual logits (default 1.0)") { |v| prior_scale = v.to_f }
  p.on("--prior-floor F", "Probability floor for missing prior entries (default 1e-12)") { |v| prior_floor = v.to_f }
  p.on("--residual-scale F", "Scale residual logits during training (default 1.0)") { |v| residual_scale = v.to_f }
  p.on("--residual-l2 F", "Per-node residual-logit L2 gradient penalty (default 0)") { |v| residual_l2 = v.to_f }
  p.on("--output-init-scale F", "Override W_o random initialization scale; useful for residual-on-prior runs") { |v| output_init_scale = v.to_f }
  p.on("--checkpoint-every N", "Write epoch checkpoints every N epochs") { |v| checkpoint_every = v.to_i }
  p.on("--max-nodes N", "Diagnostic: use first N radix records only") { |v| max_nodes = v.to_i }
  p.on("--partition-depth N", "Optimizer partition depth: 0 full batch, 1 root-child batches (default 0)") { |v| partition_depth = v.to_i }
  p.on("--batch-size N", "Level matmul/gather chunk size (default 65536)") { |v| batch_size = v.to_i }
  p.on("--dry-run", "Load trie, build plans, and report shape without training") { dry_run = true }
  p.on("-h", "--help", "Help") { puts p; exit 0 }
end

raise "--trie required" if trie_dir.empty?
raise "--d-model must be > 0" if d_model <= 0
raise "--epochs must be >= 0" if epochs < 0
raise "--lr must be > 0" if lr <= 0.0
raise "--partition-depth must be 0 or 1" unless partition_depth == 0 || partition_depth == 1
raise "--batch-size must be > 0" if batch_size <= 0
raise "--save required unless --dry-run" if save_path.empty? && !dry_run
raise "--prior-scale must be finite" unless prior_scale.finite?
raise "--prior-floor must be > 0" unless prior_floor > 0.0
raise "--prior-sidecar requires --position-data" if !prior_sidecar_path.empty? && position_data_dir.empty?
raise "--residual-scale must be finite" unless residual_scale.finite?
raise "--residual-l2 must be finite and >= 0" unless residual_l2.finite? && residual_l2 >= 0.0
raise "--output-init-scale must be finite and >= 0" if output_init_scale && (!output_init_scale.not_nil!.finite? || output_init_scale.not_nil! < 0.0)

reader = RadixTrieReader.new(trie_dir, max_cached: 256)
records = load_records(reader, max_nodes)
partitions = build_partitions(records, partition_depth)
plans = partitions.map { |part| LevelPlan.new(part, reader.radix_count) }
total_transitions = plans.sum(0_i64, &.transition_count)
total_states = plans.sum(0_i64, &.state_floats)
total_events = plans.sum(0_i64, &.loss_events)
peak_plan_states = plans.empty? ? 0_i64 : plans.max_of(&.state_floats)
max_endpoint_depth = records.empty? ? 0 : records.max_of(&.endpoint_depth)

puts "AGPT level-batched tanh-recurrent trainer"
puts "  trie: #{trie_dir}"
puts "  radix_records: #{records.size}#{max_nodes > 0 ? " (truncated)" : ""}"
puts "  logical_transitions: #{total_transitions}"
puts "  logical_states: #{total_states}"
puts "  loss_events: #{total_events}"
puts "  partition_depth: #{partition_depth}"
puts "  partitions: #{plans.size}"
puts "  vocab_size: #{reader.vocab_size}"
puts "  d_model: #{d_model}"
puts "  max_endpoint_depth: #{max_endpoint_depth}"
puts "  batch_size: #{batch_size}"
puts "  dtype: Float32"
prior_sidecar = nil.as(TargetSidecar?)
radix_to_substring = nil.as(RadixToSubstring?)
if !prior_sidecar_path.empty?
  prior_sidecar = TargetSidecar.new(prior_sidecar_path)
  rts_path = File.join(position_data_dir, "prefix_radix_to_substring.bin")
  radix_to_substring = File.open(rts_path, "rb") { |io| RadixToSubstring.read_from(io) }
  raise "position radix_count mismatch: #{radix_to_substring.not_nil!.radix_count} != #{reader.radix_count}" unless radix_to_substring.not_nil!.radix_count == reader.radix_count
  raise "prior sidecar substring_count must be positive" unless prior_sidecar.not_nil!.substring_count > 0
  puts "  prior_sidecar: #{prior_sidecar_path} (substrings=#{prior_sidecar.not_nil!.substring_count}, entries=#{prior_sidecar.not_nil!.total_entries}, scale=#{prior_sidecar.not_nil!.scale}, logit_scale=#{prior_scale}, floor=#{prior_floor})"
else
  puts "  prior_sidecar: off"
end
puts "  output_init_scale: #{output_init_scale || "default"}"
puts "  residual_scale: #{residual_scale}"
puts "  residual_l2: #{residual_l2}"
puts "  optimizer: adam (lr=#{lr}, beta1=#{beta1}, beta2=#{beta2}, eps=#{eps})"
state_mb = total_states * d_model.to_i64 * 4_i64 / 1024.0 / 1024.0
peak_state_mb = peak_plan_states * d_model.to_i64 * 4_i64 / 1024.0 / 1024.0
puts "  logical_state_memory_partition_sum: #{state_mb.round(2)} MB (H only)"
puts "  logical_state_memory_peak_partition: #{peak_state_mb.round(2)} MB (H only)"

if dry_run
  params = RecurParams32.new(reader.vocab_size, d_model)
  puts "  params: #{params.total_floats} float32"
  exit 0
end

params = RecurParams32.new(reader.vocab_size, d_model)
adam = AdamState32.new(reader.vocab_size, d_model)
start_epoch = 0
if load_path.empty?
  params.fill_random!(seed, output_init_scale)
else
  start_epoch, loaded_seed = load_checkpoint(load_path, params, adam)
  seed = loaded_seed
  puts "  loaded: #{load_path} (epoch=#{start_epoch}, adam_step=#{adam.step}, seed=#{seed})"
end

(start_epoch + 1).upto(start_epoch + epochs) do |epoch|
  t0 = Time.instant
  loss_total = 0.0_f64
  events_total = 0_i64
  updates = 0
  plans.each do |plan|
    next if plan.loss_events == 0
    loss, events = train_plan(params, adam, plan, lr, beta1, beta2, eps, batch_size, prior_sidecar, radix_to_substring, prior_scale, prior_floor, residual_scale, residual_l2)
    next if events == 0
    loss_total += loss
    events_total += events
    updates += 1
  end
  raise "no loss events found" if events_total == 0
  nll = loss_total / events_total.to_f64
  ppl = Math.exp(nll)
  wall = (Time.instant - t0).total_seconds
  ck_msg = ""
  if checkpoint_every > 0 && epoch % checkpoint_every == 0
    ck = epoch_checkpoint_path(save_path, epoch)
    save_checkpoint(ck, params, adam, epoch, seed)
    ck_msg = " checkpoint=#{ck}"
  end
  printf "epoch %6d  nll %.6f  ppl %.6f  events %d  updates %d  wall %.3fs  adam_step %d%s\n",
    epoch, nll, ppl, events_total, updates, wall, adam.step, ck_msg
end

save_checkpoint(save_path, params, adam, start_epoch + epochs, seed)
puts "saved #{save_path} (epoch=#{start_epoch + epochs}, adam_step=#{adam.step})"
