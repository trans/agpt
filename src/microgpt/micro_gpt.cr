# MiniGPT — A minimal Transformer language model in Crystal
# Character-level, from scratch, with heterogeneous head support
#
# Usage:
#   crystal build --release mini_gpt.cr
#   ./mini_gpt input.txt
#
# Config is set at the bottom of this file.

module MicroGPT

# =============================================================================
# Core Matrix Type
# =============================================================================

class Mat
  # Global memory tracking
  @@allocated_bytes : Int64 = 0_i64
  @@max_bytes : Int64 = 2_i64 * 1024 * 1024 * 1024  # 2 GiB default cap

  def self.allocated_bytes : Int64
    @@allocated_bytes
  end

  def self.max_bytes : Int64
    @@max_bytes
  end

  def self.max_bytes=(limit : Int64)
    @@max_bytes = limit
  end

  private def track_alloc
    size = (@rows.to_i64 * @cols.to_i64) * sizeof(Float32)
    if @@allocated_bytes + size > @@max_bytes
      raise "Mat memory cap exceeded: #{@@allocated_bytes + size} > #{@@max_bytes} bytes " \
            "(allocated #{@@allocated_bytes / (1024 * 1024)} MiB, " \
            "requesting #{size / (1024 * 1024)} MiB, " \
            "cap #{@@max_bytes / (1024 * 1024)} MiB)"
    end
    @@allocated_bytes += size
  end

  private def track_free
    size = (@rows.to_i64 * @cols.to_i64) * sizeof(Float32)
    @@allocated_bytes -= size
  end

  getter rows : Int32
  getter cols : Int32
  @data : Array(Float32)
  @gpu_ptr : Pointer(Void) = Pointer(Void).null
  @cpu_valid : Bool = true
  @gpu_valid : Bool = false
  property store_backed : Bool = false  # true = gpu_ptr is inside a WeightStore, don't cudaFree

  def initialize(@rows, @cols)
    track_alloc
    @data = Array(Float32).new(rows * cols, 0.0_f32)
  end

  def initialize(@rows, @cols, @data : Array(Float32))
    track_alloc
  end

  # GPU-resident constructor (no CPU data yet)
  def initialize(@rows, @cols, @gpu_ptr : Pointer(Void))
    track_alloc
    @data = Array(Float32).new(rows * cols, 0.0_f32)
    @cpu_valid = false
    @gpu_valid = true
  end

  def finalize
    track_free
    free_gpu
  end

  def free_gpu
    unless @gpu_ptr.null?
      MicroGPT.backend.gpu_free(@gpu_ptr) unless @store_backed
      @gpu_ptr = Pointer(Void).null
      @gpu_valid = false
    end
  end

  # Assign a GPU pointer from a WeightStore (no allocation, no free)
  def set_store_ptr(ptr : Pointer(Void))
    free_gpu  # release any existing independent allocation
    @gpu_ptr = ptr
    @store_backed = true
    @gpu_valid = true
  end

  # Access CPU data — triggers download from GPU if needed
  def data : Array(Float32)
    sync_to_cpu
    @data
  end

  def raw_data : Array(Float32)
    @data
  end

  def [](r : Int32, c : Int32) : Float32
    sync_to_cpu
    @data[r * @cols + c]
  end

  def []=(r : Int32, c : Int32, val : Float32)
    sync_to_cpu
    @data[r * @cols + c] = val
    @gpu_valid = false
  end

  def []=(r : Int32, c : Int32, val : Float64)
    sync_to_cpu
    @data[r * @cols + c] = val.to_f32
    @gpu_valid = false
  end

  def *(other : Mat) : Mat
    MicroGPT.backend.matmul(self, other)
  end

  def t : Mat
    MicroGPT.backend.transpose(self)
  end

  def +(other : Mat) : Mat
    MicroGPT.backend.add(self, other)
  end

  def *(scalar : Float32) : Mat
    MicroGPT.backend.scale(self, scalar)
  end

  def *(scalar : Float64) : Mat
    MicroGPT.backend.scale(self, scalar.to_f32)
  end

  # --- In-place operations (no allocation) ---

  def add!(other : Mat)
    MicroGPT.backend.add!(self, other)
  end

  def scale!(scalar : Float32)
    MicroGPT.backend.scale!(self, scalar)
  end

  def scale!(scalar : Float64)
    MicroGPT.backend.scale!(self, scalar.to_f32)
  end

  def zero!
    @data.fill(0.0_f32)
    @cpu_valid = true
    @gpu_valid = false
  end

  def self.randn(rows : Int32, cols : Int32, scale : Float64 = 1.0) : Mat
    # Note: scale param stays Float64 for precision in init
    mat = Mat.new(rows, cols)
    mat.raw_data.size.times do |i|
      u1 = rand + 1e-10
      u2 = rand
      mat.raw_data[i] = (Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math::PI * u2) * scale).to_f32
    end
    mat
  end

  def self.zeros(rows : Int32, cols : Int32) : Mat
    Mat.new(rows, cols)
  end

  # --- GPU memory management ---

  def byte_size : UInt64
    (@rows * @cols).to_u64 * sizeof(Float32)
  end

  def gpu_ptr : Pointer(Void)
    sync_to_gpu
    @gpu_ptr
  end

  def gpu_valid? : Bool
    @gpu_valid
  end

  def cpu_valid? : Bool
    @cpu_valid
  end

  def sync_to_gpu
    return if @gpu_valid
    if @gpu_ptr.null?
      @gpu_ptr = MicroGPT.backend.gpu_alloc(byte_size)
    end
    MicroGPT.backend.gpu_upload(@gpu_ptr, @data.to_unsafe.as(Void*), byte_size)
    @gpu_valid = true
  end

  def sync_to_cpu
    return if @cpu_valid
    return if @gpu_ptr.null?
    MicroGPT.backend.gpu_download(@data.to_unsafe.as(Void*), @gpu_ptr, byte_size)
    @cpu_valid = true
  end

  # Mark GPU data as authoritative (used by backends after GPU computation)
  def mark_gpu_only
    @gpu_valid = true
    @cpu_valid = false
  end

  # Mark CPU data as modified
  def invalidate_gpu
    @gpu_valid = false
  end
end

# =============================================================================
# Helper Functions
# =============================================================================

module MathUtils
  def softmax_rows(x : Mat) : Mat
    result = Mat.new(x.rows, x.cols)
    x.rows.times do |i|
      max = x[i, 0]
      x.cols.times { |j| max = x[i, j] if x[i, j] > max }
      sum = 0.0
      x.cols.times do |j|
        result[i, j] = Math.exp(x[i, j] - max)
        sum += result[i, j]
      end
      x.cols.times { |j| result[i, j] /= sum }
    end
    result
  end

  def softmax_backward(s : Mat, ds : Mat) : Mat
    result = Mat.new(s.rows, s.cols)
    s.rows.times do |i|
      dot = 0.0
      s.cols.times { |j| dot += ds[i, j] * s[i, j] }
      s.cols.times { |j| result[i, j] = s[i, j] * (ds[i, j] - dot) }
    end
    result
  end

  def concat_cols(mats : Array(Mat)) : Mat
    rows = mats.first.rows
    total_cols = mats.sum(&.cols)
    result = Mat.new(rows, total_cols)
    col_offset = 0
    mats.each do |m|
      rows.times do |i|
        m.cols.times { |j| result[i, col_offset + j] = m[i, j] }
      end
      col_offset += m.cols
    end
    result
  end

  def split_cols(m : Mat, sizes : Array(Int32)) : Array(Mat)
    result = [] of Mat
    col_offset = 0
    sizes.each do |size|
      chunk = Mat.new(m.rows, size)
      m.rows.times do |i|
        size.times { |j| chunk[i, j] = m[i, col_offset + j] }
      end
      result << chunk
      col_offset += size
    end
    result
  end

  # Stack matrices vertically: [A(r×c), B(r×c), ...] → (n*r × c)
  def stack_mats(mats : Array(Mat)) : Mat
    rows_per = mats[0].rows
    cols = mats[0].cols
    block = rows_per * cols
    result = Mat.new(rows_per * mats.size, cols)
    dst = result.raw_data.to_unsafe
    mats.each_with_index do |m, i|
      (dst + i * block).copy_from(m.data.to_unsafe, block)
    end
    result
  end

  # Unstack: (n*r × c) → n matrices of (r × c)
  def unstack_mats(m : Mat, count : Int32, rows_per : Int32) : Array(Mat)
    cols = m.cols
    block = rows_per * cols
    src = m.data.to_unsafe
    result = Array(Mat).new(count)
    count.times do |i|
      mat = Mat.new(rows_per, cols)
      mat.raw_data.to_unsafe.copy_from(src + i * block, block)
      result << mat
    end
    result
  end
end

# =============================================================================
# Adam Optimizer
# =============================================================================

class AdamParam
  getter m : Mat
  getter v : Mat
  property t : Int32

  def initialize(rows : Int32, cols : Int32)
    @m = Mat.new(rows, cols)
    @v = Mat.new(rows, cols)
    @t = 0
  end

  def step(param : Mat, grad : Mat, lr : Float64)
    @t += 1
    MicroGPT.backend.adam_step(param, grad, @m, @v, lr, @t)
  end
end

# =============================================================================
# Layer Interface
# =============================================================================

module Layer
  abstract def forward(x : Mat) : Mat
  abstract def backward(grad : Mat) : Mat
  abstract def update(lr : Float64)
end

# =============================================================================
# Linear Layer
# =============================================================================

class Linear
  include Layer

  getter w : Mat
  getter b : Mat
  getter dw : Mat
  getter db : Mat
  getter adam_w : AdamParam
  getter adam_b : AdamParam

  @input : Mat?
  @fwd_buf : Mat?  # pre-allocated forward output buffer
  @dx_buf : Mat?   # pre-allocated backward output buffer

  def initialize(d_in : Int32, d_out : Int32)
    scale = Math.sqrt(2.0 / d_in)
    @w = Mat.randn(d_in, d_out, scale)
    @b = Mat.new(1, d_out)
    @dw = Mat.new(d_in, d_out)
    @db = Mat.new(1, d_out)
    @adam_w = AdamParam.new(d_in, d_out)
    @adam_b = AdamParam.new(1, d_out)
  end

  def forward(x : Mat) : Mat
    @input = x
    buf = @fwd_buf
    if buf && buf.rows == x.rows && buf.cols == @w.cols
      MicroGPT.backend.matmul_into(x, @w, buf)
    else
      buf = x * @w
      @fwd_buf = buf
    end
    MicroGPT.backend.bias_add(buf, @b)
    buf
  end

  # Matmul only (no bias) — for use with fused_bias_relu
  def forward_matmul_only(x : Mat) : Mat
    @input = x
    buf = @fwd_buf
    if buf && buf.rows == x.rows && buf.cols == @w.cols
      MicroGPT.backend.matmul_into(x, @w, buf)
    else
      buf = x * @w
      @fwd_buf = buf
    end
    buf
  end

  def backward(grad : Mat) : Mat
    x = @input.not_nil!
    @dw = x.t * grad
    buf = @dx_buf
    if buf && buf.rows == grad.rows && buf.cols == @w.rows
      MicroGPT.backend.matmul_into(grad, @w.t, buf)
    else
      buf = grad * @w.t
      @dx_buf = buf
    end
    @db.zero!
    gd = grad.data
    dbd = @db.raw_data
    grad.rows.times do |i|
      grad.cols.times { |j| dbd[j] += gd[i * grad.cols + j] }
    end
    buf
  end

  def backward_accumulate(grad : Mat) : Mat
    x = @input.not_nil!
    @dw.add!(x.t * grad)
    buf = @dx_buf
    if buf && buf.rows == grad.rows && buf.cols == @w.rows
      MicroGPT.backend.matmul_into(grad, @w.t, buf)
    else
      buf = grad * @w.t
      @dx_buf = buf
    end
    gd = grad.data
    dbd = @db.raw_data
    grad.rows.times do |i|
      grad.cols.times { |j| dbd[j] += gd[i * grad.cols + j] }
    end
    buf
  end

  def update(lr : Float64)
    @adam_w.step(@w, @dw, lr)
    @adam_b.step(@b, @db, lr)
  end
end

# =============================================================================
# ReLU Activation
# =============================================================================

class ReLU
  include Layer

  @mask : Mat?

  def set_mask(m : Mat)
    @mask = m
  end

  def forward(x : Mat) : Mat
    output, mask = MicroGPT.backend.relu_forward(x)
    @mask = mask
    output
  end

  def backward(grad : Mat) : Mat
    MicroGPT.backend.relu_backward(grad, @mask.not_nil!)
  end

  def backward_accumulate(grad : Mat) : Mat
    backward(grad)
  end

  def update(lr : Float64) end
end

# =============================================================================
# Layer Normalization
# =============================================================================

class LayerNorm
  include Layer

  getter gamma : Mat
  getter beta : Mat
  getter dgamma : Mat
  getter dbeta : Mat
  getter adam_gamma : AdamParam
  getter adam_beta : AdamParam

  @input_norm : Mat?
  @std_inv : Mat?

  def initialize(d : Int32)
    @gamma = Mat.new(1, d)
    @beta = Mat.new(1, d)
    d.times { |i| @gamma[0, i] = 1.0 }
    @dgamma = Mat.new(1, d)
    @dbeta = Mat.new(1, d)
    @adam_gamma = AdamParam.new(1, d)
    @adam_beta = AdamParam.new(1, d)
  end

  def forward(x : Mat) : Mat
    output, norm, std_inv = MicroGPT.backend.layer_norm_forward(x, @gamma, @beta)
    @input_norm = norm
    @std_inv = std_inv
    output
  end

  def backward(grad : Mat) : Mat
    dx, @dgamma, @dbeta = MicroGPT.backend.layer_norm_backward(
      grad, @input_norm.not_nil!, @std_inv.not_nil!, @gamma)
    dx
  end

  def backward_accumulate(grad : Mat) : Mat
    dx, dg, db = MicroGPT.backend.layer_norm_backward(
      grad, @input_norm.not_nil!, @std_inv.not_nil!, @gamma)
    @dgamma.add!(dg)
    @dbeta.add!(db)
    dx
  end

  def update(lr : Float64)
    @adam_gamma.step(@gamma, @dgamma, lr)
    @adam_beta.step(@beta, @dbeta, lr)
  end
end

# =============================================================================
# Rotary Position Embedding (RoPE)
# =============================================================================
# Applies rotation to pairs of dimensions in Q/K based on position.
# For dim pair (2i, 2i+1) at position p:
#   q'[2i]   = q[2i]*cos(θ) - q[2i+1]*sin(θ)
#   q'[2i+1] = q[2i]*sin(θ) + q[2i+1]*cos(θ)
# where θ = p / 10000^(2i/d)
# No learned parameters — purely algorithmic.

class RoPE
  getter max_seq : Int32
  getter dim : Int32
  getter cos_cache : Mat  # [max_seq, dim]
  getter sin_cache : Mat  # [max_seq, dim]

  def initialize(@max_seq : Int32, @dim : Int32, base : Float64 = 10000.0)
    @cos_cache = Mat.new(@max_seq, @dim)
    @sin_cache = Mat.new(@max_seq, @dim)

    half = @dim // 2
    @max_seq.times do |pos|
      half.times do |i|
        theta = pos.to_f64 / (base ** (2.0 * i / @dim))
        c = Math.cos(theta).to_f32
        s = Math.sin(theta).to_f32
        @cos_cache[pos, 2 * i] = c
        @cos_cache[pos, 2 * i + 1] = c
        @sin_cache[pos, 2 * i] = s
        @sin_cache[pos, 2 * i + 1] = s
      end
    end
  end

  # Apply rotation in-place to a [seq_len, dim] matrix
  def apply!(x : Mat)
    if MicroGPT.backend.is_a?(CuBLASBackend)
      MicroGPT.backend.rope_apply(x, @cos_cache, @sin_cache)
      return
    end
    seq_len = x.rows
    half = @dim // 2
    seq_len.times do |pos|
      half.times do |i|
        c = @cos_cache[pos, 2 * i]
        s = @sin_cache[pos, 2 * i]
        x0 = x[pos, 2 * i]
        x1 = x[pos, 2 * i + 1]
        x[pos, 2 * i]     = x0 * c - x1 * s
        x[pos, 2 * i + 1] = x0 * s + x1 * c
      end
    end
  end

  # Inverse rotation (for backward pass): negate sin
  def apply_inverse!(x : Mat)
    if MicroGPT.backend.is_a?(CuBLASBackend)
      MicroGPT.backend.rope_apply_inverse(x, @cos_cache, @sin_cache)
      return
    end
    seq_len = x.rows
    half = @dim // 2
    seq_len.times do |pos|
      half.times do |i|
        c = @cos_cache[pos, 2 * i]
        s = @sin_cache[pos, 2 * i]
        x0 = x[pos, 2 * i]
        x1 = x[pos, 2 * i + 1]
        x[pos, 2 * i]     = x0 * c + x1 * s
        x[pos, 2 * i + 1] = -x0 * s + x1 * c
      end
    end
  end
end

# =============================================================================
# Single Attention Head (lightweight — no weights, just attention computation)
# =============================================================================

class AttentionHead
  getter head_dim : Int32
  property attn_weights : Mat?

  property q : Mat?
  property k : Mat?
  property v : Mat?

  def initialize(@head_dim : Int32)
  end

  # Forward: compute attention from pre-projected Q, K, V
  # If mask is provided, it's added to scores (pre-computed -inf pattern)
  def forward(q : Mat, k : Mat, v : Mat, mask : Mat? = nil) : Mat
    @q = q
    @k = k
    @v = v

    scale = 1.0 / Math.sqrt(@head_dim.to_f64)
    scores = q * k.t

    if m = mask
      scores.scale!(scale)
      scores.add!(m)
      weights = MicroGPT.backend.softmax_rows(scores)
    else
      # Fused: scale + causal mask + softmax (single kernel on GPU)
      weights = MicroGPT.backend.fused_attn_softmax(scores, scale)
    end

    @attn_weights = weights
    weights * v
  end

  # Backward: returns {dq, dk, dv} for the fused projection backward
  def backward(grad : Mat) : {Mat, Mat, Mat}
    q = @q.not_nil!
    k = @k.not_nil!
    v = @v.not_nil!
    w = @attn_weights.not_nil!
    scale = 1.0 / Math.sqrt(@head_dim.to_f64)

    dv = w.t * grad
    d_weights = grad * v.t
    # Fused: softmax backward + scale (single kernel on GPU)
    d_scores = MicroGPT.backend.fused_attn_softmax_backward(w, d_weights, scale)

    dq = d_scores * k
    dk = d_scores.t * q

    {dq, dk, dv}
  end
end

# =============================================================================
# Multi-Head Attention (fused Q/K/V projection, per-head attention)
# =============================================================================

class MultiHeadAttention
  include Layer
  include MathUtils

  getter heads : Array(AttentionHead)
  getter head_dims : Array(Int32)
  getter wq : Linear  # fused: d_model × d_model
  getter wk : Linear
  getter wv : Linear
  getter wo : Linear
  getter ropes : Array(RoPE)

  @uniform : Bool

  # Batched attention state (stored for backward)
  @q_parts : Array(Mat)?
  @k_parts : Array(Mat)?
  @v_parts : Array(Mat)?
  @weights_list : Array(Mat)?

  # Uniform heads
  def initialize(d_model : Int32, n_heads : Int32, max_seq_len : Int32 = 128)
    head_dim = d_model // n_heads
    @head_dims = Array.new(n_heads, head_dim)
    @heads = @head_dims.map { |dim| AttentionHead.new(dim) }
    @ropes = @head_dims.map { |dim| RoPE.new(max_seq_len, dim) }
    @wq = Linear.new(d_model, d_model)
    @wk = Linear.new(d_model, d_model)
    @wv = Linear.new(d_model, d_model)
    @wo = Linear.new(d_model, d_model)
    @uniform = true
  end

  # Heterogeneous heads — dimensions must sum to d_model
  def initialize(d_model : Int32, head_dims : Array(Int32), max_seq_len : Int32 = 128)
    total = head_dims.sum
    raise "head dims must sum to d_model (#{total} != #{d_model})" unless total == d_model
    @head_dims = head_dims
    @heads = head_dims.map { |dim| AttentionHead.new(dim) }
    @ropes = head_dims.map { |dim| RoPE.new(max_seq_len, dim) }
    @wq = Linear.new(d_model, d_model)
    @wk = Linear.new(d_model, d_model)
    @wv = Linear.new(d_model, d_model)
    @wo = Linear.new(d_model, d_model)
    @uniform = head_dims.all? { |d| d == head_dims[0] }
  end

  def forward(x : Mat, mask : Mat? = nil) : Mat
    # Fused projection: 3 matmuls instead of n_heads × 3
    q_all = @wq.forward(x)
    k_all = @wk.forward(x)
    v_all = @wv.forward(x)

    q_parts = split_cols(q_all, @head_dims)
    k_parts = split_cols(k_all, @head_dims)
    v_parts = split_cols(v_all, @head_dims)

    # Apply RoPE to Q and K (per-head, in-place)
    @heads.size.times do |i|
      @ropes[i].apply!(q_parts[i])
      @ropes[i].apply!(k_parts[i])
    end

    head_outputs = if @uniform && @heads.size > 1 && @heads.size <= 0  # DISABLED for benchmarking
      batched_attend_forward(q_parts, k_parts, v_parts)
    else
      result = Array(Mat).new(@heads.size)
      @heads.each_with_index do |head, i|
        result << head.forward(q_parts[i], k_parts[i], v_parts[i], mask)
      end
      # Store for backward (per-head path)
      @q_parts = q_parts
      @k_parts = k_parts
      @v_parts = v_parts
      result
    end

    concat = concat_cols(head_outputs)
    @wo.forward(concat)
  end

  def backward(grad : Mat) : Mat
    d_concat = @wo.backward(grad)
    head_grads = split_cols(d_concat, @head_dims)

    dq_all, dk_all, dv_all = if @uniform && @heads.size > 1 && @heads.size <= 0  # DISABLED for benchmarking
      batched_attend_backward(head_grads)
    else
      dq_parts = Array(Mat).new(@heads.size)
      dk_parts = Array(Mat).new(@heads.size)
      dv_parts = Array(Mat).new(@heads.size)
      @heads.each_with_index do |head, i|
        dq, dk, dv = head.backward(head_grads[i])
        # Inverse RoPE on gradients (rotation is orthogonal, so inverse = transpose = negate sin)
        @ropes[i].apply_inverse!(dq)
        @ropes[i].apply_inverse!(dk)
        dq_parts << dq
        dk_parts << dk
        dv_parts << dv
      end
      {concat_cols(dq_parts), concat_cols(dk_parts), concat_cols(dv_parts)}
    end

    dx_q = @wq.backward(dq_all)
    dx_k = @wk.backward(dk_all)
    dx_v = @wv.backward(dv_all)

    dx_q.add!(dx_k)
    dx_q.add!(dx_v)
    dx_q
  end

  # --- Batched attention (uniform heads only) ---

  private def batched_attend_forward(q_parts : Array(Mat), k_parts : Array(Mat),
                                      v_parts : Array(Mat)) : Array(Mat)
    n = @heads.size
    seq_len = q_parts[0].rows
    scale = 1.0 / Math.sqrt(@head_dims[0].to_f64)

    # Batched Q × K^T → n score matrices (seq_len × seq_len)
    scores_list = MicroGPT.backend.batched_matmul(q_parts, k_parts, false, true)

    # Stack all scores → one (n*seq_len × seq_len) matrix
    stacked = stack_mats(scores_list)
    stacked = stacked * scale
    MicroGPT.backend.causal_mask_batched(stacked, seq_len)
    stacked = MicroGPT.backend.softmax_rows(stacked)

    # Unstack back to per-head weights
    weights_list = unstack_mats(stacked, n, seq_len)

    # Store for backward + analysis
    @q_parts = q_parts
    @k_parts = k_parts
    @v_parts = v_parts
    @weights_list = weights_list
    @heads.each_with_index { |h, i| h.attn_weights = weights_list[i] }

    # Batched W × V → n output matrices (seq_len × head_dim)
    MicroGPT.backend.batched_matmul(weights_list, v_parts, false, false)
  end

  private def batched_attend_backward(head_grads : Array(Mat)) : {Mat, Mat, Mat}
    weights_list = @weights_list.not_nil!
    q_parts = @q_parts.not_nil!
    k_parts = @k_parts.not_nil!
    v_parts = @v_parts.not_nil!
    scale = 1.0 / Math.sqrt(@head_dims[0].to_f64)
    n = @heads.size
    seq_len = head_grads[0].rows

    # Batched dv = W^T × grad, d_weights = grad × V^T
    dv_list = MicroGPT.backend.batched_matmul(weights_list, head_grads, true, false)
    d_weights_list = MicroGPT.backend.batched_matmul(head_grads, v_parts, false, true)

    # Stacked softmax backward + scale
    stacked_w = stack_mats(weights_list)
    stacked_dw = stack_mats(d_weights_list)
    stacked_dscores = MicroGPT.backend.softmax_backward(stacked_w, stacked_dw)
    stacked_dscores = stacked_dscores * scale
    dscores_list = unstack_mats(stacked_dscores, n, seq_len)

    # Batched dq = dscores × K, dk = dscores^T × Q
    dq_list = MicroGPT.backend.batched_matmul(dscores_list, k_parts, false, false)
    dk_list = MicroGPT.backend.batched_matmul(dscores_list, q_parts, true, false)

    {concat_cols(dq_list), concat_cols(dk_list), concat_cols(dv_list)}
  end

  def backward_accumulate(grad : Mat) : Mat
    d_concat = @wo.backward_accumulate(grad)
    head_grads = split_cols(d_concat, @head_dims)

    dq_parts = Array(Mat).new(@heads.size)
    dk_parts = Array(Mat).new(@heads.size)
    dv_parts = Array(Mat).new(@heads.size)
    @heads.each_with_index do |head, i|
      dq, dk, dv = head.backward(head_grads[i])
      dq_parts << dq
      dk_parts << dk
      dv_parts << dv
    end
    dq_all = concat_cols(dq_parts)
    dk_all = concat_cols(dk_parts)
    dv_all = concat_cols(dv_parts)

    dx_q = @wq.backward_accumulate(dq_all)
    dx_k = @wk.backward_accumulate(dk_all)
    dx_v = @wv.backward_accumulate(dv_all)

    dx_q.add!(dx_k)
    dx_q.add!(dx_v)
    dx_q
  end

  def update(lr : Float64)
    @wq.update(lr)
    @wk.update(lr)
    @wv.update(lr)
    @wo.update(lr)
  end
end

# =============================================================================
# Feed-Forward Block
# =============================================================================

class FeedForward
  include Layer

  getter l1 : Linear
  getter l2 : Linear
  getter act : ReLU

  def initialize(d_model : Int32, d_ff : Int32)
    @l1 = Linear.new(d_model, d_ff)
    @l2 = Linear.new(d_ff, d_model)
    @act = ReLU.new
  end

  def forward(x : Mat) : Mat
    # Fused bias + ReLU: matmul → bias+relu in one kernel (saves 1 launch)
    h = @l1.forward_matmul_only(x)
    h, mask = MicroGPT.backend.fused_bias_relu(h, @l1.b)
    @act.set_mask(mask)
    @l2.forward(h)
  end

  def backward(grad : Mat) : Mat
    @l1.backward(@act.backward(@l2.backward(grad)))
  end

  def backward_accumulate(grad : Mat) : Mat
    @l1.backward_accumulate(@act.backward(@l2.backward_accumulate(grad)))
  end

  def update(lr : Float64)
    @l1.update(lr)
    @l2.update(lr)
  end
end

# =============================================================================
# Transformer Block
# =============================================================================

class TransformerBlock
  include Layer

  getter attn : MultiHeadAttention
  getter ff : FeedForward
  getter ln1 : LayerNorm
  getter ln2 : LayerNorm

  @res1_input : Mat?
  @res2_input : Mat?

  # Uniform heads
  def initialize(config : Config)
    @attn = MultiHeadAttention.new(config.d_model, config.n_heads, config.seq_len)
    @ff = FeedForward.new(config.d_model, config.d_ff)
    @ln1 = LayerNorm.new(config.d_model)
    @ln2 = LayerNorm.new(config.d_model)
  end

  # Heterogeneous heads
  def initialize(config : Config, head_dims : Array(Int32))
    @attn = MultiHeadAttention.new(config.d_model, head_dims, config.seq_len)
    @ff = FeedForward.new(config.d_model, config.d_ff)
    @ln1 = LayerNorm.new(config.d_model)
    @ln2 = LayerNorm.new(config.d_model)
  end

  # Pre-norm: x + attn(norm(x)), then x + ff(norm(x))
  def forward(x : Mat, mask : Mat? = nil) : Mat
    @res1_input = x
    a = @attn.forward(@ln1.forward(x), mask)
    a.add!(x)  # a += x, reuse a as residual output

    @res2_input = a
    f = @ff.forward(@ln2.forward(a))
    f.add!(a)  # f += a, reuse f as residual output
    f
  end

  def backward(grad : Mat) : Mat
    d_ff = @ln2.backward(@ff.backward(grad))
    grad.add!(d_ff)  # grad += d_ff in-place

    d_attn = @ln1.backward(@attn.backward(grad))
    grad.add!(d_attn)  # grad += d_attn in-place
    grad
  end

  def backward_accumulate(grad : Mat) : Mat
    d_ff = @ln2.backward_accumulate(@ff.backward_accumulate(grad))
    grad.add!(d_ff)

    d_attn = @ln1.backward_accumulate(@attn.backward_accumulate(grad))
    grad.add!(d_attn)
    grad
  end

  def update(lr : Float64)
    @attn.update(lr)
    @ff.update(lr)
    @ln1.update(lr)
    @ln2.update(lr)
  end
end

# =============================================================================
# Embedding (token + positional)
# =============================================================================

class Embedding
  getter token_emb : Mat
  getter d_token_emb : Mat
  getter adam_tok : AdamParam
  @last_ids : Array(Int32)?

  def initialize(vocab_size : Int32, d_model : Int32, @max_seq_len : Int32 = 128)
    scale = Math.sqrt(1.0 / d_model)
    @token_emb = Mat.randn(vocab_size, d_model, scale)
    @d_token_emb = Mat.new(vocab_size, d_model)
    @adam_tok = AdamParam.new(vocab_size, d_model)
  end

  def forward(ids : Array(Int32)) : Mat
    @last_ids = ids
    seq_len = ids.size
    d_model = @token_emb.cols
    b = MicroGPT.backend
    if b.is_a?(CuBLASBackend)
      b.embedding_gather(@token_emb, ids, seq_len, d_model)
    else
      result = Mat.new(seq_len, d_model)
      ids.each_with_index do |id, pos|
        d_model.times { |j| result[pos, j] = @token_emb[id, j] }
      end
      result
    end
  end

  def backward(grad : Mat)
    ids = @last_ids.not_nil!
    b = MicroGPT.backend
    if b.is_a?(CuBLASBackend)
      b.embedding_scatter_add(grad, ids, @d_token_emb, ids.size, @token_emb.cols)
    else
      @d_token_emb = Mat.new(@token_emb.rows, @token_emb.cols)
      ids.each_with_index do |id, pos|
        @token_emb.cols.times do |j|
          @d_token_emb[id, j] = @d_token_emb[id, j] + grad[pos, j]
        end
      end
    end
  end

  def backward_accumulate(grad : Mat)
    ids = @last_ids.not_nil!
    b = MicroGPT.backend
    if b.is_a?(CuBLASBackend)
      b.embedding_scatter_add(grad, ids, @d_token_emb, ids.size, @token_emb.cols)
    else
      ids.each_with_index do |id, pos|
        @token_emb.cols.times do |j|
          @d_token_emb[id, j] = @d_token_emb[id, j] + grad[pos, j]
        end
      end
    end
  end

  def update(lr : Float64)
    @adam_tok.step(@token_emb, @d_token_emb, lr)
  end
end

# =============================================================================
# Output Head + Loss
# =============================================================================

class OutputHead
  include MathUtils

  getter proj : Linear

  def initialize(d_model : Int32, vocab_size : Int32)
    @proj = Linear.new(d_model, vocab_size)
  end

  def forward(x : Mat) : Mat
    @proj.forward(x)
  end

  def loss_and_backward(logits : Mat, targets : Array(Int32)) : {Float64, Mat}
    seq_len = logits.rows
    vocab_size = logits.cols

    probs = MicroGPT.backend.softmax_rows(logits)

    loss = 0.0
    targets.each_with_index { |t, i| loss -= Math.log(probs[i, t] + 1e-10) }
    loss /= seq_len

    d_logits = Mat.new(seq_len, vocab_size)
    seq_len.times do |i|
      vocab_size.times { |j| d_logits[i, j] = probs[i, j] }
      d_logits[i, targets[i]] -= 1.0
    end
    d_logits.scale!(1.0 / seq_len)

    d_hidden = @proj.backward(d_logits)
    {loss, d_hidden}
  end

  def update(lr : Float64)
    @proj.update(lr)
  end
end

# =============================================================================
# Configuration
# =============================================================================

class Config
  property d_model : Int32 = 64
  property n_heads : Int32 = 4
  property n_layers : Int32 = 2
  property d_ff : Int32 = 256
  property vocab_size : Int32 = 0
  property seq_len : Int32 = 128
  property learning_rate : Float64 = 3e-4

  def head_dim : Int32
    d_model // n_heads
  end
end

# =============================================================================
# Character-Level Dataset
# =============================================================================

class CharDataset
  getter chars : Array(Char)
  getter char_to_id : Hash(Char, Int32)
  getter id_to_char : Hash(Int32, Char)
  getter data : Array(Int32)
  getter vocab_size : Int32
  getter stride : Int32
  @cursor : Int32 = 0
  @epoch : Int32 = 0

  def initialize(text : String, @stride : Int32 = 0)
    @chars = text.chars.uniq.sort
    @char_to_id = {} of Char => Int32
    @id_to_char = {} of Int32 => Char
    @chars.each_with_index do |c, i|
      @char_to_id[c] = i
      @id_to_char[i] = c
    end
    @vocab_size = @chars.size
    @data = text.chars.map { |c| @char_to_id[c] }
  end

  def sample(seq_len : Int32, lookahead : Int32 = 0) : {Array(Int32), Array(Array(Int32))}
    s = @stride > 0 ? @stride : seq_len
    needed = seq_len + 1 + lookahead
    if @cursor + needed > @data.size
      @cursor = 0
      @epoch += 1
    end
    input = @data[@cursor, seq_len]
    targets = Array(Array(Int32)).new(lookahead + 1) do |k|
      @data[@cursor + k + 1, seq_len]
    end
    @cursor += s
    {input, targets}
  end

  def epoch : Int32
    @epoch
  end

  def encode(text : String) : Array(Int32)
    text.chars.map { |c| @char_to_id[c] }
  end

  def decode(ids : Array(Int32)) : String
    ids.map { |id| @id_to_char[id] }.join
  end
end

# =============================================================================
# Bigram Table — P(next|current) from corpus statistics
# =============================================================================
# Zero-parameter lookup: given current token, returns probability distribution
# over next token. Built once from training data, never updated.

class BigramTable
  getter table : Mat  # [vocab_size, vocab_size] — row i = P(next|current=i)
  getter vocab_size : Int32

  def initialize(data : Array(Int32), @vocab_size : Int32)
    # Count bigram occurrences
    counts = Mat.new(@vocab_size, @vocab_size)
    (data.size - 1).times do |i|
      counts[data[i], data[i + 1]] += 1.0_f32
    end

    # Normalize rows to probabilities (with smoothing)
    @table = Mat.new(@vocab_size, @vocab_size)
    @vocab_size.times do |r|
      row_sum = 0.0_f32
      @vocab_size.times { |c| row_sum += counts[r, c] + 1.0_f32 }  # +1 Laplace smoothing
      @vocab_size.times do |c|
        @table[r, c] = (counts[r, c] + 1.0_f32) / row_sum
      end
    end
  end

  # Lookup: returns [seq_len, vocab_size] matrix of bigram distributions
  def lookup(input_ids : Array(Int32)) : Mat
    seq_len = input_ids.size
    result = Mat.new(seq_len, @vocab_size)
    seq_len.times do |pos|
      tok = input_ids[pos]
      @vocab_size.times { |j| result[pos, j] = @table[tok, j] }
    end
    result
  end
end

class TrigramTable
  getter vocab_size : Int32
  @table : Array(Float32)  # flat [vocab_size * vocab_size * vocab_size] — P(next|prev,current)

  def initialize(data : Array(Int32), @vocab_size : Int32)
    v = @vocab_size
    counts = Array(Float32).new(v * v * v, 0.0_f32)

    # Count trigram occurrences
    (data.size - 2).times do |i|
      prev = data[i]
      cur = data[i + 1]
      nxt = data[i + 2]
      counts[prev * v * v + cur * v + nxt] += 1.0_f32
    end

    # Normalize: for each (prev, cur), normalize over next
    @table = Array(Float32).new(v * v * v, 0.0_f32)
    v.times do |prev|
      v.times do |cur|
        base = prev * v * v + cur * v
        row_sum = 0.0_f32
        v.times { |nxt| row_sum += counts[base + nxt] + 1.0_f32 }
        v.times do |nxt|
          @table[base + nxt] = (counts[base + nxt] + 1.0_f32) / row_sum
        end
      end
    end
    STDERR.puts "Trigram table built: #{v}×#{v}×#{v} (#{@table.size} entries)"
  end

  # Lookup: returns [seq_len, vocab_size] matrix of trigram distributions
  # At pos 0, falls back to uniform (no previous context)
  def lookup(input_ids : Array(Int32)) : Mat
    seq_len = input_ids.size
    v = @vocab_size
    result = Mat.new(seq_len, v)
    uniform = 1.0_f32 / v
    seq_len.times do |pos|
      if pos == 0
        v.times { |j| result[pos, j] = uniform }
      else
        prev = input_ids[pos - 1]
        cur = input_ids[pos]
        base = prev * v * v + cur * v
        v.times { |j| result[pos, j] = @table[base + j] }
      end
    end
    result
  end
end

# Calculator expert: parses "X + Y = Z" lines and predicts answer characters
# Returns [seq_len, vocab_size] like BigramTable — peaked at correct next char
class CalculatorExpert
  getter vocab_size : Int32
  @id_to_char : Hash(Int32, Char)
  @char_to_id : Hash(Char, Int32)

  def initialize(@vocab_size : Int32, @id_to_char : Hash(Int32, Char), @char_to_id : Hash(Char, Int32))
  end

  def lookup(input_ids : Array(Int32)) : Mat
    seq_len = input_ids.size
    v = @vocab_size
    result = Mat.new(seq_len, v)
    uniform = 1.0_f32 / v

    # Decode input to chars for parsing
    chars = input_ids.map { |id| @id_to_char.fetch(id, '\0') }

    # Find all "X + Y = " patterns in the window and mark answer positions
    # For each position, we want: what is the correct NEXT character?
    # Build a map: position → predicted next char
    predictions = Array(Char?).new(seq_len, nil)

    # Scan for complete equations by finding '=' positions
    seq_len.times do |eq_pos|
      next unless chars[eq_pos] == '='

      # Parse backward from '=' to find "X + Y"
      # Expected: "...X + Y =" with spaces
      # Scan left past space before '='
      p = eq_pos - 1
      next if p < 0 || chars[p] != ' '

      # Parse Y (right operand) — scan left past digits
      p -= 1
      y_end = p
      while p >= 0 && chars[p].ascii_number?
        p -= 1
      end
      y_start = p + 1
      next if y_start > y_end

      # Expect " + " before Y
      next if p < 0 || chars[p] != ' '
      p -= 1
      next if p < 0 || chars[p] != '+'
      p -= 1
      next if p < 0 || chars[p] != ' '

      # Parse X (left operand)
      p -= 1
      x_end = p
      while p >= 0 && chars[p].ascii_number?
        p -= 1
      end
      x_start = p + 1
      next if x_start > x_end

      # Compute answer
      x_str = String.build { |s| (x_start..x_end).each { |i| s << chars[i] } }
      y_str = String.build { |s| (y_start..y_end).each { |i| s << chars[i] } }
      x_val = x_str.to_i? || next
      y_val = y_str.to_i? || next
      answer = x_val + y_val
      answer_str = " #{answer}\n"

      # Mark predictions: at position eq_pos, next char is answer_str[0] = ' '
      # at position eq_pos+1, next char is answer_str[1] = first digit, etc.
      answer_str.size.times do |k|
        target_pos = eq_pos + k
        break if target_pos >= seq_len
        predictions[target_pos] = answer_str[k]
      end
    end

    # Build output matrix
    seq_len.times do |pos|
      if (pred_char = predictions[pos]) && (pred_id = @char_to_id[pred_char]?)
        # Peaked distribution on correct next character
        smooth = 0.01_f32 / v
        v.times { |j| result[pos, j] = smooth }
        result[pos, pred_id] = 0.99_f32
      else
        # Uniform — no prediction
        v.times { |j| result[pos, j] = uniform }
      end
    end

    result
  end
end

# =============================================================================
# The Model
# =============================================================================

class MiniGPT
  include MathUtils

  getter config : Config
  getter embedding : Embedding
  getter blocks : Array(TransformerBlock)
  getter final_norm : LayerNorm
  getter output : OutputHead

  # Uniform heads
  def initialize(@config : Config)
    @embedding = Embedding.new(config.vocab_size, config.d_model, config.seq_len)
    @blocks = Array(TransformerBlock).new(config.n_layers) { TransformerBlock.new(config) }
    @final_norm = LayerNorm.new(config.d_model)
    @output = OutputHead.new(config.d_model, config.vocab_size)
  end

  # Heterogeneous heads
  def initialize(@config : Config, head_dims : Array(Int32))
    @embedding = Embedding.new(config.vocab_size, config.d_model, config.seq_len)
    @blocks = Array(TransformerBlock).new(config.n_layers) { TransformerBlock.new(config, head_dims) }
    @final_norm = LayerNorm.new(config.d_model)
    @output = OutputHead.new(config.d_model, config.vocab_size)
  end

  def forward(ids : Array(Int32), mask : Mat? = nil) : Mat
    x = @embedding.forward(ids)
    @blocks.each { |b| x = b.forward(x, mask) }
    @output.forward(@final_norm.forward(x))
  end

  def train_step(input_ids : Array(Int32), target_ids : Array(Int32)) : Float64
    logits = forward(input_ids)
    loss, grad = @output.loss_and_backward(logits, target_ids)

    grad = @final_norm.backward(grad)
    @blocks.reverse_each { |b| grad = b.backward(grad) }
    @embedding.backward(grad)

    lr = @config.learning_rate
    @embedding.update(lr)
    @blocks.each &.update(lr)
    @final_norm.update(lr)
    @output.update(lr)

    loss
  end

  def generate(start_ids : Array(Int32), max_tokens : Int32, temperature : Float64 = 1.0) : Array(Int32)
    ids = start_ids.dup
    max_tokens.times do
      context = ids.size > @config.seq_len ? ids[-@config.seq_len..] : ids
      logits = forward(context)

      last_row = logits.rows - 1
      vocab_size = logits.cols

      if temperature <= 0.01
        # Greedy
        best_id = 0
        best_val = logits[last_row, 0]
        vocab_size.times do |j|
          if logits[last_row, j] > best_val
            best_val = logits[last_row, j]
            best_id = j
          end
        end
        ids << best_id
      else
        # Temperature sampling
        scaled = Mat.new(1, vocab_size)
        vocab_size.times { |j| scaled[0, j] = logits[last_row, j] / temperature }
        probs = MicroGPT.backend.softmax_rows(scaled)

        # Sample from distribution
        r = rand
        cumulative = 0.0
        chosen = vocab_size - 1
        vocab_size.times do |j|
          cumulative += probs[0, j]
          if r <= cumulative
            chosen = j
            break
          end
        end
        ids << chosen
      end
    end
    ids
  end

  def param_count : Int64
    count = 0_i64
    count += @embedding.token_emb.data.size
    @blocks.each do |b|
      count += b.attn.wq.w.data.size + b.attn.wq.b.data.size
      count += b.attn.wk.w.data.size + b.attn.wk.b.data.size
      count += b.attn.wv.w.data.size + b.attn.wv.b.data.size
      count += b.attn.wo.w.data.size + b.attn.wo.b.data.size
      count += b.ff.l1.w.data.size + b.ff.l1.b.data.size
      count += b.ff.l2.w.data.size + b.ff.l2.b.data.size
      count += b.ln1.gamma.data.size + b.ln1.beta.data.size
      count += b.ln2.gamma.data.size + b.ln2.beta.data.size
    end
    count += @final_norm.gamma.data.size + @final_norm.beta.data.size
    count += @output.proj.w.data.size + @output.proj.b.data.size
    count
  end

  # Collect all weight matrices in deterministic order
  private def weight_mats : Array(Mat)
    mats = [] of Mat
    mats << @embedding.token_emb
    @blocks.each do |b|
      mats << b.attn.wq.w << b.attn.wq.b
      mats << b.attn.wk.w << b.attn.wk.b
      mats << b.attn.wv.w << b.attn.wv.b
      mats << b.attn.wo.w << b.attn.wo.b
      mats << b.ln1.gamma << b.ln1.beta
      mats << b.ff.l1.w << b.ff.l1.b
      mats << b.ff.l2.w << b.ff.l2.b
      mats << b.ln2.gamma << b.ln2.beta
    end
    mats << @final_norm.gamma << @final_norm.beta
    mats << @output.proj.w << @output.proj.b
    mats
  end

  def save(path : String)
    File.open(path, "wb") do |f|
      # Header: magic + config
      f.write_bytes(0x4D475054_u32, IO::ByteFormat::LittleEndian) # "MGPT"
      f.write_bytes(@config.d_model.to_i32, IO::ByteFormat::LittleEndian)
      f.write_bytes(@config.n_heads.to_i32, IO::ByteFormat::LittleEndian)
      f.write_bytes(@config.n_layers.to_i32, IO::ByteFormat::LittleEndian)
      f.write_bytes(@config.d_ff.to_i32, IO::ByteFormat::LittleEndian)
      f.write_bytes(@config.vocab_size.to_i32, IO::ByteFormat::LittleEndian)
      f.write_bytes(@config.seq_len.to_i32, IO::ByteFormat::LittleEndian)

      # Weight matrices
      weight_mats.each do |mat|
        f.write_bytes(mat.rows.to_i32, IO::ByteFormat::LittleEndian)
        f.write_bytes(mat.cols.to_i32, IO::ByteFormat::LittleEndian)
        mat.raw_data.each do |v|
          f.write_bytes(v, IO::ByteFormat::LittleEndian)
        end
      end
    end
  end

  def load(path : String)
    File.open(path, "rb") do |f|
      magic = f.read_bytes(UInt32, IO::ByteFormat::LittleEndian)
      raise "Not a MicroGPT model file" unless magic == 0x4D475054_u32

      d_model = f.read_bytes(Int32, IO::ByteFormat::LittleEndian)
      n_heads = f.read_bytes(Int32, IO::ByteFormat::LittleEndian)
      n_layers = f.read_bytes(Int32, IO::ByteFormat::LittleEndian)
      d_ff = f.read_bytes(Int32, IO::ByteFormat::LittleEndian)
      vocab_size = f.read_bytes(Int32, IO::ByteFormat::LittleEndian)
      seq_len = f.read_bytes(Int32, IO::ByteFormat::LittleEndian)

      unless d_model == @config.d_model && n_layers == @config.n_layers && vocab_size == @config.vocab_size
        raise "Config mismatch: checkpoint has d_model=#{d_model} n_layers=#{n_layers} vocab=#{vocab_size}, " \
              "but model has d_model=#{@config.d_model} n_layers=#{@config.n_layers} vocab=#{@config.vocab_size}"
      end

      weight_mats.each do |mat|
        rows = f.read_bytes(Int32, IO::ByteFormat::LittleEndian)
        cols = f.read_bytes(Int32, IO::ByteFormat::LittleEndian)
        raise "Shape mismatch: expected #{mat.rows}x#{mat.cols}, got #{rows}x#{cols}" unless rows == mat.rows && cols == mat.cols
        (rows * cols).times do |i|
          mat.raw_data[i] = f.read_bytes(Float32, IO::ByteFormat::LittleEndian)
        end
      end
    end
  end
end

end


