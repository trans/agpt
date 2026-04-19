module MicroGPT

# =============================================================================
# Backend Adapter Interface
# =============================================================================

module Backend
  # Core linear algebra
  abstract def matmul(a : Mat, b : Mat) : Mat
  abstract def matmul_into(a : Mat, b : Mat, dst : Mat)
  abstract def transpose(a : Mat) : Mat
  abstract def add(a : Mat, b : Mat) : Mat
  abstract def add!(a : Mat, b : Mat)           # a += b in-place
  abstract def scale(a : Mat, scalar : Float32) : Mat
  abstract def scale!(a : Mat, scalar : Float32) # a *= scalar in-place
  abstract def release(a : Mat)
  abstract def batched_matmul(a_list : Array(Mat), b_list : Array(Mat),
                              transpose_a : Bool, transpose_b : Bool) : Array(Mat)

  # GPU memory management
  abstract def gpu_alloc(size : UInt64) : Pointer(Void)
  abstract def gpu_free(ptr : Pointer(Void))
  abstract def gpu_upload(dst : Pointer(Void), src : Pointer(Void), size : UInt64)
  abstract def gpu_download(dst : Pointer(Void), src : Pointer(Void), size : UInt64)

  # Element-wise / reduction operations
  abstract def softmax_rows(x : Mat) : Mat
  abstract def softmax_backward(s : Mat, ds : Mat) : Mat
  abstract def causal_mask(scores : Mat)        # in-place: upper triangle → -1e9
  abstract def causal_mask_batched(scores : Mat, seq_len : Int32)  # in-place: mask repeats per seq_len block
  abstract def bias_add(data : Mat, bias : Mat)  # in-place: data[i,j] += bias[0,j]
  abstract def relu_forward(x : Mat) : {Mat, Mat}  # → {output, mask}
  abstract def relu_backward(grad : Mat, mask : Mat) : Mat
  abstract def layer_norm_forward(x : Mat, gamma : Mat, beta : Mat) : {Mat, Mat, Mat}  # → {output, norm, std_inv}
  abstract def layer_norm_backward(grad : Mat, norm : Mat, std_inv : Mat, gamma : Mat) : {Mat, Mat, Mat}  # → {dx, dgamma, dbeta}
  abstract def adam_step(param : Mat, grad : Mat, m : Mat, v : Mat, lr : Float64, t : Int32)
  # RoPE: apply/inverse on GPU if available, default is no-op (handled by RoPE class on CPU)
  def rope_apply(x : Mat, cos_cache : Mat, sin_cache : Mat)
    # Default: no-op, RoPE class handles it on CPU
  end

  def rope_apply_inverse(x : Mat, cos_cache : Mat, sin_cache : Mat)
    # Default: no-op, RoPE class handles it on CPU
  end

  # Fused attention: scale + causal mask + softmax in one pass
  # Default implementation falls back to separate ops
  def fused_attn_softmax(scores : Mat, scale : Float64) : Mat
    scores.scale!(scale)
    causal_mask(scores)
    softmax_rows(scores)
  end

  def fused_attn_softmax_backward(s : Mat, ds : Mat, scale : Float64) : Mat
    d_scores = softmax_backward(s, ds)
    d_scores.scale!(scale)
    d_scores
  end

  # Fused bias + ReLU: add bias then apply ReLU in one pass
  # Default falls back to separate ops
  def fused_bias_relu(x : Mat, bias : Mat) : {Mat, Mat}
    bias_add(x, bias)
    relu_forward(x)
  end

  abstract def sync
end

# =============================================================================
# Pure Crystal Backend (default)
# =============================================================================

class CrystalBackend
  include Backend

  def release(a : Mat) end
  def gpu_alloc(size : UInt64) : Pointer(Void); Pointer(Void).null; end
  def gpu_free(ptr : Pointer(Void)) end
  def gpu_upload(dst : Pointer(Void), src : Pointer(Void), size : UInt64) end
  def gpu_download(dst : Pointer(Void), src : Pointer(Void), size : UInt64) end
  def sync
  end

  def matmul(a : Mat, b : Mat) : Mat
    raise "dimension mismatch: #{a.cols} vs #{b.rows}" unless a.cols == b.rows
    result = Mat.new(a.rows, b.cols)
    matmul_into(a, b, result)
    result
  end

  def matmul_into(a : Mat, b : Mat, dst : Mat)
    raise "dimension mismatch: #{a.cols} vs #{b.rows}" unless a.cols == b.rows
    ad = a.data
    bd = b.data
    rd = dst.raw_data
    a.rows.times do |i|
      b.cols.times do |j|
        sum = 0.0_f32
        a.cols.times do |k|
          sum += ad[i * a.cols + k] * bd[k * b.cols + j]
        end
        rd[i * b.cols + j] = sum
      end
    end
  end

  def transpose(a : Mat) : Mat
    result = Mat.new(a.cols, a.rows)
    ad = a.data
    rd = result.raw_data
    a.rows.times do |i|
      a.cols.times do |j|
        rd[j * a.rows + i] = ad[i * a.cols + j]
      end
    end
    result
  end

  def add(a : Mat, b : Mat) : Mat
    raise "dimension mismatch" unless a.rows == b.rows && a.cols == b.cols
    result = Mat.new(a.rows, a.cols)
    ad = a.data
    bd = b.data
    rd = result.raw_data
    ad.size.times { |i| rd[i] = ad[i] + bd[i] }
    result
  end

  def add!(a : Mat, b : Mat)
    ad = a.raw_data
    bd = b.data
    ad.size.times { |i| ad[i] += bd[i] }
    a.invalidate_gpu
  end

  def scale(a : Mat, scalar : Float32) : Mat
    result = Mat.new(a.rows, a.cols)
    ad = a.data
    rd = result.raw_data
    ad.size.times { |i| rd[i] = ad[i] * scalar }
    result
  end

  def scale!(a : Mat, scalar : Float32)
    ad = a.raw_data
    ad.size.times { |i| ad[i] *= scalar }
    a.invalidate_gpu
  end

  def batched_matmul(a_list : Array(Mat), b_list : Array(Mat),
                     transpose_a : Bool, transpose_b : Bool) : Array(Mat)
    a_list.zip(b_list).map do |a, b|
      aa = transpose_a ? transpose(a) : a
      bb = transpose_b ? transpose(b) : b
      matmul(aa, bb)
    end
  end

  def softmax_rows(x : Mat) : Mat
    result = Mat.new(x.rows, x.cols)
    xd = x.data
    rd = result.raw_data
    x.rows.times do |i|
      off = i * x.cols
      max = xd[off]
      x.cols.times { |j| v = xd[off + j]; max = v if v > max }
      sum = 0.0_f32
      x.cols.times do |j|
        rd[off + j] = Math.exp(xd[off + j] - max).to_f32
        sum += rd[off + j]
      end
      x.cols.times { |j| rd[off + j] /= sum }
    end
    result
  end

  def softmax_backward(s : Mat, ds : Mat) : Mat
    result = Mat.new(s.rows, s.cols)
    sd = s.data
    dsd = ds.data
    rd = result.raw_data
    s.rows.times do |i|
      off = i * s.cols
      dot = 0.0_f32
      s.cols.times { |j| dot += dsd[off + j] * sd[off + j] }
      s.cols.times { |j| rd[off + j] = sd[off + j] * (dsd[off + j] - dot) }
    end
    result
  end

  def causal_mask(scores : Mat)
    scores.rows.times do |i|
      ((i + 1)...scores.cols).each do |j|
        scores[i, j] = -1e9_f32
      end
    end
  end

  def causal_mask_batched(scores : Mat, seq_len : Int32)
    sd = scores.data
    scores.rows.times do |i|
      pos = i % seq_len
      ((pos + 1)...scores.cols).each do |j|
        sd[i * scores.cols + j] = -1e9_f32
      end
    end
    scores.invalidate_gpu
  end

  def bias_add(data : Mat, bias : Mat)
    dd = data.data
    bd = bias.data
    data.rows.times do |i|
      data.cols.times do |j|
        dd[i * data.cols + j] += bd[j]
      end
    end
    data.invalidate_gpu
  end

  def relu_forward(x : Mat) : {Mat, Mat}
    n = x.rows * x.cols
    output = Mat.new(x.rows, x.cols)
    mask = Mat.new(x.rows, x.cols)
    xd = x.data
    od = output.raw_data
    md = mask.raw_data
    n.times do |i|
      if xd[i] > 0.0
        od[i] = xd[i]
        md[i] = 1.0
      end
    end
    {output, mask}
  end

  def relu_backward(grad : Mat, mask : Mat) : Mat
    result = Mat.new(grad.rows, grad.cols)
    gd = grad.data
    md = mask.data
    rd = result.raw_data
    gd.size.times { |i| rd[i] = gd[i] * md[i] }
    result
  end

  def layer_norm_forward(x : Mat, gamma : Mat, beta : Mat) : {Mat, Mat, Mat}
    eps = 1e-5_f32
    output = Mat.new(x.rows, x.cols)
    norm = Mat.new(x.rows, x.cols)
    std_inv = Mat.new(x.rows, 1)
    xd = x.data
    gd = gamma.data
    bd = beta.data
    od = output.raw_data
    nd = norm.raw_data
    sd = std_inv.raw_data

    x.rows.times do |i|
      off = i * x.cols
      mean = 0.0_f32
      x.cols.times { |j| mean += xd[off + j] }
      mean /= x.cols

      var = 0.0_f32
      x.cols.times { |j| d = xd[off + j] - mean; var += d * d }
      var /= x.cols

      inv = (1.0 / Math.sqrt(var + eps)).to_f32
      sd[i] = inv

      x.cols.times do |j|
        nd[off + j] = (xd[off + j] - mean) * inv
        od[off + j] = nd[off + j] * gd[j] + bd[j]
      end
    end

    {output, norm, std_inv}
  end

  def layer_norm_backward(grad : Mat, norm : Mat, std_inv : Mat, gamma : Mat) : {Mat, Mat, Mat}
    n = grad.cols.to_f32
    dx = Mat.new(grad.rows, grad.cols)
    dgamma = Mat.new(1, grad.cols)
    dbeta = Mat.new(1, grad.cols)
    grd = grad.data
    nrd = norm.data
    sid = std_inv.data
    gad = gamma.data
    dxd = dx.raw_data
    dgd = dgamma.raw_data
    dbd = dbeta.raw_data

    grad.rows.times do |i|
      off = i * grad.cols
      grad.cols.times do |j|
        dgd[j] += grd[off + j] * nrd[off + j]
        dbd[j] += grd[off + j]
      end
    end

    grad.rows.times do |i|
      off = i * grad.cols
      dot = 0.0_f32
      sum = 0.0_f32
      grad.cols.times do |j|
        gn = grd[off + j] * gad[j]
        dot += gn * nrd[off + j]
        sum += gn
      end
      sinv = sid[i]
      grad.cols.times do |j|
        gn = grd[off + j] * gad[j]
        dxd[off + j] = sinv * (gn - (nrd[off + j] * dot + sum) / n)
      end
    end

    {dx, dgamma, dbeta}
  end

  def adam_step(param : Mat, grad : Mat, m : Mat, v : Mat, lr : Float64, t : Int32)
    beta1 = 0.9_f32
    beta2 = 0.999_f32
    eps = 1e-8_f32
    bc1 = (1.0 - 0.9 ** t).to_f32
    bc2 = (1.0 - 0.999 ** t).to_f32
    lr32 = lr.to_f32
    pd = param.data
    gd = grad.data
    md = m.data
    vd = v.data
    pd.size.times do |i|
      md[i] = beta1 * md[i] + (1.0_f32 - beta1) * gd[i]
      vd[i] = beta2 * vd[i] + (1.0_f32 - beta2) * gd[i] * gd[i]
      m_hat = md[i] / bc1
      v_hat = vd[i] / bc2
      pd[i] -= lr32 * m_hat / (Math.sqrt(v_hat).to_f32 + eps)
    end
    param.invalidate_gpu
    m.invalidate_gpu
    v.invalidate_gpu
  end
end

# =============================================================================
# OpenBLAS Backend (inherits CPU element-wise ops, overrides BLAS linear algebra)
# =============================================================================

@[Link(ldflags: "-lopenblas_64")]
lib LibCBLAS
  enum Order
    RowMajor = 101
    ColMajor = 102
  end

  enum Transpose
    NoTrans   = 111
    Trans     = 112
    ConjTrans = 113
  end

  fun cblas_sgemm(order : Order, trans_a : Transpose, trans_b : Transpose,
                  m : Int64, n : Int64, k : Int64,
                  alpha : Float32, a : Float32*, lda : Int64,
                  b : Float32*, ldb : Int64,
                  beta : Float32, c : Float32*, ldc : Int64)
  fun cblas_saxpy(n : Int64, alpha : Float32, x : Float32*, incx : Int64,
                  y : Float32*, incy : Int64)
  fun cblas_sscal(n : Int64, alpha : Float32, x : Float32*, incx : Int64)
  fun cblas_scopy(n : Int64, x : Float32*, incx : Int64, y : Float32*, incy : Int64)
end

class OpenBLASBackend < CrystalBackend
  def matmul(a : Mat, b : Mat) : Mat
    raise "dimension mismatch: #{a.cols} vs #{b.rows}" unless a.cols == b.rows
    result = Mat.new(a.rows, b.cols)
    matmul_into(a, b, result)
    result
  end

  def matmul_into(a : Mat, b : Mat, dst : Mat)
    raise "dimension mismatch: #{a.cols} vs #{b.rows}" unless a.cols == b.rows
    m = a.rows.to_i64
    n = b.cols.to_i64
    k = a.cols.to_i64

    LibCBLAS.cblas_sgemm(
      LibCBLAS::Order::RowMajor,
      LibCBLAS::Transpose::NoTrans,
      LibCBLAS::Transpose::NoTrans,
      m, n, k,
      1.0_f32, a.data.to_unsafe, k,
      b.data.to_unsafe, n,
      0.0_f32, dst.raw_data.to_unsafe, n
    )
  end

  def add(a : Mat, b : Mat) : Mat
    raise "dimension mismatch" unless a.rows == b.rows && a.cols == b.cols
    n = a.data.size.to_i64
    result = Mat.new(a.rows, a.cols)
    LibCBLAS.cblas_scopy(n, a.data.to_unsafe, 1_i64, result.raw_data.to_unsafe, 1_i64)
    LibCBLAS.cblas_saxpy(n, 1.0_f32, b.data.to_unsafe, 1_i64, result.raw_data.to_unsafe, 1_i64)
    result
  end

  def add!(a : Mat, b : Mat)
    n = a.data.size.to_i64
    LibCBLAS.cblas_saxpy(n, 1.0_f32, b.data.to_unsafe, 1_i64, a.raw_data.to_unsafe, 1_i64)
    a.invalidate_gpu
  end

  def scale(a : Mat, scalar : Float32) : Mat
    n = a.data.size.to_i64
    result = Mat.new(a.rows, a.cols)
    LibCBLAS.cblas_scopy(n, a.data.to_unsafe, 1_i64, result.raw_data.to_unsafe, 1_i64)
    LibCBLAS.cblas_sscal(n, scalar, result.raw_data.to_unsafe, 1_i64)
    result
  end

  def scale!(a : Mat, scalar : Float32)
    n = a.data.size.to_i64
    LibCBLAS.cblas_sscal(n, scalar, a.raw_data.to_unsafe, 1_i64)
    a.invalidate_gpu
  end

  def batched_matmul(a_list : Array(Mat), b_list : Array(Mat),
                     transpose_a : Bool, transpose_b : Bool) : Array(Mat)
    ta = transpose_a ? LibCBLAS::Transpose::Trans : LibCBLAS::Transpose::NoTrans
    tb = transpose_b ? LibCBLAS::Transpose::Trans : LibCBLAS::Transpose::NoTrans

    a_list.zip(b_list).map do |a, b|
      m = (transpose_a ? a.cols : a.rows).to_i64
      n = (transpose_b ? b.rows : b.cols).to_i64
      k = (transpose_a ? a.rows : a.cols).to_i64

      result = Mat.new(m.to_i32, n.to_i32)

      LibCBLAS.cblas_sgemm(
        LibCBLAS::Order::RowMajor, ta, tb,
        m, n, k,
        1.0_f32, a.data.to_unsafe, a.cols.to_i64,
        b.data.to_unsafe, b.cols.to_i64,
        0.0_f32, result.raw_data.to_unsafe, n
      )

      result
    end
  end
end

# =============================================================================
# cuBLAS Backend
# =============================================================================

@[Link(ldflags: "-L/opt/cuda/lib64 -lcudart -lcublas")]
lib LibCUDA
  alias CublasHandle = Void*

  enum CublasStatus
    Success         =  0
    NotInitialized  =  1
    AllocFailed     =  3
    InvalidValue    =  7
    ArchMismatch    =  8
    MappingError    = 11
    ExecutionFailed = 13
    InternalError   = 14
  end

  enum CublasOperation
    N = 0  # No transpose
    T = 1  # Transpose
    C = 2  # Conjugate transpose
  end

  # cuBLAS handle management
  fun cublasCreate_v2(handle : CublasHandle*) : CublasStatus
  fun cublasDestroy_v2(handle : CublasHandle) : CublasStatus

  # C = alpha * op(A) * op(B) + beta * C
  fun cublasSgemm_v2(handle : CublasHandle,
                     transa : CublasOperation, transb : CublasOperation,
                     m : Int32, n : Int32, k : Int32,
                     alpha : Float32*, a : Float32*, lda : Int32,
                     b : Float32*, ldb : Int32,
                     beta : Float32*, c : Float32*, ldc : Int32) : CublasStatus

  # Batched GEMM
  fun cublasSgemmBatched(handle : CublasHandle,
                         transa : CublasOperation, transb : CublasOperation,
                         m : Int32, n : Int32, k : Int32,
                         alpha : Float32*,
                         a_array : Float32**, lda : Int32,
                         b_array : Float32**, ldb : Int32,
                         beta : Float32*,
                         c_array : Float32**, ldc : Int32,
                         batch_count : Int32) : CublasStatus

  # C = alpha * op(A) + beta * op(B)
  fun cublasSgeam(handle : CublasHandle,
                  transa : CublasOperation, transb : CublasOperation,
                  m : Int32, n : Int32,
                  alpha : Float32*, a : Float32*, lda : Int32,
                  beta : Float32*, b : Float32*, ldb : Int32,
                  c : Float32*, ldc : Int32) : CublasStatus

  # x = alpha * x
  fun cublasSscal_v2(handle : CublasHandle,
                     n : Int32, alpha : Float32*,
                     x : Float32*, incx : Int32) : CublasStatus

  # y = alpha * x + y
  fun cublasSaxpy_v2(handle : CublasHandle,
                     n : Int32, alpha : Float32*,
                     x : Float32*, incx : Int32,
                     y : Float32*, incy : Int32) : CublasStatus

  # CUDA memory management
  fun cudaMalloc(devptr : Void**, size : UInt64) : Int32
  fun cudaFree(devptr : Void*) : Int32
  fun cudaMemcpy(dst : Void*, src : Void*, count : UInt64, kind : Int32) : Int32
  fun cudaMemset(devptr : Void*, value : Int32, count : UInt64) : Int32
end

# CUDA custom kernel bindings (from src/cuda/kernels.cu, linked via build/kernels.o)
lib LibCUDAKernels
  fun cuda_softmax_rows(input : Float32*, output : Float32*, rows : Int32, cols : Int32)
  fun cuda_softmax_backward(s : Float32*, ds : Float32*, result : Float32*, rows : Int32, cols : Int32)
  fun cuda_causal_mask(data : Float32*, n : Int32)
  fun cuda_causal_mask_batched(data : Float32*, total_rows : Int32, cols : Int32, seq_len : Int32)
  fun cuda_bias_add(data : Float32*, bias : Float32*, rows : Int32, cols : Int32)
  fun cuda_relu_forward(input : Float32*, output : Float32*, mask : Float32*, n : Int32)
  fun cuda_relu_backward(grad : Float32*, mask : Float32*, output : Float32*, n : Int32)
  fun cuda_layer_norm_forward(input : Float32*, output : Float32*, norm_out : Float32*,
                               std_inv_out : Float32*, gamma : Float32*, beta : Float32*,
                               rows : Int32, cols : Int32)
  fun cuda_layer_norm_backward(grad : Float32*, norm : Float32*, std_inv : Float32*,
                                gamma : Float32*, dx : Float32*, dgamma : Float32*,
                                dbeta : Float32*, rows : Int32, cols : Int32)
  fun cuda_adam_step(param : Float32*, grad : Float32*, m : Float32*, v : Float32*,
                      lr : Float32, beta1 : Float32, beta2 : Float32, eps : Float32,
                      t : Int32, n : Int32)
  fun cuda_rope_apply(x : Float32*, cos_cache : Float32*, sin_cache : Float32*, seq_len : Int32, dim : Int32)
  fun cuda_rope_apply_inverse(x : Float32*, cos_cache : Float32*, sin_cache : Float32*, seq_len : Int32, dim : Int32)
  fun cuda_fused_attn_softmax(scores : Float32*, output : Float32*, scale : Float32, rows : Int32, cols : Int32)
  fun cuda_fused_attn_softmax_backward(s : Float32*, ds : Float32*, result : Float32*, scale : Float32, rows : Int32, cols : Int32)
  fun cuda_embedding_gather(token_emb : Float32*, ids : Int32*, output : Float32*, seq_len : Int32, d_model : Int32)
  fun cuda_embedding_scatter_add(grad : Float32*, ids : Int32*, d_token_emb : Float32*, seq_len : Int32, d_model : Int32)
  fun cuda_fused_bias_relu(input : Float32*, bias : Float32*, output : Float32*, mask : Float32*, rows : Int32, cols : Int32)
  fun cuda_fused_softmax_ce_grad(logits : Float32*, targets : Int32*, d_logits : Float32*, loss_out : Float32*, rows : Int32, cols : Int32)
  fun cuda_adam_bulk(params : Float32*, grads : Float32*, m : Float32*, v : Float32*,
                     lr : Float32, beta1 : Float32, beta2 : Float32, eps : Float32,
                     t : Int32, n : Int32)
  fun cuda_batched_varlen_attention(
    q_packed : Float32*, k_packed : Float32*, v_packed : Float32*,
    kv_offsets : Int32*, kv_lengths : Int32*,
    output : Float32*, weights_out : Float32*,
    n_nodes : Int32, n_heads : Int32, head_dim : Int32,
    max_len : Int32, scale : Float32)
  fun cuda_batched_varlen_attention_backward(
    q_packed : Float32*, k_packed : Float32*, v_packed : Float32*,
    attn_weights : Float32*, d_out : Float32*,
    kv_offsets : Int32*, kv_lengths : Int32*,
    dq : Float32*, dk_full : Float32*, dv_full : Float32*,
    n_nodes : Int32, n_heads : Int32, head_dim : Int32,
    max_len : Int32, scale : Float32)
  fun cuda_unpack_batched_attn_output(
    packed_output : Float32*, unpacked_output : Float32*,
    n_nodes : Int32, n_heads : Int32, head_dim : Int32)
  fun cuda_sync
end

class CuBLASBackend
  include Backend

  getter handle : LibCUDA::CublasHandle

  HostToDevice   = 1
  DeviceToHost   = 2
  DeviceToDevice = 3

  # Cached GPU buffer for integer arrays (input_ids, target_ids)
  @ids_buf : Pointer(Void) = Pointer(Void).null
  @ids_buf_size : UInt64 = 0_u64
  @loss_buf : Pointer(Void) = Pointer(Void).null  # persistent loss scalar

  private def ensure_ids_buf(n : Int32) : Pointer(Void)
    needed = (n * sizeof(Int32)).to_u64
    if needed > @ids_buf_size
      LibCUDA.cudaFree(@ids_buf) unless @ids_buf.null?
      @ids_buf = Pointer(Void).null
      LibCUDA.cudaMalloc(pointerof(@ids_buf), needed)
      @ids_buf_size = needed
    end
    @ids_buf
  end

  private def ensure_loss_buf : Pointer(Void)
    if @loss_buf.null?
      LibCUDA.cudaMalloc(pointerof(@loss_buf), sizeof(Float32).to_u64)
    end
    @loss_buf
  end

  def initialize
    @handle = Pointer(Void).null
    status = LibCUDA.cublasCreate_v2(pointerof(@handle))
    raise "cuBLAS init failed: #{status}" unless status == LibCUDA::CublasStatus::Success
  end

  def finalize
    LibCUDA.cublasDestroy_v2(@handle)
  end

  def release(a : Mat) end

  def gpu_alloc(size : UInt64) : Pointer(Void)
    ptr = Pointer(Void).null
    LibCUDA.cudaMalloc(pointerof(ptr), size)
    ptr
  end

  def gpu_free(ptr : Pointer(Void))
    LibCUDA.cudaFree(ptr) unless ptr.null?
  end

  def gpu_upload(dst : Pointer(Void), src : Pointer(Void), size : UInt64)
    LibCUDA.cudaMemcpy(dst, src, size, HostToDevice)
  end

  def gpu_download(dst : Pointer(Void), src : Pointer(Void), size : UInt64)
    LibCUDA.cudaMemcpy(dst, src, size, DeviceToHost)
  end

  def sync
    LibCUDAKernels.cuda_sync
  end

  # --- Core linear algebra (GPU-resident) ---

  def matmul(a : Mat, b : Mat) : Mat
    raise "dimension mismatch: #{a.cols} vs #{b.rows}" unless a.cols == b.rows
    m = a.rows
    n = b.cols
    k = a.cols

    d_a = a.gpu_ptr
    d_b = b.gpu_ptr

    c_size = (m * n).to_u64 * sizeof(Float32)
    d_c = Pointer(Void).null
    LibCUDA.cudaMalloc(pointerof(d_c), c_size)

    # cuBLAS is column-major; row-major C = A*B becomes C^T = B^T * A^T
    alpha = 1.0_f32
    beta = 0.0_f32
    # TODO(agpt-paper): cublasSgemm fails with InvalidValue ~step 500 when
    # bin/microgpt trains with --seq-len ≥ 2048 (--backend cublas). Smaller
    # seq_len and the AGPT trainer bin/agpt_train are unaffected. Symptom is
    # this exception firing mid-training, not at init. Suspected causes:
    #   - an intermediate tensor dimension (attention scores at L×L or a
    #     packed-batch matmul) crosses a cuBLAS parameter limit — possibly
    #     int32 byte-count overflow in one of the size calcs.
    #   - a stride/alignment condition only hit at larger N.
    # Workaround: cap window-training sweep at seq_len=1024. Full repro in
    # scripts/window_saturation_sweep.sh (currently stops at 1024).
    status = LibCUDA.cublasSgemm_v2(
      @handle,
      LibCUDA::CublasOperation::N,
      LibCUDA::CublasOperation::N,
      n, m, k,
      pointerof(alpha), d_b.as(Float32*), n,
      d_a.as(Float32*), k,
      pointerof(beta), d_c.as(Float32*), n
    )
    raise "cublasSgemm failed: #{status} (m=#{m} n=#{n} k=#{k})" unless status == LibCUDA::CublasStatus::Success

    Mat.new(m, n, d_c)
  end

  def matmul_into(a : Mat, b : Mat, dst : Mat)
    raise "dimension mismatch: #{a.cols} vs #{b.rows}" unless a.cols == b.rows
    m = a.rows
    n = b.cols
    k = a.cols

    alpha = 1.0_f32
    beta = 0.0_f32
    status = LibCUDA.cublasSgemm_v2(
      @handle,
      LibCUDA::CublasOperation::N,
      LibCUDA::CublasOperation::N,
      n, m, k,
      pointerof(alpha), b.gpu_ptr.as(Float32*), n,
      a.gpu_ptr.as(Float32*), k,
      pointerof(beta), dst.gpu_ptr.as(Float32*), n
    )
    raise "cublasSgemm failed: #{status}" unless status == LibCUDA::CublasStatus::Success
    dst.mark_gpu_only
  end

  def transpose(a : Mat) : Mat
    # Row-major (a.rows × a.cols) is column-major (a.cols × a.rows).
    # We want output row-major (a.cols × a.rows) = column-major (a.rows × a.cols).
    # So C is m×n column-major where m=a.rows, n=a.cols.
    # op(A) = A^T: A is n×m = (a.cols × a.rows) column-major, lda = a.cols.
    m = a.rows
    n = a.cols
    d_a = a.gpu_ptr

    c_size = (a.rows * a.cols).to_u64 * sizeof(Float32)
    d_c = Pointer(Void).null
    LibCUDA.cudaMalloc(pointerof(d_c), c_size)

    alpha = 1.0_f32
    beta = 0.0_f32
    LibCUDA.cublasSgeam(
      @handle,
      LibCUDA::CublasOperation::T, LibCUDA::CublasOperation::N,
      m, n,
      pointerof(alpha), d_a.as(Float32*), n,
      pointerof(beta), d_a.as(Float32*), m,
      d_c.as(Float32*), m
    )

    Mat.new(a.cols, a.rows, d_c)
  end

  def add(a : Mat, b : Mat) : Mat
    raise "dimension mismatch" unless a.rows == b.rows && a.cols == b.cols
    n = a.rows * a.cols

    d_a = a.gpu_ptr
    d_b = b.gpu_ptr

    c_size = n.to_u64 * sizeof(Float32)
    d_c = Pointer(Void).null
    LibCUDA.cudaMalloc(pointerof(d_c), c_size)
    LibCUDA.cudaMemcpy(d_c, d_a, c_size, DeviceToDevice)

    alpha = 1.0_f32
    LibCUDA.cublasSaxpy_v2(@handle, n, pointerof(alpha),
      d_b.as(Float32*), 1, d_c.as(Float32*), 1)

    Mat.new(a.rows, a.cols, d_c)
  end

  def add!(a : Mat, b : Mat)
    n = a.rows * a.cols
    alpha = 1.0_f32
    LibCUDA.cublasSaxpy_v2(@handle, n, pointerof(alpha),
      b.gpu_ptr.as(Float32*), 1, a.gpu_ptr.as(Float32*), 1)
    a.mark_gpu_only
  end

  def scale(a : Mat, scalar : Float32) : Mat
    n = a.rows * a.cols
    d_a = a.gpu_ptr

    c_size = n.to_u64 * sizeof(Float32)
    d_c = Pointer(Void).null
    LibCUDA.cudaMalloc(pointerof(d_c), c_size)
    LibCUDA.cudaMemcpy(d_c, d_a, c_size, DeviceToDevice)

    s = scalar
    LibCUDA.cublasSscal_v2(@handle, n, pointerof(s), d_c.as(Float32*), 1)

    Mat.new(a.rows, a.cols, d_c)
  end

  def scale!(a : Mat, scalar : Float32)
    n = a.rows * a.cols
    s = scalar
    LibCUDA.cublasSscal_v2(@handle, n, pointerof(s), a.gpu_ptr.as(Float32*), 1)
    a.mark_gpu_only
  end

  def batched_matmul(a_list : Array(Mat), b_list : Array(Mat),
                     transpose_a : Bool, transpose_b : Bool) : Array(Mat)
    batch_count = a_list.size
    return [] of Mat if batch_count == 0

    uniform = a_list.all? { |a| a.rows == a_list[0].rows && a.cols == a_list[0].cols } &&
              b_list.all? { |b| b.rows == b_list[0].rows && b.cols == b_list[0].cols }

    unless uniform
      return a_list.zip(b_list).map do |a, b|
        aa = transpose_a ? transpose(a) : a
        bb = transpose_b ? transpose(b) : b
        matmul(aa, bb)
      end
    end

    cublas_op_b = transpose_a ? LibCUDA::CublasOperation::T : LibCUDA::CublasOperation::N
    cublas_op_a = transpose_b ? LibCUDA::CublasOperation::T : LibCUDA::CublasOperation::N

    m_out = transpose_a ? a_list[0].cols : a_list[0].rows
    n_out = transpose_b ? b_list[0].rows : b_list[0].cols
    k_inner = transpose_a ? a_list[0].rows : a_list[0].cols

    cb_m = n_out
    cb_n = m_out
    cb_k = k_inner

    d_a_ptrs = Array(Pointer(Void)).new(batch_count)
    d_b_ptrs = Array(Pointer(Void)).new(batch_count)
    d_c_ptrs = Array(Pointer(Void)).new(batch_count)

    c_size = (m_out * n_out).to_u64 * sizeof(Float32)

    batch_count.times do |i|
      d_a_ptrs << a_list[i].gpu_ptr
      d_b_ptrs << b_list[i].gpu_ptr

      d_c = Pointer(Void).null
      LibCUDA.cudaMalloc(pointerof(d_c), c_size)
      d_c_ptrs << d_c
    end

    h_a_dev = d_b_ptrs.map(&.as(Float32*))
    h_b_dev = d_a_ptrs.map(&.as(Float32*))
    h_c_dev = d_c_ptrs.map(&.as(Float32*))

    ptr_size = batch_count.to_u64 * sizeof(Pointer(Float32))
    d_a_arr = Pointer(Void).null
    d_b_arr = Pointer(Void).null
    d_c_arr = Pointer(Void).null
    LibCUDA.cudaMalloc(pointerof(d_a_arr), ptr_size)
    LibCUDA.cudaMalloc(pointerof(d_b_arr), ptr_size)
    LibCUDA.cudaMalloc(pointerof(d_c_arr), ptr_size)

    LibCUDA.cudaMemcpy(d_a_arr, h_a_dev.to_unsafe.as(Void*), ptr_size, HostToDevice)
    LibCUDA.cudaMemcpy(d_b_arr, h_b_dev.to_unsafe.as(Void*), ptr_size, HostToDevice)
    LibCUDA.cudaMemcpy(d_c_arr, h_c_dev.to_unsafe.as(Void*), ptr_size, HostToDevice)

    alpha = 1.0_f32
    beta = 0.0_f32

    lda_val = b_list[0].cols
    ldb_val = a_list[0].cols
    ldc_val = n_out

    status = LibCUDA.cublasSgemmBatched(
      @handle,
      cublas_op_a, cublas_op_b,
      cb_m, cb_n, cb_k,
      pointerof(alpha),
      d_a_arr.as(Float32**), lda_val,
      d_b_arr.as(Float32**), ldb_val,
      pointerof(beta),
      d_c_arr.as(Float32**), ldc_val,
      batch_count
    )
    raise "cublasSgemmBatched failed: #{status}" unless status == LibCUDA::CublasStatus::Success

    results = Array(Mat).new(batch_count)
    batch_count.times do |i|
      results << Mat.new(m_out, n_out, d_c_ptrs[i])
    end

    LibCUDA.cudaFree(d_a_arr)
    LibCUDA.cudaFree(d_b_arr)
    LibCUDA.cudaFree(d_c_arr)

    results
  end

  # --- Element-wise operations (CUDA kernels, GPU-resident) ---

  def softmax_rows(x : Mat) : Mat
    d_in = x.gpu_ptr.as(Float32*)
    d_out = Pointer(Void).null
    LibCUDA.cudaMalloc(pointerof(d_out), x.byte_size)
    LibCUDAKernels.cuda_softmax_rows(d_in, d_out.as(Float32*), x.rows, x.cols)
    Mat.new(x.rows, x.cols, d_out)
  end

  def softmax_backward(s : Mat, ds : Mat) : Mat
    d_s = s.gpu_ptr.as(Float32*)
    d_ds = ds.gpu_ptr.as(Float32*)
    d_out = Pointer(Void).null
    LibCUDA.cudaMalloc(pointerof(d_out), s.byte_size)
    LibCUDAKernels.cuda_softmax_backward(d_s, d_ds, d_out.as(Float32*), s.rows, s.cols)
    Mat.new(s.rows, s.cols, d_out)
  end

  def rope_apply(x : Mat, cos_cache : Mat, sin_cache : Mat)
    d_x = x.gpu_ptr.as(Float32*)
    d_cos = cos_cache.gpu_ptr.as(Float32*)
    d_sin = sin_cache.gpu_ptr.as(Float32*)
    LibCUDAKernels.cuda_rope_apply(d_x, d_cos, d_sin, x.rows, x.cols)
    x.mark_gpu_only
  end

  def rope_apply_inverse(x : Mat, cos_cache : Mat, sin_cache : Mat)
    d_x = x.gpu_ptr.as(Float32*)
    d_cos = cos_cache.gpu_ptr.as(Float32*)
    d_sin = sin_cache.gpu_ptr.as(Float32*)
    LibCUDAKernels.cuda_rope_apply_inverse(d_x, d_cos, d_sin, x.rows, x.cols)
    x.mark_gpu_only
  end

  def fused_attn_softmax(scores : Mat, scale : Float64) : Mat
    d_in = scores.gpu_ptr.as(Float32*)
    d_out = Pointer(Void).null
    LibCUDA.cudaMalloc(pointerof(d_out), scores.byte_size)
    LibCUDAKernels.cuda_fused_attn_softmax(d_in, d_out.as(Float32*), scale.to_f32, scores.rows, scores.cols)
    Mat.new(scores.rows, scores.cols, d_out)
  end

  def fused_attn_softmax_backward(s : Mat, ds : Mat, scale : Float64) : Mat
    d_s = s.gpu_ptr.as(Float32*)
    d_ds = ds.gpu_ptr.as(Float32*)
    d_out = Pointer(Void).null
    LibCUDA.cudaMalloc(pointerof(d_out), s.byte_size)
    LibCUDAKernels.cuda_fused_attn_softmax_backward(d_s, d_ds, d_out.as(Float32*), scale.to_f32, s.rows, s.cols)
    Mat.new(s.rows, s.cols, d_out)
  end

  def causal_mask(scores : Mat)
    d_data = scores.gpu_ptr.as(Float32*)
    LibCUDAKernels.cuda_causal_mask(d_data, scores.rows)
    scores.mark_gpu_only
  end

  def causal_mask_batched(scores : Mat, seq_len : Int32)
    d_data = scores.gpu_ptr.as(Float32*)
    LibCUDAKernels.cuda_causal_mask_batched(d_data, scores.rows, scores.cols, seq_len)
    scores.mark_gpu_only
  end

  def bias_add(data : Mat, bias : Mat)
    d_data = data.gpu_ptr.as(Float32*)
    d_bias = bias.gpu_ptr.as(Float32*)
    LibCUDAKernels.cuda_bias_add(d_data, d_bias, data.rows, data.cols)
    data.mark_gpu_only
  end

  def relu_forward(x : Mat) : {Mat, Mat}
    n = x.rows * x.cols
    d_in = x.gpu_ptr.as(Float32*)
    d_out = Pointer(Void).null
    d_mask = Pointer(Void).null
    LibCUDA.cudaMalloc(pointerof(d_out), x.byte_size)
    LibCUDA.cudaMalloc(pointerof(d_mask), x.byte_size)
    LibCUDAKernels.cuda_relu_forward(d_in, d_out.as(Float32*), d_mask.as(Float32*), n)
    {Mat.new(x.rows, x.cols, d_out), Mat.new(x.rows, x.cols, d_mask)}
  end

  def fused_bias_relu(x : Mat, bias : Mat) : {Mat, Mat}
    d_in = x.gpu_ptr.as(Float32*)
    d_bias = bias.gpu_ptr.as(Float32*)
    d_out = Pointer(Void).null
    d_mask = Pointer(Void).null
    LibCUDA.cudaMalloc(pointerof(d_out), x.byte_size)
    LibCUDA.cudaMalloc(pointerof(d_mask), x.byte_size)
    LibCUDAKernels.cuda_fused_bias_relu(d_in, d_bias, d_out.as(Float32*), d_mask.as(Float32*), x.rows, x.cols)
    {Mat.new(x.rows, x.cols, d_out), Mat.new(x.rows, x.cols, d_mask)}
  end

  def relu_backward(grad : Mat, mask : Mat) : Mat
    n = grad.rows * grad.cols
    d_grad = grad.gpu_ptr.as(Float32*)
    d_mask = mask.gpu_ptr.as(Float32*)
    d_out = Pointer(Void).null
    LibCUDA.cudaMalloc(pointerof(d_out), grad.byte_size)
    LibCUDAKernels.cuda_relu_backward(d_grad, d_mask, d_out.as(Float32*), n)
    Mat.new(grad.rows, grad.cols, d_out)
  end

  def layer_norm_forward(x : Mat, gamma : Mat, beta : Mat) : {Mat, Mat, Mat}
    rows = x.rows
    cols = x.cols
    d_in = x.gpu_ptr.as(Float32*)
    d_gamma = gamma.gpu_ptr.as(Float32*)
    d_beta = beta.gpu_ptr.as(Float32*)

    sinv_sz = rows.to_u64 * sizeof(Float32)
    d_out = Pointer(Void).null
    d_norm = Pointer(Void).null
    d_sinv = Pointer(Void).null
    LibCUDA.cudaMalloc(pointerof(d_out), x.byte_size)
    LibCUDA.cudaMalloc(pointerof(d_norm), x.byte_size)
    LibCUDA.cudaMalloc(pointerof(d_sinv), sinv_sz)

    LibCUDAKernels.cuda_layer_norm_forward(
      d_in, d_out.as(Float32*), d_norm.as(Float32*), d_sinv.as(Float32*),
      d_gamma, d_beta, rows, cols)

    {Mat.new(rows, cols, d_out), Mat.new(rows, cols, d_norm), Mat.new(rows, 1, d_sinv)}
  end

  def layer_norm_backward(grad : Mat, norm : Mat, std_inv : Mat, gamma : Mat) : {Mat, Mat, Mat}
    rows = grad.rows
    cols = grad.cols
    d_grad = grad.gpu_ptr.as(Float32*)
    d_norm = norm.gpu_ptr.as(Float32*)
    d_sinv = std_inv.gpu_ptr.as(Float32*)
    d_gamma = gamma.gpu_ptr.as(Float32*)

    param_sz = cols.to_u64 * sizeof(Float32)
    d_dx = Pointer(Void).null
    d_dgamma = Pointer(Void).null
    d_dbeta = Pointer(Void).null
    LibCUDA.cudaMalloc(pointerof(d_dx), grad.byte_size)
    LibCUDA.cudaMalloc(pointerof(d_dgamma), param_sz)
    LibCUDA.cudaMalloc(pointerof(d_dbeta), param_sz)

    # Zero dgamma/dbeta for atomicAdd accumulation
    LibCUDA.cudaMemset(d_dgamma, 0, param_sz)
    LibCUDA.cudaMemset(d_dbeta, 0, param_sz)

    LibCUDAKernels.cuda_layer_norm_backward(
      d_grad, d_norm, d_sinv, d_gamma,
      d_dx.as(Float32*), d_dgamma.as(Float32*), d_dbeta.as(Float32*),
      rows, cols)

    {Mat.new(rows, cols, d_dx), Mat.new(1, cols, d_dgamma), Mat.new(1, cols, d_dbeta)}
  end

  def adam_step(param : Mat, grad : Mat, m : Mat, v : Mat, lr : Float64, t : Int32)
    n = param.rows * param.cols
    LibCUDAKernels.cuda_adam_step(
      param.gpu_ptr.as(Float32*), grad.gpu_ptr.as(Float32*),
      m.gpu_ptr.as(Float32*), v.gpu_ptr.as(Float32*),
      lr.to_f32, 0.9_f32, 0.999_f32, 1e-8_f32, t, n)
    param.mark_gpu_only
    m.mark_gpu_only
    v.mark_gpu_only
  end

  # GPU-side embedding gather: output[pos, j] = token_emb[ids[pos], j]
  def embedding_gather(token_emb : Mat, ids : Array(Int32), seq_len : Int32, d_model : Int32) : Mat
    d_ids = ensure_ids_buf(seq_len)
    ids_bytes = (seq_len * sizeof(Int32)).to_u64
    LibCUDA.cudaMemcpy(d_ids, ids.to_unsafe.as(Void*), ids_bytes, HostToDevice)

    d_out = Pointer(Void).null
    LibCUDA.cudaMalloc(pointerof(d_out), (seq_len * d_model).to_u64 * sizeof(Float32))

    LibCUDAKernels.cuda_embedding_gather(
      token_emb.gpu_ptr.as(Float32*), d_ids.as(Int32*),
      d_out.as(Float32*), seq_len, d_model)

    Mat.new(seq_len, d_model, d_out)
  end

  # GPU-side embedding backward: scatter-add gradients back to d_token_emb
  def embedding_scatter_add(grad : Mat, ids : Array(Int32), d_token_emb : Mat, seq_len : Int32, d_model : Int32)
    d_ids = ensure_ids_buf(seq_len)
    ids_bytes = (seq_len * sizeof(Int32)).to_u64
    LibCUDA.cudaMemcpy(d_ids, ids.to_unsafe.as(Void*), ids_bytes, HostToDevice)

    LibCUDA.cudaMemset(d_token_emb.gpu_ptr, 0, d_token_emb.byte_size)

    LibCUDAKernels.cuda_embedding_scatter_add(
      grad.gpu_ptr.as(Float32*), d_ids.as(Int32*),
      d_token_emb.gpu_ptr.as(Float32*), seq_len, d_model)

    d_token_emb.mark_gpu_only
  end

  # Fused softmax + CE loss + gradient in one kernel launch
  def fused_softmax_ce_grad(logits : Mat, targets : Array(Int32)) : {Float64, Mat}
    seq_len = logits.rows
    vocab_size = logits.cols

    d_ids = ensure_ids_buf(seq_len)
    tgt_bytes = (seq_len * sizeof(Int32)).to_u64
    LibCUDA.cudaMemcpy(d_ids, targets.to_unsafe.as(Void*), tgt_bytes, HostToDevice)

    d_out = Pointer(Void).null
    LibCUDA.cudaMalloc(pointerof(d_out), logits.byte_size)

    d_loss = ensure_loss_buf

    LibCUDAKernels.cuda_fused_softmax_ce_grad(
      logits.gpu_ptr.as(Float32*), d_ids.as(Int32*),
      d_out.as(Float32*), d_loss.as(Float32*),
      seq_len, vocab_size)

    # Download loss scalar (4 bytes)
    loss_val = 0.0_f32
    LibCUDA.cudaMemcpy(pointerof(loss_val).as(Void*), d_loss, sizeof(Float32).to_u64, DeviceToHost)

    d_logits = Mat.new(seq_len, vocab_size, d_out)
    {(loss_val / seq_len).to_f64, d_logits}
  end
end

# =============================================================================
# GPU Weight Store — contiguous GPU buffer for all model parameters
# =============================================================================
# One cudaMalloc holds all weights + Adam m/v state.
# Each Mat gets a pointer into the store (store_backed = true).
# Upload/download the entire store in one call.

class WeightStore
  getter store_ptr : Pointer(Void)    # single GPU allocation
  getter total_bytes : UInt64
  getter total_floats : Int64
  getter mat_offsets : Array(UInt64)  # byte offset of each mat in store
  getter mats : Array(Mat)           # all weight mats (references, not copies)
  @grad_buf : Pointer(Void) = Pointer(Void).null  # reusable gradient gather buffer

  # Adam state lives at known offsets in the same buffer
  getter adam_m_offset : UInt64       # byte offset where Adam M starts
  getter adam_v_offset : UInt64       # byte offset where Adam V starts
  getter weights_bytes : UInt64

  # weight_mats: all trainable parameters
  # adam_mats: corresponding Adam m/v Mats (2 per weight mat: m, v)
  #   Order: [w0_m, w0_v, w1_m, w1_v, ...]
  def initialize(@mats : Array(Mat), adam_mats : Array(Mat)? = nil)
    # Compute total size and per-mat offsets
    @mat_offsets = Array(UInt64).new(@mats.size, 0_u64)
    offset = 0_u64
    @mats.each_with_index do |m, i|
      @mat_offsets[i] = offset
      offset += m.byte_size
    end
    @weights_bytes = offset

    # Adam M and V mirror the weight layout
    @adam_m_offset = @weights_bytes
    @adam_v_offset = @weights_bytes * 2
    @total_bytes = @weights_bytes * 3  # weights + M + V
    @total_floats = (@total_bytes / sizeof(Float32)).to_i64

    # Allocate one contiguous GPU buffer
    @store_ptr = Pointer(Void).null
    LibCUDA.cudaMalloc(pointerof(@store_ptr), @total_bytes)

    # Zero the entire buffer (Adam M and V start at zero)
    LibCUDA.cudaMemset(@store_ptr, 0, @total_bytes)

    # Upload weights and point each Mat into the store
    @mats.each_with_index do |m, i|
      dest = (@store_ptr + @mat_offsets[i]).as(Float32*)
      LibCUDA.cudaMemcpy(dest.as(Void*), m.raw_data.to_unsafe.as(Void*),
                          m.byte_size, 1)  # cudaMemcpyHostToDevice = 1
      m.set_store_ptr(dest.as(Void*))
    end

    # Point Adam m/v Mats into the store's Adam sections
    if am = adam_mats
      @mats.each_with_index do |_m, i|
        m_idx = i * 2
        v_idx = i * 2 + 1
        if m_idx < am.size && v_idx < am.size
          am[m_idx].set_store_ptr(adam_m_ptr(i).as(Void*))
          am[v_idx].set_store_ptr(adam_v_ptr(i).as(Void*))
        end
      end
    end

    # Pre-allocate gradient gather buffer
    LibCUDA.cudaMalloc(pointerof(@grad_buf), @weights_bytes)

    STDERR.puts "WeightStore: #{@mats.size} tensors, #{"%.2f" % (@total_bytes / 1024.0 / 1024.0)} MB GPU"
  end

  # Download all weights back to CPU
  def download_all
    @mats.each_with_index do |m, i|
      src = (@store_ptr + @mat_offsets[i]).as(Float32*)
      LibCUDA.cudaMemcpy(m.raw_data.to_unsafe.as(Void*), src.as(Void*),
                          m.byte_size, 2)  # cudaMemcpyDeviceToHost = 2
    end
  end

  # Get Adam M pointer for a specific mat index
  def adam_m_ptr(mat_idx : Int32) : Pointer(Float32)
    (@store_ptr + @adam_m_offset + @mat_offsets[mat_idx]).as(Float32*)
  end

  # Get Adam V pointer for a specific mat index
  def adam_v_ptr(mat_idx : Int32) : Pointer(Float32)
    (@store_ptr + @adam_v_offset + @mat_offsets[mat_idx]).as(Float32*)
  end

  # Bulk Adam: gather gradients, then one kernel launch for all parameters
  def bulk_adam_step(grad_mats : Array(Mat), lr : Float64, t : Int32)
    # Gather: copy each gradient Mat's GPU data into the contiguous buffer
    @mats.each_with_index do |_m, i|
      g = grad_mats[i]
      src = g.gpu_ptr  # ensures it's on GPU
      dst = (@grad_buf + @mat_offsets[i])
      LibCUDA.cudaMemcpy(dst, src, g.byte_size, 3)  # DeviceToDevice = 3
    end

    # One kernel launch for all parameters
    n = (@weights_bytes / sizeof(Float32)).to_i32
    LibCUDAKernels.cuda_adam_bulk(
      @store_ptr.as(Float32*),
      @grad_buf.as(Float32*),
      (@store_ptr + @adam_m_offset).as(Float32*),
      (@store_ptr + @adam_v_offset).as(Float32*),
      lr.to_f32, 0.9_f32, 0.999_f32, 1e-8_f32, t, n)

    # Mark all weight mats as GPU-only (they were just updated on GPU)
    @mats.each &.mark_gpu_only
  end

  def finalize
    unless @store_ptr.null?
      LibCUDA.cudaFree(@store_ptr)
    end
    unless @grad_buf.null?
      LibCUDA.cudaFree(@grad_buf)
    end
  end
end

# =============================================================================
# Global Backend Selection
# =============================================================================

class_property backend : Backend = CrystalBackend.new

def self.use_crystal!
  @@backend = CrystalBackend.new
end

def self.use_openblas!
  @@backend = OpenBLASBackend.new
end

def self.use_cublas!
  @@backend = CuBLASBackend.new
end

end
