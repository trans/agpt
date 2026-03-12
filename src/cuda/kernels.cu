#include <math.h>
#include <float.h>

// =============================================================================
// Softmax (row-wise)
// =============================================================================
// One block per row, threads cooperate within a row

__global__ void softmax_rows_kernel(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    if (row >= rows) return;

    const float* in_row = input + row * cols;
    float* out_row = output + row * cols;

    extern __shared__ float sdata[];

    // Find local max
    float local_max = -FLT_MAX;
    for (int j = tid; j < cols; j += nthreads) {
        if (in_row[j] > local_max) local_max = in_row[j];
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s && sdata[tid + s] > sdata[tid]) sdata[tid] = sdata[tid + s];
        __syncthreads();
    }
    float max_val = sdata[0];

    // Exp and local sum
    float local_sum = 0.0f;
    for (int j = tid; j < cols; j += nthreads) {
        float e = expf(in_row[j] - max_val);
        out_row[j] = e;
        local_sum += e;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / sdata[0];

    // Normalize
    for (int j = tid; j < cols; j += nthreads) {
        out_row[j] *= inv_sum;
    }
}

extern "C" void cuda_softmax_rows(const float* input, float* output, int rows, int cols) {
    int threads = (cols < 256) ? cols : 256;
    int t = 1;
    while (t < threads) t <<= 1;
    threads = (t < 32) ? 32 : t;  // minimum warp size
    int smem = threads * sizeof(float);
    softmax_rows_kernel<<<rows, threads, smem>>>(input, output, rows, cols);
}

// =============================================================================
// Softmax backward
// =============================================================================

__global__ void softmax_backward_kernel(const float* s, const float* ds,
                                         float* result, int rows, int cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    if (row >= rows) return;

    const float* s_row = s + row * cols;
    const float* ds_row = ds + row * cols;
    float* out_row = result + row * cols;

    extern __shared__ float sdata[];

    float local_dot = 0.0f;
    for (int j = tid; j < cols; j += nthreads) {
        local_dot += ds_row[j] * s_row[j];
    }
    sdata[tid] = local_dot;
    __syncthreads();
    for (int s2 = nthreads / 2; s2 > 0; s2 >>= 1) {
        if (tid < s2) sdata[tid] += sdata[tid + s2];
        __syncthreads();
    }
    float dot = sdata[0];

    for (int j = tid; j < cols; j += nthreads) {
        out_row[j] = s_row[j] * (ds_row[j] - dot);
    }
}

extern "C" void cuda_softmax_backward(const float* s, const float* ds,
                                       float* result, int rows, int cols) {
    int threads = (cols < 256) ? cols : 256;
    int t = 1;
    while (t < threads) t <<= 1;
    threads = (t < 32) ? 32 : t;
    int smem = threads * sizeof(float);
    softmax_backward_kernel<<<rows, threads, smem>>>(s, ds, result, rows, cols);
}

// =============================================================================
// Causal mask: set upper triangle to -1e9
// =============================================================================

__global__ void causal_mask_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * n;
    if (idx >= total) return;

    int row = idx / n;
    int col = idx % n;
    if (col > row) {
        data[idx] = -1e9f;
    }
}

extern "C" void cuda_causal_mask(float* data, int n) {
    int total = n * n;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    causal_mask_kernel<<<blocks, threads>>>(data, n);
}

// =============================================================================
// Causal mask (batched): repeating mask for stacked score matrices
// =============================================================================

__global__ void causal_mask_batched_kernel(float* data, int total_rows, int cols, int seq_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = total_rows * cols;
    if (idx >= total) return;

    int row = idx / cols;
    int col = idx % cols;
    int pos = row % seq_len;
    if (col > pos) {
        data[idx] = -1e9f;
    }
}

extern "C" void cuda_causal_mask_batched(float* data, int total_rows, int cols, int seq_len) {
    int total = total_rows * cols;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    causal_mask_batched_kernel<<<blocks, threads>>>(data, total_rows, cols, seq_len);
}

// =============================================================================
// Bias add: result[i,j] += bias[j]  (bias is 1×cols, broadcast over rows)
// =============================================================================

__global__ void bias_add_kernel(float* data, const float* bias, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx >= total) return;

    int col = idx % cols;
    data[idx] += bias[col];
}

extern "C" void cuda_bias_add(float* data, const float* bias, int rows, int cols) {
    int total = rows * cols;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    bias_add_kernel<<<blocks, threads>>>(data, bias, rows, cols);
}

// =============================================================================
// ReLU forward: output = max(0, input), mask = (input > 0) ? 1 : 0
// =============================================================================

__global__ void relu_forward_kernel(const float* input, float* output,
                                     float* mask, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (input[idx] > 0.0f) {
        output[idx] = input[idx];
        mask[idx] = 1.0f;
    } else {
        output[idx] = 0.0f;
        mask[idx] = 0.0f;
    }
}

extern "C" void cuda_relu_forward(const float* input, float* output,
                                   float* mask, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    relu_forward_kernel<<<blocks, threads>>>(input, output, mask, n);
}

// =============================================================================
// ReLU backward: output = grad * mask
// =============================================================================

__global__ void relu_backward_kernel(const float* grad, const float* mask,
                                      float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    output[idx] = grad[idx] * mask[idx];
}

extern "C" void cuda_relu_backward(const float* grad, const float* mask,
                                    float* output, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    relu_backward_kernel<<<blocks, threads>>>(grad, mask, output, n);
}

// =============================================================================
// Layer norm forward
// =============================================================================

__global__ void layer_norm_forward_kernel(const float* input, float* output,
                                           float* norm_out, float* std_inv_out,
                                           const float* gamma, const float* beta,
                                           int rows, int cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    if (row >= rows) return;

    const float* in_row = input + row * cols;
    float* out_row = output + row * cols;
    float* norm_row = norm_out + row * cols;

    extern __shared__ float sdata[];

    // Compute mean
    float local_sum = 0.0f;
    for (int j = tid; j < cols; j += nthreads) local_sum += in_row[j];
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float mean = sdata[0] / cols;

    // Compute variance
    float local_var = 0.0f;
    for (int j = tid; j < cols; j += nthreads) {
        float d = in_row[j] - mean;
        local_var += d * d;
    }
    sdata[tid] = local_var;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float var = sdata[0] / cols;

    float inv = 1.0f / sqrtf(var + 1e-5f);
    if (tid == 0) std_inv_out[row] = inv;

    // Normalize and apply affine
    for (int j = tid; j < cols; j += nthreads) {
        norm_row[j] = (in_row[j] - mean) * inv;
        out_row[j] = norm_row[j] * gamma[j] + beta[j];
    }
}

extern "C" void cuda_layer_norm_forward(const float* input, float* output,
                                         float* norm_out, float* std_inv_out,
                                         const float* gamma, const float* beta,
                                         int rows, int cols) {
    int threads = (cols < 256) ? cols : 256;
    int t = 1;
    while (t < threads) t <<= 1;
    threads = (t < 32) ? 32 : t;
    int smem = threads * sizeof(float);
    layer_norm_forward_kernel<<<rows, threads, smem>>>(input, output, norm_out, std_inv_out,
                                                        gamma, beta, rows, cols);
}

// =============================================================================
// Layer norm backward
// =============================================================================

__global__ void layer_norm_backward_kernel(const float* grad, const float* norm,
                                            const float* std_inv, const float* gamma,
                                            float* dx, float* dgamma, float* dbeta,
                                            int rows, int cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    if (row >= rows) return;

    const float* grad_row = grad + row * cols;
    const float* norm_row = norm + row * cols;
    float* dx_row = dx + row * cols;
    float sinv = std_inv[row];
    float n = (float)cols;

    extern __shared__ float sdata[];
    // Use two shared arrays: sdata[0..nthreads-1] and sdata[nthreads..2*nthreads-1]
    float* sdata2 = sdata + nthreads;

    // Accumulate dgamma, dbeta (still atomicAdd across rows)
    for (int j = tid; j < cols; j += nthreads) {
        atomicAdd(&dgamma[j], grad_row[j] * norm_row[j]);
        atomicAdd(&dbeta[j], grad_row[j]);
    }

    // Compute dx: need dot and sum reductions
    float local_dot = 0.0f;
    float local_sum = 0.0f;
    for (int j = tid; j < cols; j += nthreads) {
        float gn = grad_row[j] * gamma[j];
        local_dot += gn * norm_row[j];
        local_sum += gn;
    }
    sdata[tid] = local_dot;
    sdata2[tid] = local_sum;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
            sdata2[tid] += sdata2[tid + s];
        }
        __syncthreads();
    }
    float dot = sdata[0];
    float sum_val = sdata2[0];

    for (int j = tid; j < cols; j += nthreads) {
        float gn = grad_row[j] * gamma[j];
        dx_row[j] = sinv * (gn - (norm_row[j] * dot + sum_val) / n);
    }
}

extern "C" void cuda_layer_norm_backward(const float* grad, const float* norm,
                                          const float* std_inv, const float* gamma,
                                          float* dx, float* dgamma, float* dbeta,
                                          int rows, int cols) {
    int threads = (cols < 256) ? cols : 256;
    int t = 1;
    while (t < threads) t <<= 1;
    threads = (t < 32) ? 32 : t;
    int smem = 2 * threads * sizeof(float);  // two arrays for dot and sum
    layer_norm_backward_kernel<<<rows, threads, smem>>>(grad, norm, std_inv, gamma,
                                                         dx, dgamma, dbeta, rows, cols);
}

// =============================================================================
// Adam optimizer step
// =============================================================================

__global__ void adam_step_kernel(float* param, const float* grad,
                                 float* m, float* v,
                                 float lr, float beta1, float beta2, float eps,
                                 float bc1, float bc2, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    m[idx] = beta1 * m[idx] + (1.0f - beta1) * grad[idx];
    v[idx] = beta2 * v[idx] + (1.0f - beta2) * grad[idx] * grad[idx];
    float m_hat = m[idx] / bc1;
    float v_hat = v[idx] / bc2;
    param[idx] -= lr * m_hat / (sqrtf(v_hat) + eps);
}

extern "C" void cuda_adam_step(float* param, const float* grad,
                                float* m, float* v,
                                float lr, float beta1, float beta2, float eps,
                                int t, int n) {
    float bc1 = 1.0f - powf(beta1, t);
    float bc2 = 1.0f - powf(beta2, t);
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    adam_step_kernel<<<blocks, threads>>>(param, grad, m, v, lr, beta1, beta2, eps, bc1, bc2, n);
}

// =============================================================================
// RoPE: apply rotary position embedding in-place
// =============================================================================
// For each position pos and dimension pair (2i, 2i+1):
//   x'[2i]   = x[2i]*cos(θ) - x[2i+1]*sin(θ)
//   x'[2i+1] = x[2i]*sin(θ) + x[2i+1]*cos(θ)
// where θ = pos / base^(2i/dim)

__global__ void rope_apply_kernel(float* x, const float* cos_cache, const float* sin_cache,
                                   int seq_len, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half = dim / 2;
    int total = seq_len * half;
    if (idx >= total) return;

    int pos = idx / half;
    int i = idx % half;

    int idx0 = pos * dim + 2 * i;
    int idx1 = idx0 + 1;
    int cache_idx = pos * dim + 2 * i;

    float c = cos_cache[cache_idx];
    float s = sin_cache[cache_idx];
    float x0 = x[idx0];
    float x1 = x[idx1];
    x[idx0] = x0 * c - x1 * s;
    x[idx1] = x0 * s + x1 * c;
}

extern "C" void cuda_rope_apply(float* x, const float* cos_cache, const float* sin_cache,
                                  int seq_len, int dim) {
    int total = seq_len * (dim / 2);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    rope_apply_kernel<<<blocks, threads>>>(x, cos_cache, sin_cache, seq_len, dim);
}

// Inverse RoPE: negate sin
__global__ void rope_apply_inverse_kernel(float* x, const float* cos_cache, const float* sin_cache,
                                           int seq_len, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half = dim / 2;
    int total = seq_len * half;
    if (idx >= total) return;

    int pos = idx / half;
    int i = idx % half;

    int idx0 = pos * dim + 2 * i;
    int idx1 = idx0 + 1;
    int cache_idx = pos * dim + 2 * i;

    float c = cos_cache[cache_idx];
    float s = sin_cache[cache_idx];
    float x0 = x[idx0];
    float x1 = x[idx1];
    x[idx0] = x0 * c + x1 * s;
    x[idx1] = -x0 * s + x1 * c;
}

extern "C" void cuda_rope_apply_inverse(float* x, const float* cos_cache, const float* sin_cache,
                                          int seq_len, int dim) {
    int total = seq_len * (dim / 2);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    rope_apply_inverse_kernel<<<blocks, threads>>>(x, cos_cache, sin_cache, seq_len, dim);
}

// =============================================================================
// Fused attention: scale + causal mask + softmax (forward)
// =============================================================================
// One block per row. Reads raw Q·K^T scores, applies:
//   1. Scale by 1/sqrt(d)
//   2. Causal mask (upper triangle → -inf)
//   3. Row-wise softmax
// Single kernel, single memory read/write.

__global__ void fused_attn_softmax_kernel(const float* scores, float* output,
                                           float scale, int rows, int cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    if (row >= rows) return;

    const float* in_row = scores + row * cols;
    float* out_row = output + row * cols;

    extern __shared__ float sdata[];

    // Phase 1: Scale + mask + store, find local max
    float local_max = -FLT_MAX;
    for (int j = tid; j < cols; j += nthreads) {
        float val = in_row[j] * scale;
        if (j > row) val = -1e9f;  // causal mask
        out_row[j] = val;
        if (val > local_max) local_max = val;
    }

    // Reduce max across threads
    sdata[tid] = local_max;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s && sdata[tid + s] > sdata[tid])
            sdata[tid] = sdata[tid + s];
        __syncthreads();
    }
    float max_val = sdata[0];

    // Phase 2: Exp and local sum
    float local_sum = 0.0f;
    for (int j = tid; j < cols; j += nthreads) {
        float e = expf(out_row[j] - max_val);
        out_row[j] = e;
        local_sum += e;
    }

    // Reduce sum across threads
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / sdata[0];

    // Phase 3: Normalize
    for (int j = tid; j < cols; j += nthreads) {
        out_row[j] *= inv_sum;
    }
}

extern "C" void cuda_fused_attn_softmax(const float* scores, float* output,
                                          float scale, int rows, int cols) {
    int threads = (cols < 256) ? cols : 256;
    // Round up to next power of 2 for reduction
    int t = 1;
    while (t < threads) t <<= 1;
    threads = t;
    int smem = threads * sizeof(float);
    fused_attn_softmax_kernel<<<rows, threads, smem>>>(scores, output, scale, rows, cols);
}

// =============================================================================
// Fused attention: softmax backward + scale (backward)
// =============================================================================
// Combines softmax_backward and scaling into one kernel.
// Given softmax output s and upstream gradient ds:
//   d_scores[i,j] = scale * s[i,j] * (ds[i,j] - dot(ds[i,:], s[i,:]))

__global__ void fused_attn_softmax_backward_kernel(const float* s, const float* ds,
                                                     float* result, float scale,
                                                     int rows, int cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    if (row >= rows) return;

    const float* s_row = s + row * cols;
    const float* ds_row = ds + row * cols;
    float* out_row = result + row * cols;

    extern __shared__ float sdata[];

    // Local dot product
    float local_dot = 0.0f;
    for (int j = tid; j < cols; j += nthreads) {
        local_dot += ds_row[j] * s_row[j];
    }

    // Reduce dot across threads
    sdata[tid] = local_dot;
    __syncthreads();
    for (int s2 = nthreads / 2; s2 > 0; s2 >>= 1) {
        if (tid < s2) sdata[tid] += sdata[tid + s2];
        __syncthreads();
    }
    float dot = sdata[0];

    // Compute output
    for (int j = tid; j < cols; j += nthreads) {
        out_row[j] = scale * s_row[j] * (ds_row[j] - dot);
    }
}

extern "C" void cuda_fused_attn_softmax_backward(const float* s, const float* ds,
                                                    float* result, float scale,
                                                    int rows, int cols) {
    int threads = (cols < 256) ? cols : 256;
    int t = 1;
    while (t < threads) t <<= 1;
    threads = t;
    int smem = threads * sizeof(float);
    fused_attn_softmax_backward_kernel<<<rows, threads, smem>>>(s, ds, result, scale, rows, cols);
}

// =============================================================================
// Embedding Gather (GPU-side token lookup)
// =============================================================================
// One thread per output element: output[pos, j] = token_emb[ids[pos], j]

__global__ void embedding_gather_kernel(const float* token_emb, const int* ids,
                                         float* output, int seq_len, int d_model) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len * d_model) return;
    int pos = idx / d_model;
    int j = idx % d_model;
    int token_id = ids[pos];
    output[idx] = token_emb[token_id * d_model + j];
}

extern "C" void cuda_embedding_gather(const float* token_emb, const int* ids,
                                       float* output, int seq_len, int d_model) {
    int n = seq_len * d_model;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    embedding_gather_kernel<<<blocks, threads>>>(token_emb, ids, output, seq_len, d_model);
}

// Embedding scatter-add backward: d_token_emb[ids[pos], j] += grad[pos, j]
// Uses atomicAdd since multiple positions may map to the same token
__global__ void embedding_scatter_add_kernel(const float* grad, const int* ids,
                                              float* d_token_emb, int seq_len, int d_model) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len * d_model) return;
    int pos = idx / d_model;
    int j = idx % d_model;
    int token_id = ids[pos];
    atomicAdd(&d_token_emb[token_id * d_model + j], grad[idx]);
}

extern "C" void cuda_embedding_scatter_add(const float* grad, const int* ids,
                                            float* d_token_emb, int seq_len, int d_model) {
    int n = seq_len * d_model;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    embedding_scatter_add_kernel<<<blocks, threads>>>(grad, ids, d_token_emb, seq_len, d_model);
}

// =============================================================================
// Fused Loss: Softmax + Cross-Entropy Loss + Gradient (one pass)
// =============================================================================
// One block per row. Computes:
//   probs[i,j] = softmax(logits[i,:])
//   loss_per_row[i] = -log(probs[i, target[i]])
//   d_logits[i,j] = (probs[i,j] - one_hot(target[i],j)) / seq_len

__global__ void fused_softmax_ce_grad_kernel(const float* logits, const int* targets,
                                              float* d_logits, float* loss_out,
                                              int rows, int cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    if (row >= rows) return;

    const float* in_row = logits + row * cols;
    float* out_row = d_logits + row * cols;

    extern __shared__ float sdata[];

    // 1. Find max for numerical stability
    float local_max = -FLT_MAX;
    for (int j = tid; j < cols; j += nthreads) {
        if (in_row[j] > local_max) local_max = in_row[j];
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s && sdata[tid + s] > sdata[tid]) sdata[tid] = sdata[tid + s];
        __syncthreads();
    }
    float max_val = sdata[0];

    // 2. Compute exp and sum
    float local_sum = 0.0f;
    for (int j = tid; j < cols; j += nthreads) {
        float e = expf(in_row[j] - max_val);
        out_row[j] = e;  // temporarily store exp
        local_sum += e;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float sum_exp = sdata[0];
    float inv_sum = 1.0f / sum_exp;
    float inv_seq = 1.0f / (float)rows;

    // 3. Compute probs, loss contribution, and gradient in one pass
    int target = targets[row];
    for (int j = tid; j < cols; j += nthreads) {
        float prob = out_row[j] * inv_sum;
        // gradient: (prob - one_hot) / seq_len
        float grad = prob * inv_seq;
        if (j == target) {
            grad -= inv_seq;
            // Atomically accumulate loss: -log(prob)
            atomicAdd(loss_out, -logf(prob + 1e-10f));
        }
        out_row[j] = grad;
    }
}

extern "C" void cuda_fused_softmax_ce_grad(const float* logits, const int* targets,
                                            float* d_logits, float* loss_out,
                                            int rows, int cols) {
    // Zero the loss accumulator
    cudaMemset(loss_out, 0, sizeof(float));

    int threads = (cols < 256) ? cols : 256;
    int t = 1;
    while (t < threads) t <<= 1;
    threads = t;
    int smem = threads * sizeof(float);
    fused_softmax_ce_grad_kernel<<<rows, threads, smem>>>(logits, targets, d_logits, loss_out, rows, cols);
}

// =============================================================================
// Fused Bias + ReLU forward: output = max(0, data + bias), mask = (output > 0)
// =============================================================================
// Combines bias_add + relu_forward into one kernel, saving a launch + memory pass.
// One thread per element.

__global__ void fused_bias_relu_kernel(const float* input, const float* bias,
                                        float* output, float* mask,
                                        int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    int j = idx % cols;
    float val = input[idx] + bias[j];
    float m = (val > 0.0f) ? 1.0f : 0.0f;
    output[idx] = val * m;
    mask[idx] = m;
}

extern "C" void cuda_fused_bias_relu(const float* input, const float* bias,
                                      float* output, float* mask,
                                      int rows, int cols) {
    int n = rows * cols;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    fused_bias_relu_kernel<<<blocks, threads>>>(input, bias, output, mask, rows, cols);
}

// =============================================================================
// Bulk Adam: update all parameters in one kernel launch
// =============================================================================
// Operates on contiguous weight store: params[0..n-1], m[0..n-1], v[0..n-1]
// Gradients gathered into a contiguous buffer matching the same layout.

__global__ void adam_bulk_kernel(float* params, float* grads,
                                 float* m, float* v,
                                 float lr, float beta1, float beta2, float eps,
                                 float bc1, float bc2, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float g = grads[idx];
    float mi = beta1 * m[idx] + (1.0f - beta1) * g;
    float vi = beta2 * v[idx] + (1.0f - beta2) * g * g;
    m[idx] = mi;
    v[idx] = vi;
    float m_hat = mi / bc1;
    float v_hat = vi / bc2;
    params[idx] -= lr * m_hat / (sqrtf(v_hat) + eps);
}

extern "C" void cuda_adam_bulk(float* params, float* grads,
                                float* m, float* v,
                                float lr, float beta1, float beta2, float eps,
                                int t, int n) {
    float bc1 = 1.0f - powf(beta1, t);
    float bc2 = 1.0f - powf(beta2, t);
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    adam_bulk_kernel<<<blocks, threads>>>(params, grads, m, v, lr, beta1, beta2, eps, bc1, bc2, n);
}

// =============================================================================
// Synchronize
// =============================================================================

extern "C" void cuda_sync() {
    cudaDeviceSynchronize();
}
