#ifndef AGPT_V2_KERNELS_CUH
#define AGPT_V2_KERNELS_CUH

#include <cmath>
#include <cfloat>

#include "cuda_support.cuh"

extern "C" {
    void cuda_adam_bulk(float* params, float* grads,
                        float* m, float* v,
                        float lr, float beta1, float beta2, float eps,
                        int t, int n);
    void cuda_sgd_bulk(float* params, float* grads, float lr, int n);
    void cuda_momentum_bulk(float* params, float* grads, float* m, float lr, float beta, int n);
    void cuda_rmsprop_bulk(float* params, float* grads, float* s, float lr, float beta, float eps, int n);
    void cuda_layer_norm_forward(const float* input, float* output, float* norm_out,
                                  float* std_inv_out, const float* gamma, const float* beta,
                                  int rows, int cols);
    void cuda_layer_norm_backward(const float* grad, const float* norm,
                                  const float* std_inv, const float* gamma,
                                  float* dx, float* dgamma, float* dbeta,
                                  int rows, int cols);
    void cuda_bias_add(float* data, const float* bias, int rows, int cols);
    void cuda_fused_bias_relu(const float* input, const float* bias,
                              float* output, float* mask, int rows, int cols);
    void cuda_relu_backward(const float* grad, const float* mask, float* output, int n);
    void cuda_embedding_gather(const float* token_emb, const int* ids,
                                float* output, int seq_len, int d_model);
    void cuda_embedding_scatter_add(const float* grad, const int* ids,
                                    float* d_token_emb, int seq_len, int d_model);
    void cuda_batched_varlen_attention_L_queries(
        const float* q_packed, const float* k_packed, const float* v_packed,
        const int* query_to_node, const int* query_offsets,
        const int* kv_offsets, const int* kv_lengths,
        float* output, float* weights_out,
        int T_q, int n_heads, int head_dim, int max_kv_len, float scale);
    void cuda_batched_varlen_attention_L_queries_backward(
        const float* q_packed, const float* k_packed, const float* v_packed,
        const float* attn_weights, const float* d_out,
        const int* query_to_node, const int* query_offsets,
        const int* kv_offsets, const int* kv_lengths,
        float* dq, float* dk_full, float* dv_full,
        int T_q, int n_heads, int head_dim, int max_kv_len, float scale);
}

namespace agpt_v2 {

__global__ static void rope_batched_kernel_v2(float* x, const int* positions,
                                              const float* cos_cache, const float* sin_cache,
                                              int N, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * (dim / 2);
    if (idx >= total) return;

    int row = idx / (dim / 2);
    int half_i = idx % (dim / 2);
    int pos = positions[row];

    int j0 = 2 * half_i;
    int j1 = j0 + 1;
    float x0 = x[row * dim + j0];
    float x1 = x[row * dim + j1];

    float c = cos_cache[pos * dim + j0];
    float s = sin_cache[pos * dim + j0];

    x[row * dim + j0] = x0 * c - x1 * s;
    x[row * dim + j1] = x0 * s + x1 * c;
}

static inline void launch_rope_batched_v2(float* x, const int* positions,
                                          const float* cos_cache, const float* sin_cache,
                                          int N, int dim) {
    int total = N * (dim / 2);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    rope_batched_kernel_v2<<<blocks, threads>>>(x, positions, cos_cache, sin_cache, N, dim);
}

__global__ static void rope_batched_inverse_kernel_v2(float* x, const int* positions,
                                                      const float* cos_cache, const float* sin_cache,
                                                      int N, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * (dim / 2);
    if (idx >= total) return;

    int row = idx / (dim / 2);
    int half_i = idx % (dim / 2);
    int pos = positions[row];

    int j0 = 2 * half_i;
    int j1 = j0 + 1;
    float x0 = x[row * dim + j0];
    float x1 = x[row * dim + j1];

    float c = cos_cache[pos * dim + j0];
    float s = sin_cache[pos * dim + j0];

    x[row * dim + j0] = x0 * c + x1 * s;
    x[row * dim + j1] = -x0 * s + x1 * c;
}

static inline void launch_rope_batched_inverse_v2(float* x, const int* positions,
                                                  const float* cos_cache, const float* sin_cache,
                                                  int N, int dim) {
    int total = N * (dim / 2);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    rope_batched_inverse_kernel_v2<<<blocks, threads>>>(x, positions, cos_cache, sin_cache, N, dim);
}

__global__ static void kv_scatter_compact_bf16_v2(const float* src, const int* char_pos,
                                                   const int* compact_slot,
                                                   __nv_bfloat16* dst, int N, int d_model) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * d_model;
    if (idx >= total) return;
    int row = idx / d_model;
    int col = idx % d_model;
    int cp = char_pos[row];
    int slot = compact_slot[cp];
    if (slot < 0) return;
    dst[(long long)slot * d_model + col] = __float2bfloat16(src[row * d_model + col]);
}

static inline void launch_kv_scatter_compact_bf16_v2(const float* src, const int* char_pos,
                                                     const int* compact_slot,
                                                     __nv_bfloat16* dst, int N, int d_model) {
    int total = N * d_model;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    kv_scatter_compact_bf16_v2<<<blocks, threads>>>(src, char_pos, compact_slot, dst, N, d_model);
}

__global__ static void kv_gather_anc_compact_bf16_v2(const __nv_bfloat16* global_kv,
                                                     const int* ancestor_ids,
                                                     const int* ancestor_offsets,
                                                     const int* kv_offsets,
                                                     const int* anc_lengths,
                                                     const int* compact_slot,
                                                     float* packed_kv,
                                                     int N, int n_heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int d_model = n_heads * head_dim;
    int nidx = idx / d_model;
    int col = idx % d_model;
    if (nidx >= N) return;

    int anc_off = ancestor_offsets[nidx];
    int kv_off = kv_offsets[nidx];
    int len = anc_lengths[nidx];
    int head = col / head_dim;
    int hcol = col % head_dim;

    for (int p = 0; p < len; p++) {
        int char_pos = ancestor_ids[anc_off + p];
        int slot = compact_slot[char_pos];
        float val = (slot >= 0) ? __bfloat162float(global_kv[(long long)slot * d_model + col]) : 0.0f;
        packed_kv[((kv_off + p) * n_heads + head) * head_dim + hcol] = val;
    }
}

static inline void launch_kv_gather_anc_compact_bf16_v2(const __nv_bfloat16* global_kv,
                                                        const int* ancestor_ids,
                                                        const int* ancestor_offsets,
                                                        const int* kv_offsets,
                                                        const int* anc_lengths,
                                                        const int* compact_slot,
                                                        float* packed_kv,
                                                        int N, int n_heads, int head_dim) {
    int d_model = n_heads * head_dim;
    int total = N * d_model;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    kv_gather_anc_compact_bf16_v2<<<blocks, threads>>>(
        global_kv, ancestor_ids, ancestor_offsets, kv_offsets, anc_lengths,
        compact_slot, packed_kv, N, n_heads, head_dim);
}

__global__ static void kv_copy_own_edge_v2(const float* d_k_fresh,
                                           const int* query_offsets,
                                           const int* kv_offsets,
                                           const int* anc_lengths,
                                           const int* own_lengths,
                                           float* packed_kv,
                                           int N, int n_heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int d_model = n_heads * head_dim;
    int nidx = idx / d_model;
    int col = idx % d_model;
    if (nidx >= N) return;

    int q_off = query_offsets[nidx];
    int kv_off = kv_offsets[nidx];
    int anc_len = anc_lengths[nidx];
    int own_len = own_lengths[nidx];
    int head = col / head_dim;
    int hcol = col % head_dim;

    for (int j = 0; j < own_len; j++) {
        float val = d_k_fresh[(long long)(q_off + j) * d_model + col];
        int p = anc_len + j;
        packed_kv[((kv_off + p) * n_heads + head) * head_dim + hcol] = val;
    }
}

static inline void launch_kv_copy_own_edge_v2(const float* d_k_fresh,
                                              const int* query_offsets,
                                              const int* kv_offsets,
                                              const int* anc_lengths,
                                              const int* own_lengths,
                                              float* packed_kv,
                                              int N, int n_heads, int head_dim) {
    int d_model = n_heads * head_dim;
    int total = N * d_model;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    kv_copy_own_edge_v2<<<blocks, threads>>>(
        d_k_fresh, query_offsets, kv_offsets, anc_lengths, own_lengths,
        packed_kv, N, n_heads, head_dim);
}

__global__ static void kv_uncopy_own_edge_kernel_v2(const float* packed_grad,
                                                    const int* query_offsets,
                                                    const int* kv_offsets,
                                                    const int* anc_lengths,
                                                    const int* own_lengths,
                                                    float* d_out,
                                                    int N, int n_heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int d_model = n_heads * head_dim;
    int nidx = idx / d_model;
    int col = idx % d_model;
    if (nidx >= N) return;

    int q_off = query_offsets[nidx];
    int kv_off = kv_offsets[nidx];
    int anc_len = anc_lengths[nidx];
    int own_len = own_lengths[nidx];
    int head = col / head_dim;
    int hcol = col % head_dim;

    for (int j = 0; j < own_len; j++) {
        int p = anc_len + j;
        float val = packed_grad[((kv_off + p) * n_heads + head) * head_dim + hcol];
        d_out[(long long)(q_off + j) * d_model + col] = val;
    }
}

static inline void launch_kv_uncopy_own_edge_v2(const float* packed_grad,
                                                const int* query_offsets,
                                                const int* kv_offsets,
                                                const int* anc_lengths,
                                                const int* own_lengths,
                                                float* d_out,
                                                int N, int n_heads, int head_dim) {
    int d_model = n_heads * head_dim;
    int total = N * d_model;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    kv_uncopy_own_edge_kernel_v2<<<blocks, threads>>>(
        packed_grad, query_offsets, kv_offsets, anc_lengths, own_lengths,
        d_out, N, n_heads, head_dim);
}

__global__ static void set_compact_to_subtree_kernel_v2(int* compact_to_subtree,
                                                        const int* compact_slots,
                                                        int n_sub) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_sub) return;
    int slot = compact_slots[i];
    if (slot >= 0) compact_to_subtree[slot] = i;
}

static inline void launch_set_compact_to_subtree_v2(int* compact_to_subtree,
                                                    const int* compact_slots,
                                                    int n_sub) {
    if (n_sub <= 0) return;
    int threads = 256;
    int blocks = (n_sub + threads - 1) / threads;
    set_compact_to_subtree_kernel_v2<<<blocks, threads>>>(compact_to_subtree, compact_slots, n_sub);
}

__global__ static void scatter_anc_dkv_to_subtree_kernel_v2(const float* packed_grad,
                                                            const int* ancestor_ids,
                                                            const int* ancestor_offsets,
                                                            const int* kv_offsets,
                                                            const int* anc_lengths,
                                                            const int* compact_slot,
                                                            const int* compact_to_subtree,
                                                            float* dkv_subtree,
                                                            float grad_scale,
                                                            int N, int n_heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int d_model = n_heads * head_dim;
    int nidx = idx / d_model;
    int col = idx % d_model;
    if (nidx >= N) return;

    int anc_off = ancestor_offsets[nidx];
    int kv_off = kv_offsets[nidx];
    int len = anc_lengths[nidx];
    int head = col / head_dim;
    int hcol = col % head_dim;

    for (int p = 0; p < len; p++) {
        int char_pos = ancestor_ids[anc_off + p];
        int slot = compact_slot[char_pos];
        if (slot < 0) continue;
        int sub_idx = compact_to_subtree[slot];
        if (sub_idx < 0) continue;
        float g = packed_grad[((kv_off + p) * n_heads + head) * head_dim + hcol];
        atomicAdd(&dkv_subtree[(long long)sub_idx * d_model + col], g * grad_scale);
    }
}

static inline void launch_scatter_anc_dkv_to_subtree_v2(const float* packed_grad,
                                                        const int* ancestor_ids,
                                                        const int* ancestor_offsets,
                                                        const int* kv_offsets,
                                                        const int* anc_lengths,
                                                        const int* compact_slot,
                                                        const int* compact_to_subtree,
                                                        float* dkv_subtree,
                                                        float grad_scale,
                                                        int N, int n_heads, int head_dim) {
    int d_model = n_heads * head_dim;
    int total = N * d_model;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    scatter_anc_dkv_to_subtree_kernel_v2<<<blocks, threads>>>(
        packed_grad, ancestor_ids, ancestor_offsets, kv_offsets, anc_lengths,
        compact_slot, compact_to_subtree, dkv_subtree, grad_scale,
        N, n_heads, head_dim);
}

__global__ static void save_ln1_to_subtree_kernel_v2(const float* ln1_out,
                                                     const int* char_pos,
                                                     const int* compact_slot,
                                                     const int* compact_to_subtree,
                                                     float* h_subtree,
                                                     int T_q, int D) {
    int q = blockIdx.x;
    if (q >= T_q) return;
    int cp = char_pos[q];
    int slot = compact_slot[cp];
    if (slot < 0) return;
    int sub_idx = compact_to_subtree[slot];
    if (sub_idx < 0) return;
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        h_subtree[(long long)sub_idx * D + j] = ln1_out[(long long)q * D + j];
    }
}

static inline void launch_save_ln1_to_subtree_v2(const float* ln1_out,
                                                 const int* char_pos,
                                                 const int* compact_slot,
                                                 const int* compact_to_subtree,
                                                 float* h_subtree,
                                                 int T_q, int D) {
    int threads = (D < 256) ? D : 256;
    save_ln1_to_subtree_kernel_v2<<<T_q, threads>>>(
        ln1_out, char_pos, compact_slot, compact_to_subtree, h_subtree, T_q, D);
}

__global__ static void agpt_loss_per_query_kernel_v2(
    const float* logits,
    const int* query_to_node,
    const int* query_offsets,
    const int* radix_ids,
    const int* token_ids,
    const int* counts_offset,
    const int* counts_len,
    const int* counts_tok,
    const int* counts_val,
    const float* query_weights,
    float* d_logits,
    float* loss_out,
    int T_q, int V) {
    int q = blockIdx.x;
    if (q >= T_q) return;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    const float* in_row = logits + (long long)q * V;
    float* grad_row = d_logits + (long long)q * V;
    extern __shared__ float sdata[];

    float local_max = -FLT_MAX;
    for (int j = tid; j < V; j += nthreads) {
        if (in_row[j] > local_max) local_max = in_row[j];
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s && sdata[tid + s] > sdata[tid]) sdata[tid] = sdata[tid + s];
        __syncthreads();
    }
    float max_val = sdata[0];

    float local_sum = 0.0f;
    for (int j = tid; j < V; j += nthreads) {
        float e = expf(in_row[j] - max_val);
        grad_row[j] = e;
        local_sum += e;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / sdata[0];
    for (int j = tid; j < V; j += nthreads) grad_row[j] *= inv_sum;
    __syncthreads();

    if (tid == 0) {
        int n_idx = query_to_node[q];
        int node_end_q = query_offsets[n_idx + 1];
        bool is_endpoint = (q + 1) == node_end_q;

        if (is_endpoint) {
            int radix_id = radix_ids[n_idx];
            int start = counts_offset[radix_id];
            int end = start + counts_len[radix_id];
            if (start == end) {
                for (int j = 0; j < V; j++) grad_row[j] = 0.0f;
                loss_out[q] = 0.0f;
                return;
            }
            int total = 0;
            for (int e = start; e < end; e++) total += counts_val[e];
            float total_f = (float)total;
            float weight = query_weights ? query_weights[q] : 1.0f;
            float loss = 0.0f;
            for (int e = start; e < end; e++) {
                int tok = counts_tok[e];
                int cnt = counts_val[e];
                float p = grad_row[tok];
                loss -= (cnt / total_f) * logf(p + 1e-10f);
                grad_row[tok] -= cnt / total_f;
            }
            if (weight != 1.0f) {
                loss *= weight;
                for (int j = 0; j < V; j++) grad_row[j] *= weight;
            }
            loss_out[q] = loss;
        } else {
            int target = token_ids[q + 1];
            float p = grad_row[target];
            float loss = -logf(p + 1e-10f);
            grad_row[target] -= 1.0f;
            float weight = query_weights ? query_weights[q] : 1.0f;
            if (weight != 1.0f) {
                loss *= weight;
                for (int j = 0; j < V; j++) grad_row[j] *= weight;
            }
            loss_out[q] = loss;
        }
    }
}

static inline void launch_agpt_loss_per_query_v2(const float* logits,
                                                 const int* query_to_node,
                                                 const int* query_offsets,
                                                 const int* radix_ids,
                                                 const int* token_ids,
                                                 const int* counts_offset,
                                                 const int* counts_len,
                                                 const int* counts_tok,
                                                 const int* counts_val,
                                                 const float* query_weights,
                                                 float* d_logits,
                                                 float* loss_out,
                                                 int T_q, int V) {
    int threads = (V < 256) ? V : 256;
    int t = 1;
    while (t < threads) t <<= 1;
    threads = (t < 32) ? 32 : t;
    int smem = threads * sizeof(float);
    agpt_loss_per_query_kernel_v2<<<T_q, threads, smem>>>(
        logits, query_to_node, query_offsets, radix_ids, token_ids,
        counts_offset, counts_len, counts_tok, counts_val,
        query_weights, d_logits, loss_out, T_q, V);
}

__global__ static void bias_grad_accum_kernel_v2(const float* grad, int rows, int cols,
                                                 float scale, float* bias) {
    int col = blockIdx.x;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    if (col >= cols) return;

    extern __shared__ float sdata[];
    float local = 0.0f;
    for (int r = tid; r < rows; r += nthreads) {
        local += grad[(long long)r * cols + col];
    }
    sdata[tid] = local;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) bias[col] += scale * sdata[0];
}

static inline void launch_bias_grad_accum_v2(const float* grad, int rows, int cols,
                                             float scale, float* bias) {
    int threads = 256;
    if (rows < threads) {
        int t = 1;
        while (t < rows) t <<= 1;
        threads = (t < 32) ? 32 : t;
    }
    int smem = threads * sizeof(float);
    bias_grad_accum_kernel_v2<<<cols, threads, smem>>>(grad, rows, cols, scale, bias);
}

__global__ static void elem_add_kernel_v2(float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

static inline void launch_elem_add_v2(float* a, const float* b, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    elem_add_kernel_v2<<<blocks, threads>>>(a, b, n);
}

static inline void build_rope_cache_v2(float** d_cos, float** d_sin,
                                       int max_seq, int dim, float base = 10000.0f) {
    int half = dim / 2;
    float* h_cos = (float*)std::malloc((size_t)max_seq * dim * sizeof(float));
    float* h_sin = (float*)std::malloc((size_t)max_seq * dim * sizeof(float));

    for (int pos = 0; pos < max_seq; pos++) {
        for (int i = 0; i < half; i++) {
            float theta = pos / std::pow(base, 2.0f * i / dim);
            float c = std::cos(theta);
            float s = std::sin(theta);
            h_cos[pos * dim + 2 * i] = c;
            h_cos[pos * dim + 2 * i + 1] = c;
            h_sin[pos * dim + 2 * i] = s;
            h_sin[pos * dim + 2 * i + 1] = s;
        }
    }

    AGPT_V2_CUDA_CHECK(cudaMalloc(d_cos, (size_t)max_seq * dim * sizeof(float)));
    AGPT_V2_CUDA_CHECK(cudaMalloc(d_sin, (size_t)max_seq * dim * sizeof(float)));
    AGPT_V2_CUDA_CHECK(cudaMemcpy(*d_cos, h_cos, (size_t)max_seq * dim * sizeof(float), cudaMemcpyHostToDevice));
    AGPT_V2_CUDA_CHECK(cudaMemcpy(*d_sin, h_sin, (size_t)max_seq * dim * sizeof(float), cudaMemcpyHostToDevice));
    std::free(h_cos);
    std::free(h_sin);
}

}  // namespace agpt_v2

#endif
