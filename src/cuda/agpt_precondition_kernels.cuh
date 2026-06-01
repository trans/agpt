#ifndef AGPT_V1_PRECONDITION_KERNELS_CUH
#define AGPT_V1_PRECONDITION_KERNELS_CUH

// Precondition strand — kernel-side gather machinery.
// See notes/seq-len-extension/precondition.md.
//
// Inputs (uploaded to device once at fire start):
//   d_pre_offsets       [radix_count + 1]  per-K slice into d_pre_inst_tokens
//   d_pre_inst_tokens   [n_instances, d_pre]  flat per-instance token storage
//                                              (loaded from the precondition sidecar)
//   d_pre_sample_idx    [radix_count]      per-epoch sample index per K (host-decided,
//                                          uploaded once per fire)
//
// Per-chunk inputs:
//   d_radix_ids         [N]                chunk-local node idx -> global radix id
//
// Per-chunk output:
//   d_precondition_input_tokens   [N, d_pre]   gathered tokens per K-in-chunk
//
// One thread per (k_chunk_local, j) pair. For each K in chunk:
//   inst_idx = d_pre_sample_idx[K_global]
//   slot     = d_pre_offsets[K_global] + inst_idx
//   tok[k, j] = d_pre_inst_tokens[slot * d_pre + j]
//
// Zero-instance K's (instance_count == 0): output tokens are written as 0
// (the encoder learns to treat this as a "no precondition available" signal).
// At Shakespeare d=16 / d_pre=16 only ~15 of 1.5M nodes hit this case.

#include <cstdint>

__global__ void precondition_gather_kernel(
    const int*   __restrict__ d_radix_ids,         // [N]   chunk-local -> global
    const int*   __restrict__ d_pre_offsets,       // [radix_count + 1]
    const int*   __restrict__ d_pre_inst_tokens,   // [n_instances, d_pre] flat
    const int*   __restrict__ d_pre_sample_idx,    // [radix_count]
    int*         __restrict__ d_precondition_input_tokens,  // [N, d_pre] OUT
    int N, int d_pre)
{
    int k = blockIdx.x;                 // chunk-local node index
    int j = blockIdx.y * blockDim.x + threadIdx.x;  // token position within d_pre
    if (k >= N || j >= d_pre) return;

    int K_global = d_radix_ids[k];
    int slice_start = d_pre_offsets[K_global];
    int slice_end   = d_pre_offsets[K_global + 1];
    int instance_count = slice_end - slice_start;
    int out_idx = k * d_pre + j;
    if (instance_count <= 0) {
        d_precondition_input_tokens[out_idx] = 0;
        return;
    }
    int inst_idx = d_pre_sample_idx[K_global];
    if (inst_idx < 0) inst_idx = 0;
    if (inst_idx >= instance_count) inst_idx = inst_idx % instance_count;
    int slot = slice_start + inst_idx;
    d_precondition_input_tokens[out_idx] = d_pre_inst_tokens[(long long)slot * d_pre + j];
}

static void launch_precondition_gather(
    const int* d_radix_ids,
    const int* d_pre_offsets,
    const int* d_pre_inst_tokens,
    const int* d_pre_sample_idx,
    int*       d_precondition_input_tokens,
    int N, int d_pre)
{
    if (N <= 0 || d_pre <= 0) return;
    int threads = (d_pre < 32) ? 32 : 128;
    int t = 1; while (t < threads) t <<= 1;
    threads = (t < 32) ? 32 : t;
    dim3 blocks(N, (d_pre + threads - 1) / threads);
    precondition_gather_kernel<<<blocks, threads>>>(
        d_radix_ids, d_pre_offsets, d_pre_inst_tokens, d_pre_sample_idx,
        d_precondition_input_tokens, N, d_pre);
}

// ---- Step 5: mean-pool encoder ----
//
// The simplest possible encoder over d_precondition_input_tokens[N, d_pre]:
// look up each token's embedding in the existing model token embedding
// table (wo.token_emb), average across the d_pre positions, output one
// D-dim vector per K in the chunk.
//
// Loses sequence order (treats precondition as bag-of-chars), so it's not
// the encoder we ultimately want — but it validates the injection plumbing
// in Step 6 without entangling with sequence-aware encoder complexity.
// Step 5b will swap this for a GRU.
//
// Zero new parameters: reuses wo.token_emb. Backward (when injection lands)
// will scatter gradient into the existing embedding table.

__global__ void precondition_mean_pool_kernel(
    const int*   __restrict__ d_pre_input_tokens,   // [N, d_pre]
    const float* __restrict__ d_token_emb,          // [vocab_size, D]
    float*       __restrict__ d_precondition_state, // [N, D] OUT
    int N, int d_pre, int D, int vocab_size)
{
    int k = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (k >= N || d >= D) return;

    float acc = 0.0f;
    for (int j = 0; j < d_pre; ++j) {
        int tok = d_pre_input_tokens[k * d_pre + j];
        if (tok < 0 || tok >= vocab_size) continue;  // defensive
        acc += d_token_emb[(long long)tok * D + d];
    }
    d_precondition_state[(long long)k * D + d] = acc / (float)d_pre;
}

static void launch_precondition_mean_pool(
    const int* d_pre_input_tokens,
    const float* d_token_emb,
    float* d_precondition_state,
    int N, int d_pre, int D, int vocab_size)
{
    if (N <= 0 || d_pre <= 0 || D <= 0) return;
    int threads = (D < 64) ? 32 : 64;
    int t = 1; while (t < threads) t <<= 1;
    threads = (t < 32) ? 32 : t;
    dim3 blocks(N, (D + threads - 1) / threads);
    precondition_mean_pool_kernel<<<blocks, threads>>>(
        d_pre_input_tokens, d_token_emb, d_precondition_state,
        N, d_pre, D, vocab_size);
}

// ---- Step 6: residual injection forward + backward ----
//
// Forward: at the start of layer 0 (after embedding gather, before LN1),
// each query q at chunk-local node k accumulates W_pre · precondition_state[k]
// into its residual stream d_x[q]. W_pre is zero-initialized so the first
// step is identical to baseline.

__global__ void precondition_inject_forward_kernel(
    float*       __restrict__ d_x,                  // [T_q, D] IN/OUT (+=)
    const int*   __restrict__ d_query_to_node,      // [T_q]
    const float* __restrict__ d_W_pre,              // [D, D]
    const float* __restrict__ d_precondition_state, // [N, D]
    int T_q, int D)
{
    int q = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (q >= T_q || d >= D) return;
    int k = d_query_to_node[q];
    float acc = 0.0f;
    for (int j = 0; j < D; ++j) {
        acc += d_W_pre[d * D + j]
             * d_precondition_state[(long long)k * D + j];
    }
    d_x[(long long)q * D + d] += acc;
}

static void launch_precondition_inject_forward(
    float* d_x,
    const int* d_query_to_node,
    const float* d_W_pre,
    const float* d_precondition_state,
    int T_q, int D)
{
    if (T_q <= 0 || D <= 0) return;
    int threads = (D < 64) ? 32 : 64;
    int t = 1; while (t < threads) t <<= 1;
    threads = (t < 32) ? 32 : t;
    dim3 blocks(T_q, (D + threads - 1) / threads);
    precondition_inject_forward_kernel<<<blocks, threads>>>(
        d_x, d_query_to_node, d_W_pre, d_precondition_state, T_q, D);
}

// Backward through the injection:
//   dW_pre[d, j]              += sum_q d_x_grad[q, d] * precondition_state[k_of_q, j]
//   d_precondition_state[k, j] += sum_q-at-k d_x_grad[q, d] * W_pre[d, j]
// Both contributions land via atomicAdd (per-K sharing for the second;
// dW_pre may be written by multiple queries simultaneously).

__global__ void precondition_inject_backward_kernel(
    const float* __restrict__ d_x_grad,                    // [T_q, D]
    const int*   __restrict__ d_query_to_node,             // [T_q]
    const float* __restrict__ d_W_pre,                     // [D, D]
    const float* __restrict__ d_precondition_state,        // [N, D]
    float*       __restrict__ d_W_pre_grad,                // [D, D] +=
    float*       __restrict__ d_precondition_state_grad,   // [N, D] +=
    int T_q, int D)
{
    int q = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (q >= T_q || d >= D) return;
    int k = d_query_to_node[q];
    float g_x = d_x_grad[(long long)q * D + d];
    for (int j = 0; j < D; ++j) {
        atomicAdd(&d_W_pre_grad[d * D + j],
                  g_x * d_precondition_state[(long long)k * D + j]);
        atomicAdd(&d_precondition_state_grad[(long long)k * D + j],
                  d_W_pre[d * D + j] * g_x);
    }
}

static void launch_precondition_inject_backward(
    const float* d_x_grad,
    const int* d_query_to_node,
    const float* d_W_pre,
    const float* d_precondition_state,
    float* d_W_pre_grad,
    float* d_precondition_state_grad,
    int T_q, int D)
{
    if (T_q <= 0 || D <= 0) return;
    int threads = (D < 64) ? 32 : 64;
    int t = 1; while (t < threads) t <<= 1;
    threads = (t < 32) ? 32 : t;
    dim3 blocks(T_q, (D + threads - 1) / threads);
    precondition_inject_backward_kernel<<<blocks, threads>>>(
        d_x_grad, d_query_to_node, d_W_pre, d_precondition_state,
        d_W_pre_grad, d_precondition_state_grad, T_q, D);
}

// Backward through mean-pool: scatter d_precondition_state_grad into
// the token embedding gradient. For each (k, j) input token slot:
//   d_token_emb_grad[tok_at(k, j), :] += d_precondition_state_grad[k, :] / d_pre

__global__ void precondition_mean_pool_backward_kernel(
    const float* __restrict__ d_precondition_state_grad,  // [N, D]
    const int*   __restrict__ d_pre_input_tokens,         // [N, d_pre]
    float*       __restrict__ d_token_emb_grad,           // [vocab_size, D] +=
    int N, int d_pre, int D, int vocab_size, float inv_d_pre)
{
    int k = blockIdx.x;
    int slot = blockIdx.y;                                // index in [0, d_pre)
    int d = blockIdx.z * blockDim.x + threadIdx.x;
    if (k >= N || slot >= d_pre || d >= D) return;
    int tok = d_pre_input_tokens[k * d_pre + slot];
    if (tok < 0 || tok >= vocab_size) return;
    float g = d_precondition_state_grad[(long long)k * D + d] * inv_d_pre;
    atomicAdd(&d_token_emb_grad[(long long)tok * D + d], g);
}

static void launch_precondition_mean_pool_backward(
    const float* d_precondition_state_grad,
    const int* d_pre_input_tokens,
    float* d_token_emb_grad,
    int N, int d_pre, int D, int vocab_size)
{
    if (N <= 0 || d_pre <= 0 || D <= 0) return;
    int threads = (D < 64) ? 32 : 64;
    int t = 1; while (t < threads) t <<= 1;
    threads = (t < 32) ? 32 : t;
    dim3 blocks(N, d_pre, (D + threads - 1) / threads);
    float inv_d_pre = 1.0f / (float)d_pre;
    precondition_mean_pool_backward_kernel<<<blocks, threads>>>(
        d_precondition_state_grad, d_pre_input_tokens, d_token_emb_grad,
        N, d_pre, D, vocab_size, inv_d_pre);
}

// ---- Step 5b: GRU encoder (replaces Step 5 mean-pool) ----
//
// One-layer GRU over the d_pre token embeddings per K. At each timestep t:
//   r_t = sigmoid(W_ir·x_t + b_ir + W_hr·h_{t-1} + b_hr)
//   z_t = sigmoid(W_iz·x_t + b_iz + W_hz·h_{t-1} + b_hz)
//   m_t = W_hn·h_{t-1} + b_hn        (saved for backward)
//   n_t = tanh(W_in·x_t + b_in + r_t * m_t)
//   h_t = (1 - z_t) * n_t + z_t * h_{t-1}
//
// Input: token embeddings from the shared model token_emb table (no
// separate input embedding parameters). All 12 GRU params start at zero,
// which makes h_t = 0 throughout → baseline parity at step 1.
//
// Forward saves h_t at every timestep (needed for backward); recomputes
// r/z/m/n during backward to save scratch memory. Storage: 16 × N × D × 4
// bytes ≈ 200 MB at N=50000/D=64/d_pre=16.

// Embedding gather for one timestep: produce d_x_t[N, D] by looking up
// d_pre_input_tokens[k, t] in the model's token embedding table.
__global__ void gru_embed_one_step_kernel(
    const int*   __restrict__ d_pre_input_tokens,  // [N, d_pre]
    const float* __restrict__ d_token_emb,         // [vocab_size, D]
    float*       __restrict__ d_x_t,               // [N, D]
    int N, int d_pre, int t, int D, int vocab_size)
{
    int k = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (k >= N || d >= D) return;
    int tok = d_pre_input_tokens[k * d_pre + t];
    float v = 0.0f;
    if (tok >= 0 && tok < vocab_size) {
        v = d_token_emb[(long long)tok * D + d];
    }
    d_x_t[(long long)k * D + d] = v;
}

static void launch_gru_embed_one_step(
    const int* d_pre_input_tokens, const float* d_token_emb,
    float* d_x_t, int N, int d_pre, int t, int D, int vocab_size)
{
    if (N <= 0 || D <= 0) return;
    int threads = (D < 64) ? 32 : 64;
    int tw = 1; while (tw < threads) tw <<= 1;
    threads = (tw < 32) ? 32 : tw;
    dim3 blocks(N, (D + threads - 1) / threads);
    gru_embed_one_step_kernel<<<blocks, threads>>>(
        d_pre_input_tokens, d_token_emb, d_x_t, N, d_pre, t, D, vocab_size);
}

// One GRU timestep forward. Computes h_t from h_prev + x_t given all
// 12 GRU params. Saves h_t into the rolling state. r/z/m/n are recomputed
// at backward time.
__global__ void gru_step_forward_kernel(
    const float* __restrict__ x_t,        // [N, D]
    const float* __restrict__ h_prev,     // [N, D]
    const float* __restrict__ W_ir, const float* __restrict__ W_iz,
    const float* __restrict__ W_in, const float* __restrict__ W_hr,
    const float* __restrict__ W_hz, const float* __restrict__ W_hn,
    const float* __restrict__ b_ir, const float* __restrict__ b_iz,
    const float* __restrict__ b_in, const float* __restrict__ b_hr,
    const float* __restrict__ b_hz, const float* __restrict__ b_hn,
    float*       __restrict__ h_next,     // [N, D]
    int N, int D)
{
    int k = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (k >= N || d >= D) return;

    float r_pre = b_ir[d] + b_hr[d];
    float z_pre = b_iz[d] + b_hz[d];
    float m     = b_hn[d];
    float in_x  = b_in[d];
    for (int j = 0; j < D; ++j) {
        float xj = x_t[(long long)k * D + j];
        float hj = h_prev[(long long)k * D + j];
        r_pre += W_ir[d * D + j] * xj + W_hr[d * D + j] * hj;
        z_pre += W_iz[d * D + j] * xj + W_hz[d * D + j] * hj;
        m     += W_hn[d * D + j] * hj;
        in_x  += W_in[d * D + j] * xj;
    }
    float r = 1.0f / (1.0f + expf(-r_pre));
    float z = 1.0f / (1.0f + expf(-z_pre));
    float n = tanhf(in_x + r * m);
    float h_prev_d = h_prev[(long long)k * D + d];
    h_next[(long long)k * D + d] = (1.0f - z) * n + z * h_prev_d;
}

static void launch_gru_step_forward(
    const float* x_t, const float* h_prev,
    const float* W_ir, const float* W_iz, const float* W_in,
    const float* W_hr, const float* W_hz, const float* W_hn,
    const float* b_ir, const float* b_iz, const float* b_in,
    const float* b_hr, const float* b_hz, const float* b_hn,
    float* h_next, int N, int D)
{
    if (N <= 0 || D <= 0) return;
    int threads = (D < 64) ? 32 : 64;
    int tw = 1; while (tw < threads) tw <<= 1;
    threads = (tw < 32) ? 32 : tw;
    dim3 blocks(N, (D + threads - 1) / threads);
    gru_step_forward_kernel<<<blocks, threads>>>(
        x_t, h_prev,
        W_ir, W_iz, W_in, W_hr, W_hz, W_hn,
        b_ir, b_iz, b_in, b_hr, b_hz, b_hn,
        h_next, N, D);
}

// One GRU timestep backward. Recomputes r/z/m/n from saved h_prev + x_t
// to avoid storing them. Accumulates parameter gradients via atomicAdd
// (large fan-in across N). Writes dh_prev (which becomes the dh_next of
// the next backward step) and dx_t (consumed by mean-pool-like embedding
// scatter-add into d_token_emb_grad — handled at the call site).

__device__ __forceinline__ float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void gru_step_backward_kernel(
    const float* __restrict__ x_t,           // [N, D]
    const float* __restrict__ h_prev,        // [N, D]
    const float* __restrict__ h_next,        // [N, D]  (the saved h_t)
    const float* __restrict__ W_ir, const float* __restrict__ W_iz,
    const float* __restrict__ W_in, const float* __restrict__ W_hr,
    const float* __restrict__ W_hz, const float* __restrict__ W_hn,
    const float* __restrict__ b_ir, const float* __restrict__ b_iz,
    const float* __restrict__ b_in, const float* __restrict__ b_hr,
    const float* __restrict__ b_hz, const float* __restrict__ b_hn,
    const float* __restrict__ dh_next,       // [N, D]  gradient on h_t
    float*       __restrict__ dh_prev,       // [N, D]  OUT (gradient on h_{t-1})
    float*       __restrict__ dx_t,          // [N, D]  OUT (gradient on x_t)
    float*       __restrict__ dW_ir, float* __restrict__ dW_iz,
    float*       __restrict__ dW_in, float* __restrict__ dW_hr,
    float*       __restrict__ dW_hz, float* __restrict__ dW_hn,
    float*       __restrict__ db_ir, float* __restrict__ db_iz,
    float*       __restrict__ db_in, float* __restrict__ db_hr,
    float*       __restrict__ db_hz, float* __restrict__ db_hn,
    int N, int D)
{
    int k = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (k >= N || d >= D) return;

    // Recompute pre-activations + activations for this timestep.
    float r_pre = b_ir[d] + b_hr[d];
    float z_pre = b_iz[d] + b_hz[d];
    float m     = b_hn[d];
    float in_x  = b_in[d];
    for (int j = 0; j < D; ++j) {
        float xj = x_t[(long long)k * D + j];
        float hj = h_prev[(long long)k * D + j];
        r_pre += W_ir[d * D + j] * xj + W_hr[d * D + j] * hj;
        z_pre += W_iz[d * D + j] * xj + W_hz[d * D + j] * hj;
        m     += W_hn[d * D + j] * hj;
        in_x  += W_in[d * D + j] * xj;
    }
    float r = sigmoidf(r_pre);
    float z = sigmoidf(z_pre);
    float n_pre = in_x + r * m;
    float n = tanhf(n_pre);
    float h_prev_d = h_prev[(long long)k * D + d];

    // Receive dh_t (gradient on h_t).
    float dh = dh_next[(long long)k * D + d];

    // Split through h_t = (1 - z) * n + z * h_prev_d
    float dn = dh * (1.0f - z);
    float dz = dh * (h_prev_d - n);
    float dh_prev_self = dh * z;   // direct copy contribution to h_prev_d

    // Through tanh
    float dn_pre = dn * (1.0f - n * n);
    // n_pre = in_x + r * m
    float din_x  = dn_pre;
    float dr     = dn_pre * m;
    float dm     = dn_pre * r;

    // Through sigmoids
    float dz_pre = dz * z * (1.0f - z);
    float dr_pre = dr * r * (1.0f - r);

    // Parameter gradients (atomic-adds; high contention across N).
    atomicAdd(&db_ir[d], dr_pre);
    atomicAdd(&db_iz[d], dz_pre);
    atomicAdd(&db_in[d], din_x);
    atomicAdd(&db_hr[d], dr_pre);
    atomicAdd(&db_hz[d], dz_pre);
    atomicAdd(&db_hn[d], dm);
    for (int j = 0; j < D; ++j) {
        float xj = x_t[(long long)k * D + j];
        float hj = h_prev[(long long)k * D + j];
        atomicAdd(&dW_ir[d * D + j], dr_pre * xj);
        atomicAdd(&dW_iz[d * D + j], dz_pre * xj);
        atomicAdd(&dW_in[d * D + j], din_x  * xj);
        atomicAdd(&dW_hr[d * D + j], dr_pre * hj);
        atomicAdd(&dW_hz[d * D + j], dz_pre * hj);
        atomicAdd(&dW_hn[d * D + j], dm     * hj);
    }

    // Compute dh_prev and dx_t contributions for this d.
    // dh_prev[k, j] += W_hr[d, j]*dr_pre + W_hz[d, j]*dz_pre + W_hn[d, j]*dm
    //              + (j==d ? dh_prev_self : 0)
    // dx_t[k, j]    += W_ir[d, j]*dr_pre + W_iz[d, j]*dz_pre + W_in[d, j]*din_x
    // These need atomicAdd across (d, j) pairs.
    for (int j = 0; j < D; ++j) {
        float dh_contrib = W_hr[d * D + j] * dr_pre
                         + W_hz[d * D + j] * dz_pre
                         + W_hn[d * D + j] * dm;
        float dx_contrib = W_ir[d * D + j] * dr_pre
                         + W_iz[d * D + j] * dz_pre
                         + W_in[d * D + j] * din_x;
        atomicAdd(&dh_prev[(long long)k * D + j], dh_contrib);
        atomicAdd(&dx_t  [(long long)k * D + j], dx_contrib);
    }
    // Add the direct h_prev_d skip into dh_prev[k, d].
    atomicAdd(&dh_prev[(long long)k * D + d], dh_prev_self);
}

static void launch_gru_step_backward(
    const float* x_t, const float* h_prev, const float* h_next,
    const float* W_ir, const float* W_iz, const float* W_in,
    const float* W_hr, const float* W_hz, const float* W_hn,
    const float* b_ir, const float* b_iz, const float* b_in,
    const float* b_hr, const float* b_hz, const float* b_hn,
    const float* dh_next,
    float* dh_prev, float* dx_t,
    float* dW_ir, float* dW_iz, float* dW_in,
    float* dW_hr, float* dW_hz, float* dW_hn,
    float* db_ir, float* db_iz, float* db_in,
    float* db_hr, float* db_hz, float* db_hn,
    int N, int D)
{
    if (N <= 0 || D <= 0) return;
    int threads = (D < 64) ? 32 : 64;
    int tw = 1; while (tw < threads) tw <<= 1;
    threads = (tw < 32) ? 32 : tw;
    dim3 blocks(N, (D + threads - 1) / threads);
    gru_step_backward_kernel<<<blocks, threads>>>(
        x_t, h_prev, h_next,
        W_ir, W_iz, W_in, W_hr, W_hz, W_hn,
        b_ir, b_iz, b_in, b_hr, b_hz, b_hn,
        dh_next, dh_prev, dx_t,
        dW_ir, dW_iz, dW_in, dW_hr, dW_hz, dW_hn,
        db_ir, db_iz, db_in, db_hr, db_hz, db_hn,
        N, D);
}

// Scatter dx_t (gradient on the t-th token's embedding) into the token
// embedding gradient. Mirrors mean-pool backward but per-timestep.
__global__ void gru_embed_scatter_kernel(
    const float* __restrict__ dx_t,                // [N, D]
    const int*   __restrict__ d_pre_input_tokens,  // [N, d_pre]
    float*       __restrict__ d_token_emb_grad,    // [vocab_size, D] +=
    int N, int d_pre, int t, int D, int vocab_size)
{
    int k = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (k >= N || d >= D) return;
    int tok = d_pre_input_tokens[k * d_pre + t];
    if (tok < 0 || tok >= vocab_size) return;
    float g = dx_t[(long long)k * D + d];
    atomicAdd(&d_token_emb_grad[(long long)tok * D + d], g);
}

static void launch_gru_embed_scatter(
    const float* dx_t, const int* d_pre_input_tokens,
    float* d_token_emb_grad,
    int N, int d_pre, int t, int D, int vocab_size)
{
    if (N <= 0 || D <= 0) return;
    int threads = (D < 64) ? 32 : 64;
    int tw = 1; while (tw < threads) tw <<= 1;
    threads = (tw < 32) ? 32 : tw;
    dim3 blocks(N, (D + threads - 1) / threads);
    gru_embed_scatter_kernel<<<blocks, threads>>>(
        dx_t, d_pre_input_tokens, d_token_emb_grad,
        N, d_pre, t, D, vocab_size);
}

// Host-side seed-based sampler. Deterministic given (base_seed, K_global, epoch).
// Mirrors v2's mix_u32 pattern (cf. PositionSamplingStageV2::sample_prefix_start_bin).
static inline uint32_t precondition_mix_u32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

// Fill `sample_idx[radix_count]` with per-K seeded indices for the given epoch.
// instance_count[k] == offsets[k+1] - offsets[k].
static inline void precondition_compute_sample_idx(
    const std::vector<uint32_t>& offsets,
    int radix_count,
    uint32_t base_seed,
    int epoch_index,
    std::vector<int>& sample_idx_out)
{
    sample_idx_out.assign((size_t)radix_count, 0);
    for (int k = 0; k < radix_count; ++k) {
        uint32_t cnt = offsets[(size_t)k + 1] - offsets[(size_t)k];
        if (cnt == 0) continue;
        uint32_t h = base_seed;
        h = precondition_mix_u32(h ^ (uint32_t)k);
        h = precondition_mix_u32(h ^ ((uint32_t)epoch_index * 0x9e3779b9u));
        sample_idx_out[(size_t)k] = (int)(h % cnt);
    }
}

#endif  // AGPT_V1_PRECONDITION_KERNELS_CUH
