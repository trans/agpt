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
