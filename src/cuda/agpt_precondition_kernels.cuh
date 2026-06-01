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
