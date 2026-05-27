// Cap-recurrence: hidden-state capture at trie-edge endpoints.
//
// For each trie node in a chunk, copy the d_final_out vector at the
// endpoint query position into a persistent per-radix-id EMA buffer.
// The buffer is bf16-storage / fp32-arithmetic.
//
// Design and motivation: notes/seq-len-extension/cap-recurrence-design.md
//
// Race-safety note: this kernel writes h_cap_ema[rid, :] without atomic
// ops. Safe under the current trainer because chunks process
// sequentially on a single CUDA stream — only one block per radix_id at
// a time. If chunk-parallelism via multiple streams is introduced
// later, this kernel needs an atomic update or a per-stream staging
// buffer.

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Forward declared CUDA error check macro (defined in agpt_train.cu).
#ifndef CUDA_CHECK
#define CUDA_CHECK(x) do { cudaError_t err = (x); if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(1); } } while(0)
#endif

__global__ void agpt_capture_h_caps_kernel(
    const float* __restrict__ d_final_out,     // [T_q, D]
    const int*   __restrict__ d_query_offsets, // [N+1]
    const int*   __restrict__ d_radix_ids,     // [N]
    __nv_bfloat16*            h_cap_ema,        // [radix_count, D]
    int N,
    int D,
    float ema_alpha,
    int radix_count
) {
    int i = blockIdx.x;
    if (i >= N) return;

    int q_start = d_query_offsets[i];
    int q_end   = d_query_offsets[i + 1];
    if (q_end <= q_start) return;
    int q_endpoint = q_end - 1;

    int rid = d_radix_ids[i];
    if (rid < 0 || rid >= radix_count) return;

    const float one_minus_alpha = 1.0f - ema_alpha;

    // Tile across D in case D > blockDim.x (we typically launch
    // blockDim.x = min(D, 256), so the loop runs once for D=64).
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float new_val = d_final_out[(long long)q_endpoint * D + d];
        __nv_bfloat16 old_bf = h_cap_ema[(long long)rid * D + d];
        float old_val = __bfloat162float(old_bf);
        float ema_val = ema_alpha * old_val + one_minus_alpha * new_val;
        h_cap_ema[(long long)rid * D + d] = __float2bfloat16(ema_val);
    }
}

// Launcher. Safe to call with N == 0 (no-op).
static inline void launch_capture_h_caps(
    const float* d_final_out,
    const int*   d_query_offsets,
    const int*   d_radix_ids,
    __nv_bfloat16* h_cap_ema,
    int N,
    int D,
    int radix_count,
    float ema_alpha,
    cudaStream_t stream = 0
) {
    if (N <= 0) return;
    int threads = D < 256 ? D : 256;
    if (threads <= 0) return;
    dim3 grid(N);
    dim3 block(threads);
    agpt_capture_h_caps_kernel<<<grid, block, 0, stream>>>(
        d_final_out, d_query_offsets, d_radix_ids,
        h_cap_ema, N, D, ema_alpha, radix_count);
}

// ----------------------------------------------------------------------
// Host-side stats: count non-zero h_cap entries and report mean/std of
// L2 norms. Copies h_cap_ema to host, scans, prints to stdout.
// ----------------------------------------------------------------------
static inline void report_h_cap_stats(
    const __nv_bfloat16* d_h_cap_ema,
    int radix_count,
    int D,
    const char* label
) {
    size_t total = (size_t)radix_count * (size_t)D;
    __nv_bfloat16* h = (__nv_bfloat16*)malloc(total * sizeof(__nv_bfloat16));
    if (!h) {
        fprintf(stderr, "report_h_cap_stats: malloc failed\n");
        return;
    }
    CUDA_CHECK(cudaMemcpy(h, d_h_cap_ema, total * sizeof(__nv_bfloat16),
                          cudaMemcpyDeviceToHost));

    long long filled = 0;
    double sum_norm = 0.0;
    double sum_norm_sq = 0.0;
    double max_norm = 0.0;
    double min_norm = 1e300;

    for (int r = 0; r < radix_count; r++) {
        double sq = 0.0;
        for (int d = 0; d < D; d++) {
            float v = __bfloat162float(h[(long long)r * D + d]);
            sq += (double)v * (double)v;
        }
        if (sq > 0.0) {
            double norm = sqrt(sq);
            filled++;
            sum_norm += norm;
            sum_norm_sq += norm * norm;
            if (norm > max_norm) max_norm = norm;
            if (norm < min_norm) min_norm = norm;
        }
    }

    double mean = filled > 0 ? sum_norm / (double)filled : 0.0;
    double var  = filled > 0 ? (sum_norm_sq / (double)filled) - mean * mean : 0.0;
    double std  = var > 0.0 ? sqrt(var) : 0.0;

    fprintf(stdout, "[h_cap stats%s%s] radix_count=%d filled=%lld (%.1f%%) "
            "norm: mean=%.4f std=%.4f min=%.4f max=%.4f\n",
            label && label[0] ? " " : "",
            label && label[0] ? label : "",
            radix_count,
            filled,
            radix_count > 0 ? 100.0 * (double)filled / (double)radix_count : 0.0,
            mean, std,
            filled > 0 ? min_norm : 0.0,
            max_norm);
    fflush(stdout);

    free(h);
}

// ----------------------------------------------------------------------
// Save h_cap_ema buffer to disk in a simple binary format.
//
// Format:
//   uint32 radix_count
//   uint32 d_model
//   bf16   data[radix_count * d_model]  (little-endian raw bytes)
// ----------------------------------------------------------------------
static inline bool save_h_cap_table(
    const __nv_bfloat16* d_h_cap_ema,
    int radix_count,
    int D,
    const char* path
) {
    FILE* f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "save_h_cap_table: cannot open %s for write\n", path);
        return false;
    }

    size_t total = (size_t)radix_count * (size_t)D;
    __nv_bfloat16* h = (__nv_bfloat16*)malloc(total * sizeof(__nv_bfloat16));
    if (!h) {
        fprintf(stderr, "save_h_cap_table: malloc failed (%zu bytes)\n",
                total * sizeof(__nv_bfloat16));
        fclose(f);
        return false;
    }
    CUDA_CHECK(cudaMemcpy(h, d_h_cap_ema, total * sizeof(__nv_bfloat16),
                          cudaMemcpyDeviceToHost));

    uint32_t rc = (uint32_t)radix_count;
    uint32_t dm = (uint32_t)D;
    fwrite(&rc, sizeof(uint32_t), 1, f);
    fwrite(&dm, sizeof(uint32_t), 1, f);
    fwrite(h, sizeof(__nv_bfloat16), total, f);
    fclose(f);

    free(h);
    fprintf(stdout, "[h_cap] saved %d radix x %d dim bf16 → %s\n",
            radix_count, D, path);
    fflush(stdout);
    return true;
}
