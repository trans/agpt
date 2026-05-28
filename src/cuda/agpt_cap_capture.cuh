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
// Load h_cap_ema buffer from disk. Validates that header matches the
// expected radix_count and d_model; on mismatch, prints a warning and
// returns false (caller should keep the buffer zero-initialized).
//
// Format matches save_h_cap_table below. Returns true iff loaded.
// ----------------------------------------------------------------------
static inline bool load_h_cap_table(
    __nv_bfloat16* d_h_cap_ema,
    int expected_radix_count,
    int expected_D,
    const char* path
) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "load_h_cap_table: cannot open %s for read; "
                "starting with zero-init buffer\n", path);
        return false;
    }

    uint32_t rc = 0, dm = 0;
    if (fread(&rc, sizeof(uint32_t), 1, f) != 1 ||
        fread(&dm, sizeof(uint32_t), 1, f) != 1) {
        fprintf(stderr, "load_h_cap_table: %s header truncated; "
                "starting with zero-init buffer\n", path);
        fclose(f);
        return false;
    }

    if ((int)rc != expected_radix_count || (int)dm != expected_D) {
        fprintf(stderr, "load_h_cap_table: %s header mismatch "
                "(file rc=%u dm=%u, expected rc=%d dm=%d); "
                "starting with zero-init buffer\n",
                path, rc, dm, expected_radix_count, expected_D);
        fclose(f);
        return false;
    }

    size_t total = (size_t)rc * (size_t)dm;
    __nv_bfloat16* h = (__nv_bfloat16*)malloc(total * sizeof(__nv_bfloat16));
    if (!h) {
        fprintf(stderr, "load_h_cap_table: malloc failed (%zu bytes); "
                "starting with zero-init buffer\n",
                total * sizeof(__nv_bfloat16));
        fclose(f);
        return false;
    }

    size_t got = fread(h, sizeof(__nv_bfloat16), total, f);
    fclose(f);
    if (got != total) {
        fprintf(stderr, "load_h_cap_table: %s body truncated "
                "(got %zu of %zu); starting with zero-init buffer\n",
                path, got, total);
        free(h);
        return false;
    }

    CUDA_CHECK(cudaMemcpy(d_h_cap_ema, h, total * sizeof(__nv_bfloat16),
                          cudaMemcpyHostToDevice));
    free(h);
    fprintf(stdout, "[h_cap] loaded %u radix x %u dim bf16 ← %s\n",
            rc, dm, path);
    fflush(stdout);
    return true;
}

// ======================================================================
// Predecessor table (cap-recurrence Phase 2)
// ----------------------------------------------------------------------
// CSR-format lookup: for each radix node K, the list of (K_prev, count)
// pairs where K_prev is the trie node whose path matches the d-window
// ending at the start of K's occurrence in the corpus.
//
// File format ('PRED' magic, version 1) — see
// src/tools/agpt_build_predecessor_table.cr.
// ======================================================================

struct PredecessorTable {
    int radix_count = 0;
    int d_window = 0;
    uint64_t total_entries = 0;
    uint64_t* d_offsets = nullptr;     // [radix_count + 1]
    uint32_t* d_pred_ids = nullptr;    // [total_entries]
    uint32_t* d_pred_counts = nullptr; // [total_entries]
};

// Loads the predecessor table from disk into device-side buffers.
// Validates header against the trainer's expected radix_count.
// Returns true iff loaded; on failure prints a warning and returns false.
static inline bool load_predecessor_table(
    PredecessorTable& out,
    int expected_radix_count,
    const char* path
) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "load_predecessor_table: cannot open %s for read\n", path);
        return false;
    }
    uint32_t magic = 0, version = 0, rc = 0, dw = 0;
    uint64_t te = 0;
    if (fread(&magic, sizeof(uint32_t), 1, f) != 1 ||
        fread(&version, sizeof(uint32_t), 1, f) != 1 ||
        fread(&rc, sizeof(uint32_t), 1, f) != 1 ||
        fread(&dw, sizeof(uint32_t), 1, f) != 1 ||
        fread(&te, sizeof(uint64_t), 1, f) != 1) {
        fprintf(stderr, "load_predecessor_table: %s header truncated\n", path);
        fclose(f);
        return false;
    }
    if (magic != 0x44455250u) {
        fprintf(stderr, "load_predecessor_table: %s bad magic 0x%08x (want 0x44455250 'PRED')\n",
                path, magic);
        fclose(f);
        return false;
    }
    if ((int)rc != expected_radix_count) {
        fprintf(stderr, "load_predecessor_table: %s radix_count mismatch (file=%u, expected=%d)\n",
                path, rc, expected_radix_count);
        fclose(f);
        return false;
    }

    out.radix_count = (int)rc;
    out.d_window = (int)dw;
    out.total_entries = te;

    // Read host-side then upload.
    uint64_t* h_off = (uint64_t*)malloc((size_t)(rc + 1) * sizeof(uint64_t));
    uint32_t* h_ids = (uint32_t*)malloc((size_t)te * sizeof(uint32_t));
    uint32_t* h_cnt = (uint32_t*)malloc((size_t)te * sizeof(uint32_t));
    if (!h_off || !h_ids || !h_cnt) {
        fprintf(stderr, "load_predecessor_table: malloc failed\n");
        if (h_off) free(h_off);
        if (h_ids) free(h_ids);
        if (h_cnt) free(h_cnt);
        fclose(f);
        return false;
    }
    if (fread(h_off, sizeof(uint64_t), (size_t)(rc + 1), f) != (size_t)(rc + 1)) {
        fprintf(stderr, "load_predecessor_table: %s offsets truncated\n", path);
        free(h_off); free(h_ids); free(h_cnt); fclose(f); return false;
    }
    if (fread(h_ids, sizeof(uint32_t), (size_t)te, f) != (size_t)te) {
        fprintf(stderr, "load_predecessor_table: %s pred_ids truncated\n", path);
        free(h_off); free(h_ids); free(h_cnt); fclose(f); return false;
    }
    if (fread(h_cnt, sizeof(uint32_t), (size_t)te, f) != (size_t)te) {
        fprintf(stderr, "load_predecessor_table: %s counts truncated\n", path);
        free(h_off); free(h_ids); free(h_cnt); fclose(f); return false;
    }
    fclose(f);

    CUDA_CHECK(cudaMalloc(&out.d_offsets, (size_t)(rc + 1) * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&out.d_pred_ids, (size_t)te * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&out.d_pred_counts, (size_t)te * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(out.d_offsets, h_off, (size_t)(rc + 1) * sizeof(uint64_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(out.d_pred_ids, h_ids, (size_t)te * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(out.d_pred_counts, h_cnt, (size_t)te * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    free(h_off); free(h_ids); free(h_cnt);

    size_t total_bytes = (size_t)(rc + 1) * sizeof(uint64_t) +
                         (size_t)te * sizeof(uint32_t) * 2;
    fprintf(stdout, "[h_cap] predecessor table loaded: %d K's, %llu pairs, "
            "d_window=%d (%.1f MB device) ← %s\n",
            (int)rc, (unsigned long long)te, (int)dw,
            (double)total_bytes / (1024.0 * 1024.0), path);
    fflush(stdout);
    return true;
}

static inline void free_predecessor_table(PredecessorTable& t) {
    if (t.d_offsets) { cudaFree(t.d_offsets); t.d_offsets = nullptr; }
    if (t.d_pred_ids) { cudaFree(t.d_pred_ids); t.d_pred_ids = nullptr; }
    if (t.d_pred_counts) { cudaFree(t.d_pred_counts); t.d_pred_counts = nullptr; }
    t.radix_count = 0;
    t.d_window = 0;
    t.total_entries = 0;
}

// ----------------------------------------------------------------------
// Compute h_in[N, D] = mass-weighted average over each K's predecessors.
//
// For chunk-local node i with radix_id rid:
//   start, end = pred_offsets[rid], pred_offsets[rid + 1]
//   total_count = sum(pred_counts[start..end))
//   h_in[i, d] = (1 / total_count) * sum_j pred_counts[j] * h_cap_ema[pred_ids[j], d]
//
// If a K has no predecessors (count==0, e.g., shallow K's with corpus
// position < d_window), h_in[i] is set to zero — caller should treat
// as "no recurrence input."
//
// One block per chunk-local node, D threads per block.
// ----------------------------------------------------------------------
__global__ void agpt_compute_h_in_kernel(
    const int*      __restrict__ d_radix_ids,    // [N]
    const uint64_t* __restrict__ d_pred_offsets, // [radix_count + 1]
    const uint32_t* __restrict__ d_pred_ids,     // [total_entries]
    const uint32_t* __restrict__ d_pred_counts,  // [total_entries]
    const __nv_bfloat16* __restrict__ h_cap_ema, // [radix_count, D]
    float*                       d_h_in,         // [N, D]  (fp32)
    int N,
    int D,
    int radix_count
) {
    int i = blockIdx.x;
    if (i >= N) return;

    int rid = d_radix_ids[i];
    if (rid < 0 || rid >= radix_count) {
        // Zero-init this row.
        for (int d = threadIdx.x; d < D; d += blockDim.x) {
            d_h_in[(long long)i * D + d] = 0.0f;
        }
        return;
    }

    uint64_t start = d_pred_offsets[rid];
    uint64_t end   = d_pred_offsets[rid + 1];
    uint64_t n_preds = end - start;
    if (n_preds == 0) {
        for (int d = threadIdx.x; d < D; d += blockDim.x) {
            d_h_in[(long long)i * D + d] = 0.0f;
        }
        return;
    }

    // Compute total count (small, serial — every thread does it for now).
    // For typical n_preds ≤ a few dozen this is cheap; for shallow K's
    // (thousands of preds) it's still bounded.
    uint64_t total_count = 0;
    for (uint64_t j = start; j < end; j++) {
        total_count += d_pred_counts[j];
    }
    if (total_count == 0) {
        for (int d = threadIdx.x; d < D; d += blockDim.x) {
            d_h_in[(long long)i * D + d] = 0.0f;
        }
        return;
    }
    float inv_total = 1.0f / (float)total_count;

    // Aggregate dim d across threads.
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float sum = 0.0f;
        for (uint64_t j = start; j < end; j++) {
            uint32_t kp = d_pred_ids[j];
            uint32_t c  = d_pred_counts[j];
            float v = __bfloat162float(h_cap_ema[(long long)kp * D + d]);
            sum += (float)c * v;
        }
        d_h_in[(long long)i * D + d] = sum * inv_total;
    }
}

static inline void launch_compute_h_in(
    const int*           d_radix_ids,
    const PredecessorTable& pred,
    const __nv_bfloat16* h_cap_ema,
    float*               d_h_in,
    int N,
    int D,
    cudaStream_t stream = 0
) {
    if (N <= 0) return;
    int threads = D < 256 ? D : 256;
    dim3 grid(N);
    dim3 block(threads);
    agpt_compute_h_in_kernel<<<grid, block, 0, stream>>>(
        d_radix_ids, pred.d_offsets, pred.d_pred_ids, pred.d_pred_counts,
        h_cap_ema, d_h_in, N, D, pred.radix_count);
}

// ----------------------------------------------------------------------
// Per-fire h_in norm accumulator. Tracks running sum/count over many
// kernel launches; reported at epoch end.
// ----------------------------------------------------------------------
struct HInStatsAccumulator {
    double sum_norm = 0.0;
    double sum_norm_sq = 0.0;
    double max_norm = 0.0;
    double min_norm = 1e300;
    long long n_filled = 0;
    long long n_zero = 0;
};

static inline void accumulate_h_in_stats(
    HInStatsAccumulator& acc,
    const float* d_h_in,
    int N,
    int D
) {
    if (N <= 0) return;
    size_t total = (size_t)N * (size_t)D;
    float* h = (float*)malloc(total * sizeof(float));
    if (!h) return;
    CUDA_CHECK(cudaMemcpy(h, d_h_in, total * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; i++) {
        double sq = 0.0;
        for (int d = 0; d < D; d++) {
            float v = h[(long long)i * D + d];
            sq += (double)v * (double)v;
        }
        if (sq > 0.0) {
            double norm = sqrt(sq);
            acc.n_filled++;
            acc.sum_norm += norm;
            acc.sum_norm_sq += norm * norm;
            if (norm > acc.max_norm) acc.max_norm = norm;
            if (norm < acc.min_norm) acc.min_norm = norm;
        } else {
            acc.n_zero++;
        }
    }
    free(h);
}

static inline void report_h_in_stats(const HInStatsAccumulator& acc, const char* label) {
    long long total = acc.n_filled + acc.n_zero;
    if (total == 0) {
        fprintf(stdout, "[h_in stats%s%s] no fires sampled\n",
                label && label[0] ? " " : "", label && label[0] ? label : "");
        fflush(stdout);
        return;
    }
    double mean = acc.n_filled > 0 ? acc.sum_norm / (double)acc.n_filled : 0.0;
    double var  = acc.n_filled > 0 ? (acc.sum_norm_sq / (double)acc.n_filled) - mean * mean : 0.0;
    double std  = var > 0.0 ? sqrt(var) : 0.0;
    fprintf(stdout, "[h_in stats%s%s] fires=%lld filled=%lld zero=%lld "
            "norm: mean=%.4f std=%.4f min=%.4f max=%.4f\n",
            label && label[0] ? " " : "",
            label && label[0] ? label : "",
            total, acc.n_filled, acc.n_zero,
            mean, std,
            acc.n_filled > 0 ? acc.min_norm : 0.0,
            acc.max_norm);
    fflush(stdout);
}

// ----------------------------------------------------------------------
// Phase 2B step 1: direct h_in injection (no learnable projection).
//
// For each chunk-local node i, at its FIRST query position
// (q = h_query_offsets[i]), add `scale * h_in[i, :]` to d_x[q, :].
// No new weights; diagnostic for whether h_in carries usable signal
// before we invest in the learnable W_inject projection.
//
// One block per chunk-local node, D threads per block.
// ----------------------------------------------------------------------
__global__ void agpt_inject_h_in_direct_kernel(
    const int*   __restrict__ d_query_offsets, // [N+1]
    const float* __restrict__ d_h_in,          // [N, D]
    float*                   d_x,              // [T_q, D]
    int N,
    int D,
    float scale
) {
    int i = blockIdx.x;
    if (i >= N) return;
    int q_first = d_query_offsets[i];
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        d_x[(long long)q_first * D + d] += scale * d_h_in[(long long)i * D + d];
    }
}

static inline void launch_inject_h_in_direct(
    const int*   d_query_offsets,
    const float* d_h_in,
    float*       d_x,
    int N,
    int D,
    float scale,
    cudaStream_t stream = 0
) {
    if (N <= 0 || scale == 0.0f) return;
    int threads = D < 256 ? D : 256;
    dim3 grid(N);
    dim3 block(threads);
    agpt_inject_h_in_direct_kernel<<<grid, block, 0, stream>>>(
        d_query_offsets, d_h_in, d_x, N, D, scale);
}

// ======================================================================
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

// ======================================================================
// Phase 2B step 2: learnable injection (W_inject).
//
// Forward injects proj[i] = W_inject @ h_in[i] at each node's first
// query position; the projection GEMM and the scatter-add reuse
// launch_inject_h_in_direct (scale=1). Backward needs the upstream
// gradient dL/d_x at those same first-query positions, gathered into a
// dense [N, D] buffer so a single GEMM can form dL/dW_inject.
// ----------------------------------------------------------------------

// Gather the layer-0 input gradient (d_dx, aliased to d_x in backward)
// at each chunk-local node's FIRST query position into g[i, :].
// Inverse of agpt_inject_h_in_direct_kernel's scatter.
__global__ void agpt_gather_dx_at_qfirst_kernel(
    const int*   __restrict__ d_query_offsets, // [N+1]
    const float* __restrict__ d_dx,            // [T_q, D]
    float*                   d_g,              // [N, D]
    int N,
    int D
) {
    int i = blockIdx.x;
    if (i >= N) return;
    int q_first = d_query_offsets[i];
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        d_g[(long long)i * D + d] = d_dx[(long long)q_first * D + d];
    }
}

static inline void launch_gather_dx_at_qfirst(
    const int*   d_query_offsets,
    const float* d_dx,
    float*       d_g,
    int N,
    int D,
    cudaStream_t stream = 0
) {
    if (N <= 0) return;
    int threads = D < 256 ? D : 256;
    agpt_gather_dx_at_qfirst_kernel<<<dim3(N), dim3(threads), 0, stream>>>(
        d_query_offsets, d_dx, d_g, N, D);
}

// ======================================================================
// W_inject sidecar persistence. Kept out of the model file format so
// the experiment doesn't rev the checkpoint layout.
//
// Format:
//   uint32 d_model
//   float32 W[D*D]   (params)
//   float32 s[D*D]   (RMSProp second-moment accumulator)
// ----------------------------------------------------------------------
static inline bool save_w_inject(
    const float* d_W, const float* d_s, int D, const char* path
) {
    FILE* f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "save_w_inject: cannot open %s for write\n", path);
        return false;
    }
    size_t total = (size_t)D * (size_t)D;
    float* h = (float*)malloc(2 * total * sizeof(float));
    if (!h) { fprintf(stderr, "save_w_inject: malloc failed\n"); fclose(f); return false; }
    CUDA_CHECK(cudaMemcpy(h, d_W, total * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h + total, d_s, total * sizeof(float), cudaMemcpyDeviceToHost));
    uint32_t dm = (uint32_t)D;
    fwrite(&dm, sizeof(uint32_t), 1, f);
    fwrite(h, sizeof(float), 2 * total, f);
    fclose(f);
    free(h);
    fprintf(stdout, "[w_inject] saved %d x %d params + accumulator → %s\n", D, D, path);
    fflush(stdout);
    return true;
}

// Load W (and the RMSProp accumulator) from a sidecar. Returns false if
// the file is absent or the header mismatches; caller keeps the
// zero-initialized buffers in that case.
static inline bool load_w_inject(
    float* d_W, float* d_s, int D, const char* path
) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    uint32_t dm = 0;
    if (fread(&dm, sizeof(uint32_t), 1, f) != 1 || (int)dm != D) {
        fprintf(stderr, "load_w_inject: %s header mismatch (got D=%u, want %d)\n",
                path, dm, D);
        fclose(f);
        return false;
    }
    size_t total = (size_t)D * (size_t)D;
    float* h = (float*)malloc(2 * total * sizeof(float));
    if (!h) { fprintf(stderr, "load_w_inject: malloc failed\n"); fclose(f); return false; }
    size_t got = fread(h, sizeof(float), 2 * total, f);
    fclose(f);
    if (got != 2 * total) {
        fprintf(stderr, "load_w_inject: %s body truncated (got %zu of %zu)\n",
                path, got, 2 * total);
        free(h);
        return false;
    }
    CUDA_CHECK(cudaMemcpy(d_W, h, total * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_s, h + total, total * sizeof(float), cudaMemcpyHostToDevice));
    free(h);
    fprintf(stdout, "[w_inject] loaded %d x %d params + accumulator ← %s\n", D, D, path);
    fflush(stdout);
    return true;
}

// ======================================================================
// Phase 2B step 3 (option B): K/V-token injection via expanded KV pack.
//
// Idea: prepend one "memory" slot per node at p=0 of every attention
// layer's KV sequence. The slot's K and V come from learned projections
// of h_in (shared across layers):
//   K_inject[i] = W_k_inject @ h_in[i]
//   V_inject[i] = W_v_inject @ h_in[i]
// The attention kernel runs UNCHANGED on an expanded pack of size
// (T_kv + N) — every query implicitly gains one always-visible slot
// with no RoPE.
//
// Backward: attention writes into expanded dK/dV; extract kernel splits
// slot 0 (→ atomic-add into a fire-level dK_inject_fire accumulator
// summing across layers, since W is shared) from the rest (→
// original-shape dk/dv that feeds the existing RoPE-inverse / anc-grad
// pipeline unchanged).
// ----------------------------------------------------------------------

// Per-node: kv_offsets_exp[i] = kv_offsets[i] + i (each node gains one
// slot at the start), kv_lengths_exp[i] = kv_lengths[i] + 1. Also fills
// the sentinel at kv_offsets_exp[N].
__global__ void agpt_compute_kv_offsets_exp_kernel(
    const int* __restrict__ d_kv_offsets,     // [N+1]
    const int* __restrict__ d_kv_lengths,     // [N]
    int*                    d_kv_offsets_exp, // [N+1]
    int*                    d_kv_lengths_exp, // [N]
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > N) return;
    d_kv_offsets_exp[i] = d_kv_offsets[i] + i;
    if (i < N) d_kv_lengths_exp[i] = d_kv_lengths[i] + 1;
}

static inline void launch_compute_kv_offsets_exp(
    const int* d_kv_offsets, const int* d_kv_lengths,
    int* d_kv_offsets_exp, int* d_kv_lengths_exp,
    int N, cudaStream_t stream = 0
) {
    int threads = 128;
    int blocks  = (N + 1 + threads - 1) / threads;
    agpt_compute_kv_offsets_exp_kernel<<<blocks, threads, 0, stream>>>(
        d_kv_offsets, d_kv_lengths, d_kv_offsets_exp, d_kv_lengths_exp, N);
}

// Build the expanded KV pack from the original pack + per-node INJ.
// Called separately for K and V (same kernel, different inputs).
//   pack_exp layout: [T_kv + N, H, HD]
//   For each node i: slot 0 = inject[i] (D-dim, reshaped to H*HD),
//                    slots 1..K_i = original pack[off_orig + 0..K_i-1]
__global__ void agpt_prepend_inject_kv_kernel(
    const float* __restrict__ d_inject,        // [N, D=H*HD]
    const float* __restrict__ d_pack_orig,     // [T_kv, H, HD]
    const int*   __restrict__ d_kv_offsets,    // [N+1]   original
    const int*   __restrict__ d_kv_offsets_exp,// [N+1]   expanded
    const int*   __restrict__ d_kv_lengths,    // [N]     original
    float*                   d_pack_exp,       // [T_kv + N, H, HD]
    int N, int H, int HD
) {
    int D = H * HD;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nidx = idx / D;
    int col  = idx % D;
    if (nidx >= N) return;
    int head = col / HD;
    int hcol = col % HD;
    int K_i      = d_kv_lengths[nidx];
    int off_orig = d_kv_offsets[nidx];
    int off_exp  = d_kv_offsets_exp[nidx];
    // Slot 0 — INJ (no RoPE; per design).
    d_pack_exp[(((long long)off_exp) * H + head) * HD + hcol] =
        d_inject[(long long)nidx * D + col];
    // Slots 1..K_i — copy from original.
    for (int p = 0; p < K_i; p++) {
        d_pack_exp[(((long long)off_exp + 1 + p) * H + head) * HD + hcol] =
            d_pack_orig[(((long long)off_orig + p) * H + head) * HD + hcol];
    }
}

static inline void launch_prepend_inject_kv(
    const float* d_inject, const float* d_pack_orig,
    const int* d_kv_offsets, const int* d_kv_offsets_exp,
    const int* d_kv_lengths,
    float* d_pack_exp,
    int N, int H, int HD, cudaStream_t stream = 0
) {
    if (N <= 0) return;
    int D = H * HD;
    int total = N * D;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    agpt_prepend_inject_kv_kernel<<<blocks, threads, 0, stream>>>(
        d_inject, d_pack_orig, d_kv_offsets, d_kv_offsets_exp, d_kv_lengths,
        d_pack_exp, N, H, HD);
}

// Reverse of the prepend: split expanded dKV pack into a per-node
// inject-grad row (atomic-summed across layers into d_dinject_accum, since
// W_k_inject/W_v_inject are shared across layers) and the original-shape
// dKV pack (slots 1..K_i of expanded → slots 0..K_i-1 of original).
//
// The original-shape pack receives a fresh write (= per-layer, the
// per-layer atomicAdd accumulation already happened inside the attention
// backward into dkv_pack_exp; here we just relocate slots 1..K_i to
// where the downstream RoPE-inverse / anc-grad code expects them).
__global__ void agpt_extract_inject_dkv_kernel(
    const float* __restrict__ d_dkv_pack_exp,   // [T_kv + N, H, HD]
    const int*   __restrict__ d_kv_offsets,     // [N+1]  original
    const int*   __restrict__ d_kv_offsets_exp, // [N+1]  expanded
    const int*   __restrict__ d_kv_lengths,     // [N]    original
    float*                   d_dinject_accum,   // [N, D]  atomicAdd target
    float*                   d_dkv_pack_orig,   // [T_kv, H, HD]  fresh write
    int N, int H, int HD
) {
    int D = H * HD;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nidx = idx / D;
    int col  = idx % D;
    if (nidx >= N) return;
    int head = col / HD;
    int hcol = col % HD;
    int K_i      = d_kv_lengths[nidx];
    int off_orig = d_kv_offsets[nidx];
    int off_exp  = d_kv_offsets_exp[nidx];
    // Slot 0 of expanded → inject grad accumulator (atomic, layer-summed).
    float dinj = d_dkv_pack_exp[(((long long)off_exp) * H + head) * HD + hcol];
    atomicAdd(&d_dinject_accum[(long long)nidx * D + col], dinj);
    // Slots 1..K_i of expanded → slots 0..K_i-1 of original pack.
    for (int p = 0; p < K_i; p++) {
        d_dkv_pack_orig[(((long long)off_orig + p) * H + head) * HD + hcol] =
            d_dkv_pack_exp[(((long long)off_exp + 1 + p) * H + head) * HD + hcol];
    }
}

static inline void launch_extract_inject_dkv(
    const float* d_dkv_pack_exp,
    const int* d_kv_offsets, const int* d_kv_offsets_exp,
    const int* d_kv_lengths,
    float* d_dinject_accum, float* d_dkv_pack_orig,
    int N, int H, int HD, cudaStream_t stream = 0
) {
    if (N <= 0) return;
    int D = H * HD;
    int total = N * D;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    agpt_extract_inject_dkv_kernel<<<blocks, threads, 0, stream>>>(
        d_dkv_pack_exp, d_kv_offsets, d_kv_offsets_exp, d_kv_lengths,
        d_dinject_accum, d_dkv_pack_orig, N, H, HD);
}

// ======================================================================
// KV-inject sidecar persistence (option B).
//
// Format:
//   uint32 d_model
//   float32 W_k[D*D]
//   float32 W_v[D*D]
//   float32 s_k[D*D]   (RMSProp accumulator for W_k)
//   float32 s_v[D*D]   (RMSProp accumulator for W_v)
// ----------------------------------------------------------------------
static inline bool save_kv_inject(
    const float* d_Wk, const float* d_Wv,
    const float* d_sk, const float* d_sv,
    int D, const char* path
) {
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "save_kv_inject: cannot open %s\n", path); return false; }
    size_t total = (size_t)D * (size_t)D;
    float* h = (float*)malloc(4 * total * sizeof(float));
    if (!h) { fprintf(stderr, "save_kv_inject: malloc failed\n"); fclose(f); return false; }
    CUDA_CHECK(cudaMemcpy(h + 0*total, d_Wk, total * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h + 1*total, d_Wv, total * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h + 2*total, d_sk, total * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h + 3*total, d_sv, total * sizeof(float), cudaMemcpyDeviceToHost));
    uint32_t dm = (uint32_t)D;
    fwrite(&dm, sizeof(uint32_t), 1, f);
    fwrite(h, sizeof(float), 4 * total, f);
    fclose(f); free(h);
    fprintf(stdout, "[kv_inject] saved %d x %d {W_k,W_v,s_k,s_v} → %s\n", D, D, path);
    fflush(stdout);
    return true;
}

static inline bool load_kv_inject(
    float* d_Wk, float* d_Wv, float* d_sk, float* d_sv,
    int D, const char* path
) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    uint32_t dm = 0;
    if (fread(&dm, sizeof(uint32_t), 1, f) != 1 || (int)dm != D) {
        fprintf(stderr, "load_kv_inject: %s header mismatch (got %u, want %d)\n", path, dm, D);
        fclose(f); return false;
    }
    size_t total = (size_t)D * (size_t)D;
    float* h = (float*)malloc(4 * total * sizeof(float));
    if (!h) { fprintf(stderr, "load_kv_inject: malloc failed\n"); fclose(f); return false; }
    size_t got = fread(h, sizeof(float), 4 * total, f);
    fclose(f);
    if (got != 4 * total) {
        fprintf(stderr, "load_kv_inject: %s truncated (got %zu of %zu)\n", path, got, 4 * total);
        free(h); return false;
    }
    CUDA_CHECK(cudaMemcpy(d_Wk, h + 0*total, total * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Wv, h + 1*total, total * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sk, h + 2*total, total * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sv, h + 3*total, total * sizeof(float), cudaMemcpyHostToDevice));
    free(h);
    fprintf(stdout, "[kv_inject] loaded %d x %d {W_k,W_v,s_k,s_v} ← %s\n", D, D, path);
    fflush(stdout);
    return true;
}
