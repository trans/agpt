#ifndef AGPT_V1_BACKOFF_KERNELS_CUH
#define AGPT_V1_BACKOFF_KERNELS_CUH

// Slot-selection Step 0 — kernel-side backoff K/V slot machinery.
//
// See notes/seq-len-extension/slot-selection.md and
// todo/descendant-ancestor-scatter.md (the anc-grad design this rides on).
//
// Forward path per layer:
//   1) launch_backoff_layout (once per chunk, B-independent layout):
//      For each query q, set kv_lengths_per_q[q] (the per-query prefix length
//      passed to attention) and back_subtree_idx[q*B+bidx] / back_rope_pos[q*B+bidx]
//      (densely packed non-sentinel backoff slots, with their subtree-local
//      index and RoPE position).
//   2) launch_backoff_gather_h (per layer):
//      For each (q, bidx) with back_subtree_idx >= 0, copy
//      h_subtree[l][back_subtree_idx, :] into h_back[q*B+bidx, :]. Sentinel
//      rows get zero (harmless — sentinel positions aren't read by attention).
//   3) cuBLAS sgemm + bias add: k_backoff = h_back @ W_k^l + b_k^l (and same
//      for v_backoff). Shape: [T_q * B, D]. Done by the caller in agpt_train.cu.
//   4) launch_apply_rope_per_slot: apply RoPE to k_backoff per slot at
//      back_rope_pos[q*B+bidx] (replicated per head). v_backoff has no RoPE.
//   5) Attention runs with cuda_batched_varlen_attention_L_queries_backoff,
//      reading kv_lengths_per_q + (k_backoff, v_backoff) for prefix positions
//      p >= K_i = kv_lengths[node].
//
// Backward path per layer:
//   6) Attention backward populates dk_backoff[q*B+bidx, :] and dv_backoff[...]
//      for non-sentinel slots (sentinel slots are never read so their
//      grads stay zero from cudaMemset).
//   7) launch_apply_rope_per_slot_inverse: inverse-RoPE on dk_backoff per slot
//      at back_rope_pos. dv_backoff is untouched (V has no RoPE).
//   8) launch_backoff_scatter: for each (q, bidx) with back_subtree_idx >= 0,
//      atomic-add dk_backoff[q*B+bidx, :] into d_dkv_subtree_k_backoff[l][
//      subtree_idx, :] and dv_backoff[...] into d_dkv_subtree_v[l][...].
//      (V shares anc-grad's V accumulator since V has no RoPE asymmetry.)
//
// Fire-end (in agpt_train.cu, alongside anc-grad's existing matmuls):
//   9) dW_kw[l] += grad_scale · d_dkv_subtree_k_backoff[l]^T · h_subtree[l]
//      (V's backoff contribution is already in d_dkv_subtree_v[l] from step 8.)
//
// Memory (per-chunk scratch sized at T_q_cap):
//   h_back            [T_q_cap * B, D]   float
//   k_backoff         [T_q_cap * B, D]   float
//   v_backoff         [T_q_cap * B, D]   float
//   dk_backoff        [T_q_cap * B, D]   float
//   dv_backoff        [T_q_cap * B, D]   float
//   back_subtree_idx  [T_q_cap * B]      int
//   back_rope_pos     [T_q_cap * B * H]  int  (replicated per head for the
//                                              existing launch_rope_batched_inverse)
//   kv_lengths_per_q  [T_q_cap]          int
//
// Per-layer per-fire accumulator (alongside d_dkv_subtree_k[l]):
//   d_dkv_subtree_k_backoff[l]   [n_subtree_compact_chars, D]   float
//   (V reuses d_dkv_subtree_v[l] — no separate accumulator needed.)

#include <cstdint>

// ---------------------------------------------------------------------
// Step 1 — backoff layout (once per chunk, B-independent).
// ---------------------------------------------------------------------

__global__ void backoff_layout_kernel(
    const int* __restrict__ query_to_node,        // [T_q]
    const int* __restrict__ query_offsets,        // [N+1]
    const int* __restrict__ radix_ids,            // [N] chunk-node → global radix id
    const int* __restrict__ kv_lengths,           // [N] per-node K_i = path + own
    const int* __restrict__ K_back_subtree_idx,   // [radix_count * B] sentinel = -1
    const int* __restrict__ depth_per_query,      // [T_q] real RoPE position at each q
    int* __restrict__ back_subtree_idx,           // [T_q * B] OUT
    int* __restrict__ back_rope_pos,              // [T_q * B * H] OUT (per-head replicated)
    int* __restrict__ kv_lengths_per_q,           // [T_q] OUT
    int T_q, int B, int H, int backoff_position_kind)
{
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= T_q) return;

    int node = query_to_node[q];
    int q_start = query_offsets[node];
    int q_end   = query_offsets[node + 1];
    int L_i = q_end - q_start;
    int j   = q - q_start;
    int K_i = kv_lengths[node];
    int ancestor_len = K_i - L_i;
    int prefix_len_base = ancestor_len + j + 1;

    bool is_endpoint = (j == L_i - 1);

    int n_back = 0;
    if (is_endpoint) {
        int K_global = radix_ids[node];
        int depth = depth_per_query[q];
        // Walk sidecar levels; densely pack non-sentinels.
        for (int i = 0; i < B; i++) {
            int sub_idx = K_back_subtree_idx[K_global * B + i];
            if (sub_idx < 0) continue;
            int rpos = (backoff_position_kind == 0) ? depth : (depth + i + 1);
            int dst = q * B + n_back;
            back_subtree_idx[dst] = sub_idx;
            // back_rope_pos is replicated per head — existing launch_rope_batched_inverse
            // walks [N * H] rows of HD and uses positions[row].
            for (int h = 0; h < H; h++) {
                back_rope_pos[dst * H + h] = rpos;
            }
            n_back++;
        }
    }
    // Fill remaining slots with sentinels so the scatter skips them.
    for (int bidx = n_back; bidx < B; bidx++) {
        int dst = q * B + bidx;
        back_subtree_idx[dst] = -1;
        for (int h = 0; h < H; h++) {
            back_rope_pos[dst * H + h] = 0;
        }
    }
    kv_lengths_per_q[q] = prefix_len_base + n_back;
}

static void launch_backoff_layout(
    const int* query_to_node, const int* query_offsets,
    const int* radix_ids, const int* kv_lengths,
    const int* K_back_subtree_idx, const int* depth_per_query,
    int* back_subtree_idx, int* back_rope_pos, int* kv_lengths_per_q,
    int T_q, int B, int H, int backoff_position_kind)
{
    if (T_q <= 0) return;
    int threads = 256;
    int blocks = (T_q + threads - 1) / threads;
    backoff_layout_kernel<<<blocks, threads>>>(
        query_to_node, query_offsets, radix_ids, kv_lengths,
        K_back_subtree_idx, depth_per_query,
        back_subtree_idx, back_rope_pos, kv_lengths_per_q,
        T_q, B, H, backoff_position_kind);
}

// ---------------------------------------------------------------------
// Step 2 — gather h_subtree[l] rows into h_back, per layer.
// One thread per (slot row, channel) pair. Sentinel rows zeroed.
// ---------------------------------------------------------------------

__global__ void backoff_gather_h_kernel(
    const float* __restrict__ h_subtree_l,        // [n_subtree_chars, D]
    const int*   __restrict__ back_subtree_idx,   // [T_q * B]
    float*       __restrict__ h_back,             // [T_q * B, D] OUT
    int T_q_times_B, int D)
{
    int row = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (row >= T_q_times_B || d >= D) return;
    int sub_idx = back_subtree_idx[row];
    float val = 0.0f;
    if (sub_idx >= 0) {
        val = h_subtree_l[(long long)sub_idx * D + d];
    }
    h_back[(long long)row * D + d] = val;
}

static void launch_backoff_gather_h(
    const float* h_subtree_l, const int* back_subtree_idx,
    float* h_back, int T_q, int B, int D)
{
    int rows = T_q * B;
    if (rows <= 0 || D <= 0) return;
    int threads = (D < 128) ? 32 : 128;
    int t = 1;
    while (t < threads) t <<= 1;
    threads = (t < 32) ? 32 : t;
    dim3 blocks(rows, (D + threads - 1) / threads);
    backoff_gather_h_kernel<<<blocks, threads>>>(
        h_subtree_l, back_subtree_idx, h_back, rows, D);
}

// ---------------------------------------------------------------------
// Step 8 — scatter: dk_backoff (post-inverse-RoPE) + dv_backoff into
// d_dkv_subtree_k_backoff[l] / d_dkv_subtree_v[l] at back_subtree_idx.
//
// Atomic-add along the channel axis. Sentinel slots (idx < 0) skipped.
// ---------------------------------------------------------------------

__global__ void backoff_scatter_kernel(
    const float* __restrict__ dk_backoff,         // [T_q * B, D] (post inverse-RoPE)
    const float* __restrict__ dv_backoff,         // [T_q * B, D]
    const int*   __restrict__ back_subtree_idx,   // [T_q * B]
    float*       __restrict__ d_dkv_subtree_k_backoff_l,  // [n_subtree_chars, D]
    float*       __restrict__ d_dkv_subtree_v_l,         // [n_subtree_chars, D]
    int T_q_times_B, int D)
{
    int row = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (row >= T_q_times_B || d >= D) return;
    int sub_idx = back_subtree_idx[row];
    if (sub_idx < 0) return;
    float gk = dk_backoff[(long long)row * D + d];
    float gv = dv_backoff[(long long)row * D + d];
    atomicAdd(&d_dkv_subtree_k_backoff_l[(long long)sub_idx * D + d], gk);
    atomicAdd(&d_dkv_subtree_v_l[(long long)sub_idx * D + d], gv);
}

static void launch_backoff_scatter(
    const float* dk_backoff, const float* dv_backoff,
    const int* back_subtree_idx,
    float* d_dkv_subtree_k_backoff_l, float* d_dkv_subtree_v_l,
    int T_q, int B, int D)
{
    int rows = T_q * B;
    if (rows <= 0 || D <= 0) return;
    int threads = (D < 128) ? 32 : 128;
    int t = 1;
    while (t < threads) t <<= 1;
    threads = (t < 32) ? 32 : t;
    dim3 blocks(rows, (D + threads - 1) / threads);
    backoff_scatter_kernel<<<blocks, threads>>>(
        dk_backoff, dv_backoff, back_subtree_idx,
        d_dkv_subtree_k_backoff_l, d_dkv_subtree_v_l, rows, D);
}

#endif  // AGPT_V1_BACKOFF_KERNELS_CUH
