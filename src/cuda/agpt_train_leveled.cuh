// ============================================================================
// agpt_train_leveled.cuh
// ============================================================================
//
// LEGACY leveled-trie training path. Preserved here for future revisit —
// the per-subtree radix path (run_radix_training in agpt_train.cu) is the
// active code path used by all current recipes, but the level-by-level
// processing approach captured here has intellectual content worth coming
// back to:
//
//   - The depth-loop iteration order is structurally cleaner than the
//     chunked BFS-sort path
//   - It has the kv_scatter_add scaffolding for descendant→ancestor
//     gradient flow (though that flow is still incomplete in this path too;
//     dW_kw / dW_vw / dW_kb / dW_vb are declared but never written — see
//     comments around line 3604-3617 of agpt_train.cu pre-extraction)
//   - "Level d, then level d+1, ..." may be a useful frame for future
//     work on staleness-free training or for parallel-across-levels
//     scheduling
//
// This file is included as plain text from agpt_train.cu, preserving the
// shared translation unit. All shared kernels, types (Config, TrieData,
// WeightOffsets, etc.), helper functions, and CLI parsing remain in
// agpt_train.cu. Nothing here is callable from outside the trainer.
//
// To revisit this path: dispatch in main() at the format=0 branch of
// the trie-format detect (currently always hits the radix or
// per-subtree manifest branches first).
// ============================================================================

// ============================================================================
// GPU Training State
// ============================================================================

struct TrainState {
    // Model weights (GPU)
    float* d_weights;
    float* d_grads;
    float* d_adam_m;
    float* d_adam_v;
    int adam_t;

    // KV cache: [n_layers][total_nodes * d_model]
    float** d_kv_keys;    // array of n_layers GPU pointers
    float** d_kv_values;

    // KV gradient accumulators (for backward)
    float** d_dkv_keys;
    float** d_dkv_values;

    // RoPE cache
    float* d_rope_cos;  // [max_seq, head_dim]
    float* d_rope_sin;

    // Trie data (GPU)
    int* d_tokens;
    int* d_parents;
    int* d_depths;
    int* d_counts_offset;
    int* d_counts_tok;
    int* d_counts_val;
    int* d_ancestor_offset;
    int* d_ancestor_ids;

    // Working buffers (allocated to max depth width)
    float* d_x;           // [max_N, d_model]
    float* d_x_res1;      // residual save
    float* d_x_res2;
    float* d_ln_out;      // [max_N, d_model]
    float* d_ln_norm;     // [max_N, d_model]
    float* d_ln_std_inv;  // [max_N, 1] (padded to max_N)
    float* d_q;           // [max_N, d_model]
    float* d_k;
    float* d_v;
    float* d_attn_out;    // [max_N, d_model]
    float* d_ff_h;        // [max_N, d_ff]
    float* d_ff_mask;     // [max_N, d_ff]
    float* d_ff_out;      // [max_N, d_model]
    float* d_logits;      // [max_N, vocab]
    float* d_d_logits;    // loss gradient
    float* d_loss;        // [max_N] per-node loss

    // Backward working buffers
    float* d_dx;          // [max_N, d_model]
    float* d_d_ln_out;
    float* d_d_attn_out;
    float* d_dq;
    float* d_dk;          // for projection backward
    float* d_dv;
    float* d_d_ff_h;      // [max_N, d_ff]
    float* d_d_ff_out;    // [max_N, d_model]
    float* d_d_ln2_out;

    // Varlen attention packed buffers
    float* d_q_packed;     // [max_N * n_heads, head_dim]
    float* d_kv_packed_k;  // [max_total_kv * n_heads, head_dim]
    float* d_kv_packed_v;
    float* d_attn_weights; // [max_N * n_heads * max_prefix_len]
    float* d_attn_out_packed; // [max_N * n_heads, head_dim]

    int* d_node_ids;       // [max_N] node ids for current depth
    int* d_positions;      // [max_N] positions (depth - 1)
    int* d_kv_offsets;     // [max_N] packed KV offset per node
    int* d_kv_lengths;     // [max_N] prefix length per node
    int* d_anc_offsets;    // [max_N] ancestor offset per node (into d_ancestor_ids)

    // Backward: dq/dk/dv from varlen attention backward
    float* d_dq_packed;
    float* d_dk_packed;
    float* d_dv_packed;

    // Per-layer saved forward state for backward (arrays of n_layers GPU pointers)
    float** saved_x_res1;      // [N, D] input to each layer
    float** saved_ln1_norm;    // [N, D] LN1 normalized
    float** saved_ln1_std_inv; // [N] LN1 std_inv
    float** saved_ln1_out;     // [N, D] LN1 output (QKV input)
    float** saved_x_res2;      // [N, D] input to FFN block
    float** saved_ln2_norm;    // [N, D] LN2 normalized
    float** saved_ln2_std_inv; // [N] LN2 std_inv
    float** saved_ln2_out;     // [N, D] LN2 output (FFN input)
    float** saved_ff_h;        // [N, F] post-ReLU hidden
    float** saved_ff_mask;     // [N, F] ReLU mask
    float** saved_attn_out;    // [N, D] attention output (WO input)
    float** saved_attn_weights;// [N * H * max_depth] per-layer

    // Final layer saved state
    float* saved_final_norm;    // [N, D]
    float* saved_final_std_inv; // [N]
    float* saved_final_out;     // [N, D] final LN output

    bool kv_on_host;       // true if KV cache lives in pinned host memory
    int max_N;             // max nodes at any depth
    int max_total_kv;      // max total KV positions for a depth
};

TrainState allocate_train_state(const Config& cfg, const TrieData& trie,
                                 const WeightOffsets& wo) {
    TrainState s;
    memset(&s, 0, sizeof(s));

    int D = cfg.d_model;
    int F = cfg.d_ff;
    int V = cfg.vocab_size;
    int L = cfg.n_layers;
    int H = cfg.n_heads;
    int HD = cfg.head_dim;

    // Weights + grads + Adam
    CUDA_CHECK(cudaMalloc(&s.d_weights, wo.total_floats * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_grads,   wo.total_floats * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_adam_m,   wo.total_floats * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_adam_v,   wo.total_floats * sizeof(float)));
    CUDA_CHECK(cudaMemset(s.d_adam_m, 0, wo.total_floats * sizeof(float)));
    CUDA_CHECK(cudaMemset(s.d_adam_v, 0, wo.total_floats * sizeof(float)));
    s.adam_t = 0;

    // KV caches — use cudaMallocManaged (Unified Memory).
    // Single pointer accessible from both CPU and GPU; the CUDA driver pages
    // data on demand between VRAM and host RAM. Working set for our access
    // pattern is small (chunk_size × depth × d_model), so most of the KV
    // can live in host RAM while only the currently-touched ancestors
    // migrate to GPU. Scales beyond GPU VRAM with no explicit paging code.
    long long kv_bytes = (long long)trie.total_nodes * D * sizeof(float);
    long long total_kv_bytes = kv_bytes * 2 * L; // K+V × layers, no grads

    // Safety: check against available RAM + swap. Managed memory can be paged
    // to swap, so swap counts. But if we'd use >80% of RAM+swap, refuse.
    {
        struct sysinfo si;
        if (sysinfo(&si) == 0) {
            long long avail_total = (long long)(si.freeram + si.freeswap) * si.mem_unit;
            long long safe_limit = (avail_total * 4) / 5;
            if (total_kv_bytes > safe_limit) {
                fprintf(stderr,
                    "REFUSED: KV cache would need %.1f GB but only %.1f GB (RAM+swap) available (80%% limit = %.1f GB).\n"
                    "  Close some apps, add swap, or reduce max_depth.\n",
                    total_kv_bytes / 1e9, avail_total / 1e9, safe_limit / 1e9);
                exit(1);
            }
            long long avail_ram = (long long)si.freeram * si.mem_unit;
            if (total_kv_bytes > avail_ram) {
                fprintf(stderr,
                    "NOTE: KV cache (%.1f GB) exceeds free RAM (%.1f GB); will use swap (slow).\n",
                    total_kv_bytes / 1e9, avail_ram / 1e9);
            }
        }
    }

    s.d_kv_keys   = (float**)malloc(L * sizeof(float*));
    s.d_kv_values = (float**)malloc(L * sizeof(float*));
    s.d_dkv_keys   = (float**)malloc(L * sizeof(float*));
    s.d_dkv_values = (float**)malloc(L * sizeof(float*));
    for (int l = 0; l < L; l++) {
        CUDA_CHECK(cudaMallocManaged(&s.d_kv_keys[l],   kv_bytes));
        CUDA_CHECK(cudaMallocManaged(&s.d_kv_values[l],  kv_bytes));
        // Skip KV gradient accumulators for v1 (Wk/Wv grads approximate anyway)
        s.d_dkv_keys[l] = NULL;
        s.d_dkv_values[l] = NULL;
    }
    s.kv_on_host = false; // Using unified memory — always use GPU kernels
    printf("  KV cache: %.1f MB unified memory (driver-paged between GPU and host)\n",
           total_kv_bytes / 1e6);

    // RoPE cache — one cache for head_dim (all heads uniform)
    build_rope_cache(&s.d_rope_cos, &s.d_rope_sin, cfg.seq_len, HD);

    // Trie data upload
    CUDA_CHECK(cudaMalloc(&s.d_tokens,  trie.total_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s.d_parents, trie.total_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s.d_depths,  trie.total_nodes * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(s.d_tokens,  trie.tokens,  trie.total_nodes * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.d_parents, trie.parents, trie.total_nodes * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.d_depths,  trie.depths,  trie.total_nodes * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&s.d_counts_offset, (trie.total_nodes + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s.d_counts_tok,    trie.total_counts * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s.d_counts_val,    trie.total_counts * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(s.d_counts_offset, trie.counts_offset, (trie.total_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.d_counts_tok, trie.counts_tok, trie.total_counts * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.d_counts_val, trie.counts_val, trie.total_counts * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&s.d_ancestor_offset, (trie.total_nodes + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s.d_ancestor_ids,    trie.total_ancestor_entries * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(s.d_ancestor_offset, trie.ancestor_offset, (trie.total_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.d_ancestor_ids, trie.ancestor_ids, trie.total_ancestor_entries * sizeof(int), cudaMemcpyHostToDevice));

    // Working buffers sized to CHUNK_SIZE, not max depth width.
    // Each depth is processed in chunks to bound GPU memory.
    #define CHUNK_SIZE 50000
    s.max_N = CHUNK_SIZE;
    // Max total KV for a chunk: CHUNK_SIZE * max_depth
    s.max_total_kv = CHUNK_SIZE * trie.max_depth;

    {
        int actual_max = 0;
        for (int d = 0; d < trie.depth_file_count; d++) {
            if (trie.depth_count[d] > actual_max) actual_max = trie.depth_count[d];
        }
        printf("  Max depth width: %d nodes (chunked to %d), max total KV per chunk: %d\n",
               actual_max, CHUNK_SIZE, s.max_total_kv);
    }

    int N = s.max_N;
    int TKV = s.max_total_kv;

    // Working buffers
    CUDA_CHECK(cudaMalloc(&s.d_x,          (long long)N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_x_res1,     (long long)N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_x_res2,     (long long)N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_ln_out,     (long long)N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_ln_norm,    (long long)N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_ln_std_inv, (long long)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_q,          (long long)N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_k,          (long long)N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_v,          (long long)N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_attn_out,   (long long)N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_ff_h,       (long long)N * F * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_ff_mask,    (long long)N * F * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_ff_out,     (long long)N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_logits,     (long long)N * V * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_d_logits,   (long long)N * V * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_loss,       (long long)N * sizeof(float)));

    // Backward buffers
    CUDA_CHECK(cudaMalloc(&s.d_dx,          (long long)N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_d_ln_out,    (long long)N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_d_attn_out,  (long long)N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_dq,          (long long)N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_dk,          (long long)N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_dv,          (long long)N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_d_ff_h,      (long long)N * F * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_d_ff_out,    (long long)N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_d_ln2_out,   (long long)N * D * sizeof(float)));

    // Varlen attention buffers
    CUDA_CHECK(cudaMalloc(&s.d_q_packed,       (long long)N * H * HD * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_attn_out_packed,(long long)N * H * HD * sizeof(float)));
    if (TKV > 0) {
        CUDA_CHECK(cudaMalloc(&s.d_kv_packed_k,   (long long)TKV * H * HD * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s.d_kv_packed_v,   (long long)TKV * H * HD * sizeof(float)));
        // attn_weights: N * H entries, each up to max_depth positions
        int max_prefix = trie.max_depth;
        CUDA_CHECK(cudaMalloc(&s.d_attn_weights,   (long long)N * H * max_prefix * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s.d_dq_packed,      (long long)N * H * HD * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s.d_dk_packed,      (long long)TKV * H * HD * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s.d_dv_packed,      (long long)TKV * H * HD * sizeof(float)));
    }

    // Per-layer saved forward state
    auto alloc_layer_bufs = [&](float*** arr, long long size) {
        *arr = (float**)malloc(L * sizeof(float*));
        for (int l = 0; l < L; l++)
            CUDA_CHECK(cudaMalloc(&(*arr)[l], size * sizeof(float)));
    };
    alloc_layer_bufs(&s.saved_x_res1,      (long long)N * D);
    alloc_layer_bufs(&s.saved_ln1_norm,     (long long)N * D);
    alloc_layer_bufs(&s.saved_ln1_std_inv,  (long long)N);
    alloc_layer_bufs(&s.saved_ln1_out,      (long long)N * D);
    alloc_layer_bufs(&s.saved_x_res2,       (long long)N * D);
    alloc_layer_bufs(&s.saved_ln2_norm,     (long long)N * D);
    alloc_layer_bufs(&s.saved_ln2_std_inv,  (long long)N);
    alloc_layer_bufs(&s.saved_ln2_out,      (long long)N * D);
    alloc_layer_bufs(&s.saved_ff_h,         (long long)N * F);
    alloc_layer_bufs(&s.saved_ff_mask,      (long long)N * F);
    alloc_layer_bufs(&s.saved_attn_out,     (long long)N * D);
    {
        int mp = trie.max_depth;
        alloc_layer_bufs(&s.saved_attn_weights, (long long)N * H * mp);
    }
    CUDA_CHECK(cudaMalloc(&s.saved_final_norm,    (long long)N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.saved_final_std_inv, (long long)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.saved_final_out,     (long long)N * D * sizeof(float)));

    // Node tracking per depth
    CUDA_CHECK(cudaMalloc(&s.d_node_ids,    N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s.d_positions,   N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s.d_kv_offsets,  N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s.d_kv_lengths,  N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s.d_anc_offsets, N * sizeof(int)));

    return s;
}

// ============================================================================
// Training epoch
// ============================================================================

float train_epoch(TrainState& s, const Config& cfg, const TrieData& trie,
                   const WeightOffsets& wo, cublasHandle_t cublas) {
    int D = cfg.d_model;
    int F = cfg.d_ff;
    int V = cfg.vocab_size;
    int L = cfg.n_layers;
    int H = cfg.n_heads;
    int HD = cfg.head_dim;

    double total_loss = 0.0;
    int nodes_trained = 0;

    // Zero KV caches at start of epoch
    for (int l = 0; l < L; l++) {
        long long kv_bytes = (long long)trie.total_nodes * D * sizeof(float);
        if (s.kv_on_host) {
            memset(s.d_kv_keys[l], 0, kv_bytes);
            memset(s.d_kv_values[l], 0, kv_bytes);
        } else {
            CUDA_CHECK(cudaMemset(s.d_kv_keys[l], 0, kv_bytes));
            CUDA_CHECK(cudaMemset(s.d_kv_values[l], 0, kv_bytes));
        }
    }

    // Pre-build depth→node_id lists (once, reused across epochs)
    // Stored in TrieData but we build here for convenience
    int** depth_node_lists = (int**)malloc(trie.depth_file_count * sizeof(int*));
    for (int d = 0; d < trie.depth_file_count; d++) {
        depth_node_lists[d] = (int*)malloc(trie.depth_count[d] * sizeof(int));
    }
    {
        int* depth_idx = (int*)calloc(trie.depth_file_count, sizeof(int));
        for (int id = 0; id < trie.total_nodes; id++) {
            int d = trie.depths[id];
            if (d >= 0 && d < trie.depth_file_count) {
                depth_node_lists[d][depth_idx[d]++] = id;
            }
        }
        free(depth_idx);
    }

    // Process each depth level
    for (int depth = 1; depth < trie.depth_file_count; depth++) {
        int N_total = trie.depth_count[depth];
        if (N_total == 0) continue;

        int* all_node_ids = depth_node_lists[depth];
        int max_prefix = depth;

        // Process in chunks of CHUNK_SIZE
        for (int chunk_start = 0; chunk_start < N_total; chunk_start += CHUNK_SIZE) {
        int N = (chunk_start + CHUNK_SIZE <= N_total) ? CHUNK_SIZE : (N_total - chunk_start);

        int* h_node_ids   = (int*)malloc(N * sizeof(int));
        int* h_positions  = (int*)malloc(N * sizeof(int));
        int* h_kv_offsets = (int*)malloc(N * sizeof(int));
        int* h_kv_lengths = (int*)malloc(N * sizeof(int));
        int* h_anc_offsets = (int*)malloc(N * sizeof(int));

        int total_kv = 0;
        for (int i = 0; i < N; i++) {
            int id = all_node_ids[chunk_start + i];
            h_node_ids[i] = id;
            h_positions[i] = depth - 1;
            h_kv_offsets[i] = total_kv;
            h_kv_lengths[i] = depth;
            h_anc_offsets[i] = trie.ancestor_offset[id];
            total_kv += depth;
        }

        CUDA_CHECK(cudaMemcpy(s.d_node_ids,    h_node_ids,    N * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(s.d_positions,   h_positions,   N * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(s.d_kv_offsets,  h_kv_offsets,  N * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(s.d_kv_lengths,  h_kv_lengths,  N * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(s.d_anc_offsets, h_anc_offsets, N * sizeof(int), cudaMemcpyHostToDevice));

        // ======== FORWARD PASS ========

        // Zero gradients
        CUDA_CHECK(cudaMemset(s.d_grads, 0, wo.total_floats * sizeof(float)));
        // Zero KV gradient accumulators (skip if NULL — v1 approximate Wk/Wv grads)
        for (int l = 0; l < L; l++) {
            if (s.d_dkv_keys[l])
                CUDA_CHECK(cudaMemset(s.d_dkv_keys[l], 0, (long long)trie.total_nodes * D * sizeof(float)));
            if (s.d_dkv_values[l])
                CUDA_CHECK(cudaMemset(s.d_dkv_values[l], 0, (long long)trie.total_nodes * D * sizeof(float)));
        }

        // 1. Embedding gather — need token ids, not node ids
        int* h_token_ids = (int*)malloc(N * sizeof(int));
        for (int i = 0; i < N; i++) {
            int nid = h_node_ids[i];
            int tok = trie.tokens[nid];
            if (tok < 0 || tok >= cfg.vocab_size) {
                fprintf(stderr, "Bad token at depth %d, chunk node %d: node_id=%d token=%d\n",
                        depth, i, nid, tok);
                exit(1);
            }
            h_token_ids[i] = tok;
        }
        int* d_token_ids;
        CUDA_CHECK(cudaMalloc(&d_token_ids, N * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_token_ids, h_token_ids, N * sizeof(int), cudaMemcpyHostToDevice));

        // Redo embedding gather with actual token ids
        cuda_embedding_gather(s.d_weights + wo.token_emb, d_token_ids, s.d_x, N, D);

        // Save x for backward (embedding backward needs to scatter to token indices)
        // x_input = x (before any modification)

        // Per-layer forward
        for (int l = 0; l < L; l++) {
            float* W_qw = s.d_weights + wo.wq_w[l];
            float* W_qb = s.d_weights + wo.wq_b[l];
            float* W_kw = s.d_weights + wo.wk_w[l];
            float* W_kb = s.d_weights + wo.wk_b[l];
            float* W_vw = s.d_weights + wo.wv_w[l];
            float* W_vb = s.d_weights + wo.wv_b[l];
            float* W_ow = s.d_weights + wo.wo_w[l];
            float* W_ob = s.d_weights + wo.wo_b[l];
            float* G1   = s.d_weights + wo.ln1_gamma[l];
            float* B1   = s.d_weights + wo.ln1_beta[l];
            float* W_1w = s.d_weights + wo.l1_w[l];
            float* W_1b = s.d_weights + wo.l1_b[l];
            float* W_2w = s.d_weights + wo.l2_w[l];
            float* W_2b = s.d_weights + wo.l2_b[l];
            float* G2   = s.d_weights + wo.ln2_gamma[l];
            float* B2   = s.d_weights + wo.ln2_beta[l];

            // Save residual input for backward
            CUDA_CHECK(cudaMemcpy(s.saved_x_res1[l], s.d_x, (long long)N * D * sizeof(float), cudaMemcpyDeviceToDevice));

            // LayerNorm1 → saved per-layer
            cuda_layer_norm_forward(s.d_x, s.d_ln_out, s.saved_ln1_norm[l], s.saved_ln1_std_inv[l], G1, B1, N, D);
            CUDA_CHECK(cudaMemcpy(s.saved_ln1_out[l], s.d_ln_out, (long long)N * D * sizeof(float), cudaMemcpyDeviceToDevice));

            // Q/K/V projections
            float alpha = 1.0f, beta_zero = 0.0f;
            CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                      D, N, D, &alpha, W_qw, D, s.d_ln_out, D, &beta_zero, s.d_q, D));
            cuda_bias_add(s.d_q, W_qb, N, D);
            CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                      D, N, D, &alpha, W_kw, D, s.d_ln_out, D, &beta_zero, s.d_k, D));
            cuda_bias_add(s.d_k, W_kb, N, D);
            CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                      D, N, D, &alpha, W_vw, D, s.d_ln_out, D, &beta_zero, s.d_v, D));
            cuda_bias_add(s.d_v, W_vb, N, D);

            // RoPE — build expanded positions [N*H]
            int* h_exp_pos = (int*)malloc(N * H * sizeof(int));
            for (int i = 0; i < N; i++)
                for (int h = 0; h < H; h++)
                    h_exp_pos[i * H + h] = h_positions[i];
            int* d_exp_pos;
            CUDA_CHECK(cudaMalloc(&d_exp_pos, N * H * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(d_exp_pos, h_exp_pos, N * H * sizeof(int), cudaMemcpyHostToDevice));
            launch_rope_batched(s.d_q, d_exp_pos, s.d_rope_cos, s.d_rope_sin, N * H, HD);
            launch_rope_batched(s.d_k, d_exp_pos, s.d_rope_cos, s.d_rope_sin, N * H, HD);

            // Store K/V into global KV cache
            if (s.kv_on_host) {
                host_kv_scatter(s.d_k, h_node_ids, s.d_kv_keys[l], N, D);
                host_kv_scatter(s.d_v, h_node_ids, s.d_kv_values[l], N, D);
            } else {
                launch_kv_scatter(s.d_k, s.d_node_ids, s.d_kv_keys[l], N, D);
                launch_kv_scatter(s.d_v, s.d_node_ids, s.d_kv_values[l], N, D);
            }

            // Gather ancestor K/V into packed buffers
            if (s.kv_on_host) {
                host_kv_gather(s.d_kv_keys[l], trie.ancestor_ids, h_anc_offsets,
                                h_kv_offsets, h_kv_lengths, s.d_kv_packed_k,
                                N, H, HD, total_kv);
                host_kv_gather(s.d_kv_values[l], trie.ancestor_ids, h_anc_offsets,
                                h_kv_offsets, h_kv_lengths, s.d_kv_packed_v,
                                N, H, HD, total_kv);
            } else {
                launch_kv_gather(s.d_kv_keys[l], s.d_ancestor_ids, s.d_anc_offsets,
                                  s.d_kv_offsets, s.d_kv_lengths, s.d_kv_packed_k, N, H, HD);
                launch_kv_gather(s.d_kv_values[l], s.d_ancestor_ids, s.d_anc_offsets,
                                  s.d_kv_offsets, s.d_kv_lengths, s.d_kv_packed_v, N, H, HD);
            }

            // Varlen attention
            float scale = 1.0f / sqrtf((float)HD);
            cuda_batched_varlen_attention(
                s.d_q, s.d_kv_packed_k, s.d_kv_packed_v,
                s.d_kv_offsets, s.d_kv_lengths,
                s.d_attn_out_packed, s.saved_attn_weights[l],
                N, H, HD, max_prefix, scale);
            cuda_unpack_batched_attn_output(s.d_attn_out_packed, s.d_attn_out, N, H, HD);

            // Save attn_out for WO backward
            CUDA_CHECK(cudaMemcpy(s.saved_attn_out[l], s.d_attn_out, (long long)N * D * sizeof(float), cudaMemcpyDeviceToDevice));

            // WO projection + residual 1
            CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                      D, N, D, &alpha, W_ow, D, s.d_attn_out, D, &beta_zero, s.d_ff_out, D));
            cuda_bias_add(s.d_ff_out, W_ob, N, D);
            CUDA_CHECK(cudaMemcpy(s.d_x, s.saved_x_res1[l], (long long)N * D * sizeof(float), cudaMemcpyDeviceToDevice));
            launch_elem_add(s.d_x, s.d_ff_out, N * D);

            // Save residual 2 input
            CUDA_CHECK(cudaMemcpy(s.saved_x_res2[l], s.d_x, (long long)N * D * sizeof(float), cudaMemcpyDeviceToDevice));

            // LayerNorm2 → saved per-layer
            cuda_layer_norm_forward(s.d_x, s.d_ln_out, s.saved_ln2_norm[l], s.saved_ln2_std_inv[l], G2, B2, N, D);
            CUDA_CHECK(cudaMemcpy(s.saved_ln2_out[l], s.d_ln_out, (long long)N * D * sizeof(float), cudaMemcpyDeviceToDevice));

            // FFN: l1 (fused bias+relu) → l2 + bias + residual 2
            CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                      F, N, D, &alpha, W_1w, F, s.d_ln_out, D, &beta_zero, s.d_ff_h, F));
            cuda_fused_bias_relu(s.d_ff_h, W_1b, s.d_ff_h, s.saved_ff_mask[l], N, F);
            // Save post-ReLU hidden for backward
            CUDA_CHECK(cudaMemcpy(s.saved_ff_h[l], s.d_ff_h, (long long)N * F * sizeof(float), cudaMemcpyDeviceToDevice));

            CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                      D, N, F, &alpha, W_2w, D, s.d_ff_h, F, &beta_zero, s.d_ff_out, D));
            cuda_bias_add(s.d_ff_out, W_2b, N, D);

            CUDA_CHECK(cudaMemcpy(s.d_x, s.saved_x_res2[l], (long long)N * D * sizeof(float), cudaMemcpyDeviceToDevice));
            launch_elem_add(s.d_x, s.d_ff_out, N * D);

            CUDA_CHECK(cudaFree(d_exp_pos));
            free(h_exp_pos);
        }

        // Final norm + output projection
        float* G_fn = s.d_weights + wo.final_gamma;
        float* B_fn = s.d_weights + wo.final_beta;
        float* W_out = s.d_weights + wo.out_w;
        float* B_out = s.d_weights + wo.out_b;

        cuda_layer_norm_forward(s.d_x, s.d_ln_out, s.saved_final_norm, s.saved_final_std_inv, G_fn, B_fn, N, D);
        CUDA_CHECK(cudaMemcpy(s.saved_final_out, s.d_ln_out, (long long)N * D * sizeof(float), cudaMemcpyDeviceToDevice));

        float alpha = 1.0f, beta_zero = 0.0f;
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                  V, N, D, &alpha, W_out, V, s.d_ln_out, D, &beta_zero, s.d_logits, V));
        cuda_bias_add(s.d_logits, B_out, N, V);

        // Loss
        launch_agpt_loss(s.d_logits, s.d_node_ids,
                          s.d_counts_offset, s.d_counts_tok, s.d_counts_val,
                          s.d_d_logits, s.d_loss, N, V);

        // Sum loss on CPU
        float* h_loss = (float*)malloc(N * sizeof(float));
        CUDA_CHECK(cudaMemcpy(h_loss, s.d_loss, N * sizeof(float), cudaMemcpyDeviceToHost));
        int depth_trained = 0;
        for (int i = 0; i < N; i++) {
            if (h_loss[i] > 0.0f) { total_loss += h_loss[i]; depth_trained++; }
        }
        nodes_trained += depth_trained;
        free(h_loss);

        // ======== BACKWARD PASS ========
        // d_logits already has the loss gradient from agpt_loss_kernel.
        // Scale gradients by 1/N for this chunk's update.
        float grad_scale = 1.0f / (float)N;
        float neg_grad_scale = -grad_scale;

        // Output projection backward: d_logits[N,V] → d_final_out[N,D]
        // d_final_out = d_logits × W_out^T
        float* dG_out = s.d_grads + wo.out_w;
        float* dB_out = s.d_grads + wo.out_b;
        // dx = d_logits[N,V] × W_out^T[V,D]
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                  D, N, V, &alpha, W_out, V, s.d_d_logits, V, &beta_zero, s.d_dx, D));
        // dW_out += final_out^T[D,N] × d_logits[N,V] (scaled)
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                                  V, D, N, &grad_scale, s.d_d_logits, V, s.saved_final_out, D, &alpha, dG_out, V));
        // db_out += sum over rows of d_logits (use gemv: ones^T × d_logits)
        // Simpler: use cublasSgemv with a ones vector, or just leave as is for now
        // We can accumulate bias grad with a simple kernel
        // For now, skip bias grad accumulation (TODO)

        // Final LayerNorm backward
        float* dG_fn = s.d_grads + wo.final_gamma;
        float* dB_fn = s.d_grads + wo.final_beta;
        cuda_layer_norm_backward(s.d_dx, s.saved_final_norm, s.saved_final_std_inv,
                                  G_fn, s.d_dx, dG_fn, dB_fn, N, D);
        // Note: cuda_layer_norm_backward writes dx in-place, and accumulates dgamma/dbeta

        // Per-layer backward (reverse order)
        for (int l = L - 1; l >= 0; l--) {
            float* W_qw = s.d_weights + wo.wq_w[l];
            float* W_kw = s.d_weights + wo.wk_w[l];
            float* W_vw = s.d_weights + wo.wv_w[l];
            float* W_ow = s.d_weights + wo.wo_w[l];
            float* W_1w = s.d_weights + wo.l1_w[l];
            float* W_2w = s.d_weights + wo.l2_w[l];
            float* G1   = s.d_weights + wo.ln1_gamma[l];
            float* G2   = s.d_weights + wo.ln2_gamma[l];

            float* dW_qw = s.d_grads + wo.wq_w[l]; float* dW_qb = s.d_grads + wo.wq_b[l];
            float* dW_kw = s.d_grads + wo.wk_w[l]; float* dW_kb = s.d_grads + wo.wk_b[l];
            float* dW_vw = s.d_grads + wo.wv_w[l]; float* dW_vb = s.d_grads + wo.wv_b[l];
            float* dW_ow = s.d_grads + wo.wo_w[l]; float* dW_ob = s.d_grads + wo.wo_b[l];
            float* dG1   = s.d_grads + wo.ln1_gamma[l]; float* dB1 = s.d_grads + wo.ln1_beta[l];
            float* dW_1w = s.d_grads + wo.l1_w[l]; float* dW_1b = s.d_grads + wo.l1_b[l];
            float* dW_2w = s.d_grads + wo.l2_w[l]; float* dW_2b = s.d_grads + wo.l2_b[l];
            float* dG2   = s.d_grads + wo.ln2_gamma[l]; float* dB2 = s.d_grads + wo.ln2_beta[l];

            // d_x is the gradient flowing back. It splits at residual 2.
            // d_ff_out = d_x (one branch)
            // d_x_res2 = d_x (skip branch, accumulated later)
            CUDA_CHECK(cudaMemcpy(s.d_d_ff_out, s.d_dx, (long long)N * D * sizeof(float), cudaMemcpyDeviceToDevice));

            // FFN L2 backward: d_ff_out[N,D] → d_ff_h[N,F]
            CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                      F, N, D, &alpha, W_2w, D, s.d_d_ff_out, D, &beta_zero, s.d_d_ff_h, F));
            // dW_2 += ff_h^T × d_ff_out (scaled)
            CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                                      D, F, N, &grad_scale, s.d_d_ff_out, D, s.saved_ff_h[l], F, &alpha, dW_2w, D));

            // ReLU backward: d_ff_h *= saved_ff_mask
            cuda_relu_backward(s.d_d_ff_h, s.saved_ff_mask[l], s.d_d_ff_h, N * F);

            // FFN L1 backward: d_ff_h[N,F] → d_ln2_out[N,D]
            CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                      D, N, F, &alpha, W_1w, F, s.d_d_ff_h, F, &beta_zero, s.d_d_ln_out, D));
            // dW_1 += ln2_out^T × d_ff_h (scaled)
            CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                                      F, D, N, &grad_scale, s.d_d_ff_h, F, s.saved_ln2_out[l], D, &alpha, dW_1w, F));

            // LN2 backward
            cuda_layer_norm_backward(s.d_d_ln_out, s.saved_ln2_norm[l], s.saved_ln2_std_inv[l],
                                      G2, s.d_d_ln_out, dG2, dB2, N, D);

            // Add residual 2 skip: d_x = d_ln2_backward + d_x (from skip)
            launch_elem_add(s.d_dx, s.d_d_ln_out, N * D);

            // Now d_dx flows to attention block backward
            // WO backward: d_dx → d_attn_out[N,D]
            CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                      D, N, D, &alpha, W_ow, D, s.d_dx, D, &beta_zero, s.d_d_attn_out, D));
            // dW_o += attn_out^T × d_dx (scaled)
            CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                                      D, D, N, &grad_scale, s.d_dx, D, s.saved_attn_out[l], D, &alpha, dW_ow, D));

            // Attention backward using varlen attention backward kernel
            // Need to re-gather K/V for this layer
            if (s.kv_on_host) {
                host_kv_gather(s.d_kv_keys[l], trie.ancestor_ids, h_anc_offsets,
                                h_kv_offsets, h_kv_lengths, s.d_kv_packed_k,
                                N, H, HD, total_kv);
                host_kv_gather(s.d_kv_values[l], trie.ancestor_ids, h_anc_offsets,
                                h_kv_offsets, h_kv_lengths, s.d_kv_packed_v,
                                N, H, HD, total_kv);
            } else {
                launch_kv_gather(s.d_kv_keys[l], s.d_ancestor_ids, s.d_anc_offsets,
                                  s.d_kv_offsets, s.d_kv_lengths, s.d_kv_packed_k, N, H, HD);
                launch_kv_gather(s.d_kv_values[l], s.d_ancestor_ids, s.d_anc_offsets,
                                  s.d_kv_offsets, s.d_kv_lengths, s.d_kv_packed_v, N, H, HD);
            }

            // d_attn_out is [N, D]. The backward kernel expects [N*H, HD] packed format.
            // Same memory layout, just reinterpret.
            float scale = 1.0f / sqrtf((float)HD);

            // Reconstruct Q from saved ln1_out + weight projections + RoPE
            // (Re-project from saved LN1 output rather than storing Q separately)
            CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                      D, N, D, &alpha, s.d_weights + wo.wq_w[l], D,
                                      s.saved_ln1_out[l], D, &beta_zero, s.d_q, D));
            cuda_bias_add(s.d_q, s.d_weights + wo.wq_b[l], N, D);
            // Re-apply RoPE
            int* h_exp_pos = (int*)malloc(N * H * sizeof(int));
            for (int i = 0; i < N; i++)
                for (int h = 0; h < H; h++)
                    h_exp_pos[i * H + h] = h_positions[i];
            int* d_exp_pos;
            CUDA_CHECK(cudaMalloc(&d_exp_pos, N * H * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(d_exp_pos, h_exp_pos, N * H * sizeof(int), cudaMemcpyHostToDevice));
            launch_rope_batched(s.d_q, d_exp_pos, s.d_rope_cos, s.d_rope_sin, N * H, HD);

            cuda_batched_varlen_attention_backward(
                s.d_q, s.d_kv_packed_k, s.d_kv_packed_v,
                s.saved_attn_weights[l], s.d_d_attn_out,
                s.d_kv_offsets, s.d_kv_lengths,
                s.d_dq_packed, s.d_dk_packed, s.d_dv_packed,
                N, H, HD, max_prefix, scale);

            // Inverse RoPE on dQ to get d_q_pre_rope
            launch_rope_batched_inverse(s.d_dq_packed, d_exp_pos, s.d_rope_cos, s.d_rope_sin, N * H, HD);

            // dQ → d_ln1_out via Wq^T: d_ln1_out_q = dQ[N,D] × Wq^T[D,D]
            CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                      D, N, D, &alpha, s.d_weights + wo.wq_w[l], D,
                                      s.d_dq_packed, D, &beta_zero, s.d_d_ln_out, D));
            // dWq += ln1_out^T × dQ (scaled)
            CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                                      D, D, N, &grad_scale, s.d_dq_packed, D,
                                      s.saved_ln1_out[l], D, &alpha, dW_qw, D));

            // Scatter dk/dv back to global KV grad accumulators (skip if NULL)
            if (s.d_dkv_keys[l]) {
                launch_kv_scatter_add(s.d_dk_packed, s.d_ancestor_ids, s.d_anc_offsets,
                                       s.d_kv_offsets, s.d_kv_lengths, s.d_dkv_keys[l], N, H, HD);
                launch_kv_scatter_add(s.d_dv_packed, s.d_ancestor_ids, s.d_anc_offsets,
                                       s.d_kv_offsets, s.d_kv_lengths, s.d_dkv_values[l], N, H, HD);
            }

            // dK for current nodes: extract from global dk accumulator via scatter
            // dK[i] = d_dkv_keys[l][node_ids[i]] (these are the K grads for THIS node's position)
            // Then inverse RoPE and backprop through Wk
            // For simplicity, gather dk for current nodes from the accumulator
            // Actually, the current node's dk contribution is already in d_dk_packed
            // (the self-position in the KV). But d_dk_packed contains ALL positions' grads
            // for ALL nodes. We need just the current node's own position.
            // Simpler: use the d_dkv_keys accumulator. Gather current nodes' entries.

            // Gather current-node dK/dV from accumulators
            // d_dk_self[i] = d_dkv_keys[l][node_ids[i] * D .. +D]
            // This is like the inverse of kv_scatter: use d_node_ids to gather
            // For now, use a simple gather: d_k[i] = accumulator[node_ids[i]]
            // We need a gather kernel (reverse of scatter)

            // Actually, we can just skip per-node K/V gradients for now.
            // The main gradients flow through Q (dWq is already accumulated).
            // K/V gradients for ancestors at PREVIOUS depths would need a multi-depth
            // backward pass. For the initial version, let's just accumulate the
            // projection gradients for the current depth's K and V via the saved ln1_out.
            // dWk += ln1_out^T × dK_for_current_nodes (this is approximate — ignores
            // the fact that K affects attention at FUTURE depths too).

            // For a correct gradient: we'd need to accumulate dK/dV across all depths
            // that attend to each node, then backprop through projections.
            // That's complex. For v1, let's do the simple version: only propagate
            // gradients through Q (which is always for the current depth) and through
            // the FFN/LN paths. This is correct for the FFN/LN/embedding weights
            // and approximately correct for Wq/Wo. Wk/Wv gradients are partial.

            // LN1 backward
            cuda_layer_norm_backward(s.d_d_ln_out, s.saved_ln1_norm[l], s.saved_ln1_std_inv[l],
                                      G1, s.d_d_ln_out, dG1, dB1, N, D);

            // Add residual 1 skip
            launch_elem_add(s.d_dx, s.d_d_ln_out, N * D);

            CUDA_CHECK(cudaFree(d_exp_pos));
            free(h_exp_pos);
        }

        // Embedding backward: scatter_add d_x into d_token_emb
        float* dG_emb = s.d_grads + wo.token_emb;
        cuda_embedding_scatter_add(s.d_dx, d_token_ids, dG_emb, N, D);

        // Adam update
        s.adam_t++;
        cuda_adam_bulk(s.d_weights, s.d_grads, s.d_adam_m, s.d_adam_v,
                        cfg.lr, 0.9f, 0.999f, 1e-8f, s.adam_t, wo.total_floats);

        // Cleanup per-chunk allocations
        CUDA_CHECK(cudaFree(d_token_ids));
        free(h_token_ids);
        free(h_node_ids);
        free(h_positions);
        free(h_kv_offsets);
        free(h_kv_lengths);
        free(h_anc_offsets);

        } // end chunk loop

        CUDA_CHECK(cudaDeviceSynchronize()); // catch any async errors from this depth

        if (depth <= 3 || depth == trie.depth_file_count - 1) {
            printf("  depth %d: %d nodes, %d trained, loss=%.4f\n",
                   depth, N_total, nodes_trained,
                   nodes_trained > 0 ? (total_loss / nodes_trained) : 0.0);
        }
    }

    // Free depth node lists
    for (int d = 0; d < trie.depth_file_count; d++) free(depth_node_lists[d]);
    free(depth_node_lists);

    float mean_loss = nodes_trained > 0 ? (float)(total_loss / nodes_trained) : 0.0f;
    return mean_loss;
}
