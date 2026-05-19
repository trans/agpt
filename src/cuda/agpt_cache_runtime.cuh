#ifndef AGPT_CACHE_RUNTIME_CUH
#define AGPT_CACHE_RUNTIME_CUH

struct CacheRuntime {
    int L_layers = 0;
    long long kv_bytes = 0;
    int* d_compact_slot = nullptr;
    int* d_real_pos_of_char = nullptr;
    __nv_bfloat16** d_kv_keys = nullptr;
    __nv_bfloat16** d_kv_values = nullptr;
};

static void init_cache_runtime(CacheRuntime& runtime,
                               int L_layers,
                               long long kv_bytes,
                               const int* compact_slot,
                               long long n_compact_chars,
                               const int* real_pos_of_char,
                               long long total_edge_chars) {
    runtime.L_layers = L_layers;
    runtime.kv_bytes = kv_bytes;
    CUDA_CHECK(cudaMalloc(&runtime.d_compact_slot, n_compact_chars * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(runtime.d_compact_slot, compact_slot,
                          n_compact_chars * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&runtime.d_real_pos_of_char, total_edge_chars * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(runtime.d_real_pos_of_char, real_pos_of_char,
                          total_edge_chars * sizeof(int), cudaMemcpyHostToDevice));
    runtime.d_kv_keys = (__nv_bfloat16**)malloc(L_layers * sizeof(__nv_bfloat16*));
    runtime.d_kv_values = (__nv_bfloat16**)malloc(L_layers * sizeof(__nv_bfloat16*));
    for (int l = 0; l < L_layers; l++) {
        CUDA_CHECK(cudaMallocManaged(&runtime.d_kv_keys[l],   kv_bytes));
        CUDA_CHECK(cudaMallocManaged(&runtime.d_kv_values[l], kv_bytes));
    }
}

static void zero_cache_runtime(const CacheRuntime& runtime) {
    for (int l = 0; l < runtime.L_layers; l++) {
        CUDA_CHECK(cudaMemset(runtime.d_kv_keys[l],   0, runtime.kv_bytes));
        CUDA_CHECK(cudaMemset(runtime.d_kv_values[l], 0, runtime.kv_bytes));
    }
}

static void free_cache_runtime(CacheRuntime& runtime) {
    if (runtime.d_kv_keys) {
        for (int l = 0; l < runtime.L_layers; l++) cudaFree(runtime.d_kv_keys[l]);
        free(runtime.d_kv_keys);
    }
    if (runtime.d_kv_values) {
        for (int l = 0; l < runtime.L_layers; l++) cudaFree(runtime.d_kv_values[l]);
        free(runtime.d_kv_values);
    }
    if (runtime.d_compact_slot) cudaFree(runtime.d_compact_slot);
    if (runtime.d_real_pos_of_char) cudaFree(runtime.d_real_pos_of_char);
    runtime = CacheRuntime{};
}

static void scatter_layer_kv_to_cache(const CacheRuntime& runtime,
                                      int layer,
                                      const float* d_k,
                                      const float* d_v,
                                      const int* d_char_pos,
                                      int T_q,
                                      int D) {
    launch_kv_scatter_compact_bf16(d_k, d_char_pos, runtime.d_compact_slot, runtime.d_kv_keys[layer], T_q, D);
    launch_kv_scatter_compact_bf16(d_v, d_char_pos, runtime.d_compact_slot, runtime.d_kv_values[layer], T_q, D);
}

static void gather_layer_packed_kv(const CacheRuntime& runtime,
                                   int layer,
                                   const ChunkDeviceMetadata& meta,
                                   const int* d_query_offsets,
                                   const int* d_kv_offsets,
                                   const float* d_k,
                                   const float* d_v,
                                   float* d_kv_pack_k,
                                   float* d_kv_pack_v,
                                   const float* d_rope_cos,
                                   const float* d_rope_sin,
                                   int N,
                                   int H,
                                   int HD,
                                   bool use_delta_rope) {
    if (use_delta_rope) {
        launch_kv_gather_k_anc_delta_rope(runtime.d_kv_keys[layer], meta.d_anc_ids, meta.d_anc_offsets,
                                          d_kv_offsets, meta.d_anc_lengths, runtime.d_compact_slot,
                                          meta.d_read_pos_flat, runtime.d_real_pos_of_char,
                                          d_rope_cos, d_rope_sin,
                                          d_kv_pack_k, N, H, HD);
    } else {
        launch_kv_gather_anc_compact_bf16(runtime.d_kv_keys[layer], meta.d_anc_ids, meta.d_anc_offsets,
                                          d_kv_offsets, meta.d_anc_lengths, runtime.d_compact_slot,
                                          d_kv_pack_k, N, H, HD);
    }
    launch_kv_gather_anc_compact_bf16(runtime.d_kv_values[layer], meta.d_anc_ids, meta.d_anc_offsets,
                                      d_kv_offsets, meta.d_anc_lengths, runtime.d_compact_slot,
                                      d_kv_pack_v, N, H, HD);
    launch_kv_copy_own_edge(d_k, d_query_offsets, d_kv_offsets,
                            meta.d_anc_lengths, meta.d_own_lengths,
                            d_kv_pack_k, N, H, HD);
    launch_kv_copy_own_edge(d_v, d_query_offsets, d_kv_offsets,
                            meta.d_anc_lengths, meta.d_own_lengths,
                            d_kv_pack_v, N, H, HD);
}

#endif
