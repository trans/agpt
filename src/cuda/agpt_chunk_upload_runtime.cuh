#ifndef AGPT_CHUNK_UPLOAD_RUNTIME_CUH
#define AGPT_CHUNK_UPLOAD_RUNTIME_CUH

struct DeviceIntScratch {
    int* ptr = nullptr;
    int cap = 0;
};

struct ChunkDeviceScratchPool {
    DeviceIntScratch anc_ids;
    DeviceIntScratch anc_offsets;
    DeviceIntScratch anc_lengths;
    DeviceIntScratch own_lengths;
    DeviceIntScratch read_pos_flat;
    DeviceIntScratch query_depth;
    DeviceIntScratch query_d_split;
};

struct ChunkUploadRuntime {
    int H = 0;
    int* d_radix_ids = nullptr;
    int* d_query_to_node = nullptr;
    int* d_query_offsets = nullptr;
    int* d_kv_offsets = nullptr;
    int* d_kv_lengths = nullptr;
    int* d_token_ids = nullptr;
    int* d_rope_positions = nullptr;
    int* d_char_pos = nullptr;
    ChunkDeviceScratchPool scratch;
};

struct ChunkDeviceMetadata {
    int* d_anc_ids = nullptr;
    int* d_anc_offsets = nullptr;
    int* d_anc_lengths = nullptr;
    int* d_own_lengths = nullptr;
    int* d_read_pos_flat = nullptr;
    int* d_query_depth = nullptr;
    int* d_query_d_split = nullptr;
};

static void ensure_device_int_scratch(DeviceIntScratch& scratch, int needed) {
    if (needed > scratch.cap) {
        if (scratch.ptr) cudaFree(scratch.ptr);
        CUDA_CHECK(cudaMalloc(&scratch.ptr, (needed > 0 ? needed : 1) * sizeof(int)));
        scratch.cap = needed;
    }
}

static void free_device_int_scratch(DeviceIntScratch& scratch) {
    if (scratch.ptr) cudaFree(scratch.ptr);
    scratch.ptr = nullptr;
    scratch.cap = 0;
}

static void free_chunk_device_scratch_pool(ChunkDeviceScratchPool& scratch) {
    free_device_int_scratch(scratch.anc_ids);
    free_device_int_scratch(scratch.anc_offsets);
    free_device_int_scratch(scratch.anc_lengths);
    free_device_int_scratch(scratch.own_lengths);
    free_device_int_scratch(scratch.read_pos_flat);
    free_device_int_scratch(scratch.query_depth);
    free_device_int_scratch(scratch.query_d_split);
}

static void init_chunk_upload_runtime(ChunkUploadRuntime& runtime, int N_cap, int T_q_cap, int H) {
    runtime.H = H;
    CUDA_CHECK(cudaMalloc(&runtime.d_radix_ids,      N_cap * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&runtime.d_query_to_node,  T_q_cap * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&runtime.d_query_offsets,  (N_cap + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&runtime.d_kv_offsets,     (N_cap + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&runtime.d_kv_lengths,     N_cap * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&runtime.d_token_ids,      T_q_cap * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&runtime.d_rope_positions, T_q_cap * H * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&runtime.d_char_pos,       T_q_cap * sizeof(int)));
}

static void free_chunk_upload_runtime(ChunkUploadRuntime& runtime) {
    free_chunk_device_scratch_pool(runtime.scratch);
    cudaFree(runtime.d_radix_ids);
    cudaFree(runtime.d_query_to_node);
    cudaFree(runtime.d_query_offsets);
    cudaFree(runtime.d_kv_offsets);
    cudaFree(runtime.d_kv_lengths);
    cudaFree(runtime.d_token_ids);
    cudaFree(runtime.d_rope_positions);
    cudaFree(runtime.d_char_pos);
    runtime = ChunkUploadRuntime{};
}

static ChunkDeviceMetadata upload_chunk_metadata_to_device(const ChunkMetadata& meta,
                                                           ChunkUploadRuntime& runtime) {
    ChunkDeviceScratchPool& scratch = runtime.scratch;
    ensure_device_int_scratch(scratch.anc_ids, meta.T_anc);
    ensure_device_int_scratch(scratch.anc_offsets, meta.N + 1);
    ensure_device_int_scratch(scratch.anc_lengths, meta.N);
    ensure_device_int_scratch(scratch.own_lengths, meta.N);
    ensure_device_int_scratch(scratch.read_pos_flat, meta.T_anc);
    ensure_device_int_scratch(scratch.query_depth, meta.T_q);
    ensure_device_int_scratch(scratch.query_d_split, meta.T_q);

    if (meta.T_anc > 0) {
        CUDA_CHECK(cudaMemcpy(scratch.anc_ids.ptr, meta.h_anc_ids, meta.T_anc * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(scratch.read_pos_flat.ptr, meta.h_read_pos_flat, meta.T_anc * sizeof(int), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(scratch.anc_offsets.ptr, meta.h_anc_offsets, (meta.N + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(scratch.anc_lengths.ptr, meta.h_anc_lengths, meta.N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(scratch.own_lengths.ptr, meta.h_own_lengths, meta.N * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(runtime.d_radix_ids,      meta.h_radix_ids,      meta.N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(runtime.d_query_offsets,  meta.h_query_offsets,  (meta.N + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(runtime.d_kv_offsets,     meta.h_kv_offsets,     (meta.N + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(runtime.d_kv_lengths,     meta.h_kv_lengths,     meta.N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(runtime.d_query_to_node,  meta.h_query_to_node,  meta.T_q * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(runtime.d_token_ids,      meta.h_token_ids,      meta.T_q * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(runtime.d_rope_positions, meta.h_rope_positions, meta.T_q * runtime.H * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(runtime.d_char_pos,       meta.h_char_pos,       meta.T_q * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(scratch.query_depth.ptr,   meta.h_query_depth,   meta.T_q * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(scratch.query_d_split.ptr, meta.h_query_d_split, meta.T_q * sizeof(int), cudaMemcpyHostToDevice));

    ChunkDeviceMetadata out;
    out.d_anc_ids = scratch.anc_ids.ptr;
    out.d_anc_offsets = scratch.anc_offsets.ptr;
    out.d_anc_lengths = scratch.anc_lengths.ptr;
    out.d_own_lengths = scratch.own_lengths.ptr;
    out.d_read_pos_flat = scratch.read_pos_flat.ptr;
    out.d_query_depth = scratch.query_depth.ptr;
    out.d_query_d_split = scratch.query_d_split.ptr;
    return out;
}

#endif
