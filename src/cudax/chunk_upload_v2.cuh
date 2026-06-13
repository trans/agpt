#ifndef AGPT_V2_CHUNK_UPLOAD_V2_CUH
#define AGPT_V2_CHUNK_UPLOAD_V2_CUH

#include <cstdlib>

#include "chunk_metadata_v2.cuh"
#include "cuda_support.cuh"

namespace agpt_v2 {

struct DeviceIntScratchV2 {
    int* ptr = nullptr;
    int cap = 0;
};

struct ChunkDeviceScratchPoolV2 {
    DeviceIntScratchV2 anc_ids;
    DeviceIntScratchV2 anc_offsets;
    DeviceIntScratchV2 anc_lengths;
    DeviceIntScratchV2 own_lengths;
    DeviceIntScratchV2 read_pos_flat;
    DeviceIntScratchV2 query_depth;
    DeviceIntScratchV2 target_counts_offset;
    DeviceIntScratchV2 target_counts_len;
    DeviceIntScratchV2 target_counts_tok;
    DeviceIntScratchV2 target_counts_val;
};

struct ChunkUploadRuntimeV2 {
    int n_heads = 0;
    int* d_radix_ids = nullptr;
    int* d_query_to_node = nullptr;
    int* d_query_offsets = nullptr;
    int* d_kv_offsets = nullptr;
    int* d_kv_lengths = nullptr;
    int* d_token_ids = nullptr;
    float* d_query_weights = nullptr;
    float* d_query_grad_scales = nullptr;
    int* d_rope_positions = nullptr;
    int* d_char_pos = nullptr;
    ChunkDeviceScratchPoolV2 scratch;
};

struct ChunkDeviceMetadataV2 {
    int* d_anc_ids = nullptr;
    int* d_anc_offsets = nullptr;
    int* d_anc_lengths = nullptr;
    int* d_own_lengths = nullptr;
    int* d_read_pos_flat = nullptr;
    int* d_query_depth = nullptr;
    float* d_query_weights = nullptr;
    float* d_query_grad_scales = nullptr;
    int* d_target_counts_offset = nullptr;
    int* d_target_counts_len = nullptr;
    int* d_target_counts_tok = nullptr;
    int* d_target_counts_val = nullptr;
};

static inline void ensure_device_int_scratch_v2(DeviceIntScratchV2& scratch, int needed) {
    if (needed > scratch.cap) {
        if (scratch.ptr) cudaFree(scratch.ptr);
        AGPT_V2_CUDA_CHECK(cudaMalloc(&scratch.ptr, (needed > 0 ? needed : 1) * sizeof(int)));
        scratch.cap = needed;
    }
}

static inline void free_device_int_scratch_v2(DeviceIntScratchV2& scratch) {
    if (scratch.ptr) cudaFree(scratch.ptr);
    scratch.ptr = nullptr;
    scratch.cap = 0;
}

static inline void free_chunk_device_scratch_pool_v2(ChunkDeviceScratchPoolV2& scratch) {
    free_device_int_scratch_v2(scratch.anc_ids);
    free_device_int_scratch_v2(scratch.anc_offsets);
    free_device_int_scratch_v2(scratch.anc_lengths);
    free_device_int_scratch_v2(scratch.own_lengths);
    free_device_int_scratch_v2(scratch.read_pos_flat);
    free_device_int_scratch_v2(scratch.query_depth);
    free_device_int_scratch_v2(scratch.target_counts_offset);
    free_device_int_scratch_v2(scratch.target_counts_len);
    free_device_int_scratch_v2(scratch.target_counts_tok);
    free_device_int_scratch_v2(scratch.target_counts_val);
}

static inline void init_chunk_upload_runtime_v2(ChunkUploadRuntimeV2& runtime,
                                                int node_cap,
                                                int query_cap,
                                                int n_heads) {
    runtime.n_heads = n_heads;
    AGPT_V2_CUDA_CHECK(cudaMalloc(&runtime.d_radix_ids, node_cap * sizeof(int)));
    AGPT_V2_CUDA_CHECK(cudaMalloc(&runtime.d_query_to_node, query_cap * sizeof(int)));
    AGPT_V2_CUDA_CHECK(cudaMalloc(&runtime.d_query_offsets, (node_cap + 1) * sizeof(int)));
    AGPT_V2_CUDA_CHECK(cudaMalloc(&runtime.d_kv_offsets, (node_cap + 1) * sizeof(int)));
    AGPT_V2_CUDA_CHECK(cudaMalloc(&runtime.d_kv_lengths, node_cap * sizeof(int)));
    AGPT_V2_CUDA_CHECK(cudaMalloc(&runtime.d_token_ids, query_cap * sizeof(int)));
    AGPT_V2_CUDA_CHECK(cudaMalloc(&runtime.d_query_weights, query_cap * sizeof(float)));
    AGPT_V2_CUDA_CHECK(cudaMalloc(&runtime.d_query_grad_scales, query_cap * sizeof(float)));
    AGPT_V2_CUDA_CHECK(cudaMalloc(&runtime.d_rope_positions, query_cap * n_heads * sizeof(int)));
    AGPT_V2_CUDA_CHECK(cudaMalloc(&runtime.d_char_pos, query_cap * sizeof(int)));
}

static inline void free_chunk_upload_runtime_v2(ChunkUploadRuntimeV2& runtime) {
    free_chunk_device_scratch_pool_v2(runtime.scratch);
    if (runtime.d_radix_ids) cudaFree(runtime.d_radix_ids);
    if (runtime.d_query_to_node) cudaFree(runtime.d_query_to_node);
    if (runtime.d_query_offsets) cudaFree(runtime.d_query_offsets);
    if (runtime.d_kv_offsets) cudaFree(runtime.d_kv_offsets);
    if (runtime.d_kv_lengths) cudaFree(runtime.d_kv_lengths);
    if (runtime.d_token_ids) cudaFree(runtime.d_token_ids);
    if (runtime.d_query_weights) cudaFree(runtime.d_query_weights);
    if (runtime.d_query_grad_scales) cudaFree(runtime.d_query_grad_scales);
    if (runtime.d_rope_positions) cudaFree(runtime.d_rope_positions);
    if (runtime.d_char_pos) cudaFree(runtime.d_char_pos);
    runtime = ChunkUploadRuntimeV2{};
}

static inline ChunkDeviceMetadataV2 upload_chunk_metadata_v2(const ChunkMetadataV2& meta,
                                                             ChunkUploadRuntimeV2& runtime) {
    ensure_device_int_scratch_v2(runtime.scratch.anc_ids, meta.T_anc);
    ensure_device_int_scratch_v2(runtime.scratch.anc_offsets, meta.N + 1);
    ensure_device_int_scratch_v2(runtime.scratch.anc_lengths, meta.N);
    ensure_device_int_scratch_v2(runtime.scratch.own_lengths, meta.N);
    ensure_device_int_scratch_v2(runtime.scratch.read_pos_flat, meta.T_anc);
    ensure_device_int_scratch_v2(runtime.scratch.query_depth, meta.T_q);
    if (meta.h_target_counts_offset && meta.h_target_counts_len) {
        ensure_device_int_scratch_v2(runtime.scratch.target_counts_offset, meta.N + 1);
        ensure_device_int_scratch_v2(runtime.scratch.target_counts_len, meta.N);
        ensure_device_int_scratch_v2(runtime.scratch.target_counts_tok, meta.target_counts_total > 0 ? meta.target_counts_total : 1);
        ensure_device_int_scratch_v2(runtime.scratch.target_counts_val, meta.target_counts_total > 0 ? meta.target_counts_total : 1);
    }

    if (meta.T_anc > 0) {
        AGPT_V2_CUDA_CHECK(cudaMemcpy(runtime.scratch.anc_ids.ptr, meta.h_anc_ids, meta.T_anc * sizeof(int), cudaMemcpyHostToDevice));
        AGPT_V2_CUDA_CHECK(cudaMemcpy(runtime.scratch.read_pos_flat.ptr, meta.h_read_pos_flat, meta.T_anc * sizeof(int), cudaMemcpyHostToDevice));
    }
    AGPT_V2_CUDA_CHECK(cudaMemcpy(runtime.scratch.anc_offsets.ptr, meta.h_anc_offsets, (meta.N + 1) * sizeof(int), cudaMemcpyHostToDevice));
    AGPT_V2_CUDA_CHECK(cudaMemcpy(runtime.scratch.anc_lengths.ptr, meta.h_anc_lengths, meta.N * sizeof(int), cudaMemcpyHostToDevice));
    AGPT_V2_CUDA_CHECK(cudaMemcpy(runtime.scratch.own_lengths.ptr, meta.h_own_lengths, meta.N * sizeof(int), cudaMemcpyHostToDevice));

    AGPT_V2_CUDA_CHECK(cudaMemcpy(runtime.d_radix_ids, meta.h_radix_ids, meta.N * sizeof(int), cudaMemcpyHostToDevice));
    AGPT_V2_CUDA_CHECK(cudaMemcpy(runtime.d_query_offsets, meta.h_query_offsets, (meta.N + 1) * sizeof(int), cudaMemcpyHostToDevice));
    AGPT_V2_CUDA_CHECK(cudaMemcpy(runtime.d_kv_offsets, meta.h_kv_offsets, (meta.N + 1) * sizeof(int), cudaMemcpyHostToDevice));
    AGPT_V2_CUDA_CHECK(cudaMemcpy(runtime.d_kv_lengths, meta.h_kv_lengths, meta.N * sizeof(int), cudaMemcpyHostToDevice));
    AGPT_V2_CUDA_CHECK(cudaMemcpy(runtime.d_query_to_node, meta.h_query_to_node, meta.T_q * sizeof(int), cudaMemcpyHostToDevice));
    AGPT_V2_CUDA_CHECK(cudaMemcpy(runtime.d_token_ids, meta.h_token_ids, meta.T_q * sizeof(int), cudaMemcpyHostToDevice));
    AGPT_V2_CUDA_CHECK(cudaMemcpy(runtime.d_query_weights, meta.h_query_weights, meta.T_q * sizeof(float), cudaMemcpyHostToDevice));
    AGPT_V2_CUDA_CHECK(cudaMemcpy(runtime.d_query_grad_scales, meta.h_query_grad_scales, meta.T_q * sizeof(float), cudaMemcpyHostToDevice));
    AGPT_V2_CUDA_CHECK(cudaMemcpy(runtime.d_rope_positions, meta.h_rope_positions, (long long)meta.T_q * runtime.n_heads * sizeof(int), cudaMemcpyHostToDevice));
    AGPT_V2_CUDA_CHECK(cudaMemcpy(runtime.d_char_pos, meta.h_char_pos, meta.T_q * sizeof(int), cudaMemcpyHostToDevice));
    AGPT_V2_CUDA_CHECK(cudaMemcpy(runtime.scratch.query_depth.ptr, meta.h_query_depth, meta.T_q * sizeof(int), cudaMemcpyHostToDevice));
    if (meta.h_target_counts_offset && meta.h_target_counts_len) {
        AGPT_V2_CUDA_CHECK(cudaMemcpy(runtime.scratch.target_counts_offset.ptr, meta.h_target_counts_offset, (meta.N + 1) * sizeof(int), cudaMemcpyHostToDevice));
        AGPT_V2_CUDA_CHECK(cudaMemcpy(runtime.scratch.target_counts_len.ptr, meta.h_target_counts_len, meta.N * sizeof(int), cudaMemcpyHostToDevice));
        if (meta.target_counts_total > 0) {
            AGPT_V2_CUDA_CHECK(cudaMemcpy(runtime.scratch.target_counts_tok.ptr, meta.h_target_counts_tok, meta.target_counts_total * sizeof(int), cudaMemcpyHostToDevice));
            AGPT_V2_CUDA_CHECK(cudaMemcpy(runtime.scratch.target_counts_val.ptr, meta.h_target_counts_val, meta.target_counts_total * sizeof(int), cudaMemcpyHostToDevice));
        }
    }

    ChunkDeviceMetadataV2 out;
    out.d_anc_ids = runtime.scratch.anc_ids.ptr;
    out.d_anc_offsets = runtime.scratch.anc_offsets.ptr;
    out.d_anc_lengths = runtime.scratch.anc_lengths.ptr;
    out.d_own_lengths = runtime.scratch.own_lengths.ptr;
    out.d_read_pos_flat = runtime.scratch.read_pos_flat.ptr;
    out.d_query_depth = runtime.scratch.query_depth.ptr;
    out.d_query_weights = runtime.d_query_weights;
    out.d_query_grad_scales = runtime.d_query_grad_scales;
    if (meta.h_target_counts_offset && meta.h_target_counts_len) {
        out.d_target_counts_offset = runtime.scratch.target_counts_offset.ptr;
        out.d_target_counts_len = runtime.scratch.target_counts_len.ptr;
        out.d_target_counts_tok = runtime.scratch.target_counts_tok.ptr;
        out.d_target_counts_val = runtime.scratch.target_counts_val.ptr;
    }
    return out;
}

}  // namespace agpt_v2

#endif
