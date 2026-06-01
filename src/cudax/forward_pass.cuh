#ifndef AGPT_V2_FORWARD_PASS_CUH
#define AGPT_V2_FORWARD_PASS_CUH

#include <cmath>

#include "../common/diag_tensor_dump.h"

#include "checkpoint_io_v2.cuh"
#include "chunk_metadata_v2.cuh"
#include "chunk_upload_v2.cuh"
#include "buffer_layout_v2.cuh"
#include "kernels_v2.cuh"
#include "runtime_objects.cuh"

namespace agpt_v2 {

struct ForwardPassResult {
    bool ok = true;
    const char* message = "forward full-depth scored chunk executed";
    float mean_loss = 0.0f;
    int trained_queries = 0;
    double trained_events = 0.0;
};

struct LossTablesV2 {
    const int* d_counts_offset = nullptr;
    const int* d_counts_len = nullptr;
    const int* d_counts_tok = nullptr;
    const int* d_counts_val = nullptr;
};

struct ForwardDiagDumpConfigV2 {
    const char* tensor_dir = nullptr;
    int epoch = 0;
    int root_id = 0;
    int chunk_idx = 0;
    bool active = false;
};

#include "forward_stages_v2.cuh"

static inline ForwardPassResult run_forward_prefix_v2(const TrainerConfig& cfg,
                                                      const ModelLayout& layout,
                                                      const ChunkMetadataV2& meta,
                                                      const ChunkDeviceMetadataV2& device_meta,
                                                      const ChunkUploadRuntimeV2& upload,
                                                      const LossTablesV2& loss_tables,
                                                      TrainerRuntimeV2& runtime,
                                                      UnitAncGradRuntimeV2* anc_runtime = nullptr,
                                                      const ForwardDiagDumpConfigV2* diag = nullptr) {
    ForwardPassResult result;
    int T_q = meta.T_q;
    int D = cfg.d_model;
    int F = cfg.d_ff;
    int V = cfg.vocab_size;
    ChunkBufferLayoutV2 buf = make_chunk_buffer_layout_v2(runtime.chunk, T_q, D, F, V);
    cublasHandle_t cublas = runtime.cublas;
    float* d_rope_cos = runtime.d_rope_cos;
    float* d_rope_sin = runtime.d_rope_sin;

    if (diag && diag->active) {
        agpt_diag::emit_tensor_int_bin(diag->tensor_dir, diag->epoch, diag->root_id, diag->chunk_idx, 0,
                                       "fwd_token_ids", upload.d_token_ids, T_q);
        agpt_diag::emit_tensor_int_bin(diag->tensor_dir, diag->epoch, diag->root_id, diag->chunk_idx, 0,
                                       "fwd_query_offsets", upload.d_query_offsets, meta.N + 1);
        agpt_diag::emit_tensor_int_bin(diag->tensor_dir, diag->epoch, diag->root_id, diag->chunk_idx, 0,
                                       "fwd_kv_offsets", upload.d_kv_offsets, meta.N + 1);
        agpt_diag::emit_tensor_int_bin(diag->tensor_dir, diag->epoch, diag->root_id, diag->chunk_idx, 0,
                                       "fwd_kv_lengths", upload.d_kv_lengths, meta.N);
        agpt_diag::emit_tensor_int_bin(diag->tensor_dir, diag->epoch, diag->root_id, diag->chunk_idx, 0,
                                       "fwd_query_to_node", upload.d_query_to_node, T_q);
        agpt_diag::emit_tensor_int_bin(diag->tensor_dir, diag->epoch, diag->root_id, diag->chunk_idx, 0,
                                       "fwd_radix_ids", upload.d_radix_ids, meta.N);
    }

    run_forward_embedding_stage_v2(layout, upload, runtime, buf, T_q, D, diag);
    for (int l = 0; l < cfg.n_layers; l++) {
        run_forward_transformer_layer_stage_v2(cfg, layout, meta, device_meta, upload, runtime, buf, cublas, d_rope_cos, d_rope_sin, anc_runtime, l, diag);
    }

    run_forward_output_stage_v2(cfg, layout, meta, device_meta, loss_tables, upload, runtime, buf, cublas);

    AGPT_V2_CUDA_CHECK(cudaDeviceSynchronize());
    float* h_loss = (float*)std::malloc((size_t)T_q * sizeof(float));
    AGPT_V2_CUDA_CHECK(cudaMemcpy(h_loss, buf.output.loss, (size_t)T_q * sizeof(float), cudaMemcpyDeviceToHost));
    double loss_sum = 0.0;
    double event_sum = 0.0;
    int trained = 0;
    for (int i = 0; i < T_q; i++) {
        if (!std::isfinite(h_loss[i])) {
            result.ok = false;
            result.message = "non-finite forward loss";
        } else if (h_loss[i] > 0.0f) {
            loss_sum += h_loss[i];
            event_sum += (double)meta.h_query_weights[i];
            trained++;
        }
    }
    std::free(h_loss);
    result.trained_queries = trained;
    result.trained_events = event_sum;
    result.mean_loss = event_sum > 0.0 ? (float)(loss_sum / event_sum) : 0.0f;
    return result;
}

}  // namespace agpt_v2

#endif
