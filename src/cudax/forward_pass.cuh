#ifndef AGPT_V2_FORWARD_PASS_CUH
#define AGPT_V2_FORWARD_PASS_CUH

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
};

struct LossTablesV2 {
    const int* d_counts_offset = nullptr;
    const int* d_counts_tok = nullptr;
    const int* d_counts_val = nullptr;
};

#include "forward_stages_v2.cuh"

static inline ForwardPassResult run_forward_prefix_v2(const TrainerConfig& cfg,
                                                      const ModelLayout& layout,
                                                      const ChunkMetadataV2& meta,
                                                      const ChunkDeviceMetadataV2& device_meta,
                                                      const ChunkUploadRuntimeV2& upload,
                                                      const LossTablesV2& loss_tables,
                                                      TrainerRuntimeV2& runtime,
                                                      UnitAncGradRuntimeV2* anc_runtime = nullptr) {
    ForwardPassResult result;
    int T_q = meta.T_q;
    int D = cfg.d_model;
    int F = cfg.d_ff;
    int V = cfg.vocab_size;
    ChunkBufferLayoutV2 buf = make_chunk_buffer_layout_v2(runtime.chunk, T_q, D, F, V);
    cublasHandle_t cublas = runtime.cublas;
    float* d_rope_cos = runtime.d_rope_cos;
    float* d_rope_sin = runtime.d_rope_sin;

    run_forward_embedding_stage_v2(layout, upload, runtime, buf, T_q, D);
    for (int l = 0; l < cfg.n_layers; l++) {
        run_forward_transformer_layer_stage_v2(cfg, layout, meta, device_meta, upload, runtime, buf, cublas, d_rope_cos, d_rope_sin, anc_runtime, l);
    }

    run_forward_output_stage_v2(cfg, layout, meta, loss_tables, upload, runtime, buf, cublas);

    AGPT_V2_CUDA_CHECK(cudaDeviceSynchronize());
    float* h_loss = (float*)std::malloc((size_t)T_q * sizeof(float));
    AGPT_V2_CUDA_CHECK(cudaMemcpy(h_loss, buf.output.loss, (size_t)T_q * sizeof(float), cudaMemcpyDeviceToHost));
    double loss_sum = 0.0;
    int trained = 0;
    for (int i = 0; i < T_q; i++) {
        if (h_loss[i] > 0.0f) {
            loss_sum += h_loss[i];
            trained++;
        }
    }
    std::free(h_loss);
    result.trained_queries = trained;
    result.mean_loss = trained > 0 ? (float)(loss_sum / trained) : 0.0f;
    return result;
}

}  // namespace agpt_v2

#endif
