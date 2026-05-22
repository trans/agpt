#ifndef AGPT_V2_BACKWARD_PASS_CUH
#define AGPT_V2_BACKWARD_PASS_CUH

#include <cmath>
#include <cstdlib>

#include "buffer_layout_v2.cuh"
#include "forward_pass.cuh"

namespace agpt_v2 {

struct BackwardPassResult {
    bool ok = true;
    const char* message = "full-depth backward through transformer + embedding executed";
    float out_w_grad_l2 = 0.0f;
    float final_gamma_grad_l2 = 0.0f;
    float l2_w_grad_l2 = 0.0f;
    float l1_w_grad_l2 = 0.0f;
    float ln2_gamma_grad_l2 = 0.0f;
    float wo_w_grad_l2 = 0.0f;
    float dq_grad_l2 = 0.0f;
    float wq_w_grad_l2 = 0.0f;
    float wk_w_grad_l2 = 0.0f;
    float wv_w_grad_l2 = 0.0f;
    float ln1_gamma_grad_l2 = 0.0f;
    float emb_grad_l2 = 0.0f;
};

#include "backward_stages_v2.cuh"

static inline BackwardPassResult run_backward_output_head_v2(const TrainerConfig& cfg,
                                                             const ModelLayout& layout,
                                                             const ChunkMetadataV2& meta,
                                                             const ChunkDeviceMetadataV2& device_meta,
                                                             const ChunkUploadRuntimeV2& upload,
                                                             const ForwardPassResult& forward,
                                                             TrainerRuntimeV2& runtime,
                                                             UnitAncGradRuntimeV2* anc_runtime = nullptr,
                                                             bool clear_grads = true,
                                                             bool finalize_anc_grad = false) {
    BackwardPassResult result;
    int T_q = meta.T_q;
    int D = cfg.d_model;
    int F = cfg.d_ff;
    int V = cfg.vocab_size;
    ChunkBufferLayoutV2 buf = make_chunk_buffer_layout_v2(runtime.chunk, T_q, D, F, V);

    float grad_scale = (forward.trained_queries > 0) ? 1.0f : 0.0f;

    if (clear_grads) {
        AGPT_V2_CUDA_CHECK(cudaMemset(runtime.d_grads, 0, runtime.contract.weight_and_grad_bytes / 2));
    }

    cublasHandle_t cublas = runtime.cublas;
    float* d_rope_cos = runtime.d_rope_cos;
    float* d_rope_sin = runtime.d_rope_sin;
    run_backward_output_stage_v2(cfg, layout, meta, forward, runtime, buf, cublas, grad_scale);
    for (int l = cfg.n_layers - 1; l >= 0; l--) {
        run_backward_transformer_layer_stage_v2(cfg, layout, meta, device_meta, upload, runtime, buf, cublas, grad_scale, anc_runtime, l, d_rope_cos, d_rope_sin);
    }

    if (finalize_anc_grad && anc_runtime && anc_runtime->enabled && anc_runtime->subtree_compact_chars > 0) {
        int H = cfg.n_heads;
        int HD = cfg.d_model / cfg.n_heads;
        int D = cfg.d_model;
        int n_sub = anc_runtime->subtree_compact_chars;
        float anc_alpha = 1.0f;
        float anc_one = 1.0f;
        for (int l = 0; l < cfg.n_layers; l++) {
            launch_rope_batched_inverse_v2(anc_runtime->d_dkv_subtree_k[l],
                                           anc_runtime->d_subtree_real_pos,
                                           d_rope_cos, d_rope_sin, n_sub * H, HD);
            float* dW_kw = runtime.d_grads + layout.wk_w[l];
            float* dW_vw = runtime.d_grads + layout.wv_w[l];
            AGPT_V2_CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, D, D, n_sub,
                                             &anc_alpha, anc_runtime->d_dkv_subtree_k[l], D,
                                             anc_runtime->d_h_subtree[l], D, &anc_one, dW_kw, D));
            AGPT_V2_CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, D, D, n_sub,
                                             &anc_alpha, anc_runtime->d_dkv_subtree_v[l], D,
                                             anc_runtime->d_h_subtree[l], D, &anc_one, dW_vw, D));
        }
    }

    run_backward_embedding_stage_v2(cfg, layout, upload, runtime, buf, T_q);

    AGPT_V2_CUDA_CHECK(cudaDeviceSynchronize());

    return result;
}

}  // namespace agpt_v2

#endif
