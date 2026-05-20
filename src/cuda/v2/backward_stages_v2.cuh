#ifndef AGPT_V2_BACKWARD_STAGES_CUH
#define AGPT_V2_BACKWARD_STAGES_CUH

static inline void run_backward_output_stage_v2(const TrainerConfig& cfg,
                                                const ModelLayout& layout,
                                                const ChunkMetadataV2& meta,
                                                const ForwardPassResult& forward,
                                                TrainerRuntimeV2& runtime,
                                                ChunkBufferLayoutV2& buf,
                                                cublasHandle_t cublas,
                                                float grad_scale) {
    int T_q = meta.T_q;
    int D = cfg.d_model;
    int V = cfg.vocab_size;
    float alpha = 1.0f;
    float beta_zero = 0.0f;
    float* W_out = runtime.d_weights + layout.out_w;
    float* Gf = runtime.d_weights + layout.final_gamma;
    float* dG_out = runtime.d_grads + layout.out_w;
    float* dB_out = runtime.d_grads + layout.out_b;
    float* dGf = runtime.d_grads + layout.final_gamma;
    float* dBf = runtime.d_grads + layout.final_beta;

    AGPT_V2_CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, D, T_q, V,
                                     &alpha, W_out, V, buf.output.d_logits, V, &beta_zero, buf.query.q, D));
    AGPT_V2_CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, V, D, T_q,
                                     &grad_scale, buf.output.d_logits, V, buf.output.final_out, D, &alpha, dG_out, V));
    launch_bias_grad_accum_v2(buf.output.d_logits, T_q, V, grad_scale, dB_out);

    cuda_layer_norm_backward(buf.query.q, buf.output.final_norm, buf.output.final_std_inv,
                             Gf, buf.query.x, dGf, dBf, T_q, D);
}

static inline void run_backward_transformer_layer_stage_v2(const TrainerConfig& cfg,
                                                           const ModelLayout& layout,
                                                           const ChunkMetadataV2& meta,
                                                           const ChunkDeviceMetadataV2& device_meta,
                                                           const ChunkUploadRuntimeV2& upload,
                                                           TrainerRuntimeV2& runtime,
                                                           ChunkBufferLayoutV2& buf,
                                                           cublasHandle_t cublas,
                                                           float grad_scale,
                                                           int layer,
                                                           float* d_rope_cos,
                                                           float* d_rope_sin) {
    int T_q = meta.T_q;
    int D = cfg.d_model;
    int F = cfg.d_ff;
    int H = cfg.n_heads;
    int HD = cfg.d_model / cfg.n_heads;

    SavedLayerStateLayoutV2 saved = make_layer_saved_state_v2(runtime.chunk, layer, T_q, D, F, H, meta.max_kv_len);
    float alpha = 1.0f;
    float beta_zero = 0.0f;

    float* W_1w = runtime.d_weights + layout.l1_w[layer];
    float* W_2w = runtime.d_weights + layout.l2_w[layer];
    float* W_ow = runtime.d_weights + layout.wo_w[layer];
    float* W_qw = runtime.d_weights + layout.wq_w[layer];
    float* W_kw = runtime.d_weights + layout.wk_w[layer];
    float* W_vw = runtime.d_weights + layout.wv_w[layer];
    float* G2 = runtime.d_weights + layout.ln2_gamma[layer];
    float* G1 = runtime.d_weights + layout.ln1_gamma[layer];
    float* dW_2w = runtime.d_grads + layout.l2_w[layer];
    float* dW_2b = runtime.d_grads + layout.l2_b[layer];
    float* dW_1w = runtime.d_grads + layout.l1_w[layer];
    float* dW_1b = runtime.d_grads + layout.l1_b[layer];
    float* dW_ow = runtime.d_grads + layout.wo_w[layer];
    float* dW_ob = runtime.d_grads + layout.wo_b[layer];
    float* dG2 = runtime.d_grads + layout.ln2_gamma[layer];
    float* dB2 = runtime.d_grads + layout.ln2_beta[layer];
    float* dW_qw = runtime.d_grads + layout.wq_w[layer];
    float* dW_qb = runtime.d_grads + layout.wq_b[layer];
    float* dW_kw = runtime.d_grads + layout.wk_w[layer];
    float* dW_kb = runtime.d_grads + layout.wk_b[layer];
    float* dW_vw = runtime.d_grads + layout.wv_w[layer];
    float* dW_vb = runtime.d_grads + layout.wv_b[layer];
    float* dG1 = runtime.d_grads + layout.ln1_gamma[layer];
    float* dB1 = runtime.d_grads + layout.ln1_beta[layer];

    AGPT_V2_CUDA_CHECK(cudaMemcpy(buf.query.ff_out, buf.query.x, (size_t)((long long)T_q * D * sizeof(float)), cudaMemcpyDeviceToDevice));
    AGPT_V2_CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, F, T_q, D,
                                     &alpha, W_2w, D, buf.query.ff_out, D, &beta_zero, buf.query.ff_h, F));
    AGPT_V2_CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, D, F, T_q,
                                     &grad_scale, buf.query.ff_out, D, saved.ff_h, F, &alpha, dW_2w, D));
    launch_bias_grad_accum_v2(buf.query.ff_out, T_q, D, grad_scale, dW_2b);

    cuda_relu_backward(buf.query.ff_h, saved.ff_mask, buf.query.ff_h, T_q * F);

    AGPT_V2_CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, D, T_q, F,
                                     &alpha, W_1w, F, buf.query.ff_h, F, &beta_zero, buf.query.ln_out, D));
    AGPT_V2_CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, F, D, T_q,
                                     &grad_scale, buf.query.ff_h, F, saved.ln2_out, D, &alpha, dW_1w, F));
    launch_bias_grad_accum_v2(buf.query.ff_h, T_q, F, grad_scale, dW_1b);

    cuda_layer_norm_backward(buf.query.ln_out, saved.ln2_norm, saved.ln2_std_inv,
                             G2, buf.query.ln_out, dG2, dB2, T_q, D);
    launch_elem_add_v2(buf.query.x, buf.query.ln_out, T_q * D);

    AGPT_V2_CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, D, T_q, D,
                                     &alpha, W_ow, D, buf.query.x, D, &beta_zero, buf.query.attn_out, D));
    AGPT_V2_CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, D, D, T_q,
                                     &grad_scale, buf.query.x, D, saved.attn_out, D, &alpha, dW_ow, D));
    launch_bias_grad_accum_v2(buf.query.x, T_q, D, grad_scale, dW_ob);

    launch_kv_gather_anc_compact_bf16_v2(runtime.cache.d_k_layers[layer],
                                         device_meta.d_anc_ids, device_meta.d_anc_offsets,
                                         upload.d_kv_offsets, device_meta.d_anc_lengths,
                                         runtime.cache.d_compact_slot, buf.packed.kv_pack_k,
                                         meta.N, H, HD);
    launch_kv_gather_anc_compact_bf16_v2(runtime.cache.d_v_layers[layer],
                                         device_meta.d_anc_ids, device_meta.d_anc_offsets,
                                         upload.d_kv_offsets, device_meta.d_anc_lengths,
                                         runtime.cache.d_compact_slot, buf.packed.kv_pack_v,
                                         meta.N, H, HD);
    launch_kv_copy_own_edge_v2(saved.k, upload.d_query_offsets, upload.d_kv_offsets,
                               device_meta.d_anc_lengths, device_meta.d_own_lengths,
                               buf.packed.kv_pack_k, meta.N, H, HD);
    launch_kv_copy_own_edge_v2(saved.v, upload.d_query_offsets, upload.d_kv_offsets,
                               device_meta.d_anc_lengths, device_meta.d_own_lengths,
                               buf.packed.kv_pack_v, meta.N, H, HD);
    AGPT_V2_CUDA_CHECK(cudaMemset(buf.packed.d_dk_pack, 0, (size_t)((long long)meta.T_kv * D * sizeof(float))));
    AGPT_V2_CUDA_CHECK(cudaMemset(buf.packed.d_dv_pack, 0, (size_t)((long long)meta.T_kv * D * sizeof(float))));
    cuda_batched_varlen_attention_L_queries_backward(
        saved.q, buf.packed.kv_pack_k, buf.packed.kv_pack_v, saved.attn_weights, buf.query.attn_out,
        upload.d_query_to_node, upload.d_query_offsets, upload.d_kv_offsets, upload.d_kv_lengths,
        buf.packed.d_dq_pack, buf.packed.d_dk_pack, buf.packed.d_dv_pack,
        T_q, H, HD, meta.max_kv_len, 1.0f / sqrtf((float)HD));

    launch_rope_batched_inverse_v2(buf.packed.d_dq_pack, upload.d_rope_positions, d_rope_cos, d_rope_sin, T_q * H, HD);
    AGPT_V2_CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, D, T_q, D,
                                     &alpha, W_qw, D, buf.packed.d_dq_pack, D, &beta_zero, buf.query.ln_out, D));
    AGPT_V2_CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, D, D, T_q,
                                     &grad_scale, buf.packed.d_dq_pack, D, saved.ln1_out, D, &alpha, dW_qw, D));
    launch_bias_grad_accum_v2(buf.packed.d_dq_pack, T_q, D, grad_scale, dW_qb);

    AGPT_V2_CUDA_CHECK(cudaMemset(buf.query.k, 0, (size_t)((long long)T_q * D * sizeof(float))));
    AGPT_V2_CUDA_CHECK(cudaMemset(buf.query.v, 0, (size_t)((long long)T_q * D * sizeof(float))));
    launch_kv_uncopy_own_edge_v2(buf.packed.d_dk_pack, upload.d_query_offsets, upload.d_kv_offsets,
                                 device_meta.d_anc_lengths, device_meta.d_own_lengths,
                                 buf.query.k, meta.N, H, HD);
    launch_kv_uncopy_own_edge_v2(buf.packed.d_dv_pack, upload.d_query_offsets, upload.d_kv_offsets,
                                 device_meta.d_anc_lengths, device_meta.d_own_lengths,
                                 buf.query.v, meta.N, H, HD);
    launch_rope_batched_inverse_v2(buf.query.k, upload.d_rope_positions, d_rope_cos, d_rope_sin, T_q * H, HD);

    AGPT_V2_CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, D, T_q, D,
                                     &alpha, W_kw, D, buf.query.k, D, &alpha, buf.query.ln_out, D));
    AGPT_V2_CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, D, D, T_q,
                                     &grad_scale, buf.query.k, D, saved.ln1_out, D, &alpha, dW_kw, D));
    launch_bias_grad_accum_v2(buf.query.k, T_q, D, grad_scale, dW_kb);

    AGPT_V2_CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, D, T_q, D,
                                     &alpha, W_vw, D, buf.query.v, D, &alpha, buf.query.ln_out, D));
    AGPT_V2_CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, D, D, T_q,
                                     &grad_scale, buf.query.v, D, saved.ln1_out, D, &alpha, dW_vw, D));
    launch_bias_grad_accum_v2(buf.query.v, T_q, D, grad_scale, dW_vb);

    cuda_layer_norm_backward(buf.query.ln_out, saved.ln1_norm, saved.ln1_std_inv,
                             G1, buf.query.ln_out, dG1, dB1, T_q, D);
    launch_elem_add_v2(buf.query.x, buf.query.ln_out, T_q * D);
}

static inline void run_backward_embedding_stage_v2(const TrainerConfig& cfg,
                                                   const ModelLayout& layout,
                                                   const ChunkUploadRuntimeV2& upload,
                                                   TrainerRuntimeV2& runtime,
                                                   ChunkBufferLayoutV2& buf,
                                                   int T_q) {
    float* dG_emb = runtime.d_grads + layout.token_emb;
    cuda_embedding_scatter_add(buf.query.x, upload.d_token_ids, dG_emb, T_q, cfg.d_model);
}

#endif
