#ifndef AGPT_V2_FORWARD_STAGES_CUH
#define AGPT_V2_FORWARD_STAGES_CUH

static inline void run_forward_embedding_stage_v2(const ModelLayout& layout,
                                                  const ChunkUploadRuntimeV2& upload,
                                                  TrainerRuntimeV2& runtime,
                                                  ChunkBufferLayoutV2& buf,
                                                  int T_q,
                                                  int D,
                                                  const ForwardDiagDumpConfigV2* diag) {
    cuda_embedding_gather(runtime.d_weights + layout.token_emb, upload.d_token_ids, buf.query.x, T_q, D);
    if (diag && diag->active) {
        agpt_diag::emit_tensor_bin(diag->tensor_dir, diag->epoch, diag->root_id, diag->chunk_idx, 0,
                                   "fwd_x_post_embed", buf.query.x, T_q * D);
    }
}

static inline void run_forward_transformer_layer_stage_v2(const TrainerConfig& cfg,
                                                          const ModelLayout& layout,
                                                          const ChunkMetadataV2& meta,
                                                          const ChunkDeviceMetadataV2& device_meta,
                                                          const ChunkUploadRuntimeV2& upload,
                                                          TrainerRuntimeV2& runtime,
                                                          ChunkBufferLayoutV2& buf,
                                                          cublasHandle_t cublas,
                                                          float* d_rope_cos,
                                                          float* d_rope_sin,
                                                          UnitAncGradRuntimeV2* anc_runtime,
                                                          int layer,
                                                          const ForwardDiagDumpConfigV2* diag) {
    int T_q = meta.T_q;
    int D = cfg.d_model;
    int H = cfg.n_heads;
    int HD = cfg.d_model / cfg.n_heads;
    int F = cfg.d_ff;
    SavedLayerStateLayoutV2 saved = make_layer_saved_state_v2(runtime.chunk, layer, T_q, D, F, H, meta.max_kv_len);

    float* W_qw = runtime.d_weights + layout.wq_w[layer];
    float* W_qb = runtime.d_weights + layout.wq_b[layer];
    float* W_kw = runtime.d_weights + layout.wk_w[layer];
    float* W_kb = runtime.d_weights + layout.wk_b[layer];
    float* W_vw = runtime.d_weights + layout.wv_w[layer];
    float* W_vb = runtime.d_weights + layout.wv_b[layer];
    float* W_ow = runtime.d_weights + layout.wo_w[layer];
    float* W_ob = runtime.d_weights + layout.wo_b[layer];
    float* W_1w = runtime.d_weights + layout.l1_w[layer];
    float* W_1b = runtime.d_weights + layout.l1_b[layer];
    float* W_2w = runtime.d_weights + layout.l2_w[layer];
    float* W_2b = runtime.d_weights + layout.l2_b[layer];
    float* G1 = runtime.d_weights + layout.ln1_gamma[layer];
    float* B1 = runtime.d_weights + layout.ln1_beta[layer];
    float* G2 = runtime.d_weights + layout.ln2_gamma[layer];
    float* B2 = runtime.d_weights + layout.ln2_beta[layer];

    float alpha = 1.0f;
    float beta_zero = 0.0f;
    AGPT_V2_CUDA_CHECK(cudaMemcpy(saved.x_res1, buf.query.x, (size_t)((long long)T_q * D * sizeof(float)), cudaMemcpyDeviceToDevice));
    cuda_layer_norm_forward(buf.query.x, saved.ln1_out, saved.ln1_norm, saved.ln1_std_inv, G1, B1, T_q, D);
    if (diag && diag->active && layer == 0) {
        agpt_diag::emit_tensor_bin(diag->tensor_dir, diag->epoch, diag->root_id, diag->chunk_idx, layer,
                                   "fwd_x_post_ln1", saved.ln1_out, T_q * D);
    }
    if (anc_runtime && anc_runtime->enabled && anc_runtime->subtree_compact_chars > 0) {
        launch_save_ln1_to_subtree_v2(saved.ln1_out, upload.d_char_pos,
                                      runtime.cache.d_compact_slot,
                                      anc_runtime->d_compact_to_subtree_idx,
                                      anc_runtime->d_h_subtree[layer],
                                      T_q, D);
    }

    AGPT_V2_CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, D, T_q, D,
                                     &alpha, W_qw, D, saved.ln1_out, D, &beta_zero, buf.query.q, D));
    cuda_bias_add(buf.query.q, W_qb, T_q, D);
    AGPT_V2_CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, D, T_q, D,
                                     &alpha, W_kw, D, saved.ln1_out, D, &beta_zero, buf.query.k, D));
    cuda_bias_add(buf.query.k, W_kb, T_q, D);
    AGPT_V2_CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, D, T_q, D,
                                     &alpha, W_vw, D, saved.ln1_out, D, &beta_zero, buf.query.v, D));
    cuda_bias_add(buf.query.v, W_vb, T_q, D);

    launch_rope_batched_v2(buf.query.q, upload.d_rope_positions, d_rope_cos, d_rope_sin, T_q * H, HD);
    launch_rope_batched_v2(buf.query.k, upload.d_rope_positions, d_rope_cos, d_rope_sin, T_q * H, HD);
    AGPT_V2_CUDA_CHECK(cudaMemcpy(saved.q, buf.query.q, (size_t)((long long)T_q * D * sizeof(float)), cudaMemcpyDeviceToDevice));
    AGPT_V2_CUDA_CHECK(cudaMemcpy(saved.k, buf.query.k, (size_t)((long long)T_q * D * sizeof(float)), cudaMemcpyDeviceToDevice));
    AGPT_V2_CUDA_CHECK(cudaMemcpy(saved.v, buf.query.v, (size_t)((long long)T_q * D * sizeof(float)), cudaMemcpyDeviceToDevice));
    if (diag && diag->active && layer == 0) {
        agpt_diag::emit_tensor_bin(diag->tensor_dir, diag->epoch, diag->root_id, diag->chunk_idx, layer,
                                   "fwd_q", buf.query.q, T_q * D);
        agpt_diag::emit_tensor_bin(diag->tensor_dir, diag->epoch, diag->root_id, diag->chunk_idx, layer,
                                   "fwd_k_pre_scatter", buf.query.k, T_q * D);
        agpt_diag::emit_tensor_bin(diag->tensor_dir, diag->epoch, diag->root_id, diag->chunk_idx, layer,
                                   "fwd_v_pre_scatter", buf.query.v, T_q * D);
    }

    launch_kv_scatter_compact_bf16_v2(buf.query.k, upload.d_char_pos,
                                      runtime.cache.d_compact_slot, runtime.cache.d_k_layers[layer], T_q, D);
    launch_kv_scatter_compact_bf16_v2(buf.query.v, upload.d_char_pos,
                                      runtime.cache.d_compact_slot, runtime.cache.d_v_layers[layer], T_q, D);

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
    launch_kv_copy_own_edge_v2(buf.query.k, upload.d_query_offsets, upload.d_kv_offsets,
                               device_meta.d_anc_lengths, device_meta.d_own_lengths,
                               buf.packed.kv_pack_k, meta.N, H, HD);
    launch_kv_copy_own_edge_v2(buf.query.v, upload.d_query_offsets, upload.d_kv_offsets,
                               device_meta.d_anc_lengths, device_meta.d_own_lengths,
                               buf.packed.kv_pack_v, meta.N, H, HD);
    if (diag && diag->active && layer == 0) {
        agpt_diag::emit_tensor_bin(diag->tensor_dir, diag->epoch, diag->root_id, diag->chunk_idx, layer,
                                   "fwd_kv_pack_k", buf.packed.kv_pack_k, meta.T_kv * D);
        agpt_diag::emit_tensor_bin(diag->tensor_dir, diag->epoch, diag->root_id, diag->chunk_idx, layer,
                                   "fwd_kv_pack_v", buf.packed.kv_pack_v, meta.T_kv * D);
    }

    cuda_batched_varlen_attention_L_queries(
        buf.query.q, buf.packed.kv_pack_k, buf.packed.kv_pack_v,
        upload.d_query_to_node, upload.d_query_offsets, upload.d_kv_offsets, upload.d_kv_lengths,
        buf.query.attn_out, buf.packed.attn_weights,
        T_q, H, HD, meta.max_kv_len, 1.0f / sqrtf((float)HD));
    if (diag && diag->active && layer == 0) {
        agpt_diag::emit_tensor_bin(diag->tensor_dir, diag->epoch, diag->root_id, diag->chunk_idx, layer,
                                   "fwd_attn_out", buf.query.attn_out, T_q * D);
    }
    AGPT_V2_CUDA_CHECK(cudaMemcpy(saved.attn_out, buf.query.attn_out, (size_t)((long long)T_q * D * sizeof(float)), cudaMemcpyDeviceToDevice));
    AGPT_V2_CUDA_CHECK(cudaMemcpy(saved.attn_weights, buf.packed.attn_weights,
                                  (size_t)((long long)T_q * H * meta.max_kv_len * sizeof(float)),
                                  cudaMemcpyDeviceToDevice));

    AGPT_V2_CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, D, T_q, D,
                                     &alpha, W_ow, D, buf.query.attn_out, D, &beta_zero, buf.query.ff_out, D));
    cuda_bias_add(buf.query.ff_out, W_ob, T_q, D);
    launch_elem_add_v2(buf.query.x, buf.query.ff_out, T_q * D);
    AGPT_V2_CUDA_CHECK(cudaMemcpy(saved.x_res2, buf.query.x, (size_t)((long long)T_q * D * sizeof(float)), cudaMemcpyDeviceToDevice));

    cuda_layer_norm_forward(buf.query.x, saved.ln2_out, saved.ln2_norm, saved.ln2_std_inv, G2, B2, T_q, D);
    AGPT_V2_CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, F, T_q, D,
                                     &alpha, W_1w, F, saved.ln2_out, D, &beta_zero, buf.query.ff_h, F));
    cuda_fused_bias_relu(buf.query.ff_h, W_1b, buf.query.ff_h, saved.ff_mask, T_q, F);
    AGPT_V2_CUDA_CHECK(cudaMemcpy(saved.ff_h, buf.query.ff_h, (size_t)((long long)T_q * F * sizeof(float)), cudaMemcpyDeviceToDevice));
    AGPT_V2_CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, D, T_q, F,
                                     &alpha, W_2w, D, buf.query.ff_h, F, &beta_zero, buf.query.ff_out, D));
    cuda_bias_add(buf.query.ff_out, W_2b, T_q, D);
    launch_elem_add_v2(buf.query.x, buf.query.ff_out, T_q * D);
}

static inline void run_forward_output_stage_v2(const TrainerConfig& cfg,
                                               const ModelLayout& layout,
                                               const ChunkMetadataV2& meta,
                                               const ChunkDeviceMetadataV2& device_meta,
                                               const LossTablesV2& loss_tables,
                                               const ChunkUploadRuntimeV2& upload,
                                               TrainerRuntimeV2& runtime,
                                               ChunkBufferLayoutV2& buf,
                                               cublasHandle_t cublas) {
    int T_q = meta.T_q;
    int D = cfg.d_model;
    float alpha = 1.0f;
    float beta_zero = 0.0f;
    float* Gf = runtime.d_weights + layout.final_gamma;
    float* Bf = runtime.d_weights + layout.final_beta;
    float* W_out = runtime.d_weights + layout.out_w;
    float* B_out = runtime.d_weights + layout.out_b;

    cuda_layer_norm_forward(buf.query.x, buf.output.final_out, buf.output.final_norm, buf.output.final_std_inv, Gf, Bf, T_q, D);
    AGPT_V2_CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, cfg.vocab_size, T_q, D,
                                     &alpha, W_out, cfg.vocab_size, buf.output.final_out, D, &beta_zero, buf.output.logits, cfg.vocab_size));
    cuda_bias_add(buf.output.logits, B_out, T_q, cfg.vocab_size);
    launch_agpt_loss_per_query_v2(
        buf.output.logits, upload.d_query_to_node, upload.d_query_offsets, upload.d_radix_ids, upload.d_token_ids,
        loss_tables.d_counts_offset, loss_tables.d_counts_len, loss_tables.d_counts_tok, loss_tables.d_counts_val,
        device_meta.d_query_weights, buf.output.d_logits, buf.output.loss, T_q, cfg.vocab_size);
}

#endif
