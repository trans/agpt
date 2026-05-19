#ifndef AGPT_TRANSFORMER_CHUNK_RUNTIME_CUH
#define AGPT_TRANSFORMER_CHUNK_RUNTIME_CUH

struct TransformerChunkRuntime {
    float *d_x = nullptr, *d_x_res1 = nullptr, *d_x_res2 = nullptr, *d_ln_out = nullptr;
    float *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_attn_out = nullptr;
    float *d_ff_h = nullptr, *d_ff_mask = nullptr, *d_ff_out = nullptr;

    float *d_final_out = nullptr, *d_final_norm_save = nullptr, *d_final_std_inv_save = nullptr;
    float *d_logits = nullptr, *d_d_logits = nullptr, *d_loss = nullptr, *d_d_final_out = nullptr;

    float *d_dk_own = nullptr, *d_dv_own = nullptr;

    float** sv_x_res1 = nullptr;
    float** sv_ln1_norm = nullptr;
    float** sv_ln1_std_inv = nullptr;
    float** sv_ln1_out = nullptr;
    float** sv_x_res2 = nullptr;
    float** sv_ln2_norm = nullptr;
    float** sv_ln2_std_inv = nullptr;
    float** sv_ln2_out = nullptr;
    float** sv_ff_h = nullptr;
    float** sv_ff_mask = nullptr;
    float** sv_attn_out = nullptr;
    float** sv_attn_weights = nullptr;
    float** sv_q = nullptr;
    float** sv_k = nullptr;
    float** sv_v = nullptr;

    float *d_q_pack_flat = nullptr, *d_kv_pack_k = nullptr, *d_kv_pack_v = nullptr;
    float *d_dq_pack = nullptr, *d_dk_pack = nullptr, *d_dv_pack = nullptr;
    long long T_kv_max = 0;
    int L_layers = 0;
};

static void init_transformer_chunk_runtime(TransformerChunkRuntime& runtime,
                                           int T_q_cap,
                                           int N_cap,
                                           int D,
                                           int F,
                                           int V,
                                           int L_layers,
                                           int H,
                                           int HD,
                                           int max_kv_per_node) {
    runtime.L_layers = L_layers;
    CUDA_CHECK(cudaMalloc(&runtime.d_x,         (long long)T_q_cap * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&runtime.d_x_res1,    (long long)T_q_cap * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&runtime.d_x_res2,    (long long)T_q_cap * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&runtime.d_ln_out,    (long long)T_q_cap * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&runtime.d_q,         (long long)T_q_cap * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&runtime.d_k,         (long long)T_q_cap * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&runtime.d_v,         (long long)T_q_cap * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&runtime.d_attn_out,  (long long)T_q_cap * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&runtime.d_ff_h,      (long long)T_q_cap * F * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&runtime.d_ff_mask,   (long long)T_q_cap * F * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&runtime.d_ff_out,    (long long)T_q_cap * D * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&runtime.d_final_out,          (long long)N_cap * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&runtime.d_final_norm_save,    (long long)N_cap * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&runtime.d_final_std_inv_save, (long long)N_cap * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&runtime.d_logits,             (long long)N_cap * V * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&runtime.d_d_logits,           (long long)N_cap * V * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&runtime.d_loss,               (long long)N_cap * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&runtime.d_d_final_out,        (long long)N_cap * D * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&runtime.d_dk_own, (long long)T_q_cap * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&runtime.d_dv_own, (long long)T_q_cap * D * sizeof(float)));

    runtime.sv_x_res1 = (float**)malloc(L_layers * sizeof(float*));
    runtime.sv_ln1_norm = (float**)malloc(L_layers * sizeof(float*));
    runtime.sv_ln1_std_inv = (float**)malloc(L_layers * sizeof(float*));
    runtime.sv_ln1_out = (float**)malloc(L_layers * sizeof(float*));
    runtime.sv_x_res2 = (float**)malloc(L_layers * sizeof(float*));
    runtime.sv_ln2_norm = (float**)malloc(L_layers * sizeof(float*));
    runtime.sv_ln2_std_inv = (float**)malloc(L_layers * sizeof(float*));
    runtime.sv_ln2_out = (float**)malloc(L_layers * sizeof(float*));
    runtime.sv_ff_h = (float**)malloc(L_layers * sizeof(float*));
    runtime.sv_ff_mask = (float**)malloc(L_layers * sizeof(float*));
    runtime.sv_attn_out = (float**)malloc(L_layers * sizeof(float*));
    runtime.sv_attn_weights = (float**)malloc(L_layers * sizeof(float*));
    runtime.sv_q = (float**)malloc(L_layers * sizeof(float*));
    runtime.sv_k = (float**)malloc(L_layers * sizeof(float*));
    runtime.sv_v = (float**)malloc(L_layers * sizeof(float*));
    for (int l = 0; l < L_layers; l++) {
        CUDA_CHECK(cudaMalloc(&runtime.sv_x_res1[l],      (long long)T_q_cap * D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&runtime.sv_ln1_norm[l],    (long long)T_q_cap * D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&runtime.sv_ln1_std_inv[l], (long long)T_q_cap * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&runtime.sv_ln1_out[l],     (long long)T_q_cap * D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&runtime.sv_x_res2[l],      (long long)T_q_cap * D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&runtime.sv_ln2_norm[l],    (long long)T_q_cap * D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&runtime.sv_ln2_std_inv[l], (long long)T_q_cap * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&runtime.sv_ln2_out[l],     (long long)T_q_cap * D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&runtime.sv_ff_h[l],        (long long)T_q_cap * F * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&runtime.sv_ff_mask[l],     (long long)T_q_cap * F * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&runtime.sv_attn_out[l],    (long long)T_q_cap * D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&runtime.sv_attn_weights[l],(long long)T_q_cap * H * max_kv_per_node * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&runtime.sv_q[l],           (long long)T_q_cap * D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&runtime.sv_k[l],           (long long)T_q_cap * D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&runtime.sv_v[l],           (long long)T_q_cap * D * sizeof(float)));
    }

    runtime.T_kv_max = (long long)T_q_cap * max_kv_per_node;
    if (runtime.T_kv_max > (long long)T_q_cap * 2000) runtime.T_kv_max = (long long)T_q_cap * 2000;
    CUDA_CHECK(cudaMalloc(&runtime.d_q_pack_flat, (long long)T_q_cap * H * HD * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&runtime.d_kv_pack_k,   runtime.T_kv_max * H * HD * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&runtime.d_kv_pack_v,   runtime.T_kv_max * H * HD * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&runtime.d_dq_pack,     (long long)T_q_cap * H * HD * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&runtime.d_dk_pack,     runtime.T_kv_max * H * HD * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&runtime.d_dv_pack,     runtime.T_kv_max * H * HD * sizeof(float)));
}

static void free_transformer_chunk_runtime(TransformerChunkRuntime& runtime) {
    cudaFree(runtime.d_x); cudaFree(runtime.d_x_res1); cudaFree(runtime.d_x_res2); cudaFree(runtime.d_ln_out);
    cudaFree(runtime.d_q); cudaFree(runtime.d_k); cudaFree(runtime.d_v); cudaFree(runtime.d_attn_out);
    cudaFree(runtime.d_ff_h); cudaFree(runtime.d_ff_mask); cudaFree(runtime.d_ff_out);
    cudaFree(runtime.d_final_out); cudaFree(runtime.d_final_norm_save); cudaFree(runtime.d_final_std_inv_save);
    cudaFree(runtime.d_logits); cudaFree(runtime.d_d_logits); cudaFree(runtime.d_loss); cudaFree(runtime.d_d_final_out);
    cudaFree(runtime.d_dk_own); cudaFree(runtime.d_dv_own);
    if (runtime.sv_x_res1) {
        for (int l = 0; l < runtime.L_layers; l++) {
            cudaFree(runtime.sv_x_res1[l]); cudaFree(runtime.sv_ln1_norm[l]); cudaFree(runtime.sv_ln1_std_inv[l]); cudaFree(runtime.sv_ln1_out[l]);
            cudaFree(runtime.sv_x_res2[l]); cudaFree(runtime.sv_ln2_norm[l]); cudaFree(runtime.sv_ln2_std_inv[l]); cudaFree(runtime.sv_ln2_out[l]);
            cudaFree(runtime.sv_ff_h[l]); cudaFree(runtime.sv_ff_mask[l]); cudaFree(runtime.sv_attn_out[l]); cudaFree(runtime.sv_attn_weights[l]);
            cudaFree(runtime.sv_q[l]); cudaFree(runtime.sv_k[l]); cudaFree(runtime.sv_v[l]);
        }
    }
    free(runtime.sv_x_res1); free(runtime.sv_ln1_norm); free(runtime.sv_ln1_std_inv); free(runtime.sv_ln1_out);
    free(runtime.sv_x_res2); free(runtime.sv_ln2_norm); free(runtime.sv_ln2_std_inv); free(runtime.sv_ln2_out);
    free(runtime.sv_ff_h); free(runtime.sv_ff_mask); free(runtime.sv_attn_out); free(runtime.sv_attn_weights);
    free(runtime.sv_q); free(runtime.sv_k); free(runtime.sv_v);
    cudaFree(runtime.d_q_pack_flat); cudaFree(runtime.d_kv_pack_k); cudaFree(runtime.d_kv_pack_v);
    cudaFree(runtime.d_dq_pack); cudaFree(runtime.d_dk_pack); cudaFree(runtime.d_dv_pack);
    runtime = TransformerChunkRuntime{};
}

#endif
