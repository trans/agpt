// Stub implementations for when CUDA kernels are not available.
// These satisfy the linker but should never be called at runtime
// (only the cublas backend calls them).

void cuda_softmax_rows(const float* input, float* output, int rows, int cols) {}
void cuda_softmax_backward(const float* s, const float* ds, float* result, int rows, int cols) {}
void cuda_causal_mask(float* data, int n) {}
void cuda_causal_mask_batched(float* data, int total_rows, int cols, int seq_len) {}
void cuda_bias_add(float* data, const float* bias, int rows, int cols) {}
void cuda_relu_forward(const float* input, float* output, float* mask, int n) {}
void cuda_relu_backward(const float* grad, const float* mask, float* output, int n) {}
void cuda_layer_norm_forward(const float* input, float* output, float* norm_out,
                              float* std_inv_out, const float* gamma, const float* beta,
                              int rows, int cols) {}
void cuda_layer_norm_backward(const float* grad, const float* norm, const float* std_inv,
                               const float* gamma, float* dx, float* dgamma, float* dbeta,
                               int rows, int cols) {}
void cuda_adam_step(float* param, const float* grad, float* m, float* v,
                     float lr, float beta1, float beta2, float eps, int t, int n) {}
void cuda_rope_apply(float* x, const float* cos_cache, const float* sin_cache, int seq_len, int dim) {}
void cuda_rope_apply_inverse(float* x, const float* cos_cache, const float* sin_cache, int seq_len, int dim) {}
void cuda_fused_attn_softmax(const float* scores, float* output, float scale, int rows, int cols) {}
void cuda_fused_attn_softmax_backward(const float* s, const float* ds, float* result, float scale, int rows, int cols) {}
void cuda_embedding_gather(const float* token_emb, const int* ids,
                            float* output, int seq_len, int d_model) {}
void cuda_embedding_scatter_add(const float* grad, const int* ids,
                                 float* d_token_emb, int seq_len, int d_model) {}
void cuda_fused_softmax_ce_grad(const float* logits, const int* targets,
                                 float* d_logits, float* loss_out,
                                 int rows, int cols) {}
void cuda_fused_bias_relu(const float* input, const float* bias,
                          float* output, float* mask, int rows, int cols) {}
void cuda_adam_bulk(float* params, float* grads, float* m, float* v,
                     float lr, float beta1, float beta2, float eps,
                     int t, int n) {}
void cuda_sync() {}
