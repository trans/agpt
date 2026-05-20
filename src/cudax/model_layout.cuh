#ifndef AGPT_V2_MODEL_LAYOUT_CUH
#define AGPT_V2_MODEL_LAYOUT_CUH

#include "types.cuh"

namespace agpt_v2 {

// Explicit parameter layout contract for the v2 trainer.
// v2 should treat model layout as a first-class interface rather than
// letting offsets leak through the training loop.
struct ModelLayout {
    RuntimeShape shape;
    int token_emb = 0;
    int* wq_w = nullptr;
    int* wq_b = nullptr;
    int* wk_w = nullptr;
    int* wk_b = nullptr;
    int* wv_w = nullptr;
    int* wv_b = nullptr;
    int* wo_w = nullptr;
    int* wo_b = nullptr;
    int* ln1_gamma = nullptr;
    int* ln1_beta = nullptr;
    int* l1_w = nullptr;
    int* l1_b = nullptr;
    int* l2_w = nullptr;
    int* l2_b = nullptr;
    int* ln2_gamma = nullptr;
    int* ln2_beta = nullptr;
    int final_gamma = 0;
    int final_beta = 0;
    int out_w = 0;
    int out_b = 0;
    int total_floats = 0;
};

static inline ModelLayout make_model_layout(const RuntimeShape& shape) {
    ModelLayout layout;
    layout.shape = shape;
    int L = shape.n_layers;
    int D = shape.d_model;
    int F = shape.d_ff;
    int V = shape.vocab_size;
    layout.wq_w = (int*)std::malloc(L * sizeof(int));
    layout.wq_b = (int*)std::malloc(L * sizeof(int));
    layout.wk_w = (int*)std::malloc(L * sizeof(int));
    layout.wk_b = (int*)std::malloc(L * sizeof(int));
    layout.wv_w = (int*)std::malloc(L * sizeof(int));
    layout.wv_b = (int*)std::malloc(L * sizeof(int));
    layout.wo_w = (int*)std::malloc(L * sizeof(int));
    layout.wo_b = (int*)std::malloc(L * sizeof(int));
    layout.ln1_gamma = (int*)std::malloc(L * sizeof(int));
    layout.ln1_beta = (int*)std::malloc(L * sizeof(int));
    layout.l1_w = (int*)std::malloc(L * sizeof(int));
    layout.l1_b = (int*)std::malloc(L * sizeof(int));
    layout.l2_w = (int*)std::malloc(L * sizeof(int));
    layout.l2_b = (int*)std::malloc(L * sizeof(int));
    layout.ln2_gamma = (int*)std::malloc(L * sizeof(int));
    layout.ln2_beta = (int*)std::malloc(L * sizeof(int));
    int off = 0;
    layout.token_emb = off; off += V * D;
    for (int i = 0; i < L; i++) {
        layout.wq_w[i] = off; off += D * D;
        layout.wq_b[i] = off; off += D;
        layout.wk_w[i] = off; off += D * D;
        layout.wk_b[i] = off; off += D;
        layout.wv_w[i] = off; off += D * D;
        layout.wv_b[i] = off; off += D;
        layout.wo_w[i] = off; off += D * D;
        layout.wo_b[i] = off; off += D;
        layout.ln1_gamma[i] = off; off += D;
        layout.ln1_beta[i] = off; off += D;
        layout.l1_w[i] = off; off += D * F;
        layout.l1_b[i] = off; off += F;
        layout.l2_w[i] = off; off += F * D;
        layout.l2_b[i] = off; off += D;
        layout.ln2_gamma[i] = off; off += D;
        layout.ln2_beta[i] = off; off += D;
    }
    layout.final_gamma = off; off += D;
    layout.final_beta = off; off += D;
    layout.out_w = off; off += D * V;
    layout.out_b = off; off += V;
    layout.total_floats = off;
    return layout;
}

static inline void free_model_layout(ModelLayout& layout) {
    std::free(layout.wq_w);
    std::free(layout.wq_b);
    std::free(layout.wk_w);
    std::free(layout.wk_b);
    std::free(layout.wv_w);
    std::free(layout.wv_b);
    std::free(layout.wo_w);
    std::free(layout.wo_b);
    std::free(layout.ln1_gamma);
    std::free(layout.ln1_beta);
    std::free(layout.l1_w);
    std::free(layout.l1_b);
    std::free(layout.l2_w);
    std::free(layout.l2_b);
    std::free(layout.ln2_gamma);
    std::free(layout.ln2_beta);
    layout = ModelLayout{};
}

}  // namespace agpt_v2

#endif
