#ifndef AGPT_COMMON_INIT_WEIGHTS_H
#define AGPT_COMMON_INIT_WEIGHTS_H

// Shared weight initialization for both v1 (src/cuda/agpt_train.cu)
// and v2 (src/cudax/agpt_seed.cu). Single source of truth so the two
// trainers produce byte-identical seed checkpoints from the same
// (seed, dims). Layout-struct-agnostic: caller passes per-matrix
// offsets via the InitLayout descriptor below.
//
// Scheme mirrors microgpt's micro_gpt.cr (~line 498 for linears, ~line
// 1152 for token embeddings) so --init produces a baseline
// statistically equivalent to the legacy microgpt-generated seed
// checkpoints that set the historical PPL baselines on this codebase:
//
//   token_emb (V × D):       N(0, sqrt(1/D))      — embedding init
//   linear weights (fan_in): N(0, sqrt(2/fan_in)) — Kaiming/He
//     wq_w, wk_w, wv_w, wo_w, l1_w, out_w: fan_in = D (d_model)
//     l2_w:                                 fan_in = F (d_ff)
//   biases: 0.0
//   LayerNorm gamma: 1.0; beta: 0.0
//
// Earlier --init code used flat N(0, 0.02) (GPT-2 paper schedule)
// which produced weights ~6-18× smaller than microgpt across every
// linear layer and cost ~2.2 PPL at Shakespeare 1M d=16 10 SE.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>

namespace agpt_init {

// Offsets for one transformer layer. Caller fills from its layout
// struct; this header has no dependency on either v1's WeightOffsets
// or v2's ModelLayout.
struct LayerOffsets {
    int wq_w, wq_b;
    int wk_w, wk_b;
    int wv_w, wv_b;
    int wo_w, wo_b;
    int ln1_gamma, ln1_beta;
    int l1_w, l1_b;
    int l2_w, l2_b;
    int ln2_gamma, ln2_beta;
};

// Full layout descriptor passed to init_random_weights.
struct InitLayout {
    int d_model;
    int d_ff;
    int vocab_size;
    int n_layers;
    int token_emb;
    const LayerOffsets* layers;   // length n_layers
    int final_gamma, final_beta;
    int out_w, out_b;
};

inline void init_random_weights(float* buf,
                                const InitLayout& l,
                                uint32_t seed,
                                bool verbose = true) {
    std::mt19937 rng(seed);

    auto fill_normal = [&](int off, int len, float stddev) {
        std::normal_distribution<float> nd(0.0f, stddev);
        for (int i = 0; i < len; i++) buf[off + i] = nd(rng);
    };
    auto fill_zero = [&](int off, int len) {
        for (int i = 0; i < len; i++) buf[off + i] = 0.0f;
    };
    auto fill_one = [&](int off, int len) {
        for (int i = 0; i < len; i++) buf[off + i] = 1.0f;
    };

    const int D = l.d_model;
    const int F = l.d_ff;
    const int V = l.vocab_size;
    const int L = l.n_layers;

    const float std_emb       = std::sqrt(1.0f / (float)D);
    const float std_kaiming_D = std::sqrt(2.0f / (float)D);
    const float std_kaiming_F = std::sqrt(2.0f / (float)F);

    fill_normal(l.token_emb, V * D, std_emb);
    for (int i = 0; i < L; i++) {
        const LayerOffsets& la = l.layers[i];
        fill_normal(la.wq_w, D * D, std_kaiming_D); fill_zero(la.wq_b, D);
        fill_normal(la.wk_w, D * D, std_kaiming_D); fill_zero(la.wk_b, D);
        fill_normal(la.wv_w, D * D, std_kaiming_D); fill_zero(la.wv_b, D);
        fill_normal(la.wo_w, D * D, std_kaiming_D); fill_zero(la.wo_b, D);
        fill_one  (la.ln1_gamma, D); fill_zero(la.ln1_beta, D);
        fill_normal(la.l1_w, D * F, std_kaiming_D); fill_zero(la.l1_b, F);
        fill_normal(la.l2_w, F * D, std_kaiming_F); fill_zero(la.l2_b, D);
        fill_one  (la.ln2_gamma, D); fill_zero(la.ln2_beta, D);
    }
    fill_one  (l.final_gamma, D);  fill_zero(l.final_beta, D);
    fill_normal(l.out_w, D * V, std_kaiming_D);  fill_zero(l.out_b, V);

    if (verbose) {
        std::printf("  Init: microgpt-style — emb N(0, %.4f), Kaiming N(0, %.4f) [fan_in=D], "
                    "N(0, %.4f) [fan_in=F=%d], biases 0, LN gamma=1 (seed=%u)\n",
                    std_emb, std_kaiming_D, std_kaiming_F, F, seed);
        std::printf("  Model: d=%d ff=%d vocab=%d layers=%d\n", D, F, V, L);
    }
}

}  // namespace agpt_init

#endif
