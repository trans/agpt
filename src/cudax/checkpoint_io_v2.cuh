#ifndef AGPT_V2_CHECKPOINT_IO_CUH
#define AGPT_V2_CHECKPOINT_IO_CUH

#include <cstdio>
#include <cstdlib>

#include "io.cuh"
#include "model_layout.cuh"

namespace agpt_v2 {

static inline float* load_model_weights_v2(const char* path, const ModelLayout& layout) {
    FILE* f = std::fopen(path, "rb");
    if (!f) {
        std::fprintf(stderr, "agpt_train_v2: cannot open model: %s\n", path);
        std::exit(1);
    }

    unsigned magic = read_u32(f);
    if (magic != MGPT_MAGIC) {
        std::fprintf(stderr, "agpt_train_v2: bad model magic in %s\n", path);
        std::exit(1);
    }

    RuntimeShape file_shape;
    file_shape.d_model = read_i32(f);
    file_shape.n_heads = read_i32(f);
    file_shape.n_layers = read_i32(f);
    file_shape.d_ff = read_i32(f);
    file_shape.vocab_size = read_i32(f);
    file_shape.seq_len = read_i32(f);
    file_shape.head_dim = file_shape.d_model / file_shape.n_heads;

    if (file_shape.d_model != layout.shape.d_model ||
        file_shape.n_heads != layout.shape.n_heads ||
        file_shape.n_layers != layout.shape.n_layers ||
        file_shape.d_ff != layout.shape.d_ff ||
        file_shape.vocab_size != layout.shape.vocab_size) {
        std::fprintf(stderr, "agpt_train_v2: model header mismatch while loading weights\n");
        std::exit(1);
    }

    float* weights = (float*)std::malloc((size_t)layout.total_floats * sizeof(float));
    int n_mats = 1 + layout.shape.n_layers * 16 + 4;
    int offset = 0;
    for (int m = 0; m < n_mats; m++) {
        int rows = read_i32(f);
        int cols = read_i32(f);
        int count = rows * cols;
        if (std::fread(&weights[offset], sizeof(float), count, f) != (size_t)count) {
            std::fprintf(stderr, "agpt_train_v2: failed reading weights from %s\n", path);
            std::exit(1);
        }
        offset += count;
    }
    std::fclose(f);

    if (offset != layout.total_floats) {
        std::fprintf(stderr, "agpt_train_v2: weight count mismatch read=%d expected=%d\n",
                     offset, layout.total_floats);
        std::exit(1);
    }
    return weights;
}

static inline void save_model_weights_v2(const char* path,
                                         const ModelLayout& layout,
                                         const float* weights) {
    FILE* f = std::fopen(path, "wb");
    if (!f) {
        std::fprintf(stderr, "agpt_train_v2: cannot write model: %s\n", path);
        std::exit(1);
    }

    unsigned magic = MGPT_MAGIC;
    std::fwrite(&magic, 4, 1, f);
    std::fwrite(&layout.shape.d_model, 4, 1, f);
    std::fwrite(&layout.shape.n_heads, 4, 1, f);
    std::fwrite(&layout.shape.n_layers, 4, 1, f);
    std::fwrite(&layout.shape.d_ff, 4, 1, f);
    std::fwrite(&layout.shape.vocab_size, 4, 1, f);
    std::fwrite(&layout.shape.seq_len, 4, 1, f);

    auto write_mat = [&](int offset, int rows, int cols) {
        std::fwrite(&rows, 4, 1, f);
        std::fwrite(&cols, 4, 1, f);
        std::fwrite(&weights[offset], sizeof(float), (size_t)rows * (size_t)cols, f);
    };

    int L = layout.shape.n_layers;
    int D = layout.shape.d_model;
    int F = layout.shape.d_ff;
    int V = layout.shape.vocab_size;
    write_mat(layout.token_emb, V, D);
    for (int i = 0; i < L; i++) {
        write_mat(layout.wq_w[i], D, D); write_mat(layout.wq_b[i], 1, D);
        write_mat(layout.wk_w[i], D, D); write_mat(layout.wk_b[i], 1, D);
        write_mat(layout.wv_w[i], D, D); write_mat(layout.wv_b[i], 1, D);
        write_mat(layout.wo_w[i], D, D); write_mat(layout.wo_b[i], 1, D);
        write_mat(layout.ln1_gamma[i], 1, D); write_mat(layout.ln1_beta[i], 1, D);
        write_mat(layout.l1_w[i], D, F); write_mat(layout.l1_b[i], 1, F);
        write_mat(layout.l2_w[i], F, D); write_mat(layout.l2_b[i], 1, D);
        write_mat(layout.ln2_gamma[i], 1, D); write_mat(layout.ln2_beta[i], 1, D);
    }
    write_mat(layout.final_gamma, 1, D); write_mat(layout.final_beta, 1, D);
    write_mat(layout.out_w, D, V); write_mat(layout.out_b, 1, V);
    std::fclose(f);
}

static inline void save_optimizer_state_v2(const char* path,
                                           const float* opt_v,
                                           int total_floats) {
    FILE* f = std::fopen(path, "wb");
    if (!f) {
        std::fprintf(stderr, "agpt_train_v2: cannot write optimizer state: %s\n", path);
        std::exit(1);
    }
    unsigned magic = 0x5654504f;  // "OPTV"
    std::fwrite(&magic, 4, 1, f);
    std::fwrite(&total_floats, 4, 1, f);
    std::fwrite(opt_v, sizeof(float), (size_t)total_floats, f);
    std::fclose(f);
}

static inline float* load_optimizer_state_v2(const char* path, int expected_total_floats) {
    FILE* f = std::fopen(path, "rb");
    if (!f) {
        std::fprintf(stderr, "agpt_train_v2: cannot open optimizer state: %s\n", path);
        std::exit(1);
    }
    unsigned magic = read_u32(f);
    if (magic != 0x5654504f) {
        std::fprintf(stderr, "agpt_train_v2: bad optimizer-state magic in %s\n", path);
        std::exit(1);
    }
    int total_floats = read_i32(f);
    if (total_floats != expected_total_floats) {
        std::fprintf(stderr, "agpt_train_v2: optimizer-state size mismatch read=%d expected=%d\n",
                     total_floats, expected_total_floats);
        std::exit(1);
    }
    float* opt_v = (float*)std::malloc((size_t)total_floats * sizeof(float));
    if (std::fread(opt_v, sizeof(float), (size_t)total_floats, f) != (size_t)total_floats) {
        std::fprintf(stderr, "agpt_train_v2: failed reading optimizer state from %s\n", path);
        std::exit(1);
    }
    std::fclose(f);
    return opt_v;
}

}  // namespace agpt_v2

#endif
