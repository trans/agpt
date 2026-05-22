#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

#include "checkpoint_io_v2.cuh"
#include "io.cuh"
#include "model_layout.cuh"

namespace agpt_v2 {

static void usage() {
    std::fprintf(stderr,
                 "usage: agpt-seed --trie-dir DIR --save FILE "
                 "[--d-model N] [--n-heads N] [--n-layers N] [--d-ff N] [--seed N]\n");
    std::exit(1);
}

static void zero_slice(float* weights, int offset, int count) {
    std::memset(weights + offset, 0, (size_t)count * sizeof(float));
}

static void fill_normal_slice(float* weights,
                              int offset,
                              int count,
                              std::mt19937& rng,
                              std::normal_distribution<float>& nd) {
    for (int i = 0; i < count; i++) {
        weights[offset + i] = nd(rng);
    }
}

static void fill_scalar_slice(float* weights, int offset, int count, float value) {
    for (int i = 0; i < count; i++) {
        weights[offset + i] = value;
    }
}

static void init_random_weights(float* weights, const ModelLayout& layout, int seed) {
    const int L = layout.shape.n_layers;
    const int D = layout.shape.d_model;
    const int F = layout.shape.d_ff;
    const int V = layout.shape.vocab_size;

    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 0.02f);

    fill_normal_slice(weights, layout.token_emb, V * D, rng, nd);
    for (int i = 0; i < L; i++) {
        fill_normal_slice(weights, layout.wq_w[i], D * D, rng, nd);
        zero_slice(weights, layout.wq_b[i], D);
        fill_normal_slice(weights, layout.wk_w[i], D * D, rng, nd);
        zero_slice(weights, layout.wk_b[i], D);
        fill_normal_slice(weights, layout.wv_w[i], D * D, rng, nd);
        zero_slice(weights, layout.wv_b[i], D);
        fill_normal_slice(weights, layout.wo_w[i], D * D, rng, nd);
        zero_slice(weights, layout.wo_b[i], D);

        fill_scalar_slice(weights, layout.ln1_gamma[i], D, 1.0f);
        zero_slice(weights, layout.ln1_beta[i], D);

        fill_normal_slice(weights, layout.l1_w[i], D * F, rng, nd);
        zero_slice(weights, layout.l1_b[i], F);
        fill_normal_slice(weights, layout.l2_w[i], F * D, rng, nd);
        zero_slice(weights, layout.l2_b[i], D);

        fill_scalar_slice(weights, layout.ln2_gamma[i], D, 1.0f);
        zero_slice(weights, layout.ln2_beta[i], D);
    }

    fill_scalar_slice(weights, layout.final_gamma, D, 1.0f);
    zero_slice(weights, layout.final_beta, D);
    fill_normal_slice(weights, layout.out_w, D * V, rng, nd);
    zero_slice(weights, layout.out_b, V);
}

}  // namespace agpt_v2

int main(int argc, char** argv) {
    using namespace agpt_v2;

    const char* trie_dir = nullptr;
    const char* save_path = nullptr;
    int d_model = 64;
    int n_heads = 4;
    int n_layers = 2;
    int d_ff = 256;
    int seed = 42;

    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "--trie-dir") && i + 1 < argc) {
            trie_dir = argv[++i];
        } else if (!std::strcmp(argv[i], "--save") && i + 1 < argc) {
            save_path = argv[++i];
        } else if (!std::strcmp(argv[i], "--d-model") && i + 1 < argc) {
            d_model = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--n-heads") && i + 1 < argc) {
            n_heads = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--n-layers") && i + 1 < argc) {
            n_layers = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--d-ff") && i + 1 < argc) {
            d_ff = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) {
            seed = std::atoi(argv[++i]);
        } else {
            usage();
        }
    }

    if (!trie_dir || !save_path) usage();
    if (d_model <= 0 || n_heads <= 0 || n_layers <= 0 || d_ff <= 0) {
        std::fprintf(stderr, "agpt-seed: model dimensions must be positive\n");
        return 1;
    }
    if (d_model % n_heads != 0) {
        std::fprintf(stderr, "agpt-seed: d_model (%d) must be divisible by n_heads (%d)\n",
                     d_model, n_heads);
        return 1;
    }

    RadixMetaSummary meta = load_radix_meta_summary(trie_dir);
    if (meta.depth_file_count <= 1) {
        std::fprintf(stderr, "agpt-seed: trie depth_file_count too small: %d\n", meta.depth_file_count);
        return 1;
    }
    if (meta.vocab_size <= 0) {
        std::fprintf(stderr, "agpt-seed: trie vocab_size too small: %d\n", meta.vocab_size);
        return 1;
    }

    RuntimeShape shape;
    shape.d_model = d_model;
    shape.n_heads = n_heads;
    shape.n_layers = n_layers;
    shape.d_ff = d_ff;
    shape.vocab_size = meta.vocab_size;
    shape.seq_len = meta.depth_file_count - 1;
    shape.head_dim = d_model / n_heads;

    ModelLayout layout = make_model_layout(shape);
    float* weights = (float*)std::malloc((size_t)layout.total_floats * sizeof(float));
    if (!weights) {
        std::fprintf(stderr, "agpt-seed: failed to allocate %zu bytes for weights\n",
                     (size_t)layout.total_floats * sizeof(float));
        return 1;
    }

    init_random_weights(weights, layout, seed);
    save_model_weights_v2(save_path, layout, weights);

    std::printf("agpt-seed: wrote %s\n", save_path);
    std::printf("  d_model=%d n_heads=%d n_layers=%d d_ff=%d vocab=%d seq_len=%d seed=%d\n",
                shape.d_model, shape.n_heads, shape.n_layers, shape.d_ff,
                shape.vocab_size, shape.seq_len, seed);

    std::free(weights);
    free_model_layout(layout);
    return 0;
}
