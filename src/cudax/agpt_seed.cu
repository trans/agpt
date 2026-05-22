#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "checkpoint_io_v2.cuh"
#include "io.cuh"
#include "model_layout.cuh"
#include "../common/init_weights.h"

namespace agpt_v2 {

static void usage() {
    std::fprintf(stderr,
                 "usage: agpt-seed --trie-dir DIR --save FILE "
                 "[--d-model N] [--n-heads N] [--n-layers N] [--d-ff N] [--seed N]\n");
    std::exit(1);
}

// Adapter to the shared agpt_init::init_random_weights().  v2's
// ModelLayout and v1's WeightOffsets carry the same offset structure
// (verified byte-identical), so both bind into agpt_init::InitLayout
// the same way and produce identical seed bytes for matching dims.
static void init_random_weights(float* weights, const ModelLayout& layout, int seed) {
    const int L = layout.shape.n_layers;
    std::vector<agpt_init::LayerOffsets> layers(L);
    for (int i = 0; i < L; i++) {
        layers[i] = {
            layout.wq_w[i], layout.wq_b[i],
            layout.wk_w[i], layout.wk_b[i],
            layout.wv_w[i], layout.wv_b[i],
            layout.wo_w[i], layout.wo_b[i],
            layout.ln1_gamma[i], layout.ln1_beta[i],
            layout.l1_w[i], layout.l1_b[i],
            layout.l2_w[i], layout.l2_b[i],
            layout.ln2_gamma[i], layout.ln2_beta[i],
        };
    }
    agpt_init::InitLayout il = {
        layout.shape.d_model,
        layout.shape.d_ff,
        layout.shape.vocab_size,
        L,
        layout.token_emb,
        layers.data(),
        layout.final_gamma, layout.final_beta,
        layout.out_w, layout.out_b,
    };
    agpt_init::init_random_weights(weights, il, (uint32_t)seed);
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
