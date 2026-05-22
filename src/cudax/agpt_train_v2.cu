#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include "types.cuh"
#include "io.cuh"
#include "model_layout.cuh"
#include "cache_layout.cuh"
#include "training_unit.cuh"
#include "chunk_plan.cuh"
#include "chunk_metadata_v2.cuh"
#include "chunk_upload_v2.cuh"
#include "execution_plan.cuh"
#include "runtime_contracts.cuh"
#include "runtime_objects.cuh"
#include "forward_pass.cuh"
#include "backward_pass.cuh"
#include "optimizer_step.cuh"

namespace {

enum class V2Mode {
    Plan,
    InstantiateRuntime,
    Upload,
    Forward,
    BackwardHead,
    OneStepSgd,
    OneStepRmsprop,
    MultiStepSgd,
    MultiStepRmsprop,
    SaveReloadSgd,
    SaveReloadRmsprop,
    TrainEpoch,
    TrainSmall,
};

static const char* v2_mode_name(V2Mode mode) {
    switch (mode) {
        case V2Mode::Plan: return "plan";
        case V2Mode::InstantiateRuntime: return "instantiate-runtime";
        case V2Mode::Upload: return "upload";
        case V2Mode::Forward: return "forward";
        case V2Mode::BackwardHead: return "backward-head";
        case V2Mode::OneStepSgd: return "one-step-sgd";
        case V2Mode::OneStepRmsprop: return "one-step-rmsprop";
        case V2Mode::MultiStepSgd: return "multi-step-sgd";
        case V2Mode::MultiStepRmsprop: return "multi-step-rmsprop";
        case V2Mode::SaveReloadSgd: return "save-reload-sgd";
        case V2Mode::SaveReloadRmsprop: return "save-reload-rmsprop";
        case V2Mode::TrainEpoch: return "train-epoch";
        case V2Mode::TrainSmall: return "train-small";
    }
    return "unknown";
}

static bool parse_v2_mode(const char* text, V2Mode& out) {
    if (std::strcmp(text, "plan") == 0) {
        out = V2Mode::Plan;
        return true;
    }
    if (std::strcmp(text, "instantiate-runtime") == 0) {
        out = V2Mode::InstantiateRuntime;
        return true;
    }
    if (std::strcmp(text, "upload") == 0) {
        out = V2Mode::Upload;
        return true;
    }
    if (std::strcmp(text, "forward") == 0) {
        out = V2Mode::Forward;
        return true;
    }
    if (std::strcmp(text, "backward-head") == 0) {
        out = V2Mode::BackwardHead;
        return true;
    }
    if (std::strcmp(text, "one-step-sgd") == 0) {
        out = V2Mode::OneStepSgd;
        return true;
    }
    if (std::strcmp(text, "one-step-rmsprop") == 0 || std::strcmp(text, "rmsprop") == 0) {
        out = V2Mode::OneStepRmsprop;
        return true;
    }
    if (std::strcmp(text, "multi-step-sgd") == 0) {
        out = V2Mode::MultiStepSgd;
        return true;
    }
    if (std::strcmp(text, "multi-step-rmsprop") == 0) {
        out = V2Mode::MultiStepRmsprop;
        return true;
    }
    if (std::strcmp(text, "save-reload-sgd") == 0 || std::strcmp(text, "save-reload") == 0) {
        out = V2Mode::SaveReloadSgd;
        return true;
    }
    if (std::strcmp(text, "save-reload-rmsprop") == 0 || std::strcmp(text, "save-reload-opt") == 0) {
        out = V2Mode::SaveReloadRmsprop;
        return true;
    }
    if (std::strcmp(text, "train-epoch") == 0) {
        out = V2Mode::TrainEpoch;
        return true;
    }
    if (std::strcmp(text, "train-small") == 0) {
        out = V2Mode::TrainSmall;
        return true;
    }
    return false;
}

static const char* v2_lr_schedule_name(agpt_v2::LrSchedule schedule) {
    switch (schedule) {
        case agpt_v2::LrSchedule::Constant: return "constant";
        case agpt_v2::LrSchedule::WarmupCosine: return "warmup-cosine";
    }
    return "unknown";
}

static bool parse_lr_schedule(const char* text, agpt_v2::LrSchedule& out) {
    if (std::strcmp(text, "constant") == 0) {
        out = agpt_v2::LrSchedule::Constant;
        return true;
    }
    if (std::strcmp(text, "warmup-cosine") == 0 || std::strcmp(text, "warmup_cosine") == 0) {
        out = agpt_v2::LrSchedule::WarmupCosine;
        return true;
    }
    return false;
}

static float scheduled_lr(const agpt_v2::TrainerConfig& cfg,
                          long long step_index,
                          long long total_steps,
                          long long warmup_steps) {
    if (cfg.lr_schedule == agpt_v2::LrSchedule::Constant || total_steps <= 1) {
        return cfg.lr;
    }
    if (warmup_steps < 0) warmup_steps = 0;
    if (warmup_steps > total_steps) warmup_steps = total_steps;
    if (warmup_steps > 0 && step_index < warmup_steps) {
        float scale = (float)(step_index + 1) / (float)warmup_steps;
        if (scale < 0.0f) scale = 0.0f;
        if (scale > 1.0f) scale = 1.0f;
        return cfg.lr * scale;
    }
    long long decay_steps = total_steps - warmup_steps;
    if (decay_steps <= 0) {
        return cfg.lr;
    }
    float progress = (float)(step_index - warmup_steps) / (float)decay_steps;
    if (progress < 0.0f) progress = 0.0f;
    if (progress > 1.0f) progress = 1.0f;
    constexpr float kPi = 3.14159265358979323846f;
    float cosine = 0.5f * (1.0f + std::cos(kPi * progress));
    return cfg.lr * cosine;
}

static void scale_gradients_for_fire(cublasHandle_t cublas,
                                     float* d_grads,
                                     int total_floats,
                                     long long fire_events) {
    if (fire_events <= 0) return;
    float inv_n = 1.0f / (float)fire_events;
    AGPT_V2_CUBLAS_CHECK(cublasSscal(cublas, total_floats, &inv_n, d_grads, 1));
}

struct FireDiagOptionsV2 {
    const char* path = nullptr;
    int epoch = 0;
    int root_id = -1;
    bool exit_after = false;
};

struct FireDiagBlockV2 {
    const char* name = nullptr;
    int offset = 0;
    int length = 0;
};

static FireDiagOptionsV2 read_fire_diag_options_v2() {
    FireDiagOptionsV2 opts;
    opts.path = std::getenv("AGPT_DIAG_FIRE_PATH");
    const char* epoch = std::getenv("AGPT_DIAG_FIRE_EPOCH");
    const char* root_id = std::getenv("AGPT_DIAG_FIRE_ROOT_ID");
    const char* exit_after = std::getenv("AGPT_DIAG_FIRE_EXIT_AFTER");
    if (epoch) opts.epoch = std::atoi(epoch);
    if (root_id) opts.root_id = std::atoi(root_id);
    if (exit_after && exit_after[0] && std::strcmp(exit_after, "0") != 0) opts.exit_after = true;
    if (!opts.path || !opts.path[0]) opts.path = nullptr;
    if (opts.epoch <= 0 || opts.root_id < 0) opts.path = nullptr;
    return opts;
}

static double l2_norm_host_v2(const float* data, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += (double)data[i] * (double)data[i];
    return std::sqrt(sum);
}

static double l2_diff_host_v2(const float* a, const float* b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double d = (double)b[i] - (double)a[i];
        sum += d * d;
    }
    return std::sqrt(sum);
}

static void copy_device_floats_v2(float* h_dst, const float* d_src, int n) {
    AGPT_V2_CUDA_CHECK(cudaMemcpy(h_dst, d_src, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost));
}

static void dump_fire_diag_v2(FILE* f,
                              const char* phase,
                              const FireDiagBlockV2* blocks,
                              int block_count,
                              const float* whole_a,
                              const float* whole_b,
                              const float* opt_v,
                              int total_floats) {
    if (std::strcmp(phase, "post_step") == 0) {
        std::fprintf(f, "phase=%s delta_w_total_l2=%.9f opt_v_total_l2=%.9f\n",
                     phase, l2_diff_host_v2(whole_a, whole_b, total_floats), l2_norm_host_v2(opt_v, total_floats));
        for (int i = 0; i < block_count; i++) {
            const FireDiagBlockV2& b = blocks[i];
            std::fprintf(f, "phase=%s block=%s delta_w_l2=%.9f opt_v_l2=%.9f\n",
                         phase, b.name,
                         l2_diff_host_v2(whole_a + b.offset, whole_b + b.offset, b.length),
                         l2_norm_host_v2(opt_v + b.offset, b.length));
        }
    } else {
        std::fprintf(f, "phase=%s grads_total_l2=%.9f\n", phase, l2_norm_host_v2(whole_a, total_floats));
        for (int i = 0; i < block_count; i++) {
            const FireDiagBlockV2& b = blocks[i];
            std::fprintf(f, "phase=%s block=%s grads_l2=%.9f\n",
                         phase, b.name, l2_norm_host_v2(whole_a + b.offset, b.length));
        }
    }
}

static void dump_fire_state_v2(FILE* f,
                               const char* phase,
                               const FireDiagBlockV2* blocks,
                               int block_count,
                               const float* weights,
                               const float* opt_v,
                               int total_floats) {
    std::fprintf(f, "phase=%s weights_total_l2=%.9f opt_v_total_l2=%.9f\n",
                 phase, l2_norm_host_v2(weights, total_floats), l2_norm_host_v2(opt_v, total_floats));
    for (int i = 0; i < block_count; i++) {
        const FireDiagBlockV2& b = blocks[i];
        std::fprintf(f, "phase=%s block=%s weights_l2=%.9f opt_v_l2=%.9f\n",
                     phase, b.name,
                     l2_norm_host_v2(weights + b.offset, b.length),
                     l2_norm_host_v2(opt_v + b.offset, b.length));
    }
}

}  // namespace

int main(int argc, char** argv) {
    agpt_v2::TrainerConfig cfg;
    const char* model_path = nullptr;
    const char* trie_dir = nullptr;
    const char* save_path = nullptr;
    V2Mode mode = V2Mode::Plan;
    int steps = 3;
    int unit_limit = 0;

    cfg.epochs = 1;
    cfg.partition_depth = 1;
    cfg.chunk_queries = 50000;
    cfg.lr = 3e-4f;
    cfg.lr_schedule = agpt_v2::LrSchedule::Constant;
    cfg.warmup_epochs = 0;
    cfg.accumulate = true;

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (std::strcmp(argv[i], "--trie-dir") == 0 && i + 1 < argc) trie_dir = argv[++i];
        else if (std::strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) cfg.epochs = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--partition-depth") == 0 && i + 1 < argc) cfg.partition_depth = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--chunk-queries") == 0 && i + 1 < argc) cfg.chunk_queries = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--lr") == 0 && i + 1 < argc) cfg.lr = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--lr-schedule") == 0 && i + 1 < argc) {
            if (!parse_lr_schedule(argv[++i], cfg.lr_schedule)) {
                std::fprintf(stderr, "agpt_train_v2: unsupported --lr-schedule value: %s\n", argv[i]);
                return 1;
            }
        }
        else if (std::strcmp(argv[i], "--warmup-epochs") == 0 && i + 1 < argc) cfg.warmup_epochs = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--anc-grad") == 0) cfg.anc_grad = true;
        else if (std::strcmp(argv[i], "--save") == 0 && i + 1 < argc) save_path = argv[++i];
        else if (std::strcmp(argv[i], "--steps") == 0 && i + 1 < argc) steps = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--units") == 0 && i + 1 < argc) unit_limit = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            if (!parse_v2_mode(argv[++i], mode)) {
                std::fprintf(stderr, "agpt_train_v2: unsupported --mode value: %s\n", argv[i]);
                return 1;
            }
        }
        else if (std::strcmp(argv[i], "--accumulate") == 0) cfg.accumulate = true;
        else if (std::strcmp(argv[i], "--no-accumulate") == 0) cfg.accumulate = false;
        else if (std::strcmp(argv[i], "--quiet") == 0) cfg.quiet = true;
        else if (std::strcmp(argv[i], "--instantiate-runtime") == 0) mode = V2Mode::InstantiateRuntime;
        else if (std::strcmp(argv[i], "--instantiate-chunk-upload") == 0) mode = V2Mode::Upload;
        else if (std::strcmp(argv[i], "--run-forward-prefix") == 0) mode = V2Mode::Forward;
        else if (std::strcmp(argv[i], "--run-backward-head") == 0) mode = V2Mode::BackwardHead;
        else {
            std::fprintf(stderr, "agpt_train_v2: unknown or unsupported arg: %s\n", argv[i]);
            return 1;
        }
    }

    if (!model_path || !trie_dir) {
        std::fprintf(stderr,
                     "Usage: agpt_train_v2 --model <path> --trie-dir <path>\n"
                     "  [--epochs N] [--partition-depth 1] [--chunk-queries N] [--lr F] [--lr-schedule constant|warmup-cosine]\n"
                     "  [--warmup-epochs N] [--steps N]\n"
                     "  [--anc-grad]\n"
                     "  [--units N]\n"
                     "  [--save PATH]\n"
                     "  [--mode plan|instantiate-runtime|upload|forward|backward-head|one-step-sgd|one-step-rmsprop|multi-step-sgd|multi-step-rmsprop|save-reload-sgd|save-reload-rmsprop|train-epoch|train-small]\n"
                     "  [--accumulate|--no-accumulate] [--quiet]\n"
                     "  compatibility aliases: [--instantiate-runtime] [--instantiate-chunk-upload]\n"
                         "                         [--run-forward-prefix] [--run-backward-head]\n");
        return 1;
    }
    if (cfg.partition_depth != 1) {
        std::fprintf(stderr,
                     "agpt_train_v2: only --partition-depth 1 is supported in the v2 baseline planner\n");
        return 1;
    }
    if (cfg.chunk_queries <= 0) cfg.chunk_queries = 50000;
    if (steps <= 0) steps = 3;

    agpt_v2::ModelHeader header = agpt_v2::load_model_header(model_path);
    agpt_v2::RuntimeShape shape = header.shape;
    cfg.d_model = shape.d_model;
    cfg.n_heads = shape.n_heads;
    cfg.n_layers = shape.n_layers;
    cfg.d_ff = shape.d_ff;
    cfg.vocab_size = shape.vocab_size;
    cfg.seq_len = shape.seq_len;

    agpt_v2::ModelLayout model = agpt_v2::make_model_layout(shape);
    agpt_v2::CacheLayout cache = agpt_v2::make_cache_layout(shape);
    agpt_v2::RadixTrieStructure trie = agpt_v2::load_radix_structure_minimal(trie_dir);
    agpt_v2::TrainingPlan training_plan = agpt_v2::build_pd1_training_plan(trie);
    FireDiagOptionsV2 fire_diag = read_fire_diag_options_v2();
    if (fire_diag.path) std::remove(fire_diag.path);
    agpt_v2::ExecutionPlan plan = agpt_v2::build_execution_plan(trie, training_plan, cfg.chunk_queries);
    agpt_v2::ChunkPlanList largest_chunks = {};
    if (plan.largest_by_queries) {
        largest_chunks = agpt_v2::build_chunk_plan_for_unit(trie, *plan.largest_by_queries, cfg.chunk_queries);
    }
    agpt_v2::TrainerRuntimeContract runtime_contract =
        agpt_v2::build_trainer_runtime_contract(shape, cache, plan, largest_chunks);
    agpt_v2::ChunkMetadataV2 first_chunk_meta{};
    bool have_first_chunk_meta = false;
    if (plan.largest_by_queries && largest_chunks.chunk_count > 0) {
        first_chunk_meta = agpt_v2::build_chunk_metadata_v2(cfg, shape, trie, *plan.largest_by_queries, largest_chunks.chunks[0]);
        have_first_chunk_meta = true;
    }

    std::printf("AGPT CUDA Trainer V2\n");
    std::printf("  mode: %s\n", v2_mode_name(mode));
    std::printf("  model: d=%d heads=%d layers=%d ff=%d vocab=%d seq=%d head_dim=%d\n",
                shape.d_model, shape.n_heads, shape.n_layers, shape.d_ff,
                shape.vocab_size, shape.seq_len, shape.head_dim);
    std::printf("  trie: %d radix nodes, %lld edge chars, %d endpoint depths\n",
                trie.radix_count, trie.total_edge_chars, trie.depth_file_count);
    std::printf("  config: epochs=%d lr=%.6f schedule=%s warmup_epochs=%d partition_depth=%d chunk_queries=%d accumulate=%s\n",
                cfg.epochs, cfg.lr, v2_lr_schedule_name(cfg.lr_schedule), cfg.warmup_epochs,
                cfg.partition_depth, cfg.chunk_queries, cfg.accumulate ? "true" : "false");
    if (cfg.anc_grad) {
        std::printf("  anc-grad: enabled (descendant->ancestor scatter into Wk/Wv)\n");
    }
    std::printf("  cache contract: K=%s compact_slot_indexed=%s\n",
                cache.k_space == agpt_v2::KCoordinateSpace::PostRope ? "post-RoPE" : "pre-RoPE",
                cache.compact_slot_indexed ? "true" : "false");
    std::printf("  pd=1 plan: %d root-child training units, %lld node-visits, %lld query positions,\n"
                "             %lld compact chars, ~%lld chunks/epoch at chunk_queries=%d\n",
                plan.training_unit_count, plan.total_node_count, plan.total_query_count,
                plan.total_compact_char_count, plan.estimated_chunk_count, cfg.chunk_queries);
    if (plan.largest_by_queries) {
        std::printf("  largest-by-query unit: rc=%d nodes=%d queries=%lld compact_chars=%lld depth=%d est_chunks=%d\n",
                    plan.largest_by_queries->root_child_id,
                    plan.largest_by_queries->node_count,
                    plan.largest_by_queries->query_count,
                    plan.largest_by_queries->compact_char_count,
                    plan.largest_by_queries->max_endpoint_depth,
                    largest_chunks.chunk_count);
    }
    if (plan.largest_by_compact_chars) {
        agpt_v2::ChunkPlanList compact_chunks =
            agpt_v2::build_chunk_plan_for_unit(trie, *plan.largest_by_compact_chars, cfg.chunk_queries);
        std::printf("  largest-by-compact unit: rc=%d nodes=%d queries=%lld compact_chars=%lld depth=%d est_chunks=%d\n",
                    plan.largest_by_compact_chars->root_child_id,
                    plan.largest_by_compact_chars->node_count,
                    plan.largest_by_compact_chars->query_count,
                    plan.largest_by_compact_chars->compact_char_count,
                    plan.largest_by_compact_chars->max_endpoint_depth,
                    compact_chunks.chunk_count);
        agpt_v2::free_chunk_plan_list(compact_chunks);
    }
    if (largest_chunks.chunk_count > 0) {
        int preview = largest_chunks.chunk_count < 3 ? largest_chunks.chunk_count : 3;
        std::printf("  largest unit chunk preview:\n");
        for (int i = 0; i < preview; i++) {
            const agpt_v2::ChunkPlan& chunk = largest_chunks.chunks[i];
            std::printf("    chunk %d: node_range=[%d,%d) nodes=%d queries=%lld compact_chars=%lld\n",
                        chunk.chunk_index, chunk.start_node_index, chunk.end_node_index,
                        chunk.node_count, chunk.query_count, chunk.compact_char_count);
        }
    }
    if (have_first_chunk_meta) {
        std::printf("  first chunk metadata: N=%d T_q=%d T_kv=%d T_anc=%d max_kv_len=%d\n",
                    first_chunk_meta.N, first_chunk_meta.T_q, first_chunk_meta.T_kv,
                    first_chunk_meta.T_anc, first_chunk_meta.max_kv_len);
    }
    std::printf("  runtime contract:\n");
    std::printf("    cache: compact_chars=%lld layers=%d d_model=%d total=%.1f MB (%s K, managed=%s)\n",
                runtime_contract.cache.compact_char_capacity,
                runtime_contract.cache.layer_count,
                runtime_contract.cache.d_model,
                (double)runtime_contract.cache.total_bytes / 1.0e6,
                runtime_contract.cache.k_space == agpt_v2::KCoordinateSpace::PostRope ? "post-RoPE" : "pre-RoPE",
                runtime_contract.cache.uses_managed_memory ? "true" : "false");
    std::printf("    chunk: query_cap=%d node_cap=%d kv_cap=%lld max_kv_len=%d total=%.1f MB\n",
                runtime_contract.chunk.query_capacity,
                runtime_contract.chunk.node_capacity,
                runtime_contract.chunk.kv_capacity,
                runtime_contract.chunk.max_kv_len,
                (double)runtime_contract.chunk.total_bytes / 1.0e6);
    std::printf("    params+grads: %.1f MB  optimizer: %.1f MB  combined-estimate: %.1f MB\n",
                (double)runtime_contract.weight_and_grad_bytes / 1.0e6,
                (double)runtime_contract.optimizer_state_bytes / 1.0e6,
                (double)runtime_contract.total_bytes / 1.0e6);
    bool instantiate_runtime = (mode != V2Mode::Plan);
    bool instantiate_chunk_upload =
        (mode == V2Mode::Upload || mode == V2Mode::Forward ||
         mode == V2Mode::BackwardHead || mode == V2Mode::OneStepSgd ||
        mode == V2Mode::OneStepRmsprop || mode == V2Mode::MultiStepSgd ||
         mode == V2Mode::MultiStepRmsprop || mode == V2Mode::SaveReloadSgd ||
         mode == V2Mode::SaveReloadRmsprop || mode == V2Mode::TrainEpoch ||
         mode == V2Mode::TrainSmall);
    bool run_forward_prefix =
        (mode == V2Mode::Forward || mode == V2Mode::BackwardHead || mode == V2Mode::OneStepSgd ||
         mode == V2Mode::OneStepRmsprop || mode == V2Mode::MultiStepSgd ||
         mode == V2Mode::MultiStepRmsprop || mode == V2Mode::SaveReloadSgd ||
         mode == V2Mode::SaveReloadRmsprop || mode == V2Mode::TrainEpoch ||
         mode == V2Mode::TrainSmall);
    bool run_backward_head =
        (mode == V2Mode::BackwardHead || mode == V2Mode::OneStepSgd ||
         mode == V2Mode::OneStepRmsprop || mode == V2Mode::MultiStepSgd ||
         mode == V2Mode::MultiStepRmsprop || mode == V2Mode::SaveReloadSgd ||
         mode == V2Mode::SaveReloadRmsprop || mode == V2Mode::TrainEpoch ||
         mode == V2Mode::TrainSmall);
    bool run_one_step_sgd = (mode == V2Mode::OneStepSgd);
    bool run_one_step_rmsprop = (mode == V2Mode::OneStepRmsprop);
    bool run_multi_step_sgd = (mode == V2Mode::MultiStepSgd);
    bool run_multi_step_rmsprop = (mode == V2Mode::MultiStepRmsprop);
    bool run_save_reload_sgd = (mode == V2Mode::SaveReloadSgd);
    bool run_save_reload_rmsprop = (mode == V2Mode::SaveReloadRmsprop);
    bool run_train_epoch = (mode == V2Mode::TrainEpoch);
    bool run_train_small = (mode == V2Mode::TrainSmall);

    int* d_counts_offset = nullptr;
    int* d_counts_tok = nullptr;
    int* d_counts_val = nullptr;
    if (instantiate_runtime) {
        agpt_v2::TrainerRuntimeV2 runtime{};
        init_trainer_runtime_v2(runtime, runtime_contract, trie);
        agpt_v2::zero_cache_runtime_v2(runtime.cache);
        float* h_weights = agpt_v2::load_model_weights_v2(model_path, model);
        AGPT_V2_CUDA_CHECK(cudaMemcpy(runtime.d_weights, h_weights,
                                      (size_t)model.total_floats * sizeof(float),
                                      cudaMemcpyHostToDevice));
        AGPT_V2_CUDA_CHECK(cudaMalloc(&d_counts_offset, (size_t)(trie.radix_count + 1) * sizeof(int)));
        AGPT_V2_CUDA_CHECK(cudaMemcpy(d_counts_offset, trie.counts_offset,
                                      (size_t)(trie.radix_count + 1) * sizeof(int),
                                      cudaMemcpyHostToDevice));
        AGPT_V2_CUDA_CHECK(cudaMalloc(&d_counts_tok, (size_t)(trie.total_counts > 0 ? trie.total_counts : 1) * sizeof(int)));
        AGPT_V2_CUDA_CHECK(cudaMalloc(&d_counts_val, (size_t)(trie.total_counts > 0 ? trie.total_counts : 1) * sizeof(int)));
        if (trie.total_counts > 0) {
            AGPT_V2_CUDA_CHECK(cudaMemcpy(d_counts_tok, trie.counts_tok,
                                          (size_t)trie.total_counts * sizeof(int),
                                          cudaMemcpyHostToDevice));
            AGPT_V2_CUDA_CHECK(cudaMemcpy(d_counts_val, trie.counts_val,
                                          (size_t)trie.total_counts * sizeof(int),
                                          cudaMemcpyHostToDevice));
        }
        std::printf("  runtime objects: instantiated successfully\n");
        if (instantiate_chunk_upload && have_first_chunk_meta) {
            agpt_v2::ChunkUploadRuntimeV2 upload{};
            init_chunk_upload_runtime_v2(upload, runtime_contract.chunk.node_capacity,
                                         runtime_contract.chunk.query_capacity, shape.n_heads);
            agpt_v2::ChunkDeviceMetadataV2 device_meta = upload_chunk_metadata_v2(first_chunk_meta, upload);
            (void)device_meta;
            std::printf("  chunk upload: first chunk uploaded successfully\n");
            if (run_train_epoch) {
                agpt_v2::LossTablesV2 loss_tables{d_counts_offset, d_counts_tok, d_counts_val};
                int epochs = cfg.epochs > 0 ? cfg.epochs : 1;
                int units_to_run = plan.training_unit_count;
                if (unit_limit > 0 && unit_limit < units_to_run) units_to_run = unit_limit;
                if (units_to_run < 1) units_to_run = 1;
                AGPT_V2_CUDA_CHECK(cudaMemset(runtime.d_opt_v, 0, (size_t)model.total_floats * sizeof(float)));
                std::printf("  train-epoch: epochs=%d units=%d accumulate=true optimizer=stateful RMSProp\n",
                            epochs, units_to_run);
                long long total_unit_steps = (long long)epochs * (long long)units_to_run;
                long long warmup_unit_steps = (long long)cfg.warmup_epochs * (long long)units_to_run;
                if (total_unit_steps < 1) total_unit_steps = 1;
                for (int epoch = 0; epoch < epochs; epoch++) {
                    double epoch_loss_sum = 0.0;
                    long long epoch_trained = 0;
                    agpt_v2::zero_cache_runtime_v2(runtime.cache);
                    std::printf("  train-epoch: epoch %d/%d\n", epoch + 1, epochs);
                    for (int u = 0; u < units_to_run; u++) {
                        const agpt_v2::TrainingUnit& unit = training_plan.units[u];
                        agpt_v2::ChunkPlanList unit_chunks =
                            agpt_v2::build_chunk_plan_for_unit(trie, unit, cfg.chunk_queries);
                        if (unit_chunks.chunk_count <= 0) {
                            std::printf("    unit %d/%d rc=%d chunks=0 skipped\n",
                                        u + 1, units_to_run, unit.root_child_id);
                            agpt_v2::free_chunk_plan_list(unit_chunks);
                            continue;
                        }

                        long long global_unit_step = (long long)epoch * (long long)units_to_run + (long long)u;
                        float current_lr = scheduled_lr(cfg, global_unit_step, total_unit_steps, warmup_unit_steps);
                        AGPT_V2_CUDA_CHECK(cudaMemset(runtime.d_grads, 0, runtime.contract.weight_and_grad_bytes / 2));
                        agpt_v2::UnitAncGradRuntimeV2 unit_anc{};
                        if (cfg.anc_grad) {
                            agpt_v2::init_unit_anc_grad_runtime_v2(unit_anc, runtime.contract, cfg, unit, trie);
                            agpt_v2::zero_unit_anc_grad_runtime_v2(unit_anc, runtime.contract);
                        }
                        bool run_fire_diag = fire_diag.path
                                          && (epoch + 1) == fire_diag.epoch
                                          && unit.root_child_id == fire_diag.root_id;
                        float* fire_diag_chunk_grads = nullptr;
                        if (run_fire_diag) {
                            fire_diag_chunk_grads = (float*)std::malloc((size_t)model.total_floats * sizeof(float));
                        }
                        double unit_loss_sum = 0.0;
                        long long unit_trained = 0;
                        for (int s = 0; s < unit_chunks.chunk_count; s++) {
                            const agpt_v2::ChunkPlan& chunk = unit_chunks.chunks[s];
                            agpt_v2::ChunkMetadataV2 chunk_meta =
                                agpt_v2::build_chunk_metadata_v2(cfg, shape, trie, unit, chunk);
                            agpt_v2::ChunkDeviceMetadataV2 chunk_device_meta =
                                upload_chunk_metadata_v2(chunk_meta, upload);
                            agpt_v2::ForwardPassResult chunk_fwd =
                                agpt_v2::run_forward_prefix_v2(cfg, model, chunk_meta, chunk_device_meta, upload, loss_tables, runtime,
                                                               cfg.anc_grad ? &unit_anc : nullptr);
                            agpt_v2::BackwardPassResult chunk_bwd =
                                agpt_v2::run_backward_output_head_v2(cfg, model, chunk_meta, chunk_device_meta, upload, chunk_fwd, runtime,
                                                                     cfg.anc_grad ? &unit_anc : nullptr,
                                                                     s == 0, s + 1 == unit_chunks.chunk_count);
                            (void)chunk_bwd;
                            unit_loss_sum += (double)chunk_fwd.mean_loss * (double)chunk_fwd.trained_queries;
                            unit_trained += chunk_fwd.trained_queries;
                            epoch_loss_sum += (double)chunk_fwd.mean_loss * (double)chunk_fwd.trained_queries;
                            epoch_trained += chunk_fwd.trained_queries;
                            if (run_fire_diag) {
                                copy_device_floats_v2(fire_diag_chunk_grads, runtime.d_grads, model.total_floats);
                                FILE* fire_diag_chunk_file = std::fopen(fire_diag.path, "a");
                                if (fire_diag_chunk_file) {
                                    std::fprintf(fire_diag_chunk_file,
                                                 "chunk=%d N=%d T_q=%d T_kv=%d max_kv_len=%d trained_queries=%d mean_loss=%.9f accum_grads_total_l2=%.9f\n",
                                                 s + 1, chunk_meta.N, chunk_meta.T_q, chunk_meta.T_kv, chunk_meta.max_kv_len,
                                                 chunk_fwd.trained_queries, chunk_fwd.mean_loss,
                                                 l2_norm_host_v2(fire_diag_chunk_grads, model.total_floats));
                                    std::fclose(fire_diag_chunk_file);
                                }
                            }
                            agpt_v2::free_chunk_metadata_v2(chunk_meta);
                        }

                        FireDiagBlockV2* fire_diag_blocks = nullptr;
                        char (*fire_diag_names)[32] = nullptr;
                        int fire_diag_block_count = 0;
                        float* fire_diag_grads_pre = nullptr;
                        float* fire_diag_grads_post = nullptr;
                        float* fire_diag_weights_pre = nullptr;
                        float* fire_diag_weights_post = nullptr;
                        float* fire_diag_opt_v_pre = nullptr;
                        float* fire_diag_opt_v = nullptr;
                        if (run_fire_diag) {
                            fire_diag_block_count = 3 + 3 * cfg.n_layers;
                            fire_diag_blocks = (FireDiagBlockV2*)std::malloc((size_t)fire_diag_block_count * sizeof(FireDiagBlockV2));
                            fire_diag_names = (char (*)[32])std::malloc((size_t)fire_diag_block_count * 32);
                            int bi = 0;
                            std::snprintf(fire_diag_names[bi], 32, "token_emb");
                            fire_diag_blocks[bi].name = fire_diag_names[bi];
                            fire_diag_blocks[bi].offset = model.token_emb;
                            fire_diag_blocks[bi].length = cfg.vocab_size * cfg.d_model;
                            bi++;
                            for (int l = 0; l < cfg.n_layers; l++) {
                                std::snprintf(fire_diag_names[bi], 32, "wq_w_l%d", l);
                                fire_diag_blocks[bi].name = fire_diag_names[bi];
                                fire_diag_blocks[bi].offset = model.wq_w[l];
                                fire_diag_blocks[bi].length = cfg.d_model * cfg.d_model;
                                bi++;
                                std::snprintf(fire_diag_names[bi], 32, "wk_w_l%d", l);
                                fire_diag_blocks[bi].name = fire_diag_names[bi];
                                fire_diag_blocks[bi].offset = model.wk_w[l];
                                fire_diag_blocks[bi].length = cfg.d_model * cfg.d_model;
                                bi++;
                                std::snprintf(fire_diag_names[bi], 32, "wv_w_l%d", l);
                                fire_diag_blocks[bi].name = fire_diag_names[bi];
                                fire_diag_blocks[bi].offset = model.wv_w[l];
                                fire_diag_blocks[bi].length = cfg.d_model * cfg.d_model;
                                bi++;
                            }
                            std::snprintf(fire_diag_names[bi], 32, "final_gamma");
                            fire_diag_blocks[bi].name = fire_diag_names[bi];
                            fire_diag_blocks[bi].offset = model.final_gamma;
                            fire_diag_blocks[bi].length = cfg.d_model;
                            bi++;
                            std::snprintf(fire_diag_names[bi], 32, "out_w");
                            fire_diag_blocks[bi].name = fire_diag_names[bi];
                            fire_diag_blocks[bi].offset = model.out_w;
                            fire_diag_blocks[bi].length = cfg.d_model * cfg.vocab_size;
                            bi++;

                            size_t fire_diag_bytes = (size_t)model.total_floats * sizeof(float);
                            fire_diag_grads_pre = (float*)std::malloc(fire_diag_bytes);
                            fire_diag_grads_post = (float*)std::malloc(fire_diag_bytes);
                            fire_diag_weights_pre = (float*)std::malloc(fire_diag_bytes);
                            fire_diag_weights_post = (float*)std::malloc(fire_diag_bytes);
                            fire_diag_opt_v_pre = (float*)std::malloc(fire_diag_bytes);
                            fire_diag_opt_v = (float*)std::malloc(fire_diag_bytes);
                            copy_device_floats_v2(fire_diag_grads_pre, runtime.d_grads, model.total_floats);
                            copy_device_floats_v2(fire_diag_weights_pre, runtime.d_weights, model.total_floats);
                            copy_device_floats_v2(fire_diag_opt_v_pre, runtime.d_opt_v, model.total_floats);
                        }

                        scale_gradients_for_fire(runtime.cublas, runtime.d_grads, model.total_floats, unit_trained);
                        if (run_fire_diag) {
                            copy_device_floats_v2(fire_diag_grads_post, runtime.d_grads, model.total_floats);
                        }
                        agpt_v2::OptimizerStepResult step =
                            agpt_v2::run_optimizer_step_rmsprop_stateful(current_lr, runtime.d_weights, runtime.d_grads, runtime.d_opt_v, model.total_floats);
                        bool fire_diag_exit_now = false;
                        if (run_fire_diag) {
                            copy_device_floats_v2(fire_diag_weights_post, runtime.d_weights, model.total_floats);
                            copy_device_floats_v2(fire_diag_opt_v, runtime.d_opt_v, model.total_floats);
                            FILE* fire_diag_file = std::fopen(fire_diag.path, "a");
                            if (!fire_diag_file) {
                                std::fprintf(stderr, "agpt_train_v2: could not open AGPT_DIAG_FIRE_PATH=%s for write\n", fire_diag.path);
                            } else {
                                std::fprintf(fire_diag_file,
                                             "epoch=%d root_id=%d rc=%d chunks_processed=%d fire_events=%lld fire_mass=%d step_lr=%.9g optimizer=%s\n",
                                             epoch + 1, unit.root_child_id, unit.root_child_id, unit_chunks.chunk_count,
                                             unit_trained, 0, current_lr, "rmsprop");
                                dump_fire_state_v2(fire_diag_file, "pre_step_state", fire_diag_blocks, fire_diag_block_count,
                                                   fire_diag_weights_pre, fire_diag_opt_v_pre, model.total_floats);
                                dump_fire_diag_v2(fire_diag_file, "pre_scale", fire_diag_blocks, fire_diag_block_count,
                                                  fire_diag_grads_pre, nullptr, nullptr, model.total_floats);
                                dump_fire_diag_v2(fire_diag_file, "post_scale", fire_diag_blocks, fire_diag_block_count,
                                                  fire_diag_grads_post, nullptr, nullptr, model.total_floats);
                                dump_fire_diag_v2(fire_diag_file, "post_step", fire_diag_blocks, fire_diag_block_count,
                                                  fire_diag_weights_pre, fire_diag_weights_post, fire_diag_opt_v, model.total_floats);
                                std::fclose(fire_diag_file);
                            }
                            std::free(fire_diag_blocks);
                            std::free(fire_diag_names);
                            std::free(fire_diag_grads_pre);
                            std::free(fire_diag_grads_post);
                            std::free(fire_diag_weights_pre);
                            std::free(fire_diag_weights_post);
                            std::free(fire_diag_opt_v_pre);
                            std::free(fire_diag_opt_v);
                            fire_diag_exit_now = fire_diag.exit_after;
                        }
                        double unit_mean = unit_trained > 0 ? (unit_loss_sum / (double)unit_trained) : 0.0;
                        std::printf("    unit %d/%d rc=%d chunks=%d trained_queries=%lld mean_loss=%.6f lr=%.6g step=%s\n",
                                    u + 1, units_to_run, unit.root_child_id, unit_chunks.chunk_count,
                                    unit_trained, unit_mean, current_lr, step.message);
                        std::free(fire_diag_chunk_grads);
                        agpt_v2::free_unit_anc_grad_runtime_v2(unit_anc, runtime.contract);
                        agpt_v2::free_chunk_plan_list(unit_chunks);
                        if (fire_diag_exit_now) return 0;
                    }
                    double epoch_mean = epoch_trained > 0 ? (epoch_loss_sum / (double)epoch_trained) : 0.0;
                    std::printf("  train-epoch: epoch %d summary trained_queries=%lld mean_loss=%.6f\n",
                                epoch + 1, epoch_trained, epoch_mean);
                }
                if (save_path) {
                    std::printf("  train-epoch: saving final weights to %s\n", save_path);
                    float* h_updated = (float*)std::malloc((size_t)model.total_floats * sizeof(float));
                    AGPT_V2_CUDA_CHECK(cudaMemcpy(h_updated, runtime.d_weights,
                                                  (size_t)model.total_floats * sizeof(float),
                                                  cudaMemcpyDeviceToHost));
                    agpt_v2::save_model_weights_v2(save_path, model, h_updated);
                    std::printf("  train-epoch: saved final weights to %s\n", save_path);
                    std::free(h_updated);
                }
            } else if (run_train_small) {
                agpt_v2::LossTablesV2 loss_tables{d_counts_offset, d_counts_tok, d_counts_val};
                const agpt_v2::TrainingUnit& unit = *plan.largest_by_queries;
                int n_steps = steps;
                if (n_steps > largest_chunks.chunk_count) n_steps = largest_chunks.chunk_count;
                if (n_steps < 1) n_steps = 1;
                AGPT_V2_CUDA_CHECK(cudaMemset(runtime.d_grads, 0, runtime.contract.weight_and_grad_bytes / 2));
                AGPT_V2_CUDA_CHECK(cudaMemset(runtime.d_opt_v, 0, (size_t)model.total_floats * sizeof(float)));
                agpt_v2::UnitAncGradRuntimeV2 unit_anc{};
                if (cfg.anc_grad) {
                    agpt_v2::init_unit_anc_grad_runtime_v2(unit_anc, runtime.contract, cfg, unit, trie);
                    agpt_v2::zero_unit_anc_grad_runtime_v2(unit_anc, runtime.contract);
                }
                std::printf("  train-small: unit rc=%d chunks=%d accumulate=true optimizer=stateful RMSProp\n",
                            unit.root_child_id, n_steps);
                agpt_v2::ForwardPassResult first_before{};
                long long unit_trained = 0;
                for (int s = 0; s < n_steps; s++) {
                    const agpt_v2::ChunkPlan& chunk = largest_chunks.chunks[s];
                    agpt_v2::ChunkMetadataV2 chunk_meta =
                        agpt_v2::build_chunk_metadata_v2(cfg, shape, trie, unit, chunk);
                    agpt_v2::ChunkDeviceMetadataV2 chunk_device_meta =
                        upload_chunk_metadata_v2(chunk_meta, upload);
                    agpt_v2::ForwardPassResult chunk_fwd =
                        agpt_v2::run_forward_prefix_v2(cfg, model, chunk_meta, chunk_device_meta, upload, loss_tables, runtime,
                                                       cfg.anc_grad ? &unit_anc : nullptr);
                    if (s == 0) first_before = chunk_fwd;
                    agpt_v2::BackwardPassResult chunk_bwd =
                        agpt_v2::run_backward_output_head_v2(cfg, model, chunk_meta, chunk_device_meta, upload, chunk_fwd, runtime,
                                                             cfg.anc_grad ? &unit_anc : nullptr,
                                                             s == 0, s + 1 == n_steps);
                    (void)chunk_bwd;
                    unit_trained += chunk_fwd.trained_queries;
                    std::printf("    chunk %d/%d: accumulated loss=%.6f queries=%d nodes=%d\n",
                                s + 1, n_steps, chunk_fwd.mean_loss, chunk_meta.T_q, chunk_meta.N);
                    agpt_v2::free_chunk_metadata_v2(chunk_meta);
                }
                scale_gradients_for_fire(runtime.cublas, runtime.d_grads, model.total_floats, unit_trained);
                agpt_v2::OptimizerStepResult step =
                    agpt_v2::run_optimizer_step_rmsprop_stateful(cfg.lr, runtime.d_weights, runtime.d_grads, runtime.d_opt_v, model.total_floats);
                std::printf("  train-small-step: %s  (first_chunk_before=%.6f accumulated_unit_chunks=%d)\n",
                            step.message, first_before.mean_loss, n_steps);
                agpt_v2::free_unit_anc_grad_runtime_v2(unit_anc, runtime.contract);
            } else if (run_forward_prefix) {
                agpt_v2::LossTablesV2 loss_tables{d_counts_offset, d_counts_tok, d_counts_val};
                agpt_v2::UnitAncGradRuntimeV2 unit_anc{};
                if (cfg.anc_grad && plan.largest_by_queries) {
                    agpt_v2::init_unit_anc_grad_runtime_v2(unit_anc, runtime.contract, cfg, *plan.largest_by_queries, trie);
                    agpt_v2::zero_unit_anc_grad_runtime_v2(unit_anc, runtime.contract);
                }
                agpt_v2::ForwardPassResult fwd =
                    agpt_v2::run_forward_prefix_v2(cfg, model, first_chunk_meta, device_meta, upload, loss_tables, runtime,
                                                   cfg.anc_grad ? &unit_anc : nullptr);
                std::printf("  forward prefix: %s  (trained_queries=%d mean_loss=%.6f)\n",
                            fwd.message, fwd.trained_queries, fwd.mean_loss);
                if (run_backward_head) {
                    agpt_v2::BackwardPassResult bwd =
                        agpt_v2::run_backward_output_head_v2(cfg, model, first_chunk_meta, device_meta, upload, fwd, runtime,
                                                             cfg.anc_grad ? &unit_anc : nullptr,
                                                             true, true);
                    std::printf("  backward head: %s  (||dW_out||=%.6f ||d_final_gamma||=%.6f"
                                " ||dW_2||=%.6f ||dW_1||=%.6f ||d_ln2_gamma||=%.6f"
                                " ||dW_o||=%.6f ||dQ||=%.6f"
                                " ||dW_q||=%.6f ||dW_k||=%.6f ||dW_v||=%.6f ||d_ln1_gamma||=%.6f"
                                " ||dE||=%.6f)\n",
                                bwd.message, bwd.out_w_grad_l2, bwd.final_gamma_grad_l2,
                                bwd.l2_w_grad_l2, bwd.l1_w_grad_l2, bwd.ln2_gamma_grad_l2,
                                bwd.wo_w_grad_l2, bwd.dq_grad_l2,
                                bwd.wq_w_grad_l2, bwd.wk_w_grad_l2, bwd.wv_w_grad_l2, bwd.ln1_gamma_grad_l2,
                                bwd.emb_grad_l2);
                    if (run_one_step_sgd) {
                        scale_gradients_for_fire(runtime.cublas, runtime.d_grads, model.total_floats, fwd.trained_queries);
                        agpt_v2::OptimizerStepResult step =
                            agpt_v2::run_optimizer_step_sgd(cfg, runtime.d_weights, runtime.d_grads, model.total_floats);
                        agpt_v2::ForwardPassResult fwd_after =
                            agpt_v2::run_forward_prefix_v2(cfg, model, first_chunk_meta, device_meta, upload, loss_tables, runtime);
                        std::printf("  one-step-sgd: %s  (loss_before=%.6f loss_after=%.6f delta=%.6f)\n",
                                    step.message, fwd.mean_loss, fwd_after.mean_loss, fwd_after.mean_loss - fwd.mean_loss);
                    } else if (run_one_step_rmsprop) {
                        scale_gradients_for_fire(runtime.cublas, runtime.d_grads, model.total_floats, fwd.trained_queries);
                        agpt_v2::OptimizerStepResult step =
                            agpt_v2::run_optimizer_step_rmsprop(cfg, runtime.d_weights, runtime.d_grads, runtime.d_opt_v, model.total_floats);
                        if (cfg.anc_grad) agpt_v2::zero_unit_anc_grad_runtime_v2(unit_anc, runtime.contract);
                        agpt_v2::ForwardPassResult fwd_after =
                            agpt_v2::run_forward_prefix_v2(cfg, model, first_chunk_meta, device_meta, upload, loss_tables, runtime,
                                                           cfg.anc_grad ? &unit_anc : nullptr);
                        std::printf("  one-step-rmsprop: %s  (loss_before=%.6f loss_after=%.6f delta=%.6f)\n",
                                    step.message, fwd.mean_loss, fwd_after.mean_loss, fwd_after.mean_loss - fwd.mean_loss);
                    } else if (run_multi_step_sgd) {
                        agpt_v2::ForwardPassResult cur_fwd = fwd;
                        std::printf("  multi-step-sgd: starting loss=%.6f steps=%d\n", cur_fwd.mean_loss, steps);
                        for (int s = 0; s < steps; s++) {
                            agpt_v2::BackwardPassResult cur_bwd =
                                agpt_v2::run_backward_output_head_v2(cfg, model, first_chunk_meta, device_meta, upload, cur_fwd, runtime,
                                                                     cfg.anc_grad ? &unit_anc : nullptr,
                                                                     true, true);
                            (void)cur_bwd;
                            scale_gradients_for_fire(runtime.cublas, runtime.d_grads, model.total_floats, cur_fwd.trained_queries);
                            agpt_v2::OptimizerStepResult step =
                                agpt_v2::run_optimizer_step_sgd(cfg, runtime.d_weights, runtime.d_grads, model.total_floats);
                            if (cfg.anc_grad) agpt_v2::zero_unit_anc_grad_runtime_v2(unit_anc, runtime.contract);
                            agpt_v2::ForwardPassResult next_fwd =
                                agpt_v2::run_forward_prefix_v2(cfg, model, first_chunk_meta, device_meta, upload, loss_tables, runtime,
                                                               cfg.anc_grad ? &unit_anc : nullptr);
                            std::printf("    step %d: %s  loss_before=%.6f loss_after=%.6f delta=%.6f\n",
                                        s + 1, step.message, cur_fwd.mean_loss, next_fwd.mean_loss,
                                        next_fwd.mean_loss - cur_fwd.mean_loss);
                            cur_fwd = next_fwd;
                        }
                    } else if (run_multi_step_rmsprop) {
                        AGPT_V2_CUDA_CHECK(cudaMemset(runtime.d_opt_v, 0, (size_t)model.total_floats * sizeof(float)));
                        agpt_v2::ForwardPassResult cur_fwd = fwd;
                        std::printf("  multi-step-rmsprop: starting loss=%.6f steps=%d\n", cur_fwd.mean_loss, steps);
                        for (int s = 0; s < steps; s++) {
                            agpt_v2::BackwardPassResult cur_bwd =
                                agpt_v2::run_backward_output_head_v2(cfg, model, first_chunk_meta, device_meta, upload, cur_fwd, runtime,
                                                                     cfg.anc_grad ? &unit_anc : nullptr,
                                                                     true, true);
                            (void)cur_bwd;
                            scale_gradients_for_fire(runtime.cublas, runtime.d_grads, model.total_floats, cur_fwd.trained_queries);
                            agpt_v2::OptimizerStepResult step =
                                agpt_v2::run_optimizer_step_rmsprop_stateful(cfg, runtime.d_weights, runtime.d_grads, runtime.d_opt_v, model.total_floats);
                            if (cfg.anc_grad) agpt_v2::zero_unit_anc_grad_runtime_v2(unit_anc, runtime.contract);
                            agpt_v2::ForwardPassResult next_fwd =
                                agpt_v2::run_forward_prefix_v2(cfg, model, first_chunk_meta, device_meta, upload, loss_tables, runtime,
                                                               cfg.anc_grad ? &unit_anc : nullptr);
                            std::printf("    step %d: %s  loss_before=%.6f loss_after=%.6f delta=%.6f\n",
                                        s + 1, step.message, cur_fwd.mean_loss, next_fwd.mean_loss,
                                        next_fwd.mean_loss - cur_fwd.mean_loss);
                            cur_fwd = next_fwd;
                        }
                    } else if (run_save_reload_sgd) {
                        const char* roundtrip_path = save_path ? save_path : "/tmp/agpt_v2_roundtrip.model";
                        scale_gradients_for_fire(runtime.cublas, runtime.d_grads, model.total_floats, fwd.trained_queries);
                        agpt_v2::OptimizerStepResult step =
                            agpt_v2::run_optimizer_step_sgd(cfg, runtime.d_weights, runtime.d_grads, model.total_floats);
                        float* h_updated = (float*)std::malloc((size_t)model.total_floats * sizeof(float));
                        AGPT_V2_CUDA_CHECK(cudaMemcpy(h_updated, runtime.d_weights,
                                                      (size_t)model.total_floats * sizeof(float),
                                                      cudaMemcpyDeviceToHost));
                        agpt_v2::save_model_weights_v2(roundtrip_path, model, h_updated);
                        float* h_reloaded = agpt_v2::load_model_weights_v2(roundtrip_path, model);
                        double max_abs_diff = 0.0;
                        for (int i = 0; i < model.total_floats; i++) {
                            double diff = std::fabs((double)h_updated[i] - (double)h_reloaded[i]);
                            if (diff > max_abs_diff) max_abs_diff = diff;
                        }
                        AGPT_V2_CUDA_CHECK(cudaMemcpy(runtime.d_weights, h_reloaded,
                                                      (size_t)model.total_floats * sizeof(float),
                                                      cudaMemcpyHostToDevice));
                        if (cfg.anc_grad) agpt_v2::zero_unit_anc_grad_runtime_v2(unit_anc, runtime.contract);
                        agpt_v2::ForwardPassResult fwd_after =
                            agpt_v2::run_forward_prefix_v2(cfg, model, first_chunk_meta, device_meta, upload, loss_tables, runtime,
                                                           cfg.anc_grad ? &unit_anc : nullptr);
                        std::printf("  save-reload-sgd: %s  (loss_before=%.6f loss_after=%.6f delta=%.6f max_abs_diff=%.3e file=%s)\n",
                                    step.message, fwd.mean_loss, fwd_after.mean_loss, fwd_after.mean_loss - fwd.mean_loss,
                                    max_abs_diff, roundtrip_path);
                        std::free(h_updated);
                        std::free(h_reloaded);
                    } else if (run_save_reload_rmsprop) {
                        const char* roundtrip_path = save_path ? save_path : "/tmp/agpt_v2_roundtrip.model";
                        const char* opt_path = "/tmp/agpt_v2_roundtrip.optv";
                        scale_gradients_for_fire(runtime.cublas, runtime.d_grads, model.total_floats, fwd.trained_queries);
                        agpt_v2::OptimizerStepResult step =
                            agpt_v2::run_optimizer_step_rmsprop_stateful(cfg, runtime.d_weights, runtime.d_grads, runtime.d_opt_v, model.total_floats);
                        float* h_updated = (float*)std::malloc((size_t)model.total_floats * sizeof(float));
                        float* h_opt = (float*)std::malloc((size_t)model.total_floats * sizeof(float));
                        AGPT_V2_CUDA_CHECK(cudaMemcpy(h_updated, runtime.d_weights,
                                                      (size_t)model.total_floats * sizeof(float),
                                                      cudaMemcpyDeviceToHost));
                        AGPT_V2_CUDA_CHECK(cudaMemcpy(h_opt, runtime.d_opt_v,
                                                      (size_t)model.total_floats * sizeof(float),
                                                      cudaMemcpyDeviceToHost));
                        agpt_v2::save_model_weights_v2(roundtrip_path, model, h_updated);
                        agpt_v2::save_optimizer_state_v2(opt_path, h_opt, model.total_floats);
                        float* h_reloaded = agpt_v2::load_model_weights_v2(roundtrip_path, model);
                        float* h_opt_reloaded = agpt_v2::load_optimizer_state_v2(opt_path, model.total_floats);
                        double max_abs_diff_w = 0.0;
                        double max_abs_diff_v = 0.0;
                        for (int i = 0; i < model.total_floats; i++) {
                            double diff_w = std::fabs((double)h_updated[i] - (double)h_reloaded[i]);
                            double diff_v = std::fabs((double)h_opt[i] - (double)h_opt_reloaded[i]);
                            if (diff_w > max_abs_diff_w) max_abs_diff_w = diff_w;
                            if (diff_v > max_abs_diff_v) max_abs_diff_v = diff_v;
                        }
                        AGPT_V2_CUDA_CHECK(cudaMemcpy(runtime.d_weights, h_reloaded,
                                                      (size_t)model.total_floats * sizeof(float),
                                                      cudaMemcpyHostToDevice));
                        AGPT_V2_CUDA_CHECK(cudaMemcpy(runtime.d_opt_v, h_opt_reloaded,
                                                      (size_t)model.total_floats * sizeof(float),
                                                      cudaMemcpyHostToDevice));
                        if (cfg.anc_grad) agpt_v2::zero_unit_anc_grad_runtime_v2(unit_anc, runtime.contract);
                        agpt_v2::ForwardPassResult fwd_after =
                            agpt_v2::run_forward_prefix_v2(cfg, model, first_chunk_meta, device_meta, upload, loss_tables, runtime,
                                                           cfg.anc_grad ? &unit_anc : nullptr);
                        std::printf("  save-reload-rmsprop: %s  (loss_before=%.6f loss_after=%.6f delta=%.6f max_abs_diff_w=%.3e max_abs_diff_v=%.3e file=%s)\n",
                                    step.message, fwd.mean_loss, fwd_after.mean_loss, fwd_after.mean_loss - fwd.mean_loss,
                                    max_abs_diff_w, max_abs_diff_v, roundtrip_path);
                        std::free(h_updated);
                        std::free(h_opt);
                        std::free(h_reloaded);
                        std::free(h_opt_reloaded);
                    }
                }
                agpt_v2::free_unit_anc_grad_runtime_v2(unit_anc, runtime.contract);
            }
            free_chunk_upload_runtime_v2(upload);
            std::printf("  chunk upload: freed successfully\n");
        } else if (instantiate_chunk_upload) {
            std::printf("  chunk upload: no chunk metadata available to upload\n");
        } else {
            std::printf("  chunk upload: not instantiated (pass --instantiate-chunk-upload to exercise metadata upload)\n");
        }
        free_trainer_runtime_v2(runtime);
        if (d_counts_offset) cudaFree(d_counts_offset);
        if (d_counts_tok) cudaFree(d_counts_tok);
        if (d_counts_val) cudaFree(d_counts_val);
        std::printf("  runtime objects: freed successfully\n");
        std::free(h_weights);
    } else {
        std::printf("  runtime objects: not instantiated (pass --instantiate-runtime to exercise CUDA allocation)\n");
        std::printf("  chunk upload: not instantiated (pass --instantiate-chunk-upload to exercise metadata upload)\n");
    }
    std::printf("  status: v2 currently validates file formats, plans baseline pd=1 execution,\n"
                "          and exercises the full-depth chunk upload/cache/forward/loss path,\n"
                "          plus output-head/final-LN backward, train-epoch/train-small accumulation, and one-step SGD/RMSProp/multi-step/save-reload sanity modes when requested.\n");

    (void)model;
    if (have_first_chunk_meta) agpt_v2::free_chunk_metadata_v2(first_chunk_meta);
    agpt_v2::free_chunk_plan_list(largest_chunks);
    agpt_v2::free_training_plan(training_plan);
    agpt_v2::free_radix_trie_structure(trie);
    agpt_v2::free_model_layout(model);
    return 0;
}
