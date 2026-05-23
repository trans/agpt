#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>

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
#include "growth_trie_v2.cuh"

namespace {

struct DiagFireProbeV2 {
    const char* tensor_dir = nullptr;
    int epoch = 0;
    int root_id = 0;
    bool exit_after = false;
    bool enabled = false;
};

static int read_env_int_or_default_v2(const char* name, int fallback) {
    const char* env = std::getenv(name);
    if (!env || !env[0]) return fallback;
    return std::atoi(env);
}

static bool read_env_flag_v2(const char* name) {
    const char* env = std::getenv(name);
    if (!env || !env[0]) return false;
    return !(std::strcmp(env, "0") == 0 || std::strcmp(env, "false") == 0 || std::strcmp(env, "False") == 0);
}

static DiagFireProbeV2 read_diag_fire_probe_v2() {
    DiagFireProbeV2 cfg{};
    cfg.tensor_dir = std::getenv("AGPT_DIAG_TENSOR_DIR");
    if (!cfg.tensor_dir || !cfg.tensor_dir[0]) return cfg;
    cfg.epoch = read_env_int_or_default_v2("AGPT_DIAG_FIRE_EPOCH", 1);
    cfg.root_id = read_env_int_or_default_v2("AGPT_DIAG_FIRE_ROOT_ID", 1);
    cfg.exit_after = read_env_flag_v2("AGPT_DIAG_FIRE_EXIT_AFTER");
    cfg.enabled = true;
    return cfg;
}

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
    TrainGrowth,
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
        case V2Mode::TrainGrowth: return "train-growth";
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
    if (std::strcmp(text, "train-growth") == 0) {
        out = V2Mode::TrainGrowth;
        return true;
    }
    return false;
}

static const char* v2_optimizer_name(agpt_v2::OptimizerKind optimizer) {
    switch (optimizer) {
        case agpt_v2::OptimizerKind::Adam: return "adam";
        case agpt_v2::OptimizerKind::SGD: return "sgd";
        case agpt_v2::OptimizerKind::Momentum: return "momentum";
        case agpt_v2::OptimizerKind::RMSProp: return "rmsprop";
    }
    return "unknown";
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

static bool parse_optimizer_kind(const char* text, agpt_v2::OptimizerKind& out) {
    if (std::strcmp(text, "adam") == 0) {
        out = agpt_v2::OptimizerKind::Adam;
        return true;
    }
    if (std::strcmp(text, "sgd") == 0) {
        out = agpt_v2::OptimizerKind::SGD;
        return true;
    }
    if (std::strcmp(text, "momentum") == 0) {
        out = agpt_v2::OptimizerKind::Momentum;
        return true;
    }
    if (std::strcmp(text, "rmsprop") == 0) {
        out = agpt_v2::OptimizerKind::RMSProp;
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

static double wall_seconds_v2() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

static int effective_seq_len_from_trie_v2(const agpt_v2::RadixTrieStructure& trie) {
    int effective = trie.depth_file_count - 1;
    if (effective < 1) effective = 1;
    return effective;
}

static long long active_count_entries_v2(const agpt_v2::RadixTrieStructure& trie) {
    long long total = 0;
    for (int r = 0; r < trie.radix_count; r++) total += trie.counts_len[r];
    return total;
}

static agpt_v2::ChunkPlanList build_capacity_chunk_list_for_plan_v2(
    const agpt_v2::RadixTrieStructure& trie,
    const agpt_v2::TrainingPlan& training_plan,
    int chunk_queries) {
    agpt_v2::ChunkPlanList capacity{};
    capacity.chunk_count = 1;
    capacity.chunks = (agpt_v2::ChunkPlan*)std::calloc(1, sizeof(agpt_v2::ChunkPlan));
    for (int u = 0; u < training_plan.unit_count; u++) {
        agpt_v2::ChunkPlanList chunks =
            agpt_v2::build_chunk_plan_for_unit(trie, training_plan.units[u], chunk_queries);
        for (int c = 0; c < chunks.chunk_count; c++) {
            const agpt_v2::ChunkPlan& chunk = chunks.chunks[c];
            agpt_v2::ChunkPlan& cap = capacity.chunks[0];
            if (chunk.node_count > cap.node_count) cap.node_count = chunk.node_count;
            if (chunk.query_count > cap.query_count) cap.query_count = chunk.query_count;
            if (chunk.kv_count > cap.kv_count) cap.kv_count = chunk.kv_count;
            if (chunk.compact_char_count > cap.compact_char_count) cap.compact_char_count = chunk.compact_char_count;
            if (chunk.max_kv_len > cap.max_kv_len) cap.max_kv_len = chunk.max_kv_len;
        }
        agpt_v2::free_chunk_plan_list(chunks);
    }
    return capacity;
}

struct DeviceLossTablesV2 {
    int* d_counts_offset = nullptr;
    int* d_counts_len = nullptr;
    int* d_counts_tok = nullptr;
    int* d_counts_val = nullptr;
};

static DeviceLossTablesV2 upload_loss_tables_v2(const agpt_v2::RadixTrieStructure& trie) {
    DeviceLossTablesV2 out{};
    int radix_count_for_tables = trie.radix_count > 0 ? trie.radix_count : 1;
    AGPT_V2_CUDA_CHECK(cudaMalloc(&out.d_counts_offset, (size_t)radix_count_for_tables * sizeof(int)));
    AGPT_V2_CUDA_CHECK(cudaMemcpy(out.d_counts_offset, trie.counts_offset,
                                  (size_t)radix_count_for_tables * sizeof(int),
                                  cudaMemcpyHostToDevice));
    AGPT_V2_CUDA_CHECK(cudaMalloc(&out.d_counts_len, (size_t)radix_count_for_tables * sizeof(int)));
    AGPT_V2_CUDA_CHECK(cudaMemcpy(out.d_counts_len, trie.counts_len,
                                  (size_t)radix_count_for_tables * sizeof(int),
                                  cudaMemcpyHostToDevice));
    AGPT_V2_CUDA_CHECK(cudaMalloc(&out.d_counts_tok, (size_t)(trie.total_counts > 0 ? trie.total_counts : 1) * sizeof(int)));
    AGPT_V2_CUDA_CHECK(cudaMalloc(&out.d_counts_val, (size_t)(trie.total_counts > 0 ? trie.total_counts : 1) * sizeof(int)));
    if (trie.total_counts > 0) {
        AGPT_V2_CUDA_CHECK(cudaMemcpy(out.d_counts_tok, trie.counts_tok,
                                      (size_t)trie.total_counts * sizeof(int),
                                      cudaMemcpyHostToDevice));
        AGPT_V2_CUDA_CHECK(cudaMemcpy(out.d_counts_val, trie.counts_val,
                                      (size_t)trie.total_counts * sizeof(int),
                                      cudaMemcpyHostToDevice));
    }
    return out;
}

static agpt_v2::LossTablesV2 make_loss_tables_view_v2(const DeviceLossTablesV2& tables) {
    return agpt_v2::LossTablesV2{
        tables.d_counts_offset,
        tables.d_counts_len,
        tables.d_counts_tok,
        tables.d_counts_val,
    };
}

static void free_device_loss_tables_v2(DeviceLossTablesV2& tables) {
    if (tables.d_counts_offset) cudaFree(tables.d_counts_offset);
    if (tables.d_counts_len) cudaFree(tables.d_counts_len);
    if (tables.d_counts_tok) cudaFree(tables.d_counts_tok);
    if (tables.d_counts_val) cudaFree(tables.d_counts_val);
    tables = DeviceLossTablesV2{};
}

static void run_train_epoch_on_radix_host_v2(const agpt_v2::TrainerConfig& cfg,
                                             const agpt_v2::RuntimeShape& shape,
                                             const agpt_v2::ModelLayout& model,
                                             const agpt_v2::RadixTrieStructure& trie,
                                             float* h_weights,
                                             float* h_opt_m,
                                             float* h_opt_v,
                                             int epochs,
                                             int unit_limit,
                                             long long total_unit_steps,
                                             long long warmup_unit_steps,
                                             int& optimizer_step_index) {
    double t_total0 = wall_seconds_v2();
    agpt_v2::CacheLayout cache = agpt_v2::make_cache_layout(shape);
    agpt_v2::TrainingPlan training_plan = agpt_v2::build_pd1_training_plan(trie);
    if (training_plan.unit_count <= 0) {
        std::printf("  train-growth-stage: skipped, no pd=1 training units at radix_nodes=%d\n",
                    trie.radix_count);
        agpt_v2::free_training_plan(training_plan);
        return;
    }
    agpt_v2::ExecutionPlan plan = agpt_v2::build_execution_plan(trie, training_plan, cfg.chunk_queries);
    agpt_v2::ChunkPlanList capacity_chunks =
        build_capacity_chunk_list_for_plan_v2(trie, training_plan, cfg.chunk_queries);
    agpt_v2::TrainerRuntimeContract runtime_contract =
        agpt_v2::build_trainer_runtime_contract(shape, cache, plan, capacity_chunks,
                                                trie.compact_slot_capacity);
    double t_plan1 = wall_seconds_v2();

    int units_to_run = plan.training_unit_count;
    if (unit_limit > 0 && unit_limit < units_to_run) units_to_run = unit_limit;
    if (units_to_run < 1) units_to_run = 1;

    agpt_v2::TrainerRuntimeV2 runtime{};
    init_trainer_runtime_v2(runtime, runtime_contract, trie);
    agpt_v2::zero_cache_runtime_v2(runtime.cache);
    AGPT_V2_CUDA_CHECK(cudaMemcpy(runtime.d_weights, h_weights,
                                  (size_t)model.total_floats * sizeof(float),
                                  cudaMemcpyHostToDevice));
    AGPT_V2_CUDA_CHECK(cudaMemcpy(runtime.d_opt_m, h_opt_m,
                                  (size_t)model.total_floats * sizeof(float),
                                  cudaMemcpyHostToDevice));
    AGPT_V2_CUDA_CHECK(cudaMemcpy(runtime.d_opt_v, h_opt_v,
                                  (size_t)model.total_floats * sizeof(float),
                                  cudaMemcpyHostToDevice));

    DeviceLossTablesV2 device_loss_tables = upload_loss_tables_v2(trie);
    agpt_v2::LossTablesV2 loss_tables = make_loss_tables_view_v2(device_loss_tables);

    agpt_v2::ChunkUploadRuntimeV2 upload{};
    init_chunk_upload_runtime_v2(upload, runtime_contract.chunk.node_capacity,
                                 runtime_contract.chunk.query_capacity, shape.n_heads);
    AGPT_V2_CUDA_CHECK(cudaDeviceSynchronize());
    double t_setup1 = wall_seconds_v2();

    std::printf("  train-growth-stage: epochs=%d units=%d optimizer=%s radix_nodes=%d edge_chars=%lld\n",
                epochs, units_to_run, v2_optimizer_name(cfg.optimizer),
                trie.radix_count, trie.total_edge_chars);
    if (total_unit_steps < 1) total_unit_steps = 1;
    for (int epoch = 0; epoch < epochs; epoch++) {
        double epoch_loss_sum = 0.0;
        long long epoch_trained = 0;
        agpt_v2::zero_cache_runtime_v2(runtime.cache);
        std::printf("    stage-epoch %d/%d\n", epoch + 1, epochs);
        for (int u = 0; u < units_to_run; u++) {
            const agpt_v2::TrainingUnit& unit = training_plan.units[u];
            agpt_v2::ChunkPlanList unit_chunks =
                agpt_v2::build_chunk_plan_for_unit(trie, unit, cfg.chunk_queries);
            if (unit_chunks.chunk_count <= 0) {
                agpt_v2::free_chunk_plan_list(unit_chunks);
                continue;
            }

            float current_lr = scheduled_lr(cfg, optimizer_step_index, total_unit_steps, warmup_unit_steps);
            AGPT_V2_CUDA_CHECK(cudaMemset(runtime.d_grads, 0, runtime.contract.weight_and_grad_bytes / 2));
            agpt_v2::UnitAncGradRuntimeV2 unit_anc{};
            if (cfg.anc_grad) {
                agpt_v2::init_unit_anc_grad_runtime_v2(unit_anc, runtime.contract, cfg, unit, trie);
                agpt_v2::zero_unit_anc_grad_runtime_v2(unit_anc, runtime.contract);
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
                    agpt_v2::run_forward_prefix_v2(cfg, model, chunk_meta, chunk_device_meta,
                                                   upload, loss_tables, runtime,
                                                   cfg.anc_grad ? &unit_anc : nullptr);
                agpt_v2::BackwardPassResult chunk_bwd =
                    agpt_v2::run_backward_output_head_v2(cfg, model, chunk_meta, chunk_device_meta,
                                                         upload, chunk_fwd, runtime,
                                                         cfg.anc_grad ? &unit_anc : nullptr,
                                                         s == 0, s + 1 == unit_chunks.chunk_count);
                (void)chunk_bwd;
                unit_loss_sum += (double)chunk_fwd.mean_loss * (double)chunk_fwd.trained_queries;
                unit_trained += chunk_fwd.trained_queries;
                epoch_loss_sum += (double)chunk_fwd.mean_loss * (double)chunk_fwd.trained_queries;
                epoch_trained += chunk_fwd.trained_queries;
                agpt_v2::free_chunk_metadata_v2(chunk_meta);
            }

            scale_gradients_for_fire(runtime.cublas, runtime.d_grads, model.total_floats, unit_trained);
            agpt_v2::OptimizerStepResult step =
                agpt_v2::run_optimizer_step_stateful(cfg, current_lr, runtime.d_weights, runtime.d_grads,
                                                     runtime.d_opt_m, runtime.d_opt_v,
                                                     model.total_floats, ++optimizer_step_index);
            double unit_mean = unit_trained > 0 ? unit_loss_sum / (double)unit_trained : 0.0;
            std::printf("      unit %d/%d rc=%d chunks=%d trained_queries=%lld mean_loss=%.6f lr=%.6g step=%s\n",
                        u + 1, units_to_run, unit.root_child_id, unit_chunks.chunk_count,
                        unit_trained, unit_mean, current_lr, step.message);
            agpt_v2::free_unit_anc_grad_runtime_v2(unit_anc, runtime.contract);
            agpt_v2::free_chunk_plan_list(unit_chunks);
        }
        double epoch_mean = epoch_trained > 0 ? epoch_loss_sum / (double)epoch_trained : 0.0;
        std::printf("    stage-epoch %d summary trained_queries=%lld mean_loss=%.6f\n",
                    epoch + 1, epoch_trained, epoch_mean);
    }
    AGPT_V2_CUDA_CHECK(cudaDeviceSynchronize());
    double t_train1 = wall_seconds_v2();

    AGPT_V2_CUDA_CHECK(cudaMemcpy(h_weights, runtime.d_weights,
                                  (size_t)model.total_floats * sizeof(float),
                                  cudaMemcpyDeviceToHost));
    AGPT_V2_CUDA_CHECK(cudaMemcpy(h_opt_m, runtime.d_opt_m,
                                  (size_t)model.total_floats * sizeof(float),
                                  cudaMemcpyDeviceToHost));
    AGPT_V2_CUDA_CHECK(cudaMemcpy(h_opt_v, runtime.d_opt_v,
                                  (size_t)model.total_floats * sizeof(float),
                                  cudaMemcpyDeviceToHost));
    AGPT_V2_CUDA_CHECK(cudaDeviceSynchronize());
    double t_copy1 = wall_seconds_v2();

    free_chunk_upload_runtime_v2(upload);
    free_device_loss_tables_v2(device_loss_tables);
    agpt_v2::free_trainer_runtime_v2(runtime);
    agpt_v2::free_chunk_plan_list(capacity_chunks);
    agpt_v2::free_training_plan(training_plan);
    AGPT_V2_CUDA_CHECK(cudaDeviceSynchronize());
    double t_cleanup1 = wall_seconds_v2();
    std::printf("  train-growth-stage-timing: plan=%.3fs setup=%.3fs train=%.3fs copy_back=%.3fs cleanup=%.3fs total=%.3fs\n",
                t_plan1 - t_total0,
                t_setup1 - t_plan1,
                t_train1 - t_setup1,
                t_copy1 - t_train1,
                t_cleanup1 - t_copy1,
                t_cleanup1 - t_total0);
}

}  // namespace

int main(int argc, char** argv) {
    agpt_v2::TrainerConfig cfg;
    const char* model_path = nullptr;
    const char* trie_dir = nullptr;
    const char* corpus_path = nullptr;
    const char* growth_frontiers_arg = nullptr;
    const char* save_path = nullptr;
    V2Mode mode = V2Mode::Plan;
    int steps = 3;
    int unit_limit = 0;
    int growth_max_depth = 0;

    cfg.epochs = 1;
    cfg.partition_depth = 1;
    cfg.chunk_queries = 50000;
    cfg.lr = 3e-4f;
    cfg.momentum_beta = 0.9f;
    cfg.rmsprop_beta = 0.999f;
    cfg.lr_schedule = agpt_v2::LrSchedule::Constant;
    cfg.optimizer = agpt_v2::OptimizerKind::RMSProp;
    cfg.warmup_epochs = 0;
    cfg.accumulate = true;

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (std::strcmp(argv[i], "--trie-dir") == 0 && i + 1 < argc) trie_dir = argv[++i];
        else if (std::strcmp(argv[i], "--corpus") == 0 && i + 1 < argc) corpus_path = argv[++i];
        else if (std::strcmp(argv[i], "--growth-frontiers") == 0 && i + 1 < argc) growth_frontiers_arg = argv[++i];
        else if ((std::strcmp(argv[i], "--growth-max-depth") == 0 ||
                  std::strcmp(argv[i], "--max-depth") == 0) && i + 1 < argc) growth_max_depth = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) cfg.epochs = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--partition-depth") == 0 && i + 1 < argc) cfg.partition_depth = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--chunk-queries") == 0 && i + 1 < argc) cfg.chunk_queries = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--lr") == 0 && i + 1 < argc) cfg.lr = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--optimizer") == 0 && i + 1 < argc) {
            if (!parse_optimizer_kind(argv[++i], cfg.optimizer)) {
                std::fprintf(stderr, "agpt_train_v2: unsupported --optimizer value: %s\n", argv[i]);
                return 1;
            }
        }
        else if (std::strcmp(argv[i], "--momentum-beta") == 0 && i + 1 < argc) cfg.momentum_beta = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--rmsprop-beta") == 0 && i + 1 < argc) cfg.rmsprop_beta = std::atof(argv[++i]);
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

    bool missing_required = !model_path ||
        (mode == V2Mode::TrainGrowth ? (!corpus_path || !growth_frontiers_arg) : !trie_dir);
    if (missing_required) {
        std::fprintf(stderr,
                     "Usage: agpt_train_v2 --model <path> --trie-dir <path>\n"
                     "       agpt_train_v2 --mode train-growth --model <path> --corpus <path> --growth-frontiers LIST [--growth-max-depth N]\n"
                     "  [--epochs N] [--partition-depth 1] [--chunk-queries N] [--lr F] [--optimizer adam|sgd|momentum|rmsprop]\n"
                     "  [--momentum-beta F] [--rmsprop-beta F] [--lr-schedule constant|warmup-cosine]\n"
                     "  [--warmup-epochs N] [--steps N]\n"
                     "  [--anc-grad]\n"
                     "  [--units N]\n"
                     "  [--save PATH]\n"
                     "  [--mode plan|instantiate-runtime|upload|forward|backward-head|one-step-sgd|one-step-rmsprop|multi-step-sgd|multi-step-rmsprop|save-reload-sgd|save-reload-rmsprop|train-epoch|train-small|train-growth]\n"
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
    DiagFireProbeV2 diag_probe = read_diag_fire_probe_v2();

    agpt_v2::ModelHeader header = agpt_v2::load_model_header(model_path);
    agpt_v2::RuntimeShape shape = header.shape;
    int header_seq_len = shape.seq_len;
    cfg.d_model = shape.d_model;
    cfg.n_heads = shape.n_heads;
    cfg.n_layers = shape.n_layers;
    cfg.d_ff = shape.d_ff;
    cfg.vocab_size = shape.vocab_size;
    if (mode == V2Mode::TrainGrowth) {
        int vocab_size_from_corpus = 0;
        std::vector<int> tokens = agpt_v2::tokenize_corpus_sorted_unique_utf8_v2(corpus_path, &vocab_size_from_corpus);
        if ((int)tokens.size() < 2) {
            std::fprintf(stderr, "agpt_train_v2: train-growth corpus needs at least 2 tokens\n");
            return 1;
        }
        if (vocab_size_from_corpus != shape.vocab_size) {
            std::fprintf(stderr,
                         "agpt_train_v2: train-growth vocab mismatch corpus=%d model=%d\n",
                         vocab_size_from_corpus, shape.vocab_size);
            return 1;
        }
        int max_depth = growth_max_depth > 0 ? growth_max_depth : header_seq_len;
        if (max_depth < 1) max_depth = 1;
        int max_possible_depth = (int)tokens.size() - 1;
        if (max_depth > max_possible_depth) max_depth = max_possible_depth;
        shape.seq_len = max_depth;
        cfg.seq_len = shape.seq_len;

        int full_starts = (int)tokens.size() - 1;
        std::vector<int> frontiers = agpt_v2::parse_growth_frontiers_v2(growth_frontiers_arg, full_starts);
        if (frontiers.empty()) frontiers.push_back(full_starts);
        agpt_v2::ModelLayout model = agpt_v2::make_model_layout(shape);
        float* h_weights = agpt_v2::load_model_weights_v2(model_path, model);
        float* h_opt_m = (float*)std::calloc((size_t)model.total_floats, sizeof(float));
        float* h_opt_v = (float*)std::calloc((size_t)model.total_floats, sizeof(float));
        if (!h_weights || !h_opt_m || !h_opt_v) {
            std::fprintf(stderr, "agpt_train_v2: train-growth failed allocating host state\n");
            std::free(h_weights);
            std::free(h_opt_m);
            std::free(h_opt_v);
            agpt_v2::free_model_layout(model);
            return 1;
        }

        int epochs = cfg.epochs > 0 ? cfg.epochs : 1;
        int estimated_units = unit_limit > 0 ? unit_limit : cfg.vocab_size;
        long long total_unit_steps = (long long)frontiers.size() * (long long)epochs * (long long)estimated_units;
        long long warmup_unit_steps = (long long)cfg.warmup_epochs * (long long)estimated_units;
        int optimizer_step_index = 0;
        const char* growth_radix_env = std::getenv("AGPT_GROWTH_RADIX");
        bool use_incremental_growth_radix =
            !(growth_radix_env && std::strcmp(growth_radix_env, "rebuild") == 0);
        agpt_v2::GrowthTrieStateV2 growth_rebuild;
        agpt_v2::GrowthIncrementalRadixStateV2 growth_incremental;
        if (use_incremental_growth_radix) {
            growth_incremental = agpt_v2::make_growth_incremental_radix_state_v2(std::move(tokens), max_depth);
        } else {
            growth_rebuild = agpt_v2::make_growth_trie_state_v2(std::move(tokens), max_depth);
        }

        std::printf("AGPT CUDA Trainer V2\n");
        std::printf("  mode: %s\n", v2_mode_name(mode));
        std::printf("  model: d=%d heads=%d layers=%d ff=%d vocab=%d seq=%d head_dim=%d\n",
                    shape.d_model, shape.n_heads, shape.n_layers, shape.d_ff,
                    shape.vocab_size, shape.seq_len, shape.head_dim);
        if (header_seq_len != shape.seq_len) {
            std::printf("  seq_len reconcile: model header says %d, growth max_depth=%d -> effective %d. Overriding.\n",
                        header_seq_len, max_depth, shape.seq_len);
        }
        std::printf("  corpus: %s tokens=%zu full_starts=%d\n",
                    corpus_path,
                    use_incremental_growth_radix ? growth_incremental.tokens.size() : growth_rebuild.tokens.size(),
                    full_starts);
        std::printf("  growth: stages=%zu epochs_per_stage=%d optimizer=%s schedule=%s materializer=%s estimated_total_unit_steps=%lld\n",
                    frontiers.size(), epochs, v2_optimizer_name(cfg.optimizer),
                    v2_lr_schedule_name(cfg.lr_schedule),
                    use_incremental_growth_radix ? "incremental-radix" : "rebuild",
                    total_unit_steps);
        std::printf("  config: lr=%.6f warmup_epochs=%d partition_depth=%d chunk_queries=%d anc_grad=%s\n",
                    cfg.lr, cfg.warmup_epochs, cfg.partition_depth, cfg.chunk_queries,
                    cfg.anc_grad ? "true" : "false");

        for (int i = 0; i < (int)frontiers.size(); i++) {
            int frontier = frontiers[i];
            double t_stage0 = wall_seconds_v2();
            if (use_incremental_growth_radix) {
                agpt_v2::growth_incremental_ingest_until_v2(growth_incremental, frontier);
            } else {
                agpt_v2::growth_ingest_until_v2(growth_rebuild, frontier);
            }
            double t_ingest1 = wall_seconds_v2();
            agpt_v2::RadixTrieStructure trie =
                use_incremental_growth_radix
                    ? agpt_v2::growth_incremental_radix_view_v2(growth_incremental)
                    : agpt_v2::growth_build_radix_view_v2(growth_rebuild);
            double t_materialize1 = wall_seconds_v2();
            long long active_counts = active_count_entries_v2(trie);
            std::printf("  growth-stage %d/%zu: frontier_starts=%d ingested_starts=%d radix_nodes=%d edge_chars=%lld counts=%lld",
                        i + 1, frontiers.size(), frontier,
                        use_incremental_growth_radix ? growth_incremental.ingested_starts : growth_rebuild.ingested_starts,
                        trie.radix_count, trie.total_edge_chars, active_counts);
            if ((long long)trie.total_counts != active_counts) {
                std::printf(" flat_counts=%d", trie.total_counts);
            }
            std::printf("\n");
            run_train_epoch_on_radix_host_v2(cfg, shape, model, trie,
                                             h_weights, h_opt_m, h_opt_v,
                                             epochs, unit_limit,
                                             total_unit_steps, warmup_unit_steps,
                                             optimizer_step_index);
            double t_train1 = wall_seconds_v2();
            if (!use_incremental_growth_radix) {
                agpt_v2::free_radix_trie_structure(trie);
            }
            double t_free1 = wall_seconds_v2();
            std::printf("  growth-stage-timing %d/%zu: ingest=%.3fs materialize=%.3fs train_stage=%.3fs free_radix=%.3fs total=%.3fs\n",
                        i + 1, frontiers.size(),
                        t_ingest1 - t_stage0,
                        t_materialize1 - t_ingest1,
                        t_train1 - t_materialize1,
                        t_free1 - t_train1,
                        t_free1 - t_stage0);
        }

        if (save_path) {
            agpt_v2::save_model_weights_v2(save_path, model, h_weights);
            std::printf("  train-growth: saved final weights to %s\n", save_path);
        } else {
            std::printf("  train-growth: no --save path supplied; final weights were not written\n");
        }
        std::printf("  train-growth: completed stages=%zu optimizer_steps=%d\n",
                    frontiers.size(), optimizer_step_index);

        std::free(h_weights);
        std::free(h_opt_m);
        std::free(h_opt_v);
        agpt_v2::free_model_layout(model);
        return 0;
    }
    agpt_v2::RadixTrieStructure trie = agpt_v2::load_radix_structure_minimal(trie_dir);
    shape.seq_len = effective_seq_len_from_trie_v2(trie);
    cfg.seq_len = shape.seq_len;
    agpt_v2::ModelLayout model = agpt_v2::make_model_layout(shape);
    agpt_v2::CacheLayout cache = agpt_v2::make_cache_layout(shape);
    agpt_v2::TrainingPlan training_plan = agpt_v2::build_pd1_training_plan(trie);
    agpt_v2::ExecutionPlan plan = agpt_v2::build_execution_plan(trie, training_plan, cfg.chunk_queries);
    agpt_v2::ChunkPlanList largest_chunks = {};
    if (plan.largest_by_queries) {
        largest_chunks = agpt_v2::build_chunk_plan_for_unit(trie, *plan.largest_by_queries, cfg.chunk_queries);
    }
    agpt_v2::ChunkPlanList capacity_chunks =
        build_capacity_chunk_list_for_plan_v2(trie, training_plan, cfg.chunk_queries);
    agpt_v2::TrainerRuntimeContract runtime_contract =
        agpt_v2::build_trainer_runtime_contract(shape, cache, plan, capacity_chunks,
                                                trie.compact_slot_capacity);
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
    if (header_seq_len != shape.seq_len) {
        std::printf("  seq_len reconcile: model header says %d, trie max_depth=%d -> effective %d. Overriding.\n",
                    header_seq_len, trie.depth_file_count - 1, shape.seq_len);
    }
    std::printf("  trie: %d radix nodes, %lld edge chars, %d endpoint depths\n",
                trie.radix_count, trie.total_edge_chars, trie.depth_file_count);
    std::printf("  config: epochs=%d lr=%.6f optimizer=%s schedule=%s warmup_epochs=%d partition_depth=%d chunk_queries=%d accumulate=%s\n",
                cfg.epochs, cfg.lr, v2_optimizer_name(cfg.optimizer), v2_lr_schedule_name(cfg.lr_schedule), cfg.warmup_epochs,
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

    DeviceLossTablesV2 device_loss_tables{};
    if (instantiate_runtime) {
        agpt_v2::TrainerRuntimeV2 runtime{};
        init_trainer_runtime_v2(runtime, runtime_contract, trie);
        agpt_v2::zero_cache_runtime_v2(runtime.cache);
        float* h_weights = agpt_v2::load_model_weights_v2(model_path, model);
        AGPT_V2_CUDA_CHECK(cudaMemcpy(runtime.d_weights, h_weights,
                                      (size_t)model.total_floats * sizeof(float),
                                      cudaMemcpyHostToDevice));
        device_loss_tables = upload_loss_tables_v2(trie);
        std::printf("  runtime objects: instantiated successfully\n");
        if (instantiate_chunk_upload && have_first_chunk_meta) {
            agpt_v2::ChunkUploadRuntimeV2 upload{};
            init_chunk_upload_runtime_v2(upload, runtime_contract.chunk.node_capacity,
                                         runtime_contract.chunk.query_capacity, shape.n_heads);
            agpt_v2::ChunkDeviceMetadataV2 device_meta = upload_chunk_metadata_v2(first_chunk_meta, upload);
            (void)device_meta;
            std::printf("  chunk upload: first chunk uploaded successfully\n");
            if (run_train_epoch) {
                agpt_v2::LossTablesV2 loss_tables = make_loss_tables_view_v2(device_loss_tables);
                int epochs = cfg.epochs > 0 ? cfg.epochs : 1;
                int units_to_run = plan.training_unit_count;
                if (unit_limit > 0 && unit_limit < units_to_run) units_to_run = unit_limit;
                if (units_to_run < 1) units_to_run = 1;
                int optimizer_step_index = 0;
                AGPT_V2_CUDA_CHECK(cudaMemset(runtime.d_opt_m, 0, (size_t)model.total_floats * sizeof(float)));
                AGPT_V2_CUDA_CHECK(cudaMemset(runtime.d_opt_v, 0, (size_t)model.total_floats * sizeof(float)));
                std::printf("  train-epoch: epochs=%d units=%d accumulate=%s optimizer=%s\n",
                            epochs, units_to_run, cfg.accumulate ? "true" : "false", v2_optimizer_name(cfg.optimizer));
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
                        double unit_loss_sum = 0.0;
                        long long unit_trained = 0;
                        for (int s = 0; s < unit_chunks.chunk_count; s++) {
                            const agpt_v2::ChunkPlan& chunk = unit_chunks.chunks[s];
                            agpt_v2::ChunkMetadataV2 chunk_meta =
                                agpt_v2::build_chunk_metadata_v2(cfg, shape, trie, unit, chunk);
                            agpt_v2::ChunkDeviceMetadataV2 chunk_device_meta =
                                upload_chunk_metadata_v2(chunk_meta, upload);
                            agpt_v2::ForwardDiagDumpConfigV2 diag_dump{};
                            if (diag_probe.enabled &&
                                diag_probe.epoch == (epoch + 1) &&
                                diag_probe.root_id == unit.root_child_id) {
                                diag_dump.tensor_dir = diag_probe.tensor_dir;
                                diag_dump.epoch = epoch + 1;
                                diag_dump.root_id = unit.root_child_id;
                                diag_dump.chunk_idx = s + 1;
                                diag_dump.active = true;
                            }
                            agpt_v2::ForwardPassResult chunk_fwd =
                                agpt_v2::run_forward_prefix_v2(cfg, model, chunk_meta, chunk_device_meta, upload, loss_tables, runtime,
                                                               cfg.anc_grad ? &unit_anc : nullptr,
                                                               diag_dump.active ? &diag_dump : nullptr);
                            if (diag_dump.active && diag_probe.exit_after) {
                                std::printf("  diag-fire-exit: dumped forward tensors at epoch=%d root_id=%d chunk=%d\n",
                                            diag_dump.epoch, diag_dump.root_id, diag_dump.chunk_idx);
                                agpt_v2::free_chunk_metadata_v2(chunk_meta);
                                agpt_v2::free_unit_anc_grad_runtime_v2(unit_anc, runtime.contract);
                                agpt_v2::free_chunk_plan_list(unit_chunks);
                                if (save_path) {
                                    std::printf("  diag-fire-exit: skipping save due to early exit\n");
                                }
                                free_device_loss_tables_v2(device_loss_tables);
                                free_chunk_upload_runtime_v2(upload);
                                agpt_v2::free_trainer_runtime_v2(runtime);
                                agpt_v2::free_chunk_metadata_v2(first_chunk_meta);
                                agpt_v2::free_chunk_plan_list(largest_chunks);
                                agpt_v2::free_chunk_plan_list(capacity_chunks);
                                agpt_v2::free_training_plan(training_plan);
                                agpt_v2::free_radix_trie_structure(trie);
                                return 0;
                            }
                            agpt_v2::BackwardPassResult chunk_bwd =
                                agpt_v2::run_backward_output_head_v2(cfg, model, chunk_meta, chunk_device_meta, upload, chunk_fwd, runtime,
                                                                     cfg.anc_grad ? &unit_anc : nullptr,
                                                                     s == 0, s + 1 == unit_chunks.chunk_count);
                            (void)chunk_bwd;
                            unit_loss_sum += (double)chunk_fwd.mean_loss * (double)chunk_fwd.trained_queries;
                            unit_trained += chunk_fwd.trained_queries;
                            epoch_loss_sum += (double)chunk_fwd.mean_loss * (double)chunk_fwd.trained_queries;
                            epoch_trained += chunk_fwd.trained_queries;
                            agpt_v2::free_chunk_metadata_v2(chunk_meta);
                        }

                        scale_gradients_for_fire(runtime.cublas, runtime.d_grads, model.total_floats, unit_trained);
                        agpt_v2::OptimizerStepResult step =
                            agpt_v2::run_optimizer_step_stateful(cfg, current_lr, runtime.d_weights, runtime.d_grads,
                                                                 runtime.d_opt_m, runtime.d_opt_v,
                                                                 model.total_floats, ++optimizer_step_index);
                        double unit_mean = unit_trained > 0 ? (unit_loss_sum / (double)unit_trained) : 0.0;
                        std::printf("    unit %d/%d rc=%d chunks=%d trained_queries=%lld mean_loss=%.6f lr=%.6g step=%s\n",
                                    u + 1, units_to_run, unit.root_child_id, unit_chunks.chunk_count,
                                    unit_trained, unit_mean, current_lr, step.message);
                        agpt_v2::free_unit_anc_grad_runtime_v2(unit_anc, runtime.contract);
                        agpt_v2::free_chunk_plan_list(unit_chunks);
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
                agpt_v2::LossTablesV2 loss_tables = make_loss_tables_view_v2(device_loss_tables);
                const agpt_v2::TrainingUnit& unit = *plan.largest_by_queries;
                int n_steps = steps;
                if (n_steps > largest_chunks.chunk_count) n_steps = largest_chunks.chunk_count;
                if (n_steps < 1) n_steps = 1;
                AGPT_V2_CUDA_CHECK(cudaMemset(runtime.d_grads, 0, runtime.contract.weight_and_grad_bytes / 2));
                AGPT_V2_CUDA_CHECK(cudaMemset(runtime.d_opt_m, 0, (size_t)model.total_floats * sizeof(float)));
                AGPT_V2_CUDA_CHECK(cudaMemset(runtime.d_opt_v, 0, (size_t)model.total_floats * sizeof(float)));
                agpt_v2::UnitAncGradRuntimeV2 unit_anc{};
                if (cfg.anc_grad) {
                    agpt_v2::init_unit_anc_grad_runtime_v2(unit_anc, runtime.contract, cfg, unit, trie);
                    agpt_v2::zero_unit_anc_grad_runtime_v2(unit_anc, runtime.contract);
                }
                std::printf("  train-small: unit rc=%d chunks=%d accumulate=true optimizer=%s\n",
                            unit.root_child_id, n_steps, v2_optimizer_name(cfg.optimizer));
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
                    agpt_v2::run_optimizer_step_stateful(cfg, cfg.lr, runtime.d_weights, runtime.d_grads,
                                                         runtime.d_opt_m, runtime.d_opt_v, model.total_floats, 1);
                std::printf("  train-small-step: %s  (first_chunk_before=%.6f accumulated_unit_chunks=%d)\n",
                            step.message, first_before.mean_loss, n_steps);
                agpt_v2::free_unit_anc_grad_runtime_v2(unit_anc, runtime.contract);
            } else if (run_forward_prefix) {
                agpt_v2::LossTablesV2 loss_tables = make_loss_tables_view_v2(device_loss_tables);
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
        free_device_loss_tables_v2(device_loss_tables);
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
    agpt_v2::free_chunk_plan_list(capacity_chunks);
    agpt_v2::free_training_plan(training_plan);
    agpt_v2::free_radix_trie_structure(trie);
    agpt_v2::free_model_layout(model);
    return 0;
}
