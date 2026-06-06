#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cerrno>
#include <climits>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <yam/yam.h>

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
#include "position_sampling_v2.cuh"

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

enum class GrowthEpochScheduleV2 {
    Fixed,
    LinearRamp,
    LinearDecay,
};

static const char* growth_epoch_schedule_name_v2(GrowthEpochScheduleV2 schedule) {
    switch (schedule) {
        case GrowthEpochScheduleV2::Fixed: return "fixed";
        case GrowthEpochScheduleV2::LinearRamp: return "linear-ramp";
        case GrowthEpochScheduleV2::LinearDecay: return "linear-decay";
    }
    return "unknown";
}

static bool parse_growth_epoch_schedule_v2(const char* text, GrowthEpochScheduleV2& out) {
    if (std::strcmp(text, "fixed") == 0) {
        out = GrowthEpochScheduleV2::Fixed;
        return true;
    }
    if (std::strcmp(text, "linear") == 0 || std::strcmp(text, "linear-ramp") == 0 || std::strcmp(text, "ramp") == 0) {
        out = GrowthEpochScheduleV2::LinearRamp;
        return true;
    }
    if (std::strcmp(text, "linear-decay") == 0 || std::strcmp(text, "decay") == 0 ||
        std::strcmp(text, "reverse-ramp") == 0 || std::strcmp(text, "inverted") == 0) {
        out = GrowthEpochScheduleV2::LinearDecay;
        return true;
    }
    return false;
}

static int growth_epochs_for_stage_v2(GrowthEpochScheduleV2 schedule,
                                      int stage_index,
                                      int total_stages,
                                      int min_epochs,
                                      int max_epochs) {
    if (schedule == GrowthEpochScheduleV2::Fixed || max_epochs <= min_epochs || total_stages <= 1) {
        return max_epochs;
    }
    // Inclusive ramps: linear-ramp increases min..max; linear-decay decreases max..min.
    int effective_stage = schedule == GrowthEpochScheduleV2::LinearDecay
        ? total_stages - 1 - stage_index
        : stage_index;
    int numerator = effective_stage * (max_epochs - min_epochs);
    int denominator = total_stages - 1;
    return min_epochs + numerator / denominator;
}

static std::vector<int> make_growth_division_frontiers_v2(int final_frontier, int divisions) {
    std::vector<int> frontiers;
    if (divisions <= 0 || final_frontier <= 0) return frontiers;
    frontiers.reserve(divisions);
    int prev = 0;
    for (int i = 1; i <= divisions; i++) {
        long long v = ((long long)final_frontier * i + divisions - 1) / divisions;
        if (v <= prev) v = prev + 1;
        if (v > final_frontier) v = final_frontier;
        frontiers.push_back((int)v);
        prev = (int)v;
    }
    frontiers.erase(std::unique(frontiers.begin(), frontiers.end()), frontiers.end());
    return frontiers;
}

static bool checkpoint_epoch_requested_v2(const std::vector<int>& checkpoint_epochs, int epoch) {
    return std::find(checkpoint_epochs.begin(), checkpoint_epochs.end(), epoch) != checkpoint_epochs.end();
}

static std::string epoch_checkpoint_path_v2(const std::string& save_path, int epoch) {
    char suffix[64];
    std::snprintf(suffix, sizeof(suffix), ".epoch_%06d.model", epoch);
    const std::string model_suffix = ".model";
    if (save_path.size() >= model_suffix.size() &&
        save_path.compare(save_path.size() - model_suffix.size(), model_suffix.size(), model_suffix) == 0) {
        return save_path.substr(0, save_path.size() - model_suffix.size()) + suffix;
    }
    return save_path + suffix;
}

static void ensure_parent_dir_for_path_v2(const std::string& path) {
    std::filesystem::path p(path);
    std::filesystem::path parent = p.parent_path();
    if (!parent.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
        if (ec) {
            std::fprintf(stderr, "agpt_train_v2: cannot create model parent directory %s: %s\n",
                         parent.string().c_str(), ec.message().c_str());
            std::exit(1);
        }
    }
}

static void save_device_weights_checkpoint_v2(const char* label,
                                              int epoch,
                                              const std::string& path,
                                              const agpt_v2::ModelLayout& model,
                                              const float* d_weights) {
    std::printf("  %s: saving epoch %d checkpoint to %s\n", label, epoch, path.c_str());
    float* h_updated = (float*)std::malloc((size_t)model.total_floats * sizeof(float));
    if (!h_updated) {
        std::fprintf(stderr, "agpt_train_v2: failed to allocate checkpoint host buffer\n");
        std::exit(1);
    }
    AGPT_V2_CUDA_CHECK(cudaMemcpy(h_updated, d_weights,
                                  (size_t)model.total_floats * sizeof(float),
                                  cudaMemcpyDeviceToHost));
    ensure_parent_dir_for_path_v2(path);
    agpt_v2::save_model_weights_v2(path.c_str(), model, h_updated);
    std::free(h_updated);
    std::printf("  %s: saved epoch %d checkpoint to %s\n", label, epoch, path.c_str());
}

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

static void abort_bad_forward_v2(const char* scope,
                                 int epoch,
                                 int unit_index,
                                 int units_to_run,
                                 int root_child_id,
                                 int chunk_index,
                                 int chunk_count,
                                 const agpt_v2::ForwardPassResult& fwd) {
    std::fprintf(stderr,
                 "ERROR: v2 %s aborted: forward pass failed at epoch=%d unit=%d/%d rc=%d chunk=%d/%d: %s "
                 "trained_queries=%d trained_events=%.0f mean_loss=%.6f\n",
                 scope, epoch, unit_index + 1, units_to_run, root_child_id, chunk_index + 1, chunk_count,
                 fwd.message ? fwd.message : "unknown forward failure",
                 fwd.trained_queries, fwd.trained_events, fwd.mean_loss);
    std::exit(1);
}

static void abort_empty_training_unit_v2(const char* scope,
                                         int epoch,
                                         int unit_index,
                                         int units_to_run,
                                         int root_child_id,
                                         int chunk_count,
                                         long long trained_queries,
                                         double trained_events) {
    std::fprintf(stderr,
                 "ERROR: v2 %s aborted: unit produced no trainable events at epoch=%d unit=%d/%d rc=%d "
                 "chunks=%d trained_queries=%lld trained_events=%.0f; refusing optimizer step\n",
                 scope, epoch, unit_index + 1, units_to_run, root_child_id, chunk_count,
                 trained_queries, trained_events);
    std::exit(1);
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

static bool parse_rope_position_mode_v2(const char* text, agpt_v2::RopePositionModeV2& out) {
    if (std::strcmp(text, "depth") == 0) {
        out = agpt_v2::RopePositionModeV2::Depth;
        return true;
    }
    if (std::strcmp(text, "sampled-bin") == 0 || std::strcmp(text, "sampled_bin") == 0) {
        out = agpt_v2::RopePositionModeV2::SampledBin;
        return true;
    }
    if (std::strcmp(text, "phase-sweep") == 0 || std::strcmp(text, "phase_sweep") == 0) {
        out = agpt_v2::RopePositionModeV2::PhaseSweep;
        return true;
    }
    if (std::strcmp(text, "phase-weighted") == 0 || std::strcmp(text, "phase_weighted") == 0) {
        out = agpt_v2::RopePositionModeV2::PhaseWeighted;
        return true;
    }
    if (std::strcmp(text, "phase-conditioned") == 0 || std::strcmp(text, "phase_conditioned") == 0 ||
        std::strcmp(text, "phase-target") == 0 || std::strcmp(text, "phase_target") == 0 ||
        std::strcmp(text, "phase-conditioned-target") == 0 || std::strcmp(text, "phase_conditioned_target") == 0) {
        out = agpt_v2::RopePositionModeV2::PhaseConditioned;
        return true;
    }
    if (std::strcmp(text, "sampled-unit-phase") == 0 || std::strcmp(text, "sampled_unit_phase") == 0 ||
        std::strcmp(text, "sampled-node-phase") == 0 || std::strcmp(text, "sampled_node_phase") == 0) {
        out = agpt_v2::RopePositionModeV2::PhaseWeighted;
        return true;
    }
    return false;
}

static const char* rope_position_mode_name_v2(agpt_v2::RopePositionModeV2 mode) {
    switch (mode) {
        case agpt_v2::RopePositionModeV2::Depth: return "depth";
        case agpt_v2::RopePositionModeV2::SampledBin: return "sampled-bin";
        case agpt_v2::RopePositionModeV2::PhaseSweep: return "phase-sweep";
        case agpt_v2::RopePositionModeV2::PhaseWeighted: return "phase-weighted";
        case agpt_v2::RopePositionModeV2::PhaseConditioned: return "phase-conditioned";
    }
    return "unknown";
}

static bool rope_position_mode_uses_position_data_v2(agpt_v2::RopePositionModeV2 mode) {
    return mode == agpt_v2::RopePositionModeV2::SampledBin ||
           mode == agpt_v2::RopePositionModeV2::PhaseSweep ||
           mode == agpt_v2::RopePositionModeV2::PhaseWeighted ||
           mode == agpt_v2::RopePositionModeV2::PhaseConditioned;
}

static bool rope_position_mode_is_phase_v2(agpt_v2::RopePositionModeV2 mode) {
    return mode == agpt_v2::RopePositionModeV2::PhaseSweep ||
           mode == agpt_v2::RopePositionModeV2::PhaseWeighted ||
           mode == agpt_v2::RopePositionModeV2::PhaseConditioned;
}

static int required_rope_seq_len_v2(agpt_v2::RopePositionModeV2 mode, int context_seq_len, int position_window) {
    int required = context_seq_len > 0 ? context_seq_len : 1;
    if (position_window > required) required = position_window;
    if (rope_position_mode_is_phase_v2(mode) && position_window > 0) {
        int phase_required = position_window + (context_seq_len > 0 ? context_seq_len : 1) - 1;
        if (phase_required > required) required = phase_required;
    }
    return required;
}

#include "yaml_config_v2.cuh"

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
                                     double fire_events) {
    if (fire_events <= 0.0) return;
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
    int chunk_queries,
    const agpt_v2::SuccessorPrefixTableV2* successor_table = nullptr) {
    agpt_v2::ChunkPlanList capacity{};
    capacity.chunk_count = 1;
    capacity.chunks = (agpt_v2::ChunkPlan*)std::calloc(1, sizeof(agpt_v2::ChunkPlan));
    for (int u = 0; u < training_plan.unit_count; u++) {
        agpt_v2::ChunkPlanList chunks =
            agpt_v2::build_chunk_plan_for_unit(trie, training_plan.units[u], chunk_queries, successor_table);
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

static std::vector<agpt_v2::ChunkPlanList> build_unit_chunk_plan_cache_v2(
    const agpt_v2::RadixTrieStructure& trie,
    const agpt_v2::TrainingPlan& training_plan,
    int chunk_queries,
    const agpt_v2::SuccessorPrefixTableV2* successor_table = nullptr) {
    std::vector<agpt_v2::ChunkPlanList> cached;
    cached.reserve((size_t)training_plan.unit_count);
    for (int u = 0; u < training_plan.unit_count; u++) {
        cached.push_back(agpt_v2::build_chunk_plan_for_unit(trie, training_plan.units[u], chunk_queries, successor_table));
    }
    return cached;
}

static void free_unit_chunk_plan_cache_v2(std::vector<agpt_v2::ChunkPlanList>& cached) {
    for (agpt_v2::ChunkPlanList& chunks : cached) {
        agpt_v2::free_chunk_plan_list(chunks);
    }
    cached.clear();
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
                                             int& optimizer_step_index,
                                             const agpt_v2::PositionSamplingStageV2* pos_stage = nullptr) {
    double t_total0 = wall_seconds_v2();
    agpt_v2::CacheLayout cache = agpt_v2::make_cache_layout(shape);
    agpt_v2::TrainingPlan training_plan =
        agpt_v2::build_training_plan_for_partition_depth(trie, cfg.partition_depth);
    if (training_plan.unit_count <= 0) {
        std::printf("  train-growth-stage: skipped, no pd=%d training units at radix_nodes=%d\n",
                    cfg.partition_depth, trie.radix_count);
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
        double epoch_events = 0.0;
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
                agpt_v2::init_unit_anc_grad_runtime_v2(unit_anc, runtime.contract, cfg, unit, trie,
                                                       pos_stage, epoch, optimizer_step_index);
                agpt_v2::zero_unit_anc_grad_runtime_v2(unit_anc, runtime.contract);
            }

            double unit_loss_sum = 0.0;
            double unit_events = 0.0;
            long long unit_trained = 0;
            for (int s = 0; s < unit_chunks.chunk_count; s++) {
                const agpt_v2::ChunkPlan& chunk = unit_chunks.chunks[s];
                agpt_v2::ChunkMetadataV2 chunk_meta =
                    agpt_v2::build_chunk_metadata_v2(cfg, shape, trie, unit, chunk,
                                                     pos_stage, epoch, optimizer_step_index);
                agpt_v2::ChunkDeviceMetadataV2 chunk_device_meta =
                    upload_chunk_metadata_v2(chunk_meta, upload);
                agpt_v2::ForwardPassResult chunk_fwd =
                    agpt_v2::run_forward_prefix_v2(cfg, model, chunk_meta, chunk_device_meta,
                                                   upload, loss_tables, runtime,
                                                   cfg.anc_grad ? &unit_anc : nullptr);
                if (!chunk_fwd.ok) {
                    abort_bad_forward_v2("train-growth", epoch + 1, u, units_to_run,
                                         unit.root_child_id, s, unit_chunks.chunk_count, chunk_fwd);
                }
                agpt_v2::BackwardPassResult chunk_bwd =
                    agpt_v2::run_backward_output_head_v2(cfg, model, chunk_meta, chunk_device_meta,
                                                         upload, chunk_fwd, runtime,
                                                         cfg.anc_grad ? &unit_anc : nullptr,
                                                         s == 0, s + 1 == unit_chunks.chunk_count);
                (void)chunk_bwd;
                unit_loss_sum += (double)chunk_fwd.mean_loss * chunk_fwd.trained_events;
                unit_events += chunk_fwd.trained_events;
                unit_trained += chunk_fwd.trained_queries;
                epoch_loss_sum += (double)chunk_fwd.mean_loss * chunk_fwd.trained_events;
                epoch_events += chunk_fwd.trained_events;
                epoch_trained += chunk_fwd.trained_queries;
                agpt_v2::free_chunk_metadata_v2(chunk_meta);
            }

            if (unit_events <= 0.0 || unit_trained <= 0) {
                abort_empty_training_unit_v2("train-growth", epoch + 1, u, units_to_run,
                                             unit.root_child_id, unit_chunks.chunk_count,
                                             unit_trained, unit_events);
            }
            scale_gradients_for_fire(runtime.cublas, runtime.d_grads, model.total_floats, unit_events);
            agpt_v2::OptimizerStepResult step =
                agpt_v2::run_optimizer_step_stateful(cfg, current_lr, runtime.d_weights, runtime.d_grads,
                                                     runtime.d_opt_m, runtime.d_opt_v,
                                                     model.total_floats, ++optimizer_step_index);
            double unit_mean = unit_events > 0.0 ? unit_loss_sum / unit_events : 0.0;
            std::printf("      unit %d/%d rc=%d chunks=%d trained_queries=%lld trained_events=%.0f mean_loss=%.6f lr=%.6g step=%s\n",
                        u + 1, units_to_run, unit.root_child_id, unit_chunks.chunk_count,
                        unit_trained, unit_events, unit_mean, current_lr, step.message);
            agpt_v2::free_unit_anc_grad_runtime_v2(unit_anc, runtime.contract);
            agpt_v2::free_chunk_plan_list(unit_chunks);
        }
        double epoch_mean = epoch_events > 0.0 ? epoch_loss_sum / epoch_events : 0.0;
        std::printf("    stage-epoch %d summary trained_queries=%lld trained_events=%.0f mean_loss=%.6f\n",
                    epoch + 1, epoch_trained, epoch_events, epoch_mean);
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
    const char* config_path = nullptr;
    const char* model_path = nullptr;
    const char* trie_dir = nullptr;
    const char* corpus_path = nullptr;
    const char* growth_frontiers_arg = nullptr;
    const char* position_data_dir = nullptr;
    const char* save_path = nullptr;
    V2Mode mode = V2Mode::Plan;
    int steps = 3;
    int unit_limit = 0;
    int growth_max_depth = 0;
    int growth_min_epochs = 1;
    int growth_divisions = 0;
    int growth_final_frontier = 0;
    double growth_train_frac = 1.0;
    GrowthEpochScheduleV2 growth_epoch_schedule = GrowthEpochScheduleV2::Fixed;
    bool explicit_anc_grad = false;
    bool ablate_anc_grad = false;
    bool seed_override_set = false;
    int seed_override = 0;
    bool validate_only = false;

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

    bool saw_non_config_arg = false;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
            config_path = argv[++i];
        } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed_override = std::atoi(argv[++i]);
            seed_override_set = true;
        } else if (std::strcmp(argv[i], "--validate-only") == 0) {
            validate_only = true;
        } else if (std::strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            steps = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--units") == 0 && i + 1 < argc) {
            unit_limit = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            if (!parse_v2_mode(argv[++i], mode)) {
                std::fprintf(stderr, "agpt_train_v2: unsupported --mode value: %s\n", argv[i]);
                return 1;
            }
        } else if (std::strcmp(argv[i], "--instantiate-runtime") == 0) {
            mode = V2Mode::InstantiateRuntime;
        } else if (std::strcmp(argv[i], "--instantiate-chunk-upload") == 0) {
            mode = V2Mode::Upload;
        } else if (std::strcmp(argv[i], "--run-forward-prefix") == 0) {
            mode = V2Mode::Forward;
        } else if (std::strcmp(argv[i], "--run-backward-head") == 0) {
            mode = V2Mode::BackwardHead;
        } else {
            saw_non_config_arg = true;
        }
    }
    if (validate_only && !config_path) {
        std::fprintf(stderr, "agpt_train_v2: --validate-only requires --config\n");
        return 1;
    }
    if (config_path && saw_non_config_arg) {
        std::fprintf(stderr, "agpt_train_v2: --config may only be combined with --seed, --validate-only, --steps/--units, and diagnostic mode flags\n");
        return 1;
    }

    for (int i = 1; i < argc; i++) {
        if (config_path) {
            break;
        } else if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (std::strcmp(argv[i], "--trie-dir") == 0 && i + 1 < argc) trie_dir = argv[++i];
        else if (std::strcmp(argv[i], "--corpus") == 0 && i + 1 < argc) corpus_path = argv[++i];
        else if (std::strcmp(argv[i], "--growth-frontiers") == 0 && i + 1 < argc) growth_frontiers_arg = argv[++i];
        else if (std::strcmp(argv[i], "--growth-divisions") == 0 && i + 1 < argc) growth_divisions = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--growth-final-frontier") == 0 && i + 1 < argc) growth_final_frontier = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--growth-train-frac") == 0 && i + 1 < argc) growth_train_frac = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--position-data") == 0 && i + 1 < argc) position_data_dir = argv[++i];
        else if (std::strcmp(argv[i], "--pos-sample-seed") == 0 && i + 1 < argc) cfg.pos_sample_seed = (unsigned)std::strtoul(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "--rope-position-mode") == 0 && i + 1 < argc) {
            if (!parse_rope_position_mode_v2(argv[++i], cfg.rope_position_mode)) {
                std::fprintf(stderr, "agpt_train_v2: unsupported --rope-position-mode value: %s\n", argv[i]);
                return 1;
            }
        }
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
        else if (std::strcmp(argv[i], "--growth-min-epochs") == 0 && i + 1 < argc) growth_min_epochs = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--growth-epoch-schedule") == 0 && i + 1 < argc) {
            if (!parse_growth_epoch_schedule_v2(argv[++i], growth_epoch_schedule)) {
                std::fprintf(stderr, "agpt_train_v2: unsupported --growth-epoch-schedule value: %s\n", argv[i]);
                return 1;
            }
        }
        else if (std::strcmp(argv[i], "--growth-epoch-ramp") == 0 && i + 1 < argc) {
            if (!parse_growth_epoch_schedule_v2(argv[++i], growth_epoch_schedule)) {
                std::fprintf(stderr, "agpt_train_v2: unsupported --growth-epoch-ramp value: %s\n", argv[i]);
                return 1;
            }
        }
        else if (std::strcmp(argv[i], "--anc-grad") == 0) {
            cfg.anc_grad = true;
            explicit_anc_grad = true;
        }
        else if (std::strcmp(argv[i], "--ablate-anc-grad") == 0) ablate_anc_grad = true;
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

    YamlConfigV2 yaml_cfg;
    if (config_path) {
        if (!apply_yaml_config_v2(config_path, cfg, yaml_cfg, mode, steps, unit_limit,
                                  growth_max_depth, growth_min_epochs, growth_divisions,
                                  growth_final_frontier, growth_train_frac, growth_epoch_schedule,
                                  explicit_anc_grad, ablate_anc_grad)) {
            return 1;
        }
        if (seed_override_set) {
            yaml_cfg.seed = seed_override;
            yaml_cfg.seed_set = true;
            cfg.pos_sample_seed = (unsigned)seed_override;
        }
        model_path = yaml_cfg.model_path.c_str();
        trie_dir = yaml_cfg.trie_dir.empty() ? nullptr : yaml_cfg.trie_dir.c_str();
        corpus_path = yaml_cfg.corpus_path.c_str();
        save_path = yaml_cfg.save_path.empty() ? nullptr : yaml_cfg.save_path.c_str();
        position_data_dir = yaml_cfg.position_data_dir.empty() ? nullptr : yaml_cfg.position_data_dir.c_str();
        std::sort(yaml_cfg.checkpoint_epochs.begin(), yaml_cfg.checkpoint_epochs.end());
        yaml_cfg.checkpoint_epochs.erase(
            std::unique(yaml_cfg.checkpoint_epochs.begin(), yaml_cfg.checkpoint_epochs.end()),
            yaml_cfg.checkpoint_epochs.end());
    }

    bool has_growth_schedule =
        growth_frontiers_arg || growth_divisions > 0 || growth_final_frontier > 0 || growth_train_frac < 1.0;
    bool missing_required = !model_path ||
        (mode == V2Mode::TrainGrowth ? (!corpus_path || !has_growth_schedule) : !trie_dir);
    if (missing_required) {
        std::fprintf(stderr,
                     "Usage: agpt_train_v2 --config <path> [--seed N] [--validate-only]\n"
                     "Usage: agpt_train_v2 --model <path> --trie-dir <path>\n"
                     "       agpt_train_v2 --mode train-growth --model <path> --corpus <path>\n"
                     "  [--growth-frontiers LIST | --growth-divisions N [--growth-final-frontier N | --growth-train-frac F]]\n"
                     "  [--growth-max-depth N]\n"
                     "  [--epochs N] [--partition-depth 0|1] [--chunk-queries N] [--lr F] [--optimizer adam|sgd|momentum|rmsprop]\n"
                     "  [--momentum-beta F] [--rmsprop-beta F] [--lr-schedule constant|warmup-cosine]\n"
                     "  [--warmup-epochs N] [--growth-min-epochs N] [--growth-epoch-schedule fixed|linear-ramp|linear-decay] [--steps N]\n"
                     "  [--rope-position-mode depth|sampled-bin|phase-sweep|phase-weighted|phase-conditioned] [--position-data DIR] [--pos-sample-seed N]\n"
                     "  [--anc-grad] [--ablate-anc-grad]\n"
                     "  [--units N]\n"
                     "  [--save PATH]\n"
                     "  [--mode plan|instantiate-runtime|upload|forward|backward-head|one-step-sgd|one-step-rmsprop|multi-step-sgd|multi-step-rmsprop|save-reload-sgd|save-reload-rmsprop|train-epoch|train-small|train-growth]\n"
                     "  [--accumulate|--no-accumulate] [--quiet]\n"
                     "  compatibility aliases: [--instantiate-runtime] [--instantiate-chunk-upload]\n"
                         "                         [--run-forward-prefix] [--run-backward-head]\n");
        return 1;
    }
    if (cfg.partition_depth < 0 || cfg.partition_depth > 1) {
        std::fprintf(stderr,
                     "agpt_train_v2: only --partition-depth 0 or 1 is supported in the v2 baseline planner\n");
        return 1;
    }
    if (explicit_anc_grad && ablate_anc_grad) {
        std::fprintf(stderr,
                     "agpt_train_v2: --anc-grad and --ablate-anc-grad are mutually exclusive\n");
        return 1;
    }
    if (mode == V2Mode::TrainGrowth) {
        cfg.anc_grad = !ablate_anc_grad;
    } else if (ablate_anc_grad) {
        std::fprintf(stderr,
                     "agpt_train_v2: --ablate-anc-grad is only meaningful for --mode train-growth\n");
        return 1;
    }
    if (cfg.rope_position_mode == agpt_v2::RopePositionModeV2::SampledBin &&
        (mode != V2Mode::TrainGrowth || !position_data_dir)) {
        std::fprintf(stderr,
                     "agpt_train_v2: --rope-position-mode sampled-bin currently requires train-growth and --position-data DIR\n");
        return 1;
    }
    if ((cfg.rope_position_mode == agpt_v2::RopePositionModeV2::PhaseSweep ||
         cfg.rope_position_mode == agpt_v2::RopePositionModeV2::PhaseWeighted ||
         cfg.rope_position_mode == agpt_v2::RopePositionModeV2::PhaseConditioned) &&
        !position_data_dir) {
        std::fprintf(stderr,
                     "agpt_train_v2: --rope-position-mode %s requires --position-data DIR\n",
                     rope_position_mode_name_v2(cfg.rope_position_mode));
        return 1;
    }
    if (cfg.chunk_queries <= 0) cfg.chunk_queries = 50000;
    if (steps <= 0) steps = 3;
    if (config_path && !validate_only && !save_path) {
        std::fprintf(stderr,
                     "WARN: model.save_file not set; trained model not persisted.\n");
    }
    DiagFireProbeV2 diag_probe = read_diag_fire_probe_v2();

    agpt_v2::ModelHeader header = agpt_v2::load_model_header(model_path);
    agpt_v2::RuntimeShape shape = header.shape;
    shape.rope_seq_len = shape.seq_len;
    int header_seq_len = shape.seq_len;
    if (config_path) {
        if ((yaml_cfg.has_model_d_model && yaml_cfg.model_d_model != shape.d_model) ||
            (yaml_cfg.has_model_n_layers && yaml_cfg.model_n_layers != shape.n_layers) ||
            (yaml_cfg.has_model_n_heads && yaml_cfg.model_n_heads != shape.n_heads) ||
            (yaml_cfg.has_model_d_ff && yaml_cfg.model_d_ff != shape.d_ff) ||
            (yaml_cfg.has_model_head_dim && yaml_cfg.model_head_dim != shape.head_dim)) {
            std::fprintf(stderr,
                         "agpt_train_v2: YAML model architecture does not match checkpoint header "
                         "(checkpoint d_model=%d n_layers=%d n_heads=%d d_ff=%d head_dim=%d)\n",
                         shape.d_model, shape.n_layers, shape.n_heads, shape.d_ff, shape.head_dim);
            return 1;
        }
        if (yaml_cfg.has_seq_len && yaml_cfg.seq_len != header_seq_len) {
            std::fprintf(stderr,
                         "agpt_train_v2: YAML train.seq_len (%d) must match checkpoint seq_len (%d)\n",
                         yaml_cfg.seq_len, header_seq_len);
            return 1;
        }
    }
    cfg.d_model = shape.d_model;
    cfg.n_heads = shape.n_heads;
    cfg.n_layers = shape.n_layers;
    cfg.d_ff = shape.d_ff;
    cfg.vocab_size = shape.vocab_size;
    if (mode == V2Mode::TrainGrowth) {
        agpt_v2::PositionSamplingDataV2 pos_data;
        bool have_pos_data = false;
        if (rope_position_mode_uses_position_data_v2(cfg.rope_position_mode)) {
            pos_data = agpt_v2::load_position_sampling_data_v2(position_data_dir);
            have_pos_data = true;
            if (pos_data.prefix_table.window_size <= 0) {
                std::fprintf(stderr, "agpt_train_v2: position table has invalid window_size=%d\n",
                             pos_data.prefix_table.window_size);
                return 1;
            }
        }
        if (validate_only) {
            std::FILE* corpus_file = std::fopen(corpus_path, "rb");
            if (!corpus_file) {
                std::fprintf(stderr, "agpt_train_v2: failed to read corpus.path for YAML validation: %s\n",
                             corpus_path);
                return 1;
            }
            std::fclose(corpus_file);
            std::printf("agpt_train_v2: YAML config validated (mode=train-growth)\n");
            return 0;
        }
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
        shape.rope_seq_len = shape.seq_len;
        if (have_pos_data) {
            shape.rope_seq_len = required_rope_seq_len_v2(
                cfg.rope_position_mode, shape.seq_len, pos_data.prefix_table.window_size);
        }
        cfg.seq_len = shape.seq_len;
        cfg.rope_seq_len = shape.rope_seq_len;

        int full_starts = (int)tokens.size() - 1;
        if (growth_train_frac <= 0.0 || growth_train_frac > 1.0) {
            std::fprintf(stderr, "agpt_train_v2: --growth-train-frac must be in (0, 1]\n");
            return 1;
        }
        if (growth_final_frontier < 0) growth_final_frontier = 0;
        if (growth_final_frontier > full_starts) growth_final_frontier = full_starts;
        if (growth_final_frontier == 0) {
            growth_final_frontier = (int)std::floor((double)full_starts * growth_train_frac);
        }
        if (growth_final_frontier < 1) growth_final_frontier = 1;
        if (growth_divisions < 0) growth_divisions = 0;

        std::vector<int> frontiers;
        const char* growth_schedule_source = "explicit-frontiers";
        if (growth_frontiers_arg) {
            frontiers = agpt_v2::parse_growth_frontiers_v2(growth_frontiers_arg, full_starts);
        } else if (growth_divisions > 0) {
            frontiers = make_growth_division_frontiers_v2(growth_final_frontier, growth_divisions);
            growth_schedule_source = "generated-divisions";
        }
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
        if (growth_min_epochs <= 0) growth_min_epochs = 1;
        if (growth_epoch_schedule != GrowthEpochScheduleV2::Fixed && growth_min_epochs > epochs) {
            std::fprintf(stderr,
                         "agpt_train_v2: --growth-min-epochs (%d) must be <= --epochs (%d) for ramp schedules\n",
                         growth_min_epochs, epochs);
            std::free(h_weights);
            std::free(h_opt_m);
            std::free(h_opt_v);
            agpt_v2::free_model_layout(model);
            return 1;
        }
        int estimated_units = unit_limit > 0 ? unit_limit : cfg.vocab_size;
        long long scheduled_epochs = 0;
        for (int i = 0; i < (int)frontiers.size(); i++) {
            scheduled_epochs += growth_epochs_for_stage_v2(growth_epoch_schedule, i, (int)frontiers.size(),
                                                           growth_min_epochs, epochs);
        }
        long long total_unit_steps = scheduled_epochs * (long long)estimated_units;
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
            if (have_pos_data) {
                std::printf("  seq_len reconcile: model header says %d, growth max_depth=%d -> context %d, rope_cache=%d. Overriding context.\n",
                            header_seq_len, max_depth, shape.seq_len, shape.rope_seq_len);
            } else {
                std::printf("  seq_len reconcile: model header says %d, growth max_depth=%d -> effective %d. Overriding.\n",
                            header_seq_len, max_depth, shape.seq_len);
            }
        }
        std::printf("  corpus: %s tokens=%zu full_starts=%d\n",
                    corpus_path,
                    use_incremental_growth_radix ? growth_incremental.tokens.size() : growth_rebuild.tokens.size(),
                    full_starts);
        std::printf("  growth: stages=%zu epochs_per_stage=%d min_epochs=%d epoch_schedule=%s scheduled_epochs=%lld optimizer=%s schedule=%s materializer=%s estimated_total_unit_steps=%lld\n",
                    frontiers.size(), epochs, growth_min_epochs,
                    growth_epoch_schedule_name_v2(growth_epoch_schedule), scheduled_epochs, v2_optimizer_name(cfg.optimizer),
                    v2_lr_schedule_name(cfg.lr_schedule),
                    use_incremental_growth_radix ? "incremental-radix" : "rebuild",
                    total_unit_steps);
        std::printf("  growth-frontiers: source=%s divisions=%d train_frac=%.6f final_frontier=%d\n",
                    growth_schedule_source, growth_divisions, growth_train_frac, frontiers.back());
        std::printf("  config: lr=%.6f warmup_epochs=%d partition_depth=%d chunk_queries=%d anc_grad=%s\n",
                    cfg.lr, cfg.warmup_epochs, cfg.partition_depth, cfg.chunk_queries,
                    cfg.anc_grad ? "true" : "false");
        std::printf("  rope-position: mode=%s", rope_position_mode_name_v2(cfg.rope_position_mode));
        if (have_pos_data) {
            std::printf(" position_data=%s window=%d rope_cache=%d substrings=%d seed=%u",
                        position_data_dir, pos_data.prefix_table.window_size,
                        shape.rope_seq_len,
                        pos_data.prefix_table.substring_count, cfg.pos_sample_seed);
            if (pos_data.prefix_targets.window_size > 0) {
                std::printf(" phase_targets=%lld", (long long)pos_data.prefix_targets.total_entries);
            }
        }
        std::printf("\n");

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
            agpt_v2::PositionSamplingStageV2 pos_stage;
            const agpt_v2::PositionSamplingStageV2* pos_stage_ptr = nullptr;
            if (have_pos_data) {
                pos_stage = agpt_v2::build_position_sampling_stage_v2(pos_data, trie, cfg.pos_sample_seed);
                pos_stage_ptr = &pos_stage;
            }
            double t_materialize1 = wall_seconds_v2();
            long long active_counts = active_count_entries_v2(trie);
            std::printf("  growth-stage %d/%zu: frontier_starts=%d ingested_starts=%d radix_nodes=%d edge_chars=%lld counts=%lld",
                        i + 1, frontiers.size(), frontier,
                        use_incremental_growth_radix ? growth_incremental.ingested_starts : growth_rebuild.ingested_starts,
                        trie.radix_count, trie.total_edge_chars, active_counts);
            if ((long long)trie.total_counts != active_counts) {
                std::printf(" flat_counts=%d", trie.total_counts);
            }
            if (pos_stage_ptr) {
                std::printf(" pos_matches=%d/%d",
                            agpt_v2::count_position_sampling_matches_v2(*pos_stage_ptr),
                            trie.radix_count);
            }
            std::printf("\n");
            int epochs_this_stage = growth_epochs_for_stage_v2(
                growth_epoch_schedule, i, (int)frontiers.size(), growth_min_epochs, epochs);
            if (growth_epoch_schedule != GrowthEpochScheduleV2::Fixed) {
                std::printf("  growth-stage-epochs %d/%zu: epochs=%d min_epochs=%d max_epochs=%d schedule=%s\n",
                            i + 1, frontiers.size(), epochs_this_stage, growth_min_epochs, epochs,
                            growth_epoch_schedule_name_v2(growth_epoch_schedule));
            }
            run_train_epoch_on_radix_host_v2(cfg, shape, model, trie,
                                             h_weights, h_opt_m, h_opt_v,
                                             epochs_this_stage, unit_limit,
                                             total_unit_steps, warmup_unit_steps,
                                             optimizer_step_index, pos_stage_ptr);
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
            ensure_parent_dir_for_path_v2(save_path);
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
    shape.rope_seq_len = shape.seq_len;
    agpt_v2::SuccessorPrefixTableV2 successor_table{};
    agpt_v2::SuccessorPrefixTableV2* successor_table_ptr = nullptr;
    if (config_path && !yaml_cfg.successor_prefix_table.empty()) {
        successor_table = agpt_v2::load_successor_prefix_table_v2(
            yaml_cfg.successor_prefix_table.c_str(), trie.radix_count, shape.seq_len);
        successor_table_ptr = &successor_table;
        int successor_rope_len = successor_table.d_max * 2;
        if (successor_rope_len > shape.rope_seq_len) shape.rope_seq_len = successor_rope_len;
    }
    if (config_path && yaml_cfg.has_max_depth && yaml_cfg.max_depth != shape.seq_len) {
        std::fprintf(stderr,
                     "agpt_train_v2: YAML train.max_depth (%d) must match trie effective depth (%d)\n",
                     yaml_cfg.max_depth, shape.seq_len);
        agpt_v2::free_radix_trie_structure(trie);
        return 1;
    }
    agpt_v2::PositionSamplingDataV2 pos_data;
    bool have_pos_data = false;
    if (rope_position_mode_uses_position_data_v2(cfg.rope_position_mode)) {
        pos_data = agpt_v2::load_position_sampling_data_v2(position_data_dir);
        have_pos_data = true;
        if (pos_data.prefix_table.window_size <= 0) {
            std::fprintf(stderr, "agpt_train_v2: position table has invalid window_size=%d\n",
                         pos_data.prefix_table.window_size);
            agpt_v2::free_radix_trie_structure(trie);
            return 1;
        }
        shape.rope_seq_len = required_rope_seq_len_v2(
            cfg.rope_position_mode, shape.seq_len, pos_data.prefix_table.window_size);
        if (successor_table_ptr) {
            int successor_rope_len = successor_table.d_max * 2;
            if (successor_rope_len > shape.rope_seq_len) shape.rope_seq_len = successor_rope_len;
        }
    }
    if (validate_only) {
        std::printf("agpt_train_v2: YAML config validated (mode=%s, trie_depth=%d, context_seq_len=%d, rope_seq_len=%d, successor_prefix=%s)\n",
                    v2_mode_name(mode), effective_seq_len_from_trie_v2(trie), shape.seq_len, shape.rope_seq_len,
                    successor_table_ptr ? "true" : "false");
        agpt_v2::free_successor_prefix_table_v2(successor_table);
        agpt_v2::free_radix_trie_structure(trie);
        return 0;
    }
    cfg.seq_len = shape.seq_len;
    cfg.rope_seq_len = shape.rope_seq_len;
    agpt_v2::ModelLayout model = agpt_v2::make_model_layout(shape);
    agpt_v2::CacheLayout cache = agpt_v2::make_cache_layout(shape);
    agpt_v2::TrainingPlan training_plan =
        agpt_v2::build_training_plan_for_partition_depth(trie, cfg.partition_depth);
    agpt_v2::ExecutionPlan plan = agpt_v2::build_execution_plan(trie, training_plan, cfg.chunk_queries);
    agpt_v2::ChunkPlanList largest_chunks = {};
    if (plan.largest_by_queries) {
        largest_chunks = agpt_v2::build_chunk_plan_for_unit(trie, *plan.largest_by_queries, cfg.chunk_queries, successor_table_ptr);
    }
    agpt_v2::ChunkPlanList capacity_chunks =
        build_capacity_chunk_list_for_plan_v2(trie, training_plan, cfg.chunk_queries, successor_table_ptr);
    std::vector<agpt_v2::ChunkPlanList> unit_chunk_cache =
        build_unit_chunk_plan_cache_v2(trie, training_plan, cfg.chunk_queries, successor_table_ptr);
    agpt_v2::TrainerRuntimeContract runtime_contract =
        agpt_v2::build_trainer_runtime_contract(shape, cache, plan, capacity_chunks,
                                                trie.compact_slot_capacity);
    agpt_v2::PositionSamplingStageV2 pos_stage;
    const agpt_v2::PositionSamplingStageV2* pos_stage_ptr = nullptr;
    if (have_pos_data) {
        pos_stage = agpt_v2::build_position_sampling_stage_v2(pos_data, trie, cfg.pos_sample_seed);
        pos_stage_ptr = &pos_stage;
    }
    agpt_v2::ChunkMetadataV2 first_chunk_meta{};
    bool have_first_chunk_meta = false;
    if (plan.largest_by_queries && largest_chunks.chunk_count > 0) {
        first_chunk_meta = agpt_v2::build_chunk_metadata_v2(cfg, shape, trie, *plan.largest_by_queries,
                                                            largest_chunks.chunks[0], pos_stage_ptr,
                                                            0, 0, successor_table_ptr);
        have_first_chunk_meta = true;
    }

    std::printf("AGPT CUDA Trainer V2\n");
    std::printf("  mode: %s\n", v2_mode_name(mode));
    std::printf("  model: d=%d heads=%d layers=%d ff=%d vocab=%d seq=%d head_dim=%d\n",
                shape.d_model, shape.n_heads, shape.n_layers, shape.d_ff,
                shape.vocab_size, shape.seq_len, shape.head_dim);
    if (header_seq_len != shape.seq_len) {
        if (have_pos_data) {
            std::printf("  seq_len reconcile: model header says %d, trie max_depth=%d -> context %d, rope_cache=%d. Overriding context.\n",
                        header_seq_len, trie.depth_file_count - 1, shape.seq_len, shape.rope_seq_len);
        } else {
            std::printf("  seq_len reconcile: model header says %d, trie max_depth=%d -> effective %d. Overriding.\n",
                        header_seq_len, trie.depth_file_count - 1, shape.seq_len);
        }
    }
    std::printf("  trie: %d radix nodes, %lld edge chars, %d endpoint depths\n",
                trie.radix_count, trie.total_edge_chars, trie.depth_file_count);
    std::printf("  config: epochs=%d lr=%.6f optimizer=%s schedule=%s warmup_epochs=%d partition_depth=%d chunk_queries=%d accumulate=%s\n",
                cfg.epochs, cfg.lr, v2_optimizer_name(cfg.optimizer), v2_lr_schedule_name(cfg.lr_schedule), cfg.warmup_epochs,
                cfg.partition_depth, cfg.chunk_queries, cfg.accumulate ? "true" : "false");
    if (!yaml_cfg.checkpoint_epochs.empty()) {
        std::printf("  checkpoint_epochs:");
        for (int epoch : yaml_cfg.checkpoint_epochs) std::printf(" %d", epoch);
        std::printf("\n");
        if (!save_path) {
            std::printf("  checkpoint_epochs: ignored because model.save_file is not set\n");
        }
    }
    if (cfg.anc_grad) {
        std::printf("  anc-grad: enabled (descendant->ancestor scatter into Wk/Wv)\n");
    }
    std::printf("  rope-position: mode=%s", rope_position_mode_name_v2(cfg.rope_position_mode));
    if (have_pos_data) {
        std::printf(" position_data=%s window=%d rope_cache=%d substrings=%d seed=%u pos_matches=%d/%d",
                    position_data_dir, pos_data.prefix_table.window_size,
                    shape.rope_seq_len,
                    pos_data.prefix_table.substring_count, cfg.pos_sample_seed,
                    agpt_v2::count_position_sampling_matches_v2(*pos_stage_ptr),
                    trie.radix_count);
        if (pos_data.prefix_targets.window_size > 0) {
            std::printf(" phase_targets=%lld", (long long)pos_data.prefix_targets.total_entries);
        }
        if (cfg.rope_position_mode == agpt_v2::RopePositionModeV2::PhaseSweep ||
            cfg.rope_position_mode == agpt_v2::RopePositionModeV2::PhaseWeighted ||
            cfg.rope_position_mode == agpt_v2::RopePositionModeV2::PhaseConditioned) {
            std::printf(" phase_span=%d",
                        agpt_v2::presentation_phase_span_v2(pos_stage_ptr, trie));
            if (cfg.rope_position_offset >= 0) {
                std::printf(" fixed_offset=%d", cfg.rope_position_offset);
            } else if (cfg.rope_phase_shuffle) {
                std::printf(" phase_order=shuffle phase_order_seed=%u", cfg.rope_phase_shuffle_seed);
            } else {
                std::printf(" phase_order=sequential");
            }
        }
    }
    if (successor_table_ptr) {
        std::printf(" successor_prefix=%s deterministic=%llu skipped_fanout=%llu",
                    successor_table.mode == 2 ? "head" : "end",
                    successor_table.deterministic_count,
                    successor_table.skipped_fanout_count);
    }
    std::printf("\n");
    std::printf("  cache contract: K=%s compact_slot_indexed=%s\n",
                cache.k_space == agpt_v2::KCoordinateSpace::PostRope ? "post-RoPE" : "pre-RoPE",
                cache.compact_slot_indexed ? "true" : "false");
    std::printf("  pd=%d plan: %d training units, %lld node-visits, %lld query positions,\n"
                "           %lld compact chars, ~%lld chunks/epoch at chunk_queries=%d\n",
                cfg.partition_depth,
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
            agpt_v2::build_chunk_plan_for_unit(trie, *plan.largest_by_compact_chars, cfg.chunk_queries, successor_table_ptr);
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
                bool use_phase_mode =
                    (cfg.rope_position_mode == agpt_v2::RopePositionModeV2::PhaseSweep ||
                     cfg.rope_position_mode == agpt_v2::RopePositionModeV2::PhaseWeighted ||
                     cfg.rope_position_mode == agpt_v2::RopePositionModeV2::PhaseConditioned) &&
                    pos_stage_ptr != nullptr;
                int presentation_phase_span = use_phase_mode
                    ? agpt_v2::presentation_phase_span_v2(pos_stage_ptr, trie)
                    : -1;
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
                double train_loop_start = wall_seconds_v2();
                for (int epoch = 0; epoch < epochs; epoch++) {
                    agpt_v2::LossTablesV2 epoch_loss_tables = loss_tables;
                    int epoch_phase = -1;
                    if (use_phase_mode) {
                        epoch_phase = agpt_v2::sample_prefix_start_unit_phase_v2(
                            pos_stage_ptr, trie, 0, epoch, 0, cfg.rope_position_offset,
                            cfg.rope_phase_shuffle, cfg.rope_phase_shuffle_seed);
                        std::printf("  %s: epoch=%d presentation_start_phase=%d phase_span=%d offset=%s phase_order=%s target=%s weights=%s\n",
                                    rope_position_mode_name_v2(cfg.rope_position_mode),
                                    epoch + 1, epoch_phase, presentation_phase_span,
                                    cfg.rope_position_offset >= 0 ? "fixed" : "sweep",
                                    cfg.rope_phase_shuffle ? "shuffle" : "sequential",
                                    cfg.rope_position_mode == agpt_v2::RopePositionModeV2::PhaseConditioned ? "phase" : "global",
                                    (cfg.rope_position_mode == agpt_v2::RopePositionModeV2::PhaseWeighted ||
                                     cfg.rope_position_mode == agpt_v2::RopePositionModeV2::PhaseConditioned) ? "phase" : "global");
                    }
                    double epoch_loss_sum = 0.0;
                    double epoch_events = 0.0;
                    long long epoch_trained = 0;
                    int skipped_phase_zero_units = 0;
                    long long phase_target_nodes = 0;
                    long long phase_target_zero_nodes = 0;
                    long long phase_target_singleton_nodes = 0;
                    long long phase_target_prefix_mass = 0;
                    long long phase_target_global_mass = 0;
                    long long phase_target_local_mass = 0;
                    double phase_target_global_entropy_mass = 0.0;
                    double phase_target_local_entropy_mass = 0.0;
                    agpt_v2::zero_cache_runtime_v2(runtime.cache);
                    std::printf("  train-epoch: epoch %d/%d\n", epoch + 1, epochs);
                    for (int u = 0; u < units_to_run; u++) {
                        const agpt_v2::TrainingUnit& unit = training_plan.units[u];
                        if (cfg.rope_position_mode == agpt_v2::RopePositionModeV2::PhaseWeighted ||
                            cfg.rope_position_mode == agpt_v2::RopePositionModeV2::PhaseConditioned) {
                            int root_phase_mass = agpt_v2::prefix_position_mass_for_presentation_start_v2(
                                pos_stage_ptr, trie, unit.root_child_id, epoch_phase);
                            if (root_phase_mass <= 0) {
                                skipped_phase_zero_units++;
                                continue;
                            }
                        }
                        const agpt_v2::ChunkPlanList& unit_chunks = unit_chunk_cache[u];
                        if (unit_chunks.chunk_count <= 0) {
                            std::printf("    unit %d/%d rc=%d chunks=0 skipped\n",
                                        u + 1, units_to_run, unit.root_child_id);
                            continue;
                        }

                        long long global_unit_step = (long long)epoch * (long long)units_to_run + (long long)u;
                        float current_lr = scheduled_lr(cfg, global_unit_step, total_unit_steps, warmup_unit_steps);
                        AGPT_V2_CUDA_CHECK(cudaMemset(runtime.d_grads, 0, runtime.contract.weight_and_grad_bytes / 2));
                        agpt_v2::UnitAncGradRuntimeV2 unit_anc{};
                        if (cfg.anc_grad) {
                            agpt_v2::init_unit_anc_grad_runtime_v2(unit_anc, runtime.contract, cfg, unit, trie,
                                                                   pos_stage_ptr, epoch, optimizer_step_index);
                            agpt_v2::zero_unit_anc_grad_runtime_v2(unit_anc, runtime.contract);
                        }
                        double unit_loss_sum = 0.0;
                        double unit_events = 0.0;
                        long long unit_trained = 0;
                        for (int s = 0; s < unit_chunks.chunk_count; s++) {
                            const agpt_v2::ChunkPlan& chunk = unit_chunks.chunks[s];
                            agpt_v2::ChunkMetadataV2 chunk_meta =
                                agpt_v2::build_chunk_metadata_v2(cfg, shape, trie, unit, chunk,
                                                                 pos_stage_ptr, epoch, optimizer_step_index,
                                                                 successor_table_ptr);
                            phase_target_nodes += chunk_meta.phase_target_nodes;
                            phase_target_zero_nodes += chunk_meta.phase_target_zero_nodes;
                            phase_target_singleton_nodes += chunk_meta.phase_target_singleton_nodes;
                            phase_target_prefix_mass += chunk_meta.phase_target_prefix_mass;
                            phase_target_global_mass += chunk_meta.phase_target_global_mass;
                            phase_target_local_mass += chunk_meta.phase_target_local_mass;
                            phase_target_global_entropy_mass += chunk_meta.phase_target_global_entropy_mass;
                            phase_target_local_entropy_mass += chunk_meta.phase_target_local_entropy_mass;
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
                                agpt_v2::run_forward_prefix_v2(cfg, model, chunk_meta, chunk_device_meta, upload, epoch_loss_tables, runtime,
                                                               cfg.anc_grad ? &unit_anc : nullptr,
                                                               diag_dump.active ? &diag_dump : nullptr);
                            if (!chunk_fwd.ok) {
                                abort_bad_forward_v2("train-epoch", epoch + 1, u, units_to_run,
                                                     unit.root_child_id, s, unit_chunks.chunk_count, chunk_fwd);
                            }
                            if (diag_dump.active && diag_probe.exit_after) {
                                std::printf("  diag-fire-exit: dumped forward tensors at epoch=%d root_id=%d chunk=%d\n",
                                            diag_dump.epoch, diag_dump.root_id, diag_dump.chunk_idx);
                                agpt_v2::free_chunk_metadata_v2(chunk_meta);
                                agpt_v2::free_unit_anc_grad_runtime_v2(unit_anc, runtime.contract);
                                if (save_path) {
                                    std::printf("  diag-fire-exit: skipping save due to early exit\n");
                                }
                                free_device_loss_tables_v2(device_loss_tables);
                                free_chunk_upload_runtime_v2(upload);
                                agpt_v2::free_trainer_runtime_v2(runtime);
                                agpt_v2::free_chunk_metadata_v2(first_chunk_meta);
                                agpt_v2::free_chunk_plan_list(largest_chunks);
                                agpt_v2::free_chunk_plan_list(capacity_chunks);
                                free_unit_chunk_plan_cache_v2(unit_chunk_cache);
                                agpt_v2::free_training_plan(training_plan);
                                agpt_v2::free_radix_trie_structure(trie);
                                return 0;
                            }
                            agpt_v2::BackwardPassResult chunk_bwd =
                                agpt_v2::run_backward_output_head_v2(cfg, model, chunk_meta, chunk_device_meta, upload, chunk_fwd, runtime,
                                                                     cfg.anc_grad ? &unit_anc : nullptr,
                                                                     s == 0, s + 1 == unit_chunks.chunk_count);
                            (void)chunk_bwd;
                            unit_loss_sum += (double)chunk_fwd.mean_loss * chunk_fwd.trained_events;
                            unit_events += chunk_fwd.trained_events;
                            unit_trained += chunk_fwd.trained_queries;
                            epoch_loss_sum += (double)chunk_fwd.mean_loss * chunk_fwd.trained_events;
                            epoch_events += chunk_fwd.trained_events;
                            epoch_trained += chunk_fwd.trained_queries;
                            agpt_v2::free_chunk_metadata_v2(chunk_meta);
                        }

                        if (unit_events <= 0.0 || unit_trained <= 0) {
                            abort_empty_training_unit_v2("train-epoch", epoch + 1, u, units_to_run,
                                                         unit.root_child_id, unit_chunks.chunk_count,
                                                         unit_trained, unit_events);
                        }
                        scale_gradients_for_fire(runtime.cublas, runtime.d_grads, model.total_floats, unit_events);
                        agpt_v2::OptimizerStepResult step =
                            agpt_v2::run_optimizer_step_stateful(cfg, current_lr, runtime.d_weights, runtime.d_grads,
                                                                 runtime.d_opt_m, runtime.d_opt_v,
                                                                 model.total_floats, ++optimizer_step_index);
                        double unit_mean = unit_events > 0.0 ? (unit_loss_sum / unit_events) : 0.0;
                        std::printf("    unit %d/%d rc=%d chunks=%d trained_queries=%lld trained_events=%.0f mean_loss=%.6f lr=%.6g step=%s\n",
                                    u + 1, units_to_run, unit.root_child_id, unit_chunks.chunk_count,
                                    unit_trained, unit_events, unit_mean, current_lr, step.message);
                        agpt_v2::free_unit_anc_grad_runtime_v2(unit_anc, runtime.contract);
                    }
                    double epoch_mean = epoch_events > 0.0 ? (epoch_loss_sum / epoch_events) : 0.0;
                    std::printf("  train-epoch: epoch %d summary trained_queries=%lld trained_events=%.0f mean_loss=%.6f",
                                epoch + 1, epoch_trained, epoch_events, epoch_mean);
                    if (use_phase_mode) {
                        std::printf(" presentation_start_phase=%d skipped_zero_root_units=%d",
                                    epoch_phase, skipped_phase_zero_units);
                    }
                    if (cfg.rope_position_mode == agpt_v2::RopePositionModeV2::PhaseConditioned) {
                        double zero_pct = phase_target_nodes > 0
                            ? 100.0 * (double)phase_target_zero_nodes / (double)phase_target_nodes
                            : 0.0;
                        double singleton_pct = phase_target_nodes > 0
                            ? 100.0 * (double)phase_target_singleton_nodes / (double)phase_target_nodes
                            : 0.0;
                        double retained_pct = phase_target_global_mass > 0
                            ? 100.0 * (double)phase_target_local_mass / (double)phase_target_global_mass
                            : 0.0;
                        double prefix_retained_pct = phase_target_prefix_mass > 0
                            ? 100.0 * (double)phase_target_local_mass / (double)phase_target_prefix_mass
                            : 0.0;
                        double global_h = phase_target_global_mass > 0
                            ? phase_target_global_entropy_mass / (double)phase_target_global_mass
                            : 0.0;
                        double local_h = phase_target_local_mass > 0
                            ? phase_target_local_entropy_mass / (double)phase_target_local_mass
                            : 0.0;
                        std::printf(" phase_targets=nodes:%lld zero:%lld(%.1f%%) singleton:%lld(%.1f%%) target_mass:%lld/prefix:%lld(%.1f%%) allphase:%lld(%.1f%%) H:phase=%.4f/global=%.4f",
                                    phase_target_nodes,
                                    phase_target_zero_nodes, zero_pct,
                                    phase_target_singleton_nodes, singleton_pct,
                                    phase_target_local_mass, phase_target_prefix_mass, prefix_retained_pct,
                                    phase_target_global_mass, retained_pct,
                                    local_h, global_h);
                    }
                    std::printf("\n");
                    if (save_path && checkpoint_epoch_requested_v2(yaml_cfg.checkpoint_epochs, epoch + 1)) {
                        std::string checkpoint_path = epoch_checkpoint_path_v2(save_path, epoch + 1);
                        double checkpoint_train_wall = wall_seconds_v2() - train_loop_start;
                        std::printf("  train-epoch-checkpoint: epoch=%d train_wall_seconds=%.6f path=%s\n",
                                    epoch + 1, checkpoint_train_wall, checkpoint_path.c_str());
                        save_device_weights_checkpoint_v2("train-epoch", epoch + 1,
                                                          checkpoint_path, model, runtime.d_weights);
                    }
                }
                if (save_path) {
                    std::printf("  train-epoch: saving final weights to %s\n", save_path);
                    float* h_updated = (float*)std::malloc((size_t)model.total_floats * sizeof(float));
                    AGPT_V2_CUDA_CHECK(cudaMemcpy(h_updated, runtime.d_weights,
                                                  (size_t)model.total_floats * sizeof(float),
                                                  cudaMemcpyDeviceToHost));
                    ensure_parent_dir_for_path_v2(save_path);
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
                    agpt_v2::init_unit_anc_grad_runtime_v2(unit_anc, runtime.contract, cfg, unit, trie,
                                                           pos_stage_ptr, 0, 0);
                    agpt_v2::zero_unit_anc_grad_runtime_v2(unit_anc, runtime.contract);
                }
                std::printf("  train-small: unit rc=%d chunks=%d accumulate=true optimizer=%s\n",
                            unit.root_child_id, n_steps, v2_optimizer_name(cfg.optimizer));
                agpt_v2::ForwardPassResult first_before{};
                double unit_events = 0.0;
                long long unit_trained = 0;
                for (int s = 0; s < n_steps; s++) {
                    const agpt_v2::ChunkPlan& chunk = largest_chunks.chunks[s];
                    agpt_v2::ChunkMetadataV2 chunk_meta =
                        agpt_v2::build_chunk_metadata_v2(cfg, shape, trie, unit, chunk,
                                                         pos_stage_ptr, 0, 0, successor_table_ptr);
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
                    unit_events += chunk_fwd.trained_events;
                    unit_trained += chunk_fwd.trained_queries;
                    std::printf("    chunk %d/%d: accumulated loss=%.6f queries=%d events=%.0f nodes=%d\n",
                                s + 1, n_steps, chunk_fwd.mean_loss, chunk_meta.T_q, chunk_fwd.trained_events, chunk_meta.N);
                    agpt_v2::free_chunk_metadata_v2(chunk_meta);
                }
                scale_gradients_for_fire(runtime.cublas, runtime.d_grads, model.total_floats, unit_events);
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
                    agpt_v2::init_unit_anc_grad_runtime_v2(unit_anc, runtime.contract, cfg, *plan.largest_by_queries, trie,
                                                           pos_stage_ptr, 0, 0);
                    agpt_v2::zero_unit_anc_grad_runtime_v2(unit_anc, runtime.contract);
                }
                agpt_v2::ForwardPassResult fwd =
                    agpt_v2::run_forward_prefix_v2(cfg, model, first_chunk_meta, device_meta, upload, loss_tables, runtime,
                                                   cfg.anc_grad ? &unit_anc : nullptr);
                std::printf("  forward prefix: %s  (trained_queries=%d trained_events=%.0f mean_loss=%.6f)\n",
                            fwd.message, fwd.trained_queries, fwd.trained_events, fwd.mean_loss);
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
                        scale_gradients_for_fire(runtime.cublas, runtime.d_grads, model.total_floats, fwd.trained_events);
                        agpt_v2::OptimizerStepResult step =
                            agpt_v2::run_optimizer_step_sgd(cfg, runtime.d_weights, runtime.d_grads, model.total_floats);
                        agpt_v2::ForwardPassResult fwd_after =
                            agpt_v2::run_forward_prefix_v2(cfg, model, first_chunk_meta, device_meta, upload, loss_tables, runtime);
                        std::printf("  one-step-sgd: %s  (loss_before=%.6f loss_after=%.6f delta=%.6f)\n",
                                    step.message, fwd.mean_loss, fwd_after.mean_loss, fwd_after.mean_loss - fwd.mean_loss);
                    } else if (run_one_step_rmsprop) {
                        scale_gradients_for_fire(runtime.cublas, runtime.d_grads, model.total_floats, fwd.trained_events);
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
                            scale_gradients_for_fire(runtime.cublas, runtime.d_grads, model.total_floats, cur_fwd.trained_events);
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
                            scale_gradients_for_fire(runtime.cublas, runtime.d_grads, model.total_floats, cur_fwd.trained_events);
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
                        scale_gradients_for_fire(runtime.cublas, runtime.d_grads, model.total_floats, fwd.trained_events);
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
                        scale_gradients_for_fire(runtime.cublas, runtime.d_grads, model.total_floats, fwd.trained_events);
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
    std::printf("  status: v2 currently validates file formats, plans baseline pd=0/pd=1 execution,\n"
                "          and exercises the full-depth chunk upload/cache/forward/loss path,\n"
                "          plus output-head/final-LN backward, train-epoch/train-small accumulation, and one-step SGD/RMSProp/multi-step/save-reload sanity modes when requested.\n");

    (void)model;
    if (have_first_chunk_meta) agpt_v2::free_chunk_metadata_v2(first_chunk_meta);
    agpt_v2::free_chunk_plan_list(largest_chunks);
    agpt_v2::free_chunk_plan_list(capacity_chunks);
    free_unit_chunk_plan_cache_v2(unit_chunk_cache);
    agpt_v2::free_training_plan(training_plan);
    agpt_v2::free_successor_prefix_table_v2(successor_table);
    agpt_v2::free_radix_trie_structure(trie);
    agpt_v2::free_model_layout(model);
    return 0;
}
