#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <random>
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

// Gradient-population dump (rnd/gradient-population). When AGPT_GRAD_DUMP_DIR
// is set in train-epoch mode, every training unit's fired (event-mean)
// gradient is appended as one float32 row to <dir>/grads.f32 with a metadata
// line in <dir>/units.tsv, and the optimizer step is SKIPPED so every unit
// answers against the same frozen weights. AGPT_GRAD_DUMP_APPLY=1 keeps the
// step (dump the sequential trajectory instead of the frozen fan-out).
struct GradDumpProbeV2 {
    const char* dir = nullptr;
    bool apply_step = false;
    bool enabled = false;
    FILE* grads = nullptr;
    FILE* units = nullptr;
    std::vector<float> host;
};

static GradDumpProbeV2 read_grad_dump_probe_v2() {
    GradDumpProbeV2 cfg{};
    cfg.dir = std::getenv("AGPT_GRAD_DUMP_DIR");
    if (!cfg.dir || !cfg.dir[0]) return cfg;
    cfg.apply_step = read_env_flag_v2("AGPT_GRAD_DUMP_APPLY");
    cfg.enabled = true;
    return cfg;
}

static void grad_dump_write_layout_v2(const GradDumpProbeV2& probe,
                                      const agpt_v2::ModelLayout& model,
                                      const agpt_v2::TrainerConfig& cfg,
                                      int unit_count) {
    std::string path = std::string(probe.dir) + "/layout.json";
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) { std::perror("grad-dump: layout.json"); std::exit(1); }
    const agpt_v2::RuntimeShape& s = model.shape;
    std::fprintf(f, "{\n  \"total_floats\": %d,\n  \"unit_count\": %d,\n  \"partition_depth\": %d,\n",
                 model.total_floats, unit_count, cfg.partition_depth);
    std::fprintf(f, "  \"apply_step\": %s,\n  \"anc_grad\": %s,\n",
                 probe.apply_step ? "true" : "false", cfg.anc_grad ? "true" : "false");
    std::fprintf(f, "  \"shape\": {\"d_model\": %d, \"n_heads\": %d, \"n_layers\": %d, \"d_ff\": %d, \"vocab\": %d, \"seq_len\": %d},\n",
                 s.d_model, s.n_heads, s.n_layers, s.d_ff, s.vocab_size, s.seq_len);
    std::fprintf(f, "  \"sections\": [\n");
    int D = s.d_model, F = s.d_ff, V = s.vocab_size;
    std::fprintf(f, "    [\"token_emb\", %d, %d]", model.token_emb, V * D);
    for (int l = 0; l < s.n_layers; l++) {
        std::fprintf(f, ",\n    [\"l%d.wq_w\", %d, %d],\n    [\"l%d.wq_b\", %d, %d]", l, model.wq_w[l], D * D, l, model.wq_b[l], D);
        std::fprintf(f, ",\n    [\"l%d.wk_w\", %d, %d],\n    [\"l%d.wk_b\", %d, %d]", l, model.wk_w[l], D * D, l, model.wk_b[l], D);
        std::fprintf(f, ",\n    [\"l%d.wv_w\", %d, %d],\n    [\"l%d.wv_b\", %d, %d]", l, model.wv_w[l], D * D, l, model.wv_b[l], D);
        std::fprintf(f, ",\n    [\"l%d.wo_w\", %d, %d],\n    [\"l%d.wo_b\", %d, %d]", l, model.wo_w[l], D * D, l, model.wo_b[l], D);
        std::fprintf(f, ",\n    [\"l%d.ln1_gamma\", %d, %d],\n    [\"l%d.ln1_beta\", %d, %d]", l, model.ln1_gamma[l], D, l, model.ln1_beta[l], D);
        std::fprintf(f, ",\n    [\"l%d.l1_w\", %d, %d],\n    [\"l%d.l1_b\", %d, %d]", l, model.l1_w[l], D * F, l, model.l1_b[l], F);
        std::fprintf(f, ",\n    [\"l%d.l2_w\", %d, %d],\n    [\"l%d.l2_b\", %d, %d]", l, model.l2_w[l], F * D, l, model.l2_b[l], D);
        std::fprintf(f, ",\n    [\"l%d.ln2_gamma\", %d, %d],\n    [\"l%d.ln2_beta\", %d, %d]", l, model.ln2_gamma[l], D, l, model.ln2_beta[l], D);
    }
    std::fprintf(f, ",\n    [\"final_gamma\", %d, %d],\n    [\"final_beta\", %d, %d]", model.final_gamma, D, model.final_beta, D);
    std::fprintf(f, ",\n    [\"out_w\", %d, %d],\n    [\"out_b\", %d, %d]\n  ]\n}\n", model.out_w, D * V, model.out_b, V);
    std::fclose(f);
}

static void grad_dump_open_v2(GradDumpProbeV2& probe, int total_floats) {
    std::string gpath = std::string(probe.dir) + "/grads.f32";
    std::string upath = std::string(probe.dir) + "/units.tsv";
    probe.grads = std::fopen(gpath.c_str(), "wb");
    probe.units = std::fopen(upath.c_str(), "w");
    if (!probe.grads || !probe.units) { std::perror("grad-dump: open"); std::exit(1); }
    std::fprintf(probe.units,
                 "row\tepoch\tunit_index\tanchor_id\troot_child_id\tanchor_depth\tanchor_endpoint_depth\t"
                 "context_tokens\tnode_count\tquery_count\ttrained_queries\ttrained_events\tmean_loss\tgrad_l2\n");
    probe.host.assign((size_t)total_floats, 0.0f);
}

// Token path root -> anchor, truncated to `depth` chars (the partition
// prefix), as comma-separated token ids. Decoded offline via the vocab.
static std::string grad_dump_context_tokens_v2(const agpt_v2::RadixTrieStructure& trie,
                                               int anchor, int depth) {
    std::string out;
    if (anchor <= 0 || anchor >= trie.radix_count) return out;
    std::vector<int> toks;
    int a0 = trie.ancestor_char_offsets[anchor];
    int a1 = trie.ancestor_char_offsets[anchor + 1];
    for (int i = a0; i < a1; i++) toks.push_back(trie.edge_tokens_flat[trie.ancestor_char_ids[i]]);
    for (int e = 0; e < trie.edge_lens[anchor]; e++) toks.push_back(trie.edge_tokens_flat[trie.edge_starts[anchor] + e]);
    if (depth > 0 && (int)toks.size() > depth) toks.resize((size_t)depth);
    char buf[16];
    for (size_t i = 0; i < toks.size(); i++) {
        std::snprintf(buf, sizeof(buf), "%s%d", i ? "," : "", toks[i]);
        out += buf;
    }
    return out;
}

static void grad_dump_row_v2(GradDumpProbeV2& probe, long long row, int epoch,
                             const agpt_v2::TrainingUnit& unit,
                             const agpt_v2::RadixTrieStructure& trie,
                             int partition_depth,
                             const float* d_grads, int total_floats,
                             long long trained_queries, double trained_events, double mean_loss) {
    AGPT_V2_CUDA_CHECK(cudaMemcpy(probe.host.data(), d_grads,
                                  (size_t)total_floats * sizeof(float), cudaMemcpyDeviceToHost));
    double l2 = 0.0;
    for (int i = 0; i < total_floats; i++) l2 += (double)probe.host[i] * (double)probe.host[i];
    l2 = std::sqrt(l2);
    if (std::fwrite(probe.host.data(), sizeof(float), (size_t)total_floats, probe.grads) != (size_t)total_floats) {
        std::perror("grad-dump: write grads.f32"); std::exit(1);
    }
    int anchor = unit.anchor_id;
    int anchor_depth = 0, anchor_endpoint = 0;
    if (anchor > 0 && anchor < trie.radix_count) {
        anchor_depth = trie.edge_first_char_depths[anchor];
        anchor_endpoint = anchor_depth + trie.edge_lens[anchor] - 1;
    }
    std::string ctx = grad_dump_context_tokens_v2(trie, anchor, partition_depth);
    std::fprintf(probe.units, "%lld\t%d\t%d\t%d\t%d\t%d\t%d\t%s\t%d\t%lld\t%lld\t%.0f\t%.6f\t%.8g\n",
                 row, epoch, unit.unit_index, anchor, unit.root_child_id, anchor_depth, anchor_endpoint,
                 ctx.c_str(), unit.node_count, unit.query_count, trained_queries, trained_events, mean_loss, l2);
}

static void grad_dump_close_v2(GradDumpProbeV2& probe) {
    if (probe.grads) std::fclose(probe.grads);
    if (probe.units) std::fclose(probe.units);
    probe.grads = nullptr;
    probe.units = nullptr;
}

// ---------------------------------------------------------------------------
// L-BFGS on the exact aggregated gradient (rnd/gradient-population, Exp 7).
// In train-epoch mode with optimizer=lbfgs, every "epoch" is one function
// evaluation: a full pass over all units with d_grads accumulated (no
// per-unit scaling, no per-unit step). All vectors live on the device; the
// two-loop recursion and line search use cuBLAS and only move scalars.
// ---------------------------------------------------------------------------
struct LbfgsStateV2 {
    int P = 0;
    int m = 0;
    float* d_theta = nullptr;   // accepted iterate
    float* d_g = nullptr;       // gradient at accepted iterate
    float* d_dir = nullptr;     // search direction
    float* d_q = nullptr;       // two-loop scratch
    float* d_acc = nullptr;     // gradient accumulated across units within one pass
    float* d_S = nullptr;       // m x P ring buffer of s_i
    float* d_Y = nullptr;       // m x P ring buffer of y_i
    std::vector<double> rho, alpha_i;
    int count = 0, head = 0;    // ring buffer fill and next slot
    bool have_accepted = false;
    double f_acc = 0.0;
    double alpha = 1.0;
    int backtracks = 0;
    int iterations = 0;         // accepted steps
    int evaluations = 0;        // passes
    int resets = 0;
};

static void lbfgs_init_v2(LbfgsStateV2& st, int P, int m) {
    st.P = P; st.m = m < 1 ? 1 : m;
    size_t vb = (size_t)P * sizeof(float);
    AGPT_V2_CUDA_CHECK(cudaMalloc(&st.d_theta, vb));
    AGPT_V2_CUDA_CHECK(cudaMalloc(&st.d_g, vb));
    AGPT_V2_CUDA_CHECK(cudaMalloc(&st.d_dir, vb));
    AGPT_V2_CUDA_CHECK(cudaMalloc(&st.d_q, vb));
    AGPT_V2_CUDA_CHECK(cudaMalloc(&st.d_acc, vb));
    AGPT_V2_CUDA_CHECK(cudaMalloc(&st.d_S, vb * (size_t)st.m));
    AGPT_V2_CUDA_CHECK(cudaMalloc(&st.d_Y, vb * (size_t)st.m));
    st.rho.assign((size_t)st.m, 0.0);
    st.alpha_i.assign((size_t)st.m, 0.0);
}

static void lbfgs_free_v2(LbfgsStateV2& st) {
    cudaFree(st.d_theta); cudaFree(st.d_g); cudaFree(st.d_dir); cudaFree(st.d_q); cudaFree(st.d_acc);
    cudaFree(st.d_S); cudaFree(st.d_Y);
    st = LbfgsStateV2{};
}

static double lbfgs_dot_v2(cublasHandle_t h, const float* a, const float* b, int n) {
    float r = 0.0f;
    AGPT_V2_CUBLAS_CHECK(cublasSdot(h, n, a, 1, b, 1, &r));
    return (double)r;
}
static double lbfgs_nrm2_v2(cublasHandle_t h, const float* a, int n) {
    float r = 0.0f;
    AGPT_V2_CUBLAS_CHECK(cublasSnrm2(h, n, a, 1, &r));
    return (double)r;
}
static void lbfgs_axpy_v2(cublasHandle_t h, float a, const float* x, float* y, int n) {
    AGPT_V2_CUBLAS_CHECK(cublasSaxpy(h, n, &a, x, 1, y, 1));
}
static void lbfgs_scal_v2(cublasHandle_t h, float a, float* x, int n) {
    AGPT_V2_CUBLAS_CHECK(cublasSscal(h, n, &a, x, 1));
}
static void lbfgs_copy_v2(cublasHandle_t h, const float* x, float* y, int n) {
    AGPT_V2_CUBLAS_CHECK(cublasScopy(h, n, x, 1, y, 1));
}

// d_dir <- -H_k g using the two-loop recursion over the ring buffer.
static void lbfgs_direction_v2(cublasHandle_t h, LbfgsStateV2& st) {
    int P = st.P;
    lbfgs_copy_v2(h, st.d_g, st.d_q, P);
    // newest -> oldest
    for (int k = 0; k < st.count; k++) {
        int i = (st.head - 1 - k + st.m * 2) % st.m;
        const float* Si = st.d_S + (size_t)i * P;
        const float* Yi = st.d_Y + (size_t)i * P;
        st.alpha_i[(size_t)i] = st.rho[(size_t)i] * lbfgs_dot_v2(h, Si, st.d_q, P);
        lbfgs_axpy_v2(h, (float)(-st.alpha_i[(size_t)i]), Yi, st.d_q, P);
    }
    if (st.count > 0) {
        int newest = (st.head - 1 + st.m) % st.m;
        const float* Sn = st.d_S + (size_t)newest * P;
        const float* Yn = st.d_Y + (size_t)newest * P;
        double gamma = lbfgs_dot_v2(h, Sn, Yn, P) / std::max(lbfgs_dot_v2(h, Yn, Yn, P), 1e-30);
        lbfgs_scal_v2(h, (float)gamma, st.d_q, P);
    }
    // oldest -> newest
    for (int k = st.count - 1; k >= 0; k--) {
        int i = (st.head - 1 - k + st.m * 2) % st.m;
        const float* Si = st.d_S + (size_t)i * P;
        const float* Yi = st.d_Y + (size_t)i * P;
        double beta = st.rho[(size_t)i] * lbfgs_dot_v2(h, Yi, st.d_q, P);
        lbfgs_axpy_v2(h, (float)(st.alpha_i[(size_t)i] - beta), Si, st.d_q, P);
    }
    lbfgs_copy_v2(h, st.d_q, st.d_dir, P);
    lbfgs_scal_v2(h, -1.0f, st.d_dir, P);
    double dg = lbfgs_dot_v2(h, st.d_dir, st.d_g, P);
    if (!(dg < 0.0)) {  // not a descent direction (or NaN): reset to steepest descent
        st.count = 0; st.head = 0; st.resets++;
        lbfgs_copy_v2(h, st.d_g, st.d_dir, P);
        lbfgs_scal_v2(h, -1.0f, st.d_dir, P);
    }
}

// d_weights <- theta_acc + alpha * dir  (the next trial point)
static void lbfgs_set_trial_v2(cublasHandle_t h, LbfgsStateV2& st, float* d_weights) {
    lbfgs_copy_v2(h, st.d_theta, d_weights, st.P);
    lbfgs_axpy_v2(h, (float)st.alpha, st.d_dir, d_weights, st.P);
}

// One evaluation just finished at d_weights with mean gradient in d_grads and
// mean loss f. Decide accept / backtrack, update history, and set the next
// trial point into d_weights. Returns a status string for the log line.
static const char* lbfgs_update_v2(cublasHandle_t h, const agpt_v2::TrainerConfig& cfg,
                                   LbfgsStateV2& st, float* d_weights, float* d_grads, double f) {
    int P = st.P;
    st.evaluations++;
    if (!st.have_accepted) {
        lbfgs_copy_v2(h, d_weights, st.d_theta, P);
        lbfgs_copy_v2(h, d_grads, st.d_g, P);
        st.f_acc = f;
        st.have_accepted = true;
        lbfgs_direction_v2(h, st);
        double gn = lbfgs_nrm2_v2(h, st.d_g, P);
        st.alpha = std::min(1.0, 1.0 / std::max(gn, 1e-12));
        st.backtracks = 0;
        lbfgs_set_trial_v2(h, st, d_weights);
        return "init";
    }
    double dg = lbfgs_dot_v2(h, st.d_dir, st.d_g, P);
    bool armijo = std::isfinite(f) && (f <= st.f_acc + (double)cfg.lbfgs_c1 * st.alpha * dg);
    if (armijo) {
        // s = alpha*dir ; y = g_trial - g_acc
        int slot = st.head;
        float* Ss = st.d_S + (size_t)slot * P;
        float* Ys = st.d_Y + (size_t)slot * P;
        lbfgs_copy_v2(h, st.d_dir, Ss, P);
        lbfgs_scal_v2(h, (float)st.alpha, Ss, P);
        lbfgs_copy_v2(h, d_grads, Ys, P);
        lbfgs_axpy_v2(h, -1.0f, st.d_g, Ys, P);
        double sy = lbfgs_dot_v2(h, Ss, Ys, P);
        double sn = lbfgs_nrm2_v2(h, Ss, P), yn = lbfgs_nrm2_v2(h, Ys, P);
        bool curv_ok = sy > 1e-10 * sn * yn;
        if (curv_ok) {
            st.rho[(size_t)slot] = 1.0 / sy;
            st.head = (st.head + 1) % st.m;
            if (st.count < st.m) st.count++;
        }
        lbfgs_copy_v2(h, d_weights, st.d_theta, P);   // accept trial
        lbfgs_copy_v2(h, d_grads, st.d_g, P);
        st.f_acc = f;
        st.iterations++;
        lbfgs_direction_v2(h, st);
        st.alpha = 1.0;
        st.backtracks = 0;
        lbfgs_set_trial_v2(h, st, d_weights);
        return curv_ok ? "accept" : "accept(no-curv-pair)";
    }
    st.backtracks++;
    if (st.backtracks > cfg.lbfgs_max_backtracks) {
        // give up on this direction: reset history, steepest descent, small step
        st.count = 0; st.head = 0; st.resets++;
        lbfgs_copy_v2(h, st.d_g, st.d_dir, P);
        lbfgs_scal_v2(h, -1.0f, st.d_dir, P);
        double gn = lbfgs_nrm2_v2(h, st.d_g, P);
        st.alpha = std::min(1.0, 1.0 / std::max(gn, 1e-12));
        st.backtracks = 0;
        lbfgs_set_trial_v2(h, st, d_weights);
        return "reset";
    }
    st.alpha *= 0.5;
    lbfgs_set_trial_v2(h, st, d_weights);
    return "backtrack";
}

// ---------------------------------------------------------------------------
// Exact ancestor backward, second pass (experimental.anc_grad_exact).
// After a unit's normal chunks (loss backward, ancestor K/V gradient
// accumulated in unit_anc), revisit the unit's internal nodes (edge_mass > 1:
// the only nodes any descendant attends to), grouped by endpoint depth,
// deepest group first. Nodes with equal endpoint depth are never
// ancestor/descendant, so a group is safe to batch. Each group is chunked
// against the runtime's fixed node/query/kv capacities, its forward is
// recomputed, and run_backward_ancestor_path_v2 injects the accumulated
// gradient. Must run before the unit's fire scaling / optimizer step.
// ---------------------------------------------------------------------------
static double wall_seconds_v2();

struct AncExactStatsV2 {
    int groups = 0;
    int chunks = 0;
    long long nodes = 0;
    long long queries = 0;
    double seconds = 0.0;
};

static AncExactStatsV2 run_anc_exact_pass_v2(const agpt_v2::TrainerConfig& cfg,
                                             const agpt_v2::RuntimeShape& shape,
                                             const agpt_v2::ModelLayout& model,
                                             const agpt_v2::RadixTrieStructure& trie,
                                             const agpt_v2::TrainingUnit& unit,
                                             const agpt_v2::PositionSamplingStageV2* pos_stage,
                                             int epoch,
                                             int optimizer_step_index,
                                             agpt_v2::ChunkUploadRuntimeV2& upload,
                                             const agpt_v2::LossTablesV2& loss_tables,
                                             agpt_v2::TrainerRuntimeV2& runtime,
                                             agpt_v2::UnitAncGradRuntimeV2& anc,
                                             const agpt_v2::TrainerRuntimeContract& contract) {
    AncExactStatsV2 st;
    double t0 = wall_seconds_v2();
    if (!anc.enabled || anc.subtree_compact_chars <= 0) return st;

    int max_depth = 0;
    for (int i = 0; i < unit.node_count; i++) {
        int r = unit.radix_ids[i];
        if (trie.edge_mass[r] <= 1) continue;
        int ep = trie.edge_first_char_depths[r] + trie.edge_lens[r] - 1;
        if (ep > max_depth) max_depth = ep;
    }
    std::vector<std::vector<int>> by_depth((size_t)max_depth + 1);
    std::vector<std::vector<unsigned char>> ctx_by_depth((size_t)max_depth + 1);
    for (int i = 0; i < unit.node_count; i++) {
        int r = unit.radix_ids[i];
        if (trie.edge_mass[r] <= 1) continue;
        int ep = trie.edge_first_char_depths[r] + trie.edge_lens[r] - 1;
        if (ep < 0) continue;
        by_depth[(size_t)ep].push_back(r);
        ctx_by_depth[(size_t)ep].push_back(unit.context_only ? unit.context_only[i] : (unsigned char)0);
    }

    const int node_cap = contract.chunk.node_capacity;
    const int query_cap = contract.chunk.query_capacity;
    const long long kv_cap = contract.chunk.kv_capacity;
    const int max_kv_cap = contract.chunk.max_kv_len;

    for (int d = max_depth; d >= 0; d--) {
        std::vector<int>& ids = by_depth[(size_t)d];
        if (ids.empty()) continue;
        agpt_v2::TrainingUnit g{};
        g.kind = unit.kind;
        g.unit_index = unit.unit_index;
        g.root_child_id = unit.root_child_id;
        g.anchor_id = unit.anchor_id;
        g.node_count = (int)ids.size();
        g.radix_ids = ids.data();
        g.context_only = ctx_by_depth[(size_t)d].data();
        st.groups++;

        int start = 0;
        while (start < g.node_count) {
            long long q_sum = 0, kv_sum = 0, compact_sum = 0;
            int max_kv_len = 0;
            int end = start;
            while (end < g.node_count) {
                int r = g.radix_ids[end];
                int q_next = trie.edge_lens[r];
                int kv_next = trie.edge_first_char_depths[r] + trie.edge_lens[r] - 1;
                if (end > start && (end - start + 1 > node_cap || q_sum + q_next > query_cap ||
                                    kv_sum + kv_next > kv_cap)) break;
                q_sum += q_next;
                kv_sum += kv_next;
                compact_sum += trie.edge_lens[r];
                if (kv_next > max_kv_len) max_kv_len = kv_next;
                end++;
            }
            if (end - start > node_cap || q_sum > query_cap || kv_sum > kv_cap || max_kv_len > max_kv_cap) {
                std::fprintf(stderr, "anc-exact: node %d exceeds chunk capacity (nodes %d/%d q %lld/%d kv %lld/%lld maxkv %d/%d)\n",
                             g.radix_ids[start], end - start, node_cap, q_sum, query_cap, kv_sum, kv_cap,
                             max_kv_len, max_kv_cap);
                std::exit(1);
            }
            agpt_v2::ChunkPlan chunk{};
            chunk.chunk_index = st.chunks;
            chunk.start_node_index = start;
            chunk.end_node_index = end;
            chunk.node_count = end - start;
            chunk.query_count = q_sum;
            chunk.kv_count = kv_sum;
            chunk.compact_char_count = compact_sum;
            chunk.max_kv_len = max_kv_len;

            agpt_v2::ChunkMetadataV2 meta =
                agpt_v2::build_chunk_metadata_v2(cfg, shape, trie, g, chunk, pos_stage,
                                                 epoch, optimizer_step_index, nullptr, nullptr);
            agpt_v2::ChunkDeviceMetadataV2 dmeta = upload_chunk_metadata_v2(meta, upload);
            agpt_v2::ForwardPassResult fwd =
                agpt_v2::run_forward_prefix_v2(cfg, model, meta, dmeta, upload, loss_tables, runtime, &anc, nullptr);
            if (!fwd.ok) {
                std::fprintf(stderr, "anc-exact: recomputed forward failed (%s) at depth %d\n", fwd.message, d);
                std::exit(1);
            }
            agpt_v2::run_backward_ancestor_path_v2(cfg, model, meta, dmeta, upload, runtime, &anc);
            agpt_v2::free_chunk_metadata_v2(meta);
            st.chunks++;
            st.nodes += chunk.node_count;
            st.queries += q_sum;
            start = end;
        }
        // g borrows the vectors' storage; nothing to free.
    }
    st.seconds = wall_seconds_v2() - t0;
    return st;
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
        case agpt_v2::OptimizerKind::LBFGS: return "lbfgs";
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
    if (std::strcmp(text, "lbfgs") == 0 || std::strcmp(text, "l-bfgs") == 0) {
        out = agpt_v2::OptimizerKind::LBFGS;
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

static const char* lightning_anchor_mode_name_v2(agpt_v2::LightningAnchorModeV2 mode) {
    switch (mode) {
        case agpt_v2::LightningAnchorModeV2::TraversalStop: return "traversal-stop";
        case agpt_v2::LightningAnchorModeV2::RandomDescendants: return "random-descendants";
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
    float min_ratio = cfg.lr_min_ratio;
    if (min_ratio < 0.0f) min_ratio = 0.0f;
    if (min_ratio > 1.0f) min_ratio = 1.0f;
    return cfg.lr * (min_ratio + (1.0f - min_ratio) * cosine);
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

static agpt_v2::ChunkPlanList build_lightning_capacity_chunk_list_v2(
    const agpt_v2::TrainerConfig& cfg,
    const agpt_v2::RuntimeShape& shape,
    const agpt_v2::SuccessorPrefixTableV2* successor_table = nullptr) {
    int chunk_queries = cfg.chunk_queries > 0 ? cfg.chunk_queries : 50000;
    int context = shape.seq_len > 0 ? shape.seq_len : 1;
    int successor_context = successor_table ? context : 0;
    int max_kv_len = context + successor_context;
    if (max_kv_len < 1) max_kv_len = 1;

    long long query_cap = (long long)chunk_queries + (long long)max_kv_len;
    long long node_cap_ll = query_cap;
    if (node_cap_ll < 1) node_cap_ll = 1;
    if (node_cap_ll > 1000000000LL) node_cap_ll = 1000000000LL;

    agpt_v2::ChunkPlanList capacity{};
    capacity.chunk_count = 1;
    capacity.chunks = (agpt_v2::ChunkPlan*)std::calloc(1, sizeof(agpt_v2::ChunkPlan));
    capacity.chunks[0].chunk_index = 0;
    capacity.chunks[0].start_node_index = 0;
    capacity.chunks[0].end_node_index = (int)node_cap_ll;
    capacity.chunks[0].node_count = (int)node_cap_ll;
    capacity.chunks[0].query_count = query_cap;
    capacity.chunks[0].kv_count = query_cap * (long long)max_kv_len;
    capacity.chunks[0].compact_char_count = query_cap;
    capacity.chunks[0].max_kv_len = max_kv_len;
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
    agpt_v2::TrainingPlan training_plan = cfg.lightning_enabled
        ? agpt_v2::build_lightning_training_plan_v2(trie, cfg)
        : agpt_v2::build_training_plan_for_partition_depth(trie, cfg.partition_depth);
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
    const char* target_sidecar_path = nullptr;
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
        else if (std::strcmp(argv[i], "--target-sidecar") == 0 && i + 1 < argc) target_sidecar_path = argv[++i];
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
        else if (std::strcmp(argv[i], "--lr-min-ratio") == 0 && i + 1 < argc) cfg.lr_min_ratio = std::atof(argv[++i]);
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
        target_sidecar_path = yaml_cfg.target_sidecar.empty() ? nullptr : yaml_cfg.target_sidecar.c_str();
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
                     "  [--warmup-epochs N] [--lr-min-ratio F] [--growth-min-epochs N] [--growth-epoch-schedule fixed|linear-ramp|linear-decay] [--steps N]\n"
                     "  [--rope-position-mode depth|sampled-bin|phase-sweep|phase-weighted|phase-conditioned] [--position-data DIR] [--pos-sample-seed N]\n"
                     "  [--target-sidecar PATH]\n"
                     "  [--anc-grad] [--ablate-anc-grad]\n"
                     "  [--units N]\n"
                     "  [--save PATH]\n"
                     "  [--mode plan|instantiate-runtime|upload|forward|backward-head|one-step-sgd|one-step-rmsprop|multi-step-sgd|multi-step-rmsprop|save-reload-sgd|save-reload-rmsprop|train-epoch|train-small|train-growth]\n"
                     "  [--accumulate|--no-accumulate] [--quiet]\n"
                     "  compatibility aliases: [--instantiate-runtime] [--instantiate-chunk-upload]\n"
                         "                         [--run-forward-prefix] [--run-backward-head]\n");
        return 1;
    }
    if (cfg.partition_depth < 0) {
        std::fprintf(stderr,
                     "agpt_train_v2: --partition-depth must be non-negative\n");
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
    if (target_sidecar_path && !position_data_dir) {
        std::fprintf(stderr,
                     "agpt_train_v2: --target-sidecar requires --position-data DIR for radix->substring mapping\n");
        return 1;
    }
    if (target_sidecar_path && mode == V2Mode::TrainGrowth) {
        std::fprintf(stderr,
                     "agpt_train_v2: --target-sidecar is only supported for fixed-trie training\n");
        return 1;
    }
    if (cfg.chunk_queries <= 0) cfg.chunk_queries = 50000;
    if (steps <= 0) steps = 3;
    if (config_path && !validate_only && !save_path) {
        std::fprintf(stderr,
                     "WARN: model.save_file not set; trained model not persisted.\n");
    }
    DiagFireProbeV2 diag_probe = read_diag_fire_probe_v2();
    GradDumpProbeV2 grad_dump = read_grad_dump_probe_v2();

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
        if (rope_position_mode_uses_position_data_v2(cfg.rope_position_mode) || target_sidecar_path) {
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
        std::printf("  config: lr=%.6f warmup_epochs=%d lr_min_ratio=%.3f partition_depth=%d chunk_queries=%d anc_grad=%s\n",
                    cfg.lr, cfg.warmup_epochs, cfg.lr_min_ratio, cfg.partition_depth, cfg.chunk_queries,
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
    if (rope_position_mode_uses_position_data_v2(cfg.rope_position_mode) || target_sidecar_path) {
        pos_data = agpt_v2::load_position_sampling_data_v2(position_data_dir);
        have_pos_data = true;
        if (pos_data.prefix_table.window_size <= 0) {
            std::fprintf(stderr, "agpt_train_v2: position table has invalid window_size=%d\n",
                         pos_data.prefix_table.window_size);
            agpt_v2::free_radix_trie_structure(trie);
            return 1;
        }
        if (rope_position_mode_uses_position_data_v2(cfg.rope_position_mode)) {
            shape.rope_seq_len = required_rope_seq_len_v2(
                cfg.rope_position_mode, shape.seq_len, pos_data.prefix_table.window_size);
        }
        if (successor_table_ptr) {
            int successor_rope_len = successor_table.d_max * 2;
            if (successor_rope_len > shape.rope_seq_len) shape.rope_seq_len = successor_rope_len;
        }
    }
    agpt_v2::TargetSidecarTableV2 target_sidecar{};
    const agpt_v2::TargetSidecarTableV2* target_sidecar_ptr = nullptr;
    if (target_sidecar_path) {
        target_sidecar = agpt_v2::load_target_sidecar_table_v2(target_sidecar_path);
        if (target_sidecar.substring_count != pos_data.prefix_table.substring_count) {
            std::fprintf(stderr,
                         "agpt_train_v2: target sidecar substring_count=%u does not match position_data substrings=%d\n",
                         target_sidecar.substring_count, pos_data.prefix_table.substring_count);
            agpt_v2::free_successor_prefix_table_v2(successor_table);
            agpt_v2::free_radix_trie_structure(trie);
            return 1;
        }
        target_sidecar_ptr = &target_sidecar;
    }
    if (validate_only) {
        std::printf("agpt_train_v2: YAML config validated (mode=%s, trie_depth=%d, context_seq_len=%d, rope_seq_len=%d, successor_prefix=%s, target_sidecar=%s)\n",
                    v2_mode_name(mode), effective_seq_len_from_trie_v2(trie), shape.seq_len, shape.rope_seq_len,
                    successor_table_ptr ? "true" : "false",
                    target_sidecar_ptr ? "true" : "false");
        agpt_v2::free_successor_prefix_table_v2(successor_table);
        agpt_v2::free_radix_trie_structure(trie);
        return 0;
    }
    cfg.seq_len = shape.seq_len;
    cfg.rope_seq_len = shape.rope_seq_len;
    agpt_v2::ModelLayout model = agpt_v2::make_model_layout(shape);
    agpt_v2::CacheLayout cache = agpt_v2::make_cache_layout(shape);
    agpt_v2::LightningChildIndexV2 lightning_child_index{};
    if (cfg.lightning_enabled) {
        lightning_child_index = agpt_v2::build_lightning_child_index_v2(trie);
    }
    agpt_v2::TrainingPlan training_plan{};
    if (cfg.lightning_enabled) {
        training_plan.unit_count = 1;
        training_plan.units = (agpt_v2::TrainingUnit*)std::calloc(1, sizeof(agpt_v2::TrainingUnit));
        training_plan.units[0] = agpt_v2::build_lightning_sample_unit_v2(trie, lightning_child_index, cfg, 0);
    } else if (!yaml_cfg.partition_depth_map.empty()) {
        // experimental.partition_depth_map: text file, one "<token_id> <depth>" per line
        // ('#' comments allowed); roots not listed use train.partition_depth.
        std::vector<int> depth_by_token((size_t)shape.vocab_size, -1);
        FILE* mf = std::fopen(yaml_cfg.partition_depth_map.c_str(), "r");
        if (!mf) {
            std::fprintf(stderr, "agpt_train_v2: cannot open experimental.partition_depth_map %s\n",
                         yaml_cfg.partition_depth_map.c_str());
            return 1;
        }
        char line[256];
        int mapped = 0;
        while (std::fgets(line, sizeof(line), mf)) {
            if (line[0] == '#' || line[0] == '\n') continue;
            int tok = -1, d = -1;
            if (std::sscanf(line, "%d %d", &tok, &d) == 2 && tok >= 0 && tok < shape.vocab_size && d >= 1) {
                depth_by_token[(size_t)tok] = d;
                mapped++;
            }
        }
        std::fclose(mf);
        training_plan = agpt_v2::build_mixed_partition_plan_v2(trie, cfg.partition_depth, depth_by_token);
        std::printf("  partition_depth_map: %s (%d roots mapped, default pd=%d) -> %d training units\n",
                    yaml_cfg.partition_depth_map.c_str(), mapped, cfg.partition_depth, training_plan.unit_count);
    } else {
        training_plan = agpt_v2::build_training_plan_for_partition_depth(trie, cfg.partition_depth);
    }
    agpt_v2::ExecutionPlan plan = agpt_v2::build_execution_plan(trie, training_plan, cfg.chunk_queries);
    agpt_v2::ChunkPlanList largest_chunks = {};
    if (plan.largest_by_queries) {
        largest_chunks = agpt_v2::build_chunk_plan_for_unit(trie, *plan.largest_by_queries, cfg.chunk_queries, successor_table_ptr);
    }
    agpt_v2::ChunkPlanList capacity_chunks = cfg.lightning_enabled
        ? build_lightning_capacity_chunk_list_v2(cfg, shape, successor_table_ptr)
        : build_capacity_chunk_list_for_plan_v2(trie, training_plan, cfg.chunk_queries, successor_table_ptr);
    std::vector<agpt_v2::ChunkPlanList> unit_chunk_cache;
    if (!cfg.lightning_enabled) {
        unit_chunk_cache = build_unit_chunk_plan_cache_v2(trie, training_plan, cfg.chunk_queries, successor_table_ptr);
    }
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
                                                            0, 0, successor_table_ptr,
                                                            target_sidecar_ptr);
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
    std::printf("  config: epochs=%d lr=%.6f optimizer=%s schedule=%s warmup_epochs=%d lr_min_ratio=%.3f partition_depth=%d chunk_queries=%d accumulate=%s\n",
                cfg.epochs, cfg.lr, v2_optimizer_name(cfg.optimizer), v2_lr_schedule_name(cfg.lr_schedule), cfg.warmup_epochs, cfg.lr_min_ratio,
                cfg.partition_depth, cfg.chunk_queries, cfg.accumulate ? "true" : "false");
    if (cfg.dropout_node_keep_prob < 1.0f) {
        std::printf("  dropout: node_keep_prob=%.3f seed=%u\n",
                    cfg.dropout_node_keep_prob, cfg.dropout_seed);
    }
    if (cfg.entropy_gate_min_scale < 1.0f) {
        std::printf("  entropy-gate: min_scale=%.3f\n", cfg.entropy_gate_min_scale);
    }
    if (cfg.entropy_grad_min_scale < 1.0f) {
        std::printf("  entropy-grad: min_scale=%.3f\n", cfg.entropy_grad_min_scale);
    }
    if (cfg.lightning_enabled) {
        if (cfg.lightning_anchor_mode == agpt_v2::LightningAnchorModeV2::RandomDescendants) {
            std::printf("  lightning: enabled updates=%d anchor_mode=%s query_budget=ignored chunk_queries=%d seed=%u repeats_per_sample=%d\n",
                        cfg.lightning_updates,
                        lightning_anchor_mode_name_v2(cfg.lightning_anchor_mode),
                        cfg.chunk_queries,
                        cfg.lightning_seed,
                        cfg.lightning_repeats_per_sample);
        } else {
            std::printf("  lightning: enabled updates=%d anchor_mode=%s query_budget=%d seed=%u stop_p=%.3f sample_fanout=%d anchors_per_step=%d repeats_per_sample=%d\n",
                        cfg.lightning_updates,
                        lightning_anchor_mode_name_v2(cfg.lightning_anchor_mode),
                        cfg.lightning_query_budget > 0 ? cfg.lightning_query_budget : cfg.chunk_queries,
                        cfg.lightning_seed,
                        cfg.lightning_stop_p,
                        cfg.lightning_sample_fanout,
                        cfg.lightning_anchors_per_step,
                        cfg.lightning_repeats_per_sample);
        }
    }
    if (!yaml_cfg.checkpoint_epochs.empty()) {
        std::printf("  checkpoint_epochs:");
        for (int epoch : yaml_cfg.checkpoint_epochs) std::printf(" %d", epoch);
        std::printf("\n");
        if (!save_path) {
            std::printf("  checkpoint_epochs: ignored because model.save_file is not set\n");
        }
    }
    if (!cfg.anc_grad) {
        cfg.anc_grad_exact = false;  // nothing to make exact
    } else if (cfg.anc_grad_exact) {
        const char* unsupported = nullptr;
        if (mode != V2Mode::TrainEpoch) unsupported = "non-train-epoch mode";
        else if (cfg.rope_position_mode != agpt_v2::RopePositionModeV2::Depth) unsupported = "non-depth RoPE mode";
        else if (successor_table_ptr) unsupported = "successor prefix table";
        else if (target_sidecar_ptr) unsupported = "target sidecar";
        else if (cfg.lightning_enabled) unsupported = "lightning sampling";
        else if (cfg.dropout_node_keep_prob < 1.0f) unsupported = "node dropout";
        if (unsupported) {
            if (cfg.anc_grad_exact_explicit) {
                std::fprintf(stderr, "agpt_train_v2: experimental.anc_grad_exact: true is not supported with %s\n",
                             unsupported);
                return 1;
            }
            cfg.anc_grad_exact = false;
            std::printf("  anc-grad-exact: OFF (default-on, but not supported with %s; gradient through ancestor "
                        "K/V stops at Wk/Wv)\n", unsupported);
        } else {
            std::printf("  anc-grad-exact: enabled%s (second pass over internal nodes, deepest endpoint depth first; "
                        "ancestor K/V gradient carried through the full ancestor computation)\n",
                        cfg.anc_grad_exact_explicit ? "" : " (default)");
        }
    } else {
        std::printf("  anc-grad-exact: OFF (explicit experimental.anc_grad_exact: false; truncated ancestor gradient)\n");
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
        if (target_sidecar_ptr) {
            std::printf(" target_sidecar=%s sidecar_entries=%llu sidecar_scale=%u sidecar_mix=%.3f",
                        target_sidecar_path,
                        (unsigned long long)target_sidecar.total_entries,
                        target_sidecar.scale,
                        cfg.target_sidecar_mix);
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
    if (cfg.lightning_enabled) {
        std::printf("  lightning plan: streaming updates=%d; probe nodes=%lld queries=%lld compact_chars=%lld chunks=%lld at chunk_queries=%d\n",
                    cfg.lightning_updates,
                    plan.total_node_count, plan.total_query_count,
                    plan.total_compact_char_count, plan.estimated_chunk_count, cfg.chunk_queries);
    } else {
        std::printf("  pd=%d plan: %d training units, %lld node-visits, %lld query positions,\n"
                    "           %lld compact chars, ~%lld chunks/epoch at chunk_queries=%d\n",
                    cfg.partition_depth, plan.training_unit_count, plan.total_node_count,
                    plan.total_query_count, plan.total_compact_char_count,
                    plan.estimated_chunk_count, cfg.chunk_queries);
    }
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
                int units_to_run = cfg.lightning_enabled ? cfg.lightning_updates : plan.training_unit_count;
                if (unit_limit > 0 && unit_limit < units_to_run) units_to_run = unit_limit;
                if (units_to_run < 1) units_to_run = 1;
                int repeats_per_sample = cfg.lightning_enabled ? cfg.lightning_repeats_per_sample : 1;
                if (repeats_per_sample < 1) repeats_per_sample = 1;
                int optimizer_step_index = 0;
                AGPT_V2_CUDA_CHECK(cudaMemset(runtime.d_opt_m, 0, (size_t)model.total_floats * sizeof(float)));
                AGPT_V2_CUDA_CHECK(cudaMemset(runtime.d_opt_v, 0, (size_t)model.total_floats * sizeof(float)));
                std::printf("  train-epoch: epochs=%d units=%d repeats_per_sample=%d accumulate=%s optimizer=%s\n",
                            epochs, units_to_run, repeats_per_sample,
                            cfg.accumulate ? "true" : "false", v2_optimizer_name(cfg.optimizer));
                long long grad_dump_rows = 0;
                if (grad_dump.enabled) {
                    grad_dump_write_layout_v2(grad_dump, model, cfg, units_to_run);
                    grad_dump_open_v2(grad_dump, model.total_floats);
                    std::printf("  grad-dump: dir=%s apply_step=%s floats_per_row=%d\n",
                                grad_dump.dir, grad_dump.apply_step ? "true" : "false", model.total_floats);
                }
                long long total_unit_steps = (long long)epochs * (long long)units_to_run * (long long)repeats_per_sample;
                long long warmup_unit_steps = (long long)cfg.warmup_epochs * (long long)units_to_run * (long long)repeats_per_sample;
                if (total_unit_steps < 1) total_unit_steps = 1;
                const bool lbfgs_mode = (cfg.optimizer == agpt_v2::OptimizerKind::LBFGS);
                LbfgsStateV2 lbfgs{};
                if (lbfgs_mode) {
                    lbfgs_init_v2(lbfgs, model.total_floats, cfg.lbfgs_history);
                    std::printf("  lbfgs: history=%d c1=%g max_backtracks=%d; each epoch = one full-pass evaluation (units accumulated, no per-unit step)\n",
                                cfg.lbfgs_history, cfg.lbfgs_c1, cfg.lbfgs_max_backtracks);
                }
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
                    double anc_exact_epoch_seconds = 0.0;
                    long long anc_exact_epoch_queries = 0;
                    long long anc_exact_epoch_chunks = 0;
                    long long phase_target_nodes = 0;
                    long long phase_target_zero_nodes = 0;
                    long long phase_target_singleton_nodes = 0;
                    long long phase_target_prefix_mass = 0;
                    long long phase_target_global_mass = 0;
                    long long phase_target_local_mass = 0;
                    double phase_target_global_entropy_mass = 0.0;
                    double phase_target_local_entropy_mass = 0.0;
                    agpt_v2::zero_cache_runtime_v2(runtime.cache);
                    if (lbfgs_mode) {
                        AGPT_V2_CUDA_CHECK(cudaMemset(lbfgs.d_acc, 0, (size_t)model.total_floats * sizeof(float)));
                    }
                    std::printf("  train-epoch: epoch %d/%d\n", epoch + 1, epochs);
                    // experimental.unit_order_seed: seeded per-epoch shuffle of the unit
                    // order (rnd/gradient-population pool experiments: each pool member
                    // is a chain with its own random node order).
                    std::vector<int> unit_order((size_t)units_to_run);
                    for (int u = 0; u < units_to_run; u++) unit_order[(size_t)u] = u;
                    if (yaml_cfg.unit_order_seed >= 0) {
                        std::mt19937 rng((unsigned)yaml_cfg.unit_order_seed * 1000003u + (unsigned)epoch);
                        std::shuffle(unit_order.begin(), unit_order.end(), rng);
                        std::printf("  unit-order: seed=%d epoch=%d shuffled (first units: %d %d %d)\n",
                                    yaml_cfg.unit_order_seed, epoch + 1,
                                    unit_order[0], units_to_run > 1 ? unit_order[1] : -1, units_to_run > 2 ? unit_order[2] : -1);
                    }
                    for (int uo = 0; uo < units_to_run; uo++) {
                        int u = unit_order[(size_t)uo];
                        agpt_v2::TrainingUnit streamed_unit{};
                        agpt_v2::ChunkPlanList streamed_chunks{};
                        const agpt_v2::TrainingUnit* unit_ptr = nullptr;
                        const agpt_v2::ChunkPlanList* unit_chunks_ptr = nullptr;
                        if (cfg.lightning_enabled) {
                            streamed_unit = agpt_v2::build_lightning_sample_unit_v2(
                                trie, lightning_child_index, cfg,
                                epoch * units_to_run + u);
                            streamed_chunks = agpt_v2::build_chunk_plan_for_unit(
                                trie, streamed_unit, cfg.chunk_queries, successor_table_ptr);
                            unit_ptr = &streamed_unit;
                            unit_chunks_ptr = &streamed_chunks;
                        } else {
                            unit_ptr = &training_plan.units[u];
                            unit_chunks_ptr = &unit_chunk_cache[u];
                        }
                        const agpt_v2::TrainingUnit& unit = *unit_ptr;
                        if (cfg.rope_position_mode == agpt_v2::RopePositionModeV2::PhaseWeighted ||
                            cfg.rope_position_mode == agpt_v2::RopePositionModeV2::PhaseConditioned) {
                            int root_phase_mass = agpt_v2::prefix_position_mass_for_presentation_start_v2(
                                pos_stage_ptr, trie, unit.root_child_id, epoch_phase);
                            if (root_phase_mass <= 0) {
                                skipped_phase_zero_units++;
                                if (cfg.lightning_enabled) {
                                    agpt_v2::free_chunk_plan_list(streamed_chunks);
                                    agpt_v2::free_training_unit(streamed_unit);
                                }
                                continue;
                            }
                        }
                        const agpt_v2::ChunkPlanList& unit_chunks = *unit_chunks_ptr;
                        if (unit_chunks.chunk_count <= 0) {
                            std::printf("    unit %d/%d rc=%d chunks=0 skipped\n",
                                        u + 1, units_to_run, unit.root_child_id);
                            if (cfg.lightning_enabled) {
                                agpt_v2::free_chunk_plan_list(streamed_chunks);
                                agpt_v2::free_training_unit(streamed_unit);
                            }
                            continue;
                        }

                        for (int repeat = 0; repeat < repeats_per_sample; repeat++) {
                            float current_lr = scheduled_lr(cfg, optimizer_step_index, total_unit_steps, warmup_unit_steps);
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
                                                                     successor_table_ptr,
                                                                     target_sidecar_ptr);
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
                                    if (cfg.lightning_enabled) {
                                        agpt_v2::free_chunk_plan_list(streamed_chunks);
                                        agpt_v2::free_training_unit(streamed_unit);
                                    }
                                    agpt_v2::free_training_plan(training_plan);
                                    agpt_v2::free_radix_trie_structure(trie);
                                    return 0;
                                }
                                agpt_v2::BackwardPassResult chunk_bwd =
                                    agpt_v2::run_backward_output_head_v2(cfg, model, chunk_meta, chunk_device_meta, upload, chunk_fwd, runtime,
                                                                         cfg.anc_grad ? &unit_anc : nullptr,
                                                                         s == 0, (s + 1 == unit_chunks.chunk_count) && !cfg.anc_grad_exact);
                                (void)chunk_bwd;
                                unit_loss_sum += (double)chunk_fwd.mean_loss * chunk_fwd.trained_events;
                                unit_events += chunk_fwd.trained_events;
                                unit_trained += chunk_fwd.trained_queries;
                                epoch_loss_sum += (double)chunk_fwd.mean_loss * chunk_fwd.trained_events;
                                epoch_events += chunk_fwd.trained_events;
                                epoch_trained += chunk_fwd.trained_queries;
                                agpt_v2::free_chunk_metadata_v2(chunk_meta);
                            }

                            if (cfg.anc_grad_exact && cfg.anc_grad) {
                                AncExactStatsV2 ax = run_anc_exact_pass_v2(cfg, shape, model, trie, unit, pos_stage_ptr,
                                                                           epoch, optimizer_step_index, upload,
                                                                           epoch_loss_tables, runtime, unit_anc,
                                                                           runtime_contract);
                                anc_exact_epoch_seconds += ax.seconds;
                                anc_exact_epoch_queries += ax.queries;
                                anc_exact_epoch_chunks += ax.chunks;
                            }
                            if (unit_events <= 0.0 || unit_trained <= 0) {
                                abort_empty_training_unit_v2("train-epoch", epoch + 1, u, units_to_run,
                                                             unit.root_child_id, unit_chunks.chunk_count,
                                                             unit_trained, unit_events);
                            }
                            if (lbfgs_mode) {
                                lbfgs_axpy_v2(runtime.cublas, 1.0f, runtime.d_grads, lbfgs.d_acc, model.total_floats);
                            } else {
                                scale_gradients_for_fire(runtime.cublas, runtime.d_grads, model.total_floats, unit_events);
                            }
                            double unit_mean = unit_events > 0.0 ? (unit_loss_sum / unit_events) : 0.0;
                            if (grad_dump.enabled && !lbfgs_mode) {
                                grad_dump_row_v2(grad_dump, grad_dump_rows++, epoch + 1, unit, trie,
                                                 cfg.partition_depth, runtime.d_grads, model.total_floats,
                                                 unit_trained, unit_events, unit_mean);
                            }
                            agpt_v2::OptimizerStepResult step{};
                            if (lbfgs_mode) {
                                step.message = "accumulated (lbfgs)";
                            } else if (!grad_dump.enabled || grad_dump.apply_step) {
                                step = agpt_v2::run_optimizer_step_stateful(cfg, current_lr, runtime.d_weights, runtime.d_grads,
                                                                            runtime.d_opt_m, runtime.d_opt_v,
                                                                            model.total_floats, ++optimizer_step_index);
                            } else {
                                step.message = "grad dumped, step skipped (frozen weights)";
                            }
                            if (!(lbfgs_mode && cfg.quiet)) {
                                std::printf("    unit %d/%d repeat %d/%d rc=%d chunks=%d trained_queries=%lld trained_events=%.0f mean_loss=%.6f lr=%.6g step=%s\n",
                                            u + 1, units_to_run, repeat + 1, repeats_per_sample,
                                            unit.root_child_id, unit_chunks.chunk_count,
                                            unit_trained, unit_events, unit_mean, current_lr, step.message);
                            }
                            agpt_v2::free_unit_anc_grad_runtime_v2(unit_anc, runtime.contract);
                        }
                        if (cfg.lightning_enabled) {
                            agpt_v2::free_chunk_plan_list(streamed_chunks);
                            agpt_v2::free_training_unit(streamed_unit);
                        }
                    }
                    double epoch_mean = epoch_events > 0.0 ? (epoch_loss_sum / epoch_events) : 0.0;
                    std::printf("  train-epoch: epoch %d summary trained_queries=%lld trained_events=%.0f mean_loss=%.6f",
                                epoch + 1, epoch_trained, epoch_events, epoch_mean);
                    if (use_phase_mode) {
                        std::printf(" presentation_start_phase=%d skipped_zero_root_units=%d",
                                    epoch_phase, skipped_phase_zero_units);
                    }
                    if (cfg.rope_position_mode == agpt_v2::RopePositionModeV2::PhaseConditioned ||
                        target_sidecar_ptr) {
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
                        std::printf(" %s=nodes:%lld zero:%lld(%.1f%%) singleton:%lld(%.1f%%) target_mass:%lld/prefix:%lld(%.1f%%) allphase:%lld(%.1f%%) H:local=%.4f/global=%.4f",
                                    target_sidecar_ptr ? "target_sidecar" : "phase_targets",
                                    phase_target_nodes,
                                    phase_target_zero_nodes, zero_pct,
                                    phase_target_singleton_nodes, singleton_pct,
                                    phase_target_local_mass, phase_target_prefix_mass, prefix_retained_pct,
                                    phase_target_global_mass, retained_pct,
                                    local_h, global_h);
                    }
                    std::printf("\n");
                    if (cfg.anc_grad_exact) {
                        std::printf("  anc-exact: epoch %d second pass queries=%lld chunks=%lld seconds=%.2f\n",
                                    epoch + 1, anc_exact_epoch_queries, anc_exact_epoch_chunks, anc_exact_epoch_seconds);
                    }
                    if (lbfgs_mode) {
                        lbfgs_copy_v2(runtime.cublas, lbfgs.d_acc, runtime.d_grads, model.total_floats);
                        if (epoch_events > 0.0) {
                            scale_gradients_for_fire(runtime.cublas, runtime.d_grads, model.total_floats, epoch_events);
                        }
                        double gnorm = lbfgs_nrm2_v2(runtime.cublas, runtime.d_grads, model.total_floats);
                        double alpha_used = lbfgs.alpha;
                        const char* status = lbfgs_update_v2(runtime.cublas, cfg, lbfgs, runtime.d_weights, runtime.d_grads, epoch_mean);
                        std::printf("  lbfgs: pass=%d f=%.6f |g|=%.5g status=%s alpha_used=%.4g next_alpha=%.4g f_acc=%.6f iters=%d hist=%d/%d resets=%d wall=%.1fs\n",
                                    lbfgs.evaluations, epoch_mean, gnorm, status, alpha_used, lbfgs.alpha, lbfgs.f_acc,
                                    lbfgs.iterations, lbfgs.count, lbfgs.m, lbfgs.resets, wall_seconds_v2() - train_loop_start);
                    }
                    if (save_path && checkpoint_epoch_requested_v2(yaml_cfg.checkpoint_epochs, epoch + 1)) {
                        std::string checkpoint_path = epoch_checkpoint_path_v2(save_path, epoch + 1);
                        double checkpoint_train_wall = wall_seconds_v2() - train_loop_start;
                        std::printf("  train-epoch-checkpoint: epoch=%d train_wall_seconds=%.6f path=%s\n",
                                    epoch + 1, checkpoint_train_wall, checkpoint_path.c_str());
                        save_device_weights_checkpoint_v2("train-epoch", epoch + 1,
                                                          checkpoint_path, model,
                                                          lbfgs_mode ? lbfgs.d_theta : runtime.d_weights);
                    }
                }
                if (lbfgs_mode) {
                    // leave the accepted iterate (not the pending trial point) in d_weights for the final save
                    if (lbfgs.have_accepted) lbfgs_copy_v2(runtime.cublas, lbfgs.d_theta, runtime.d_weights, model.total_floats);
                    std::printf("  lbfgs: done evaluations=%d accepted_iterations=%d f_acc=%.6f resets=%d\n",
                                lbfgs.evaluations, lbfgs.iterations, lbfgs.f_acc, lbfgs.resets);
                    lbfgs_free_v2(lbfgs);
                }
                if (grad_dump.enabled) {
                    grad_dump_close_v2(grad_dump);
                    std::printf("  grad-dump: wrote %lld rows to %s/grads.f32\n", grad_dump_rows, grad_dump.dir);
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
                                                         pos_stage_ptr, 0, 0, successor_table_ptr,
                                                         target_sidecar_ptr);
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
