// AGPT CUDA Training Engine
// Standalone program: reads MGPT checkpoint + leveled trie index,
// runs BFS trie-walk training entirely on GPU, writes updated weights.
//
// Usage: agpt_train --model <path> --trie-dir <path> --epochs N --lr 0.0003
//
// Build: nvcc -O2 src/cuda/agpt_train.cu src/cuda/kernels.cu -lcublas -o bin/agpt_train

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <limits.h>
#include <unistd.h>
#include <sys/sysinfo.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// ============================================================================
// Error checking
// ============================================================================

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = (call); \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
        exit(1); \
    } \
} while(0)

// ============================================================================
// Existing kernels (extern declarations — linked from kernels.cu)
// ============================================================================

extern "C" {
    void cuda_softmax_rows(const float* input, float* output, int rows, int cols);
    void cuda_softmax_backward(const float* s, const float* ds, float* result, int rows, int cols);
    void cuda_layer_norm_forward(const float* input, float* output, float* norm_out,
                                  float* std_inv_out, const float* gamma, const float* beta,
                                  int rows, int cols);
    void cuda_layer_norm_backward(const float* grad, const float* norm, const float* std_inv,
                                   const float* gamma, float* dx, float* dgamma, float* dbeta,
                                   int rows, int cols);
    void cuda_bias_add(float* data, const float* bias, int rows, int cols);
    void cuda_fused_bias_relu(const float* input, const float* bias,
                              float* output, float* mask, int rows, int cols);
    void cuda_relu_backward(const float* grad, const float* mask, float* output, int n);
    void cuda_embedding_gather(const float* token_emb, const int* ids,
                                float* output, int seq_len, int d_model);
    void cuda_embedding_scatter_add(const float* grad, const int* ids,
                                     float* d_token_emb, int seq_len, int d_model);
    void cuda_adam_bulk(float* params, float* grads, float* m, float* v,
                         float lr, float beta1, float beta2, float eps,
                         int t, int n);
    void cuda_sgd_bulk(float* params, float* grads, float lr, int n);
    void cuda_momentum_bulk(float* params, float* grads, float* m, float lr, float beta, int n);
    void cuda_rmsprop_bulk(float* params, float* grads, float* s, float lr, float beta, float eps, int n);
    void cuda_weight_decay(float* params, float lr, float wd, int n);
    void cuda_grad_clip_by_norm(float* grads, float max_norm, int n,
                                 float* partials_scratch, float* norm_scratch);
    void cuda_batched_varlen_attention(
        const float* q_packed, const float* k_packed, const float* v_packed,
        const int* kv_offsets, const int* kv_lengths,
        float* output, float* weights_out,
        int n_nodes, int n_heads, int head_dim, int max_len, float scale);
    void cuda_batched_varlen_attention_backward(
        const float* q_packed, const float* k_packed, const float* v_packed,
        const float* attn_weights, const float* d_out,
        const int* kv_offsets, const int* kv_lengths,
        float* dq, float* dk_full, float* dv_full,
        int n_nodes, int n_heads, int head_dim, int max_len, float scale);
    void cuda_unpack_batched_attn_output(
        const float* packed_output, float* unpacked_output,
        int n_nodes, int n_heads, int head_dim);
    void cuda_sync();
}

// ============================================================================
// Model config
// ============================================================================

struct Config {
    int d_model;
    int n_heads;
    int n_layers;
    int d_ff;
    int vocab_size;
    int seq_len;
    float lr;
    int head_dim;       // derived: d_model / n_heads
    int chunk_queries;  // CLI --chunk-queries; 0 → default 50000
    bool ce_only = false;   // force single-target CE at endpoints (SGD-semantic; disables KL aggregation)
    float hotspot_coverage = 0.0f;  // adaptive split: between epochs, split subtrees covering top X% of excess-loss
                                     //   0.0 (default) disables splitting; 0.8 splits top subtrees covering 80% of excess
    int lr_rule = 0;  // per-subtree LR multiplier rule:
                      //   0=none (baseline), 1=inv-depth (1/depth),
                      //   2=inv-sqrt-depth (1/sqrt(depth)),
                      //   3=sqrt-batch (sqrt(tokens/mean_tokens)),
                      //   4=residual (prev-epoch score/mean_score)
    bool shuffle_order = false;  // randomize partition-group visit order each super-epoch
    unsigned shuffle_seed = 0xa17b1edu;  // RNG seed for shuffle-order (independent of lightning seed)
    int lbfgs_k = 10;  // L-BFGS history size (K most-recent (s,y) pairs); 0 disables
    int mini_batch_groups = 1;  // accumulate gradients across K consecutive partition groups before each
                                 // optimizer step (only effective with --no-accumulate). 1 = current
                                 // per-group behavior. Combined with --shuffle-order, K random groups
                                 // are batched per step — addresses pd=2 memorization where few unique
                                 // partitions get repeated solo many times.
    bool anc_grad = false;      // Descendant→ancestor gradient flow for Wk/Wv.
                                 // When set, the ANCESTOR slice of d_dk_pack/d_dv_pack
                                 // (currently dropped) gets scatter-added into per-subtree
                                 // accumulators, and the eventual dW_k/dW_v update at
                                 // subtree-fire end consumes those accumulators along
                                 // with the existing own-edge contribution. Requires
                                 // pd=1 (cross-group cache staleness at pd>1 would
                                 // confound the new gradient flow). See
                                 // todo/descendant-ancestor-scatter.md.
    // (anc_grad_scale removed — fire-end uses 1/subtree_events as the principled
    //  LM-update-unit denominator; no knob.)
    bool per_rc_adam = false;   // Stage 1 of topological optimizer state: per-root-child
                                 // Adam/RMSprop moments. Each rc subtree gets its own m, v, t
                                 // instead of sharing global state. Tests whether topological
                                 // localization of optimizer state matters at all. Per-rc state
                                 // is not persisted across runs (one-shot experimental flag).
                                 // Memory cost: n_root_children * total_floats * 8 bytes (two
                                 // float arrays). Requires --no-accumulate.
    const char* per_rc_v_dump_path = nullptr;  // when set, dump per-rc v buffer to this path at
                                                // end of training for offline diagnostic analysis
                                                // (per-bucket norms, cosine similarities, etc.)
};

// Curriculum modes: how subtrees are scheduled across an epoch.
// - Flat:        each epoch = one pass over all subtries at max trie depth.
// - Progressive: each epoch = d=1 pass, then d=2 pass, ..., then d=max pass.
//                At curriculum step d, only subtree nodes with endpoint_depth ≤ d
//                are trained (invariants doc: "bounded subtries up to depth d").
// - Random [TODO]: random interior subtree sampling; requires RoPE offset.
enum class CurriculumMode { Flat, Progressive };

// Optimizer choice. AGPT's aggregated gradients are low-variance, so Adam's
// per-parameter adaptation may be unnecessary — cheaper optimizers can match
// or beat Adam at fewer steps with tuned lr.
//  - Adam:     default, adaptive lr per param with momentum
//  - SGD:      plain w -= lr * g; tests whether AGPT needs any optimizer smarts
//  - Momentum: SGD + velocity; tests if just gradient smoothing is what helps
//  - RMSProp:  per-param variance without momentum; isolates Adam's two mechanisms
enum class OptimizerKind { Adam, SGD, Momentum, RMSProp, LBFGS };

// Mass-weight compression schemes. Each assigns a per-query weight
// w_i = compress(edge_mass_i) / mean_j(compress(edge_mass_j)), which
// scales the loss+gradient of each endpoint query. Off disables
// weighting (equal per radix endpoint).
//
//   Off    : w_i = 1  (AGPT default — equal per context, ignores count)
//   Log    : w_i = log(1 + count_i) / mean                 (compressed)
//   Sqrt   : w_i = sqrt(count_i)    / mean                 (partial)
//   Linear : w_i = count_i          / mean                 (matches SGD
//            frequency weighting — common patterns dominate training)
enum class MassWeightMode { Off, Log, Sqrt, Linear };

// Learning rate schedules.
//  - Constant:     lr stays at base_lr throughout training.
//  - Cosine:       lr decays as 0.5·base_lr·(1 + cos(π·progress)), progress ∈ [0,1] over total steps
//  - WarmupCosine: linear ramp from 0 to base_lr over first warmup_steps, then cosine decay
//                  over the remaining steps.
// Relevant for AGPT because the "converges fast then overfits" pattern benefits from
// aggressive early steps followed by near-zero late steps.
enum class LRSchedule { Constant, Cosine, WarmupCosine };

// Lightning Training — stochastic subtree sampling.
// Each super-epoch issues `steps` stochastic samples instead of the deterministic
// 65-root-child sweep. Each sampled subtree is a bounded training unit: its own
// d_grads zero, accumulated across chunks, one optimizer step. See
// notes/lightning-training.md for design rationale.
enum class LightningSampler { L1_Uniform, L2_RcDepth, L3_MassWalk, L4_Path };
struct LightningConfig {
    int               steps      = 0;          // 0 = disabled (deterministic sweep)
    LightningSampler  sampler    = LightningSampler::L3_MassWalk;
    float             p_stop     = 0.3f;       // L3 stopping probability at each level
    unsigned          seed       = 0x5c115e1u; // sampler RNG seed
    // Virtual-tree training: K>1 extends effective context past D* by
    // looping root-walks at mass>1 leaves, reusing the compact cache via
    // delta-RoPE at gather time. K=1 is plain AGPT (no virtual extension).
    int               virtual_cycles = 1;
    // Per-sample LR scaling by subtree mass (adaptive form of #4 from the
    // design discussion — LR scaling beats gradient scaling under RMSProp/Adam
    // because gradient scaling cancels in the adaptive divisor).
    //   Off    — no LR scaling
    //   Log    — w = log(1+mass) / mean(log(1+mass))   (gentlest, ~4× range)
    //   Sqrt   — w = sqrt(mass)  / mean(sqrt(mass))    (~100× range)
    //   Linear — w = mass        / mean(mass)          (can be 10000×+; unstable)
    // The linear mode reproduces the exact #4 proposal but empirically blows up
    // RMSProp when a single high-mass sample dominates. Log is the recommended
    // starting point.
    MassWeightMode    mass_lr    = MassWeightMode::Off;
    // Adaptive cap on per-step subtree size (proxied by mass — at d=32, where
    // most leaves are mass-1, mass(r) ≈ subtree_radix_size(r)). When the mass-
    // walk lands on a node with mass > max_mass, force-descend (override
    // p_stop) until the sampled subtree fits. 0 = no cap. Without this, on
    // skewed corpora (e.g. Gutenberg's biggest root-child has 2.47M nodes), a
    // single L3 step can train millions of nodes, blowing up wall-clock.
    long long         max_mass   = 0;
};

// L-BFGS optimizer state. Maintains rolling K-history of (s_i, y_i) pairs and
// a previous gradient. Two-loop recursion uses cuBLAS for axpy/dot/scal/copy.
// All buffers are device-side; rho values cached on host (small, K floats).
struct LBFGSState {
    int K = 0;            // history size
    int n = 0;            // total params
    float* d_g_prev = nullptr;    // previous gradient (n)
    float* d_step = nullptr;      // step taken last iteration = -lr * direction (n)
    float* d_s_hist = nullptr;    // K x n flattened (param differences)
    float* d_y_hist = nullptr;    // K x n flattened (gradient differences)
    float* rho_hist = nullptr;    // K floats (host)
    float* alpha = nullptr;       // K floats (host scratch for two-loop)
    int pushed_count = 0;  // number of accepted (s,y) pairs in history (== valid history size up to K)
    float* d_q = nullptr;         // n scratch for two-loop
    bool first_step = true;
};

// L-BFGS one-step update. Two-loop recursion using cuBLAS.
//
// State semantics:
//   d_g_prev: gradient from previous call (used to compute y_new).
//   d_step:   the step (-lr * direction) we took LAST CALL. Used as s_new.
//   d_s_hist[slot*n..]: param-difference history at history slot.
//   d_y_hist[slot*n..]: gradient-difference history at history slot.
//   rho_hist[slot]:     1 / (y^T s) for the entry at slot.
//   pushed_count:       number of (s,y) pairs ever pushed into history. The
//                       valid history size is min(pushed_count, K). Slot for
//                       the i-th pushed pair (0-indexed) is i % K.
//
// Per call:
//   1. If first_step: take SGD step θ -= lr * g, save d_step and d_g_prev,
//      mark first_step=false, return. (No (s,y) pair yet.)
//   2. Else:
//      a. y_new = d_grads - d_g_prev
//      b. ys = y_new^T d_step (curvature)
//      c. If ys is sufficiently positive AND finite, push (d_step, y_new, 1/ys)
//         into history at slot (pushed_count % K), increment pushed_count.
//         Otherwise SKIP the push (curvature condition violated).
//      d. If pushed_count > 0: two-loop recursion using all valid history.
//         Direction d = H * d_grads. Take θ -= lr * d.
//         If pushed_count == 0 (no valid pairs ever): SGD step instead.
//      e. Save d_step = -lr * (direction taken). Save d_g_prev = d_grads.
//
// Bug fixes from initial impl (2026-05-03 review):
//   - pushed_count separate from call count; skipped pairs don't lie about
//     history size and the two-loop never reads uninitialized slots.
//   - γ uses the most recent VALID pair's ρ even when this step skipped its
//     push, instead of falling back to identity.
static void cuda_lbfgs_step(LBFGSState* st, cublasHandle_t cublas,
                            float* d_weights, float* d_grads, float lr) {
    int n = st->n;
    int K = st->K;
    float neg_lr = -lr;

    if (st->first_step) {
        // First call: no history → plain SGD.
        CUBLAS_CHECK(cublasSaxpy(cublas, n, &neg_lr, d_grads, 1, d_weights, 1));
        // Save the step taken = -lr * g (overwrite d_step from zero state)
        CUDA_CHECK(cudaMemset(st->d_step, 0, n * sizeof(float)));
        CUBLAS_CHECK(cublasSaxpy(cublas, n, &neg_lr, d_grads, 1, st->d_step, 1));
        CUBLAS_CHECK(cublasScopy(cublas, n, d_grads, 1, st->d_g_prev, 1));
        st->first_step = false;
        return;
    }

    // ---- Compute y_new = d_grads - d_g_prev into d_q (scratch) ----
    CUBLAS_CHECK(cublasScopy(cublas, n, d_grads, 1, st->d_q, 1));
    float neg1 = -1.0f;
    CUBLAS_CHECK(cublasSaxpy(cublas, n, &neg1, st->d_g_prev, 1, st->d_q, 1));
    // d_q now holds y_new.

    // Curvature condition. Use a relative threshold (Wolfe-style):
    //   ys > eps_rel * sqrt(ys * yy)   ⟺   ys / sqrt(ys*yy) > eps_rel
    // Equivalently: ys > eps_rel * ||y|| * ||s||
    float ys = 0.0f, yy = 0.0f, ss = 0.0f;
    CUBLAS_CHECK(cublasSdot(cublas, n, st->d_q, 1, st->d_step, 1, &ys));
    CUBLAS_CHECK(cublasSdot(cublas, n, st->d_q, 1, st->d_q, 1, &yy));
    CUBLAS_CHECK(cublasSdot(cublas, n, st->d_step, 1, st->d_step, 1, &ss));

    const float CURVATURE_EPS = 1e-6f;  // relative tolerance for ys > eps * |y| * |s|
    float yy_ss = yy * ss;
    bool ys_finite = std::isfinite(ys) && std::isfinite(yy) && std::isfinite(ss);
    bool ys_valid  = ys_finite && (ys > CURVATURE_EPS * std::sqrt(yy_ss > 0.0f ? yy_ss : 1.0f));

    if (ys_valid) {
        // Push (d_step, y_new, 1/ys) into history at slot (pushed_count % K).
        int slot = st->pushed_count % K;
        CUBLAS_CHECK(cublasScopy(cublas, n, st->d_step, 1, st->d_s_hist + (size_t)slot * n, 1));
        CUBLAS_CHECK(cublasScopy(cublas, n, st->d_q, 1, st->d_y_hist + (size_t)slot * n, 1));
        st->rho_hist[slot] = 1.0f / ys;
        st->pushed_count++;
    }

    int hist_size = (st->pushed_count < K) ? st->pushed_count : K;

    if (hist_size == 0) {
        // No valid history yet (curvature failed every prior call). Fall back
        // to SGD this step, but keep tracking d_step / d_g_prev so subsequent
        // attempts to build curvature pairs can succeed.
        CUBLAS_CHECK(cublasSaxpy(cublas, n, &neg_lr, d_grads, 1, d_weights, 1));
        CUDA_CHECK(cudaMemset(st->d_step, 0, n * sizeof(float)));
        CUBLAS_CHECK(cublasSaxpy(cublas, n, &neg_lr, d_grads, 1, st->d_step, 1));
        CUBLAS_CHECK(cublasScopy(cublas, n, d_grads, 1, st->d_g_prev, 1));
        return;
    }

    // ---- Two-loop recursion ----
    // q = current gradient
    CUBLAS_CHECK(cublasScopy(cublas, n, d_grads, 1, st->d_q, 1));

    // Slot of i-th most recent pushed pair (0=newest, hist_size-1=oldest):
    //   slot = ((pushed_count - 1 - i) % K + K) % K
    // We store alpha at the SAME slot index used for the s/y read, so the
    // second loop retrieves a matching alpha for each slot.

    // First loop: i = newest to oldest
    for (int idx = 0; idx < hist_size; idx++) {
        int slot = ((st->pushed_count - 1 - idx) % K + K) % K;
        float si_q = 0.0f;
        CUBLAS_CHECK(cublasSdot(cublas, n,
                                st->d_s_hist + (size_t)slot * n, 1,
                                st->d_q, 1, &si_q));
        float a = st->rho_hist[slot] * si_q;
        st->alpha[slot] = a;
        float neg_a = -a;
        CUBLAS_CHECK(cublasSaxpy(cublas, n, &neg_a,
                                 st->d_y_hist + (size_t)slot * n, 1,
                                 st->d_q, 1));
    }

    // Initial Hessian scaling: γ = (s_last^T y_last) / (y_last^T y_last)
    // Use the MOST RECENT valid pair (slot of the newest push), even if THIS
    // call didn't push. This is the standard L-BFGS H_0 estimate.
    //
    // AGPT_LBFGS_GAMMA_ONE=1 env var forces γ=1.0 (identity H_0 init). This
    // is a diagnostic option for when the standard γ collapses to lr (which
    // happens when consecutive partition fires have orthogonal gradients —
    // see notes/todo/l-bfgs.md).
    {
        static const bool gamma_one = (getenv("AGPT_LBFGS_GAMMA_ONE") != nullptr);
        if (!gamma_one) {
            int slot_last = (st->pushed_count - 1) % K;
            float yy_last = 0.0f;
            CUBLAS_CHECK(cublasSdot(cublas, n,
                                    st->d_y_hist + (size_t)slot_last * n, 1,
                                    st->d_y_hist + (size_t)slot_last * n, 1, &yy_last));
            // gamma = (s^T y) / (y^T y) = (1 / ρ_slot_last) / (y^T y)
            float gamma = 1.0f;
            if (yy_last > 1e-12f && std::isfinite(yy_last) && st->rho_hist[slot_last] > 0.0f) {
                gamma = 1.0f / (st->rho_hist[slot_last] * yy_last);
            }
            if (gamma > 0.0f && std::isfinite(gamma)) {
                CUBLAS_CHECK(cublasSscal(cublas, n, &gamma, st->d_q, 1));
            }
        }
        // else: leave d_q as-is (γ=1 implicit).
    }

    // Second loop: i = oldest to newest
    for (int idx = hist_size - 1; idx >= 0; idx--) {
        int slot = ((st->pushed_count - 1 - idx) % K + K) % K;
        float yi_r = 0.0f;
        CUBLAS_CHECK(cublasSdot(cublas, n,
                                st->d_y_hist + (size_t)slot * n, 1,
                                st->d_q, 1, &yi_r));
        float beta = st->rho_hist[slot] * yi_r;
        float coef = st->alpha[slot] - beta;
        CUBLAS_CHECK(cublasSaxpy(cublas, n, &coef,
                                 st->d_s_hist + (size_t)slot * n, 1,
                                 st->d_q, 1));
    }

    // d_q now holds H * g. Take step θ -= lr * d_q.
    CUBLAS_CHECK(cublasSaxpy(cublas, n, &neg_lr, st->d_q, 1, d_weights, 1));

    // Save the step taken = -lr * d_q, and current grad as g_prev.
    CUDA_CHECK(cudaMemset(st->d_step, 0, n * sizeof(float)));
    CUBLAS_CHECK(cublasSaxpy(cublas, n, &neg_lr, st->d_q, 1, st->d_step, 1));
    CUBLAS_CHECK(cublasScopy(cublas, n, d_grads, 1, st->d_g_prev, 1));
}

// 32-bit xorshift. Same output across platforms; reproducible from a seed.
static inline unsigned xorshift32(unsigned* state) {
    unsigned x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x ? x : 0x1u;
    return *state;
}

static inline float xorshift_float01(unsigned* state) {
    // uniform in [0, 1)
    return (float)xorshift32(state) / (float)4294967296.0;
}

static float compute_lr(float base_lr, int step, int total_steps,
                         int warmup_steps, LRSchedule sched) {
    if (total_steps <= 1) return base_lr;
    if (sched == LRSchedule::Constant) return base_lr;
    if (sched == LRSchedule::WarmupCosine && step < warmup_steps) {
        return base_lr * ((float)(step + 1) / (float)warmup_steps);
    }
    // cosine tail — progress from end-of-warmup (or 0 for pure cosine) to total
    int cos_start = (sched == LRSchedule::WarmupCosine) ? warmup_steps : 0;
    int cos_end   = total_steps;
    if (cos_end <= cos_start) return base_lr;
    float progress = (float)(step - cos_start) / (float)(cos_end - cos_start);
    if (progress > 1.0f) progress = 1.0f;
    return 0.5f * base_lr * (1.0f + cosf(3.14159265358979323846f * progress));
}

// ============================================================================
// Weight layout: flat buffer with computed offsets
// ============================================================================
// Order matches Crystal's weight_mats:
//   token_emb
//   per block: wq.w, wq.b, wk.w, wk.b, wv.w, wv.b, wo.w, wo.b,
//              ln1.gamma, ln1.beta, ff.l1.w, ff.l1.b, ff.l2.w, ff.l2.b,
//              ln2.gamma, ln2.beta
//   final_norm.gamma, final_norm.beta
//   output.w, output.b

struct WeightOffsets {
    int token_emb;        // [vocab_size, d_model]

    // Per-layer offsets (arrays of size n_layers)
    int* wq_w;    // [d_model, d_model]
    int* wq_b;    // [1, d_model]
    int* wk_w;
    int* wk_b;
    int* wv_w;
    int* wv_b;
    int* wo_w;
    int* wo_b;
    int* ln1_gamma;  // [1, d_model]
    int* ln1_beta;
    int* l1_w;    // [d_model, d_ff]
    int* l1_b;    // [1, d_ff]
    int* l2_w;    // [d_ff, d_model]
    int* l2_b;    // [1, d_model]
    int* ln2_gamma;
    int* ln2_beta;

    int final_gamma;  // [1, d_model]
    int final_beta;
    int out_w;        // [d_model, vocab_size]
    int out_b;        // [1, vocab_size]

    int total_floats;
};

WeightOffsets compute_offsets(const Config& cfg) {
    WeightOffsets wo;
    int L = cfg.n_layers;
    int D = cfg.d_model;
    int F = cfg.d_ff;
    int V = cfg.vocab_size;

    wo.wq_w = (int*)malloc(L * sizeof(int));
    wo.wq_b = (int*)malloc(L * sizeof(int));
    wo.wk_w = (int*)malloc(L * sizeof(int));
    wo.wk_b = (int*)malloc(L * sizeof(int));
    wo.wv_w = (int*)malloc(L * sizeof(int));
    wo.wv_b = (int*)malloc(L * sizeof(int));
    wo.wo_w = (int*)malloc(L * sizeof(int));
    wo.wo_b = (int*)malloc(L * sizeof(int));
    wo.ln1_gamma = (int*)malloc(L * sizeof(int));
    wo.ln1_beta  = (int*)malloc(L * sizeof(int));
    wo.l1_w = (int*)malloc(L * sizeof(int));
    wo.l1_b = (int*)malloc(L * sizeof(int));
    wo.l2_w = (int*)malloc(L * sizeof(int));
    wo.l2_b = (int*)malloc(L * sizeof(int));
    wo.ln2_gamma = (int*)malloc(L * sizeof(int));
    wo.ln2_beta  = (int*)malloc(L * sizeof(int));

    int off = 0;
    wo.token_emb = off; off += V * D;

    for (int i = 0; i < L; i++) {
        wo.wq_w[i] = off; off += D * D;
        wo.wq_b[i] = off; off += D;
        wo.wk_w[i] = off; off += D * D;
        wo.wk_b[i] = off; off += D;
        wo.wv_w[i] = off; off += D * D;
        wo.wv_b[i] = off; off += D;
        wo.wo_w[i] = off; off += D * D;
        wo.wo_b[i] = off; off += D;
        wo.ln1_gamma[i] = off; off += D;
        wo.ln1_beta[i]  = off; off += D;
        wo.l1_w[i] = off; off += D * F;
        wo.l1_b[i] = off; off += F;
        wo.l2_w[i] = off; off += F * D;
        wo.l2_b[i] = off; off += D;
        wo.ln2_gamma[i] = off; off += D;
        wo.ln2_beta[i]  = off; off += D;
    }

    wo.final_gamma = off; off += D;
    wo.final_beta  = off; off += D;
    wo.out_w = off; off += D * V;
    wo.out_b = off; off += V;

    wo.total_floats = off;
    return wo;
}

// ============================================================================
// Trie structure (CPU side, then uploaded)
// ============================================================================

struct TrieData {
    int total_nodes;
    int max_depth;
    int depth_file_count;

    // Flat arrays indexed by node_id (sorted by depth within each level)
    int* tokens;       // token_id per node
    int* parents;      // parent_id per node
    int* depths;       // depth per node

    // Per-depth: how many nodes, starting index in the flat arrays
    int* depth_start;  // [depth_file_count + 1] — exclusive end at [d+1]
    int* depth_count;  // [depth_file_count]

    // Next-token counts (targets for loss)
    int* counts_offset; // [total_nodes + 1] — offset into counts_tok/counts_val
    int* counts_tok;    // flat token ids
    int* counts_val;    // flat count values
    int total_counts;   // total entries in counts_tok/counts_val

    // Ancestor chain for each node (for building varlen attention kv_offsets)
    // ancestor_offset[i]..ancestor_offset[i+1] are the ancestor node_ids for node i
    int* ancestor_offset; // [total_nodes + 1]
    int* ancestor_ids;    // flat ancestor node ids (in order from root to parent)
    int total_ancestor_entries;
};

// --------------------------------------------------------------------
// Radix trie data (when input dir contains radix_depth_NNN.bin files)
// --------------------------------------------------------------------
struct RadixTrieData {
    int radix_count;
    int depth_file_count;        // endpoint depth file count
    long long total_edge_chars;  // total character positions

    // Per-radix-node arrays
    int* parents;                // [radix_count]
    int* edge_starts;            // [radix_count] — offset into edge_tokens_flat
    int* edge_lens;              // [radix_count]
    int* edge_first_char_depths; // [radix_count]
    int* edge_mass;              // [radix_count] — prefix mass at head of edge (v2+)
    int* edge_tokens_flat;       // [total_edge_chars]

    // d_split[r] = depth at which the path root→r first becomes mass=1.
    // Populated post-load by walking each node's parent chain. INT_MAX for
    // multi-mass leaves (whose paths never reach mass=1 within the trie).
    // Used for per-leaf depth-routed K/V gradient (the "real radix point"
    // boundary, vs a flat global threshold).
    int* d_split;                // [radix_count]

    // mean_edge_mass[d] for d in [0, depth_file_count+1). Average edge_mass
    // across all radix nodes whose edge spans depth d. Used by joint-mass
    // weighting (AGPT_JOINT_MASS=1) — at a query at depth d_q, the
    // "complementary" suffix-side mass is approximated by mean_edge_mass at
    // the complementary depth (depth_file_count - d_q).
    double* mean_edge_mass;      // [depth_file_count + 1]

    int* endpoint_depth_start;   // [depth_file_count + 1]
    int* endpoint_depth_count;   // [depth_file_count]

    // Endpoint counts (training targets)
    int* counts_offset;          // [radix_count + 1]
    int* counts_tok;
    int* counts_val;
    int total_counts;

    // Ancestor character-position chains per radix node.
    // For radix node r, ancestor_char_offsets[r]..ancestor_char_offsets[r+1]
    // are the CHARACTER POSITIONS (into the global KV cache / edge_tokens_flat)
    // that make up the ancestry of r's edge (parent edges concatenated, root to leaf).
    int* ancestor_char_offsets;  // [radix_count + 1]
    int* ancestor_char_ids;      // flat
    long long total_ancestor_chars;
};

#define LEVELED_MAGIC 0x4C475041u
#define RADIX_MAGIC   0x52445841u

// Read one little-endian int32 from file
static int read_i32(FILE* f) {
    int v;
    fread(&v, 4, 1, f);
    return v;
}

static int read_i16(FILE* f) {
    int16_t v;
    fread(&v, 2, 1, f);
    return (int)v;  // sign-extend to int for storage compatibility
}

static unsigned read_u32(FILE* f) {
    unsigned v;
    fread(&v, 4, 1, f);
    return v;
}

static void read_u64(FILE* f) {
    unsigned long long v;
    fread(&v, 8, 1, f);
}

// Detect trie format. Return codes:
//   0 = leveled trie (single trie, multiple depth files)
//   1 = radix trie, global layout (single trie, per-endpoint-depth files)
//   2 = radix trie, per-subtree layout (one file per root-child; manifest.bin exists)
int detect_trie_format(const char* dir) {
    char path[1024];
    // Per-subtree is indicated by manifest.bin existing alongside meta.bin.
    snprintf(path, sizeof(path), "%s/manifest.bin", dir);
    FILE* mf = fopen(path, "rb");
    if (mf) {
        fclose(mf);
        return 2;
    }
    snprintf(path, sizeof(path), "%s/meta.bin", dir);
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    unsigned magic = read_u32(f);
    fclose(f);
    if (magic == RADIX_MAGIC)   return 1;
    if (magic == LEVELED_MAGIC) return 0;
    fprintf(stderr, "Unknown trie magic 0x%08x\n", magic);
    exit(1);
}

// Per-subtree support: manifest entry and loader for a single subtree file.
// Each subtree is self-contained — its ancestor chain lives entirely within its
// own records (because radix descendants of a root-child never escape to other
// root-children's subtrees). This means a subtree file can be loaded and
// trained independently, with KV cache scoped to this subtree's character
// positions only. Big win for memory at d≥16.

struct SubtreeManifestEntry {
    int root_child_id;
    int n_nodes;
    long long total_edge_chars;
    int max_endpoint_depth;
};

struct SubtreeManifest {
    int n_subtrees;
    SubtreeManifestEntry* entries;  // [n_subtrees]
    char dir[1024];                  // parent directory containing "subtrees/"
};

SubtreeManifest load_subtree_manifest(const char* dir) {
    SubtreeManifest m;
    memset(&m, 0, sizeof(m));
    strncpy(m.dir, dir, sizeof(m.dir) - 1);
    char path[1024];
    snprintf(path, sizeof(path), "%s/manifest.bin", dir);
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    unsigned magic = read_u32(f);
    if (magic != RADIX_MAGIC) { fprintf(stderr, "Bad manifest magic\n"); exit(1); }
    int version = read_i32(f);
    if (version != 2 && version != 3) { fprintf(stderr, "Unsupported manifest version %d (need 2 or 3)\n", version); exit(1); }
    m.n_subtrees = read_i32(f);
    m.entries = (SubtreeManifestEntry*)calloc(m.n_subtrees, sizeof(SubtreeManifestEntry));
    for (int i = 0; i < m.n_subtrees; i++) {
        m.entries[i].root_child_id = read_i32(f);
        m.entries[i].n_nodes = read_i32(f);
        fread(&m.entries[i].total_edge_chars, 8, 1, f);
        m.entries[i].max_endpoint_depth = read_i32(f);
    }
    fclose(f);
    return m;
}

// A single-subtree SoA. Indexing is LOCAL to this subtree — there's no global
// radix_id or global char position. Fields have the same semantics as in
// RadixTrieData but only for this subtree's members.
struct SubtreeData {
    int root_child_id;
    int n_nodes;
    int total_edge_chars;
    int max_endpoint_depth;

    // Per-local-radix-node arrays, indexed by local_id (0 = root-child of this subtree)
    int* parents;              // [n_nodes] — local parent id, or -1 if this node IS the root-child
    int* edge_starts;          // [n_nodes] — local char position of edge's first char
    int* edge_lens;             // [n_nodes]
    int* edge_first_char_depths; // [n_nodes]
    int* edge_mass;            // [n_nodes]
    int* edge_tokens_flat;     // [total_edge_chars]

    int* counts_offset;        // [n_nodes + 1]
    int* counts_tok;
    int* counts_val;
    int total_counts;

    // Ancestor character-position chains (local).
    int* ancestor_char_offsets; // [n_nodes + 1]
    int* ancestor_char_ids;     // flat local char positions
    long long total_ancestor_chars;
};

SubtreeData load_subtree(const SubtreeManifest& m, int manifest_index) {
    SubtreeData s;
    memset(&s, 0, sizeof(s));
    const SubtreeManifestEntry& e = m.entries[manifest_index];
    s.root_child_id = e.root_child_id;
    s.max_endpoint_depth = e.max_endpoint_depth;

    char path[1024];
    snprintf(path, sizeof(path), "%s/subtrees/radix_subtree_%06d.bin", m.dir, e.root_child_id);
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    unsigned magic = read_u32(f);
    if (magic != RADIX_MAGIC) { fprintf(stderr, "Bad subtree magic in %s\n", path); exit(1); }
    int version = read_i32(f);
    if (version != 2 && version != 3) { fprintf(stderr, "Bad subtree version %d in %s (need v2 or v3)\n", version, path); exit(1); }
    const bool narrow_tokens = (version >= 3);
    const int token_bytes = narrow_tokens ? 2 : 4;
    const int counts_entry_bytes = narrow_tokens ? 6 : 8;
    int stored_rc = read_i32(f);
    if (stored_rc != e.root_child_id) { fprintf(stderr, "Subtree rc mismatch\n"); exit(1); }
    s.n_nodes = read_i32(f);
    fread(&s.total_edge_chars, 8, 1, f); // oops: declared int, see below
    // The file format stores i64 but our struct uses int. For Shakespeare/Gutenberg
    // subtrees this is always < 2^31 so it fits. Truncate on read:
    int max_ep = read_i32(f);
    (void)max_ep;

    s.parents                = (int*)calloc(s.n_nodes, sizeof(int));
    s.edge_starts            = (int*)calloc(s.n_nodes, sizeof(int));
    s.edge_lens              = (int*)calloc(s.n_nodes, sizeof(int));
    s.edge_first_char_depths = (int*)calloc(s.n_nodes, sizeof(int));
    s.edge_mass              = (int*)calloc(s.n_nodes, sizeof(int));
    s.edge_tokens_flat       = (int*)calloc(s.total_edge_chars, sizeof(int));
    s.counts_offset          = (int*)calloc(s.n_nodes + 1, sizeof(int));

    // First pass: read structure. Records in the file use GLOBAL radix_ids.
    // We need to remap to local ids in [0, n_nodes). Records are BFS-sorted, so
    // local_id = order of appearance works. The root-child (= radix_id matches
    // root_child_id) gets local_id 0; others get incremental ids as they appear.
    int* global_to_local = (int*)malloc(s.n_nodes * sizeof(int)); // will remap later
    int* global_ids = (int*)malloc(s.n_nodes * sizeof(int));
    int* entry_counts_per_local = (int*)calloc(s.n_nodes, sizeof(int));
    long long edge_fill_pos = 0;
    long long total_counts_local = 0;

    for (int i = 0; i < s.n_nodes; i++) {
        int global_rid = read_i32(f);
        int global_parent = read_i32(f);
        int fcd = read_i32(f);
        int elen = read_i32(f);
        global_ids[i] = global_rid;
        // Stash global parent in parents[] temporarily; remap after pass 1
        s.parents[i] = global_parent;
        s.edge_starts[i] = (int)edge_fill_pos;
        s.edge_lens[i] = elen;
        s.edge_first_char_depths[i] = fcd;
        for (int e2 = 0; e2 < elen; e2++) {
            s.edge_tokens_flat[edge_fill_pos + e2] = narrow_tokens ? read_i16(f) : read_i32(f);
        }
        edge_fill_pos += elen;
        s.edge_mass[i] = read_i32(f);
        int ec = read_i32(f);
        entry_counts_per_local[i] = ec;
        total_counts_local += ec;
        fseek(f, ec * counts_entry_bytes, SEEK_CUR);
    }
    s.total_counts = (int)total_counts_local;

    // Build global_rid → local_id map
    // (Linear search is fine for small n_nodes; for huge subtrees, consider hashing)
    // To avoid O(n^2) we use a hash table (std::unordered_map-ish via sort).
    // Simpler: sort by global_rid into an auxiliary array and binary-search.
    // n_nodes up to ~250k per subtree; binary search is fast enough.
    {
        // Build an indirect sort: idx array sorted by global_ids
        int* sort_idx = (int*)malloc(s.n_nodes * sizeof(int));
        for (int i = 0; i < s.n_nodes; i++) sort_idx[i] = i;
        // qsort with comparator referencing global_ids via a global pointer.
        static int* g_global_ids_ptr = NULL;
        g_global_ids_ptr = global_ids;
        auto cmp = +[](const void* a, const void* b) -> int {
            int ia = *(const int*)a; int ib = *(const int*)b;
            int ga = g_global_ids_ptr[ia]; int gb = g_global_ids_ptr[ib];
            return (ga > gb) - (ga < gb);
        };
        qsort(sort_idx, s.n_nodes, sizeof(int), cmp);
        // Helper binary search: given global_id, return local_id or -1
        auto find_local = [&](int gid) -> int {
            int lo = 0, hi = s.n_nodes - 1;
            while (lo <= hi) {
                int mid = (lo + hi) / 2;
                int g = global_ids[sort_idx[mid]];
                if (g == gid) return sort_idx[mid];
                if (g < gid) lo = mid + 1; else hi = mid - 1;
            }
            return -1;
        };

        // Remap parents: s.parents[i] currently holds the GLOBAL parent_radix_id.
        // Convert to LOCAL id. If the parent is the virtual root (0), it means
        // this node IS the root-child of the subtree → local parent = -1.
        for (int i = 0; i < s.n_nodes; i++) {
            int gp = s.parents[i];
            if (gp == 0) {
                s.parents[i] = -1;
            } else {
                int lp = find_local(gp);
                // If parent is outside this subtree (shouldn't happen by construction), -1.
                s.parents[i] = lp;
            }
        }
        free(sort_idx);
    }
    free(global_ids);

    // Prefix sum for counts_offset
    s.counts_offset[0] = 0;
    for (int i = 0; i < s.n_nodes; i++) {
        s.counts_offset[i + 1] = s.counts_offset[i] + entry_counts_per_local[i];
    }
    free(entry_counts_per_local);

    s.counts_tok = (int*)malloc(s.total_counts * sizeof(int));
    s.counts_val = (int*)malloc(s.total_counts * sizeof(int));

    // Second pass: read counts
    fseek(f, 0, SEEK_SET);
    read_u32(f); read_i32(f); read_i32(f); read_i32(f); fread(&edge_fill_pos, 8, 1, f); read_i32(f);
    for (int i = 0; i < s.n_nodes; i++) {
        read_i32(f); read_i32(f); read_i32(f);                // rid, parent, fcd
        int elen = s.edge_lens[i];
        fseek(f, 4, SEEK_CUR);                                  // skip edge_len
        fseek(f, elen * token_bytes, SEEK_CUR);                 // skip edge tokens (v2: 4B, v3: 2B)
        fseek(f, 4, SEEK_CUR);                                  // skip edge_mass
        int ec = read_i32(f);
        int out_off = s.counts_offset[i];
        for (int ee = 0; ee < ec; ee++) {
            s.counts_tok[out_off + ee] = narrow_tokens ? read_i16(f) : read_i32(f);
            s.counts_val[out_off + ee] = read_i32(f);
        }
    }
    fclose(f);

    // Build ancestor_char_ids: for each local node i, concatenate parent's
    // ancestors + parent's edge chars. Since records are BFS-sorted by endpoint
    // depth, parent always appears before child, so forward scan is valid.
    s.ancestor_char_offsets = (int*)malloc((s.n_nodes + 1) * sizeof(int));
    long long* anc_lens = (long long*)calloc(s.n_nodes, sizeof(long long));
    long long total_anc_chars = 0;
    for (int i = 0; i < s.n_nodes; i++) {
        int p = s.parents[i];
        anc_lens[i] = (p < 0) ? 0 : anc_lens[p] + s.edge_lens[p];
        total_anc_chars += anc_lens[i];
    }
    s.total_ancestor_chars = total_anc_chars;
    s.ancestor_char_ids = (int*)malloc(total_anc_chars * sizeof(int));
    s.ancestor_char_offsets[0] = 0;
    for (int i = 0; i < s.n_nodes; i++) {
        s.ancestor_char_offsets[i + 1] = s.ancestor_char_offsets[i] + (int)anc_lens[i];
    }
    for (int i = 0; i < s.n_nodes; i++) {
        int p = s.parents[i];
        if (p < 0) continue;
        int out = s.ancestor_char_offsets[i];
        int parent_anc_off = s.ancestor_char_offsets[p];
        int parent_anc_len = (int)anc_lens[p];
        memcpy(&s.ancestor_char_ids[out], &s.ancestor_char_ids[parent_anc_off],
               parent_anc_len * sizeof(int));
        int parent_edge_start = s.edge_starts[p];
        int parent_edge_len = s.edge_lens[p];
        for (int ee = 0; ee < parent_edge_len; ee++) {
            s.ancestor_char_ids[out + parent_anc_len + ee] = parent_edge_start + ee;
        }
    }
    free(anc_lens);

    return s;
}

void free_subtree(SubtreeData& s) {
    free(s.parents); free(s.edge_starts); free(s.edge_lens);
    free(s.edge_first_char_depths); free(s.edge_mass); free(s.edge_tokens_flat);
    free(s.counts_offset); free(s.counts_tok); free(s.counts_val);
    free(s.ancestor_char_offsets); free(s.ancestor_char_ids);
    memset(&s, 0, sizeof(s));
}

// Adapter: wrap a SubtreeData in a RadixTrieData view so run_radix_training can
// consume it unchanged. The mismatch is that the global radix format reserves
// radix_id=0 as the virtual root (run_radix_training's root-child detection
// scans r>=1 and checks parents[r]==0), while SubtreeData has local_id 0 as the
// real root-child with parent=-1.
//
// Fix: synthesize a virtual-root entry at index 0 and shift every subtree node
// up by one. Allocates fresh arrays for the shifted index buffers
// (parents/edge_starts/counts_offset/ancestor_char_offsets). Borrows the data
// arrays where contents don't need remapping (edge_tokens_flat, counts_tok,
// counts_val, ancestor_char_ids — these hold character positions and token ids,
// which are subtree-local and don't need shifting).
//
// free_radix_view frees only the arrays we allocated here.
struct RadixView {
    RadixTrieData t;
    int* owned_parents;
    int* owned_edge_starts;
    int* owned_edge_lens;
    int* owned_edge_first_char_depths;
    int* owned_edge_mass;
    int* owned_counts_offset;
    int* owned_ancestor_char_offsets;
};

RadixView subtree_to_radix_view(const SubtreeData& s) {
    RadixView v;
    memset(&v, 0, sizeof(v));
    int N = s.n_nodes + 1;  // +1 for virtual root at index 0

    v.owned_parents              = (int*)calloc(N, sizeof(int));
    v.owned_edge_starts          = (int*)calloc(N, sizeof(int));
    v.owned_edge_lens            = (int*)calloc(N, sizeof(int));
    v.owned_edge_first_char_depths = (int*)calloc(N, sizeof(int));
    v.owned_edge_mass            = (int*)calloc(N, sizeof(int));
    v.owned_counts_offset        = (int*)calloc(N + 1, sizeof(int));
    v.owned_ancestor_char_offsets = (int*)calloc(N + 1, sizeof(int));

    // Virtual root at index 0.
    v.owned_parents[0] = 0;
    v.owned_edge_starts[0] = 0;
    v.owned_edge_lens[0] = 0;
    v.owned_edge_first_char_depths[0] = 0;
    v.owned_edge_mass[0] = 0;
    v.owned_counts_offset[0] = 0;
    v.owned_counts_offset[1] = 0;  // virtual root has no counts
    v.owned_ancestor_char_offsets[0] = 0;
    v.owned_ancestor_char_offsets[1] = 0;  // virtual root has no ancestors

    for (int i = 0; i < s.n_nodes; i++) {
        int g = i + 1;  // global index in the view
        // Parent: -1 in subtree means "IS the root-child" → point at virtual-root 0.
        // Any other local id p maps to p+1 in the view.
        v.owned_parents[g] = (s.parents[i] < 0) ? 0 : (s.parents[i] + 1);
        v.owned_edge_starts[g]           = s.edge_starts[i];
        v.owned_edge_lens[g]             = s.edge_lens[i];
        v.owned_edge_first_char_depths[g] = s.edge_first_char_depths[i];
        v.owned_edge_mass[g]             = s.edge_mass[i];
        v.owned_counts_offset[g + 1]        = s.counts_offset[i + 1];
        v.owned_ancestor_char_offsets[g + 1] = s.ancestor_char_offsets[i + 1];
    }

    v.t.radix_count           = N;
    v.t.depth_file_count      = s.max_endpoint_depth + 1;
    v.t.total_edge_chars      = s.total_edge_chars;
    v.t.parents               = v.owned_parents;
    v.t.edge_starts           = v.owned_edge_starts;
    v.t.edge_lens             = v.owned_edge_lens;
    v.t.edge_first_char_depths = v.owned_edge_first_char_depths;
    v.t.edge_mass             = v.owned_edge_mass;
    v.t.edge_tokens_flat      = s.edge_tokens_flat;      // borrowed
    v.t.endpoint_depth_start  = NULL;                    // unused by run_radix_training
    v.t.endpoint_depth_count  = NULL;
    v.t.counts_offset         = v.owned_counts_offset;
    v.t.counts_tok            = s.counts_tok;            // borrowed
    v.t.counts_val            = s.counts_val;            // borrowed
    v.t.total_counts          = s.total_counts;
    v.t.ancestor_char_offsets = v.owned_ancestor_char_offsets;
    v.t.ancestor_char_ids     = s.ancestor_char_ids;     // borrowed
    v.t.total_ancestor_chars  = s.total_ancestor_chars;
    return v;
}

void free_radix_view(RadixView& v) {
    free(v.owned_parents); free(v.owned_edge_starts); free(v.owned_edge_lens);
    free(v.owned_edge_first_char_depths); free(v.owned_edge_mass);
    free(v.owned_counts_offset); free(v.owned_ancestor_char_offsets);
    memset(&v, 0, sizeof(v));
}

// --------------------------------------------------------------------
// Radix trie loader
// --------------------------------------------------------------------
RadixTrieData load_radix_trie(const char* dir) {
    RadixTrieData t;
    memset(&t, 0, sizeof(t));

    char path[1024];
    // meta.bin
    snprintf(path, sizeof(path), "%s/meta.bin", dir);
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    unsigned magic = read_u32(f);
    if (magic != RADIX_MAGIC) { fprintf(stderr, "Bad radix magic\n"); exit(1); }
    int version = read_i32(f);
    if (version != 2 && version != 3) { fprintf(stderr, "Radix format version %d unsupported (need v2 or v3). Rebuild index.\n", version); exit(1); }
    // v3 stores edge tokens and counts_tok as int16 on disk (was int32 in v2).
    // We promote to int32 in memory for compatibility with existing kernels.
    // Phase 2 will narrow in-memory storage; this phase only reads the new format.
    const bool narrow_tokens = (version >= 3);
    const int token_bytes = narrow_tokens ? 2 : 4;
    t.radix_count = read_i32(f);
    t.depth_file_count = read_i32(f);
    fread(&t.total_edge_chars, 8, 1, f);
    read_i32(f); // corpus_token_count
    read_i32(f); // vocab_size
    read_u64(f); // corpus_hash
    int tlen = read_i32(f);
    fseek(f, tlen, SEEK_CUR);
    fclose(f);

    printf("  Radix trie: %d nodes, %lld total edge chars, %d endpoint depths\n",
           t.radix_count, t.total_edge_chars, t.depth_file_count);

    // Allocate flat arrays
    t.parents                = (int*)calloc(t.radix_count, sizeof(int));
    t.edge_starts            = (int*)calloc(t.radix_count, sizeof(int));
    t.edge_lens              = (int*)calloc(t.radix_count, sizeof(int));
    t.edge_first_char_depths = (int*)calloc(t.radix_count, sizeof(int));
    t.edge_mass              = (int*)calloc(t.radix_count, sizeof(int));
    t.edge_tokens_flat       = (int*)calloc((long long)t.total_edge_chars, sizeof(int));
    t.endpoint_depth_start   = (int*)calloc(t.depth_file_count + 1, sizeof(int));
    t.endpoint_depth_count   = (int*)calloc(t.depth_file_count, sizeof(int));
    t.counts_offset          = (int*)calloc(t.radix_count + 1, sizeof(int));

    // Pass 1: read structure + build counts_offset (counting)
    long long edge_fill_pos = 0;
    long long total_counts_local = 0;
    int* entry_counts_per_node = (int*)calloc(t.radix_count, sizeof(int));
    // v2: each counts entry is 8 bytes (i32 token + i32 count); v3: 6 bytes (i16 + i32)
    const int counts_entry_bytes = narrow_tokens ? 6 : 8;
    for (int d = 0; d < t.depth_file_count; d++) {
        snprintf(path, sizeof(path), "%s/radix_depth_%03d.bin", dir, d);
        f = fopen(path, "rb");
        if (!f) continue;  // empty depth file — skip
        unsigned m = read_u32(f);
        if (m != RADIX_MAGIC) { fprintf(stderr, "Bad radix depth magic\n"); exit(1); }
        int stored_depth = read_i32(f);
        if (stored_depth != d) { fprintf(stderr, "Radix depth mismatch\n"); exit(1); }
        int n = read_i32(f);
        t.endpoint_depth_start[d] = (int)edge_fill_pos;  // not quite right for start, but unused
        t.endpoint_depth_count[d] = n;

        for (int i = 0; i < n; i++) {
            int rid = read_i32(f);
            int parent = read_i32(f);
            int fcd = read_i32(f);
            int elen = read_i32(f);
            // Store into arrays indexed by rid
            t.parents[rid] = parent;
            t.edge_starts[rid] = (int)edge_fill_pos;
            t.edge_lens[rid] = elen;
            t.edge_first_char_depths[rid] = fcd;
            for (int e = 0; e < elen; e++) {
                t.edge_tokens_flat[edge_fill_pos + e] = narrow_tokens ? read_i16(f) : read_i32(f);
            }
            edge_fill_pos += elen;
            t.edge_mass[rid] = read_i32(f);  // v2 prefix mass
            int ec = read_i32(f);
            entry_counts_per_node[rid] = ec;
            total_counts_local += ec;
            fseek(f, ec * counts_entry_bytes, SEEK_CUR);
        }
        fclose(f);
    }

    t.total_counts = (int)total_counts_local;
    t.counts_tok = (int*)malloc(t.total_counts * sizeof(int));
    t.counts_val = (int*)malloc(t.total_counts * sizeof(int));

    // Prefix sum for counts_offset
    t.counts_offset[0] = 0;
    for (int i = 0; i < t.radix_count; i++) {
        t.counts_offset[i + 1] = t.counts_offset[i] + entry_counts_per_node[i];
    }
    free(entry_counts_per_node);

    // Fix endpoint_depth_start to be a proper radix_id range boundary.
    // Since we don't enforce a specific id ordering per depth, we build
    // endpoint_depth_start by prefix-summing the counts.
    t.endpoint_depth_start[0] = 0;
    for (int d = 0; d < t.depth_file_count; d++) {
        t.endpoint_depth_start[d + 1] = t.endpoint_depth_start[d] + t.endpoint_depth_count[d];
    }

    // Pass 2: read counts
    long long ci = 0;
    // We need to place counts for rid at [counts_offset[rid] .. counts_offset[rid+1]]
    for (int d = 0; d < t.depth_file_count; d++) {
        snprintf(path, sizeof(path), "%s/radix_depth_%03d.bin", dir, d);
        f = fopen(path, "rb");
        if (!f) continue;
        read_u32(f); read_i32(f);
        int n = read_i32(f);
        for (int i = 0; i < n; i++) {
            int rid = read_i32(f);
            fseek(f, 3 * 4, SEEK_CUR); // parent, fcd, edge_len
            int elen = t.edge_lens[rid];
            fseek(f, elen * token_bytes, SEEK_CUR); // skip edge tokens (v2: 4B each; v3: 2B each)
            fseek(f, 4, SEEK_CUR);                   // skip edge_mass (v2 + v3 both i32)
            int ec = read_i32(f);
            int out_off = t.counts_offset[rid];
            for (int e = 0; e < ec; e++) {
                t.counts_tok[out_off + e] = narrow_tokens ? read_i16(f) : read_i32(f);
                t.counts_val[out_off + e] = read_i32(f);
            }
        }
        fclose(f);
    }

    // Build ancestor character-position chain for each radix node.
    // For radix r, ancestor chars = ancestor edges concatenated (root → leaf order).
    // If radix r's parent is p, ancestor_chars[r] = ancestor_chars[p] + edge_chars(p).
    // Walk radix nodes in radix_id order assuming parent_id < child_id (true because
    // the builder assigns ids in a BFS-like order — parent is always emitted before child).
    t.ancestor_char_offsets = (int*)malloc((t.radix_count + 1) * sizeof(int));

    // First pass: compute lengths
    long long* anc_lens = (long long*)calloc(t.radix_count, sizeof(long long));
    anc_lens[0] = 0;  // root has no ancestors
    long long total_anc_chars = 0;
    for (int r = 1; r < t.radix_count; r++) {
        int p = t.parents[r];
        if (p < 0 || p >= t.radix_count) {
            // Should not happen (parent 0 = virtual root)
            anc_lens[r] = 0;
        } else {
            // Parent's ancestry + parent's own edge (parent's edge is our ancestor)
            anc_lens[r] = anc_lens[p] + t.edge_lens[p];
        }
        total_anc_chars += anc_lens[r];
    }
    t.total_ancestor_chars = total_anc_chars;
    t.ancestor_char_ids = (int*)malloc(total_anc_chars * sizeof(int));

    // Offsets
    t.ancestor_char_offsets[0] = 0;
    for (int r = 0; r < t.radix_count; r++) {
        t.ancestor_char_offsets[r + 1] = t.ancestor_char_offsets[r] + (int)anc_lens[r];
    }

    // Fill ancestor_char_ids: for each r, copy parent's ancestry + parent's edge chars.
    for (int r = 1; r < t.radix_count; r++) {
        int p = t.parents[r];
        if (p < 0 || p >= t.radix_count) continue;
        int out = t.ancestor_char_offsets[r];
        int parent_anc_off = t.ancestor_char_offsets[p];
        int parent_anc_len = (int)anc_lens[p];
        // Copy parent's ancestor chars
        memcpy(&t.ancestor_char_ids[out], &t.ancestor_char_ids[parent_anc_off],
               parent_anc_len * sizeof(int));
        // Then append parent's own edge character positions
        int parent_edge_start = t.edge_starts[p];
        int parent_edge_len = t.edge_lens[p];
        for (int e = 0; e < parent_edge_len; e++) {
            t.ancestor_char_ids[out + parent_anc_len + e] = parent_edge_start + e;
        }
    }
    free(anc_lens);

    printf("  Radix loaded: %d counts entries, %lld ancestor char entries\n",
           t.total_counts, t.total_ancestor_chars);

    // mean_edge_mass[d] = average edge_mass over all radix nodes whose edge
    // spans depth d. Used as a proxy for "expected mass at this depth" in
    // joint mass weighting (prefix tree and suffix tree have statistically
    // identical per-depth distributions, so the prefix tree's per-depth mean
    // serves as the suffix-side complementary mass at the corresponding
    // complementary depth).
    {
        int max_d = t.depth_file_count + 1;
        long long* sum_per_d = (long long*)calloc(max_d, sizeof(long long));
        long long* cnt_per_d = (long long*)calloc(max_d, sizeof(long long));
        for (int r = 1; r < t.radix_count; r++) {
            int fcd = t.edge_first_char_depths[r];
            int elen = t.edge_lens[r];
            long long m = (long long)t.edge_mass[r];
            for (int d = fcd; d < fcd + elen; d++) {
                if (d >= 0 && d < max_d) {
                    sum_per_d[d] += m;
                    cnt_per_d[d] += 1;
                }
            }
        }
        t.mean_edge_mass = (double*)malloc(max_d * sizeof(double));
        for (int d = 0; d < max_d; d++) {
            t.mean_edge_mass[d] = (cnt_per_d[d] > 0)
                ? (double)sum_per_d[d] / (double)cnt_per_d[d]
                : 1.0;
        }
        free(sum_per_d); free(cnt_per_d);
    }

    // d_split: depth at which each node's path first becomes unique (mass=1).
    // Walk the parent chain root→node, find shallowest ancestor with edge_mass=1.
    // Used by per-leaf depth-routing (variable-threshold variant).
    t.d_split = (int*)malloc(t.radix_count * sizeof(int));
    {
        int* chain = (int*)malloc((t.depth_file_count + 1) * sizeof(int));
        int n_resolved = 0;
        long long sum_dsplit = 0;
        for (int r = 0; r < t.radix_count; r++) {
            int len = 0;
            int cur = r;
            while (cur > 0 && len <= t.depth_file_count) {
                chain[len++] = cur;
                cur = t.parents[cur];
                if (cur == r) break;  // safety: shouldn't happen but avoid loop
            }
            // chain[0]=r, chain[len-1] = shallowest ancestor (just under root).
            // Walk root→r (reverse) and pick first node with edge_mass==1.
            int dsplit = INT_MAX;
            for (int i = len - 1; i >= 0; i--) {
                int n = chain[i];
                if (t.edge_mass[n] == 1) {
                    dsplit = t.edge_first_char_depths[n];
                    break;
                }
            }
            t.d_split[r] = dsplit;
            if (dsplit != INT_MAX) { n_resolved++; sum_dsplit += dsplit; }
        }
        free(chain);
        if (n_resolved > 0) {
            printf("  d_split: %d/%d nodes have a mass=1 ancestor (mean depth %.2f)\n",
                   n_resolved, t.radix_count, (double)sum_dsplit / n_resolved);
        }
    }

    return t;
}

TrieData load_trie(const char* dir) {
    TrieData t;
    memset(&t, 0, sizeof(t));

    // Read meta.bin
    char path[1024];
    snprintf(path, sizeof(path), "%s/meta.bin", dir);
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }

    unsigned magic = read_u32(f);
    if (magic != LEVELED_MAGIC) { fprintf(stderr, "Bad meta magic\n"); exit(1); }
    int version = read_i32(f);
    if (version != 1) { fprintf(stderr, "Bad version %d\n", version); exit(1); }
    int max_depth_raw = read_i32(f);
    read_i32(f); // max_starts
    read_i32(f); // start_offset
    read_i32(f); // starts_used
    t.total_nodes = read_i32(f);
    t.depth_file_count = read_i32(f);
    read_i32(f); // corpus_token_count
    read_i32(f); // vocab_size
    read_u64(f); // corpus_hash
    int tlen = read_i32(f);
    fseek(f, tlen, SEEK_CUR); // skip tokenizer_tag
    fclose(f);

    t.max_depth = t.depth_file_count - 1;
    printf("  Trie: %d nodes, %d depth files (max_depth=%d)\n",
           t.total_nodes, t.depth_file_count, t.max_depth);

    // Allocate flat arrays
    t.tokens  = (int*)calloc(t.total_nodes, sizeof(int));
    t.parents = (int*)calloc(t.total_nodes, sizeof(int));
    t.depths  = (int*)calloc(t.total_nodes, sizeof(int));
    t.depth_start = (int*)calloc(t.depth_file_count + 1, sizeof(int));
    t.depth_count = (int*)calloc(t.depth_file_count, sizeof(int));
    t.counts_offset = (int*)calloc(t.total_nodes + 1, sizeof(int));

    // First pass: count total entries for counts arrays
    int total_counts = 0;
    // Also need a temp array to map node_id → flat index
    // Since nodes are stored in depth files in order, we can use node_id directly
    // as index into the flat arrays.

    // Read each depth file
    int flat_idx = 0;
    for (int d = 0; d < t.depth_file_count; d++) {
        snprintf(path, sizeof(path), "%s/depth_%03d.bin", dir, d);
        f = fopen(path, "rb");
        if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }

        magic = read_u32(f);
        if (magic != LEVELED_MAGIC) { fprintf(stderr, "Bad depth magic in %s\n", path); exit(1); }
        int stored_depth = read_i32(f);
        if (stored_depth != d) { fprintf(stderr, "Depth mismatch in %s\n", path); exit(1); }
        int n = read_i32(f);

        t.depth_start[d] = flat_idx;
        t.depth_count[d] = n;

        for (int i = 0; i < n; i++) {
            int id = read_i32(f);
            int parent = read_i32(f);
            int token = read_i32(f);
            int depth = read_i32(f);
            read_i32(f); // child_count
            read_i32(f); // first_child
            int entry_count = read_i32(f);

            // Store in flat arrays indexed by node_id
            t.tokens[id] = token;
            t.parents[id] = parent;
            t.depths[id] = depth;

            t.counts_offset[id] = total_counts;
            total_counts += entry_count;

            // Skip entries for now (we'll re-read)
            fseek(f, entry_count * 8, SEEK_CUR);
            flat_idx++;
        }
        fclose(f);
    }
    t.depth_start[t.depth_file_count] = flat_idx;
    t.total_counts = total_counts;
    t.counts_offset[t.total_nodes] = total_counts; // sentinel

    // Allocate counts arrays
    t.counts_tok = (int*)malloc(total_counts * sizeof(int));
    t.counts_val = (int*)malloc(total_counts * sizeof(int));

    // Second pass: read counts
    int counts_idx = 0;
    for (int d = 0; d < t.depth_file_count; d++) {
        snprintf(path, sizeof(path), "%s/depth_%03d.bin", dir, d);
        f = fopen(path, "rb");
        read_u32(f); read_i32(f); // magic, depth
        int n = read_i32(f);

        for (int i = 0; i < n; i++) {
            int id = read_i32(f);
            read_i32(f); read_i32(f); read_i32(f); // parent, token, depth
            read_i32(f); read_i32(f); // child_count, first_child
            int entry_count = read_i32(f);

            for (int e = 0; e < entry_count; e++) {
                int tok = read_i32(f);
                int cnt = read_i32(f);
                t.counts_tok[counts_idx] = tok;
                t.counts_val[counts_idx] = cnt;
                counts_idx++;
            }
        }
        fclose(f);
    }

    // Build proper counts_offset: scan by node_id
    // We already set counts_offset[id] during first pass, but we need to
    // ensure it's a proper offset array. Let's rebuild it properly.
    // The issue is that node_ids may not be contiguous or ordered.
    // Actually they are: node_id 0 is root, then depth-1 nodes get consecutive ids,
    // etc. So counts_offset[id] should be correct from the first pass.
    // But we need the sentinel: counts_offset[id+1] gives end for node id.
    // Problem: for nodes without counts, counts_offset[id] == counts_offset[id+1]
    // should hold. Let's fix this with a forward fill.
    // Actually the streaming builder assigns node ids sequentially by depth,
    // so for nodes with no counts (entry_count=0), offset[id] == offset[id]
    // and offset[id+1] should be the same. This is already correct because
    // we only advance total_counts by entry_count.
    // But we need to handle gaps: root (id 0) and nodes with 0 entries need
    // their counts_offset set correctly. Let's do a proper scan.

    // Rebuild counts_offset properly using a separate pass
    {
        int* temp_offset = (int*)calloc(t.total_nodes + 1, sizeof(int));
        // First, count entries per node
        int* entry_counts = (int*)calloc(t.total_nodes, sizeof(int));

        for (int d = 0; d < t.depth_file_count; d++) {
            snprintf(path, sizeof(path), "%s/depth_%03d.bin", dir, d);
            f = fopen(path, "rb");
            read_u32(f); read_i32(f);
            int n = read_i32(f);
            for (int i = 0; i < n; i++) {
                int id = read_i32(f);
                fseek(f, 5 * 4, SEEK_CUR); // parent, token, depth, child_count, first_child
                int ec = read_i32(f);
                entry_counts[id] = ec;
                fseek(f, ec * 8, SEEK_CUR);
            }
            fclose(f);
        }

        // Prefix sum
        temp_offset[0] = 0;
        for (int i = 0; i < t.total_nodes; i++) {
            temp_offset[i + 1] = temp_offset[i] + entry_counts[i];
        }
        free(t.counts_offset);
        t.counts_offset = temp_offset;
        free(entry_counts);
    }

    // Build ancestor chains
    // For each node, the ancestor chain is: [root_child, ..., grandparent, parent]
    // (the node ids along the path from depth 1 to parent, NOT including node itself)
    // Length = depth - 1 for nodes at depth d (they attend to d-1 ancestor positions + self = d total,
    //   but the varlen attention expects the full KV including self).
    // Actually: node at depth d has position d-1. It attends to d KV entries:
    //   ancestors at depths 1..d-1 plus itself at depth d.
    // So ancestor chain for attention = [ancestor_d1, ancestor_d2, ..., ancestor_d(d-1), self]
    // Length = d

    // We'll build ancestor_ids as: for node at depth d, store the d node_ids
    // from depth 1 ancestor down to the node itself.
    // The varlen attention kernel uses these to gather K/V from the global KV cache.

    int total_ancestor = 0;
    for (int d = 0; d < t.depth_file_count; d++) {
        // Nodes at depth d each contribute d ancestor entries (including self)
        total_ancestor += t.depth_count[d] * d;
    }
    t.total_ancestor_entries = total_ancestor;
    t.ancestor_offset = (int*)malloc((t.total_nodes + 1) * sizeof(int));
    t.ancestor_ids    = (int*)malloc(total_ancestor * sizeof(int));

    // For BFS: build ancestor chains depth by depth
    // ancestor_chain[node_id] = [id_at_depth_1, id_at_depth_2, ..., id_at_depth_d]
    // For depth d nodes: chain = parent's chain + [node_id]

    // We'll use a temp buffer: for each node, store its chain.
    // Since nodes are processed BFS, parent chain is always available.
    // Use ancestor_offset/ancestor_ids directly.

    // First pass: compute offsets
    {
        int off = 0;
        // Process nodes in id order (which is depth order due to BFS build)
        // But we need depth info. Let's iterate by depth.
        // Set offset for root (depth 0, id 0)
        t.ancestor_offset[0] = 0; // root has 0 ancestors

        for (int d = 0; d < t.depth_file_count; d++) {
            snprintf(path, sizeof(path), "%s/depth_%03d.bin", dir, d);
            f = fopen(path, "rb");
            read_u32(f); read_i32(f);
            int n = read_i32(f);
            for (int i = 0; i < n; i++) {
                int id = read_i32(f);
                fseek(f, 5 * 4, SEEK_CUR); // skip parent, token, depth, child_count, first_child
                int ec = read_i32(f);       // entry_count
                fseek(f, ec * 8, SEEK_CUR); // skip entries

                t.ancestor_offset[id] = off;
                off += d; // d ancestors (including self) for node at depth d
            }
            fclose(f);
        }
        t.ancestor_offset[t.total_nodes] = off;
    }

    // Second pass: fill ancestor_ids
    for (int d = 1; d < t.depth_file_count; d++) {
        snprintf(path, sizeof(path), "%s/depth_%03d.bin", dir, d);
        f = fopen(path, "rb");
        read_u32(f); read_i32(f);
        int n = read_i32(f);
        for (int i = 0; i < n; i++) {
            int id = read_i32(f);
            int parent = read_i32(f);
            fseek(f, 3 * 4, SEEK_CUR); // skip token, depth, child_count
            fseek(f, 1 * 4, SEEK_CUR); // skip first_child
            int ec = read_i32(f);       // entry_count
            fseek(f, ec * 8, SEEK_CUR); // skip entries

            int off = t.ancestor_offset[id];
            if (d == 1) {
                // Only self
                t.ancestor_ids[off] = id;
            } else {
                // Copy parent's chain, then append self
                int parent_off = t.ancestor_offset[parent];
                int parent_len = d - 1; // parent is at depth d-1
                memcpy(&t.ancestor_ids[off], &t.ancestor_ids[parent_off],
                       parent_len * sizeof(int));
                t.ancestor_ids[off + parent_len] = id;
            }
        }
        fclose(f);
    }

    printf("  Trie loaded: %d total count entries, %d ancestor entries\n",
           t.total_counts, t.total_ancestor_entries);
    return t;
}

// ============================================================================
// Model checkpoint I/O
// ============================================================================

#define MGPT_MAGIC 0x4D475054u
// Optimizer-state footer magic ("OPT1"). When present after the weights
// section, the file contains: OPT_MAGIC (4B) + total_floats (i32) + adam_t (i32)
// + adam_m (float32[total_floats]) + adam_v (float32[total_floats]).
// Backward compat: files without this footer load normally (cold optimizer).
#define OPT_MAGIC  0x31545056u  // "OPT1" little-endian

float* load_model_weights(const char* path, Config* cfg) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open model: %s\n", path); exit(1); }

    unsigned magic = read_u32(f);
    if (magic != MGPT_MAGIC) { fprintf(stderr, "Bad model magic\n"); exit(1); }

    cfg->d_model   = read_i32(f);
    cfg->n_heads   = read_i32(f);
    cfg->n_layers  = read_i32(f);
    cfg->d_ff      = read_i32(f);
    cfg->vocab_size = read_i32(f);
    cfg->seq_len   = read_i32(f);
    cfg->head_dim  = cfg->d_model / cfg->n_heads;

    printf("  Model: d=%d heads=%d layers=%d ff=%d vocab=%d seq=%d head_dim=%d\n",
           cfg->d_model, cfg->n_heads, cfg->n_layers, cfg->d_ff,
           cfg->vocab_size, cfg->seq_len, cfg->head_dim);

    WeightOffsets wo = compute_offsets(*cfg);
    float* weights = (float*)malloc(wo.total_floats * sizeof(float));

    // Read weight matrices in Crystal's weight_mats order
    // Each mat: rows(i32), cols(i32), data(float32 * rows * cols)
    int offset = 0;
    // Count expected matrices:
    // 1 (token_emb) + n_layers * 16 (per block) + 4 (final_norm + output)
    int n_mats = 1 + cfg->n_layers * 16 + 4;

    for (int m = 0; m < n_mats; m++) {
        int rows = read_i32(f);
        int cols = read_i32(f);
        int count = rows * cols;
        fread(&weights[offset], sizeof(float), count, f);
        offset += count;
    }
    fclose(f);

    if (offset != wo.total_floats) {
        fprintf(stderr, "Weight count mismatch: read %d, expected %d\n", offset, wo.total_floats);
        exit(1);
    }

    printf("  Loaded %d weight floats (%.1f KB)\n", wo.total_floats,
           wo.total_floats * 4.0f / 1024.0f);
    return weights;
}

void save_model_weights(const char* path, const Config& cfg,
                        const float* weights, const WeightOffsets& wo) {
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot write model: %s\n", path); exit(1); }

    unsigned magic = MGPT_MAGIC;
    fwrite(&magic, 4, 1, f);
    fwrite(&cfg.d_model, 4, 1, f);
    fwrite(&cfg.n_heads, 4, 1, f);
    fwrite(&cfg.n_layers, 4, 1, f);
    fwrite(&cfg.d_ff, 4, 1, f);
    fwrite(&cfg.vocab_size, 4, 1, f);
    fwrite(&cfg.seq_len, 4, 1, f);

    // Write matrices in same order
    int D = cfg.d_model, F = cfg.d_ff, V = cfg.vocab_size;

    auto write_mat = [&](int offset, int rows, int cols) {
        fwrite(&rows, 4, 1, f);
        fwrite(&cols, 4, 1, f);
        fwrite(&weights[offset], sizeof(float), rows * cols, f);
    };

    write_mat(wo.token_emb, V, D);
    for (int i = 0; i < cfg.n_layers; i++) {
        write_mat(wo.wq_w[i], D, D); write_mat(wo.wq_b[i], 1, D);
        write_mat(wo.wk_w[i], D, D); write_mat(wo.wk_b[i], 1, D);
        write_mat(wo.wv_w[i], D, D); write_mat(wo.wv_b[i], 1, D);
        write_mat(wo.wo_w[i], D, D); write_mat(wo.wo_b[i], 1, D);
        write_mat(wo.ln1_gamma[i], 1, D); write_mat(wo.ln1_beta[i], 1, D);
        write_mat(wo.l1_w[i], D, F); write_mat(wo.l1_b[i], 1, F);
        write_mat(wo.l2_w[i], F, D); write_mat(wo.l2_b[i], 1, D);
        write_mat(wo.ln2_gamma[i], 1, D); write_mat(wo.ln2_beta[i], 1, D);
    }
    write_mat(wo.final_gamma, 1, D); write_mat(wo.final_beta, 1, D);
    write_mat(wo.out_w, D, V); write_mat(wo.out_b, 1, V);
    fclose(f);
}

// Try to load optimizer-state footer from a model checkpoint.
// Returns true if found and loaded; false if checkpoint is the older
// weights-only format (or doesn't exist / has a malformed footer).
// Buffers must be pre-allocated with total_floats floats each.
bool load_optimizer_state(const char* path, int total_floats,
                          float* h_adam_m, float* h_adam_v, int* adam_t) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;

    unsigned magic = read_u32(f);
    if (magic != MGPT_MAGIC) { fclose(f); return false; }
    int d_model_   = read_i32(f);
    int n_heads_   = read_i32(f); (void)n_heads_;
    int n_layers_  = read_i32(f);
    int d_ff_      = read_i32(f); (void)d_ff_;
    int vocab_size_ = read_i32(f); (void)vocab_size_;
    int seq_len_   = read_i32(f); (void)seq_len_; (void)d_model_;

    int n_mats = 1 + n_layers_ * 16 + 4;
    for (int m = 0; m < n_mats; m++) {
        int rows = read_i32(f);
        int cols = read_i32(f);
        long long bytes = (long long)rows * cols * sizeof(float);
        if (fseek(f, bytes, SEEK_CUR) != 0) { fclose(f); return false; }
    }

    // Try OPT footer
    unsigned opt_magic = 0;
    if (fread(&opt_magic, 4, 1, f) != 1) { fclose(f); return false; }
    if (opt_magic != OPT_MAGIC) { fclose(f); return false; }
    int stored_total = read_i32(f);
    if (stored_total != total_floats) {
        fprintf(stderr, "  Warning: opt-state total_floats mismatch (file=%d, expected=%d); ignoring\n",
                stored_total, total_floats);
        fclose(f);
        return false;
    }
    int stored_t = read_i32(f);
    *adam_t = stored_t;
    size_t got_m = fread(h_adam_m, sizeof(float), total_floats, f);
    size_t got_v = fread(h_adam_v, sizeof(float), total_floats, f);
    fclose(f);
    if ((int)got_m != total_floats || (int)got_v != total_floats) {
        fprintf(stderr, "  Warning: opt-state truncated; ignoring (got %zu/%zu floats)\n", got_m, got_v);
        return false;
    }
    return true;
}

// Append optimizer-state footer to an existing model checkpoint.
// Used after save_model_weights to make a streaming-compatible checkpoint.
void append_optimizer_state(const char* path, int total_floats,
                             const float* h_adam_m, const float* h_adam_v, int adam_t) {
    FILE* f = fopen(path, "ab");
    if (!f) { fprintf(stderr, "Cannot append optimizer state to %s\n", path); return; }
    unsigned opt_magic = OPT_MAGIC;
    fwrite(&opt_magic, 4, 1, f);
    fwrite(&total_floats, 4, 1, f);
    fwrite(&adam_t, 4, 1, f);
    fwrite(h_adam_m, sizeof(float), total_floats, f);
    fwrite(h_adam_v, sizeof(float), total_floats, f);
    fclose(f);
}

// ============================================================================
// NEW KERNELS
// ============================================================================

// --- RoPE with per-row position indices ---
// Each row i uses position pos[i] to look up cos/sin cache.
// x: [N, dim], positions: [N], cos_cache/sin_cache: [max_seq, dim]
__global__ void rope_batched_kernel(float* x, const int* positions,
                                     const float* cos_cache, const float* sin_cache,
                                     int N, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * (dim / 2);
    if (idx >= total) return;

    int row = idx / (dim / 2);
    int half_i = idx % (dim / 2);
    int pos = positions[row];

    int j0 = 2 * half_i;
    int j1 = j0 + 1;
    float x0 = x[row * dim + j0];
    float x1 = x[row * dim + j1];

    float c = cos_cache[pos * dim + j0];
    float s = sin_cache[pos * dim + j0];

    x[row * dim + j0] = x0 * c - x1 * s;
    x[row * dim + j1] = x0 * s + x1 * c;
}

// Inverse RoPE for backward
__global__ void rope_batched_inverse_kernel(float* x, const int* positions,
                                             const float* cos_cache, const float* sin_cache,
                                             int N, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * (dim / 2);
    if (idx >= total) return;

    int row = idx / (dim / 2);
    int half_i = idx % (dim / 2);
    int pos = positions[row];

    int j0 = 2 * half_i;
    int j1 = j0 + 1;
    float x0 = x[row * dim + j0];
    float x1 = x[row * dim + j1];

    float c = cos_cache[pos * dim + j0];
    float s = sin_cache[pos * dim + j0];

    // Inverse rotation: transpose of rotation matrix
    x[row * dim + j0] =  x0 * c + x1 * s;
    x[row * dim + j1] = -x0 * s + x1 * c;
}

void launch_rope_batched(float* x, const int* positions,
                          const float* cos_cache, const float* sin_cache,
                          int N, int dim) {
    int total = N * (dim / 2);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    rope_batched_kernel<<<blocks, threads>>>(x, positions, cos_cache, sin_cache, N, dim);
}

// --- Scalar-position RoPE: rotate every row by the same angle ---
// Used for virtual-tree cycle shifts where all queries in a chunk share the
// same shift. Composes multiplicatively with an already-rotated buffer:
//   x_before = Rot(θ(real_pos)) · x_raw   (from prior launch_rope_batched)
//   x_after  = Rot(θ(shift))    · x_before = Rot(θ(real_pos + shift)) · x_raw
__global__ void rope_batched_scalar_kernel(float* x, int scalar_pos,
                                            const float* cos_cache, const float* sin_cache,
                                            int N, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * (dim / 2);
    if (idx >= total) return;
    int row = idx / (dim / 2);
    int half_i = idx % (dim / 2);

    int j0 = 2 * half_i;
    int j1 = j0 + 1;
    float x0 = x[row * dim + j0];
    float x1 = x[row * dim + j1];

    float c = cos_cache[scalar_pos * dim + j0];
    float s = sin_cache[scalar_pos * dim + j0];

    x[row * dim + j0] = x0 * c - x1 * s;
    x[row * dim + j1] = x0 * s + x1 * c;
}

void launch_rope_batched_scalar(float* x, int scalar_pos,
                                 const float* cos_cache, const float* sin_cache,
                                 int N, int dim) {
    int total = N * (dim / 2);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    rope_batched_scalar_kernel<<<blocks, threads>>>(x, scalar_pos, cos_cache, sin_cache, N, dim);
}

// Inverse scalar-position RoPE (for backward dQ).
__global__ void rope_batched_scalar_inverse_kernel(float* x, int scalar_pos,
                                                    const float* cos_cache, const float* sin_cache,
                                                    int N, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * (dim / 2);
    if (idx >= total) return;
    int row = idx / (dim / 2);
    int half_i = idx % (dim / 2);

    int j0 = 2 * half_i;
    int j1 = j0 + 1;
    float x0 = x[row * dim + j0];
    float x1 = x[row * dim + j1];

    float c = cos_cache[scalar_pos * dim + j0];
    float s = sin_cache[scalar_pos * dim + j0];

    x[row * dim + j0] =  x0 * c + x1 * s;
    x[row * dim + j1] = -x0 * s + x1 * c;
}

void launch_rope_batched_scalar_inverse(float* x, int scalar_pos,
                                         const float* cos_cache, const float* sin_cache,
                                         int N, int dim) {
    int total = N * (dim / 2);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    rope_batched_scalar_inverse_kernel<<<blocks, threads>>>(x, scalar_pos, cos_cache, sin_cache, N, dim);
}

void launch_rope_batched_inverse(float* x, const int* positions,
                                  const float* cos_cache, const float* sin_cache,
                                  int N, int dim) {
    int total = N * (dim / 2);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    rope_batched_inverse_kernel<<<blocks, threads>>>(x, positions, cos_cache, sin_cache, N, dim);
}

// --- KV scatter: store projected K/V into global KV cache ---
// src: [N, d_model], node_ids: [N], dst: [total_nodes, d_model]
// For each row i: dst[node_ids[i]] = src[i]
__global__ void kv_scatter_kernel(const float* src, const int* node_ids,
                                   float* dst, int N, int d_model) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * d_model;
    if (idx >= total) return;

    int row = idx / d_model;
    int col = idx % d_model;
    int nid = node_ids[row];
    dst[nid * d_model + col] = src[row * d_model + col];
}

void launch_kv_scatter(const float* src, const int* node_ids,
                        float* dst, int N, int d_model) {
    int total = N * d_model;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    kv_scatter_kernel<<<blocks, threads>>>(src, node_ids, dst, N, d_model);
}

// --- KV gather: gather ancestor K/V into packed buffer for varlen attention ---
// For node i with kv_length = ancestor_count[i]:
//   for p in 0..kv_length-1:
//     for h in 0..n_heads-1:
//       packed[kv_offset[i]*n_heads*hd + (p*n_heads+h)*hd .. +hd]
//         = global_kv[ancestor_ids[anc_off[i]+p] * d_model + h*hd .. +hd]
//
// The varlen attention kernel expects K/V packed as:
//   [total_kv_positions, head_dim] with heads interleaved at each position.
// So for position p of node i, head h:
//   packed[(kv_offset[i] + p) * n_heads + h] * head_dim + j]
//
// global_kv is stored as [total_nodes, d_model] where d_model = n_heads * head_dim

__global__ void kv_gather_kernel(const float* global_kv,
                                  const int* ancestor_ids,
                                  const int* ancestor_offsets, // per-node offset into ancestor_ids
                                  const int* kv_offsets,       // per-node offset into packed output
                                  const int* kv_lengths,       // per-node prefix length
                                  float* packed_kv,
                                  int N, int n_heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Total work items = sum of kv_lengths * n_heads * head_dim across nodes
    // Instead: one thread per (node, position, head, dim_element) is wasteful.
    // Simpler: one thread per output element in packed_kv.
    // packed_kv has total_packed_positions * n_heads * head_dim elements.
    // But we don't know total_packed_positions easily in the kernel.
    // Better: iterate over nodes, each node's work = kv_length * n_heads * head_dim.

    // Alternative simpler approach: iterate (node, position_in_prefix, dim_col)
    // where dim_col spans the full d_model = n_heads * head_dim.
    // This is straightforward: N * max_kv_length * d_model threads, with bounds check.

    // Actually, simplest: grid over N nodes × max_len × d_model.
    // But max_len varies. Let's use a 1D grid and compute which (node, pos, col) we map to.

    // For simplicity, let's just iterate in the kernel with a grid over N * d_model:
    // Each thread handles one (node, col), and loops over prefix positions.
    int d_model = n_heads * head_dim;
    int nidx = idx / d_model;
    int col = idx % d_model;
    if (nidx >= N) return;

    int anc_off = ancestor_offsets[nidx];
    int kv_off = kv_offsets[nidx];
    int len = kv_lengths[nidx];

    // Map col in d_model to (head, head_col)
    int head = col / head_dim;
    int hcol = col % head_dim;

    for (int p = 0; p < len; p++) {
        int ancestor = ancestor_ids[anc_off + p];
        float val = global_kv[ancestor * d_model + col];
        // packed layout: [(kv_off + p) * n_heads + head] * head_dim + hcol
        packed_kv[((kv_off + p) * n_heads + head) * head_dim + hcol] = val;
    }
}

void launch_kv_gather(const float* global_kv,
                       const int* ancestor_ids,
                       const int* ancestor_offsets,
                       const int* kv_offsets,
                       const int* kv_lengths,
                       float* packed_kv,
                       int N, int n_heads, int head_dim) {
    int d_model = n_heads * head_dim;
    int total = N * d_model;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    kv_gather_kernel<<<blocks, threads>>>(global_kv, ancestor_ids, ancestor_offsets,
                                           kv_offsets, kv_lengths, packed_kv,
                                           N, n_heads, head_dim);
}

// --- BF16 variants: KV cache storage is bf16, packed buffers + attention stay fp32 ---
// Scatter: convert fp32 → bf16 on write into the global cache.
__global__ void kv_scatter_kernel_bf16(const float* src, const int* node_ids,
                                        __nv_bfloat16* dst, int N, int d_model) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * d_model;
    if (idx >= total) return;
    int row = idx / d_model;
    int col = idx % d_model;
    int nid = node_ids[row];
    dst[nid * d_model + col] = __float2bfloat16(src[row * d_model + col]);
}

void launch_kv_scatter_bf16(const float* src, const int* node_ids,
                             __nv_bfloat16* dst, int N, int d_model) {
    int total = N * d_model;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    kv_scatter_kernel_bf16<<<blocks, threads>>>(src, node_ids, dst, N, d_model);
}

// Gather: read bf16, convert to fp32 on write into the packed buffer.
__global__ void kv_gather_kernel_bf16(const __nv_bfloat16* global_kv,
                                       const int* ancestor_ids,
                                       const int* ancestor_offsets,
                                       const int* kv_offsets,
                                       const int* kv_lengths,
                                       float* packed_kv,
                                       int N, int n_heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int d_model = n_heads * head_dim;
    int nidx = idx / d_model;
    int col = idx % d_model;
    if (nidx >= N) return;

    int anc_off = ancestor_offsets[nidx];
    int kv_off = kv_offsets[nidx];
    int len = kv_lengths[nidx];
    int head = col / head_dim;
    int hcol = col % head_dim;

    for (int p = 0; p < len; p++) {
        int ancestor = ancestor_ids[anc_off + p];
        float val = __bfloat162float(global_kv[ancestor * d_model + col]);
        packed_kv[((kv_off + p) * n_heads + head) * head_dim + hcol] = val;
    }
}

void launch_kv_gather_bf16(const __nv_bfloat16* global_kv,
                            const int* ancestor_ids,
                            const int* ancestor_offsets,
                            const int* kv_offsets,
                            const int* kv_lengths,
                            float* packed_kv,
                            int N, int n_heads, int head_dim) {
    int d_model = n_heads * head_dim;
    int total = N * d_model;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    kv_gather_kernel_bf16<<<blocks, threads>>>(global_kv, ancestor_ids, ancestor_offsets,
                                                kv_offsets, kv_lengths, packed_kv,
                                                N, n_heads, head_dim);
}

// --- Compact-cache scatter: write K/V to bf16 cache indexed by compact_slot ---
// char_pos[row] is the GLOBAL character position of query row. compact_slot[cp]
// remaps to a compact-cache index or -1 for mass=1 positions (which we skip).
__global__ void kv_scatter_compact_bf16(const float* src, const int* char_pos,
                                         const int* compact_slot,
                                         __nv_bfloat16* dst, int N, int d_model) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * d_model;
    if (idx >= total) return;
    int row = idx / d_model;
    int col = idx % d_model;
    int cp = char_pos[row];
    int slot = compact_slot[cp];
    if (slot < 0) return;  // mass=1 char: skip
    dst[(long long)slot * d_model + col] = __float2bfloat16(src[row * d_model + col]);
}

void launch_kv_scatter_compact_bf16(const float* src, const int* char_pos,
                                     const int* compact_slot,
                                     __nv_bfloat16* dst, int N, int d_model) {
    int total = N * d_model;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    kv_scatter_compact_bf16<<<blocks, threads>>>(src, char_pos, compact_slot, dst, N, d_model);
}

// --- Compact-cache gather for ANCESTORS only ---
// Ancestors are always mass>1 so they always have a compact_slot >= 0.
// Writes the first anc_lengths[i] rows of each query's packed prefix.
__global__ void kv_gather_anc_compact_bf16(const __nv_bfloat16* global_kv,
                                            const int* ancestor_ids,
                                            const int* ancestor_offsets, // per-node offset into ancestor_ids
                                            const int* kv_offsets,       // per-node offset into packed output
                                            const int* anc_lengths,      // per-node ANCESTOR-only length
                                            const int* compact_slot,
                                            float* packed_kv,
                                            int N, int n_heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int d_model = n_heads * head_dim;
    int nidx = idx / d_model;
    int col = idx % d_model;
    if (nidx >= N) return;

    int anc_off = ancestor_offsets[nidx];
    int kv_off  = kv_offsets[nidx];
    int len     = anc_lengths[nidx];
    int head = col / head_dim;
    int hcol = col % head_dim;

    for (int p = 0; p < len; p++) {
        int char_pos = ancestor_ids[anc_off + p];
        int slot = compact_slot[char_pos];
        // slot < 0 should not happen for ancestors (always mass>1); guard anyway.
        float val = (slot >= 0) ? __bfloat162float(global_kv[(long long)slot * d_model + col]) : 0.0f;
        packed_kv[((kv_off + p) * n_heads + head) * head_dim + hcol] = val;
    }
}

void launch_kv_gather_anc_compact_bf16(const __nv_bfloat16* global_kv,
                                        const int* ancestor_ids,
                                        const int* ancestor_offsets,
                                        const int* kv_offsets,
                                        const int* anc_lengths,
                                        const int* compact_slot,
                                        float* packed_kv,
                                        int N, int n_heads, int head_dim) {
    int d_model = n_heads * head_dim;
    int total = N * d_model;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    kv_gather_anc_compact_bf16<<<blocks, threads>>>(global_kv, ancestor_ids, ancestor_offsets,
                                                     kv_offsets, anc_lengths, compact_slot, packed_kv,
                                                     N, n_heads, head_dim);
}

// --- K-specific ancestor gather with delta-RoPE ---
// Stored K is post-real-RoPE (rotated by θ(real_pos)). For virtual-tree reuse,
// the same cache entry needs to serve a different read position. We apply a
// delta rotation Δθ = θ(read_pos) - θ(real_pos) per dim-pair.
//
// Using the identity:
//   cos(a - b) = cos(a)cos(b) + sin(a)sin(b)
//   sin(a - b) = sin(a)cos(b) - cos(a)sin(b)
// we can compute the delta rotation from two position lookups in the same
// RoPE cos/sin tables, without needing a separate delta cache.
//
// When read_pos == real_pos (K=1 training, no virtual), Δθ = 0 and the
// kernel is bit-identical to kv_gather_anc_compact_bf16.
__global__ void kv_gather_k_anc_delta_rope_kernel(const __nv_bfloat16* global_k,
                                                    const int* ancestor_ids,       // char_pos
                                                    const int* ancestor_offsets,
                                                    const int* kv_offsets,
                                                    const int* anc_lengths,
                                                    const int* compact_slot,
                                                    const int* read_pos_flat,      // [T_anc] RoPE read position
                                                    const int* real_pos_of_char,   // [total_edge_chars] char_pos → real RoPE pos
                                                    const float* rope_cos,
                                                    const float* rope_sin,
                                                    float* packed_k,
                                                    int N, int n_heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int d_model = n_heads * head_dim;
    int total = N * (d_model / 2);   // one thread per (node, half-dim)
    if (idx >= total) return;

    int nidx = idx / (d_model / 2);
    int hi   = idx % (d_model / 2);   // half-dim index

    int anc_off = ancestor_offsets[nidx];
    int kv_off  = kv_offsets[nidx];
    int len     = anc_lengths[nidx];

    int j0 = 2 * hi;
    int j1 = j0 + 1;
    int head = j0 / head_dim;
    int hcol0 = j0 % head_dim;
    // (hcol1 = hcol0 + 1, always in the same head since head_dim is even)

    for (int p = 0; p < len; p++) {
        int char_pos = ancestor_ids[anc_off + p];
        int slot = compact_slot[char_pos];
        float x0, x1;
        if (slot >= 0) {
            x0 = __bfloat162float(global_k[(long long)slot * d_model + j0]);
            x1 = __bfloat162float(global_k[(long long)slot * d_model + j1]);
        } else {
            x0 = 0.0f; x1 = 0.0f;
        }

        // Delta rotation: Rot(θ_read - θ_real) applied to (x0, x1).
        // Stored entry used rope angle at real_pos; target is read_pos.
        int rp = real_pos_of_char[char_pos];
        int wp = read_pos_flat[anc_off + p];
        float cr = rope_cos[rp * head_dim + hcol0];
        float sr = rope_sin[rp * head_dim + hcol0];
        float cw = rope_cos[wp * head_dim + hcol0];
        float sw = rope_sin[wp * head_dim + hcol0];
        float cd = cw * cr + sw * sr;   // cos(θ_read - θ_real)
        float sd = sw * cr - cw * sr;   // sin(θ_read - θ_real)

        float y0 = x0 * cd - x1 * sd;
        float y1 = x0 * sd + x1 * cd;

        // Packed layout: [(kv_off + p) * n_heads + head] * head_dim + hcol
        int base = ((kv_off + p) * n_heads + head) * head_dim;
        packed_k[base + hcol0] = y0;
        packed_k[base + hcol0 + 1] = y1;
    }
}

void launch_kv_gather_k_anc_delta_rope(const __nv_bfloat16* global_k,
                                        const int* ancestor_ids,
                                        const int* ancestor_offsets,
                                        const int* kv_offsets,
                                        const int* anc_lengths,
                                        const int* compact_slot,
                                        const int* read_pos_flat,
                                        const int* real_pos_of_char,
                                        const float* rope_cos,
                                        const float* rope_sin,
                                        float* packed_k,
                                        int N, int n_heads, int head_dim) {
    int d_model = n_heads * head_dim;
    int total = N * (d_model / 2);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    kv_gather_k_anc_delta_rope_kernel<<<blocks, threads>>>(
        global_k, ancestor_ids, ancestor_offsets, kv_offsets, anc_lengths,
        compact_slot, read_pos_flat, real_pos_of_char, rope_cos, rope_sin,
        packed_k, N, n_heads, head_dim);
}

// --- Bias gradient: accumulate column-sum of a [rows, cols] gradient tensor ---
// bias[c] += scale * sum_{r} grad[r, c]. Simple one-block-per-column reduction;
// rows typically O(T_q) which is small enough that a single block per column is fine.
__global__ void bias_grad_accum_kernel(const float* grad, int rows, int cols,
                                        float scale, float* bias) {
    int c = blockIdx.x;
    if (c >= cols) return;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    extern __shared__ float sdata[];
    float local = 0.0f;
    for (int r = tid; r < rows; r += nthreads) local += grad[r * cols + c];
    sdata[tid] = local;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) bias[c] += scale * sdata[0];
}

void launch_bias_grad_accum(const float* grad, int rows, int cols,
                             float scale, float* bias) {
    int threads = 256;
    int smem = threads * sizeof(float);
    bias_grad_accum_kernel<<<cols, threads, smem>>>(grad, rows, cols, scale, bias);
}

// --- Extract own-edge portion of packed K/V gradient back to [T_q, D] layout ---
// Reverse of kv_copy_own_edge. For query i, the packed slots
// [kv_offset[i] + anc_length[i] .. kv_offset[i] + anc_length[i] + own_length[i])
// map to d_out[query_offsets[i] + j, col] for j in [0, own_length[i]).
// Each T_q position belongs to exactly one query, so the uncopy is a bijection.
__global__ void kv_uncopy_own_edge_kernel(const float* packed_grad,
                                            const int* query_offsets,
                                            const int* kv_offsets,
                                            const int* anc_lengths,
                                            const int* own_lengths,
                                            float* d_out,     // [T_q, D], zeroed by caller
                                            int N, int n_heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int d_model = n_heads * head_dim;
    int nidx = idx / d_model;
    int col = idx % d_model;
    if (nidx >= N) return;

    int q_off  = query_offsets[nidx];
    int kv_off = kv_offsets[nidx];
    int anc_len = anc_lengths[nidx];
    int own_len = own_lengths[nidx];
    int head = col / head_dim;
    int hcol = col % head_dim;

    for (int j = 0; j < own_len; j++) {
        int p = anc_len + j;
        float val = packed_grad[((kv_off + p) * n_heads + head) * head_dim + hcol];
        d_out[(long long)(q_off + j) * d_model + col] = val;
    }
}

void launch_kv_uncopy_own_edge(const float* packed_grad,
                                const int* query_offsets,
                                const int* kv_offsets,
                                const int* anc_lengths,
                                const int* own_lengths,
                                float* d_out,
                                int N, int n_heads, int head_dim) {
    int d_model = n_heads * head_dim;
    int total = N * d_model;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    kv_uncopy_own_edge_kernel<<<blocks, threads>>>(packed_grad, query_offsets, kv_offsets,
                                                    anc_lengths, own_lengths, d_out,
                                                    N, n_heads, head_dim);
}

// --- --anc-grad: scatter-add packed ANCESTOR gradient slice into subtree buffer ---
// For each node n_idx in the chunk and each ancestor slot p ∈ [0, anc_lengths[n_idx]),
// the packed K-grad at (kv_offsets[n_idx] + p) is dL/dK_post_rope at that
// ancestor's char_pos. We accumulate it into d_dkv_subtree_k[sub_idx], where
// sub_idx = d_compact_to_subtree_idx[compact_slot[ancestor_char_pos]].
// Multiple descendants may contribute to the same ancestor → atomic-add.
// Same kernel handles V (no RoPE), just pass d_dv_pack / d_dkv_subtree_v.
// Skipped slots: sub_idx < 0 (ancestor not in current subtree — shouldn't
// happen at pd=1 since all ancestors are in the subtree, but kernel is safe).
__global__ void scatter_anc_dkv_to_subtree_kernel(const float* packed_grad,
                                                    const int* ancestor_ids,
                                                    const int* ancestor_offsets,
                                                    const int* kv_offsets,
                                                    const int* anc_lengths,
                                                    const int* compact_slot,
                                                    const int* compact_to_subtree,
                                                    float* dkv_subtree,
                                                    float grad_scale,
                                                    int N, int n_heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int d_model = n_heads * head_dim;
    int nidx = idx / d_model;
    int col = idx % d_model;
    if (nidx >= N) return;

    int anc_off = ancestor_offsets[nidx];
    int kv_off  = kv_offsets[nidx];
    int len     = anc_lengths[nidx];
    int head = col / head_dim;
    int hcol = col % head_dim;

    for (int p = 0; p < len; p++) {
        int char_pos = ancestor_ids[anc_off + p];
        int slot = compact_slot[char_pos];
        if (slot < 0) continue;
        int sub_idx = compact_to_subtree[slot];
        if (sub_idx < 0) continue;
        float g = packed_grad[((kv_off + p) * n_heads + head) * head_dim + hcol];
        // Pre-scale by 1/T_q_chunk so this ancestor event contributes with the
        // same per-event weight that own-edge applies to its own events. This
        // keeps anc-grad and own-edge magnitudes consistent without any
        // post-hoc normalizer at fire-end.
        atomicAdd(&dkv_subtree[(long long)sub_idx * d_model + col], g * grad_scale);
    }
}

void launch_scatter_anc_dkv_to_subtree(const float* packed_grad,
                                        const int* ancestor_ids,
                                        const int* ancestor_offsets,
                                        const int* kv_offsets,
                                        const int* anc_lengths,
                                        const int* compact_slot,
                                        const int* compact_to_subtree,
                                        float* dkv_subtree,
                                        float grad_scale,
                                        int N, int n_heads, int head_dim) {
    int d_model = n_heads * head_dim;
    int total = N * d_model;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    scatter_anc_dkv_to_subtree_kernel<<<blocks, threads>>>(packed_grad, ancestor_ids,
                                                             ancestor_offsets, kv_offsets,
                                                             anc_lengths, compact_slot,
                                                             compact_to_subtree, dkv_subtree,
                                                             grad_scale,
                                                             N, n_heads, head_dim);
}

// --- --anc-grad: per-query save of ln1_out into the subtree-scoped buffer ---
// For each query in the current chunk, look up its compact_slot, then
// d_compact_to_subtree_idx, and copy ln1_out[q] into h_subtree[sub_idx].
// Queries whose char is mass=1 (slot < 0) or not in the current subtree
// (sub_idx < 0) are skipped.
__global__ void save_ln1_to_subtree_kernel(const float* ln1_out,
                                            const int* char_pos,
                                            const int* compact_slot,
                                            const int* compact_to_subtree,
                                            float* h_subtree,
                                            int T_q, int D) {
    int q = blockIdx.x;
    if (q >= T_q) return;
    int cp = char_pos[q];
    int slot = compact_slot[cp];
    if (slot < 0) return;
    int sub_idx = compact_to_subtree[slot];
    if (sub_idx < 0) return;
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        h_subtree[(long long)sub_idx * D + j] = ln1_out[(long long)q * D + j];
    }
}

void launch_save_ln1_to_subtree(const float* ln1_out,
                                 const int* char_pos,
                                 const int* compact_slot,
                                 const int* compact_to_subtree,
                                 float* h_subtree,
                                 int T_q, int D) {
    int threads = (D < 256) ? D : 256;
    save_ln1_to_subtree_kernel<<<T_q, threads>>>(ln1_out, char_pos, compact_slot,
                                                  compact_to_subtree, h_subtree, T_q, D);
}

// --- Mask the per-query gradient buffer (own-edge dK or dV) by query depth ---
// Used to implement depth-routed gradient: K-weight gradient takes only
// shallow queries (depth ≤ d_k); V-weight gradient takes only deep queries
// (depth > d_k). The d_d_ln_out propagation already happened with the FULL
// gradient before this is called — masking only affects the dWk/dWv gemms.
//   mode = 0  →  zero query rows where depth >  threshold  (K side: keep shallow)
//   mode = 1  →  zero query rows where depth <= threshold  (V side: keep deep)
__global__ void mask_grad_by_query_depth_kernel(float* grad,
                                                 const int* query_depth,
                                                 int threshold, int mode,
                                                 int T_q, int d_model) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int q   = idx / d_model;
    int col = idx % d_model;
    if (q >= T_q) return;
    int d = query_depth[q];
    bool zero = (mode == 0) ? (d > threshold) : (d <= threshold);
    if (zero) grad[(long long)q * d_model + col] = 0.0f;
}

void launch_mask_grad_by_query_depth(float* grad, const int* query_depth,
                                      int threshold, int mode,
                                      int T_q, int d_model) {
    int total = T_q * d_model;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    mask_grad_by_query_depth_kernel<<<blocks, threads>>>(grad, query_depth, threshold, mode, T_q, d_model);
}

// Per-leaf variant: threshold comes from `query_d_split[q]` instead of a
// scalar. Implements the variable-radix-cap routing where each leaf's
// decision/identity boundary is its own d* (depth at which the path first
// becomes mass=1).
__global__ void mask_grad_by_query_dsplit_kernel(float* grad,
                                                  const int* query_depth,
                                                  const int* query_d_split,
                                                  int mode,
                                                  int T_q, int d_model) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int q   = idx / d_model;
    int col = idx % d_model;
    if (q >= T_q) return;
    int d = query_depth[q];
    int k = query_d_split[q];
    bool zero = (mode == 0) ? (d > k) : (d <= k);
    if (zero) grad[(long long)q * d_model + col] = 0.0f;
}

void launch_mask_grad_by_query_dsplit(float* grad, const int* query_depth,
                                       const int* query_d_split, int mode,
                                       int T_q, int d_model) {
    int total = T_q * d_model;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    mask_grad_by_query_dsplit_kernel<<<blocks, threads>>>(grad, query_depth, query_d_split,
                                                          mode, T_q, d_model);
}

// Decision-only loss mask: zero d_loss[i] and d_d_logits[i,:] for queries
// where depth > d_split[node] + buffer. The buffer keeps a configurable
// number of post-decision-point events (so we don't strip every tail event,
// only the deeper ones). buffer=0 = strict decision-only; buffer>0 keeps
// events for a few chars into the unary cap. Multi-mass intermediate nodes
// have d_split=INT_MAX so all their queries are kept regardless.
__global__ void mask_loss_decision_only_kernel(float* d_loss, float* d_d_logits,
                                                const int* query_depth,
                                                const int* query_d_split,
                                                int buffer,
                                                int T_q, int V) {
    int q = blockIdx.x;
    if (q >= T_q) return;
    int d = query_depth[q];
    int k = query_d_split[q];
    // Saturate against overflow — k can be INT_MAX for multi-mass nodes.
    long long boundary = (long long)k + (long long)buffer;
    if ((long long)d <= boundary) return;
    // Skipped event: zero loss and gradient row.
    if (threadIdx.x == 0) d_loss[q] = 0.0f;
    for (int v = threadIdx.x; v < V; v += blockDim.x) {
        d_d_logits[(long long)q * V + v] = 0.0f;
    }
}

void launch_mask_loss_decision_only(float* d_loss, float* d_d_logits,
                                     const int* query_depth, const int* query_d_split,
                                     int buffer,
                                     int T_q, int V) {
    int threads = 128;
    mask_loss_decision_only_kernel<<<T_q, threads>>>(d_loss, d_d_logits,
                                                      query_depth, query_d_split,
                                                      buffer, T_q, V);
}

// --- Copy own-edge K/V from fresh d_k[T_q, D] into the packed prefix buffer ---
// For query i, own_len[i] positions starting at query_offsets[i] in d_k_fresh
// are appended after the query's anc_length[i] ancestor slots in packed_kv.
__global__ void kv_copy_own_edge(const float* d_k_fresh,
                                  const int* query_offsets,    // per-node offset into d_k_fresh (== q_off)
                                  const int* kv_offsets,       // per-node start in packed_kv
                                  const int* anc_lengths,      // ancestor count (own-edge starts after)
                                  const int* own_lengths,      // own-edge count per query
                                  float* packed_kv,
                                  int N, int n_heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int d_model = n_heads * head_dim;
    int nidx = idx / d_model;
    int col = idx % d_model;
    if (nidx >= N) return;

    int q_off  = query_offsets[nidx];
    int kv_off = kv_offsets[nidx];
    int anc_len = anc_lengths[nidx];
    int own_len = own_lengths[nidx];
    int head = col / head_dim;
    int hcol = col % head_dim;

    for (int j = 0; j < own_len; j++) {
        float val = d_k_fresh[(long long)(q_off + j) * d_model + col];
        int p = anc_len + j;
        packed_kv[((kv_off + p) * n_heads + head) * head_dim + hcol] = val;
    }
}

void launch_kv_copy_own_edge(const float* d_k_fresh,
                              const int* query_offsets,
                              const int* kv_offsets,
                              const int* anc_lengths,
                              const int* own_lengths,
                              float* packed_kv,
                              int N, int n_heads, int head_dim) {
    int d_model = n_heads * head_dim;
    int total = N * d_model;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    kv_copy_own_edge<<<blocks, threads>>>(d_k_fresh, query_offsets, kv_offsets,
                                           anc_lengths, own_lengths, packed_kv,
                                           N, n_heads, head_dim);
}

// --- KV scatter-add backward: accumulate dK/dV from packed gradients back to global ---
// Reverse of kv_gather: for each position in each node's prefix,
// add the packed gradient back to the global kv gradient buffer.
__global__ void kv_scatter_add_kernel(const float* packed_dkv,
                                       const int* ancestor_ids,
                                       const int* ancestor_offsets,
                                       const int* kv_offsets,
                                       const int* kv_lengths,
                                       float* global_dkv,
                                       int N, int n_heads, int head_dim) {
    int d_model = n_heads * head_dim;
    int nidx = (blockIdx.x * blockDim.x + threadIdx.x) / d_model;
    int col = (blockIdx.x * blockDim.x + threadIdx.x) % d_model;
    if (nidx >= N) return;

    int anc_off = ancestor_offsets[nidx];
    int kv_off = kv_offsets[nidx];
    int len = kv_lengths[nidx];
    int head = col / head_dim;
    int hcol = col % head_dim;

    for (int p = 0; p < len; p++) {
        int ancestor = ancestor_ids[anc_off + p];
        float val = packed_dkv[((kv_off + p) * n_heads + head) * head_dim + hcol];
        atomicAdd(&global_dkv[ancestor * d_model + col], val);
    }
}

void launch_kv_scatter_add(const float* packed_dkv,
                            const int* ancestor_ids,
                            const int* ancestor_offsets,
                            const int* kv_offsets,
                            const int* kv_lengths,
                            float* global_dkv,
                            int N, int n_heads, int head_dim) {
    int d_model = n_heads * head_dim;
    int total = N * d_model;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    kv_scatter_add_kernel<<<blocks, threads>>>(packed_dkv, ancestor_ids, ancestor_offsets,
                                                kv_offsets, kv_lengths, global_dkv,
                                                N, n_heads, head_dim);
}

// --- AGPT loss: softmax + sparse CE against count distributions ---
// logits: [N, vocab_size]
// counts_offset: [N+1] (into counts_tok/counts_val)
// Output: d_logits (gradient), loss_per_node (scalar per node)
// For node i:
//   probs = softmax(logits[i])
//   total = sum(counts_val[offset[i]..offset[i+1]])
//   loss = -sum(count_k/total * log(prob_k))
//   d_logits[i,j] = probs[j] - count_j/total (for tokens in counts, 0 otherwise)

__global__ void agpt_loss_kernel(const float* logits,
                                  const int* node_ids,       // [N] global node ids
                                  const int* counts_offset,  // [total_nodes+1]
                                  const int* counts_tok,
                                  const int* counts_val,
                                  float* d_logits,           // [N, V]
                                  float* loss_out,           // [N]
                                  int N, int V) {
    int i = blockIdx.x;
    if (i >= N) return;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    const float* in_row = logits + i * V;
    float* grad_row = d_logits + i * V;

    extern __shared__ float sdata[];

    // 1. Find max for numerical stability
    float local_max = -FLT_MAX;
    for (int j = tid; j < V; j += nthreads) {
        if (in_row[j] > local_max) local_max = in_row[j];
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s && sdata[tid + s] > sdata[tid]) sdata[tid] = sdata[tid + s];
        __syncthreads();
    }
    float max_val = sdata[0];

    // 2. Exp and sum
    float local_sum = 0.0f;
    for (int j = tid; j < V; j += nthreads) {
        float e = expf(in_row[j] - max_val);
        grad_row[j] = e;  // temporarily store exp
        local_sum += e;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / sdata[0];

    // 3. Normalize to get probabilities (stored in grad_row for now)
    for (int j = tid; j < V; j += nthreads) {
        grad_row[j] *= inv_sum;
    }
    __syncthreads();

    // 4. Compute loss and gradient from sparse counts
    // Only thread 0 does the sparse part (counts are small, typically 1-65 entries)
    if (tid == 0) {
        int nid = node_ids[i];
        int start = counts_offset[nid];
        int end = counts_offset[nid + 1];

        if (start == end) {
            // No counts — zero gradient, zero loss
            loss_out[i] = 0.0f;
            return;
        }

        int total = 0;
        for (int e = start; e < end; e++) {
            total += counts_val[e];
        }
        float total_f = (float)total;

        float loss = 0.0f;
        for (int e = start; e < end; e++) {
            int tok = counts_tok[e];
            int cnt = counts_val[e];
            float p = grad_row[tok];
            loss -= (cnt / total_f) * logf(p + 1e-10f);
            // grad: prob - target
            grad_row[tok] -= cnt / total_f;
        }
        loss_out[i] = loss;
    }
}

void launch_agpt_loss(const float* logits, const int* node_ids,
                       const int* counts_offset, const int* counts_tok,
                       const int* counts_val, float* d_logits, float* loss_out,
                       int N, int V) {
    int threads = (V < 256) ? V : 256;
    // round up to power of 2
    int t = 1;
    while (t < threads) t <<= 1;
    threads = (t < 32) ? 32 : t;
    int smem = threads * sizeof(float);
    agpt_loss_kernel<<<N, threads, smem>>>(logits, node_ids, counts_offset,
                                            counts_tok, counts_val,
                                            d_logits, loss_out, N, V);
}

// --- AGPT loss per QUERY (radix: intermediate positions + endpoints) ---
// At each query position q we do softmax(logits[q]) and compute loss.
// Intermediate position (q + 1 < query_offsets[radix_idx+1]):
//   target = d_token_ids[q + 1]  (the next character in the edge)
//   counts are effectively {target: 1}, total = 1 (deterministic unary continuation)
//   loss = -log(p_target), grad = p - onehot(target)
// Endpoint position (q + 1 == query_offsets[radix_idx+1]):
//   counts from radix_counts_{offset,tok,val}[radix_ids[radix_idx]]
//   loss = -Σ(c_t/total) log(p_t), grad = p - c/total
//
// For a pure unary chain, this makes radix ABC contribute exactly the same three
// loss terms (one per character) as non-radix A→B→C.
// entropy_lambda > 0 applies "icing" weight: w = 1 + λ · (H / log V), where H is
// the entropy of the empirical next-token distribution at an endpoint. Boosts
// loss & gradient at high-branching (information-rich) positions. Deterministic
// unary-intermediate positions have H=0 → w=1 (unchanged). Orthogonal to
// corpus-mass weighting.
//
// mass_weights != NULL applies corpus-mass weighting: each query's loss and
// gradient are scaled by mass_weights[q] = edge_mass[radix_id] / mean_mass.
// This restores corpus-frequency exposure: "the" (seen 10,000×) pulls with
// proportional weight vs "xyz" (seen 3×). The head-of-edge count is used
// (not endpoint count) so truncation drops don't reduce the weight.
__global__ void agpt_loss_per_query_kernel(
    const float* logits,          // [T_q, V]
    const int* query_to_node,     // [T_q] chunk-local radix index per query
    const int* query_offsets,     // [N+1] chunk-local node boundaries in T_q
    const int* radix_ids,         // [N] global radix_id per chunk position
    const int* token_ids,         // [T_q] token id per query (for intermediate target lookup)
    const int* counts_offset,     // [radix_count+1] global counts index
    const int* counts_tok,
    const int* counts_val,
    const float* mass_weights,    // [T_q] per-query mass weight, or NULL to disable
    // Fold-table arrays (optional; non-NULL fold_lengths enables fold path).
    // For caps with fold target, replaces the degenerate cap counts with the
    // composite next-char distribution P(c | W) where W is the cap's suffix
    // tail. Per-radix sparse top-K entries; probabilities renormalized to 1.
    const int* fold_offsets,      // [radix_count] start index into fold_tokens/probs (NULL if no fold)
    const int* fold_lengths,      // [radix_count] entries per radix_id (0 = no fold target)
    const int* fold_tokens,       // [total_fold_entries] flat token indices
    const float* fold_probs,      // [total_fold_entries] flat probabilities (sum to 1 per cap)
    // Virtual-tree side-table (optional; non-NULL vtree_lengths enables it).
    // Per (cap, tunnel-position) composite distribution that replaces the
    // one-hot intermediate target at tunnel positions p ∈ [0, expansion_depth).
    // Slot index = radix_id * vtree_expansion_depth + position_in_edge.
    const int* vtree_offsets,     // [radix_count * expansion_depth] offset into vtree_tokens/probs
    const int* vtree_lengths,     // [radix_count * expansion_depth] entries per slot (0 = no override)
    const int* vtree_tokens,      // flat token indices
    const float* vtree_probs,     // flat probabilities (sum to 1 per slot)
    int vtree_expansion_depth,    // 0 if no virtual tree
    float* d_logits,              // [T_q, V] — written with gradient
    float* loss_out,              // [T_q]
    int T_q, int V,
    float entropy_lambda,         // 0 disables icing
    float intermediate_weight,    // scale applied at unary-intermediate positions (endpoints unchanged)
    int    ce_only)               // 1 = force single-target CE at endpoints too (SGD semantic); 0 = KL default
{
    int q = blockIdx.x;
    if (q >= T_q) return;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    const float* in_row = logits + q * V;
    float* grad_row = d_logits + q * V;

    extern __shared__ float sdata[];

    // Softmax (max, exp, sum, normalize) — same as agpt_loss_kernel
    float local_max = -FLT_MAX;
    for (int j = tid; j < V; j += nthreads) if (in_row[j] > local_max) local_max = in_row[j];
    sdata[tid] = local_max; __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s && sdata[tid + s] > sdata[tid]) sdata[tid] = sdata[tid + s];
        __syncthreads();
    }
    float max_val = sdata[0];

    float local_sum = 0.0f;
    for (int j = tid; j < V; j += nthreads) {
        float e = expf(in_row[j] - max_val);
        grad_row[j] = e;
        local_sum += e;
    }
    sdata[tid] = local_sum; __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / sdata[0];
    for (int j = tid; j < V; j += nthreads) grad_row[j] *= inv_sum;
    __syncthreads();

    if (tid == 0) {
        int n_idx = query_to_node[q];
        int node_end_q = query_offsets[n_idx + 1];
        bool is_endpoint = (q + 1) == node_end_q;

        // ce_only=1 + endpoint-has-next-token: route through single-target CE
        // (same code path as intermediate). Target = token_ids[q+1] which, for a
        // non-final radix endpoint, is the first character of the next radix
        // node's edge — i.e. the character we sampled as this branch's
        // continuation. This is SGD-semantic. For the FINAL endpoint in the
        // chunk (no q+1), fall through to the KL path below.
        bool ce_as_intermediate = (ce_only && is_endpoint && (q + 1 < T_q));

        if (is_endpoint && !ce_as_intermediate) {
            // Endpoint: use stored counts (may be branching). AGPT KL semantic.
            int radix_id = radix_ids[n_idx];

            // Fold-table override: if this radix has a fold target, use it
            // instead of the (degenerate) cap counts. Fold probabilities are
            // pre-normalized to sum to 1.0 per cap, so the loss/gradient
            // formulas match the existing counts-based path with cnt/total_f
            // replaced by prob.
            int fold_len = (fold_lengths != NULL) ? fold_lengths[radix_id] : 0;
            if (fold_len > 0) {
                int fold_off = fold_offsets[radix_id];
                float weight = 1.0f;
                if (entropy_lambda > 0.0f && fold_len > 1) {
                    float H = 0.0f;
                    for (int e = 0; e < fold_len; e++) {
                        float prob = fold_probs[fold_off + e];
                        if (prob > 0.0f) H -= prob * logf(prob);
                    }
                    weight = 1.0f + entropy_lambda * (H / logf((float)V));
                }
                if (mass_weights != NULL) weight *= mass_weights[q];
                float loss = 0.0f;
                for (int e = 0; e < fold_len; e++) {
                    int tok = fold_tokens[fold_off + e];
                    float prob = fold_probs[fold_off + e];
                    float p = grad_row[tok];
                    loss -= prob * logf(p + 1e-10f);
                    grad_row[tok] -= prob;
                }
                if (weight != 1.0f) {
                    loss *= weight;
                    for (int j = 0; j < V; j++) grad_row[j] *= weight;
                }
                loss_out[q] = loss;
                return;
            }

            int start = counts_offset[radix_id];
            int end = counts_offset[radix_id + 1];
            if (start == end) {
                loss_out[q] = 0.0f;
                return;
            }
            int total = 0;
            for (int e = start; e < end; e++) total += counts_val[e];
            float total_f = (float)total;

            // Entropy weighting ("icing"): w = 1 + λ · H/log(V).
            // H = 0 for deterministic (single-entry) distributions, so single-branch
            // endpoints get weight 1 (unchanged).
            float weight = 1.0f;
            if (entropy_lambda > 0.0f && (end - start) > 1) {
                float H = 0.0f;
                for (int e = start; e < end; e++) {
                    float q_e = counts_val[e] / total_f;
                    if (q_e > 0.0f) H -= q_e * logf(q_e);
                }
                float log_V = logf((float)V);
                weight = 1.0f + entropy_lambda * (H / log_V);
            }

            // Combine entropy icing with corpus-mass weighting (both multiplicative).
            if (mass_weights != NULL) weight *= mass_weights[q];

            float loss = 0.0f;
            for (int e = start; e < end; e++) {
                int tok = counts_tok[e];
                int cnt = counts_val[e];
                float p = grad_row[tok];
                loss -= (cnt / total_f) * logf(p + 1e-10f);
                grad_row[tok] -= cnt / total_f;
            }
            if (weight != 1.0f) {
                loss *= weight;
                // Scale entire gradient row by weight (both softmax part and target sub)
                for (int j = 0; j < V; j++) grad_row[j] *= weight;
            }
            loss_out[q] = loss;
        } else {
            // Intermediate unary: target is the next token in the edge.
            // Virtual-tree override: at tunnel positions inside a cap, replace
            // the one-hot with a corpus-aggregated composite distribution from
            // shifted-prefix walks. The cap's L tunnel positions are otherwise
            // L deterministic one-hots that the model can't reach (cap-as-SGD
            // pathology). VTRE substitutes a real distribution at the first
            // expansion_depth positions where shifted walks still find
            // non-degenerate evidence; remaining tunnel positions stay one-hot.
            if (vtree_lengths != NULL && vtree_expansion_depth > 0) {
                int radix_id = radix_ids[n_idx];
                int counts_start = counts_offset[radix_id];
                int counts_end = counts_offset[radix_id + 1];
                bool is_cap = (counts_end - counts_start) == 1;
                if (is_cap) {
                    int node_start_q = query_offsets[n_idx];
                    int pos_in_edge = q - node_start_q;
                    if (pos_in_edge < vtree_expansion_depth) {
                        int slot = radix_id * vtree_expansion_depth + pos_in_edge;
                        int vlen = vtree_lengths[slot];
                        if (vlen > 0) {
                            int voff = vtree_offsets[slot];
                            float w = intermediate_weight;
                            if (entropy_lambda > 0.0f && vlen > 1) {
                                float H = 0.0f;
                                for (int e = 0; e < vlen; e++) {
                                    float prob = vtree_probs[voff + e];
                                    if (prob > 0.0f) H -= prob * logf(prob);
                                }
                                w *= 1.0f + entropy_lambda * (H / logf((float)V));
                            }
                            if (mass_weights != NULL) w *= mass_weights[q];
                            float loss = 0.0f;
                            for (int e = 0; e < vlen; e++) {
                                int tok = vtree_tokens[voff + e];
                                float prob = vtree_probs[voff + e];
                                float p = grad_row[tok];
                                loss -= prob * logf(p + 1e-10f);
                                grad_row[tok] -= prob;
                            }
                            if (w != 1.0f) {
                                loss *= w;
                                for (int j = 0; j < V; j++) grad_row[j] *= w;
                            }
                            loss_out[q] = loss;
                            return;
                        }
                    }
                }
            }
            int target = token_ids[q + 1];
            float p = grad_row[target];
            float loss = -logf(p + 1e-10f);
            grad_row[target] -= 1.0f;
            // Combined weight: mass-weighting × intermediate-weight scalar. The
            // intermediate-weight knob lets callers reduce the pull of
            // deterministic unary-chain predictions (which can cause "run-on
            // word" generation artifacts) without affecting endpoint branching.
            float w = intermediate_weight;
            if (mass_weights != NULL) w *= mass_weights[q];
            if (w != 1.0f) {
                loss *= w;
                for (int j = 0; j < V; j++) grad_row[j] *= w;
            }
            loss_out[q] = loss;
        }
    }
}

void launch_agpt_loss_per_query(const float* logits, const int* query_to_node,
                                 const int* query_offsets, const int* radix_ids,
                                 const int* token_ids,
                                 const int* counts_offset, const int* counts_tok,
                                 const int* counts_val,
                                 const float* mass_weights,
                                 const int* fold_offsets, const int* fold_lengths,
                                 const int* fold_tokens, const float* fold_probs,
                                 const int* vtree_offsets, const int* vtree_lengths,
                                 const int* vtree_tokens, const float* vtree_probs,
                                 int vtree_expansion_depth,
                                 float* d_logits, float* loss_out,
                                 int T_q, int V, float entropy_lambda,
                                 float intermediate_weight, int ce_only) {
    int threads = (V < 256) ? V : 256;
    int t = 1; while (t < threads) t <<= 1;
    threads = (t < 32) ? 32 : t;
    int smem = threads * sizeof(float);
    agpt_loss_per_query_kernel<<<T_q, threads, smem>>>(
        logits, query_to_node, query_offsets, radix_ids, token_ids,
        counts_offset, counts_tok, counts_val, mass_weights,
        fold_offsets, fold_lengths, fold_tokens, fold_probs,
        vtree_offsets, vtree_lengths, vtree_tokens, vtree_probs,
        vtree_expansion_depth,
        d_logits, loss_out, T_q, V, entropy_lambda, intermediate_weight, ce_only);
}

// --- Element-wise add: a += b ---
__global__ void elem_add_kernel(float* a, const float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) a[idx] += b[idx];
}

void launch_elem_add(float* a, const float* b, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    elem_add_kernel<<<blocks, threads>>>(a, b, n);
}

// --- Zero buffer ---
__global__ void zero_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = 0.0f;
}

void launch_zero(float* data, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    zero_kernel<<<blocks, threads>>>(data, n);
}

// ============================================================================
// CPU-side KV scatter/gather (for host-memory KV cache)
// ============================================================================

// Scatter K/V from GPU buffer to host KV cache:
// Download src[N, D] from GPU, then for each i: host_kv[node_ids[i] * D .. +D] = src[i]
void host_kv_scatter(const float* d_src, const int* h_node_ids,
                      float* h_kv, int N, int D) {
    float* h_src = (float*)malloc((long long)N * D * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_src, d_src, (long long)N * D * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; i++) {
        int nid = h_node_ids[i];
        memcpy(&h_kv[(long long)nid * D], &h_src[(long long)i * D], D * sizeof(float));
    }
    free(h_src);
}

// Gather ancestor K/V from host into packed GPU buffer:
// For each node i, gather ancestors' K/V from host, pack, upload to GPU.
void host_kv_gather(const float* h_kv, const int* h_ancestor_ids,
                     const int* h_ancestor_offsets, // per-node (chunk-local) offset into ancestor_ids
                     const int* h_kv_offsets,       // per-node offset into packed output
                     const int* h_kv_lengths,       // per-node prefix length
                     float* d_packed_kv,            // GPU output
                     int N, int n_heads, int head_dim,
                     int total_kv_positions) {
    int D = n_heads * head_dim;
    int HD = head_dim;
    int H = n_heads;
    // Allocate CPU packed buffer
    long long packed_size = (long long)total_kv_positions * H * HD;
    float* h_packed = (float*)calloc(packed_size, sizeof(float));

    for (int i = 0; i < N; i++) {
        int anc_off = h_ancestor_offsets[i];
        int kv_off = h_kv_offsets[i];
        int len = h_kv_lengths[i];
        for (int p = 0; p < len; p++) {
            int ancestor = h_ancestor_ids[anc_off + p];
            for (int col = 0; col < D; col++) {
                int head = col / HD;
                int hcol = col % HD;
                float val = h_kv[(long long)ancestor * D + col];
                h_packed[((long long)(kv_off + p) * H + head) * HD + hcol] = val;
            }
        }
    }

    CUDA_CHECK(cudaMemcpy(d_packed_kv, h_packed, packed_size * sizeof(float), cudaMemcpyHostToDevice));
    free(h_packed);
}

// ============================================================================
// RoPE cache (precomputed on CPU, uploaded to GPU)
// ============================================================================

void build_rope_cache(float** d_cos, float** d_sin, int max_seq, int dim, float base = 10000.0f) {
    int half = dim / 2;
    float* h_cos = (float*)malloc(max_seq * dim * sizeof(float));
    float* h_sin = (float*)malloc(max_seq * dim * sizeof(float));

    for (int pos = 0; pos < max_seq; pos++) {
        for (int i = 0; i < half; i++) {
            float theta = pos / powf(base, 2.0f * i / dim);
            float c = cosf(theta);
            float s = sinf(theta);
            h_cos[pos * dim + 2 * i]     = c;
            h_cos[pos * dim + 2 * i + 1] = c;
            h_sin[pos * dim + 2 * i]     = s;
            h_sin[pos * dim + 2 * i + 1] = s;
        }
    }

    CUDA_CHECK(cudaMalloc(d_cos, max_seq * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(d_sin, max_seq * dim * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(*d_cos, h_cos, max_seq * dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(*d_sin, h_sin, max_seq * dim * sizeof(float), cudaMemcpyHostToDevice));
    free(h_cos);
    free(h_sin);
}

// Leveled (legacy) trainer code path. See file for rationale.
// Preserved for future revisit.
#include "agpt_train_leveled.cuh"


// ============================================================================
// Radix training
// ============================================================================
//
// Processes radix nodes chunked by endpoint depth. For each radix node with
// edge of length L_i:
//   - embed L_i tokens
//   - forward all L_i positions through the model (LN/QKV/attn/WO/LN/FFN)
//   - L-query varlen attention: each of L_i queries attends to ancestors +
//     edge positions 0..j
//   - final LN + output proj applied only at endpoint (last edge position)
//   - loss only at endpoint
//   - backward: gradient enters only at endpoint, propagates through forward
//
// KV cache is per-character-position (total_edge_chars). Character position
// of radix node r's edge[j] = edge_starts[r] + j.

// Forward declarations
extern "C" void cuda_batched_varlen_attention_L_queries(
    const float* q_packed, const float* k_packed, const float* v_packed,
    const int* query_to_node, const int* query_offsets,
    const int* kv_offsets, const int* kv_lengths,
    float* output, float* weights_out,
    int T_q, int n_heads, int head_dim, int max_kv_len, float scale);
extern "C" void cuda_batched_varlen_attention_L_queries_backward(
    const float* q_packed, const float* k_packed, const float* v_packed,
    const float* attn_weights, const float* d_out,
    const int* query_to_node, const int* query_offsets,
    const int* kv_offsets, const int* kv_lengths,
    float* dq, float* dk_full, float* dv_full,
    int T_q, int n_heads, int head_dim, int max_kv_len, float scale);

// Scatter per-radix-node endpoint logit gradient into per-query buffer.
// d_final_per_node [N, D] → d_x_per_query [T_q, D] with zeros at non-endpoints.
__global__ void scatter_endpoint_grad_kernel(
    const float* d_endpoint,  // [N, D]
    float* d_per_query,       // [T_q, D]
    const int* query_offsets, // [N+1]
    int N, int D)
{
    int n = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (n >= N || d >= D) return;
    int end_q = query_offsets[n + 1] - 1;  // last query of node n = endpoint
    d_per_query[end_q * D + d] = d_endpoint[n * D + d];
}

// Gather endpoint positions' activations: for each radix node, pick out the
// last query's row. src [T_q, D], dst [N, D].
__global__ void gather_endpoint_rows_kernel(
    const float* src,         // [T_q, D]
    float* dst,               // [N, D]
    const int* query_offsets, // [N+1]
    int N, int D)
{
    int n = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (n >= N || d >= D) return;
    int end_q = query_offsets[n + 1] - 1;
    dst[n * D + d] = src[end_q * D + d];
}

// File-scope fold-table device pointers. Set in main() when --fold-table is
// passed; remain NULL otherwise. The loss kernel reads g_d_fold_lengths !=
// NULL to gate the fold path. Single-process/single-threaded trainer, so a
// global here avoids threading the four pointers through all training entry
// points.
static const int*   g_d_fold_offsets = NULL;
static const int*   g_d_fold_lengths = NULL;
static const int*   g_d_fold_tokens  = NULL;
static const float* g_d_fold_probs   = NULL;

// File-scope virtual-tree side-table device pointers + expansion depth.
// Set by load_virtual_tree() when --virtual-tree is passed; remain NULL
// otherwise. Same single-process rationale as the fold-table globals above.
static const int*   g_d_vtree_offsets = NULL;
static const int*   g_d_vtree_lengths = NULL;
static const int*   g_d_vtree_tokens  = NULL;
static const float* g_d_vtree_probs   = NULL;
static int          g_vtree_expansion_depth = 0;

// Load a fold-table file produced by `bin/agpt_build_fold_table` and upload
// its arrays to GPU. Sets g_d_fold_* globals on success.
//
// File format:
//   magic (u32 = 'FOLD')
//   version (u32 = 1)
//   n_radix (u32) — must match expected_radix_count
//   vocab_size (u32) — must match expected_vocab
//   top_k (u32) — informational
//   offsets[n_radix] (i32)
//   lengths[n_radix] (i32)
//   entries[total]: i32 token + f32 prob
static void load_fold_table(const char* path, int expected_radix_count, int expected_vocab) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "fold-table: cannot open %s\n", path);
        exit(1);
    }
    uint32_t magic = 0, version = 0, n_radix = 0, vocab = 0, top_k = 0;
    if (fread(&magic, 4, 1, f) != 1 || magic != 0x444C4F46u) {
        fprintf(stderr, "fold-table: bad magic in %s (got 0x%08x)\n", path, magic);
        exit(1);
    }
    if (fread(&version, 4, 1, f) != 1 || version != 1) {
        fprintf(stderr, "fold-table: unsupported version %u (expected 1)\n", version);
        exit(1);
    }
    if (fread(&n_radix, 4, 1, f) != 1 || (int)n_radix != expected_radix_count) {
        fprintf(stderr, "fold-table: n_radix=%u does not match trie radix_count=%d\n",
                n_radix, expected_radix_count);
        exit(1);
    }
    if (fread(&vocab, 4, 1, f) != 1 || (int)vocab != expected_vocab) {
        fprintf(stderr, "fold-table: vocab_size=%u does not match cfg.vocab_size=%d\n",
                vocab, expected_vocab);
        exit(1);
    }
    if (fread(&top_k, 4, 1, f) != 1) { fprintf(stderr, "fold-table: short read at top_k\n"); exit(1); }

    int* h_offsets = (int*)malloc((size_t)n_radix * sizeof(int));
    int* h_lengths = (int*)malloc((size_t)n_radix * sizeof(int));
    if (fread(h_offsets, sizeof(int), n_radix, f) != n_radix) {
        fprintf(stderr, "fold-table: short read at offsets\n"); exit(1);
    }
    if (fread(h_lengths, sizeof(int), n_radix, f) != n_radix) {
        fprintf(stderr, "fold-table: short read at lengths\n"); exit(1);
    }

    long long total_entries = 0;
    int n_with_fold = 0;
    for (uint32_t i = 0; i < n_radix; i++) {
        total_entries += h_lengths[i];
        if (h_lengths[i] > 0) n_with_fold++;
    }

    int* h_tokens = (int*)malloc((size_t)total_entries * sizeof(int));
    float* h_probs = (float*)malloc((size_t)total_entries * sizeof(float));
    for (long long i = 0; i < total_entries; i++) {
        int tok = 0; float prob = 0.0f;
        if (fread(&tok, sizeof(int), 1, f) != 1 || fread(&prob, sizeof(float), 1, f) != 1) {
            fprintf(stderr, "fold-table: short read at entry %lld/%lld\n", i, total_entries);
            exit(1);
        }
        h_tokens[i] = tok;
        h_probs[i] = prob;
    }
    fclose(f);

    int *d_off, *d_len, *d_tok;
    float* d_prb;
    CUDA_CHECK(cudaMalloc(&d_off, (size_t)n_radix * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_len, (size_t)n_radix * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_tok, (size_t)total_entries * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_prb, (size_t)total_entries * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_off, h_offsets, (size_t)n_radix * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_len, h_lengths, (size_t)n_radix * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tok, h_tokens, (size_t)total_entries * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_prb, h_probs, (size_t)total_entries * sizeof(float), cudaMemcpyHostToDevice));
    free(h_offsets); free(h_lengths); free(h_tokens); free(h_probs);

    g_d_fold_offsets = d_off;
    g_d_fold_lengths = d_len;
    g_d_fold_tokens  = d_tok;
    g_d_fold_probs   = d_prb;

    printf("Loaded fold-table from %s: %u radix slots, %d with fold, %lld entries, top_k=%u\n",
           path, n_radix, n_with_fold, total_entries, top_k);
}

// Load a VTRE virtual-tree side-table produced by `bin/agpt_build_virtual_tree`
// and upload its arrays to GPU. Sets g_d_vtree_* + g_vtree_expansion_depth
// globals on success.
//
// File format ('VTRE' v1):
//   magic (u32 = 'VTRE')
//   version (u32 = 1)
//   n_radix (u32) — must match expected_radix_count
//   vocab_size (u32) — must match expected_vocab
//   top_k (u32) — informational
//   expansion_depth (u32)
//   offsets[n_radix * expansion_depth] (i32)
//   lengths[n_radix * expansion_depth] (i32)
//   entries[total]: i32 token + f32 prob
static void load_virtual_tree(const char* path, int expected_radix_count, int expected_vocab) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "virtual-tree: cannot open %s\n", path);
        exit(1);
    }
    uint32_t magic = 0, version = 0, n_radix = 0, vocab = 0, top_k = 0, expansion_depth = 0;
    if (fread(&magic, 4, 1, f) != 1 || magic != 0x45525456u) {
        fprintf(stderr, "virtual-tree: bad magic in %s (got 0x%08x, expected 'VTRE')\n", path, magic);
        exit(1);
    }
    if (fread(&version, 4, 1, f) != 1 || version != 1) {
        fprintf(stderr, "virtual-tree: unsupported version %u (expected 1)\n", version);
        exit(1);
    }
    if (fread(&n_radix, 4, 1, f) != 1 || (int)n_radix != expected_radix_count) {
        fprintf(stderr, "virtual-tree: n_radix=%u does not match trie radix_count=%d\n",
                n_radix, expected_radix_count);
        exit(1);
    }
    if (fread(&vocab, 4, 1, f) != 1 || (int)vocab != expected_vocab) {
        fprintf(stderr, "virtual-tree: vocab_size=%u does not match cfg.vocab_size=%d\n",
                vocab, expected_vocab);
        exit(1);
    }
    if (fread(&top_k, 4, 1, f) != 1) { fprintf(stderr, "virtual-tree: short read at top_k\n"); exit(1); }
    if (fread(&expansion_depth, 4, 1, f) != 1 || expansion_depth < 1) {
        fprintf(stderr, "virtual-tree: bad expansion_depth %u\n", expansion_depth);
        exit(1);
    }

    long long n_slots = (long long)n_radix * (long long)expansion_depth;
    int* h_offsets = (int*)malloc((size_t)n_slots * sizeof(int));
    int* h_lengths = (int*)malloc((size_t)n_slots * sizeof(int));
    if ((long long)fread(h_offsets, sizeof(int), (size_t)n_slots, f) != n_slots) {
        fprintf(stderr, "virtual-tree: short read at offsets\n"); exit(1);
    }
    if ((long long)fread(h_lengths, sizeof(int), (size_t)n_slots, f) != n_slots) {
        fprintf(stderr, "virtual-tree: short read at lengths\n"); exit(1);
    }

    long long total_entries = 0;
    long long n_filled = 0;
    for (long long i = 0; i < n_slots; i++) {
        total_entries += h_lengths[i];
        if (h_lengths[i] > 0) n_filled++;
    }

    int* h_tokens = (int*)malloc((size_t)total_entries * sizeof(int));
    float* h_probs = (float*)malloc((size_t)total_entries * sizeof(float));
    for (long long i = 0; i < total_entries; i++) {
        int tok = 0; float prob = 0.0f;
        if (fread(&tok, sizeof(int), 1, f) != 1 || fread(&prob, sizeof(float), 1, f) != 1) {
            fprintf(stderr, "virtual-tree: short read at entry %lld/%lld\n", i, total_entries);
            exit(1);
        }
        h_tokens[i] = tok;
        h_probs[i] = prob;
    }
    fclose(f);

    int *d_off, *d_len, *d_tok;
    float* d_prb;
    CUDA_CHECK(cudaMalloc(&d_off, (size_t)n_slots * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_len, (size_t)n_slots * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_tok, (size_t)total_entries * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_prb, (size_t)total_entries * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_off, h_offsets, (size_t)n_slots * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_len, h_lengths, (size_t)n_slots * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tok, h_tokens, (size_t)total_entries * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_prb, h_probs, (size_t)total_entries * sizeof(float), cudaMemcpyHostToDevice));
    free(h_offsets); free(h_lengths); free(h_tokens); free(h_probs);

    g_d_vtree_offsets = d_off;
    g_d_vtree_lengths = d_len;
    g_d_vtree_tokens  = d_tok;
    g_d_vtree_probs   = d_prb;
    g_vtree_expansion_depth = (int)expansion_depth;

    printf("Loaded virtual-tree from %s: %u radix slots × expansion=%u, %lld slots filled, %lld entries, top_k=%u\n",
           path, n_radix, expansion_depth, n_filled, total_entries, top_k);
}

// run_radix_training optional parameters (declared here via overload-less defaults).
// When invoked from the per-subtree wrapper, these thread optimizer state across
// calls so RMSProp/Adam running averages don't reset per subtree, and suppress
// the usual startup banner for a clean repeated-call log.
struct TrainPersistence {
    float* h_adam_m_io = nullptr;  // if non-null: load in on entry, copy out on exit
    float* h_adam_v_io = nullptr;
    int*   adam_t_io   = nullptr;  // read on entry as starting step, write on exit
    bool   quiet       = false;    // suppress banner + per-epoch lines
    int    total_opt_steps_override = 0;  // for LR schedule when the caller knows the true horizon (steps)
    int    warmup_steps_override    = 0;  // caller-known warmup length (0 = derive from warmup_epochs)
    int    total_epochs_override    = 0;  // caller-known SE-budget for LR schedule (computed → total_opt_steps); 0 = derive from epochs arg
};

#include "agpt_chunk_metadata.cuh"
#include "agpt_chunk_upload_runtime.cuh"
#include "agpt_cache_runtime.cuh"
#include "agpt_transformer_chunk_runtime.cuh"

// ============================================================================
// Experimental flags
// ============================================================================
//
// Environment-variable-driven knobs for active experiments. Consolidates the
// scattered getenv() calls into one place so `grep ExperimentalFlags` shows
// every experimental control surface at a glance.
//
// Promote to CLI flags when an experiment graduates from "is this worth
// keeping?" to "yes, this is a documented option."
//
struct ExperimentalFlags {
    // AGPT_TIMING_BREAKDOWN: per-kernel cudaEvent timing breakdown each epoch.
    // ~5-10% overhead. Diagnostic only.
    bool timing_breakdown = false;

    // AGPT_DEPTH_ROUTE_K: depth threshold for K/V gradient routing.
    // Shallow queries (d ≤ d_k) feed dWk only; deep queries feed dWv only.
    // 0 disables (default). Mutually exclusive with depth_route_perleaf.
    int depth_route_k = 0;

    // AGPT_DEPTH_ROUTE_PERLEAF: per-leaf d* routing — each query's threshold
    // is its node's d_split (depth at which path becomes mass=1) instead of
    // the static depth_route_k. Takes precedence over depth_route_k when set.
    int depth_route_perleaf = 0;

    // AGPT_DECISION_ONLY: skip loss + gradient at queries where depth > d_split.
    // Trains only on decision events; deterministic-tail events contribute
    // neither loss nor gradient.
    int decision_only = 0;

    // AGPT_DECISION_BUFFER: chars past d_split to keep before zeroing.
    // 0 = strict decision-only; larger = less aggressive cut. Relevant when
    // decision_only=1.
    int decision_buffer = 0;

    // AGPT_JOINT_MASS: multiply mass weight by complementary-depth suffix
    // factor. Two implementations:
    //   1. Aggregate proxy (default): suffix_factor = mean_edge_mass[D_max - d_q]
    //   2. Per-position table (when char_suffix_mass_path is set):
    //      suffix_factor = char_suffix_mass[char_pos]
    // Effective only when --mass-weight is set to log/sqrt/linear.
    int joint_mass = 0;

    // AGPT_SUBTREE_DROPOUT: per super-epoch, randomly drop each rc with prob p.
    // Clamped to [0, 0.99]. 0 disables (default).
    double subtree_dropout = 0.0;
    // AGPT_SUBTREE_DROPOUT_SEED: RNG seed for the dropout sampling.
    unsigned int subtree_dropout_seed = 0xA9F71E13u;

    // AGPT_BRANCH_DROPOUT_DEPTH: when > 0, dropout sampling happens at radix
    // nodes whose first-char-depth equals this value (default depth=1 root-children).
    // E.g., depth=3 masks bigram/trigram subtrees → many more candidate masks per epoch.
    int branch_dropout_depth = 0;

    // AGPT_CHAR_SUFFIX_MASS_PATH: path to binary table of per-char-position
    // suffix mass values (used by joint_mass mode 2). Loaded at function entry
    // if joint_mass > 0 and path is set. Buffer owned by caller; freed at exit.
    const char* char_suffix_mass_path = nullptr;
};

static ExperimentalFlags read_experimental_flags() {
    ExperimentalFlags f;
    auto envi = [](const char* name, int dflt) -> int {
        const char* s = getenv(name);
        return s ? atoi(s) : dflt;
    };
    f.timing_breakdown    = (getenv("AGPT_TIMING_BREAKDOWN") != nullptr);
    f.depth_route_k       = envi("AGPT_DEPTH_ROUTE_K", 0);
    f.depth_route_perleaf = envi("AGPT_DEPTH_ROUTE_PERLEAF", 0);
    f.decision_only       = envi("AGPT_DECISION_ONLY", 0);
    f.decision_buffer     = envi("AGPT_DECISION_BUFFER", 0);
    f.joint_mass          = envi("AGPT_JOINT_MASS", 0);
    f.branch_dropout_depth = envi("AGPT_BRANCH_DROPOUT_DEPTH", 0);
    if (f.branch_dropout_depth < 0) f.branch_dropout_depth = 0;
    {
        const char* s = getenv("AGPT_SUBTREE_DROPOUT");
        if (s) f.subtree_dropout = atof(s);
        if (f.subtree_dropout < 0.0)  f.subtree_dropout = 0.0;
        if (f.subtree_dropout > 0.99) f.subtree_dropout = 0.99;
    }
    {
        const char* s = getenv("AGPT_SUBTREE_DROPOUT_SEED");
        if (s) f.subtree_dropout_seed = (unsigned int)atoll(s);
    }
    f.char_suffix_mass_path = getenv("AGPT_CHAR_SUFFIX_MASS_PATH");
    return f;
}

int run_radix_training(const Config& cfg, const WeightOffsets& wo,
                        float* h_weights, RadixTrieData& trie,
                        int epochs, float entropy_lambda, MassWeightMode mass_weight,
                        int subtree_splits, int partition_depth, bool accumulate,
                        bool single_subtree, float intermediate_weight,
                        OptimizerKind optimizer, float momentum_beta, float rmsprop_beta,
                        LRSchedule lr_schedule, int warmup_epochs,
                        float weight_decay, float grad_clip_norm, int save_every,
                        CurriculumMode curriculum, const char* save_path,
                        LightningConfig lightning = LightningConfig{},
                        TrainPersistence* persist = nullptr)
{
    const bool quiet = persist && persist->quiet;

    // ---- Experimental flags (env-var-driven; see ExperimentalFlags struct above) ----
    ExperimentalFlags flags = read_experimental_flags();
    // Local aliases preserve existing names throughout the function body.
    bool         t_enabled            = flags.timing_breakdown;
    int          depth_route_k        = flags.depth_route_k;
    int          depth_route_perleaf  = flags.depth_route_perleaf;
    int          decision_only        = flags.decision_only;
    int          decision_buffer      = flags.decision_buffer;
    int          joint_mass           = flags.joint_mass;
    double       subtree_dropout      = flags.subtree_dropout;
    unsigned int subtree_dropout_seed = flags.subtree_dropout_seed;
    int          branch_dropout_depth = flags.branch_dropout_depth;

    cudaEvent_t te_start = NULL, te_stop = NULL;
    if (t_enabled) {
        cudaEventCreate(&te_start);
        cudaEventCreate(&te_stop);
    }
    double t_us_gather_fwd = 0, t_us_gather_bwd = 0;
    double t_us_attn_fwd = 0,   t_us_attn_bwd   = 0;
    double t_us_scatter_fwd = 0;

    // char_suffix_mass: special — loads a binary table from disk (not a flag value).
    // Only loaded if joint_mass mode 2 is active.
    double* char_suffix_mass = NULL;
    long long char_suffix_mass_n = 0;
    if (flags.char_suffix_mass_path && joint_mass > 0) {
        const char* path = flags.char_suffix_mass_path;
        FILE* f = fopen(path, "rb");
        if (!f) {
            fprintf(stderr, "AGPT_CHAR_SUFFIX_MASS_PATH=%s: cannot open\n", path);
            exit(1);
        }
        long long n;
        if (fread(&n, 8, 1, f) != 1) {
            fprintf(stderr, "AGPT_CHAR_SUFFIX_MASS_PATH=%s: header read failed\n", path);
            exit(1);
        }
        char_suffix_mass = (double*)malloc(n * sizeof(double));
        if (fread(char_suffix_mass, sizeof(double), n, f) != (size_t)n) {
            fprintf(stderr, "AGPT_CHAR_SUFFIX_MASS_PATH=%s: data read failed (expected %lld doubles)\n", path, n);
            exit(1);
        }
        fclose(f);
        char_suffix_mass_n = n;
    }
    if (!quiet) {
        if (decision_only > 0) {
            printf("  decision-only: yes  (loss + gradient zeroed at depth > d_split + %d)\n",
                   decision_buffer);
        }
        if (depth_route_perleaf > 0) {
            printf("  depth-route: per-leaf d* (Wk grad ← d≤d*[node], Wv grad ← d>d*[node])\n");
        } else if (depth_route_k > 0) {
            printf("  depth-route k: %d (Wk grad ← d≤%d, Wv grad ← d>%d)\n",
                   depth_route_k, depth_route_k, depth_route_k);
        }
        if (joint_mass > 0) {
            if (char_suffix_mass != NULL) {
                printf("  joint-mass: per-position  (table loaded: %lld char offsets)\n",
                       char_suffix_mass_n);
            } else {
                printf("  joint-mass: aggregate proxy  (per-query weight = compress(edge_mass × mean_edge_mass[D_max - d_q]))\n");
            }
        }
        if (subtree_dropout > 0.0) {
            if (branch_dropout_depth > 0) {
                printf("  branch-dropout: %.2f at depth %d  (subtrees rooted at depth-%d nodes skipped with prob %.2f)\n",
                       subtree_dropout, branch_dropout_depth, branch_dropout_depth, subtree_dropout);
            } else {
                printf("  subtree-dropout: %.2f  (per super-epoch, each root-child skipped with prob %.2f)\n",
                       subtree_dropout, subtree_dropout);
            }
        }
    }
    #define TIME_K(accum, code) do { \
        if (t_enabled) cudaEventRecord(te_start); \
        code; \
        if (t_enabled) { \
            cudaEventRecord(te_stop); \
            cudaEventSynchronize(te_stop); \
            float __ms = 0; \
            cudaEventElapsedTime(&__ms, te_start, te_stop); \
            accum += __ms * 1000.0; \
        } \
    } while(0)
    if (!quiet) {
    const char* sched_name = (lr_schedule == LRSchedule::Constant)    ? "constant"
                           : (lr_schedule == LRSchedule::Cosine)       ? "cosine"
                           :                                             "warmup-cosine";
    if (lr_schedule != LRSchedule::Constant) {
        printf("  lr-schedule: %s (peak=%.4g, warmup_epochs=%d)\n", sched_name, cfg.lr, warmup_epochs);
    }
    if (weight_decay > 0.0f) {
        printf("  weight-decay: %.4g (decoupled, AdamW-style)\n", weight_decay);
    }
    if (grad_clip_norm > 0.0f) {
        printf("  grad-clip-norm: %.4g\n", grad_clip_norm);
    }
    if (save_every > 0 && save_path) {
        printf("  save-every: %d epochs (checkpoints as <save_path>.epN)\n", save_every);
    }
    const char* opt_name = (optimizer == OptimizerKind::Adam)     ? "adam"
                         : (optimizer == OptimizerKind::SGD)      ? "sgd"
                         : (optimizer == OptimizerKind::Momentum) ? "momentum"
                         : (optimizer == OptimizerKind::LBFGS)    ? "lbfgs"
                         :                                          "rmsprop";
    printf("  optimizer: %s (lr=%.4g)\n", opt_name, cfg.lr);
    if (entropy_lambda > 0.0f) {
        printf("  entropy lambda: %.3f (branching-endpoint icing enabled)\n", entropy_lambda);
    }
    if (mass_weight != MassWeightMode::Off) {
        const char* mode_name = (mass_weight == MassWeightMode::Log)    ? "log"
                              : (mass_weight == MassWeightMode::Sqrt)   ? "sqrt"
                              : (mass_weight == MassWeightMode::Linear) ? "linear"
                              :                                            "?";
        printf("  mass weighting: %s (head-of-edge count restores corpus-frequency exposure)\n", mode_name);
    }
    if (subtree_splits > 1) {
        printf("  subtree splits: %d (N sub-batches per subtree; trades some within-subtree consistency for more updates)\n", subtree_splits);
    }
    if (single_subtree) {
        printf("  single-subtree: all radix nodes form ONE subtree → 1 Adam step per subtree pass\n");
    }
    if (lightning.steps > 0) {
        const char* sname = (lightning.sampler == LightningSampler::L1_Uniform) ? "l1-uniform"
                          : (lightning.sampler == LightningSampler::L2_RcDepth) ? "l2-rc-depth"
                          : (lightning.sampler == LightningSampler::L4_Path)    ? "l4-path (SGD-equivalent)"
                          :                                                        "l3-mass-walk";
        const char* mlr = (lightning.mass_lr == MassWeightMode::Log)    ? ", mass-lr=log"
                        : (lightning.mass_lr == MassWeightMode::Sqrt)   ? ", mass-lr=sqrt"
                        : (lightning.mass_lr == MassWeightMode::Linear) ? ", mass-lr=linear"
                        :                                                  "";
        printf("  lightning: %s, %d samples/super-epoch, p_stop=%.2f, seed=0x%x%s\n",
               sname, lightning.steps, lightning.p_stop, lightning.seed, mlr);
        if (lightning.max_mass > 0) {
            printf("  lightning: max-mass=%lld (force-descend when sampled subtree exceeds this)\n",
                   lightning.max_mass);
        }
        printf("           (stochastic per-sample optimizer steps; accumulate forced off)\n");
    }
    if (intermediate_weight != 1.0f) {
        printf("  intermediate-weight: %.3f (scale loss at unary-intermediate positions)\n", intermediate_weight);
    }
    printf("  curriculum: %s\n", curriculum == CurriculumMode::Progressive ? "progressive (d=1..d=max per epoch)" : "flat (d=max each epoch)");
    }
    int D = cfg.d_model;
    int F = cfg.d_ff;
    int V = cfg.vocab_size;
    int L_layers = cfg.n_layers;
    int H = cfg.n_heads;
    int HD = cfg.head_dim;

    // Chunk by total queries per chunk (character positions processed together).
    // Chunk size is a GPU-memory partition only: gradients ACCUMULATE across chunks
    // (no Adam step between). Smaller = less working-buffer memory; larger = fewer
    // host→device metadata uploads and better matmul shapes. Has no effect on
    // gradient semantics (contrast with --subtree-splits, which DOES fire Adam
    // steps at sub-batch boundaries).
    const int CHUNK_QUERIES = cfg.chunk_queries > 0 ? cfg.chunk_queries : 50000;
    // Find max endpoint depth and max edge len
    int max_edge_len = 0;
    int max_endpoint_depth = 0;
    for (int r = 0; r < trie.radix_count; r++) {
        if (trie.edge_lens[r] > max_edge_len) max_edge_len = trie.edge_lens[r];
        int ep = trie.edge_first_char_depths[r] + trie.edge_lens[r] - 1;
        if (ep > max_endpoint_depth) max_endpoint_depth = ep;
    }
    int max_ancestor_chars = 0;
    for (int r = 0; r < trie.radix_count; r++) {
        int a = trie.ancestor_char_offsets[r + 1] - trie.ancestor_char_offsets[r];
        if (a > max_ancestor_chars) max_ancestor_chars = a;
    }
    int max_kv_per_node = max_ancestor_chars + max_edge_len;
    if (!quiet) printf("  max edge_len: %d, max ancestor chars: %d, max KV per node: %d, max endpoint depth: %d\n",
           max_edge_len, max_ancestor_chars, max_kv_per_node, max_endpoint_depth);

    // Total working buffer sizes
    int T_q_cap = CHUNK_QUERIES;
    long long T_kv_cap = (long long)CHUNK_QUERIES * 4;  // generous: avg ~4x ancestors per query
    // We'll actually allocate T_kv_cap based on max observed per chunk, but
    // can't know that without iterating. Use a safe upper bound.

    // ------------------------------------------------------------
    // Allocate GPU state
    // ------------------------------------------------------------
    float *d_weights, *d_grads, *d_adam_m, *d_adam_v;
    CUDA_CHECK(cudaMalloc(&d_weights, wo.total_floats * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grads,   wo.total_floats * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_adam_m,  wo.total_floats * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_adam_v,  wo.total_floats * sizeof(float)));
    if (persist && persist->h_adam_m_io) {
        CUDA_CHECK(cudaMemcpy(d_adam_m, persist->h_adam_m_io, wo.total_floats * sizeof(float), cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMemset(d_adam_m, 0, wo.total_floats * sizeof(float)));
    }
    if (persist && persist->h_adam_v_io) {
        CUDA_CHECK(cudaMemcpy(d_adam_v, persist->h_adam_v_io, wo.total_floats * sizeof(float), cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMemset(d_adam_v, 0, wo.total_floats * sizeof(float)));
    }
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights, wo.total_floats * sizeof(float), cudaMemcpyHostToDevice));

    // ---- L-BFGS optimizer state (allocated only if --optimizer lbfgs) ----
    LBFGSState lbfgs_state;
    if (optimizer == OptimizerKind::LBFGS) {
        lbfgs_state.K = cfg.lbfgs_k > 0 ? cfg.lbfgs_k : 10;
        lbfgs_state.n = wo.total_floats;
        lbfgs_state.first_step = true;
        lbfgs_state.pushed_count = 0;
        size_t hist_bytes = (size_t)lbfgs_state.K * (size_t)wo.total_floats * sizeof(float);
        CUDA_CHECK(cudaMalloc(&lbfgs_state.d_g_prev, wo.total_floats * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&lbfgs_state.d_step,   wo.total_floats * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&lbfgs_state.d_s_hist, hist_bytes));
        CUDA_CHECK(cudaMalloc(&lbfgs_state.d_y_hist, hist_bytes));
        CUDA_CHECK(cudaMalloc(&lbfgs_state.d_q,      wo.total_floats * sizeof(float)));
        lbfgs_state.rho_hist = (float*)calloc(lbfgs_state.K, sizeof(float));
        lbfgs_state.alpha    = (float*)calloc(lbfgs_state.K, sizeof(float));
        if (!quiet) {
            double mb = (2.0 * hist_bytes + 3.0 * wo.total_floats * sizeof(float)) / (1024.0 * 1024.0);
            printf("  lbfgs state: K=%d, n=%d, %.2f MB\n", lbfgs_state.K, wo.total_floats, mb);
        }
    }

    // Scratch for gradient clipping: one partial per block + one float for norm.
    float* d_clip_partials = NULL;
    float* d_clip_norm = NULL;
    if (grad_clip_norm > 0.0f) {
        int threads = 256;
        int blocks = (wo.total_floats + threads - 1) / threads;
        CUDA_CHECK(cudaMalloc(&d_clip_partials, blocks * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_clip_norm, sizeof(float)));
    }

    // KV cache (unified memory, per character position). Stored as bf16 — half
    // the memory of fp32. Packed buffers used for attention remain fp32, with
    // conversion done on the scatter/gather kernels. BF16 mantissa loss (8 bits)
    // is fine for stored K/V since attention softmax already flattens small
    // differences; verified empirically at d=8 within GPU reduction noise.
    //
    // Mass=1 compaction: radix nodes with edge_mass == 1 are leaves (a single
    // corpus position reached there); no other query ever attends to their K/V
    // as an ancestor. We skip caching their char positions entirely. At
    // attention time, current-query own-edge K/V is read directly from the
    // fresh d_k/d_v forward buffer (no round-trip through cache), so the
    // compact cache only needs to hold mass>1 char positions.
    //
    // Build compact_slot[char_pos] -> compact index, or -1 for mass=1.
    int* compact_slot = (int*)malloc((long long)trie.total_edge_chars * sizeof(int));
    long long n_compact_chars = 0;
    long long n_mass1_chars   = 0;
    for (int r = 0; r < trie.radix_count; r++) {
        int start = trie.edge_starts[r];
        int len   = trie.edge_lens[r];
        bool is_mass1 = (trie.edge_mass[r] == 1);
        for (int j = 0; j < len; j++) {
            int cp = start + j;
            if (is_mass1) {
                compact_slot[cp] = -1;
                n_mass1_chars++;
            } else {
                compact_slot[cp] = (int)n_compact_chars++;
            }
        }
    }
    if (!quiet) {
        double skip_pct = (trie.total_edge_chars > 0)
            ? 100.0 * (double)n_mass1_chars / (double)trie.total_edge_chars : 0.0;
        printf("  mass=1 compaction: %lld / %lld chars are mass=1 (%.1f%%); cache holds %lld positions\n",
               n_mass1_chars, trie.total_edge_chars, skip_pct, n_compact_chars);
    }
    if (n_compact_chars == 0) n_compact_chars = 1;  // avoid zero-size allocation

    // Loop-point catalog (Phase 1 of virtual-tree): a radix node r is
    // "loop-eligible" for virtual-tree attachment iff it has edge_mass > 1
    // (so K/V is in the compact cache) AND is a leaf in the radix trie (no
    // children). These are the depth-D truncation points where multiple
    // corpus paths converged and got cut off — the natural places to attach
    // a cycle-2+ root walk. Mass=1 leaves are single-corpus-suffix dead ends;
    // extending a virtual cycle there adds no branching signal and their K/V
    // isn't even cached, so they're excluded.
    unsigned char* is_loop_point = (unsigned char*)calloc(trie.radix_count, sizeof(unsigned char));
    {
        // First pass: count children per node to identify leaves.
        int* child_count = (int*)calloc(trie.radix_count, sizeof(int));
        for (int r = 1; r < trie.radix_count; r++) child_count[trie.parents[r]]++;
        long long n_loop = 0;
        long long n_mass_gt1_leaves = 0;
        long long n_mass_gt1_total = 0;
        for (int r = 1; r < trie.radix_count; r++) {
            bool is_leaf = (child_count[r] == 0);
            bool has_mass = (trie.edge_mass[r] > 1);
            if (has_mass) n_mass_gt1_total++;
            if (is_leaf && has_mass) {
                is_loop_point[r] = 1;
                n_loop++;
                n_mass_gt1_leaves++;
            }
        }
        if (!quiet) {
            double pct_of_mass_gt1 = (n_mass_gt1_total > 0)
                ? 100.0 * (double)n_loop / (double)n_mass_gt1_total : 0.0;
            printf("  loop points: %lld (mass>1 leaves; %.1f%% of %lld mass>1 nodes)\n",
                   n_loop, pct_of_mass_gt1, n_mass_gt1_total);
        }
        free(child_count);
    }
    unsigned char* d_is_loop_point;
    CUDA_CHECK(cudaMalloc(&d_is_loop_point, (long long)trie.radix_count * sizeof(unsigned char)));
    CUDA_CHECK(cudaMemcpy(d_is_loop_point, is_loop_point,
                          (long long)trie.radix_count * sizeof(unsigned char), cudaMemcpyHostToDevice));

    // Precompute real RoPE position per char_pos. Matches the forward's scatter
    // convention: pos = first_char_depth + j - 1, clamped to [0, seq_len-1].
    // Used by the delta-RoPE K gather so it can reconstruct the real rotation
    // angle of each cached entry. Virtual-tree training will pass different
    // read positions to the same cache entries.
    int* real_pos_of_char = (int*)malloc((long long)trie.total_edge_chars * sizeof(int));
    for (int r = 0; r < trie.radix_count; r++) {
        int start = trie.edge_starts[r];
        int len   = trie.edge_lens[r];
        int fcd   = trie.edge_first_char_depths[r];
        for (int j = 0; j < len; j++) {
            int pos = fcd + j - 1;
            if (pos < 0) pos = 0;
            if (pos >= cfg.seq_len) pos = cfg.seq_len - 1;
            real_pos_of_char[start + j] = pos;
        }
    }
    long long kv_bytes = n_compact_chars * (long long)D * (long long)sizeof(__nv_bfloat16);
    long long total_kv_bytes = kv_bytes * 2 * L_layers;
    {
        struct sysinfo si;
        if (sysinfo(&si) == 0) {
            long long avail_total = (long long)(si.freeram + si.freeswap) * si.mem_unit;
            long long safe_limit = (avail_total * 4) / 5;
            if (total_kv_bytes > safe_limit) {
                fprintf(stderr, "REFUSED: KV cache needs %.1f GB but only %.1f GB RAM+swap available\n",
                        total_kv_bytes / 1e9, avail_total / 1e9);
                exit(1);
            }
        }
    }
    CacheRuntime cache_runtime;
    init_cache_runtime(cache_runtime, L_layers, kv_bytes,
                       compact_slot, (long long)trie.total_edge_chars,
                       real_pos_of_char, (long long)trie.total_edge_chars);
    if (!quiet) printf("  KV cache: %.1f MB unified memory (bf16 compact)\n", total_kv_bytes / 1e6);

    // RoPE cache
    float* d_rope_cos; float* d_rope_sin;
    build_rope_cache(&d_rope_cos, &d_rope_sin, cfg.seq_len, HD);

    // Per-chunk working buffers. Sized to T_q_cap queries, T_kv_cap packed KV.
    int N_cap = CHUNK_QUERIES;  // worst-case radix count per chunk when edge_len=1

    TransformerChunkRuntime transformer_runtime;
    init_transformer_chunk_runtime(transformer_runtime, T_q_cap, N_cap, D, F, V, L_layers, H, HD, max_kv_per_node);
    float *d_x = transformer_runtime.d_x, *d_ln_out = transformer_runtime.d_ln_out;
    float *d_q = transformer_runtime.d_q, *d_k = transformer_runtime.d_k, *d_v = transformer_runtime.d_v, *d_attn_out = transformer_runtime.d_attn_out, *d_ff_h = transformer_runtime.d_ff_h, *d_ff_out = transformer_runtime.d_ff_out;
    float *d_final_out = transformer_runtime.d_final_out, *d_final_norm_save = transformer_runtime.d_final_norm_save, *d_final_std_inv_save = transformer_runtime.d_final_std_inv_save, *d_logits = transformer_runtime.d_logits, *d_d_logits = transformer_runtime.d_d_logits, *d_loss = transformer_runtime.d_loss, *d_d_final_out = transformer_runtime.d_d_final_out;
    float *d_dk_own = transformer_runtime.d_dk_own, *d_dv_own = transformer_runtime.d_dv_own;
    float** sv_x_res1 = transformer_runtime.sv_x_res1;
    float** sv_ln1_norm = transformer_runtime.sv_ln1_norm;
    float** sv_ln1_std_inv = transformer_runtime.sv_ln1_std_inv;
    float** sv_ln1_out = transformer_runtime.sv_ln1_out;
    float** sv_x_res2 = transformer_runtime.sv_x_res2;
    float** sv_ln2_norm = transformer_runtime.sv_ln2_norm;
    float** sv_ln2_std_inv = transformer_runtime.sv_ln2_std_inv;
    float** sv_ln2_out = transformer_runtime.sv_ln2_out;
    float** sv_ff_h = transformer_runtime.sv_ff_h;
    float** sv_ff_mask = transformer_runtime.sv_ff_mask;
    float** sv_attn_out = transformer_runtime.sv_attn_out;
    float** sv_attn_weights = transformer_runtime.sv_attn_weights;
    float** sv_q = transformer_runtime.sv_q;
    float** sv_k = transformer_runtime.sv_k;
    float** sv_v = transformer_runtime.sv_v;
    float *d_kv_pack_k = transformer_runtime.d_kv_pack_k, *d_kv_pack_v = transformer_runtime.d_kv_pack_v;
    float *d_dq_pack = transformer_runtime.d_dq_pack, *d_dk_pack = transformer_runtime.d_dk_pack, *d_dv_pack = transformer_runtime.d_dv_pack;
    long long T_kv_max = transformer_runtime.T_kv_max;

    // Trie upload
    int *d_radix_counts_offset, *d_radix_counts_tok, *d_radix_counts_val;
    CUDA_CHECK(cudaMalloc(&d_radix_counts_offset, (trie.radix_count + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_radix_counts_tok,    trie.total_counts * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_radix_counts_val,    trie.total_counts * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_radix_counts_offset, trie.counts_offset, (trie.radix_count + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_radix_counts_tok,    trie.counts_tok,    trie.total_counts * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_radix_counts_val,    trie.counts_val,    trie.total_counts * sizeof(int), cudaMemcpyHostToDevice));

    // Per-chunk upload buffers (allocated once, reused)
    int *d_radix_ids;         // [N_cap]
    int *d_query_to_node;     // [T_q_cap]
    int *d_query_offsets;     // [N_cap+1]
    int *d_kv_offsets;        // [N_cap+1]
    int *d_kv_lengths;        // [N_cap]
    int *d_token_ids;         // [T_q_cap] for embedding gather
    int *d_rope_positions;    // [T_q_cap * H] for RoPE (per-query, replicated per head)
    int *d_char_pos;          // [T_q_cap] global character position per query (for KV scatter)
    ChunkUploadRuntime chunk_upload_runtime;
    init_chunk_upload_runtime(chunk_upload_runtime, N_cap, T_q_cap, H);
    d_radix_ids = chunk_upload_runtime.d_radix_ids;
    d_query_to_node = chunk_upload_runtime.d_query_to_node;
    d_query_offsets = chunk_upload_runtime.d_query_offsets;
    d_kv_offsets = chunk_upload_runtime.d_kv_offsets;
    d_kv_lengths = chunk_upload_runtime.d_kv_lengths;
    d_token_ids = chunk_upload_runtime.d_token_ids;
    d_rope_positions = chunk_upload_runtime.d_rope_positions;
    d_char_pos = chunk_upload_runtime.d_char_pos;

    // Per-query mass weight buffer. Allocated unconditionally so it's
    // available for ancestor-loss-masking (Lightning prepends ancestors to
    // the BFS for K/V freshness; their loss must be zeroed via this buffer
    // even when --mass-weight is off). Cost: T_q_cap * 4 bytes (~kB).
    float* d_mass_weights = NULL;
    CUDA_CHECK(cudaMalloc(&d_mass_weights, T_q_cap * sizeof(float)));

    // cuBLAS handle. Enable TF32 tensor cores for FP32 matmuls — gives 2-3×
    // speedup on Ampere+ (A100, RTX 30xx, RTX 40xx, H100/H200) at no accuracy
    // cost vs FP32 inputs/outputs (TF32 uses FP32 range with 10-bit mantissa
    // internally). On pre-Ampere GPUs this is a no-op fall-back to FP32.
    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));
    CUBLAS_CHECK(cublasSetMathMode(cublas, CUBLAS_TF32_TENSOR_OP_MATH));

    if (!quiet) {
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        printf("  GPU memory: %.1f MB used, %.1f MB free, %.1f MB total\n",
               (total_mem - free_mem) / 1e6, free_mem / 1e6, total_mem / 1e6);
    }

    // ------------------------------------------------------------
    // Build root-child subtree grouping.
    // Each depth-1 radix node (parent == 0) is a "root-child"; its subtree is
    // itself plus all radix descendants. These partition the trie below depth 1.
    // AGPT invariants: the subtree is the training unit. Weights are fixed
    // across all chunks of a subtree; one Adam step per subtree.
    // ------------------------------------------------------------

    // root_child_of[r] = the depth-1 radix ancestor of r (== r if r is itself depth-1)
    int* root_child_of = (int*)calloc(trie.radix_count, sizeof(int));
    for (int r = 1; r < trie.radix_count; r++) {
        int cur = r;
        while (cur > 0 && trie.parents[cur] != 0) {
            cur = trie.parents[cur];
        }
        root_child_of[r] = cur;  // 0 if r has no valid ancestry; should not happen for r >= 1
    }

    // Collect root-children (depth-1 radix nodes, i.e., parent == 0)
    int n_root_children = 0;
    for (int r = 1; r < trie.radix_count; r++) {
        if (trie.parents[r] == 0) n_root_children++;
    }
    int* root_children = (int*)malloc(n_root_children * sizeof(int));
    {
        int idx = 0;
        for (int r = 1; r < trie.radix_count; r++) {
            if (trie.parents[r] == 0) root_children[idx++] = r;
        }
    }

    // For each root-child, a BFS-sorted list of its subtree's radix ids.
    // Ordering guarantees ancestors are processed before descendants (so their
    // K/V is written to the cache before any descendant reads it).
    int** subtree_nodes = (int**)malloc(n_root_children * sizeof(int*));
    int* subtree_sizes = (int*)calloc(n_root_children, sizeof(int));
    // Per-Lightning-sample count of ancestors prepended to subtree_nodes[s].
    // The bucket sort places ancestors first (lowest endpoint depth), so the
    // global indices [0, subtree_n_anc[s]) within subtree_nodes[s] are ancestors.
    // For deterministic AGPT and L4, no ancestors are prepended → stays 0.
    // Used by chunk processing to mask ancestor positions out of the loss
    // (forward K/V scatter still happens, populating cache with current-weight
    // K/V; only the first-order CE loss is suppressed to avoid over-training
    // shallow ancestors that appear in many L3 BFS sets per epoch).
    int* subtree_n_anc = (int*)calloc(n_root_children, sizeof(int));
    // rc_index[rc_id] = index into root_children[]
    // Built on demand via linear search since n_root_children ≤ vocab_size (≤ 65).

    auto rc_index_of = [&](int rc_id) -> int {
        for (int i = 0; i < n_root_children; i++) if (root_children[i] == rc_id) return i;
        return -1;
    };

    for (int r = 1; r < trie.radix_count; r++) {
        int rc = root_child_of[r];
        int i = rc_index_of(rc);
        if (i >= 0) subtree_sizes[i]++;
    }
    for (int i = 0; i < n_root_children; i++) {
        subtree_nodes[i] = (int*)malloc(subtree_sizes[i] * sizeof(int));
    }
    {
        // Fill; then sort each by endpoint depth for BFS ordering.
        int* fills = (int*)calloc(n_root_children, sizeof(int));
        for (int r = 1; r < trie.radix_count; r++) {
            int rc = root_child_of[r];
            int i = rc_index_of(rc);
            if (i >= 0) subtree_nodes[i][fills[i]++] = r;
        }
        free(fills);

        // Sort each root-child's list by endpoint depth (stable is fine).
        for (int i = 0; i < n_root_children; i++) {
            int* arr = subtree_nodes[i];
            int sz = subtree_sizes[i];
            // Simple insertion/bubble for small arrays; qsort for larger.
            if (sz > 16) {
                // Use qsort
                static const int* g_edge_first_char_depths_ptr = NULL;
                static const int* g_edge_lens_ptr = NULL;
                g_edge_first_char_depths_ptr = trie.edge_first_char_depths;
                g_edge_lens_ptr = trie.edge_lens;
                auto cmp = +[](const void* a, const void* b) -> int {
                    int ra = *(const int*)a, rb = *(const int*)b;
                    // Need trie data; use globals captured above (not ideal — switch to lambda body).
                    // Since we can't pass context to qsort, do a simple selection loop for larger.
                    (void)ra; (void)rb; return 0;
                };
                (void)cmp;
                // Fall back: simple O(n^2) for sizes up to ~10k; otherwise do bucket sort by depth
                if (sz <= 20000) {
                    for (int a = 1; a < sz; a++) {
                        int key = arr[a];
                        int key_ep = trie.edge_first_char_depths[key] + trie.edge_lens[key] - 1;
                        int b = a - 1;
                        while (b >= 0) {
                            int cur_ep = trie.edge_first_char_depths[arr[b]] + trie.edge_lens[arr[b]] - 1;
                            if (cur_ep <= key_ep) break;
                            arr[b+1] = arr[b];
                            b--;
                        }
                        arr[b+1] = key;
                    }
                } else {
                    // Bucket sort by endpoint depth
                    int max_ep = 0;
                    for (int a = 0; a < sz; a++) {
                        int ep = trie.edge_first_char_depths[arr[a]] + trie.edge_lens[arr[a]] - 1;
                        if (ep > max_ep) max_ep = ep;
                    }
                    int* bucket_counts = (int*)calloc(max_ep + 2, sizeof(int));
                    for (int a = 0; a < sz; a++) {
                        int ep = trie.edge_first_char_depths[arr[a]] + trie.edge_lens[arr[a]] - 1;
                        bucket_counts[ep + 1]++;
                    }
                    for (int e = 0; e < max_ep + 1; e++) bucket_counts[e + 1] += bucket_counts[e];
                    int* sorted = (int*)malloc(sz * sizeof(int));
                    int* cursors = (int*)calloc(max_ep + 2, sizeof(int));
                    for (int a = 0; a < sz; a++) {
                        int ep = trie.edge_first_char_depths[arr[a]] + trie.edge_lens[arr[a]] - 1;
                        sorted[bucket_counts[ep] + cursors[ep]] = arr[a];
                        cursors[ep]++;
                    }
                    memcpy(arr, sorted, sz * sizeof(int));
                    free(sorted); free(bucket_counts); free(cursors);
                }
            } else {
                // Insertion sort for very small
                for (int a = 1; a < sz; a++) {
                    int key = arr[a];
                    int key_ep = trie.edge_first_char_depths[key] + trie.edge_lens[key] - 1;
                    int b = a - 1;
                    while (b >= 0) {
                        int cur_ep = trie.edge_first_char_depths[arr[b]] + trie.edge_lens[arr[b]] - 1;
                        if (cur_ep <= key_ep) break;
                        arr[b+1] = arr[b];
                        b--;
                    }
                    arr[b+1] = key;
                }
            }
        }
    }

    // Stats
    {
        int min_sz = INT_MAX, max_sz = 0; long long total_sz = 0;
        for (int i = 0; i < n_root_children; i++) {
            if (subtree_sizes[i] < min_sz) min_sz = subtree_sizes[i];
            if (subtree_sizes[i] > max_sz) max_sz = subtree_sizes[i];
            total_sz += subtree_sizes[i];
        }
        if (!quiet) printf("  %d root-child subtrees: sizes min=%d max=%d avg=%.1f (total=%lld radix nodes)\n",
               n_root_children, min_sz, max_sz, (double)total_sz / n_root_children, total_sz);
    }

    // Single-subtree mode: collapse all 65 root-child subtrees into one.
    // This tests whether the 65-way partitioning is introducing asymmetric
    // gradient behavior (e.g., early common-letter subtrees dominating Adam's
    // running stats). Trade-off: 1 Adam step per epoch instead of 65.
    if (single_subtree) {
        int* big = (int*)malloc(trie.radix_count * sizeof(int));
        int fill = 0;
        // Collect all subtree nodes in root-child order, preserving BFS within each
        for (int i = 0; i < n_root_children; i++) {
            for (int j = 0; j < subtree_sizes[i]; j++) big[fill++] = subtree_nodes[i][j];
            free(subtree_nodes[i]);
        }
        free(subtree_nodes);
        free(subtree_sizes);
        free(subtree_n_anc);
        subtree_nodes = (int**)malloc(sizeof(int*));
        subtree_sizes = (int*)malloc(sizeof(int));
        subtree_n_anc = (int*)calloc(1, sizeof(int));  // no ancestors in deterministic AGPT
        subtree_nodes[0] = big;
        subtree_sizes[0] = fill;
        // Re-sort globally by endpoint depth so BFS order holds across the full subtree
        {
            int sz = fill;
            int max_ep = 0;
            for (int a = 0; a < sz; a++) {
                int ep = trie.edge_first_char_depths[big[a]] + trie.edge_lens[big[a]] - 1;
                if (ep > max_ep) max_ep = ep;
            }
            int* bucket_counts = (int*)calloc(max_ep + 2, sizeof(int));
            for (int a = 0; a < sz; a++) {
                int ep = trie.edge_first_char_depths[big[a]] + trie.edge_lens[big[a]] - 1;
                bucket_counts[ep + 1]++;
            }
            for (int e = 0; e < max_ep + 1; e++) bucket_counts[e + 1] += bucket_counts[e];
            int* sorted = (int*)malloc(sz * sizeof(int));
            int* cursors = (int*)calloc(max_ep + 2, sizeof(int));
            for (int a = 0; a < sz; a++) {
                int ep = trie.edge_first_char_depths[big[a]] + trie.edge_lens[big[a]] - 1;
                sorted[bucket_counts[ep] + cursors[ep]] = big[a];
                cursors[ep]++;
            }
            memcpy(big, sorted, sz * sizeof(int));
            free(sorted); free(bucket_counts); free(cursors);
        }
        n_root_children = 1;
        if (!quiet) printf("  single-subtree mode: one subtree of %d radix nodes (BFS-sorted)\n", fill);
    }

    // ------------------------------------------------------------
    // N-gram partition (--partition-depth N, N>1).
    // Re-groups whatever subtree buckets we currently have into finer
    // buckets keyed by depth-N radix ancestor. N=1 is a no-op (keeps
    // current per-root-child or single-subtree layout). N=2 = bigram
    // (~1139 groups on d=16 Shakespeare). Each group still gets ONE
    // Adam step per super-epoch, so this multiplies optimizer steps.
    // BFS-sort within each group is preserved because we bucket from
    // the already-BFS-sorted input list in order.
    // ------------------------------------------------------------
    if (partition_depth > 1 && n_root_children > 0) {
        int total = 0;
        for (int i = 0; i < n_root_children; i++) total += subtree_sizes[i];
        int* all_nodes = (int*)malloc((total > 0 ? total : 1) * sizeof(int));
        int fill2 = 0;
        for (int i = 0; i < n_root_children; i++) {
            for (int j = 0; j < subtree_sizes[i]; j++) all_nodes[fill2++] = subtree_nodes[i][j];
            free(subtree_nodes[i]);
        }
        free(subtree_nodes);
        free(subtree_sizes);

        // Depth-N ancestor of each touched radix node: walk parents until
        // the edge covers depth N (first_char_depth <= N <= endpoint_depth).
        // Returns 0 if the node's path is shallower than N (whole root-to-r
        // path has length < N); those nodes all clump into group keyed by 0.
        int* partition_ancestor = (int*)malloc(trie.radix_count * sizeof(int));
        for (int r = 0; r < trie.radix_count; r++) partition_ancestor[r] = -1;
        for (int k = 0; k < fill2; k++) {
            int r = all_nodes[k];
            if (partition_ancestor[r] != -1) continue;
            int cur = r;
            while (cur > 0) {
                int fcd = trie.edge_first_char_depths[cur];
                int ed  = fcd + trie.edge_lens[cur] - 1;
                if (fcd <= partition_depth && ed >= partition_depth) break;
                cur = trie.parents[cur];
            }
            partition_ancestor[r] = cur;  // 0 = no ancestor covers depth N
        }

        // Collect unique partition keys (group ids), preserving first-seen order.
        char* seen = (char*)calloc(trie.radix_count, 1);
        int* key_to_group = (int*)malloc(trie.radix_count * sizeof(int));
        for (int r = 0; r < trie.radix_count; r++) key_to_group[r] = -1;
        int n_groups = 0;
        int* group_keys = (int*)malloc((fill2 > 0 ? fill2 : 1) * sizeof(int));
        for (int k = 0; k < fill2; k++) {
            int key = partition_ancestor[all_nodes[k]];
            if (key < 0) key = 0;
            if (!seen[key]) {
                seen[key] = 1;
                key_to_group[key] = n_groups;
                group_keys[n_groups++] = key;
            }
        }

        int* group_sizes = (int*)calloc(n_groups, sizeof(int));
        for (int k = 0; k < fill2; k++) {
            int key = partition_ancestor[all_nodes[k]];
            if (key < 0) key = 0;
            group_sizes[key_to_group[key]]++;
        }
        int** group_nodes = (int**)malloc(n_groups * sizeof(int*));
        for (int g = 0; g < n_groups; g++) {
            group_nodes[g] = (int*)malloc((group_sizes[g] > 0 ? group_sizes[g] : 1) * sizeof(int));
        }
        int* fills_arr = (int*)calloc(n_groups, sizeof(int));
        for (int k = 0; k < fill2; k++) {
            int key = partition_ancestor[all_nodes[k]];
            if (key < 0) key = 0;
            int g = key_to_group[key];
            group_nodes[g][fills_arr[g]++] = all_nodes[k];
        }

        free(fills_arr);
        free(all_nodes);
        free(partition_ancestor);
        free(key_to_group);
        free(group_keys);
        free(seen);

        subtree_nodes = group_nodes;
        subtree_sizes = group_sizes;
        free(subtree_n_anc);
        subtree_n_anc = (int*)calloc(n_groups, sizeof(int));  // no ancestors in deterministic AGPT
        n_root_children = n_groups;   // semantics: now "partition groups"

        if (!quiet) {
            printf("  partition-depth=%d: %d groups, 1 Adam step per group per super-epoch\n",
                   partition_depth, n_groups);
        }
    }

    // ------------------------------------------------------------
    // Per-rc Adam state (Stage 1 of topological optimizer state).
    // When cfg.per_rc_adam is set, each root-child / partition-group has its
    // own (m, v, t) instead of sharing global state. Used in place of
    // d_adam_m, d_adam_v at the fire site. Memory: n_root_children *
    // total_floats * 8 bytes (two float arrays).
    // ------------------------------------------------------------
    float* d_adam_m_per_rc = NULL;
    float* d_adam_v_per_rc = NULL;
    int*   h_adam_t_per_rc = NULL;
    if (cfg.per_rc_adam) {
        size_t per_rc_bytes = (size_t)n_root_children * (size_t)wo.total_floats * sizeof(float);
        const size_t budget_bytes = (size_t)2 * 1024 * 1024 * 1024;  // 2 GB per buffer
        if (per_rc_bytes > budget_bytes) {
            fprintf(stderr,
                "ERROR: --per-rc-adam memory budget exceeded: %.2f GB per moment buffer (limit 2 GB).\n"
                "       n_root_children=%d, total_floats=%d. Try a smaller partition-depth.\n",
                (double)per_rc_bytes / 1.0e9, n_root_children, wo.total_floats);
            return 1;
        }
        CUDA_CHECK(cudaMalloc(&d_adam_m_per_rc, per_rc_bytes));
        CUDA_CHECK(cudaMalloc(&d_adam_v_per_rc, per_rc_bytes));
        CUDA_CHECK(cudaMemset(d_adam_m_per_rc, 0, per_rc_bytes));
        CUDA_CHECK(cudaMemset(d_adam_v_per_rc, 0, per_rc_bytes));
        h_adam_t_per_rc = (int*)calloc(n_root_children, sizeof(int));
        if (!quiet) {
            printf("  per-rc-adam: %d buckets x %d params = %.1f MB per moment buffer\n",
                   n_root_children, wo.total_floats, (double)per_rc_bytes / 1.0e6);
        }
    }

    // ------------------------------------------------------------
    // Descendant→ancestor gradient buffers (--anc-grad).
    // Per-subtree-scoped, indexed by COMPACT-CACHE CHARACTER POSITION
    // (not by radix node). The K/V cache itself is per-compact-char,
    // with mass=1 caps skipped. Descendant gradients flow back to
    // specific compact-char slots, so our accumulators mirror that.
    //
    // Per layer:
    //   d_dkv_subtree_k[l]: [max_n_subtree_compact_chars, D]  K-grad accumulator
    //   d_dkv_subtree_v[l]: [max_n_subtree_compact_chars, D]  V-grad accumulator
    //   h_subtree[l]:       [max_n_subtree_compact_chars, D]  saved ln1_out
    //
    // Global lookup (one buffer):
    //   d_compact_to_subtree_idx: [n_compact_chars] int
    //     map global compact-cache index → subtree-local index
    //     (-1 sentinel for chars not in current subtree)
    //
    // SCAFFOLDING ONLY at this commit — buffers allocated, not yet used.
    // Subsequent commits will wire the scatter + chain-rule reduction.
    // ------------------------------------------------------------
    float** d_dkv_subtree_k = NULL;     // [L_layers] of device pointers
    float** d_dkv_subtree_v = NULL;
    float** h_subtree       = NULL;
    int*    d_compact_to_subtree_idx = NULL;
    int*    d_subtree_real_pos = NULL;  // [max_n_subtree_compact_chars] — RoPE position
                                        // per subtree-local slot, used by fire-end RoPE-inverse
                                        // on the K-grad accumulator.
    int     max_n_subtree_compact_chars = 0;
    if (cfg.anc_grad) {
        // Compute max compact-chars per subtree: for each subtree, sum
        // own_len across its mass>1 nodes (mass=1 nodes have no cache slots).
        for (int s = 0; s < n_root_children; s++) {
            int* radix_list_s = subtree_nodes[s];
            int n_s = subtree_sizes[s];
            int count = 0;
            for (int i = 0; i < n_s; i++) {
                int r = radix_list_s[i];
                if (trie.edge_mass[r] == 1) continue;  // mass=1 cap: no cache slot
                count += trie.edge_lens[r];
            }
            if (count > max_n_subtree_compact_chars) max_n_subtree_compact_chars = count;
        }
        size_t per_layer_bytes = (size_t)max_n_subtree_compact_chars * (size_t)D * sizeof(float);
        d_dkv_subtree_k = (float**)malloc(L_layers * sizeof(float*));
        d_dkv_subtree_v = (float**)malloc(L_layers * sizeof(float*));
        h_subtree       = (float**)malloc(L_layers * sizeof(float*));
        for (int l = 0; l < L_layers; l++) {
            CUDA_CHECK(cudaMalloc(&d_dkv_subtree_k[l], per_layer_bytes));
            CUDA_CHECK(cudaMalloc(&d_dkv_subtree_v[l], per_layer_bytes));
            CUDA_CHECK(cudaMalloc(&h_subtree[l],       per_layer_bytes));
        }
        CUDA_CHECK(cudaMalloc(&d_compact_to_subtree_idx, (size_t)n_compact_chars * sizeof(int)));
        // d_subtree_real_pos sized [max_n × H]: existing launch_rope_batched_inverse
        // expects one position per (row = slot*H + head) entry. Same pos repeats
        // across heads of the same slot.
        CUDA_CHECK(cudaMalloc(&d_subtree_real_pos, (size_t)max_n_subtree_compact_chars * H * sizeof(int)));
        size_t lookup_bytes = (size_t)n_compact_chars * sizeof(int);
        size_t pos_bytes    = (size_t)max_n_subtree_compact_chars * H * sizeof(int);
        size_t total_bytes = 3 * L_layers * per_layer_bytes + lookup_bytes + pos_bytes;
        if (!quiet) {
            printf("  anc-grad: max_n_subtree_compact_chars=%d, %d layers x 3 buffers x %.2f MB + %.2f MB lookup = %.2f MB total\n",
                   max_n_subtree_compact_chars, L_layers,
                   (double)per_layer_bytes / 1.0e6,
                   (double)lookup_bytes / 1.0e6,
                   (double)total_bytes / 1.0e6);
        }
    }

    // ------------------------------------------------------------
    // Lightning Training adjacency precompute.
    // We build an inverted parents[] → children adjacency table once, plus
    // cumulative child weights used by L3's mass-weighted descent.
    // Per-epoch resampling (inside the training loop) frees the current
    // subtree_nodes[] / subtree_sizes[] and replaces them with N new samples.
    // ------------------------------------------------------------
    int* lightning_children_offsets = NULL;
    int* lightning_children_flat    = NULL;
    int lightning_active = (lightning.steps > 0);
    unsigned lightning_rng = lightning.seed;

    // Lightning resamples subtree_nodes[] each epoch, overwriting any partition
    // layout (--partition-depth, --single-subtree) — so combining them just
    // wastes the pre-build. Hard-error to surface the mistake. These flags are
    // alternative ways to shape the training unit, not orthogonal modifiers.
    // --curriculum progressive would need depth_limit[] rebuilt per-sample;
    // not implemented here — Lightning's p_stop is the stochastic analogue
    // (depth control via sampling bias, not explicit depth bounds).
    if (lightning_active && curriculum == CurriculumMode::Progressive) {
        fprintf(stderr, "Lightning is incompatible with --curriculum progressive: progressive "
                        "controls depth via an explicit d=1..D schedule, Lightning controls "
                        "depth stochastically via p_stop. Use one or the other.\n");
        exit(1);
    }
    // Note: Lightning + single_subtree is allowed. single_subtree collapses the
    // 65 root-child buckets into one list; Lightning then overwrites that list
    // with N stochastic samples. The collapse is wasted work but harmless. This
    // path is used by run_per_subtree_training when loading one root-child at a
    // time — each loaded view is effectively a single-subtree trie.
    if (lightning_active && partition_depth > 1) {
        fprintf(stderr, "Lightning resamples subtrees each epoch; --partition-depth's n-gram "
                        "bucket layout would be discarded. Do not combine.\n");
        exit(1);
    }

    // Build radix-tree child adjacency unconditionally — needed by Lightning
    // and by the hotspot-curriculum splitter.
    {
        int* cnt = (int*)calloc(trie.radix_count, sizeof(int));
        for (int r = 1; r < trie.radix_count; r++) cnt[trie.parents[r]]++;
        lightning_children_offsets = (int*)calloc(trie.radix_count + 1, sizeof(int));
        for (int r = 0; r < trie.radix_count; r++) {
            lightning_children_offsets[r + 1] = lightning_children_offsets[r] + cnt[r];
        }
        long long total_child_edges = lightning_children_offsets[trie.radix_count];
        lightning_children_flat = (int*)malloc((total_child_edges > 0 ? total_child_edges : 1) * sizeof(int));
        int* cur_idx = (int*)calloc(trie.radix_count, sizeof(int));
        for (int r = 1; r < trie.radix_count; r++) {
            int p = trie.parents[r];
            lightning_children_flat[lightning_children_offsets[p] + cur_idx[p]++] = r;
        }
        free(cur_idx);
        free(cnt);
        if (!quiet && lightning_active) {
            printf("  lightning: built adjacency (%lld child edges total; root has %d children)\n",
                   total_child_edges, lightning_children_offsets[1] - lightning_children_offsets[0]);
        }
    }

    if (lightning_active) {
        // Force accumulate=false: each Lightning sample is its own bounded
        // training unit with one optimizer step. Without this override the
        // whole super-epoch of N samples would collapse into one step, which
        // is not what Lightning is supposed to do.
        accumulate = false;
    }

    // For progressive curriculum: per-subtree cumulative "how many nodes are
    // within endpoint_depth ≤ d?" — because subtree_nodes[i] is sorted by
    // endpoint depth, this is a simple prefix scan.
    // depth_limit[rc_idx][d] = count of nodes in subtree_nodes[rc_idx] with
    // endpoint_depth ≤ d (equivalently, the exclusive upper bound index).
    int** depth_limit = NULL;
    int curriculum_max_depth = trie.depth_file_count - 1; // last valid endpoint depth
    if (curriculum == CurriculumMode::Progressive) {
        depth_limit = (int**)malloc(n_root_children * sizeof(int*));
        for (int i = 0; i < n_root_children; i++) {
            depth_limit[i] = (int*)calloc(curriculum_max_depth + 2, sizeof(int));
            int* arr = subtree_nodes[i];
            int sz = subtree_sizes[i];
            int cursor = 0;
            for (int d = 0; d <= curriculum_max_depth; d++) {
                while (cursor < sz) {
                    int r = arr[cursor];
                    int ep = trie.edge_first_char_depths[r] + trie.edge_lens[r] - 1;
                    if (ep > d) break;
                    cursor++;
                }
                depth_limit[i][d + 1] = cursor;  // exclusive upper bound
            }
        }
    }

    int adam_t = (persist && persist->adam_t_io) ? *persist->adam_t_io : 0;

    // Persistent state for LR rules that depend on previous epoch's residual.
    // Populated at end of each epoch; used at start of the next.
    double* prev_epoch_score = NULL;
    int     prev_epoch_n = 0;

    // ------------------------------------------------------------
    // Training loop
    // ------------------------------------------------------------
    // Subtree dropout keep-mask, regenerated each super-epoch.
    char* subtree_keep = NULL;
    if (subtree_dropout > 0.0) {
        subtree_keep = (char*)malloc(n_root_children * sizeof(char));
    }
    // Branch-dropout per-radix-node mask (1 = dropped, 0 = kept).
    // Used only when branch_dropout_depth > 0. Marked dropped if the node OR
    // any of its ancestors is at branch_dropout_depth and was randomly selected.
    char* branch_drop_mask = NULL;
    if (subtree_dropout > 0.0 && branch_dropout_depth > 0) {
        branch_drop_mask = (char*)calloc(trie.radix_count, sizeof(char));
    }
    unsigned int sd_rng_state = subtree_dropout_seed;
    unsigned int shuffle_rng_state = cfg.shuffle_order ? cfg.shuffle_seed : 0u;

    for (int epoch = 0; epoch < epochs; epoch++) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        // Reset per-epoch timing accumulators
        if (t_enabled) {
            t_us_gather_fwd = t_us_gather_bwd = 0;
            t_us_attn_fwd = t_us_attn_bwd = 0;
            t_us_scatter_fwd = 0;
        }

        // --shuffle-order: Fisher-Yates permutation of partition groups per
        // super-epoch. Tests order-independence at constant exposure (every
        // group still hit exactly once). Shuffles subtree_nodes/sizes/n_anc
        // in tandem; per-epoch arrays (mass/lr_mult/loss_sum) are populated
        // below from the shuffled order so they stay consistent.
        if (cfg.shuffle_order && n_root_children > 1) {
            for (int i = n_root_children - 1; i > 0; i--) {
                shuffle_rng_state ^= shuffle_rng_state << 13;
                shuffle_rng_state ^= shuffle_rng_state >> 17;
                shuffle_rng_state ^= shuffle_rng_state << 5;
                int j = (int)(shuffle_rng_state % (unsigned)(i + 1));
                int* tn = subtree_nodes[i]; subtree_nodes[i] = subtree_nodes[j]; subtree_nodes[j] = tn;
                int  ts = subtree_sizes[i]; subtree_sizes[i] = subtree_sizes[j]; subtree_sizes[j] = ts;
                int  ta = subtree_n_anc[i]; subtree_n_anc[i] = subtree_n_anc[j]; subtree_n_anc[j] = ta;
            }
            if (!quiet && epoch == 0) {
                printf("  shuffle-order: per-SE Fisher-Yates over %d partition groups (seed=0x%x)\n",
                       n_root_children, cfg.shuffle_seed);
            }
        }

        // Subtree dropout: sample which root-child subtrees to keep this epoch.
        // Simple LCG to keep it deterministic per seed.
        if (subtree_dropout > 0.0 && branch_dropout_depth == 0) {
            int n_kept = 0;
            for (int i = 0; i < n_root_children; i++) {
                sd_rng_state = sd_rng_state * 1664525u + 1013904223u;
                double u = (double)(sd_rng_state >> 8) / (double)(1u << 24);
                subtree_keep[i] = (u >= subtree_dropout) ? 1 : 0;
                if (subtree_keep[i]) n_kept++;
            }
            // Guard: if we dropped everything, keep at least one (the largest by mass).
            if (n_kept == 0) {
                int best = 0;
                for (int i = 1; i < n_root_children; i++) {
                    if (subtree_sizes[i] > subtree_sizes[best]) best = i;
                }
                subtree_keep[best] = 1;
                n_kept = 1;
            }
            if (!quiet) {
                printf("  [epoch %d] subtree-dropout: keeping %d/%d root-children\n",
                       epoch + 1, n_kept, n_root_children);
            }
        }
        // Branch dropout: sample at internal depth instead of root level.
        // Step 1: at each radix node whose fcd == branch_dropout_depth, sample
        // whether to mask. Step 2: propagate mask to all descendants.
        if (subtree_dropout > 0.0 && branch_dropout_depth > 0) {
            memset(branch_drop_mask, 0, trie.radix_count * sizeof(char));
            int n_candidates = 0, n_masked_seeds = 0;
            for (int r = 1; r < trie.radix_count; r++) {
                if (trie.edge_first_char_depths[r] != branch_dropout_depth) continue;
                n_candidates++;
                sd_rng_state = sd_rng_state * 1664525u + 1013904223u;
                double u = (double)(sd_rng_state >> 8) / (double)(1u << 24);
                if (u < subtree_dropout) {
                    branch_drop_mask[r] = 1;
                    n_masked_seeds++;
                }
            }
            // Propagate: walk parent chain for each node; if any ancestor masked, mark.
            // This is O(depth × radix_count). One pass per node.
            int n_total_masked = n_masked_seeds;
            for (int r = 1; r < trie.radix_count; r++) {
                if (branch_drop_mask[r]) continue;  // self already marked
                int cur = trie.parents[r];
                while (cur > 0) {
                    if (branch_drop_mask[cur]) {
                        branch_drop_mask[r] = 1;
                        n_total_masked++;
                        break;
                    }
                    cur = trie.parents[cur];
                }
            }
            if (!quiet) {
                printf("  [epoch %d] branch-dropout depth=%d: %d/%d seeds masked, %d total nodes (%.1f%%)\n",
                       epoch + 1, branch_dropout_depth,
                       n_masked_seeds, n_candidates, n_total_masked,
                       100.0 * (double)n_total_masked / (double)trie.radix_count);
            }
        }

        // Zero KV caches
        zero_cache_runtime(cache_runtime);

        // ------------------------------------------------------------
        // Lightning resampling: generate N stochastic samples for this
        // super-epoch and replace subtree_nodes[] / subtree_sizes[].
        // ------------------------------------------------------------
        int lightning_depth_hist[64];
        long long lightning_nodes_sum = 0;
        double* lightning_subtree_mass = NULL;
        double lightning_mean_mass = 1.0;
        double lightning_mass_min = 0.0, lightning_mass_max = 0.0;
        // Per-step BFS node count (sz) — the actual cost driver. mass_accum
        // (depth-inflated sum over subtree) is a misleading proxy because it
        // counts each corpus context once per ancestor depth.
        int lightning_size_min = 0, lightning_size_max = 0;
        double lightning_size_mean = 0.0;
        double lightning_w_min = 1.0, lightning_w_max = 1.0;
        // Per-sample virtual-tree cycle shift: shift = k_sample * D_max applied
        // to RoPE angle of Q and K (both own-edge fresh buffer and ancestor
        // delta-rotation). Relative (Q-K) positions within a sample are
        // preserved; absolute angle is shifted to teach the model to handle
        // virtual position D..K*D at inference.
        int* lightning_cycle_shift = NULL;
        int D_max = trie.depth_file_count - 1;
        if (lightning_active) {
            for (int i = 0; i < 64; i++) lightning_depth_hist[i] = 0;
            // Free previous subtree_nodes / subtree_sizes / subtree_n_anc.
            for (int i = 0; i < n_root_children; i++) free(subtree_nodes[i]);
            free(subtree_nodes);
            free(subtree_sizes);
            free(subtree_n_anc);
            n_root_children = lightning.steps;
            subtree_nodes = (int**)malloc(n_root_children * sizeof(int*));
            subtree_sizes = (int*)calloc(n_root_children, sizeof(int));
            subtree_n_anc = (int*)calloc(n_root_children, sizeof(int));
            lightning_subtree_mass = (double*)calloc(n_root_children, sizeof(double));
            lightning_cycle_shift = (int*)calloc(n_root_children, sizeof(int));
            // Pick per-sample cycle shift uniformly from {0, D, 2D, ..., (K-1)D}.
            if (lightning.virtual_cycles > 1) {
                for (int s = 0; s < n_root_children; s++) {
                    int k = (int)(xorshift32(&lightning_rng) % (unsigned)lightning.virtual_cycles);
                    int shift = k * D_max;
                    // Clamp so (shift + D_max) stays inside RoPE cache.
                    if (shift + D_max >= cfg.seq_len) shift = cfg.seq_len - D_max - 1;
                    if (shift < 0) shift = 0;
                    lightning_cycle_shift[s] = shift;
                }
            }

            // Scratch buffers reused across samples: BFS queue of radix ids.
            int* bfs_buf = (int*)malloc(trie.radix_count * sizeof(int));

            for (int s = 0; s < n_root_children; s++) {
                // --- Sample a radix node ---
                int r_sample = 0;
                if (lightning.sampler == LightningSampler::L3_MassWalk ||
                    lightning.sampler == LightningSampler::L4_Path) {
                    // L3: descend with p_stop probability, emit wherever we stop.
                    // L4: always descend to a leaf (p_stop forced to 0). This yields
                    //     a root-to-leaf path weighted by mass — statistically
                    //     equivalent to uniform-corpus-position window sampling
                    //     (SGD baseline comparator).
                    float p_stop_eff = (lightning.sampler == LightningSampler::L4_Path)
                                       ? 0.0f : lightning.p_stop;
                    int cur = 0;
                    while (1) {
                        // Stop probability applies at all nodes except the virtual root.
                        // Force-descend (override p_stop) when the current node's mass
                        // exceeds the configured cap — mass(r) is a tight proxy for
                        // subtree_radix_size(r) at d=32, so this bounds per-step work.
                        bool force_descend = (lightning.max_mass > 0 && cur != 0 &&
                                              (long long)trie.edge_mass[cur] > lightning.max_mass);
                        if (cur != 0 && p_stop_eff > 0.0f && !force_descend) {
                            float u = xorshift_float01(&lightning_rng);
                            if (u < p_stop_eff) break;
                        }
                        int cs = lightning_children_offsets[cur];
                        int ce = lightning_children_offsets[cur + 1];
                        int nc = ce - cs;
                        if (nc == 0) break;  // leaf — emit it
                        double total = 0.0;
                        for (int j = cs; j < ce; j++) {
                            total += (double)trie.edge_mass[lightning_children_flat[j]];
                        }
                        int pick;
                        if (total <= 0.0) {
                            pick = cs + (int)(xorshift32(&lightning_rng) % (unsigned)nc);
                        } else {
                            double u2 = (double)xorshift_float01(&lightning_rng) * total;
                            double acc = 0.0;
                            pick = ce - 1;
                            for (int j = cs; j < ce; j++) {
                                acc += (double)trie.edge_mass[lightning_children_flat[j]];
                                if (u2 <= acc) { pick = j; break; }
                            }
                        }
                        cur = lightning_children_flat[pick];
                    }
                    r_sample = cur;
                } else if (lightning.sampler == LightningSampler::L1_Uniform) {
                    // Uniform over all non-root radix nodes.
                    if (trie.radix_count > 1) {
                        r_sample = 1 + (int)(xorshift32(&lightning_rng) % (unsigned)(trie.radix_count - 1));
                    }
                } else {
                    // L2_RcDepth: pick random depth-1 root-child and random depth.
                    // First-pass: just pick a random root-child (= depth-1 node).
                    // Depth-stratified L2 would need a per-rc-and-depth index; defer.
                    int root_cs = lightning_children_offsets[0];
                    int root_ce = lightning_children_offsets[1];
                    int nc = root_ce - root_cs;
                    if (nc > 0) {
                        r_sample = lightning_children_flat[root_cs + (int)(xorshift32(&lightning_rng) % (unsigned)nc)];
                    }
                }
                if (r_sample == 0) {
                    // Degenerate: sampler emitted the virtual root (e.g., p_stop fires,
                    // but walk is at root). Shouldn't happen per guard above, but fall
                    // back to picking a random root-child for correctness.
                    int root_cs = lightning_children_offsets[0];
                    int root_ce = lightning_children_offsets[1];
                    int nc = root_ce - root_cs;
                    if (nc > 0) {
                        r_sample = lightning_children_flat[root_cs + (int)(xorshift32(&lightning_rng) % (unsigned)nc)];
                    }
                }

                // --- Collect training set for this sample ---
                // L1/L2/L3: training set is the sampled radix node + its descendants
                //   (a subtree rooted at r_sample). Existing behavior.
                // L4: training set is r_sample + its ancestors (the full root-to-leaf
                //   path through the sampled radix node). Every node in the path has
                //   its ancestors already in the path, so each node's attention sees
                //   the proper path prefix. This makes L4 statistically equivalent
                //   to SGD window training.
                int fill = 0;
                double mass_accum = 0.0;
                if (lightning.sampler == LightningSampler::L4_Path) {
                    // Walk r_sample → root, recording each non-root ancestor.
                    int cur = r_sample;
                    while (cur > 0) {
                        bfs_buf[fill++] = cur;
                        mass_accum += (double)trie.edge_mass[cur];
                        cur = trie.parents[cur];
                    }
                    // Reverse so path is root-child first (depth-ascending).
                    for (int a = 0, b = fill - 1; a < b; a++, b--) {
                        int t = bfs_buf[a]; bfs_buf[a] = bfs_buf[b]; bfs_buf[b] = t;
                    }
                } else {
                    // ancestors(r_sample) ∪ {r_sample} ∪ descendants(r_sample).
                    //
                    // Ancestors are prepended so that in the chunk loop their
                    // forward pass scatters fresh K/V (current weights) into
                    // the cache before any descendant query reads from those
                    // positions. Without this, the cache holds zeros (epoch-
                    // init memset) or stale K/V from prior steps, and the
                    // prefix portion of descendant attention reads garbage.
                    //
                    // CRITICAL: the count of prepended ancestors is recorded
                    // in subtree_n_anc[s]; the chunk-loss code zeros their
                    // mass_weight so they don't contribute first-order CE
                    // loss. (Naive variant — letting ancestors contribute to
                    // loss — over-trains shallow nodes that appear in many
                    // BFS sets, and PPL got worse: see git log apr 2026.)
                    // Ancestor weights still receive gradient via descendant
                    // attention's K/V grad path — that's the AGPT mechanism
                    // and remains intact.
                    int cur = trie.parents[r_sample];
                    int anc_start = fill;
                    while (cur > 0) {
                        bfs_buf[fill++] = cur;
                        mass_accum += (double)trie.edge_mass[cur];
                        cur = trie.parents[cur];
                    }
                    // Reverse ancestor segment to depth-ascending order
                    // (cosmetic; the endpoint-depth bucket sort below
                    // re-orders strictly by endpoint depth anyway, putting
                    // ancestors first and descendants last).
                    for (int a = anc_start, b = fill - 1; a < b; a++, b--) {
                        int t = bfs_buf[a]; bfs_buf[a] = bfs_buf[b]; bfs_buf[b] = t;
                    }
                    int n_anc_local = fill - anc_start;
                    subtree_n_anc[s] = n_anc_local;

                    // BFS: {r_sample} ∪ descendants(r_sample)
                    int head = fill;
                    bfs_buf[fill++] = r_sample;
                    mass_accum += (double)trie.edge_mass[r_sample];
                    while (head < fill) {
                        int cur_i = bfs_buf[head++];
                        int cs = lightning_children_offsets[cur_i];
                        int ce = lightning_children_offsets[cur_i + 1];
                        for (int j = cs; j < ce; j++) {
                            int child = lightning_children_flat[j];
                            bfs_buf[fill++] = child;
                            mass_accum += (double)trie.edge_mass[child];
                        }
                    }
                }
                lightning_subtree_mass[s] = mass_accum;

                // --- Endpoint-depth sort (bucket) ---
                int sz = fill;
                int max_ep = 0;
                for (int a = 0; a < sz; a++) {
                    int ep = trie.edge_first_char_depths[bfs_buf[a]] + trie.edge_lens[bfs_buf[a]] - 1;
                    if (ep > max_ep) max_ep = ep;
                }
                int* node_arr = (int*)malloc(sz * sizeof(int));
                int* bucket_counts = (int*)calloc(max_ep + 2, sizeof(int));
                for (int a = 0; a < sz; a++) {
                    int ep = trie.edge_first_char_depths[bfs_buf[a]] + trie.edge_lens[bfs_buf[a]] - 1;
                    bucket_counts[ep + 1]++;
                }
                for (int e = 0; e < max_ep + 1; e++) bucket_counts[e + 1] += bucket_counts[e];
                int* cursors = (int*)calloc(max_ep + 2, sizeof(int));
                for (int a = 0; a < sz; a++) {
                    int ep = trie.edge_first_char_depths[bfs_buf[a]] + trie.edge_lens[bfs_buf[a]] - 1;
                    node_arr[bucket_counts[ep] + cursors[ep]++] = bfs_buf[a];
                }
                free(bucket_counts); free(cursors);
                subtree_nodes[s] = node_arr;
                subtree_sizes[s] = sz;
                lightning_nodes_sum += sz;

                // Log sampled-node start depth (endpoint_depth of r_sample) into histogram.
                int sample_ep = trie.edge_first_char_depths[r_sample] + trie.edge_lens[r_sample] - 1;
                if (sample_ep < 0) sample_ep = 0;
                if (sample_ep >= 64) sample_ep = 63;
                lightning_depth_hist[sample_ep]++;
            }
            free(bfs_buf);

            // Compute raw mass stats (for logging), then compress + normalize
            // the per-sample weights so the step-LR multiplier at optimizer
            // time is w[s] = compress(mass[s]) / mean(compress(mass)).
            // Mean-normalization preserves overall training budget: average
            // weight across an epoch = 1.0.
            double total_raw = 0.0;
            lightning_mass_min = (n_root_children > 0) ? lightning_subtree_mass[0] : 0.0;
            lightning_mass_max = lightning_mass_min;
            for (int s = 0; s < n_root_children; s++) {
                double m = lightning_subtree_mass[s];
                total_raw += m;
                if (m < lightning_mass_min) lightning_mass_min = m;
                if (m > lightning_mass_max) lightning_mass_max = m;
            }
            lightning_mean_mass = (n_root_children > 0 && total_raw > 0.0)
                                  ? total_raw / n_root_children : 1.0;

            // Honest per-step cost stats: BFS node count (sz).
            if (n_root_children > 0) {
                lightning_size_min = subtree_sizes[0];
                lightning_size_max = subtree_sizes[0];
                long long total_sz = 0;
                for (int s = 0; s < n_root_children; s++) {
                    int sz_s = subtree_sizes[s];
                    total_sz += sz_s;
                    if (sz_s < lightning_size_min) lightning_size_min = sz_s;
                    if (sz_s > lightning_size_max) lightning_size_max = sz_s;
                }
                lightning_size_mean = (double)total_sz / (double)n_root_children;
            }

            if (lightning.mass_lr != MassWeightMode::Off) {
                // In-place: replace subtree_mass[s] with the normalized weight
                // used at optimizer time. compress(m) then divide by its mean.
                double total_compressed = 0.0;
                for (int s = 0; s < n_root_children; s++) {
                    double m = lightning_subtree_mass[s];
                    double c;
                    switch (lightning.mass_lr) {
                        case MassWeightMode::Log:    c = log(1.0 + m);             break;
                        case MassWeightMode::Sqrt:   c = (m > 0.0) ? sqrt(m) : 0.0; break;
                        case MassWeightMode::Linear: c = m;                         break;
                        default:                     c = 1.0;                       break;
                    }
                    lightning_subtree_mass[s] = c;  // will be divided by mean below
                    total_compressed += c;
                }
                double mean_c = (n_root_children > 0 && total_compressed > 0.0)
                                ? total_compressed / n_root_children : 1.0;
                lightning_w_min = (n_root_children > 0) ? lightning_subtree_mass[0] / mean_c : 1.0;
                lightning_w_max = lightning_w_min;
                for (int s = 0; s < n_root_children; s++) {
                    lightning_subtree_mass[s] /= mean_c;
                    if (lightning_subtree_mass[s] < lightning_w_min) lightning_w_min = lightning_subtree_mass[s];
                    if (lightning_subtree_mass[s] > lightning_w_max) lightning_w_max = lightning_subtree_mass[s];
                }
            }
        }

        double total_loss = 0.0;
        int nodes_trained = 0;
        int chunks_processed = 0;
        int mb_groups_in_flight = 0;  // counter for --mini-batch-groups: number of partition groups
                                       // whose gradients are currently accumulated since the last
                                       // optimizer step (or since epoch start for the first batch).

        // Curriculum loop. Flat: one pass at d=max. Progressive: d=1, then d=2, ..., d=max.
        int subtrees_trained = 0;
        int curriculum_d_start = (curriculum == CurriculumMode::Progressive) ? 1 : curriculum_max_depth;
        int curriculum_d_end = curriculum_max_depth;

        // --accumulate mode: zero gradients ONCE at the top of the epoch so
        // all splits + partition groups + root-children accumulate into the
        // same buffer. A single optimizer step fires after all loops below.
        if (accumulate) {
            CUDA_CHECK(cudaMemset(d_grads, 0, wo.total_floats * sizeof(float)));
        }

        // --- Per-subtree residual measurement (notes/measure-loss-during-training.md) ---
        // During training, aggregate per-query loss into per-subtree sums.
        // At epoch end we print a "residual" ranking = count * max(avg_loss - global_avg, 0)
        // to identify hotspot subtrees that still need refinement.
        double* subtree_loss_sum = (double*)calloc(n_root_children, sizeof(double));
        long long* subtree_tokens = (long long*)calloc(n_root_children, sizeof(long long));
        long long* subtree_mass   = (long long*)calloc(n_root_children, sizeof(long long));
        for (int rc = 0; rc < n_root_children; rc++) {
            long long m = 0;
            int* arr = subtree_nodes[rc];
            int sz = subtree_sizes[rc];
            for (int a = 0; a < sz; a++) m += (long long)trie.edge_mass[arr[a]];
            subtree_mass[rc] = m;
        }

        // --- Per-subtree LR multipliers (cfg.lr_rule) ---
        // Computed once at epoch start. Applied at per-subtree optimizer step.
        // Mean-normalized so the average multiplier is 1.0 (preserves overall
        // LR schedule magnitude).
        double* subtree_lr_mult = (double*)malloc((n_root_children > 0 ? n_root_children : 1) * sizeof(double));
        if (cfg.lr_rule == 0) {
            // none: all 1.0
            for (int rc = 0; rc < n_root_children; rc++) subtree_lr_mult[rc] = 1.0;
        } else {
            double* raw = (double*)calloc(n_root_children, sizeof(double));
            long long tokens_sum = 0;
            for (int rc = 0; rc < n_root_children; rc++) tokens_sum += subtree_mass[rc];
            double mean_tokens = (n_root_children > 0) ? (double)tokens_sum / (double)n_root_children : 1.0;
            for (int rc = 0; rc < n_root_children; rc++) {
                int shallowest = INT_MAX;
                int sz = subtree_sizes[rc];
                for (int a = 0; a < sz; a++) {
                    int r = subtree_nodes[rc][a];
                    int d = trie.edge_first_char_depths[r];
                    if (d < shallowest) shallowest = d;
                }
                if (shallowest == INT_MAX) shallowest = 1;
                double m = (double)subtree_mass[rc];
                double v = 1.0;
                switch (cfg.lr_rule) {
                    case 1: // inv-depth
                        v = 1.0 / (double)shallowest;
                        break;
                    case 2: // inv-sqrt-depth
                        v = 1.0 / sqrt((double)shallowest);
                        break;
                    case 3: // sqrt-batch (sqrt(tokens / mean_tokens))
                        v = (mean_tokens > 0.0) ? sqrt(m / mean_tokens) : 1.0;
                        break;
                    case 4: // residual (prev epoch score / mean score). Falls back to 1 on first epoch or if mismatch.
                        v = 1.0;
                        if (prev_epoch_score && prev_epoch_n == n_root_children) {
                            double sum = 0.0;
                            for (int q = 0; q < prev_epoch_n; q++) sum += prev_epoch_score[q];
                            double mean_score = (prev_epoch_n > 0 && sum > 0.0) ? sum / (double)prev_epoch_n : 1.0;
                            v = (mean_score > 0.0) ? (prev_epoch_score[rc] / mean_score) : 1.0;
                            if (v <= 0.0) v = 1e-3;  // floor: a fully-converged subtree still gets tiny nonzero LR
                        }
                        break;
                }
                raw[rc] = v;
            }
            // Normalize to mean 1.0
            double sum = 0.0;
            for (int rc = 0; rc < n_root_children; rc++) sum += raw[rc];
            double mean = (n_root_children > 0 && sum > 0.0) ? sum / (double)n_root_children : 1.0;
            for (int rc = 0; rc < n_root_children; rc++) subtree_lr_mult[rc] = raw[rc] / mean;
            free(raw);
        }

        for (int curriculum_d = curriculum_d_start; curriculum_d <= curriculum_d_end; curriculum_d++) {
        // Iterate over root-child subtrees. Each subtree is one training unit:
        // weights fixed throughout forward+backward+grad-aggregation, one Adam step.
        for (int rc_idx = 0; rc_idx < n_root_children; rc_idx++) {
            // Subtree dropout: skip this rc if not in this epoch's keep set.
            if (subtree_keep != NULL && subtree_keep[rc_idx] == 0) continue;

            int* radix_list = subtree_nodes[rc_idx];
            int n_in_subtree;
            if (curriculum == CurriculumMode::Progressive) {
                // Restrict this subtree to nodes with endpoint_depth ≤ curriculum_d.
                // BFS-sorted list means prefix [0..depth_limit[rc_idx][curriculum_d+1]) qualifies.
                n_in_subtree = depth_limit[rc_idx][curriculum_d + 1];
            } else {
                n_in_subtree = subtree_sizes[rc_idx];
            }
            if (n_in_subtree == 0) continue;

            // --anc-grad: per-subtree-fire state init
            // Per the corrected design (see todo/descendant-ancestor-scatter.md),
            // accumulators are indexed by compact-cache character position, not
            // by radix node. Walk this subtree's mass>1 nodes; for each char in
            // each node, assign the next subtree-local index and write into the
            // lookup at the global compact-cache slot.
            int n_subtree_compact_chars = 0;  // declared here, used by later anc-grad
                                                // steps (forward save, backward scatter,
                                                // fire-end reduction). Counter is per fire.
            if (cfg.anc_grad) {
                static int* h_anc_lookup = NULL;
                static int* h_subtree_pos = NULL;
                if (!h_anc_lookup)  h_anc_lookup  = (int*)malloc((size_t)n_compact_chars * sizeof(int));
                if (!h_subtree_pos) h_subtree_pos = (int*)malloc((size_t)max_n_subtree_compact_chars * H * sizeof(int));
                memset(h_anc_lookup, 0xFF, (size_t)n_compact_chars * sizeof(int));  // -1 sentinel
                for (int i = 0; i < n_in_subtree; i++) {
                    int r = radix_list[i];
                    if (trie.edge_mass[r] == 1) continue;  // mass=1 cap: no cache slot
                    int start_pos = trie.edge_starts[r];
                    int len       = trie.edge_lens[r];
                    for (int c = 0; c < len; c++) {
                        int slot = compact_slot[start_pos + c];
                        if (slot >= 0) {
                            h_anc_lookup[slot] = n_subtree_compact_chars;
                            int pos = real_pos_of_char[start_pos + c];
                            // Replicate the same position across all heads so the
                            // existing launch_rope_batched_inverse — which reads
                            // positions[row] with row = slot*H + h — works directly.
                            for (int h = 0; h < H; h++) {
                                h_subtree_pos[n_subtree_compact_chars * H + h] = pos;
                            }
                            n_subtree_compact_chars++;
                        }
                    }
                }
                // NOTE: d_dkv_subtree_k/v and h_subtree are zeroed at split start
                // (inside the splits loop), not here. Each split is a separate Adam
                // fire and needs its own anc-grad accumulators. Per-subtree constants
                // (compact_to_subtree_idx, subtree_real_pos) stay uploaded here.
                CUDA_CHECK(cudaMemcpy(d_compact_to_subtree_idx, h_anc_lookup,
                                       (size_t)n_compact_chars * sizeof(int),
                                       cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_subtree_real_pos, h_subtree_pos,
                                       (size_t)n_subtree_compact_chars * H * sizeof(int),
                                       cudaMemcpyHostToDevice));
            }

        // Split this subtree into `subtree_splits` sub-batches. Each sub-batch
        // is a bounded training unit: its own d_grads zero, chunk-accumulated
        // forward/backward, one Adam step. Setting subtree_splits=1 preserves
        // the strict-invariant behavior (one update per root-child subtree).
        int actual_splits = (subtree_splits < n_in_subtree) ? subtree_splits : n_in_subtree;
        int split_base = n_in_subtree / actual_splits;
        int split_rem  = n_in_subtree % actual_splits;
        int subtree_offset = 0;
        for (int split_i = 0; split_i < actual_splits; split_i++) {
            int split_size = split_base + (split_i < split_rem ? 1 : 0);
            int n_at_depth = split_size;   // retained variable name used below

            // AGPT invariant: weights are fixed across all chunks of this sub-batch.
            // Zero gradients once at split start; accumulate across chunks.
            // In --accumulate mode, the zero happens once per-epoch above, not here.
            // With --mini-batch-groups K, zero only at the start of each K-group batch
            // (when no in-flight gradients from prior groups are accumulating).
            if (!accumulate && mb_groups_in_flight == 0) {
                CUDA_CHECK(cudaMemset(d_grads, 0, wo.total_floats * sizeof(float)));
            }

            // --anc-grad: per-split (= per Adam fire) accumulator zero. The actual
            // per-event normalization happens at scatter time via grad_scale =
            // 1/T_q_chunk, identical to own-edge's per-event weighting. No
            // per-fire counter or post-hoc divisor is needed here.
            if (cfg.anc_grad && n_subtree_compact_chars > 0) {
                size_t fire_bytes = (size_t)n_subtree_compact_chars * D * sizeof(float);
                for (int l = 0; l < L_layers; l++) {
                    CUDA_CHECK(cudaMemset(d_dkv_subtree_k[l], 0, fire_bytes));
                    CUDA_CHECK(cudaMemset(d_dkv_subtree_v[l], 0, fire_bytes));
                    CUDA_CHECK(cudaMemset(h_subtree[l],       0, fire_bytes));
                }
            }

            // Chunk by total queries ≤ CHUNK_QUERIES
            int chunk_start = 0;
            while (chunk_start < n_at_depth) {
                int chunk_cycle_shift = 0;
                if (lightning_active && lightning_cycle_shift && lightning.virtual_cycles > 1) {
                    chunk_cycle_shift = lightning_cycle_shift[rc_idx];
                }
                ChunkBuildContext chunk_ctx{
                    cfg,
                    trie,
                    radix_list,
                    subtree_offset,
                    n_at_depth,
                    chunk_start,
                    T_q_cap,
                    N_cap,
                    H,
                    branch_drop_mask,
                    chunk_cycle_shift,
                    real_pos_of_char,
                    T_kv_max
                };
                ChunkMetadata chunk_meta;
                if (!build_chunk_metadata(chunk_ctx, chunk_meta)) {
                    chunk_start = chunk_meta.next_chunk_start;
                    continue;
                }

                int chunk_end = chunk_meta.chunk_end;
                int N = chunk_meta.N;
                int T_q = chunk_meta.T_q;
                int T_kv = chunk_meta.T_kv;
                int max_kv_len = chunk_meta.max_kv_len;
                int* h_radix_ids = chunk_meta.h_radix_ids;
                int* h_query_offsets = chunk_meta.h_query_offsets;
                int* h_query_to_node = chunk_meta.h_query_to_node;
                int* h_char_pos = chunk_meta.h_char_pos;
                int* h_query_depth = chunk_meta.h_query_depth;

                ChunkDeviceMetadata device_chunk_meta =
                    upload_chunk_metadata_to_device(chunk_meta, chunk_upload_runtime);
                int* d_anc_lengths_cache = device_chunk_meta.d_anc_lengths;
                int* d_own_lengths_cache = device_chunk_meta.d_own_lengths;
                int* d_query_depth_cache = device_chunk_meta.d_query_depth;
                int* d_query_d_split_cache = device_chunk_meta.d_query_d_split;

                // Corpus-mass weighting. Raw edge_mass varies by 5+ orders of
                // magnitude in natural-language tries (common letters vs rare
                // prefixes). Directly weighting by mass/mean causes Adam to fail
                // because one high-mass sample dominates the update direction.
                //
                // We use log(1 + mass) compression, then normalize to mean 1 within
                // the chunk. This preserves ORDER (common > rare) while bounding
                // the per-step ratio to ~log(max_mass) / log(min_mass + 1) ≈ 10x.
                // Gradient stability wins out over exact linear-mass correspondence.
                // Naive warmup: ancestors are in the BFS and contribute to loss
                // exactly like any other position. This matches L4 (path-sampling)
                // semantics, where every position in the path is trained. The
                // masked variant (anc loss = 0) was based on an over-training
                // hypothesis that turned out to be wrong: L4 also has shallow
                // ancestors appearing in many paths and works fine (PPL 29 vs
                // L3-masked 59 on Gutenberg). See git log for the experiment.
                if (mass_weight != MassWeightMode::Off) {
                    float* h_mass_weights = (float*)malloc(T_q * sizeof(float));
                    if (joint_mass > 0 && (char_suffix_mass != NULL || trie.mean_edge_mass != NULL)) {
                        // Per-query joint weight: compress(edge_mass[r] * suffix_factor).
                        // suffix_factor: per-position from char_suffix_mass[h_char_pos[q]] if
                        // table is loaded; otherwise aggregate mean_edge_mass[D_max - d_q].
                        int D_max = trie.depth_file_count;
                        double total_w = 0.0;
                        for (int q = 0; q < T_q; q++) {
                            int node_idx = h_query_to_node[q];
                            int r = h_radix_ids[node_idx];
                            double pref_m = (double)trie.edge_mass[r];
                            double suff_m;
                            if (char_suffix_mass != NULL) {
                                long long c = (long long)h_char_pos[q];
                                if (c >= 0 && c < char_suffix_mass_n) {
                                    suff_m = char_suffix_mass[c];
                                } else {
                                    suff_m = 1.0;  // fallback for out-of-range
                                }
                            } else {
                                int d_q = h_query_depth[q];
                                int comp_d = D_max - d_q;
                                if (comp_d < 0) comp_d = 0;
                                if (comp_d >= D_max) comp_d = D_max - 1;
                                suff_m = trie.mean_edge_mass[comp_d];
                            }
                            double joint = pref_m * suff_m;
                            float w;
                            switch (mass_weight) {
                                case MassWeightMode::Log:    w = (float)log(1.0 + joint); break;
                                case MassWeightMode::Sqrt:   w = (float)sqrt(joint);      break;
                                case MassWeightMode::Linear: w = (float)joint;             break;
                                default:                     w = 1.0f;                     break;
                            }
                            h_mass_weights[q] = w;
                            total_w += (double)w;
                        }
                        float mean_w = (T_q > 0) ? (float)(total_w / T_q) : 1.0f;
                        if (mean_w <= 0.0f) mean_w = 1.0f;
                        for (int q = 0; q < T_q; q++) h_mass_weights[q] /= mean_w;
                    } else {
                        // Standard per-node weight: compress(edge_mass[r]).
                        float* node_w = (float*)malloc(N * sizeof(float));
                        double total_w = 0.0;
                        for (int i = 0; i < N; i++) {
                            int r = h_radix_ids[i];
                            float count = (float)trie.edge_mass[r];
                            float w;
                            switch (mass_weight) {
                                case MassWeightMode::Log:    w = logf(1.0f + count); break;
                                case MassWeightMode::Sqrt:   w = sqrtf(count);       break;
                                case MassWeightMode::Linear: w = count;              break;
                                default:                     w = 1.0f;               break;
                            }
                            node_w[i] = w;
                            int L = trie.edge_lens[r];
                            total_w += (double)w * L;
                        }
                        float mean_w = (T_q > 0) ? (float)(total_w / T_q) : 1.0f;
                        if (mean_w <= 0.0f) mean_w = 1.0f;
                        for (int i = 0; i < N; i++) {
                            float w = node_w[i] / mean_w;
                            int q_start = h_query_offsets[i];
                            int q_end = h_query_offsets[i + 1];
                            for (int q = q_start; q < q_end; q++) h_mass_weights[q] = w;
                        }
                        free(node_w);
                    }
                    CUDA_CHECK(cudaMemcpy(d_mass_weights, h_mass_weights, T_q * sizeof(float), cudaMemcpyHostToDevice));
                    free(h_mass_weights);
                }
                bool need_mass_weights = (mass_weight != MassWeightMode::Off);

                // NOTE: gradients zeroed at subtree start; NOT per chunk.
                // This chunk's backward accumulates into d_grads (via +=).

                // ---------- FORWARD ----------

                // Embedding gather: d_x[T_q, D]
                cuda_embedding_gather(d_weights + wo.token_emb, d_token_ids, d_x, T_q, D);

                float alpha = 1.0f, beta_zero = 0.0f;
                for (int l = 0; l < L_layers; l++) {
                    float* W_qw = d_weights + wo.wq_w[l];
                    float* W_qb = d_weights + wo.wq_b[l];
                    float* W_kw = d_weights + wo.wk_w[l];
                    float* W_kb = d_weights + wo.wk_b[l];
                    float* W_vw = d_weights + wo.wv_w[l];
                    float* W_vb = d_weights + wo.wv_b[l];
                    float* W_ow = d_weights + wo.wo_w[l];
                    float* W_ob = d_weights + wo.wo_b[l];
                    float* G1   = d_weights + wo.ln1_gamma[l];
                    float* B1   = d_weights + wo.ln1_beta[l];
                    float* W_1w = d_weights + wo.l1_w[l];
                    float* W_1b = d_weights + wo.l1_b[l];
                    float* W_2w = d_weights + wo.l2_w[l];
                    float* W_2b = d_weights + wo.l2_b[l];
                    float* G2   = d_weights + wo.ln2_gamma[l];
                    float* B2   = d_weights + wo.ln2_beta[l];

                    // Save residual 1 input
                    CUDA_CHECK(cudaMemcpy(sv_x_res1[l], d_x, (long long)T_q * D * sizeof(float), cudaMemcpyDeviceToDevice));

                    // LN1
                    cuda_layer_norm_forward(d_x, d_ln_out, sv_ln1_norm[l], sv_ln1_std_inv[l], G1, B1, T_q, D);
                    CUDA_CHECK(cudaMemcpy(sv_ln1_out[l], d_ln_out, (long long)T_q * D * sizeof(float), cudaMemcpyDeviceToDevice));

                    // --anc-grad: stash ln1_out per compact-char into the subtree
                    // buffer. The fire-end chain rule needs ln1_out at each
                    // ancestor's position to compute dW_kw/dW_vw += d_dkv^T · ln1_out.
                    // Same write-once per slot regardless of how many descendants
                    // attend to that ancestor.
                    if (cfg.anc_grad) {
                        launch_save_ln1_to_subtree(d_ln_out, d_char_pos,
                                                    cache_runtime.d_compact_slot,
                                                    d_compact_to_subtree_idx,
                                                    h_subtree[l], T_q, D);
                    }

                    // Q/K/V
                    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, D, T_q, D,
                                              &alpha, W_qw, D, d_ln_out, D, &beta_zero, d_q, D));
                    cuda_bias_add(d_q, W_qb, T_q, D);
                    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, D, T_q, D,
                                              &alpha, W_kw, D, d_ln_out, D, &beta_zero, d_k, D));
                    cuda_bias_add(d_k, W_kb, T_q, D);
                    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, D, T_q, D,
                                              &alpha, W_vw, D, d_ln_out, D, &beta_zero, d_v, D));
                    cuda_bias_add(d_v, W_vb, T_q, D);

                    // RoPE
                    launch_rope_batched(d_q, d_rope_positions, d_rope_cos, d_rope_sin, T_q * H, HD);
                    launch_rope_batched(d_k, d_rope_positions, d_rope_cos, d_rope_sin, T_q * H, HD);

                    // Scatter K/V into compact cache (mass=1 char positions are
                    // skipped — they're never queried as ancestors).
                    TIME_K(t_us_scatter_fwd, {
                        scatter_layer_kv_to_cache(cache_runtime, l, d_k, d_v, d_char_pos, T_q, D);
                    });

                    // Virtual-tree shift: if this sample's cycle shift > 0, rotate
                    // Q and own-edge K by θ(shift) on top of real-position rotation.
                    // Cache is already scattered at real rotation; delta-RoPE gather
                    // handles ancestors. V has no RoPE; no change.
                    int current_shift = 0;
                    if (lightning_active && lightning_cycle_shift && lightning.virtual_cycles > 1) {
                        current_shift = lightning_cycle_shift[rc_idx];
                    }
                    if (current_shift > 0) {
                        launch_rope_batched_scalar(d_q, current_shift, d_rope_cos, d_rope_sin, T_q * H, HD);
                        launch_rope_batched_scalar(d_k, current_shift, d_rope_cos, d_rope_sin, T_q * H, HD);
                    }

                    // Save post-RoPE Q/K and V so backward can skip the Q/K/V
                    // matmuls + RoPE recompute. The save buffers are sized
                    // T_q_cap × D so any chunk fits.
                    CUDA_CHECK(cudaMemcpy(sv_q[l], d_q, (long long)T_q * D * sizeof(float), cudaMemcpyDeviceToDevice));
                    CUDA_CHECK(cudaMemcpy(sv_k[l], d_k, (long long)T_q * D * sizeof(float), cudaMemcpyDeviceToDevice));
                    CUDA_CHECK(cudaMemcpy(sv_v[l], d_v, (long long)T_q * D * sizeof(float), cudaMemcpyDeviceToDevice));

                    // Build packed prefix: [ancestors from cache | own-edge from fresh d_k/d_v]
                    // Ancestors: gather from compact cache (all ancestors are mass>1 → slot>=0).
                    // When virtual-tree training is active (K>1), K gather uses delta-RoPE so
                    // the same cache entry can serve a virtual read position; otherwise the
                    // plain gather is used (faster, no extra trig). V has no RoPE.
                    TIME_K(t_us_gather_fwd, {
                        gather_layer_packed_kv(cache_runtime, l, device_chunk_meta,
                                               d_query_offsets, d_kv_offsets,
                                               d_k, d_v,
                                               d_kv_pack_k, d_kv_pack_v,
                                               d_rope_cos, d_rope_sin,
                                               N, H, HD,
                                               lightning.virtual_cycles > 1);
                    });

                    // L-query varlen attention
                    float scale = 1.0f / sqrtf((float)HD);
                    TIME_K(t_us_attn_fwd, {
                        cuda_batched_varlen_attention_L_queries(
                            d_q, d_kv_pack_k, d_kv_pack_v,
                            d_query_to_node, d_query_offsets, d_kv_offsets, d_kv_lengths,
                            d_attn_out /* used as packed output temp */, sv_attn_weights[l],
                            T_q, H, HD, max_kv_len, scale);
                    });
                    // d_attn_out now has packed [T_q * H, HD] output — same memory layout as [T_q, D].
                    // Since D = H * HD and heads are contiguous in the last dim, this is already the
                    // right layout for [T_q, D]. Save for backward.
                    CUDA_CHECK(cudaMemcpy(sv_attn_out[l], d_attn_out, (long long)T_q * D * sizeof(float), cudaMemcpyDeviceToDevice));

                    // WO + residual
                    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, D, T_q, D,
                                              &alpha, W_ow, D, d_attn_out, D, &beta_zero, d_ff_out, D));
                    cuda_bias_add(d_ff_out, W_ob, T_q, D);
                    CUDA_CHECK(cudaMemcpy(d_x, sv_x_res1[l], (long long)T_q * D * sizeof(float), cudaMemcpyDeviceToDevice));
                    launch_elem_add(d_x, d_ff_out, T_q * D);

                    // Save x_res2
                    CUDA_CHECK(cudaMemcpy(sv_x_res2[l], d_x, (long long)T_q * D * sizeof(float), cudaMemcpyDeviceToDevice));

                    // LN2
                    cuda_layer_norm_forward(d_x, d_ln_out, sv_ln2_norm[l], sv_ln2_std_inv[l], G2, B2, T_q, D);
                    CUDA_CHECK(cudaMemcpy(sv_ln2_out[l], d_ln_out, (long long)T_q * D * sizeof(float), cudaMemcpyDeviceToDevice));

                    // FFN
                    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, F, T_q, D,
                                              &alpha, W_1w, F, d_ln_out, D, &beta_zero, d_ff_h, F));
                    cuda_fused_bias_relu(d_ff_h, W_1b, d_ff_h, sv_ff_mask[l], T_q, F);
                    CUDA_CHECK(cudaMemcpy(sv_ff_h[l], d_ff_h, (long long)T_q * F * sizeof(float), cudaMemcpyDeviceToDevice));
                    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, D, T_q, F,
                                              &alpha, W_2w, D, d_ff_h, F, &beta_zero, d_ff_out, D));
                    cuda_bias_add(d_ff_out, W_2b, T_q, D);
                    CUDA_CHECK(cudaMemcpy(d_x, sv_x_res2[l], (long long)T_q * D * sizeof(float), cudaMemcpyDeviceToDevice));
                    launch_elem_add(d_x, d_ff_out, T_q * D);
                }

                // AGPT mass conservation: apply final LN + output proj over ALL T_q
                // positions (not just endpoints). This gives every edge character its
                // own supervision signal — radix edge ABC with L=3 contributes exactly
                // 3 loss terms (one per position), matching non-radix A→B→C semantics.
                //
                // TODO (corpus-mass weighting): the current loss normalizes each trie
                // node's CE by its total count, so every trie node contributes one
                // unit-weight gradient regardless of how many corpus events it
                // represents. This is AGPT's "equal-weight per node" semantic — good
                // for rare-pattern coverage, bad for fast fitting of high-frequency
                // patterns. To switch to corpus-mass weighting (each trie node's
                // gradient scaled by its count), the radix trie's head-of-edge count
                // (= count of the first original-trie node in the edge, NOT the
                // possibly-truncated endpoint count) must be stored in the binary
                // format. Mass conservation along a unary chain guarantees this head
                // count equals the true mass flowing through every position in the
                // edge; using the endpoint count would undercount due to seq_len
                // cutoff artifacts. This is a deliberate future option, not a bug.
                float* G_fn = d_weights + wo.final_gamma;
                float* B_fn = d_weights + wo.final_beta;
                float* W_out = d_weights + wo.out_w;
                float* B_out = d_weights + wo.out_b;
                // Final LN over all T_q positions — write into d_final_out buffer
                // (which is sized to T_q_cap already).
                cuda_layer_norm_forward(d_x, d_final_out, d_final_norm_save, d_final_std_inv_save, G_fn, B_fn, T_q, D);
                // Output projection: d_final_out[T_q, D] × W_out[D, V] → d_logits[T_q, V]
                CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, V, T_q, D,
                                          &alpha, W_out, V, d_final_out, D, &beta_zero, d_logits, V));
                cuda_bias_add(d_logits, B_out, T_q, V);

                // Per-query loss: intermediate positions = single-target CE, endpoints
                // = distribution CE (KL). d_d_logits (per-query grad) written in place.
                // --ce-only forces endpoints to single-target CE too (SGD-semantic).
                launch_agpt_loss_per_query(d_logits, d_query_to_node, d_query_offsets,
                                            d_radix_ids, d_token_ids,
                                            d_radix_counts_offset, d_radix_counts_tok, d_radix_counts_val,
                                            need_mass_weights ? d_mass_weights : NULL,
                                            g_d_fold_offsets, g_d_fold_lengths,
                                            g_d_fold_tokens, g_d_fold_probs,
                                            g_d_vtree_offsets, g_d_vtree_lengths,
                                            g_d_vtree_tokens, g_d_vtree_probs,
                                            g_vtree_expansion_depth,
                                            d_d_logits, d_loss, T_q, V, entropy_lambda,
                                            intermediate_weight, cfg.ce_only ? 1 : 0);

                if (decision_only > 0) {
                    // Zero loss + grad for queries past d_split + buffer.
                    launch_mask_loss_decision_only(d_loss, d_d_logits,
                                                    d_query_depth_cache, d_query_d_split_cache,
                                                    decision_buffer, T_q, V);
                }

                float* h_loss = (float*)malloc(T_q * sizeof(float));
                CUDA_CHECK(cudaMemcpy(h_loss, d_loss, T_q * sizeof(float), cudaMemcpyDeviceToHost));
                int chunk_trained = 0;
                double chunk_loss_sum = 0.0;
                for (int i = 0; i < T_q; i++) if (h_loss[i] > 0.0f) {
                    total_loss += h_loss[i];
                    chunk_loss_sum += h_loss[i];
                    chunk_trained++;
                }
                subtree_loss_sum[rc_idx] += chunk_loss_sum;
                subtree_tokens[rc_idx]   += chunk_trained;
                nodes_trained += chunk_trained;
                free(h_loss);

                // ---------- BACKWARD ----------
                // Scale by 1/chunk_trained where chunk_trained is now the number of
                // per-query loss terms (≈ T_q, not N).
                float grad_scale = (chunk_trained > 0) ? (1.0f / (float)chunk_trained) : 0.0f;

                // Output projection backward — all T_q rows.
                float* dG_out = d_grads + wo.out_w;
                float* dB_out_g = d_grads + wo.out_b;
                // d_d_final_out[T_q, D] = d_d_logits[T_q, V] × W_out^T[V, D]
                CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, D, T_q, V,
                                          &alpha, W_out, V, d_d_logits, V, &beta_zero, d_d_final_out, D));
                // dW_out += d_final_out^T × d_d_logits (scaled); dB_out += col-sum(d_d_logits)
                CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, V, D, T_q,
                                          &grad_scale, d_d_logits, V, d_final_out, D, &alpha, dG_out, V));
                launch_bias_grad_accum(d_d_logits, T_q, V, grad_scale, dB_out_g);

                // Final LN backward over all T_q rows.
                float* dG_fn = d_grads + wo.final_gamma;
                float* dB_fn = d_grads + wo.final_beta;
                cuda_layer_norm_backward(d_d_final_out, d_final_norm_save, d_final_std_inv_save,
                                          G_fn, d_d_final_out, dG_fn, dB_fn, T_q, D);

                // d_x (reused as d_dx) receives the gradient directly at every position.
                // No scatter needed — gradient is dense across T_q positions.
                CUDA_CHECK(cudaMemcpy(d_x, d_d_final_out, (long long)T_q * D * sizeof(float), cudaMemcpyDeviceToDevice));

                // Per-layer backward (reverse)
                float* d_dx = d_x; // alias
                float* d_d_ff_out = d_ff_out; // reuse as dY for FFN backward
                float* d_d_ln_out = d_ln_out; // reuse
                float* d_d_ff_h = d_ff_h;     // reuse
                float* d_d_attn_out = d_attn_out; // reuse

                for (int l = L_layers - 1; l >= 0; l--) {
                    float* W_ow = d_weights + wo.wo_w[l];
                    float* W_1w = d_weights + wo.l1_w[l];
                    float* W_2w = d_weights + wo.l2_w[l];
                    float* G1   = d_weights + wo.ln1_gamma[l];
                    float* G2   = d_weights + wo.ln2_gamma[l];

                    float* dW_ow = d_grads + wo.wo_w[l];
                    float* dW_1w = d_grads + wo.l1_w[l];
                    float* dW_2w = d_grads + wo.l2_w[l];
                    float* dG1   = d_grads + wo.ln1_gamma[l]; float* dB1 = d_grads + wo.ln1_beta[l];
                    float* dG2   = d_grads + wo.ln2_gamma[l]; float* dB2 = d_grads + wo.ln2_beta[l];
                    float* dW_qw = d_grads + wo.wq_w[l];

                    // d_x split at residual 2: one branch through FFN, skip added later
                    CUDA_CHECK(cudaMemcpy(d_d_ff_out, d_dx, (long long)T_q * D * sizeof(float), cudaMemcpyDeviceToDevice));

                    // FFN L2 backward: d_ff_out → d_ff_h;  dW_2 += ff_h^T × d_ff_out;
                    //                                       db_2 += col-sum(d_ff_out).
                    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, F, T_q, D,
                                              &alpha, W_2w, D, d_d_ff_out, D, &beta_zero, d_d_ff_h, F));
                    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, D, F, T_q,
                                              &grad_scale, d_d_ff_out, D, sv_ff_h[l], F, &alpha, dW_2w, D));
                    {
                        float* dW_2b = d_grads + wo.l2_b[l];
                        launch_bias_grad_accum(d_d_ff_out, T_q, D, grad_scale, dW_2b);
                    }

                    // ReLU backward
                    cuda_relu_backward(d_d_ff_h, sv_ff_mask[l], d_d_ff_h, T_q * F);

                    // FFN L1 backward: d_ff_h → d_ln2_out;  dW_1 += ln2_out^T × d_ff_h;
                    //                                        db_1 += col-sum(d_ff_h).
                    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, D, T_q, F,
                                              &alpha, W_1w, F, d_d_ff_h, F, &beta_zero, d_d_ln_out, D));
                    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, F, D, T_q,
                                              &grad_scale, d_d_ff_h, F, sv_ln2_out[l], D, &alpha, dW_1w, F));
                    {
                        float* dW_1b = d_grads + wo.l1_b[l];
                        launch_bias_grad_accum(d_d_ff_h, T_q, F, grad_scale, dW_1b);
                    }

                    // LN2 backward
                    cuda_layer_norm_backward(d_d_ln_out, sv_ln2_norm[l], sv_ln2_std_inv[l],
                                              G2, d_d_ln_out, dG2, dB2, T_q, D);
                    launch_elem_add(d_dx, d_d_ln_out, T_q * D);  // residual 2 skip

                    // WO backward: d_dx → d_attn_out;  dW_o += attn_out^T × d_dx;
                    //                                   db_o += col-sum(d_dx).
                    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, D, T_q, D,
                                              &alpha, W_ow, D, d_dx, D, &beta_zero, d_d_attn_out, D));
                    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, D, D, T_q,
                                              &grad_scale, d_dx, D, sv_attn_out[l], D, &alpha, dW_ow, D));
                    {
                        float* dW_ob = d_grads + wo.wo_b[l];
                        launch_bias_grad_accum(d_dx, T_q, D, grad_scale, dW_ob);
                    }

                    // Attention backward (L-queries)
                    // Restore post-RoPE Q/K and V from forward-side save buffers.
                    // Skips ~3 matmuls + 2 RoPE applies per layer per chunk that
                    // the previous implementation did via recompute from sv_ln1_out.
                    // d_q, d_k, d_v are reused as scratch for the attention-backward
                    // kernel's own K/V input, so we need them in the d_* buffers.
                    CUDA_CHECK(cudaMemcpy(d_q, sv_q[l], (long long)T_q * D * sizeof(float), cudaMemcpyDeviceToDevice));
                    CUDA_CHECK(cudaMemcpy(d_k, sv_k[l], (long long)T_q * D * sizeof(float), cudaMemcpyDeviceToDevice));
                    CUDA_CHECK(cudaMemcpy(d_v, sv_v[l], (long long)T_q * D * sizeof(float), cudaMemcpyDeviceToDevice));

                    // Gather packed K/V for backward: ancestors from compact cache,
                    // own-edge from freshly-recomputed d_k/d_v. Delta-RoPE K gather
                    // only when virtual-tree mode is active.
                    TIME_K(t_us_gather_bwd, {
                        gather_layer_packed_kv(cache_runtime, l, device_chunk_meta,
                                               d_query_offsets, d_kv_offsets,
                                               d_k, d_v,
                                               d_kv_pack_k, d_kv_pack_v,
                                               d_rope_cos, d_rope_sin,
                                               N, H, HD,
                                               lightning.virtual_cycles > 1);
                    });

                    // Zero dK/dV packed buffers
                    CUDA_CHECK(cudaMemset(d_dk_pack, 0, (long long)T_kv * H * HD * sizeof(float)));
                    CUDA_CHECK(cudaMemset(d_dv_pack, 0, (long long)T_kv * H * HD * sizeof(float)));

                    float scale = 1.0f / sqrtf((float)HD);
                    TIME_K(t_us_attn_bwd, {
                        cuda_batched_varlen_attention_L_queries_backward(
                            d_q, d_kv_pack_k, d_kv_pack_v, sv_attn_weights[l], d_d_attn_out,
                            d_query_to_node, d_query_offsets, d_kv_offsets, d_kv_lengths,
                            d_dq_pack, d_dk_pack, d_dv_pack,
                            T_q, H, HD, max_kv_len, scale);
                    });

                    // --anc-grad: scatter ancestor-slice of d_dk_pack/d_dv_pack into
                    // the subtree-scoped accumulator. Each ancestor slot's gradient
                    // (one slot per descendant→ancestor read during attention) is
                    // atomic-added at the ancestor's compact-cache subtree index.
                    // Gradient stays POST-RoPE here; the fire-end chain rule applies
                    // RoPE-inverse on the accumulator using d_subtree_real_pos.
                    if (cfg.anc_grad) {
                        // Pre-scale by 1/T_q_chunk so anc events ride the same
                        // per-event weight as own-edge events from this chunk.
                        launch_scatter_anc_dkv_to_subtree(d_dk_pack,
                                                          device_chunk_meta.d_anc_ids,
                                                          device_chunk_meta.d_anc_offsets,
                                                          d_kv_offsets,
                                                          d_anc_lengths_cache,
                                                          cache_runtime.d_compact_slot,
                                                          d_compact_to_subtree_idx,
                                                          d_dkv_subtree_k[l],
                                                          grad_scale,
                                                          N, H, HD);
                        launch_scatter_anc_dkv_to_subtree(d_dv_pack,
                                                          device_chunk_meta.d_anc_ids,
                                                          device_chunk_meta.d_anc_offsets,
                                                          d_kv_offsets,
                                                          d_anc_lengths_cache,
                                                          cache_runtime.d_compact_slot,
                                                          d_compact_to_subtree_idx,
                                                          d_dkv_subtree_v[l],
                                                          grad_scale,
                                                          N, H, HD);
                    }

                    // Inverse RoPE on dQ. Reverse order of forward composition:
                    // Q was rotated first by real position, then by scalar shift.
                    // Undo shift first, then real-position rotation.
                    if (chunk_cycle_shift > 0) {
                        launch_rope_batched_scalar_inverse(d_dq_pack, chunk_cycle_shift, d_rope_cos, d_rope_sin, T_q * H, HD);
                    }
                    launch_rope_batched_inverse(d_dq_pack, d_rope_positions, d_rope_cos, d_rope_sin, T_q * H, HD);

                    // dQ → d_ln1_out via Wq^T; dWq += ln1_out^T × dQ;
                    //                            dwq_b += col-sum(dQ, scaled).
                    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, D, T_q, D,
                                              &alpha, d_weights + wo.wq_w[l], D,
                                              d_dq_pack, D, &beta_zero, d_d_ln_out, D));
                    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, D, D, T_q,
                                              &grad_scale, d_dq_pack, D,
                                              sv_ln1_out[l], D, &alpha, dW_qw, D));
                    {
                        float* dW_qb = d_grads + wo.wq_b[l];
                        launch_bias_grad_accum(d_dq_pack, T_q, D, grad_scale, dW_qb);
                    }

                    // --- K/V path backward (previously missing: Wk/Wv/biases frozen) ---
                    // Extract own-edge portion of packed dK/dV into [T_q, D] layout.
                    // Ancestor-portion dK/dV is still dropped (cross-chunk scatter-add
                    // into the compact cache is not implemented). Own-edge dK/dV is
                    // what feeds Wk/Wv gradients for the current chunk's positions.
                    CUDA_CHECK(cudaMemset(d_dk_own, 0, (long long)T_q * D * sizeof(float)));
                    CUDA_CHECK(cudaMemset(d_dv_own, 0, (long long)T_q * D * sizeof(float)));
                    launch_kv_uncopy_own_edge(d_dk_pack, d_query_offsets, d_kv_offsets,
                                               d_anc_lengths_cache, d_own_lengths_cache,
                                               d_dk_own, N, H, HD);
                    launch_kv_uncopy_own_edge(d_dv_pack, d_query_offsets, d_kv_offsets,
                                               d_anc_lengths_cache, d_own_lengths_cache,
                                               d_dv_own, N, H, HD);

                    // Inverse RoPE on dK (V has no RoPE). Match forward composition:
                    // scalar shift (if any) then real-position rotation.
                    if (chunk_cycle_shift > 0) {
                        launch_rope_batched_scalar_inverse(d_dk_own, chunk_cycle_shift, d_rope_cos, d_rope_sin, T_q * H, HD);
                    }
                    launch_rope_batched_inverse(d_dk_own, d_rope_positions, d_rope_cos, d_rope_sin, T_q * H, HD);

                    // dK → d_d_ln_out += d_dk_own × Wk^T;  dWk += ln1_out^T × d_dk_own;
                    //                                      dwk_b += col-sum(d_dk_own)
                    // dlnout propagation uses the FULL gradient (no depth routing) —
                    // we only route the *weight* gradient, not the path through Wk to
                    // the input embedding stack.
                    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, D, T_q, D,
                                              &alpha, d_weights + wo.wk_w[l], D,
                                              d_dk_own, D, &alpha, d_d_ln_out, D));
                    if (depth_route_perleaf > 0) {
                        // Per-leaf d* routing: each query's threshold is its node's d_split.
                        launch_mask_grad_by_query_dsplit(d_dk_own, d_query_depth_cache,
                                                          d_query_d_split_cache, /*mode=*/0, T_q, D);
                    } else if (depth_route_k > 0) {
                        // Zero queries whose char depth > d_k → only shallow events
                        // contribute to dWk (matches K = decision-zone projection).
                        launch_mask_grad_by_query_depth(d_dk_own, d_query_depth_cache,
                                                         depth_route_k, /*mode=*/0, T_q, D);
                    }
                    {
                        float* dW_kw = d_grads + wo.wk_w[l];
                        float* dW_kb = d_grads + wo.wk_b[l];
                        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, D, D, T_q,
                                                  &grad_scale, d_dk_own, D,
                                                  sv_ln1_out[l], D, &alpha, dW_kw, D));
                        launch_bias_grad_accum(d_dk_own, T_q, D, grad_scale, dW_kb);
                    }

                    // dV → d_d_ln_out += d_dv_own × Wv^T;  dWv += ln1_out^T × d_dv_own;
                    //                                      dwv_b += col-sum(d_dv_own)
                    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, D, T_q, D,
                                              &alpha, d_weights + wo.wv_w[l], D,
                                              d_dv_own, D, &alpha, d_d_ln_out, D));
                    if (depth_route_perleaf > 0) {
                        launch_mask_grad_by_query_dsplit(d_dv_own, d_query_depth_cache,
                                                          d_query_d_split_cache, /*mode=*/1, T_q, D);
                    } else if (depth_route_k > 0) {
                        // Zero queries whose char depth ≤ d_k → only deep events
                        // contribute to dWv (matches V = identity-zone projection).
                        launch_mask_grad_by_query_depth(d_dv_own, d_query_depth_cache,
                                                         depth_route_k, /*mode=*/1, T_q, D);
                    }
                    {
                        float* dW_vw = d_grads + wo.wv_w[l];
                        float* dW_vb = d_grads + wo.wv_b[l];
                        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, D, D, T_q,
                                                  &grad_scale, d_dv_own, D,
                                                  sv_ln1_out[l], D, &alpha, dW_vw, D));
                        launch_bias_grad_accum(d_dv_own, T_q, D, grad_scale, dW_vb);
                    }

                    // LN1 backward
                    cuda_layer_norm_backward(d_d_ln_out, sv_ln1_norm[l], sv_ln1_std_inv[l],
                                              G1, d_d_ln_out, dG1, dB1, T_q, D);
                    launch_elem_add(d_dx, d_d_ln_out, T_q * D);  // residual 1 skip
                }

                // Embedding backward: scatter_add d_x into token_emb grad
                float* dG_emb = d_grads + wo.token_emb;
                cuda_embedding_scatter_add(d_dx, d_token_ids, dG_emb, T_q, D);

                // NOTE: no Adam step here — gradients accumulate in d_grads
                // across all chunks of this root-child subtree; one step at subtree end.

                free_chunk_metadata(chunk_meta);

                chunks_processed++;
                chunk_start = chunk_end;
            }  // end chunk loop — one subtree done

            // --anc-grad: fire-end chain-rule reduction.
            // Ancestor K/V gradients have been scatter-added into d_dkv_subtree_{k,v}[l]
            // already pre-scaled by 1/T_q_chunk at scatter time — matching own-edge's
            // per-event weighting exactly. h_subtree[l] holds ln1_out at each ancestor's
            // position (saved during forward). All we do here is RoPE-inverse the K
            // accumulator and chain-rule via cuBLAS with scalar 1.0 (no further scaling).
            //
            //   dW_kw[l] += d_dkv_subtree_k[l] · h_subtree[l]^T   (RoPE-inv first)
            //   dW_vw[l] += d_dkv_subtree_v[l] · h_subtree[l]^T   (V has no RoPE)
            //
            // No knob, no chunks_processed, no subtree_events — the per-event weight
            // already lives in the accumulator. Each anc event contributes the same
            // 1/T_q_chunk weight that own-edge would have given its own events.
            // Cf. todo/descendant-ancestor-scatter.md.
            if (cfg.anc_grad && n_subtree_compact_chars > 0) {
                int n_sub = n_subtree_compact_chars;
                float anc_alpha = 1.0f;  // pre-scaled at scatter; no extra factor
                float anc_one = 1.0f;    // beta=1 accumulate into existing dW
                for (int l = 0; l < L_layers; l++) {
                    // RoPE-inverse on K-grad (V has no RoPE). Treats buffer as
                    // (n_sub × H) rows of HD; positions[row] gives per-head pos.
                    launch_rope_batched_inverse(d_dkv_subtree_k[l], d_subtree_real_pos,
                                                d_rope_cos, d_rope_sin, n_sub * H, HD);

                    float* dW_kw = d_grads + wo.wk_w[l];
                    float* dW_vw = d_grads + wo.wv_w[l];
                    // dW_kw [D × D] += d_dkv_subtree_k [n_sub × D] · h_subtree [n_sub × D]^T
                    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, D, D, n_sub,
                                              &anc_alpha, d_dkv_subtree_k[l], D,
                                              h_subtree[l], D, &anc_one, dW_kw, D));
                    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, D, D, n_sub,
                                              &anc_alpha, d_dkv_subtree_v[l], D,
                                              h_subtree[l], D, &anc_one, dW_vw, D));
                }
            }

            // ONE Adam step per subtree. This is the AGPT training-unit boundary:
            // all descendant-branch gradients sharing a prefix inside this subtree
            // have been accumulated into d_grads (Jacobian factorization realized
            // via additive gradient accumulation).
            // In --accumulate mode, we skip the per-split/per-rc step and fire one
            // after all loops exit (see below).
            // With --mini-batch-groups K, fire the step every K groups instead of
            // every group; gradients keep accumulating for groups in between.
            mb_groups_in_flight++;
            bool fire_step = !accumulate && (mb_groups_in_flight >= cfg.mini_batch_groups);
            if (fire_step) {
                mb_groups_in_flight = 0;
            }
            if (fire_step) {
            adam_t++;
            // Apply LR schedule: `adam_t` counts total optimizer steps taken so
            // far (monotonically across epochs). We need total_steps to compute
            // cosine progress — estimate once at first step from structural info.
            // Compute total_opt_steps: prefer caller-supplied override (the
            // per-subtree wrapper knows the real horizon across subtrees), else
            // estimate from this call's structure. Rebuilt each step — cheap,
            // and avoids the static-variable trap of stale values across calls.
            int total_opt_steps_estimate;
            if (persist && persist->total_opt_steps_override > 0) {
                total_opt_steps_estimate = persist->total_opt_steps_override;
            } else if (persist && persist->total_epochs_override > 0) {
                int per_epoch = n_root_children * subtree_splits;
                if (curriculum == CurriculumMode::Progressive) per_epoch *= curriculum_max_depth;
                total_opt_steps_estimate = per_epoch * persist->total_epochs_override;
                if (total_opt_steps_estimate < 1) total_opt_steps_estimate = 1;
            } else {
                int per_epoch = n_root_children * subtree_splits;
                if (curriculum == CurriculumMode::Progressive) per_epoch *= curriculum_max_depth;
                total_opt_steps_estimate = per_epoch * epochs;
                if (total_opt_steps_estimate < 1) total_opt_steps_estimate = 1;
            }
            int warmup_steps;
            if (persist && persist->warmup_steps_override > 0) {
                warmup_steps = persist->warmup_steps_override;
            } else {
                warmup_steps = warmup_epochs * n_root_children * subtree_splits;
                if (curriculum == CurriculumMode::Progressive) warmup_steps *= curriculum_max_depth;
            }
            float step_lr = compute_lr(cfg.lr, adam_t - 1, total_opt_steps_estimate,
                                       warmup_steps, lr_schedule);

            // Lightning --mass-lr: each sample's step_lr is multiplied by its
            // precomputed normalized weight (compressed mass / compressed-mean).
            // High-mass samples move weights proportionally more; average weight
            // across the epoch is 1.0, so the LR schedule still controls nominal
            // magnitude. Weight was computed after resampling above.
            if (lightning_active && lightning.mass_lr != MassWeightMode::Off && lightning_subtree_mass) {
                step_lr *= (float)lightning_subtree_mass[rc_idx];
            }

            // --lr-rule: per-subtree LR multiplier (mean-normalized so average is 1.0).
            if (cfg.lr_rule != 0 && subtree_lr_mult) {
                step_lr *= (float)subtree_lr_mult[rc_idx];
            }

            // Grad clipping (applies to the accumulated chunk-gradient sum for this
            // subtree-split before the optimizer uses it).
            if (grad_clip_norm > 0.0f) {
                cuda_grad_clip_by_norm(d_grads, grad_clip_norm, wo.total_floats,
                                        d_clip_partials, d_clip_norm);
            }

            // Per-rc Adam state: redirect m/v pointers and step counter to
            // this rc's bucket when --per-rc-adam is on. Otherwise use the
            // shared global moments as before.
            float* opt_m = d_adam_m;
            float* opt_v = d_adam_v;
            int    opt_t = adam_t;
            if (cfg.per_rc_adam && d_adam_m_per_rc) {
                long long off = (long long)rc_idx * (long long)wo.total_floats;
                opt_m = d_adam_m_per_rc + off;
                opt_v = d_adam_v_per_rc + off;
                h_adam_t_per_rc[rc_idx]++;
                opt_t = h_adam_t_per_rc[rc_idx];
            }
            switch (optimizer) {
                case OptimizerKind::Adam:
                    cuda_adam_bulk(d_weights, d_grads, opt_m, opt_v,
                                    step_lr, momentum_beta, rmsprop_beta, 1e-8f,
                                    opt_t, wo.total_floats);
                    break;
                case OptimizerKind::SGD:
                    cuda_sgd_bulk(d_weights, d_grads, step_lr, wo.total_floats);
                    break;
                case OptimizerKind::Momentum:
                    cuda_momentum_bulk(d_weights, d_grads, opt_m,
                                       step_lr, momentum_beta, wo.total_floats);
                    break;
                case OptimizerKind::RMSProp:
                    cuda_rmsprop_bulk(d_weights, d_grads, opt_v,
                                      step_lr, rmsprop_beta, 1e-8f, wo.total_floats);
                    break;
                case OptimizerKind::LBFGS:
                    cuda_lbfgs_step(&lbfgs_state, cublas, d_weights, d_grads, step_lr);
                    break;
            }
            // Decoupled weight decay (applies after optimizer step — AdamW style
            // across all optimizers). lr is the scheduled lr so decay also decays.
            if (weight_decay > 0.0f) {
                cuda_weight_decay(d_weights, step_lr, weight_decay, wo.total_floats);
            }
            subtrees_trained++;
            }  // end !accumulate gate

            subtree_offset += split_size;
        }  // end subtree-splits loop
        }  // end root-child subtree loop
        }  // end curriculum loop

        // --accumulate: single optimizer step at the end of the epoch, after all
        // splits, partition groups, and root-children have contributed to d_grads.
        if (accumulate) {
            adam_t++;
            int total_opt_steps_estimate;
            if (persist && persist->total_opt_steps_override > 0) {
                total_opt_steps_estimate = persist->total_opt_steps_override;
            } else if (persist && persist->total_epochs_override > 0) {
                total_opt_steps_estimate = persist->total_epochs_override;
                if (total_opt_steps_estimate < 1) total_opt_steps_estimate = 1;
            } else {
                total_opt_steps_estimate = epochs;
                if (total_opt_steps_estimate < 1) total_opt_steps_estimate = 1;
            }
            int warmup_steps_acc;
            if (persist && persist->warmup_steps_override > 0) {
                warmup_steps_acc = persist->warmup_steps_override;
            } else {
                warmup_steps_acc = warmup_epochs;
            }
            float step_lr = compute_lr(cfg.lr, adam_t - 1, total_opt_steps_estimate,
                                       warmup_steps_acc, lr_schedule);
            if (grad_clip_norm > 0.0f) {
                cuda_grad_clip_by_norm(d_grads, grad_clip_norm, wo.total_floats,
                                        d_clip_partials, d_clip_norm);
            }
            switch (optimizer) {
                case OptimizerKind::Adam:
                    cuda_adam_bulk(d_weights, d_grads, d_adam_m, d_adam_v,
                                    step_lr, momentum_beta, rmsprop_beta, 1e-8f,
                                    adam_t, wo.total_floats);
                    break;
                case OptimizerKind::SGD:
                    cuda_sgd_bulk(d_weights, d_grads, step_lr, wo.total_floats);
                    break;
                case OptimizerKind::Momentum:
                    cuda_momentum_bulk(d_weights, d_grads, d_adam_m,
                                       step_lr, momentum_beta, wo.total_floats);
                    break;
                case OptimizerKind::RMSProp:
                    cuda_rmsprop_bulk(d_weights, d_grads, d_adam_v,
                                      step_lr, rmsprop_beta, 1e-8f, wo.total_floats);
                    break;
                case OptimizerKind::LBFGS:
                    cuda_lbfgs_step(&lbfgs_state, cublas, d_weights, d_grads, step_lr);
                    break;
            }
            if (weight_decay > 0.0f) {
                cuda_weight_decay(d_weights, step_lr, weight_decay, wo.total_floats);
            }
            subtrees_trained++;
        }

        clock_gettime(CLOCK_MONOTONIC, &t1);
        double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        float mean_loss = nodes_trained > 0 ? (float)(total_loss / nodes_trained) : 0.0f;
        // --- Compute residual score per subtree (in-training measurement) ---
        // score[rc] = subtree_mass[rc] * max(avg_loss[rc] - mean_loss, 0)
        // Computed always (needed both for diagnostic printing and for the
        // --hotspot-coverage splitter).
        double* score   = (double*)calloc(n_root_children > 0 ? n_root_children : 1, sizeof(double));
        double* avgloss = (double*)calloc(n_root_children > 0 ? n_root_children : 1, sizeof(double));
        int*    order   = (int*)malloc((n_root_children > 0 ? n_root_children : 1) * sizeof(int));
        double  total_excess = 0.0;
        for (int rc = 0; rc < n_root_children; rc++) {
            avgloss[rc] = (subtree_tokens[rc] > 0)
                ? subtree_loss_sum[rc] / (double)subtree_tokens[rc] : 0.0;
            double excess = avgloss[rc] - (double)mean_loss;
            if (excess < 0.0) excess = 0.0;
            score[rc] = (double)subtree_mass[rc] * excess;
            total_excess += score[rc];
            order[rc] = rc;
        }
        // Full sort by score descending — used for residual top-10 print and
        // hotspot split coverage selection. qsort is O(n log n); the prior
        // selection sort was O(n²) and at pd=6 (283k groups) burned ~6-7 min
        // per epoch (40B compares). qsort here is ~50ms regardless of n.
        {
            static const double* g_score_for_sort;  // file-scope handoff to comparator
            g_score_for_sort = score;
            auto cmp = [](const void* a, const void* b) -> int {
                int ia = *(const int*)a, ib = *(const int*)b;
                double sa = g_score_for_sort[ia], sb = g_score_for_sort[ib];
                if (sa > sb) return -1;
                if (sa < sb) return  1;
                return ia - ib;  // tiebreak on rc id for determinism
            };
            qsort(order, n_root_children, sizeof(int), cmp);
        }

        if (!quiet) {
            printf("Epoch %d: loss=%.6f  (%.2f sec, %d subtrees, %d chunks, %d nodes)\n",
                   epoch + 1, mean_loss, elapsed, subtrees_trained, chunks_processed, nodes_trained);
            if (t_enabled) {
                double total_us = t_us_gather_fwd + t_us_gather_bwd + t_us_attn_fwd + t_us_attn_bwd + t_us_scatter_fwd;
                printf("  [timing] gather_fwd=%.2fs gather_bwd=%.2fs attn_fwd=%.2fs attn_bwd=%.2fs scatter_fwd=%.2fs  measured_total=%.2fs / wall=%.2fs (%.0f%%)\n",
                       t_us_gather_fwd / 1e6, t_us_gather_bwd / 1e6,
                       t_us_attn_fwd   / 1e6, t_us_attn_bwd   / 1e6,
                       t_us_scatter_fwd / 1e6,
                       total_us / 1e6, elapsed, 100.0 * (total_us / 1e6) / elapsed);
            }

            if (n_root_children > 0) {
                int top_k = 10;
                if (top_k > n_root_children) top_k = n_root_children;
                printf("  residual top-%d (subtree_mass × max(avg_loss − %.3f, 0)):\n", top_k, (double)mean_loss);
                printf("    %-4s %-8s %-9s %-12s %-6s\n", "rc", "rootID", "mass", "avg_loss", "score");
                for (int i = 0; i < top_k; i++) {
                    int rc = order[i];
                    int root_r = (subtree_sizes[rc] > 0) ? subtree_nodes[rc][0] : -1;
                    printf("    %-4d %-8d %-9lld %-12.4f %-6.1f\n",
                           rc, root_r, subtree_mass[rc], avgloss[rc], score[rc]);
                }
                double total_mass = 0.0, top_mass = 0.0, top_excess = 0.0;
                for (int rc = 0; rc < n_root_children; rc++) total_mass += (double)subtree_mass[rc];
                for (int i = 0; i < top_k; i++) {
                    top_mass += (double)subtree_mass[order[i]];
                    top_excess += score[order[i]];
                }
                printf("    top-%d: %.1f%% of mass, %.1f%% of excess-loss\n",
                       top_k, 100.0 * top_mass / (total_mass > 0 ? total_mass : 1.0),
                       100.0 * top_excess / (total_excess > 0 ? total_excess : 1.0));
            }

            if (lightning_active) {
                printf("  lightning depth histogram (sample endpoint depth):");
                int max_seen = 0;
                for (int d = 0; d < 64; d++) if (lightning_depth_hist[d] > 0 && d > max_seen) max_seen = d;
                for (int d = 0; d <= max_seen; d++) {
                    if (lightning_depth_hist[d] > 0) printf(" d%d:%d", d, lightning_depth_hist[d]);
                }
                printf("  size[min=%d mean=%.0f max=%d]",
                       lightning_size_min, lightning_size_mean, lightning_size_max);
                if (lightning.mass_lr != MassWeightMode::Off) {
                    printf("  lr_scale[min=%.3f max=%.3f]", lightning_w_min, lightning_w_max);
                }
                printf("\n");
            }
        }

        // --- Adaptive hotspot split between epochs ---
        // When cfg.hotspot_coverage > 0 and we're not on the last epoch, select
        // subtrees covering that fraction of total excess loss (by score) and
        // split each one level: replace {R, descendants} with {R} +
        // {child_i, child_i descendants} for each child C_i of R. Rebuilds
        // subtree_nodes[] / subtree_sizes[], then re-sorts by shallowest
        // radix-node depth so parent-only entries are processed first next epoch
        // (required for the K/V cache ordering invariant).
        if (cfg.hotspot_coverage > 0.0f && epoch + 1 < epochs && n_root_children > 0 && total_excess > 0.0) {
            if (lightning_active) {
                if (!quiet) printf("  hotspot-coverage: skipped (Lightning resamples each epoch)\n");
            } else {
                double target = cfg.hotspot_coverage * total_excess;
                double acc = 0.0;
                int n_split = 0;
                while (n_split < n_root_children && acc < target) {
                    acc += score[order[n_split]];
                    n_split++;
                }
                // Mark which subtrees get split.
                char* split_mark = (char*)calloc(n_root_children, 1);
                for (int i = 0; i < n_split; i++) split_mark[order[i]] = 1;

                // Build new arrays. For each non-split subtree, copy as-is.
                // For each split subtree rooted at R = subtree_nodes[rc][0]:
                //   emit one entry [R] (parent only)
                //   emit one entry per child C_i of R: [C_i ∪ descendants of C_i]
                int new_cap = n_root_children * 2;  // grows as we split
                int** new_nodes = (int**)malloc(new_cap * sizeof(int*));
                int*  new_sizes = (int*)malloc(new_cap * sizeof(int));
                int   new_n = 0;

                // Scratch BFS buffer for descendant collection.
                int* bfs_buf = (int*)malloc(trie.radix_count * sizeof(int));

                int actual_split = 0;
                for (int rc = 0; rc < n_root_children; rc++) {
                    if (!split_mark[rc]) {
                        // Copy subtree as-is.
                        if (new_n >= new_cap) {
                            new_cap *= 2;
                            new_nodes = (int**)realloc(new_nodes, new_cap * sizeof(int*));
                            new_sizes = (int*)realloc(new_sizes, new_cap * sizeof(int));
                        }
                        int sz = subtree_sizes[rc];
                        int* copy = (int*)malloc((sz > 0 ? sz : 1) * sizeof(int));
                        memcpy(copy, subtree_nodes[rc], sz * sizeof(int));
                        new_nodes[new_n] = copy;
                        new_sizes[new_n] = sz;
                        new_n++;
                        continue;
                    }
                    // Split this subtree. Root is shallowest node (BFS index 0).
                    int R = subtree_nodes[rc][0];
                    int cs = lightning_children_offsets[R];
                    int ce = lightning_children_offsets[R + 1];
                    int nc = ce - cs;
                    if (nc == 0) {
                        // Cannot split a leaf. Keep as-is.
                        if (new_n >= new_cap) {
                            new_cap *= 2;
                            new_nodes = (int**)realloc(new_nodes, new_cap * sizeof(int*));
                            new_sizes = (int*)realloc(new_sizes, new_cap * sizeof(int));
                        }
                        int sz = subtree_sizes[rc];
                        int* copy = (int*)malloc((sz > 0 ? sz : 1) * sizeof(int));
                        memcpy(copy, subtree_nodes[rc], sz * sizeof(int));
                        new_nodes[new_n] = copy;
                        new_sizes[new_n] = sz;
                        new_n++;
                        continue;
                    }
                    // Emit parent-only entry first.
                    if (new_n >= new_cap) {
                        new_cap *= 2;
                        new_nodes = (int**)realloc(new_nodes, new_cap * sizeof(int*));
                        new_sizes = (int*)realloc(new_sizes, new_cap * sizeof(int));
                    }
                    int* parent_only = (int*)malloc(sizeof(int));
                    parent_only[0] = R;
                    new_nodes[new_n] = parent_only;
                    new_sizes[new_n] = 1;
                    new_n++;

                    // Emit one entry per child.
                    for (int j = cs; j < ce; j++) {
                        int C = lightning_children_flat[j];
                        // BFS from C to collect {C} ∪ descendants(C).
                        int fill = 0, head = 0;
                        bfs_buf[fill++] = C;
                        while (head < fill) {
                            int cur = bfs_buf[head++];
                            int ccs = lightning_children_offsets[cur];
                            int cce = lightning_children_offsets[cur + 1];
                            for (int k = ccs; k < cce; k++) {
                                bfs_buf[fill++] = lightning_children_flat[k];
                            }
                        }
                        // Endpoint-depth sort within child subtree.
                        int sz = fill;
                        int max_ep = 0;
                        for (int a = 0; a < sz; a++) {
                            int ep = trie.edge_first_char_depths[bfs_buf[a]] + trie.edge_lens[bfs_buf[a]] - 1;
                            if (ep > max_ep) max_ep = ep;
                        }
                        int* node_arr = (int*)malloc((sz > 0 ? sz : 1) * sizeof(int));
                        int* bucket_counts = (int*)calloc(max_ep + 2, sizeof(int));
                        for (int a = 0; a < sz; a++) {
                            int ep = trie.edge_first_char_depths[bfs_buf[a]] + trie.edge_lens[bfs_buf[a]] - 1;
                            bucket_counts[ep + 1]++;
                        }
                        for (int e = 0; e < max_ep + 1; e++) bucket_counts[e + 1] += bucket_counts[e];
                        int* cursors = (int*)calloc(max_ep + 2, sizeof(int));
                        for (int a = 0; a < sz; a++) {
                            int ep = trie.edge_first_char_depths[bfs_buf[a]] + trie.edge_lens[bfs_buf[a]] - 1;
                            node_arr[bucket_counts[ep] + cursors[ep]++] = bfs_buf[a];
                        }
                        free(bucket_counts); free(cursors);

                        if (new_n >= new_cap) {
                            new_cap *= 2;
                            new_nodes = (int**)realloc(new_nodes, new_cap * sizeof(int*));
                            new_sizes = (int*)realloc(new_sizes, new_cap * sizeof(int));
                        }
                        new_nodes[new_n] = node_arr;
                        new_sizes[new_n] = sz;
                        new_n++;
                    }
                    actual_split++;
                }
                free(bfs_buf);
                free(split_mark);

                // Re-sort the new subtree list by shallowest-node-depth
                // ascending. This preserves the K/V cache ordering invariant
                // (shallow subtrees scatter ancestor K/V before deeper subtrees
                // that would read it).
                int* sort_key = (int*)malloc(new_n * sizeof(int));
                int* sort_idx = (int*)malloc(new_n * sizeof(int));
                for (int i = 0; i < new_n; i++) {
                    int shallowest = INT_MAX;
                    for (int a = 0; a < new_sizes[i]; a++) {
                        int r = new_nodes[i][a];
                        int start_depth = trie.edge_first_char_depths[r];
                        if (start_depth < shallowest) shallowest = start_depth;
                    }
                    sort_key[i] = shallowest;
                    sort_idx[i] = i;
                }
                // Insertion sort (stable enough for 100s of entries).
                for (int i = 1; i < new_n; i++) {
                    int k = sort_key[sort_idx[i]];
                    int v = sort_idx[i];
                    int j = i - 1;
                    while (j >= 0 && sort_key[sort_idx[j]] > k) {
                        sort_idx[j + 1] = sort_idx[j]; j--;
                    }
                    sort_idx[j + 1] = v;
                }
                int** sorted_nodes = (int**)malloc(new_n * sizeof(int*));
                int*  sorted_sizes = (int*)malloc(new_n * sizeof(int));
                for (int i = 0; i < new_n; i++) {
                    sorted_nodes[i] = new_nodes[sort_idx[i]];
                    sorted_sizes[i] = new_sizes[sort_idx[i]];
                }
                free(sort_key); free(sort_idx);
                free(new_nodes); free(new_sizes);

                // Free the old subtree_nodes[] and install the new one.
                for (int i = 0; i < n_root_children; i++) free(subtree_nodes[i]);
                free(subtree_nodes);
                free(subtree_sizes);
                free(subtree_n_anc);
                subtree_nodes = sorted_nodes;
                subtree_sizes = sorted_sizes;
                // Hotspot-split is a deterministic-AGPT path; no ancestors prepended.
                subtree_n_anc = (int*)calloc(new_n, sizeof(int));
                int old_n = n_root_children;
                n_root_children = new_n;

                if (!quiet) {
                    printf("  hotspot-split: %d → %d subtrees (split %d hotspots covering %.1f%% of excess)\n",
                           old_n, new_n, actual_split, 100.0 * acc / total_excess);
                }
            }
        }

        // Stash this epoch's score for the next epoch's residual LR rule.
        if (prev_epoch_score) free(prev_epoch_score);
        prev_epoch_score = (double*)malloc((n_root_children > 0 ? n_root_children : 1) * sizeof(double));
        memcpy(prev_epoch_score, score, (n_root_children > 0 ? n_root_children : 1) * sizeof(double));
        prev_epoch_n = n_root_children;

        free(score); free(avgloss); free(order);
        if (subtree_lr_mult) { free(subtree_lr_mult); subtree_lr_mult = NULL; }

        // Intermediate checkpoint every save_every epochs. External tooling
        // (bin/perplexity) can score these to find the best-held-out stopping point.
        if (save_every > 0 && save_path && (epoch + 1) % save_every == 0) {
            char ck_path[2048];
            snprintf(ck_path, sizeof(ck_path), "%s.ep%d", save_path, epoch + 1);
            CUDA_CHECK(cudaMemcpy(h_weights, d_weights, wo.total_floats * sizeof(float), cudaMemcpyDeviceToHost));
            save_model_weights(ck_path, cfg, h_weights, wo);
        }

        if (lightning_subtree_mass) { free(lightning_subtree_mass); lightning_subtree_mass = NULL; }
        if (lightning_cycle_shift) { free(lightning_cycle_shift); lightning_cycle_shift = NULL; }
        free(subtree_loss_sum);
        free(subtree_tokens);
        free(subtree_mass);
    }

    // Save weights back to host.
    CUDA_CHECK(cudaMemcpy(h_weights, d_weights, wo.total_floats * sizeof(float), cudaMemcpyDeviceToHost));
    if (save_path) {
        save_model_weights(save_path, cfg, h_weights, wo);
        if (!quiet) printf("Saved to %s\n", save_path);
    }
    // Save optimizer state back to caller if requested.
    if (persist) {
        if (persist->h_adam_m_io) CUDA_CHECK(cudaMemcpy(persist->h_adam_m_io, d_adam_m, wo.total_floats * sizeof(float), cudaMemcpyDeviceToHost));
        if (persist->h_adam_v_io) CUDA_CHECK(cudaMemcpy(persist->h_adam_v_io, d_adam_v, wo.total_floats * sizeof(float), cudaMemcpyDeviceToHost));
        if (persist->adam_t_io)   *persist->adam_t_io = adam_t;
    }

    // --- GPU cleanup (required so the per-subtree wrapper can call us
    //     repeatedly without OOMing). Order mirrors allocation. ---
    // Dump per-rc v buffer for offline diagnostic analysis (per-bucket norms,
    // cosine similarities, etc.) before freeing.
    if (cfg.per_rc_v_dump_path && d_adam_v_per_rc) {
        size_t bytes = (size_t)n_root_children * (size_t)wo.total_floats * sizeof(float);
        float* h_v = (float*)malloc(bytes);
        if (h_v) {
            CUDA_CHECK(cudaMemcpy(h_v, d_adam_v_per_rc, bytes, cudaMemcpyDeviceToHost));
            FILE* fp = fopen(cfg.per_rc_v_dump_path, "wb");
            if (fp) {
                const char magic[4] = {'P','R','V','D'};
                fwrite(magic, 1, 4, fp);
                int n_rc = n_root_children;
                int tf   = wo.total_floats;
                fwrite(&n_rc, sizeof(int), 1, fp);
                fwrite(&tf,   sizeof(int), 1, fp);
                fwrite(h_adam_t_per_rc, sizeof(int), n_rc, fp);  // per-bucket step counts
                fwrite(h_v, sizeof(float), (size_t)n_rc * (size_t)tf, fp);
                fclose(fp);
                printf("  per-rc v dumped to %s (%d buckets x %d params)\n",
                       cfg.per_rc_v_dump_path, n_rc, tf);
            } else {
                fprintf(stderr, "WARN: could not open %s for write\n", cfg.per_rc_v_dump_path);
            }
            free(h_v);
        }
    }

    free_chunk_upload_runtime(chunk_upload_runtime);

    cudaFree(d_weights); cudaFree(d_grads); cudaFree(d_adam_m); cudaFree(d_adam_v);
    if (d_dkv_subtree_k) {
        for (int l = 0; l < L_layers; l++) {
            if (d_dkv_subtree_k[l]) cudaFree(d_dkv_subtree_k[l]);
            if (d_dkv_subtree_v[l]) cudaFree(d_dkv_subtree_v[l]);
            if (h_subtree[l])       cudaFree(h_subtree[l]);
        }
        free(d_dkv_subtree_k);
        free(d_dkv_subtree_v);
        free(h_subtree);
    }
    if (d_compact_to_subtree_idx) cudaFree(d_compact_to_subtree_idx);
    if (d_subtree_real_pos)       cudaFree(d_subtree_real_pos);
    if (d_adam_m_per_rc) cudaFree(d_adam_m_per_rc);
    if (d_adam_v_per_rc) cudaFree(d_adam_v_per_rc);
    if (h_adam_t_per_rc) free(h_adam_t_per_rc);
    if (d_clip_partials) cudaFree(d_clip_partials);
    if (d_clip_norm)     cudaFree(d_clip_norm);

    // L-BFGS state cleanup (allocated only when optimizer == LBFGS)
    if (lbfgs_state.d_g_prev) cudaFree(lbfgs_state.d_g_prev);
    if (lbfgs_state.d_step)   cudaFree(lbfgs_state.d_step);
    if (lbfgs_state.d_s_hist) cudaFree(lbfgs_state.d_s_hist);
    if (lbfgs_state.d_y_hist) cudaFree(lbfgs_state.d_y_hist);
    if (lbfgs_state.d_q)      cudaFree(lbfgs_state.d_q);
    if (lbfgs_state.rho_hist) free(lbfgs_state.rho_hist);
    if (lbfgs_state.alpha)    free(lbfgs_state.alpha);
    free_cache_runtime(cache_runtime);
    cudaFree(d_rope_cos); cudaFree(d_rope_sin);
    free_transformer_chunk_runtime(transformer_runtime);
    cudaFree(d_radix_counts_offset); cudaFree(d_radix_counts_tok); cudaFree(d_radix_counts_val);
    if (d_mass_weights) cudaFree(d_mass_weights);
    free(root_child_of); free(root_children);
    for (int i = 0; i < n_root_children; i++) free(subtree_nodes[i]);
    free(subtree_nodes); free(subtree_sizes); free(subtree_n_anc);
    if (prev_epoch_score) free(prev_epoch_score);
    if (lightning_children_offsets) free(lightning_children_offsets);
    if (lightning_children_flat)    free(lightning_children_flat);
    if (compact_slot)       free(compact_slot);
    if (real_pos_of_char)   free(real_pos_of_char);
    if (is_loop_point)      free(is_loop_point);
    if (d_is_loop_point)    cudaFree(d_is_loop_point);
    if (depth_limit) {
        for (int i = 0; i < n_root_children; i++) free(depth_limit[i]);
        free(depth_limit);
    }

    cublasDestroy(cublas);
    if (!quiet) printf("Done.\n");
    return 0;
}

// Per-subtree training path (format=2 manifest). See file for rationale.
#include "agpt_train_per_subtree.cuh"


// ============================================================================
// CLI + main
// ============================================================================

int main(int argc, char** argv) {
    const char* model_path = NULL;
    const char* trie_dir = NULL;
    const char* save_path = NULL;
    const char* fold_table_path = NULL;
    const char* virtual_tree_path = NULL;
    int epochs = 1;
    float lr = 3e-4f;
    float entropy_lambda = 0.0f;
    MassWeightMode mass_weight = MassWeightMode::Off;
    int subtree_splits = 1;   // deprecated: count-based chunking. --partition-depth is preferred.
    int partition_depth = 1;  // 1 = per-root-child (65 groups); 2 = bigram (~1139); 3 = trigram; etc.
    // Default: accumulate gradients across all splits + partitions within a
    // training-unit call; fire ONE optimizer step at the end. Preserves the
    // AGPT invariant and avoids K/V staleness that comes from firing the
    // optimizer mid-subtree. Override with --no-accumulate for the old behavior.
    bool accumulate = true;
    int chunk_queries  = 0;   // 0 → default 50000 inside trainer
    bool single_subtree = false;  // treat entire trie as one subtree (1 Adam/epoch)
    float intermediate_weight = 1.0f;  // loss scale at unary-intermediate positions; 1.0 = unchanged
    bool ce_only = false;  // force single-target CE at endpoints too (SGD-semantic, disables KL aggregation)
    float hotspot_coverage = 0.0f;  // 0 disables; X>0 splits top subtrees covering top X of excess-loss between epochs
    int   lr_rule = 0;  // per-subtree LR multiplier rule (0=none, 1=inv-depth, 2=inv-sqrt-depth, 3=sqrt-batch, 4=residual)
    bool shuffle_order = false;     // --shuffle-order: random partition-group order per SE
    int  mini_batch_groups = 1;     // --mini-batch-groups K: accumulate K partition groups per opt step
    bool per_rc_adam = false;       // --per-rc-adam: per-root-child Adam/RMSprop state (Stage 1 topological optimizer)
    bool anc_grad = false;          // --anc-grad: descendant→ancestor gradient flow for Wk/Wv
    const char* per_rc_v_dump_path = nullptr;  // --dump-per-rc-v PATH: write per-rc v buffer at end of training
    unsigned shuffle_seed = 0xa17b1edu;  // --shuffle-seed: RNG seed for shuffle
    OptimizerKind optimizer = OptimizerKind::Adam;
    float momentum_beta = 0.9f;   // used by momentum + (via β₁) adam
    float rmsprop_beta = 0.999f;  // used by rmsprop + (via β₂) adam
    int   lbfgs_k = 10;           // L-BFGS history size
    LRSchedule lr_schedule = LRSchedule::Constant;
    int warmup_epochs = 0;
    int total_epochs_budget = 0;  // 0 = use --epochs as the LR-schedule horizon. Streaming
                                   // sets this to the full multi-call SE budget so each call's
                                   // LR schedule references the global step horizon, not local.
    float weight_decay = 0.0f;
    float grad_clip_norm = 0.0f;  // 0 = disabled
    int save_every = 0;            // 0 = don't save intermediates
    CurriculumMode curriculum = CurriculumMode::Flat;
    bool lr_scale_by_steps = false;  // per-subtree: auto-rescale lr to keep the same
                                      // effective "gradient budget per pass" as the
                                      // unigram-d=16 reference recipe (65 steps/pass).
    LightningConfig lightning;  // defaults: steps=0 (off), sampler=L3, p_stop=0.3, seed=0x5c115e1

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "--trie-dir") == 0 && i + 1 < argc) trie_dir = argv[++i];
        else if (strcmp(argv[i], "--fold-table") == 0 && i + 1 < argc) fold_table_path = argv[++i];
        else if (strcmp(argv[i], "--virtual-tree") == 0 && i + 1 < argc) virtual_tree_path = argv[++i];
        else if (strcmp(argv[i], "--save") == 0 && i + 1 < argc) save_path = argv[++i];
        else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) epochs = atoi(argv[++i]);
        else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) lr = atof(argv[++i]);
        else if (strcmp(argv[i], "--entropy-lambda") == 0 && i + 1 < argc) entropy_lambda = atof(argv[++i]);
        else if (strcmp(argv[i], "--mass-weight") == 0) {
            // Two-form argument:
            //   --mass-weight           → log (alias for backward compat)
            //   --mass-weight <mode>    → mode ∈ {off, log, sqrt, linear}
            // We peek the next arg; if it matches a known mode string we
            // consume it. Otherwise treat this as bare --mass-weight (= log).
            if (i + 1 < argc) {
                const char* m = argv[i + 1];
                if      (strcmp(m, "off")    == 0) { mass_weight = MassWeightMode::Off;    i++; }
                else if (strcmp(m, "log")    == 0) { mass_weight = MassWeightMode::Log;    i++; }
                else if (strcmp(m, "sqrt")   == 0) { mass_weight = MassWeightMode::Sqrt;   i++; }
                else if (strcmp(m, "linear") == 0) { mass_weight = MassWeightMode::Linear; i++; }
                else                               { mass_weight = MassWeightMode::Log; }  // bare flag
            } else {
                mass_weight = MassWeightMode::Log;
            }
        }
        else if (strcmp(argv[i], "--subtree-splits") == 0 && i + 1 < argc) subtree_splits = atoi(argv[++i]);
        else if (strcmp(argv[i], "--partition-depth") == 0 && i + 1 < argc) partition_depth = atoi(argv[++i]);
        else if (strcmp(argv[i], "--accumulate") == 0) accumulate = true;         // default; no-op, kept for explicitness
        else if (strcmp(argv[i], "--no-accumulate") == 0) accumulate = false;     // opt in to legacy fire-per-group behavior
        else if (strcmp(argv[i], "--chunk-queries") == 0 && i + 1 < argc) chunk_queries = atoi(argv[++i]);
        else if (strcmp(argv[i], "--single-subtree") == 0) single_subtree = true;
        else if (strcmp(argv[i], "--lr-scale-by-steps") == 0) lr_scale_by_steps = true;
        else if (strcmp(argv[i], "--intermediate-weight") == 0 && i + 1 < argc) intermediate_weight = atof(argv[++i]);
        else if (strcmp(argv[i], "--ce-only") == 0) ce_only = true;
        else if (strcmp(argv[i], "--hotspot-coverage") == 0 && i + 1 < argc) hotspot_coverage = atof(argv[++i]);
        else if (strcmp(argv[i], "--shuffle-order") == 0) shuffle_order = true;
        else if (strcmp(argv[i], "--mini-batch-groups") == 0 && i + 1 < argc) mini_batch_groups = atoi(argv[++i]);
        else if (strcmp(argv[i], "--per-rc-adam") == 0) per_rc_adam = true;
        else if (strcmp(argv[i], "--anc-grad") == 0) anc_grad = true;
        else if (strcmp(argv[i], "--dump-per-rc-v") == 0 && i + 1 < argc) per_rc_v_dump_path = argv[++i];
        else if (strcmp(argv[i], "--shuffle-seed") == 0 && i + 1 < argc) shuffle_seed = (unsigned)strtoul(argv[++i], NULL, 0);
        else if (strcmp(argv[i], "--lr-rule") == 0 && i + 1 < argc) {
            const char* m = argv[++i];
            if      (strcmp(m, "none")          == 0) lr_rule = 0;
            else if (strcmp(m, "inv-depth")     == 0) lr_rule = 1;
            else if (strcmp(m, "inv-sqrt-depth")== 0) lr_rule = 2;
            else if (strcmp(m, "sqrt-batch")    == 0) lr_rule = 3;
            else if (strcmp(m, "residual")      == 0) lr_rule = 4;
            else { fprintf(stderr, "Unknown --lr-rule '%s' (none|inv-depth|inv-sqrt-depth|sqrt-batch|residual)\n", m); return 1; }
        }
        else if (strcmp(argv[i], "--optimizer") == 0 && i + 1 < argc) {
            const char* o = argv[++i];
            if      (strcmp(o, "adam")     == 0) optimizer = OptimizerKind::Adam;
            else if (strcmp(o, "sgd")      == 0) optimizer = OptimizerKind::SGD;
            else if (strcmp(o, "momentum") == 0) optimizer = OptimizerKind::Momentum;
            else if (strcmp(o, "rmsprop")  == 0) optimizer = OptimizerKind::RMSProp;
            else if (strcmp(o, "lbfgs")    == 0) optimizer = OptimizerKind::LBFGS;
            else { fprintf(stderr, "Unknown optimizer '%s' (adam|sgd|momentum|rmsprop|lbfgs)\n", o); return 1; }
        }
        else if (strcmp(argv[i], "--lbfgs-k") == 0 && i + 1 < argc) lbfgs_k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--momentum-beta") == 0 && i + 1 < argc) momentum_beta = atof(argv[++i]);
        else if (strcmp(argv[i], "--rmsprop-beta") == 0 && i + 1 < argc) rmsprop_beta = atof(argv[++i]);
        else if (strcmp(argv[i], "--lr-schedule") == 0 && i + 1 < argc) {
            const char* s = argv[++i];
            if      (strcmp(s, "constant") == 0)      lr_schedule = LRSchedule::Constant;
            else if (strcmp(s, "cosine") == 0)        lr_schedule = LRSchedule::Cosine;
            else if (strcmp(s, "warmup-cosine") == 0) lr_schedule = LRSchedule::WarmupCosine;
            else { fprintf(stderr, "Unknown lr-schedule '%s' (constant|cosine|warmup-cosine)\n", s); return 1; }
        }
        else if (strcmp(argv[i], "--warmup-epochs") == 0 && i + 1 < argc) warmup_epochs = atoi(argv[++i]);
        else if (strcmp(argv[i], "--total-epochs-budget") == 0 && i + 1 < argc) total_epochs_budget = atoi(argv[++i]);
        else if (strcmp(argv[i], "--weight-decay") == 0 && i + 1 < argc) weight_decay = atof(argv[++i]);
        else if (strcmp(argv[i], "--grad-clip-norm") == 0 && i + 1 < argc) grad_clip_norm = atof(argv[++i]);
        else if (strcmp(argv[i], "--save-every") == 0 && i + 1 < argc) save_every = atoi(argv[++i]);
        else if (strcmp(argv[i], "--curriculum") == 0 && i + 1 < argc) {
            const char* c = argv[++i];
            if (strcmp(c, "flat") == 0) curriculum = CurriculumMode::Flat;
            else if (strcmp(c, "progressive") == 0) curriculum = CurriculumMode::Progressive;
            else { fprintf(stderr, "Unknown curriculum '%s' (expected: flat, progressive)\n", c); return 1; }
        }
        else if (strcmp(argv[i], "--lightning-steps") == 0 && i + 1 < argc) lightning.steps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--lightning-sampler") == 0 && i + 1 < argc) {
            const char* s = argv[++i];
            if      (strcmp(s, "l1") == 0 || strcmp(s, "uniform") == 0)   lightning.sampler = LightningSampler::L1_Uniform;
            else if (strcmp(s, "l2") == 0 || strcmp(s, "rc-depth") == 0)  lightning.sampler = LightningSampler::L2_RcDepth;
            else if (strcmp(s, "l3") == 0 || strcmp(s, "mass-walk") == 0) lightning.sampler = LightningSampler::L3_MassWalk;
            else if (strcmp(s, "l4") == 0 || strcmp(s, "path") == 0) lightning.sampler = LightningSampler::L4_Path;
            else { fprintf(stderr, "Unknown --lightning-sampler '%s' (l1|l2|l3)\n", s); return 1; }
        }
        else if (strcmp(argv[i], "--lightning-p-stop") == 0 && i + 1 < argc) lightning.p_stop = atof(argv[++i]);
        else if (strcmp(argv[i], "--lightning-max-mass") == 0 && i + 1 < argc) lightning.max_mass = atoll(argv[++i]);
        else if (strcmp(argv[i], "--lightning-seed") == 0 && i + 1 < argc) lightning.seed = (unsigned)strtoul(argv[++i], NULL, 0);
        else if (strcmp(argv[i], "--virtual-cycles") == 0 && i + 1 < argc) lightning.virtual_cycles = atoi(argv[++i]);
        else if (strcmp(argv[i], "--lightning-mass-lr") == 0) {
            // Two-form: bare --lightning-mass-lr → log (safest), or followed by
            // off|log|sqrt|linear for explicit mode.
            if (i + 1 < argc) {
                const char* m = argv[i + 1];
                if      (strcmp(m, "off")    == 0) { lightning.mass_lr = MassWeightMode::Off;    i++; }
                else if (strcmp(m, "log")    == 0) { lightning.mass_lr = MassWeightMode::Log;    i++; }
                else if (strcmp(m, "sqrt")   == 0) { lightning.mass_lr = MassWeightMode::Sqrt;   i++; }
                else if (strcmp(m, "linear") == 0) { lightning.mass_lr = MassWeightMode::Linear; i++; }
                else                               { lightning.mass_lr = MassWeightMode::Log; }
            } else {
                lightning.mass_lr = MassWeightMode::Log;
            }
        }
    }
    if (subtree_splits < 1) subtree_splits = 1;
    if (partition_depth < 1) partition_depth = 1;

    if (!model_path || !trie_dir) {
        fprintf(stderr, "Usage: agpt_train --model <path> --trie-dir <path>\n"
                        "  [--epochs N] [--lr F]\n"
                        "  [--entropy-lambda F]        — endpoint icing, λ≥0 (0=off)\n"
                        "  [--mass-weight [off|log|sqrt|linear]] — corpus-mass weighting. Bare\n"
                        "                                flag defaults to 'log' (backward compat).\n"
                        "                                log = log(1+count)/mean (compressed, stable).\n"
                        "                                sqrt = sqrt(count)/mean (moderate compression).\n"
                        "                                linear = count/mean (matches SGD's frequency\n"
                        "                                weighting — common patterns dominate).\n"
                        "                                off = equal weight per radix endpoint.\n"
                        "  [--curriculum flat|progressive]\n"
                        "  [--subtree-splits N]        — DEPRECATED count-based chunking. Use\n"
                        "                                --partition-depth instead. With --accumulate\n"
                        "                                (default) it's harmless work-division.\n"
                        "  [--partition-depth N]       — n-gram partition: group radix nodes by their\n"
                        "                                depth-N ancestor. Pure work-division by default.\n"
                        "                                1 = per-root-child (65 groups at vocab=65).\n"
                        "                                2 = per-bigram (~1139 groups at d=16 Shakespeare).\n"
                        "                                3 = per-trigram; etc.\n"
                        "  [--accumulate]              — default ON. Accumulate gradients across all\n"
                        "                                splits and partition groups within a training\n"
                        "                                unit; fire ONE optimizer step at the end.\n"
                        "  [--no-accumulate]           — opt in to legacy per-group optimizer firing\n"
                        "                                (reintroduces K/V staleness; for reproducing old\n"
                        "                                experiments only).\n"
                        "  [--chunk-queries N]         — GPU-memory chunk size (default 50000). No effect on\n"
                        "                                gradient semantics: chunks within a split accumulate.\n"
                        "  [--single-subtree]          — merge all root-child subtrees into one → 1 Adam/epoch\n"
                        "  [--intermediate-weight F]   — loss scale at unary-intermediate positions (default 1.0;\n"
                        "                                F<1 softens run-on predictions, unchanged at endpoints).\n"
                        "  [--optimizer adam|sgd|momentum|rmsprop] — default adam.\n"
                        "                                adam uses (β₁, β₂) from --momentum-beta --rmsprop-beta.\n"
                        "                                momentum/rmsprop use their single β from the same flag.\n"
                        "  [--momentum-beta F]         — default 0.9 (= Adam β₁ when optimizer=adam)\n"
                        "  [--rmsprop-beta F]          — default 0.999 (= Adam β₂ when optimizer=adam)\n"
                        "  [--lr-schedule constant|cosine|warmup-cosine] — default constant.\n"
                        "  [--warmup-epochs N]         — warmup length for warmup-cosine (default 0).\n"
                        "  [--weight-decay F]          — decoupled AdamW-style weight decay (default 0).\n"
                        "  [--grad-clip-norm F]        — clip gradient L2 norm per subtree step (default 0=off).\n"
                        "                                Needed for stable SGD/momentum training at non-tiny lr.\n"
                        "  [--save-every N]            — checkpoint as <save>.epN every N epochs for external\n"
                        "                                best-PPL selection.\n"
                        "  [--lightning-steps N]       — Lightning Training: N stochastic subtree samples\n"
                        "                                per super-epoch; one optimizer step per sample.\n"
                        "                                0 = off (default deterministic sweep).\n"
                        "                                Implies --no-accumulate. Mutually exclusive with\n"
                        "                                --single-subtree and --partition-depth N>1 because\n"
                        "                                Lightning overwrites their pre-built partition every\n"
                        "                                epoch. Also mutex with --curriculum progressive\n"
                        "                                (use p_stop as the stochastic depth-control analogue).\n"
                        "  [--lightning-sampler l1|l2|l3|l4] — sampler variant. Default l3 (mass-walk).\n"
                        "                                l4 = path (SGD-equivalent): walks root→leaf via\n"
                        "                                mass-weighted picks, trains every radix node on\n"
                        "                                the sampled path. Sample distribution matches\n"
                        "                                uniform-corpus-position window training.\n"
                        "                                l1 = uniform over all radix nodes.\n"
                        "                                l2 = uniform over depth-1 root-children.\n"
                        "                                l3 = mass-weighted top-down walk with p_stop.\n"
                        "  [--lightning-p-stop F]      — L3 stop probability at each level (default 0.3).\n"
                        "  [--lightning-max-mass N]    — L3 cap on per-step subtree mass; force-descend\n"
                        "                                past nodes with mass>N (default 0=off). Use to\n"
                        "                                bound wall-clock on skewed corpora — e.g. set to\n"
                        "                                ~50000 on Gutenberg 5M to avoid 2M-node steps.\n"
                        "  [--lightning-seed N]        — sampler RNG seed (default 0x5c115e1).\n"
                        "  [--hotspot-coverage F]      — Adaptive split between epochs. 0.0 (default)\n"
                        "                                disables. 0.8 splits the top subtrees covering\n"
                        "                                80%% of total excess-loss (residual = mass ×\n"
                        "                                max(avg_loss − mean_loss, 0)). Splits each into\n"
                        "                                parent-only + one entry per child + descendants.\n"
                        "                                Incompatible with --lightning-steps (Lightning\n"
                        "                                resamples each epoch).\n"
                        "  [--virtual-cycles K]        — K>1 extends effective context to K·D* via\n"
                        "                                root-loop at mass>1 leaves; reuses compact\n"
                        "                                cache via delta-RoPE at gather time. K=1\n"
                        "                                is plain AGPT (default).\n"
                        "  [--lightning-mass-lr [off|log|sqrt|linear]] — per-sample LR scaling by\n"
                        "                                subtree mass. Bare flag = log (safest).\n"
                        "                                Each sample's step_lr is multiplied by\n"
                        "                                compress(subtree_mass[s]) / mean(compress).\n"
                        "                                Mean-normalized so average weight = 1.0.\n"
                        "                                linear can blow up RMSProp with a single\n"
                        "                                high-mass sample dominating; log is the\n"
                        "                                stable default. off = no scaling.\n"
                        "  [--save <path>]            — output weights path (default: --model path,\n"
                        "                                i.e. overwrite-in-place).\n"
                        "  [--fold-table <path>]      — composite cap-fold side-table built by\n"
                        "                                bin/agpt_build_fold_table. When present,\n"
                        "                                radix caps with a fold target use the\n"
                        "                                suffix-W posterior P(c|W) (top-K, sum=1)\n"
                        "                                as the training target instead of the\n"
                        "                                degenerate one-hot. Only the radix trie\n"
                        "                                format (format=1) is supported.\n"
                        "  [--virtual-tree <path>]    — VTRE side-table built by\n"
                        "                                bin/agpt_build_virtual_tree. Replaces the\n"
                        "                                one-hot intermediate target at the first\n"
                        "                                expansion_depth tunnel positions of each\n"
                        "                                cap with a length-weighted composite over\n"
                        "                                shifted-prefix walks. Tunnel positions past\n"
                        "                                expansion_depth stay one-hot. Only format=1.\n");
        return 1;
    }

    // --save behavior: explicit only. If --save is omitted, training does
    // NOT write weights or optimizer state anywhere. Earlier versions defaulted
    // save_path to model_path (overwrite-in-place) to avoid "training
    // silently discards weights at exit." That fix introduced a worse footgun:
    // diagnostic runs without --save silently appended optimizer state to the
    // source model, contaminating future cold starts (caught 2026-05-20 on the
    // raw-no-scale seed1 runs — codex spotted the adam_t=130 load). Now we
    // print a clear warning instead of silently overwriting.
    if (!save_path) {
        fprintf(stderr,
                "WARNING: no --save path given. Training results (weights + "
                "optimizer state) will be DISCARDED at exit. Pass --save PATH "
                "to persist.\n");
    }

    printf("AGPT CUDA Training Engine\n");

    // Load model
    Config cfg;
    cfg.lr = lr;
    cfg.chunk_queries = chunk_queries;
    cfg.ce_only = ce_only;
    cfg.hotspot_coverage = hotspot_coverage;
    cfg.lr_rule = lr_rule;
    cfg.shuffle_order = shuffle_order;
    cfg.shuffle_seed = shuffle_seed;
    cfg.lbfgs_k = lbfgs_k;
    cfg.mini_batch_groups = (mini_batch_groups < 1) ? 1 : mini_batch_groups;
    cfg.per_rc_adam = per_rc_adam;
    cfg.anc_grad = anc_grad;
    if (cfg.anc_grad && partition_depth != 1) {
        fprintf(stderr, "ERROR: --anc-grad requires --partition-depth 1 (cross-group cache staleness at pd>1 would confound the new gradient flow)\n");
        return 1;
    }
    if (cfg.anc_grad && accumulate) {
        fprintf(stderr, "ERROR: --anc-grad requires --no-accumulate (gradient flow is per-subtree-fire)\n");
        return 1;
    }
    if (cfg.anc_grad) {
        fprintf(stderr, "INFO: --anc-grad: per-subtree descendant→ancestor gradient flow enabled (Wk/Wv get full gradient instead of own-edge-only)\n");
    }
    cfg.per_rc_v_dump_path = per_rc_v_dump_path;
    if (cfg.per_rc_adam && accumulate) {
        fprintf(stderr, "ERROR: --per-rc-adam requires --no-accumulate (per-rc Adam state is only meaningful in per-group fire mode)\n");
        return 1;
    }
    if (cfg.per_rc_adam) {
        fprintf(stderr, "WARNING: --per-rc-adam: per-rc Adam/RMSprop state is NOT persisted across runs.\n");
        fprintf(stderr, "         Streaming/multi-stage training will start each invocation cold.\n");
    }
    float* h_weights = load_model_weights(model_path, &cfg);
    WeightOffsets wo = compute_offsets(cfg);

    // Optimizer-state persistence: allocate host buffers, try to load from the
    // model checkpoint. If absent (older format), starts cold (zeros). The
    // resulting TrainPersistence is threaded through run_radix_training so
    // Adam/RMSProp moments survive across training invocations — essential
    // for streaming / multi-stage training where save_path of one run feeds
    // model_path of the next.
    float* h_adam_m = (float*)calloc(wo.total_floats, sizeof(float));
    float* h_adam_v = (float*)calloc(wo.total_floats, sizeof(float));
    int loaded_adam_t = 0;
    bool opt_loaded = load_optimizer_state(model_path, wo.total_floats,
                                           h_adam_m, h_adam_v, &loaded_adam_t);
    if (opt_loaded) {
        printf("  Loaded optimizer state from checkpoint (adam_t=%d)\n", loaded_adam_t);
    } else {
        printf("  No optimizer state in checkpoint; starting cold\n");
    }
    TrainPersistence persist;
    persist.h_adam_m_io = h_adam_m;
    persist.h_adam_v_io = h_adam_v;
    persist.adam_t_io = &loaded_adam_t;
    // For streaming / multi-call training, the user can specify a global SE
    // budget so the LR schedule references the total horizon rather than
    // each call's local --epochs. If unset, falls back to --epochs.
    if (total_epochs_budget > 0) {
        persist.total_epochs_override = total_epochs_budget;
        printf("  LR schedule horizon: %d total SE (override; this call runs %d SE)\n", total_epochs_budget, epochs);
    }

    // Detect trie format
    int format = detect_trie_format(trie_dir);
    if (fold_table_path && format != 1) {
        fprintf(stderr, "--fold-table is only supported with the radix trie format (format=1). "
                        "Detected format=%d at %s.\n", format, trie_dir);
        return 1;
    }
    if (virtual_tree_path && format != 1) {
        fprintf(stderr, "--virtual-tree is only supported with the radix trie format (format=1). "
                        "Detected format=%d at %s.\n", format, trie_dir);
        return 1;
    }
    if (format == 2) {
        printf("Per-subtree radix format detected at %s\n", trie_dir);
        SubtreeManifest manifest = load_subtree_manifest(trie_dir);
        printf("  Manifest: %d subtree files\n", manifest.n_subtrees);
        long long total_nodes = 0, total_chars = 0;
        for (int i = 0; i < manifest.n_subtrees; i++) {
            total_nodes += manifest.entries[i].n_nodes;
            total_chars += manifest.entries[i].total_edge_chars;
        }
        printf("  Total: %lld radix nodes, %lld edge chars\n", total_nodes, total_chars);

        int rc = run_per_subtree_training(cfg, wo, h_weights, manifest,
                                           /*super_epochs=*/epochs,
                                           entropy_lambda, mass_weight, subtree_splits, partition_depth, accumulate,
                                           single_subtree, intermediate_weight,
                                           optimizer, momentum_beta, rmsprop_beta,
                                           lr_schedule, warmup_epochs,
                                           weight_decay, grad_clip_norm, save_every,
                                           curriculum, save_path,
                                           lr_scale_by_steps,
                                           lightning);
        free(manifest.entries);
        free(h_adam_m); free(h_adam_v);
        return rc;
    }
    if (format == 1) {
        printf("Loading radix trie from %s...\n", trie_dir);
        RadixTrieData radix_trie = load_radix_trie(trie_dir);

        if (fold_table_path) {
            load_fold_table(fold_table_path, radix_trie.radix_count, cfg.vocab_size);
        }
        if (virtual_tree_path) {
            load_virtual_tree(virtual_tree_path, radix_trie.radix_count, cfg.vocab_size);
        }

        int rc = run_radix_training(cfg, wo, h_weights, radix_trie, epochs, entropy_lambda, mass_weight, subtree_splits, partition_depth, accumulate, single_subtree, intermediate_weight, optimizer, momentum_beta, rmsprop_beta, lr_schedule, warmup_epochs, weight_decay, grad_clip_norm, save_every, curriculum, save_path, lightning, &persist);
        // Append optimizer state to the saved checkpoint so the next training
        // call can pick up Adam/RMSprop moments mid-stream.
        if (rc == 0 && save_path) {
            append_optimizer_state(save_path, wo.total_floats, h_adam_m, h_adam_v, loaded_adam_t);
            printf("  Appended optimizer state to %s (adam_t=%d)\n", save_path, loaded_adam_t);
        }
        free(h_adam_m); free(h_adam_v);
        return rc;
    }

    // Load leveled trie
    printf("Loading trie from %s...\n", trie_dir);
    TrieData trie = load_trie(trie_dir);

    // Allocate GPU state
    printf("Allocating GPU memory...\n");
    TrainState state = allocate_train_state(cfg, trie, wo);

    // Upload weights
    CUDA_CHECK(cudaMemcpy(state.d_weights, h_weights, wo.total_floats * sizeof(float), cudaMemcpyHostToDevice));

    // cuBLAS handle with TF32 tensor cores enabled (2-3× FP32 matmul speedup
    // on Ampere+; no-op on older GPUs).
    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));
    CUBLAS_CHECK(cublasSetMathMode(cublas, CUBLAS_TF32_TENSOR_OP_MATH));

    // Report GPU memory
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("  GPU memory: %.1f MB used, %.1f MB free, %.1f MB total\n",
           (total_mem - free_mem) / 1e6, free_mem / 1e6, total_mem / 1e6);

    // Train
    for (int epoch = 0; epoch < epochs; epoch++) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        state.adam_t = epoch + 1;
        float loss = train_epoch(state, cfg, trie, wo, cublas);

        clock_gettime(CLOCK_MONOTONIC, &t1);
        double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

        printf("Epoch %d: loss=%.6f  (%.2f sec)\n", epoch + 1, loss, elapsed);
    }

    // Save if requested
    if (save_path) {
        CUDA_CHECK(cudaMemcpy(h_weights, state.d_weights, wo.total_floats * sizeof(float), cudaMemcpyDeviceToHost));
        save_model_weights(save_path, cfg, h_weights, wo);
        printf("Saved to %s\n", save_path);
    }

    cublasDestroy(cublas);
    free(h_weights);
    free(h_adam_m); free(h_adam_v);
    printf("Done.\n");
    return 0;
}
