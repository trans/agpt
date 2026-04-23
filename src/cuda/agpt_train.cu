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
enum class OptimizerKind { Adam, SGD, Momentum, RMSProp };

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
// notes/agpt/lightning-training.md for design rationale.
enum class LightningSampler { L1_Uniform, L2_RcDepth, L3_MassWalk };
struct LightningConfig {
    int               steps      = 0;          // 0 = disabled (deterministic sweep)
    LightningSampler  sampler    = LightningSampler::L3_MassWalk;
    float             p_stop     = 0.3f;       // L3 stopping probability at each level
    unsigned          seed       = 0x5c115e1u; // sampler RNG seed
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
};

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
    if (version != 2) { fprintf(stderr, "Unsupported manifest version %d\n", version); exit(1); }
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
    if (version != 2) { fprintf(stderr, "Bad subtree version in %s\n", path); exit(1); }
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
            s.edge_tokens_flat[edge_fill_pos + e2] = read_i32(f);
        }
        edge_fill_pos += elen;
        s.edge_mass[i] = read_i32(f);
        int ec = read_i32(f);
        entry_counts_per_local[i] = ec;
        total_counts_local += ec;
        fseek(f, ec * 8, SEEK_CUR);
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
        fseek(f, elen * 4, SEEK_CUR);                           // skip edge tokens
        fseek(f, 4, SEEK_CUR);                                  // skip edge_mass
        int ec = read_i32(f);
        int out_off = s.counts_offset[i];
        for (int ee = 0; ee < ec; ee++) {
            s.counts_tok[out_off + ee] = read_i32(f);
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
    if (version != 2) { fprintf(stderr, "Radix format version %d unsupported (need v2). Rebuild index with --agpt-build-radix.\n", version); exit(1); }
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
                t.edge_tokens_flat[edge_fill_pos + e] = read_i32(f);
            }
            edge_fill_pos += elen;
            t.edge_mass[rid] = read_i32(f);  // v2 prefix mass
            int ec = read_i32(f);
            entry_counts_per_node[rid] = ec;
            total_counts_local += ec;
            fseek(f, ec * 8, SEEK_CUR);
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
            fseek(f, elen * 4, SEEK_CUR); // skip edge tokens
            fseek(f, 4, SEEK_CUR);         // skip edge_mass (v2)
            int ec = read_i32(f);
            int out_off = t.counts_offset[rid];
            for (int e = 0; e < ec; e++) {
                t.counts_tok[out_off + e] = read_i32(f);
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
    float* d_logits,              // [T_q, V] — written with gradient
    float* loss_out,              // [T_q]
    int T_q, int V,
    float entropy_lambda,         // 0 disables icing
    float intermediate_weight)    // scale applied at unary-intermediate positions (endpoints unchanged)
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

        if (is_endpoint) {
            // Endpoint: use stored counts (may be branching)
            int radix_id = radix_ids[n_idx];
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
                                 float* d_logits, float* loss_out,
                                 int T_q, int V, float entropy_lambda,
                                 float intermediate_weight) {
    int threads = (V < 256) ? V : 256;
    int t = 1; while (t < threads) t <<= 1;
    threads = (t < 32) ? 32 : t;
    int smem = threads * sizeof(float);
    agpt_loss_per_query_kernel<<<T_q, threads, smem>>>(
        logits, query_to_node, query_offsets, radix_ids, token_ids,
        counts_offset, counts_tok, counts_val, mass_weights,
        d_logits, loss_out, T_q, V, entropy_lambda, intermediate_weight);
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

// ============================================================================
// GPU Training State
// ============================================================================

struct TrainState {
    // Model weights (GPU)
    float* d_weights;
    float* d_grads;
    float* d_adam_m;
    float* d_adam_v;
    int adam_t;

    // KV cache: [n_layers][total_nodes * d_model]
    float** d_kv_keys;    // array of n_layers GPU pointers
    float** d_kv_values;

    // KV gradient accumulators (for backward)
    float** d_dkv_keys;
    float** d_dkv_values;

    // RoPE cache
    float* d_rope_cos;  // [max_seq, head_dim]
    float* d_rope_sin;

    // Trie data (GPU)
    int* d_tokens;
    int* d_parents;
    int* d_depths;
    int* d_counts_offset;
    int* d_counts_tok;
    int* d_counts_val;
    int* d_ancestor_offset;
    int* d_ancestor_ids;

    // Working buffers (allocated to max depth width)
    float* d_x;           // [max_N, d_model]
    float* d_x_res1;      // residual save
    float* d_x_res2;
    float* d_ln_out;      // [max_N, d_model]
    float* d_ln_norm;     // [max_N, d_model]
    float* d_ln_std_inv;  // [max_N, 1] (padded to max_N)
    float* d_q;           // [max_N, d_model]
    float* d_k;
    float* d_v;
    float* d_attn_out;    // [max_N, d_model]
    float* d_ff_h;        // [max_N, d_ff]
    float* d_ff_mask;     // [max_N, d_ff]
    float* d_ff_out;      // [max_N, d_model]
    float* d_logits;      // [max_N, vocab]
    float* d_d_logits;    // loss gradient
    float* d_loss;        // [max_N] per-node loss

    // Backward working buffers
    float* d_dx;          // [max_N, d_model]
    float* d_d_ln_out;
    float* d_d_attn_out;
    float* d_dq;
    float* d_dk;          // for projection backward
    float* d_dv;
    float* d_d_ff_h;      // [max_N, d_ff]
    float* d_d_ff_out;    // [max_N, d_model]
    float* d_d_ln2_out;

    // Varlen attention packed buffers
    float* d_q_packed;     // [max_N * n_heads, head_dim]
    float* d_kv_packed_k;  // [max_total_kv * n_heads, head_dim]
    float* d_kv_packed_v;
    float* d_attn_weights; // [max_N * n_heads * max_prefix_len]
    float* d_attn_out_packed; // [max_N * n_heads, head_dim]

    int* d_node_ids;       // [max_N] node ids for current depth
    int* d_positions;      // [max_N] positions (depth - 1)
    int* d_kv_offsets;     // [max_N] packed KV offset per node
    int* d_kv_lengths;     // [max_N] prefix length per node
    int* d_anc_offsets;    // [max_N] ancestor offset per node (into d_ancestor_ids)

    // Backward: dq/dk/dv from varlen attention backward
    float* d_dq_packed;
    float* d_dk_packed;
    float* d_dv_packed;

    // Per-layer saved forward state for backward (arrays of n_layers GPU pointers)
    float** saved_x_res1;      // [N, D] input to each layer
    float** saved_ln1_norm;    // [N, D] LN1 normalized
    float** saved_ln1_std_inv; // [N] LN1 std_inv
    float** saved_ln1_out;     // [N, D] LN1 output (QKV input)
    float** saved_x_res2;      // [N, D] input to FFN block
    float** saved_ln2_norm;    // [N, D] LN2 normalized
    float** saved_ln2_std_inv; // [N] LN2 std_inv
    float** saved_ln2_out;     // [N, D] LN2 output (FFN input)
    float** saved_ff_h;        // [N, F] post-ReLU hidden
    float** saved_ff_mask;     // [N, F] ReLU mask
    float** saved_attn_out;    // [N, D] attention output (WO input)
    float** saved_attn_weights;// [N * H * max_depth] per-layer

    // Final layer saved state
    float* saved_final_norm;    // [N, D]
    float* saved_final_std_inv; // [N]
    float* saved_final_out;     // [N, D] final LN output

    bool kv_on_host;       // true if KV cache lives in pinned host memory
    int max_N;             // max nodes at any depth
    int max_total_kv;      // max total KV positions for a depth
};

TrainState allocate_train_state(const Config& cfg, const TrieData& trie,
                                 const WeightOffsets& wo) {
    TrainState s;
    memset(&s, 0, sizeof(s));

    int D = cfg.d_model;
    int F = cfg.d_ff;
    int V = cfg.vocab_size;
    int L = cfg.n_layers;
    int H = cfg.n_heads;
    int HD = cfg.head_dim;

    // Weights + grads + Adam
    CUDA_CHECK(cudaMalloc(&s.d_weights, wo.total_floats * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_grads,   wo.total_floats * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_adam_m,   wo.total_floats * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_adam_v,   wo.total_floats * sizeof(float)));
    CUDA_CHECK(cudaMemset(s.d_adam_m, 0, wo.total_floats * sizeof(float)));
    CUDA_CHECK(cudaMemset(s.d_adam_v, 0, wo.total_floats * sizeof(float)));
    s.adam_t = 0;

    // KV caches — use cudaMallocManaged (Unified Memory).
    // Single pointer accessible from both CPU and GPU; the CUDA driver pages
    // data on demand between VRAM and host RAM. Working set for our access
    // pattern is small (chunk_size × depth × d_model), so most of the KV
    // can live in host RAM while only the currently-touched ancestors
    // migrate to GPU. Scales beyond GPU VRAM with no explicit paging code.
    long long kv_bytes = (long long)trie.total_nodes * D * sizeof(float);
    long long total_kv_bytes = kv_bytes * 2 * L; // K+V × layers, no grads

    // Safety: check against available RAM + swap. Managed memory can be paged
    // to swap, so swap counts. But if we'd use >80% of RAM+swap, refuse.
    {
        struct sysinfo si;
        if (sysinfo(&si) == 0) {
            long long avail_total = (long long)(si.freeram + si.freeswap) * si.mem_unit;
            long long safe_limit = (avail_total * 4) / 5;
            if (total_kv_bytes > safe_limit) {
                fprintf(stderr,
                    "REFUSED: KV cache would need %.1f GB but only %.1f GB (RAM+swap) available (80%% limit = %.1f GB).\n"
                    "  Close some apps, add swap, or reduce max_depth.\n",
                    total_kv_bytes / 1e9, avail_total / 1e9, safe_limit / 1e9);
                exit(1);
            }
            long long avail_ram = (long long)si.freeram * si.mem_unit;
            if (total_kv_bytes > avail_ram) {
                fprintf(stderr,
                    "NOTE: KV cache (%.1f GB) exceeds free RAM (%.1f GB); will use swap (slow).\n",
                    total_kv_bytes / 1e9, avail_ram / 1e9);
            }
        }
    }

    s.d_kv_keys   = (float**)malloc(L * sizeof(float*));
    s.d_kv_values = (float**)malloc(L * sizeof(float*));
    s.d_dkv_keys   = (float**)malloc(L * sizeof(float*));
    s.d_dkv_values = (float**)malloc(L * sizeof(float*));
    for (int l = 0; l < L; l++) {
        CUDA_CHECK(cudaMallocManaged(&s.d_kv_keys[l],   kv_bytes));
        CUDA_CHECK(cudaMallocManaged(&s.d_kv_values[l],  kv_bytes));
        // Skip KV gradient accumulators for v1 (Wk/Wv grads approximate anyway)
        s.d_dkv_keys[l] = NULL;
        s.d_dkv_values[l] = NULL;
    }
    s.kv_on_host = false; // Using unified memory — always use GPU kernels
    printf("  KV cache: %.1f MB unified memory (driver-paged between GPU and host)\n",
           total_kv_bytes / 1e6);

    // RoPE cache — one cache for head_dim (all heads uniform)
    build_rope_cache(&s.d_rope_cos, &s.d_rope_sin, cfg.seq_len, HD);

    // Trie data upload
    CUDA_CHECK(cudaMalloc(&s.d_tokens,  trie.total_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s.d_parents, trie.total_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s.d_depths,  trie.total_nodes * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(s.d_tokens,  trie.tokens,  trie.total_nodes * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.d_parents, trie.parents, trie.total_nodes * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.d_depths,  trie.depths,  trie.total_nodes * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&s.d_counts_offset, (trie.total_nodes + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s.d_counts_tok,    trie.total_counts * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s.d_counts_val,    trie.total_counts * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(s.d_counts_offset, trie.counts_offset, (trie.total_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.d_counts_tok, trie.counts_tok, trie.total_counts * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.d_counts_val, trie.counts_val, trie.total_counts * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&s.d_ancestor_offset, (trie.total_nodes + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s.d_ancestor_ids,    trie.total_ancestor_entries * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(s.d_ancestor_offset, trie.ancestor_offset, (trie.total_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.d_ancestor_ids, trie.ancestor_ids, trie.total_ancestor_entries * sizeof(int), cudaMemcpyHostToDevice));

    // Working buffers sized to CHUNK_SIZE, not max depth width.
    // Each depth is processed in chunks to bound GPU memory.
    #define CHUNK_SIZE 50000
    s.max_N = CHUNK_SIZE;
    // Max total KV for a chunk: CHUNK_SIZE * max_depth
    s.max_total_kv = CHUNK_SIZE * trie.max_depth;

    {
        int actual_max = 0;
        for (int d = 0; d < trie.depth_file_count; d++) {
            if (trie.depth_count[d] > actual_max) actual_max = trie.depth_count[d];
        }
        printf("  Max depth width: %d nodes (chunked to %d), max total KV per chunk: %d\n",
               actual_max, CHUNK_SIZE, s.max_total_kv);
    }

    int N = s.max_N;
    int TKV = s.max_total_kv;

    // Working buffers
    CUDA_CHECK(cudaMalloc(&s.d_x,          (long long)N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_x_res1,     (long long)N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_x_res2,     (long long)N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_ln_out,     (long long)N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_ln_norm,    (long long)N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_ln_std_inv, (long long)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_q,          (long long)N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_k,          (long long)N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_v,          (long long)N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_attn_out,   (long long)N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_ff_h,       (long long)N * F * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_ff_mask,    (long long)N * F * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_ff_out,     (long long)N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_logits,     (long long)N * V * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_d_logits,   (long long)N * V * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_loss,       (long long)N * sizeof(float)));

    // Backward buffers
    CUDA_CHECK(cudaMalloc(&s.d_dx,          (long long)N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_d_ln_out,    (long long)N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_d_attn_out,  (long long)N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_dq,          (long long)N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_dk,          (long long)N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_dv,          (long long)N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_d_ff_h,      (long long)N * F * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_d_ff_out,    (long long)N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_d_ln2_out,   (long long)N * D * sizeof(float)));

    // Varlen attention buffers
    CUDA_CHECK(cudaMalloc(&s.d_q_packed,       (long long)N * H * HD * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_attn_out_packed,(long long)N * H * HD * sizeof(float)));
    if (TKV > 0) {
        CUDA_CHECK(cudaMalloc(&s.d_kv_packed_k,   (long long)TKV * H * HD * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s.d_kv_packed_v,   (long long)TKV * H * HD * sizeof(float)));
        // attn_weights: N * H entries, each up to max_depth positions
        int max_prefix = trie.max_depth;
        CUDA_CHECK(cudaMalloc(&s.d_attn_weights,   (long long)N * H * max_prefix * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s.d_dq_packed,      (long long)N * H * HD * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s.d_dk_packed,      (long long)TKV * H * HD * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s.d_dv_packed,      (long long)TKV * H * HD * sizeof(float)));
    }

    // Per-layer saved forward state
    auto alloc_layer_bufs = [&](float*** arr, long long size) {
        *arr = (float**)malloc(L * sizeof(float*));
        for (int l = 0; l < L; l++)
            CUDA_CHECK(cudaMalloc(&(*arr)[l], size * sizeof(float)));
    };
    alloc_layer_bufs(&s.saved_x_res1,      (long long)N * D);
    alloc_layer_bufs(&s.saved_ln1_norm,     (long long)N * D);
    alloc_layer_bufs(&s.saved_ln1_std_inv,  (long long)N);
    alloc_layer_bufs(&s.saved_ln1_out,      (long long)N * D);
    alloc_layer_bufs(&s.saved_x_res2,       (long long)N * D);
    alloc_layer_bufs(&s.saved_ln2_norm,     (long long)N * D);
    alloc_layer_bufs(&s.saved_ln2_std_inv,  (long long)N);
    alloc_layer_bufs(&s.saved_ln2_out,      (long long)N * D);
    alloc_layer_bufs(&s.saved_ff_h,         (long long)N * F);
    alloc_layer_bufs(&s.saved_ff_mask,      (long long)N * F);
    alloc_layer_bufs(&s.saved_attn_out,     (long long)N * D);
    {
        int mp = trie.max_depth;
        alloc_layer_bufs(&s.saved_attn_weights, (long long)N * H * mp);
    }
    CUDA_CHECK(cudaMalloc(&s.saved_final_norm,    (long long)N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.saved_final_std_inv, (long long)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.saved_final_out,     (long long)N * D * sizeof(float)));

    // Node tracking per depth
    CUDA_CHECK(cudaMalloc(&s.d_node_ids,    N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s.d_positions,   N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s.d_kv_offsets,  N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s.d_kv_lengths,  N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s.d_anc_offsets, N * sizeof(int)));

    return s;
}

// ============================================================================
// Training epoch
// ============================================================================

float train_epoch(TrainState& s, const Config& cfg, const TrieData& trie,
                   const WeightOffsets& wo, cublasHandle_t cublas) {
    int D = cfg.d_model;
    int F = cfg.d_ff;
    int V = cfg.vocab_size;
    int L = cfg.n_layers;
    int H = cfg.n_heads;
    int HD = cfg.head_dim;

    double total_loss = 0.0;
    int nodes_trained = 0;

    // Zero KV caches at start of epoch
    for (int l = 0; l < L; l++) {
        long long kv_bytes = (long long)trie.total_nodes * D * sizeof(float);
        if (s.kv_on_host) {
            memset(s.d_kv_keys[l], 0, kv_bytes);
            memset(s.d_kv_values[l], 0, kv_bytes);
        } else {
            CUDA_CHECK(cudaMemset(s.d_kv_keys[l], 0, kv_bytes));
            CUDA_CHECK(cudaMemset(s.d_kv_values[l], 0, kv_bytes));
        }
    }

    // Pre-build depth→node_id lists (once, reused across epochs)
    // Stored in TrieData but we build here for convenience
    int** depth_node_lists = (int**)malloc(trie.depth_file_count * sizeof(int*));
    for (int d = 0; d < trie.depth_file_count; d++) {
        depth_node_lists[d] = (int*)malloc(trie.depth_count[d] * sizeof(int));
    }
    {
        int* depth_idx = (int*)calloc(trie.depth_file_count, sizeof(int));
        for (int id = 0; id < trie.total_nodes; id++) {
            int d = trie.depths[id];
            if (d >= 0 && d < trie.depth_file_count) {
                depth_node_lists[d][depth_idx[d]++] = id;
            }
        }
        free(depth_idx);
    }

    // Process each depth level
    for (int depth = 1; depth < trie.depth_file_count; depth++) {
        int N_total = trie.depth_count[depth];
        if (N_total == 0) continue;

        int* all_node_ids = depth_node_lists[depth];
        int max_prefix = depth;

        // Process in chunks of CHUNK_SIZE
        for (int chunk_start = 0; chunk_start < N_total; chunk_start += CHUNK_SIZE) {
        int N = (chunk_start + CHUNK_SIZE <= N_total) ? CHUNK_SIZE : (N_total - chunk_start);

        int* h_node_ids   = (int*)malloc(N * sizeof(int));
        int* h_positions  = (int*)malloc(N * sizeof(int));
        int* h_kv_offsets = (int*)malloc(N * sizeof(int));
        int* h_kv_lengths = (int*)malloc(N * sizeof(int));
        int* h_anc_offsets = (int*)malloc(N * sizeof(int));

        int total_kv = 0;
        for (int i = 0; i < N; i++) {
            int id = all_node_ids[chunk_start + i];
            h_node_ids[i] = id;
            h_positions[i] = depth - 1;
            h_kv_offsets[i] = total_kv;
            h_kv_lengths[i] = depth;
            h_anc_offsets[i] = trie.ancestor_offset[id];
            total_kv += depth;
        }

        CUDA_CHECK(cudaMemcpy(s.d_node_ids,    h_node_ids,    N * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(s.d_positions,   h_positions,   N * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(s.d_kv_offsets,  h_kv_offsets,  N * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(s.d_kv_lengths,  h_kv_lengths,  N * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(s.d_anc_offsets, h_anc_offsets, N * sizeof(int), cudaMemcpyHostToDevice));

        // ======== FORWARD PASS ========

        // Zero gradients
        CUDA_CHECK(cudaMemset(s.d_grads, 0, wo.total_floats * sizeof(float)));
        // Zero KV gradient accumulators (skip if NULL — v1 approximate Wk/Wv grads)
        for (int l = 0; l < L; l++) {
            if (s.d_dkv_keys[l])
                CUDA_CHECK(cudaMemset(s.d_dkv_keys[l], 0, (long long)trie.total_nodes * D * sizeof(float)));
            if (s.d_dkv_values[l])
                CUDA_CHECK(cudaMemset(s.d_dkv_values[l], 0, (long long)trie.total_nodes * D * sizeof(float)));
        }

        // 1. Embedding gather — need token ids, not node ids
        int* h_token_ids = (int*)malloc(N * sizeof(int));
        for (int i = 0; i < N; i++) {
            int nid = h_node_ids[i];
            int tok = trie.tokens[nid];
            if (tok < 0 || tok >= cfg.vocab_size) {
                fprintf(stderr, "Bad token at depth %d, chunk node %d: node_id=%d token=%d\n",
                        depth, i, nid, tok);
                exit(1);
            }
            h_token_ids[i] = tok;
        }
        int* d_token_ids;
        CUDA_CHECK(cudaMalloc(&d_token_ids, N * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_token_ids, h_token_ids, N * sizeof(int), cudaMemcpyHostToDevice));

        // Redo embedding gather with actual token ids
        cuda_embedding_gather(s.d_weights + wo.token_emb, d_token_ids, s.d_x, N, D);

        // Save x for backward (embedding backward needs to scatter to token indices)
        // x_input = x (before any modification)

        // Per-layer forward
        for (int l = 0; l < L; l++) {
            float* W_qw = s.d_weights + wo.wq_w[l];
            float* W_qb = s.d_weights + wo.wq_b[l];
            float* W_kw = s.d_weights + wo.wk_w[l];
            float* W_kb = s.d_weights + wo.wk_b[l];
            float* W_vw = s.d_weights + wo.wv_w[l];
            float* W_vb = s.d_weights + wo.wv_b[l];
            float* W_ow = s.d_weights + wo.wo_w[l];
            float* W_ob = s.d_weights + wo.wo_b[l];
            float* G1   = s.d_weights + wo.ln1_gamma[l];
            float* B1   = s.d_weights + wo.ln1_beta[l];
            float* W_1w = s.d_weights + wo.l1_w[l];
            float* W_1b = s.d_weights + wo.l1_b[l];
            float* W_2w = s.d_weights + wo.l2_w[l];
            float* W_2b = s.d_weights + wo.l2_b[l];
            float* G2   = s.d_weights + wo.ln2_gamma[l];
            float* B2   = s.d_weights + wo.ln2_beta[l];

            // Save residual input for backward
            CUDA_CHECK(cudaMemcpy(s.saved_x_res1[l], s.d_x, (long long)N * D * sizeof(float), cudaMemcpyDeviceToDevice));

            // LayerNorm1 → saved per-layer
            cuda_layer_norm_forward(s.d_x, s.d_ln_out, s.saved_ln1_norm[l], s.saved_ln1_std_inv[l], G1, B1, N, D);
            CUDA_CHECK(cudaMemcpy(s.saved_ln1_out[l], s.d_ln_out, (long long)N * D * sizeof(float), cudaMemcpyDeviceToDevice));

            // Q/K/V projections
            float alpha = 1.0f, beta_zero = 0.0f;
            CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                      D, N, D, &alpha, W_qw, D, s.d_ln_out, D, &beta_zero, s.d_q, D));
            cuda_bias_add(s.d_q, W_qb, N, D);
            CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                      D, N, D, &alpha, W_kw, D, s.d_ln_out, D, &beta_zero, s.d_k, D));
            cuda_bias_add(s.d_k, W_kb, N, D);
            CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                      D, N, D, &alpha, W_vw, D, s.d_ln_out, D, &beta_zero, s.d_v, D));
            cuda_bias_add(s.d_v, W_vb, N, D);

            // RoPE — build expanded positions [N*H]
            int* h_exp_pos = (int*)malloc(N * H * sizeof(int));
            for (int i = 0; i < N; i++)
                for (int h = 0; h < H; h++)
                    h_exp_pos[i * H + h] = h_positions[i];
            int* d_exp_pos;
            CUDA_CHECK(cudaMalloc(&d_exp_pos, N * H * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(d_exp_pos, h_exp_pos, N * H * sizeof(int), cudaMemcpyHostToDevice));
            launch_rope_batched(s.d_q, d_exp_pos, s.d_rope_cos, s.d_rope_sin, N * H, HD);
            launch_rope_batched(s.d_k, d_exp_pos, s.d_rope_cos, s.d_rope_sin, N * H, HD);

            // Store K/V into global KV cache
            if (s.kv_on_host) {
                host_kv_scatter(s.d_k, h_node_ids, s.d_kv_keys[l], N, D);
                host_kv_scatter(s.d_v, h_node_ids, s.d_kv_values[l], N, D);
            } else {
                launch_kv_scatter(s.d_k, s.d_node_ids, s.d_kv_keys[l], N, D);
                launch_kv_scatter(s.d_v, s.d_node_ids, s.d_kv_values[l], N, D);
            }

            // Gather ancestor K/V into packed buffers
            if (s.kv_on_host) {
                host_kv_gather(s.d_kv_keys[l], trie.ancestor_ids, h_anc_offsets,
                                h_kv_offsets, h_kv_lengths, s.d_kv_packed_k,
                                N, H, HD, total_kv);
                host_kv_gather(s.d_kv_values[l], trie.ancestor_ids, h_anc_offsets,
                                h_kv_offsets, h_kv_lengths, s.d_kv_packed_v,
                                N, H, HD, total_kv);
            } else {
                launch_kv_gather(s.d_kv_keys[l], s.d_ancestor_ids, s.d_anc_offsets,
                                  s.d_kv_offsets, s.d_kv_lengths, s.d_kv_packed_k, N, H, HD);
                launch_kv_gather(s.d_kv_values[l], s.d_ancestor_ids, s.d_anc_offsets,
                                  s.d_kv_offsets, s.d_kv_lengths, s.d_kv_packed_v, N, H, HD);
            }

            // Varlen attention
            float scale = 1.0f / sqrtf((float)HD);
            cuda_batched_varlen_attention(
                s.d_q, s.d_kv_packed_k, s.d_kv_packed_v,
                s.d_kv_offsets, s.d_kv_lengths,
                s.d_attn_out_packed, s.saved_attn_weights[l],
                N, H, HD, max_prefix, scale);
            cuda_unpack_batched_attn_output(s.d_attn_out_packed, s.d_attn_out, N, H, HD);

            // Save attn_out for WO backward
            CUDA_CHECK(cudaMemcpy(s.saved_attn_out[l], s.d_attn_out, (long long)N * D * sizeof(float), cudaMemcpyDeviceToDevice));

            // WO projection + residual 1
            CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                      D, N, D, &alpha, W_ow, D, s.d_attn_out, D, &beta_zero, s.d_ff_out, D));
            cuda_bias_add(s.d_ff_out, W_ob, N, D);
            CUDA_CHECK(cudaMemcpy(s.d_x, s.saved_x_res1[l], (long long)N * D * sizeof(float), cudaMemcpyDeviceToDevice));
            launch_elem_add(s.d_x, s.d_ff_out, N * D);

            // Save residual 2 input
            CUDA_CHECK(cudaMemcpy(s.saved_x_res2[l], s.d_x, (long long)N * D * sizeof(float), cudaMemcpyDeviceToDevice));

            // LayerNorm2 → saved per-layer
            cuda_layer_norm_forward(s.d_x, s.d_ln_out, s.saved_ln2_norm[l], s.saved_ln2_std_inv[l], G2, B2, N, D);
            CUDA_CHECK(cudaMemcpy(s.saved_ln2_out[l], s.d_ln_out, (long long)N * D * sizeof(float), cudaMemcpyDeviceToDevice));

            // FFN: l1 (fused bias+relu) → l2 + bias + residual 2
            CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                      F, N, D, &alpha, W_1w, F, s.d_ln_out, D, &beta_zero, s.d_ff_h, F));
            cuda_fused_bias_relu(s.d_ff_h, W_1b, s.d_ff_h, s.saved_ff_mask[l], N, F);
            // Save post-ReLU hidden for backward
            CUDA_CHECK(cudaMemcpy(s.saved_ff_h[l], s.d_ff_h, (long long)N * F * sizeof(float), cudaMemcpyDeviceToDevice));

            CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                      D, N, F, &alpha, W_2w, D, s.d_ff_h, F, &beta_zero, s.d_ff_out, D));
            cuda_bias_add(s.d_ff_out, W_2b, N, D);

            CUDA_CHECK(cudaMemcpy(s.d_x, s.saved_x_res2[l], (long long)N * D * sizeof(float), cudaMemcpyDeviceToDevice));
            launch_elem_add(s.d_x, s.d_ff_out, N * D);

            CUDA_CHECK(cudaFree(d_exp_pos));
            free(h_exp_pos);
        }

        // Final norm + output projection
        float* G_fn = s.d_weights + wo.final_gamma;
        float* B_fn = s.d_weights + wo.final_beta;
        float* W_out = s.d_weights + wo.out_w;
        float* B_out = s.d_weights + wo.out_b;

        cuda_layer_norm_forward(s.d_x, s.d_ln_out, s.saved_final_norm, s.saved_final_std_inv, G_fn, B_fn, N, D);
        CUDA_CHECK(cudaMemcpy(s.saved_final_out, s.d_ln_out, (long long)N * D * sizeof(float), cudaMemcpyDeviceToDevice));

        float alpha = 1.0f, beta_zero = 0.0f;
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                  V, N, D, &alpha, W_out, V, s.d_ln_out, D, &beta_zero, s.d_logits, V));
        cuda_bias_add(s.d_logits, B_out, N, V);

        // Loss
        launch_agpt_loss(s.d_logits, s.d_node_ids,
                          s.d_counts_offset, s.d_counts_tok, s.d_counts_val,
                          s.d_d_logits, s.d_loss, N, V);

        // Sum loss on CPU
        float* h_loss = (float*)malloc(N * sizeof(float));
        CUDA_CHECK(cudaMemcpy(h_loss, s.d_loss, N * sizeof(float), cudaMemcpyDeviceToHost));
        int depth_trained = 0;
        for (int i = 0; i < N; i++) {
            if (h_loss[i] > 0.0f) { total_loss += h_loss[i]; depth_trained++; }
        }
        nodes_trained += depth_trained;
        free(h_loss);

        // ======== BACKWARD PASS ========
        // d_logits already has the loss gradient from agpt_loss_kernel.
        // Scale gradients by 1/N for this chunk's update.
        float grad_scale = 1.0f / (float)N;
        float neg_grad_scale = -grad_scale;

        // Output projection backward: d_logits[N,V] → d_final_out[N,D]
        // d_final_out = d_logits × W_out^T
        float* dG_out = s.d_grads + wo.out_w;
        float* dB_out = s.d_grads + wo.out_b;
        // dx = d_logits[N,V] × W_out^T[V,D]
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                  D, N, V, &alpha, W_out, V, s.d_d_logits, V, &beta_zero, s.d_dx, D));
        // dW_out += final_out^T[D,N] × d_logits[N,V] (scaled)
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                                  V, D, N, &grad_scale, s.d_d_logits, V, s.saved_final_out, D, &alpha, dG_out, V));
        // db_out += sum over rows of d_logits (use gemv: ones^T × d_logits)
        // Simpler: use cublasSgemv with a ones vector, or just leave as is for now
        // We can accumulate bias grad with a simple kernel
        // For now, skip bias grad accumulation (TODO)

        // Final LayerNorm backward
        float* dG_fn = s.d_grads + wo.final_gamma;
        float* dB_fn = s.d_grads + wo.final_beta;
        cuda_layer_norm_backward(s.d_dx, s.saved_final_norm, s.saved_final_std_inv,
                                  G_fn, s.d_dx, dG_fn, dB_fn, N, D);
        // Note: cuda_layer_norm_backward writes dx in-place, and accumulates dgamma/dbeta

        // Per-layer backward (reverse order)
        for (int l = L - 1; l >= 0; l--) {
            float* W_qw = s.d_weights + wo.wq_w[l];
            float* W_kw = s.d_weights + wo.wk_w[l];
            float* W_vw = s.d_weights + wo.wv_w[l];
            float* W_ow = s.d_weights + wo.wo_w[l];
            float* W_1w = s.d_weights + wo.l1_w[l];
            float* W_2w = s.d_weights + wo.l2_w[l];
            float* G1   = s.d_weights + wo.ln1_gamma[l];
            float* G2   = s.d_weights + wo.ln2_gamma[l];

            float* dW_qw = s.d_grads + wo.wq_w[l]; float* dW_qb = s.d_grads + wo.wq_b[l];
            float* dW_kw = s.d_grads + wo.wk_w[l]; float* dW_kb = s.d_grads + wo.wk_b[l];
            float* dW_vw = s.d_grads + wo.wv_w[l]; float* dW_vb = s.d_grads + wo.wv_b[l];
            float* dW_ow = s.d_grads + wo.wo_w[l]; float* dW_ob = s.d_grads + wo.wo_b[l];
            float* dG1   = s.d_grads + wo.ln1_gamma[l]; float* dB1 = s.d_grads + wo.ln1_beta[l];
            float* dW_1w = s.d_grads + wo.l1_w[l]; float* dW_1b = s.d_grads + wo.l1_b[l];
            float* dW_2w = s.d_grads + wo.l2_w[l]; float* dW_2b = s.d_grads + wo.l2_b[l];
            float* dG2   = s.d_grads + wo.ln2_gamma[l]; float* dB2 = s.d_grads + wo.ln2_beta[l];

            // d_x is the gradient flowing back. It splits at residual 2.
            // d_ff_out = d_x (one branch)
            // d_x_res2 = d_x (skip branch, accumulated later)
            CUDA_CHECK(cudaMemcpy(s.d_d_ff_out, s.d_dx, (long long)N * D * sizeof(float), cudaMemcpyDeviceToDevice));

            // FFN L2 backward: d_ff_out[N,D] → d_ff_h[N,F]
            CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                      F, N, D, &alpha, W_2w, D, s.d_d_ff_out, D, &beta_zero, s.d_d_ff_h, F));
            // dW_2 += ff_h^T × d_ff_out (scaled)
            CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                                      D, F, N, &grad_scale, s.d_d_ff_out, D, s.saved_ff_h[l], F, &alpha, dW_2w, D));

            // ReLU backward: d_ff_h *= saved_ff_mask
            cuda_relu_backward(s.d_d_ff_h, s.saved_ff_mask[l], s.d_d_ff_h, N * F);

            // FFN L1 backward: d_ff_h[N,F] → d_ln2_out[N,D]
            CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                      D, N, F, &alpha, W_1w, F, s.d_d_ff_h, F, &beta_zero, s.d_d_ln_out, D));
            // dW_1 += ln2_out^T × d_ff_h (scaled)
            CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                                      F, D, N, &grad_scale, s.d_d_ff_h, F, s.saved_ln2_out[l], D, &alpha, dW_1w, F));

            // LN2 backward
            cuda_layer_norm_backward(s.d_d_ln_out, s.saved_ln2_norm[l], s.saved_ln2_std_inv[l],
                                      G2, s.d_d_ln_out, dG2, dB2, N, D);

            // Add residual 2 skip: d_x = d_ln2_backward + d_x (from skip)
            launch_elem_add(s.d_dx, s.d_d_ln_out, N * D);

            // Now d_dx flows to attention block backward
            // WO backward: d_dx → d_attn_out[N,D]
            CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                      D, N, D, &alpha, W_ow, D, s.d_dx, D, &beta_zero, s.d_d_attn_out, D));
            // dW_o += attn_out^T × d_dx (scaled)
            CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                                      D, D, N, &grad_scale, s.d_dx, D, s.saved_attn_out[l], D, &alpha, dW_ow, D));

            // Attention backward using varlen attention backward kernel
            // Need to re-gather K/V for this layer
            if (s.kv_on_host) {
                host_kv_gather(s.d_kv_keys[l], trie.ancestor_ids, h_anc_offsets,
                                h_kv_offsets, h_kv_lengths, s.d_kv_packed_k,
                                N, H, HD, total_kv);
                host_kv_gather(s.d_kv_values[l], trie.ancestor_ids, h_anc_offsets,
                                h_kv_offsets, h_kv_lengths, s.d_kv_packed_v,
                                N, H, HD, total_kv);
            } else {
                launch_kv_gather(s.d_kv_keys[l], s.d_ancestor_ids, s.d_anc_offsets,
                                  s.d_kv_offsets, s.d_kv_lengths, s.d_kv_packed_k, N, H, HD);
                launch_kv_gather(s.d_kv_values[l], s.d_ancestor_ids, s.d_anc_offsets,
                                  s.d_kv_offsets, s.d_kv_lengths, s.d_kv_packed_v, N, H, HD);
            }

            // d_attn_out is [N, D]. The backward kernel expects [N*H, HD] packed format.
            // Same memory layout, just reinterpret.
            float scale = 1.0f / sqrtf((float)HD);

            // Reconstruct Q from saved ln1_out + weight projections + RoPE
            // (Re-project from saved LN1 output rather than storing Q separately)
            CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                      D, N, D, &alpha, s.d_weights + wo.wq_w[l], D,
                                      s.saved_ln1_out[l], D, &beta_zero, s.d_q, D));
            cuda_bias_add(s.d_q, s.d_weights + wo.wq_b[l], N, D);
            // Re-apply RoPE
            int* h_exp_pos = (int*)malloc(N * H * sizeof(int));
            for (int i = 0; i < N; i++)
                for (int h = 0; h < H; h++)
                    h_exp_pos[i * H + h] = h_positions[i];
            int* d_exp_pos;
            CUDA_CHECK(cudaMalloc(&d_exp_pos, N * H * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(d_exp_pos, h_exp_pos, N * H * sizeof(int), cudaMemcpyHostToDevice));
            launch_rope_batched(s.d_q, d_exp_pos, s.d_rope_cos, s.d_rope_sin, N * H, HD);

            cuda_batched_varlen_attention_backward(
                s.d_q, s.d_kv_packed_k, s.d_kv_packed_v,
                s.saved_attn_weights[l], s.d_d_attn_out,
                s.d_kv_offsets, s.d_kv_lengths,
                s.d_dq_packed, s.d_dk_packed, s.d_dv_packed,
                N, H, HD, max_prefix, scale);

            // Inverse RoPE on dQ to get d_q_pre_rope
            launch_rope_batched_inverse(s.d_dq_packed, d_exp_pos, s.d_rope_cos, s.d_rope_sin, N * H, HD);

            // dQ → d_ln1_out via Wq^T: d_ln1_out_q = dQ[N,D] × Wq^T[D,D]
            CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                      D, N, D, &alpha, s.d_weights + wo.wq_w[l], D,
                                      s.d_dq_packed, D, &beta_zero, s.d_d_ln_out, D));
            // dWq += ln1_out^T × dQ (scaled)
            CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                                      D, D, N, &grad_scale, s.d_dq_packed, D,
                                      s.saved_ln1_out[l], D, &alpha, dW_qw, D));

            // Scatter dk/dv back to global KV grad accumulators (skip if NULL)
            if (s.d_dkv_keys[l]) {
                launch_kv_scatter_add(s.d_dk_packed, s.d_ancestor_ids, s.d_anc_offsets,
                                       s.d_kv_offsets, s.d_kv_lengths, s.d_dkv_keys[l], N, H, HD);
                launch_kv_scatter_add(s.d_dv_packed, s.d_ancestor_ids, s.d_anc_offsets,
                                       s.d_kv_offsets, s.d_kv_lengths, s.d_dkv_values[l], N, H, HD);
            }

            // dK for current nodes: extract from global dk accumulator via scatter
            // dK[i] = d_dkv_keys[l][node_ids[i]] (these are the K grads for THIS node's position)
            // Then inverse RoPE and backprop through Wk
            // For simplicity, gather dk for current nodes from the accumulator
            // Actually, the current node's dk contribution is already in d_dk_packed
            // (the self-position in the KV). But d_dk_packed contains ALL positions' grads
            // for ALL nodes. We need just the current node's own position.
            // Simpler: use the d_dkv_keys accumulator. Gather current nodes' entries.

            // Gather current-node dK/dV from accumulators
            // d_dk_self[i] = d_dkv_keys[l][node_ids[i] * D .. +D]
            // This is like the inverse of kv_scatter: use d_node_ids to gather
            // For now, use a simple gather: d_k[i] = accumulator[node_ids[i]]
            // We need a gather kernel (reverse of scatter)

            // Actually, we can just skip per-node K/V gradients for now.
            // The main gradients flow through Q (dWq is already accumulated).
            // K/V gradients for ancestors at PREVIOUS depths would need a multi-depth
            // backward pass. For the initial version, let's just accumulate the
            // projection gradients for the current depth's K and V via the saved ln1_out.
            // dWk += ln1_out^T × dK_for_current_nodes (this is approximate — ignores
            // the fact that K affects attention at FUTURE depths too).

            // For a correct gradient: we'd need to accumulate dK/dV across all depths
            // that attend to each node, then backprop through projections.
            // That's complex. For v1, let's do the simple version: only propagate
            // gradients through Q (which is always for the current depth) and through
            // the FFN/LN paths. This is correct for the FFN/LN/embedding weights
            // and approximately correct for Wq/Wo. Wk/Wv gradients are partial.

            // LN1 backward
            cuda_layer_norm_backward(s.d_d_ln_out, s.saved_ln1_norm[l], s.saved_ln1_std_inv[l],
                                      G1, s.d_d_ln_out, dG1, dB1, N, D);

            // Add residual 1 skip
            launch_elem_add(s.d_dx, s.d_d_ln_out, N * D);

            CUDA_CHECK(cudaFree(d_exp_pos));
            free(h_exp_pos);
        }

        // Embedding backward: scatter_add d_x into d_token_emb
        float* dG_emb = s.d_grads + wo.token_emb;
        cuda_embedding_scatter_add(s.d_dx, d_token_ids, dG_emb, N, D);

        // Adam update
        s.adam_t++;
        cuda_adam_bulk(s.d_weights, s.d_grads, s.d_adam_m, s.d_adam_v,
                        cfg.lr, 0.9f, 0.999f, 1e-8f, s.adam_t, wo.total_floats);

        // Cleanup per-chunk allocations
        CUDA_CHECK(cudaFree(d_token_ids));
        free(h_token_ids);
        free(h_node_ids);
        free(h_positions);
        free(h_kv_offsets);
        free(h_kv_lengths);
        free(h_anc_offsets);

        } // end chunk loop

        CUDA_CHECK(cudaDeviceSynchronize()); // catch any async errors from this depth

        if (depth <= 3 || depth == trie.depth_file_count - 1) {
            printf("  depth %d: %d nodes, %d trained, loss=%.4f\n",
                   depth, N_total, nodes_trained,
                   nodes_trained > 0 ? (total_loss / nodes_trained) : 0.0);
        }
    }

    // Free depth node lists
    for (int d = 0; d < trie.depth_file_count; d++) free(depth_node_lists[d]);
    free(depth_node_lists);

    float mean_loss = nodes_trained > 0 ? (float)(total_loss / nodes_trained) : 0.0f;
    return mean_loss;
}

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

// run_radix_training optional parameters (declared here via overload-less defaults).
// When invoked from the per-subtree wrapper, these thread optimizer state across
// calls so RMSProp/Adam running averages don't reset per subtree, and suppress
// the usual startup banner for a clean repeated-call log.
struct TrainPersistence {
    float* h_adam_m_io = nullptr;  // if non-null: load in on entry, copy out on exit
    float* h_adam_v_io = nullptr;
    int*   adam_t_io   = nullptr;  // read on entry as starting step, write on exit
    bool   quiet       = false;    // suppress banner + per-epoch lines
    int    total_opt_steps_override = 0;  // for LR schedule when the caller knows the true horizon
    int    warmup_steps_override    = 0;  // caller-known warmup length (0 = derive from warmup_epochs)
};

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
                          :                                                        "l3-mass-walk";
        const char* mlr = (lightning.mass_lr == MassWeightMode::Log)    ? ", mass-lr=log"
                        : (lightning.mass_lr == MassWeightMode::Sqrt)   ? ", mass-lr=sqrt"
                        : (lightning.mass_lr == MassWeightMode::Linear) ? ", mass-lr=linear"
                        :                                                  "";
        printf("  lightning: %s, %d samples/super-epoch, p_stop=%.2f, seed=0x%x%s\n",
               sname, lightning.steps, lightning.p_stop, lightning.seed, mlr);
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

    int* d_compact_slot;
    CUDA_CHECK(cudaMalloc(&d_compact_slot, (long long)trie.total_edge_chars * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_compact_slot, compact_slot,
                          (long long)trie.total_edge_chars * sizeof(int), cudaMemcpyHostToDevice));

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
    __nv_bfloat16** d_kv_keys = (__nv_bfloat16**)malloc(L_layers * sizeof(__nv_bfloat16*));
    __nv_bfloat16** d_kv_values = (__nv_bfloat16**)malloc(L_layers * sizeof(__nv_bfloat16*));
    for (int l = 0; l < L_layers; l++) {
        CUDA_CHECK(cudaMallocManaged(&d_kv_keys[l],   kv_bytes));
        CUDA_CHECK(cudaMallocManaged(&d_kv_values[l], kv_bytes));
    }
    if (!quiet) printf("  KV cache: %.1f MB unified memory (bf16 compact)\n", total_kv_bytes / 1e6);

    // RoPE cache
    float* d_rope_cos; float* d_rope_sin;
    build_rope_cache(&d_rope_cos, &d_rope_sin, cfg.seq_len, HD);

    // Per-chunk working buffers. Sized to T_q_cap queries, T_kv_cap packed KV.
    int N_cap = CHUNK_QUERIES;  // worst-case radix count per chunk when edge_len=1

    // Query-side buffers (one row per query position)
    float *d_x, *d_x_res1, *d_x_res2, *d_ln_out, *d_q, *d_k, *d_v, *d_attn_out, *d_ff_h, *d_ff_mask, *d_ff_out;
    CUDA_CHECK(cudaMalloc(&d_x,         (long long)T_q_cap * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x_res1,    (long long)T_q_cap * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x_res2,    (long long)T_q_cap * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ln_out,    (long long)T_q_cap * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_q,         (long long)T_q_cap * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k,         (long long)T_q_cap * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v,         (long long)T_q_cap * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_attn_out,  (long long)T_q_cap * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ff_h,      (long long)T_q_cap * F * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ff_mask,   (long long)T_q_cap * F * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ff_out,    (long long)T_q_cap * D * sizeof(float)));

    // Endpoint-side buffers (one row per radix node = N)
    float *d_final_out, *d_final_norm_save, *d_final_std_inv_save, *d_logits, *d_d_logits, *d_loss, *d_d_final_out;
    CUDA_CHECK(cudaMalloc(&d_final_out,          (long long)N_cap * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_final_norm_save,    (long long)N_cap * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_final_std_inv_save, (long long)N_cap * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_logits,             (long long)N_cap * V * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_d_logits,           (long long)N_cap * V * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_loss,               (long long)N_cap * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_d_final_out,        (long long)N_cap * D * sizeof(float)));

    // Per-layer saved state (for backward)
    float** sv_x_res1 = (float**)malloc(L_layers * sizeof(float*));
    float** sv_ln1_norm = (float**)malloc(L_layers * sizeof(float*));
    float** sv_ln1_std_inv = (float**)malloc(L_layers * sizeof(float*));
    float** sv_ln1_out = (float**)malloc(L_layers * sizeof(float*));
    float** sv_x_res2 = (float**)malloc(L_layers * sizeof(float*));
    float** sv_ln2_norm = (float**)malloc(L_layers * sizeof(float*));
    float** sv_ln2_std_inv = (float**)malloc(L_layers * sizeof(float*));
    float** sv_ln2_out = (float**)malloc(L_layers * sizeof(float*));
    float** sv_ff_h = (float**)malloc(L_layers * sizeof(float*));
    float** sv_ff_mask = (float**)malloc(L_layers * sizeof(float*));
    float** sv_attn_out = (float**)malloc(L_layers * sizeof(float*));
    float** sv_attn_weights = (float**)malloc(L_layers * sizeof(float*));
    for (int l = 0; l < L_layers; l++) {
        CUDA_CHECK(cudaMalloc(&sv_x_res1[l],      (long long)T_q_cap * D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&sv_ln1_norm[l],    (long long)T_q_cap * D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&sv_ln1_std_inv[l], (long long)T_q_cap * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&sv_ln1_out[l],     (long long)T_q_cap * D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&sv_x_res2[l],      (long long)T_q_cap * D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&sv_ln2_norm[l],    (long long)T_q_cap * D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&sv_ln2_std_inv[l], (long long)T_q_cap * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&sv_ln2_out[l],     (long long)T_q_cap * D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&sv_ff_h[l],        (long long)T_q_cap * F * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&sv_ff_mask[l],     (long long)T_q_cap * F * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&sv_attn_out[l],    (long long)T_q_cap * D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&sv_attn_weights[l],(long long)T_q_cap * H * max_kv_per_node * sizeof(float)));
    }

    // Packed attention buffers
    float *d_q_pack_flat, *d_kv_pack_k, *d_kv_pack_v;
    // T_kv = max packed KV positions per chunk (upper bound)
    long long T_kv_max = (long long)T_q_cap * max_kv_per_node; // very generous
    // Clamp with available memory
    if (T_kv_max > (long long)T_q_cap * 2000) T_kv_max = (long long)T_q_cap * 2000;
    CUDA_CHECK(cudaMalloc(&d_q_pack_flat, (long long)T_q_cap * H * HD * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_kv_pack_k,   T_kv_max * H * HD * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_kv_pack_v,   T_kv_max * H * HD * sizeof(float)));

    // dQ/dK/dV packed
    float *d_dq_pack, *d_dk_pack, *d_dv_pack;
    CUDA_CHECK(cudaMalloc(&d_dq_pack, (long long)T_q_cap * H * HD * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dk_pack, T_kv_max * H * HD * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dv_pack, T_kv_max * H * HD * sizeof(float)));

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
    CUDA_CHECK(cudaMalloc(&d_radix_ids,      N_cap * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_query_to_node,  T_q_cap * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_query_offsets,  (N_cap + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_kv_offsets,     (N_cap + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_kv_lengths,     N_cap * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_token_ids,      T_q_cap * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_rope_positions, T_q_cap * H * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_char_pos,       T_q_cap * sizeof(int)));

    // Per-query mass weight buffer (populated when mass_weight mode != Off).
    float* d_mass_weights = NULL;
    if (mass_weight != MassWeightMode::Off) {
        CUDA_CHECK(cudaMalloc(&d_mass_weights, T_q_cap * sizeof(float)));
    }

    // cuBLAS handle
    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));

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
        subtree_nodes = (int**)malloc(sizeof(int*));
        subtree_sizes = (int*)malloc(sizeof(int));
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
        n_root_children = n_groups;   // semantics: now "partition groups"

        if (!quiet) {
            printf("  partition-depth=%d: %d groups, 1 Adam step per group per super-epoch\n",
                   partition_depth, n_groups);
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

    if (lightning_active) {
        // Build adjacency from parents[]. Each non-root r has parents[r] as its parent.
        int* cnt = (int*)calloc(trie.radix_count, sizeof(int));
        for (int r = 1; r < trie.radix_count; r++) cnt[trie.parents[r]]++;
        lightning_children_offsets = (int*)calloc(trie.radix_count + 1, sizeof(int));
        for (int r = 0; r < trie.radix_count; r++) {
            lightning_children_offsets[r + 1] = lightning_children_offsets[r] + cnt[r];
        }
        long long total_child_edges = lightning_children_offsets[trie.radix_count];
        lightning_children_flat = (int*)malloc((total_child_edges > 0 ? total_child_edges : 1) * sizeof(int));
        int* cur = (int*)calloc(trie.radix_count, sizeof(int));
        for (int r = 1; r < trie.radix_count; r++) {
            int p = trie.parents[r];
            lightning_children_flat[lightning_children_offsets[p] + cur[p]++] = r;
        }
        free(cur);
        free(cnt);

        // Force accumulate=false: each Lightning sample is its own bounded
        // training unit with one optimizer step. Without this override the
        // whole super-epoch of N samples would collapse into one step, which
        // is not what Lightning is supposed to do.
        accumulate = false;

        if (!quiet) {
            printf("  lightning: built adjacency (%lld child edges total; root has %d children)\n",
                   total_child_edges, lightning_children_offsets[1] - lightning_children_offsets[0]);
        }
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

    // ------------------------------------------------------------
    // Training loop
    // ------------------------------------------------------------
    for (int epoch = 0; epoch < epochs; epoch++) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        // Zero KV caches
        for (int l = 0; l < L_layers; l++) {
            CUDA_CHECK(cudaMemset(d_kv_keys[l],   0, kv_bytes));
            CUDA_CHECK(cudaMemset(d_kv_values[l], 0, kv_bytes));
        }

        // ------------------------------------------------------------
        // Lightning resampling: generate N stochastic samples for this
        // super-epoch and replace subtree_nodes[] / subtree_sizes[].
        // ------------------------------------------------------------
        int lightning_depth_hist[64];
        long long lightning_nodes_sum = 0;
        double* lightning_subtree_mass = NULL;
        double lightning_mean_mass = 1.0;
        double lightning_mass_min = 0.0, lightning_mass_max = 0.0;
        double lightning_w_min = 1.0, lightning_w_max = 1.0;
        if (lightning_active) {
            for (int i = 0; i < 64; i++) lightning_depth_hist[i] = 0;
            // Free previous subtree_nodes / subtree_sizes.
            for (int i = 0; i < n_root_children; i++) free(subtree_nodes[i]);
            free(subtree_nodes);
            free(subtree_sizes);
            n_root_children = lightning.steps;
            subtree_nodes = (int**)malloc(n_root_children * sizeof(int*));
            subtree_sizes = (int*)calloc(n_root_children, sizeof(int));
            lightning_subtree_mass = (double*)calloc(n_root_children, sizeof(double));

            // Scratch buffers reused across samples: BFS queue of radix ids.
            int* bfs_buf = (int*)malloc(trie.radix_count * sizeof(int));

            for (int s = 0; s < n_root_children; s++) {
                // --- Sample a radix node ---
                int r_sample = 0;
                if (lightning.sampler == LightningSampler::L3_MassWalk) {
                    int cur = 0;
                    while (1) {
                        // Stop probability applies at all nodes except the virtual root.
                        if (cur != 0) {
                            float u = xorshift_float01(&lightning_rng);
                            if (u < lightning.p_stop) break;
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

                // --- BFS: {r_sample} ∪ descendants(r_sample) ---
                int fill = 0;
                int head = 0;
                bfs_buf[fill++] = r_sample;
                double mass_accum = (double)trie.edge_mass[r_sample];
                while (head < fill) {
                    int cur = bfs_buf[head++];
                    int cs = lightning_children_offsets[cur];
                    int ce = lightning_children_offsets[cur + 1];
                    for (int j = cs; j < ce; j++) {
                        int child = lightning_children_flat[j];
                        bfs_buf[fill++] = child;
                        mass_accum += (double)trie.edge_mass[child];
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

        for (int curriculum_d = curriculum_d_start; curriculum_d <= curriculum_d_end; curriculum_d++) {
        // Iterate over root-child subtrees. Each subtree is one training unit:
        // weights fixed throughout forward+backward+grad-aggregation, one Adam step.
        for (int rc_idx = 0; rc_idx < n_root_children; rc_idx++) {
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
            if (!accumulate) {
                CUDA_CHECK(cudaMemset(d_grads, 0, wo.total_floats * sizeof(float)));
            }

            // Chunk by total queries ≤ CHUNK_QUERIES
            int chunk_start = 0;
            while (chunk_start < n_at_depth) {
                // Accumulate radix nodes until total queries exceeds CHUNK_QUERIES or we hit N_cap
                int chunk_end = chunk_start;
                int T_q = 0;
                while (chunk_end < n_at_depth) {
                    int r = radix_list[subtree_offset + chunk_end];
                    int L = trie.edge_lens[r];
                    if (T_q + L > T_q_cap || chunk_end - chunk_start >= N_cap) break;
                    T_q += L;
                    chunk_end++;
                }
                if (chunk_end == chunk_start) {
                    // One node too big — skip (shouldn't happen at sane CHUNK_QUERIES)
                    chunk_start++;
                    continue;
                }
                int N = chunk_end - chunk_start;

                // Build per-chunk host arrays
                int* h_radix_ids      = (int*)malloc(N * sizeof(int));
                int* h_query_offsets  = (int*)malloc((N + 1) * sizeof(int));
                int* h_kv_offsets     = (int*)malloc((N + 1) * sizeof(int));
                int* h_kv_lengths     = (int*)malloc(N * sizeof(int));
                int* h_query_to_node  = (int*)malloc(T_q * sizeof(int));
                int* h_token_ids      = (int*)malloc(T_q * sizeof(int));
                int* h_rope_positions = (int*)malloc(T_q * H * sizeof(int));
                int* h_char_pos       = (int*)malloc(T_q * sizeof(int));

                int q_fill = 0;
                int kv_fill = 0;
                for (int i = 0; i < N; i++) {
                    int r = radix_list[subtree_offset + chunk_start + i];
                    h_radix_ids[i] = r;
                    int L = trie.edge_lens[r];
                    int anc_len = trie.ancestor_char_offsets[r + 1] - trie.ancestor_char_offsets[r];
                    int K_i = anc_len + L;
                    h_query_offsets[i] = q_fill;
                    h_kv_offsets[i] = kv_fill;
                    h_kv_lengths[i] = K_i;

                    int edge_start = trie.edge_starts[r];
                    int fcd = trie.edge_first_char_depths[r];
                    for (int j = 0; j < L; j++) {
                        h_query_to_node[q_fill + j] = i;
                        h_token_ids[q_fill + j] = trie.edge_tokens_flat[edge_start + j];
                        int pos = fcd + j - 1;  // position = absolute depth - 1 (matches leveled)
                        if (pos < 0) pos = 0;
                        if (pos >= cfg.seq_len) pos = cfg.seq_len - 1;
                        for (int h = 0; h < H; h++)
                            h_rope_positions[(q_fill + j) * H + h] = pos;
                        h_char_pos[q_fill + j] = edge_start + j;
                    }
                    q_fill += L;
                    kv_fill += K_i;
                }
                h_query_offsets[N] = q_fill;
                h_kv_offsets[N] = kv_fill;
                int T_kv = kv_fill;

                // Guard against T_kv overflow of packed buffers
                if ((long long)T_kv > T_kv_max) {
                    fprintf(stderr, "Chunk T_kv=%d exceeds T_kv_max=%lld; skip\n", T_kv, T_kv_max);
                    free(h_radix_ids); free(h_query_offsets); free(h_kv_offsets); free(h_kv_lengths);
                    free(h_query_to_node); free(h_token_ids); free(h_rope_positions); free(h_char_pos);
                    chunk_start = chunk_end;
                    continue;
                }

                // Find max prefix length for this chunk (for kernel smem)
                int max_kv_len = 0;
                for (int i = 0; i < N; i++) if (h_kv_lengths[i] > max_kv_len) max_kv_len = h_kv_lengths[i];

                // --- Ancestor + own-edge split for compact-cache gather ---
                // Build flat ancestor ids (cache side) + per-query anc/own lengths.
                // Own-edge K/V is copied from the fresh d_k/d_v buffer; ancestors
                // come from the compact cache (mass=1 positions skipped).
                int T_anc = 0;
                for (int i = 0; i < N; i++) {
                    int r = h_radix_ids[i];
                    T_anc += trie.ancestor_char_offsets[r + 1] - trie.ancestor_char_offsets[r];
                }
                int* h_anc_ids      = (int*)malloc((T_anc > 0 ? T_anc : 1) * sizeof(int));
                int* h_anc_offsets  = (int*)malloc((N + 1) * sizeof(int));
                int* h_anc_lengths  = (int*)malloc(N * sizeof(int));
                int* h_own_lengths  = (int*)malloc(N * sizeof(int));
                {
                    int fill = 0;
                    for (int i = 0; i < N; i++) {
                        h_anc_offsets[i] = fill;
                        int r = h_radix_ids[i];
                        int anc_off = trie.ancestor_char_offsets[r];
                        int anc_len = trie.ancestor_char_offsets[r + 1] - anc_off;
                        for (int a = 0; a < anc_len; a++)
                            h_anc_ids[fill++] = trie.ancestor_char_ids[anc_off + a];
                        h_anc_lengths[i] = anc_len;
                        h_own_lengths[i] = trie.edge_lens[r];
                    }
                    h_anc_offsets[N] = fill;
                }
                static int* d_anc_ids_cache      = NULL; static int d_anc_ids_cap      = 0;
                static int* d_anc_offsets_cache  = NULL; static int d_anc_offsets_cap  = 0;
                static int* d_anc_lengths_cache  = NULL; static int d_anc_lengths_cap  = 0;
                static int* d_own_lengths_cache  = NULL; static int d_own_lengths_cap  = 0;
                if (T_anc > d_anc_ids_cap) {
                    if (d_anc_ids_cache) cudaFree(d_anc_ids_cache);
                    CUDA_CHECK(cudaMalloc(&d_anc_ids_cache, (T_anc > 0 ? T_anc : 1) * sizeof(int)));
                    d_anc_ids_cap = T_anc;
                }
                if (N + 1 > d_anc_offsets_cap) {
                    if (d_anc_offsets_cache) cudaFree(d_anc_offsets_cache);
                    CUDA_CHECK(cudaMalloc(&d_anc_offsets_cache, (N + 1) * sizeof(int)));
                    d_anc_offsets_cap = N + 1;
                }
                if (N > d_anc_lengths_cap) {
                    if (d_anc_lengths_cache) cudaFree(d_anc_lengths_cache);
                    CUDA_CHECK(cudaMalloc(&d_anc_lengths_cache, N * sizeof(int)));
                    d_anc_lengths_cap = N;
                }
                if (N > d_own_lengths_cap) {
                    if (d_own_lengths_cache) cudaFree(d_own_lengths_cache);
                    CUDA_CHECK(cudaMalloc(&d_own_lengths_cache, N * sizeof(int)));
                    d_own_lengths_cap = N;
                }
                if (T_anc > 0) {
                    CUDA_CHECK(cudaMemcpy(d_anc_ids_cache, h_anc_ids, T_anc * sizeof(int), cudaMemcpyHostToDevice));
                }
                CUDA_CHECK(cudaMemcpy(d_anc_offsets_cache, h_anc_offsets, (N + 1) * sizeof(int), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_anc_lengths_cache, h_anc_lengths, N * sizeof(int), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_own_lengths_cache, h_own_lengths, N * sizeof(int), cudaMemcpyHostToDevice));
                free(h_anc_ids); free(h_anc_offsets); free(h_anc_lengths); free(h_own_lengths);

                // Upload
                CUDA_CHECK(cudaMemcpy(d_radix_ids,      h_radix_ids,      N * sizeof(int), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_query_offsets,  h_query_offsets,  (N + 1) * sizeof(int), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_kv_offsets,     h_kv_offsets,     (N + 1) * sizeof(int), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_kv_lengths,     h_kv_lengths,     N * sizeof(int), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_query_to_node,  h_query_to_node,  T_q * sizeof(int), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_token_ids,      h_token_ids,      T_q * sizeof(int), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_rope_positions, h_rope_positions, T_q * H * sizeof(int), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_char_pos,       h_char_pos,       T_q * sizeof(int), cudaMemcpyHostToDevice));

                // Corpus-mass weighting. Raw edge_mass varies by 5+ orders of
                // magnitude in natural-language tries (common letters vs rare
                // prefixes). Directly weighting by mass/mean causes Adam to fail
                // because one high-mass sample dominates the update direction.
                //
                // We use log(1 + mass) compression, then normalize to mean 1 within
                // the chunk. This preserves ORDER (common > rare) while bounding
                // the per-step ratio to ~log(max_mass) / log(min_mass + 1) ≈ 10x.
                // Gradient stability wins out over exact linear-mass correspondence.
                if (mass_weight != MassWeightMode::Off) {
                    float* h_mass_weights = (float*)malloc(T_q * sizeof(float));
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
                    CUDA_CHECK(cudaMemcpy(d_mass_weights, h_mass_weights, T_q * sizeof(float), cudaMemcpyHostToDevice));
                    free(h_mass_weights);
                }

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
                    launch_kv_scatter_compact_bf16(d_k, d_char_pos, d_compact_slot, d_kv_keys[l],   T_q, D);
                    launch_kv_scatter_compact_bf16(d_v, d_char_pos, d_compact_slot, d_kv_values[l], T_q, D);

                    // Build packed prefix: [ancestors from cache | own-edge from fresh d_k/d_v]
                    // Ancestors: gather from compact cache (all ancestors are mass>1 → slot>=0).
                    // Own-edge: copy directly from this chunk's freshly-computed post-RoPE d_k / d_v.
                    launch_kv_gather_anc_compact_bf16(d_kv_keys[l], d_anc_ids_cache, d_anc_offsets_cache,
                                                      d_kv_offsets, d_anc_lengths_cache, d_compact_slot,
                                                      d_kv_pack_k, N, H, HD);
                    launch_kv_gather_anc_compact_bf16(d_kv_values[l], d_anc_ids_cache, d_anc_offsets_cache,
                                                      d_kv_offsets, d_anc_lengths_cache, d_compact_slot,
                                                      d_kv_pack_v, N, H, HD);
                    launch_kv_copy_own_edge(d_k, d_query_offsets, d_kv_offsets,
                                             d_anc_lengths_cache, d_own_lengths_cache,
                                             d_kv_pack_k, N, H, HD);
                    launch_kv_copy_own_edge(d_v, d_query_offsets, d_kv_offsets,
                                             d_anc_lengths_cache, d_own_lengths_cache,
                                             d_kv_pack_v, N, H, HD);

                    // L-query varlen attention
                    float scale = 1.0f / sqrtf((float)HD);
                    cuda_batched_varlen_attention_L_queries(
                        d_q, d_kv_pack_k, d_kv_pack_v,
                        d_query_to_node, d_query_offsets, d_kv_offsets, d_kv_lengths,
                        d_attn_out /* used as packed output temp */, sv_attn_weights[l],
                        T_q, H, HD, max_kv_len, scale);
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
                // = distribution CE. d_d_logits (per-query grad) written in place.
                launch_agpt_loss_per_query(d_logits, d_query_to_node, d_query_offsets,
                                            d_radix_ids, d_token_ids,
                                            d_radix_counts_offset, d_radix_counts_tok, d_radix_counts_val,
                                            (mass_weight != MassWeightMode::Off) ? d_mass_weights : NULL,
                                            d_d_logits, d_loss, T_q, V, entropy_lambda,
                                            intermediate_weight);

                float* h_loss = (float*)malloc(T_q * sizeof(float));
                CUDA_CHECK(cudaMemcpy(h_loss, d_loss, T_q * sizeof(float), cudaMemcpyDeviceToHost));
                int chunk_trained = 0;
                for (int i = 0; i < T_q; i++) if (h_loss[i] > 0.0f) { total_loss += h_loss[i]; chunk_trained++; }
                nodes_trained += chunk_trained;
                free(h_loss);

                // ---------- BACKWARD ----------
                // Scale by 1/chunk_trained where chunk_trained is now the number of
                // per-query loss terms (≈ T_q, not N).
                float grad_scale = (chunk_trained > 0) ? (1.0f / (float)chunk_trained) : 0.0f;

                // Output projection backward — all T_q rows.
                float* dG_out = d_grads + wo.out_w;
                // d_d_final_out[T_q, D] = d_d_logits[T_q, V] × W_out^T[V, D]
                CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, D, T_q, V,
                                          &alpha, W_out, V, d_d_logits, V, &beta_zero, d_d_final_out, D));
                // dW_out += d_final_out^T × d_d_logits (scaled)
                CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, V, D, T_q,
                                          &grad_scale, d_d_logits, V, d_final_out, D, &alpha, dG_out, V));

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

                    // FFN L2 backward: d_ff_out → d_ff_h
                    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, F, T_q, D,
                                              &alpha, W_2w, D, d_d_ff_out, D, &beta_zero, d_d_ff_h, F));
                    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, D, F, T_q,
                                              &grad_scale, d_d_ff_out, D, sv_ff_h[l], F, &alpha, dW_2w, D));

                    // ReLU backward
                    cuda_relu_backward(d_d_ff_h, sv_ff_mask[l], d_d_ff_h, T_q * F);

                    // FFN L1 backward: d_ff_h → d_ln2_out
                    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, D, T_q, F,
                                              &alpha, W_1w, F, d_d_ff_h, F, &beta_zero, d_d_ln_out, D));
                    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, F, D, T_q,
                                              &grad_scale, d_d_ff_h, F, sv_ln2_out[l], D, &alpha, dW_1w, F));

                    // LN2 backward
                    cuda_layer_norm_backward(d_d_ln_out, sv_ln2_norm[l], sv_ln2_std_inv[l],
                                              G2, d_d_ln_out, dG2, dB2, T_q, D);
                    launch_elem_add(d_dx, d_d_ln_out, T_q * D);  // residual 2 skip

                    // WO backward: d_dx → d_attn_out
                    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, D, T_q, D,
                                              &alpha, W_ow, D, d_dx, D, &beta_zero, d_d_attn_out, D));
                    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, D, D, T_q,
                                              &grad_scale, d_dx, D, sv_attn_out[l], D, &alpha, dW_ow, D));

                    // Attention backward (L-queries)
                    // Re-compute post-RoPE Q/K and V from saved ln1_out. We need
                    // fresh fp32 K/V values for:
                    //   - the own-edge portion of each query's prefix (mass=1
                    //     positions aren't in the cache)
                    //   - the attention-backward kernel's own K/V input
                    // Ancestor K/V still comes from the compact cache.

                    // Recompute Q (for attention backward)
                    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, D, T_q, D,
                                              &alpha, d_weights + wo.wq_w[l], D,
                                              sv_ln1_out[l], D, &beta_zero, d_q, D));
                    cuda_bias_add(d_q, d_weights + wo.wq_b[l], T_q, D);
                    launch_rope_batched(d_q, d_rope_positions, d_rope_cos, d_rope_sin, T_q * H, HD);

                    // Recompute K (post-RoPE) — overwrites d_k with fresh values
                    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, D, T_q, D,
                                              &alpha, d_weights + wo.wk_w[l], D,
                                              sv_ln1_out[l], D, &beta_zero, d_k, D));
                    cuda_bias_add(d_k, d_weights + wo.wk_b[l], T_q, D);
                    launch_rope_batched(d_k, d_rope_positions, d_rope_cos, d_rope_sin, T_q * H, HD);

                    // Recompute V (no RoPE) — overwrites d_v
                    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, D, T_q, D,
                                              &alpha, d_weights + wo.wv_w[l], D,
                                              sv_ln1_out[l], D, &beta_zero, d_v, D));
                    cuda_bias_add(d_v, d_weights + wo.wv_b[l], T_q, D);

                    // Gather packed K/V for backward: ancestors from compact cache,
                    // own-edge from freshly-recomputed d_k/d_v.
                    launch_kv_gather_anc_compact_bf16(d_kv_keys[l], d_anc_ids_cache, d_anc_offsets_cache,
                                                      d_kv_offsets, d_anc_lengths_cache, d_compact_slot,
                                                      d_kv_pack_k, N, H, HD);
                    launch_kv_gather_anc_compact_bf16(d_kv_values[l], d_anc_ids_cache, d_anc_offsets_cache,
                                                      d_kv_offsets, d_anc_lengths_cache, d_compact_slot,
                                                      d_kv_pack_v, N, H, HD);
                    launch_kv_copy_own_edge(d_k, d_query_offsets, d_kv_offsets,
                                             d_anc_lengths_cache, d_own_lengths_cache,
                                             d_kv_pack_k, N, H, HD);
                    launch_kv_copy_own_edge(d_v, d_query_offsets, d_kv_offsets,
                                             d_anc_lengths_cache, d_own_lengths_cache,
                                             d_kv_pack_v, N, H, HD);

                    // Zero dK/dV packed buffers
                    CUDA_CHECK(cudaMemset(d_dk_pack, 0, (long long)T_kv * H * HD * sizeof(float)));
                    CUDA_CHECK(cudaMemset(d_dv_pack, 0, (long long)T_kv * H * HD * sizeof(float)));

                    float scale = 1.0f / sqrtf((float)HD);
                    cuda_batched_varlen_attention_L_queries_backward(
                        d_q, d_kv_pack_k, d_kv_pack_v, sv_attn_weights[l], d_d_attn_out,
                        d_query_to_node, d_query_offsets, d_kv_offsets, d_kv_lengths,
                        d_dq_pack, d_dk_pack, d_dv_pack,
                        T_q, H, HD, max_kv_len, scale);

                    // Inverse RoPE on dQ
                    launch_rope_batched_inverse(d_dq_pack, d_rope_positions, d_rope_cos, d_rope_sin, T_q * H, HD);

                    // dQ → d_ln1_out via Wq^T; dWq += ln1_out^T × dQ
                    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, D, T_q, D,
                                              &alpha, d_weights + wo.wq_w[l], D,
                                              d_dq_pack, D, &beta_zero, d_d_ln_out, D));
                    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, D, D, T_q,
                                              &grad_scale, d_dq_pack, D,
                                              sv_ln1_out[l], D, &alpha, dW_qw, D));

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

                free(h_radix_ids); free(h_query_offsets); free(h_kv_offsets); free(h_kv_lengths);
                free(h_query_to_node); free(h_token_ids); free(h_rope_positions); free(h_char_pos);

                chunks_processed++;
                chunk_start = chunk_end;
            }  // end chunk loop — one subtree done

            // ONE Adam step per subtree. This is the AGPT training-unit boundary:
            // all descendant-branch gradients sharing a prefix inside this subtree
            // have been accumulated into d_grads (Jacobian factorization realized
            // via additive gradient accumulation).
            // In --accumulate mode, we skip the per-split/per-rc step and fire one
            // after all loops exit (see below).
            if (!accumulate) {
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

            // Grad clipping (applies to the accumulated chunk-gradient sum for this
            // subtree-split before the optimizer uses it).
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
            }
            if (weight_decay > 0.0f) {
                cuda_weight_decay(d_weights, step_lr, weight_decay, wo.total_floats);
            }
            subtrees_trained++;
        }

        clock_gettime(CLOCK_MONOTONIC, &t1);
        double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        float mean_loss = nodes_trained > 0 ? (float)(total_loss / nodes_trained) : 0.0f;
        if (!quiet) {
            printf("Epoch %d: loss=%.6f  (%.2f sec, %d subtrees, %d chunks, %d nodes)\n",
                   epoch + 1, mean_loss, elapsed, subtrees_trained, chunks_processed, nodes_trained);
            if (lightning_active) {
                double mean_size = (n_root_children > 0) ? (double)lightning_nodes_sum / n_root_children : 0.0;
                printf("  lightning depth histogram (sample endpoint depth):");
                int max_seen = 0;
                for (int d = 0; d < 64; d++) if (lightning_depth_hist[d] > 0 && d > max_seen) max_seen = d;
                for (int d = 0; d <= max_seen; d++) {
                    if (lightning_depth_hist[d] > 0) printf(" d%d:%d", d, lightning_depth_hist[d]);
                }
                printf("  mean_size=%.0f  mass[min=%.0f mean=%.0f max=%.0f]",
                       mean_size, lightning_mass_min, lightning_mean_mass, lightning_mass_max);
                if (lightning.mass_lr != MassWeightMode::Off) {
                    printf("  lr_scale[min=%.3f max=%.3f]", lightning_w_min, lightning_w_max);
                }
                printf("\n");
            }
        }

        // Intermediate checkpoint every save_every epochs. External tooling
        // (bin/perplexity) can score these to find the best-held-out stopping point.
        if (save_every > 0 && save_path && (epoch + 1) % save_every == 0) {
            char ck_path[2048];
            snprintf(ck_path, sizeof(ck_path), "%s.ep%d", save_path, epoch + 1);
            CUDA_CHECK(cudaMemcpy(h_weights, d_weights, wo.total_floats * sizeof(float), cudaMemcpyDeviceToHost));
            save_model_weights(ck_path, cfg, h_weights, wo);
        }

        if (lightning_subtree_mass) { free(lightning_subtree_mass); lightning_subtree_mass = NULL; }
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
    cudaFree(d_weights); cudaFree(d_grads); cudaFree(d_adam_m); cudaFree(d_adam_v);
    if (d_clip_partials) cudaFree(d_clip_partials);
    if (d_clip_norm)     cudaFree(d_clip_norm);
    for (int l = 0; l < L_layers; l++) {
        cudaFree(d_kv_keys[l]); cudaFree(d_kv_values[l]);
    }
    free(d_kv_keys); free(d_kv_values);
    cudaFree(d_rope_cos); cudaFree(d_rope_sin);
    cudaFree(d_x); cudaFree(d_x_res1); cudaFree(d_x_res2); cudaFree(d_ln_out);
    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_attn_out);
    cudaFree(d_ff_h); cudaFree(d_ff_mask); cudaFree(d_ff_out);
    cudaFree(d_final_out); cudaFree(d_final_norm_save); cudaFree(d_final_std_inv_save);
    cudaFree(d_logits); cudaFree(d_d_logits); cudaFree(d_loss); cudaFree(d_d_final_out);
    for (int l = 0; l < L_layers; l++) {
        cudaFree(sv_x_res1[l]); cudaFree(sv_ln1_norm[l]); cudaFree(sv_ln1_std_inv[l]); cudaFree(sv_ln1_out[l]);
        cudaFree(sv_x_res2[l]); cudaFree(sv_ln2_norm[l]); cudaFree(sv_ln2_std_inv[l]); cudaFree(sv_ln2_out[l]);
        cudaFree(sv_ff_h[l]); cudaFree(sv_ff_mask[l]); cudaFree(sv_attn_out[l]); cudaFree(sv_attn_weights[l]);
    }
    free(sv_x_res1); free(sv_ln1_norm); free(sv_ln1_std_inv); free(sv_ln1_out);
    free(sv_x_res2); free(sv_ln2_norm); free(sv_ln2_std_inv); free(sv_ln2_out);
    free(sv_ff_h); free(sv_ff_mask); free(sv_attn_out); free(sv_attn_weights);
    cudaFree(d_q_pack_flat); cudaFree(d_kv_pack_k); cudaFree(d_kv_pack_v);
    cudaFree(d_dq_pack); cudaFree(d_dk_pack); cudaFree(d_dv_pack);
    cudaFree(d_radix_counts_offset); cudaFree(d_radix_counts_tok); cudaFree(d_radix_counts_val);
    cudaFree(d_radix_ids); cudaFree(d_query_to_node); cudaFree(d_query_offsets);
    cudaFree(d_kv_offsets); cudaFree(d_kv_lengths); cudaFree(d_token_ids);
    cudaFree(d_rope_positions); cudaFree(d_char_pos);
    if (d_mass_weights) cudaFree(d_mass_weights);
    free(root_child_of); free(root_children);
    for (int i = 0; i < n_root_children; i++) free(subtree_nodes[i]);
    free(subtree_nodes); free(subtree_sizes);
    if (lightning_children_offsets) free(lightning_children_offsets);
    if (lightning_children_flat)    free(lightning_children_flat);
    if (compact_slot)    free(compact_slot);
    if (d_compact_slot)  cudaFree(d_compact_slot);
    if (depth_limit) {
        for (int i = 0; i < n_root_children; i++) free(depth_limit[i]);
        free(depth_limit);
    }

    cublasDestroy(cublas);
    if (!quiet) printf("Done.\n");
    return 0;
}

// ============================================================================
// Per-subtree training path (task #43)
// ============================================================================
//
// For deep tries (d=32+) the global KV cache won't fit in RAM. The per-subtree
// radix format splits the trie into one file per root-child, each with a
// self-contained local index space. This function loads one subtree at a
// time, sizes the KV cache to just that subtree's character count, runs one
// Adam/RMSProp step on it, frees the KV cache, and moves on.
//
// Optimizer state (Adam m/v, RMSProp s, step counter) persists in host buffers
// across subtree calls so the running averages don't reset each subtree. Weights
// persist via the caller's h_weights buffer which run_radix_training updates
// in-place.
int run_per_subtree_training(const Config& cfg_in, const WeightOffsets& wo,
                              float* h_weights,
                              const SubtreeManifest& manifest,
                              int super_epochs, float entropy_lambda, MassWeightMode mass_weight,
                              int subtree_splits, int partition_depth, bool accumulate,
                              bool single_subtree, float intermediate_weight,
                              OptimizerKind optimizer, float momentum_beta, float rmsprop_beta,
                              LRSchedule lr_schedule, int warmup_super_epochs,
                              float weight_decay, float grad_clip_norm, int save_every,
                              CurriculumMode curriculum, const char* save_path,
                              bool lr_scale_by_steps = false,
                              LightningConfig lightning = LightningConfig{})
{
    // Auto-LR scaling: the optimal LR depends on total gradient-movement per pass
    // (lr × steps_per_super_epoch ≈ constant for a fixed depth). The winning d=16
    // unigram recipe was calibrated at 65 steps/super-epoch, lr=3e-3. When the
    // user changes subtree granularity (bigram: 1465 steps, trigram later) we
    // rescale lr so the same base_lr knob keeps working. Reference step count is
    // hardcoded at 65 because that's what the shipped recipe in memory was
    // calibrated against — don't change without a fresh calibration.
    Config cfg = cfg_in;
    const int LR_SCALE_REFERENCE_STEPS = 65;
    const bool lightning_active = (lightning.steps > 0);
    int steps_per_super_epoch = lightning_active
        ? lightning.steps
        : manifest.n_subtrees * subtree_splits;
    if (lr_scale_by_steps && steps_per_super_epoch > 0) {
        float scale = (float)LR_SCALE_REFERENCE_STEPS / (float)steps_per_super_epoch;
        float scaled_lr = cfg_in.lr * scale;
        printf("Per-subtree training: %d subtrees, %d super-epochs (lr auto-scaled %.4g → %.4g × %d/%d)\n",
               manifest.n_subtrees, super_epochs, cfg_in.lr, scaled_lr,
               LR_SCALE_REFERENCE_STEPS, steps_per_super_epoch);
        cfg.lr = scaled_lr;
    } else {
        printf("Per-subtree training: %d subtrees, %d super-epochs%s\n",
               manifest.n_subtrees, super_epochs,
               lightning_active ? " (Lightning stochastic sampling)" : "");
    }
    if (lightning_active) {
        const char* sname = (lightning.sampler == LightningSampler::L1_Uniform) ? "l1-uniform"
                          : (lightning.sampler == LightningSampler::L2_RcDepth) ? "l2-rc-depth"
                          :                                                        "l3-mass-walk";
        printf("  lightning: %s, %d samples/SE total, p_stop=%.2f, seed=0x%x\n",
               sname, lightning.steps, lightning.p_stop, lightning.seed);
    }

    const char* opt_name = (optimizer == OptimizerKind::Adam)     ? "adam"
                         : (optimizer == OptimizerKind::SGD)      ? "sgd"
                         : (optimizer == OptimizerKind::Momentum) ? "momentum"
                         :                                          "rmsprop";
    printf("  optimizer: %s (lr=%.4g)\n", opt_name, cfg.lr);
    const char* sched_name = (lr_schedule == LRSchedule::Constant)    ? "constant"
                           : (lr_schedule == LRSchedule::Cosine)       ? "cosine"
                           :                                             "warmup-cosine";
    printf("  lr-schedule: %s (warmup %d super-epochs = %d steps)\n",
           sched_name, warmup_super_epochs, warmup_super_epochs * manifest.n_subtrees * subtree_splits);
    if (entropy_lambda > 0.0f) printf("  entropy lambda: %.3f\n", entropy_lambda);
    if (mass_weight != MassWeightMode::Off) {
        const char* mode_name = (mass_weight == MassWeightMode::Log)    ? "log"
                              : (mass_weight == MassWeightMode::Sqrt)   ? "sqrt"
                              : (mass_weight == MassWeightMode::Linear) ? "linear"
                              :                                            "?";
        printf("  mass weighting: %s\n", mode_name);
    }
    if (single_subtree) printf("  single-subtree (per file): 1 Adam step per subtree per super-epoch\n");

    // Allocate optimizer-state host buffers so state persists across the many
    // run_radix_training invocations below.
    float* h_adam_m = (float*)calloc(wo.total_floats, sizeof(float));
    float* h_adam_v = (float*)calloc(wo.total_floats, sizeof(float));
    int adam_t = 0;

    // Total optimizer steps across the whole training (for cosine horizon).
    int total_opt_steps = super_epochs * steps_per_super_epoch;
    int warmup_steps    = warmup_super_epochs * steps_per_super_epoch;

    printf("  total optimizer steps: %d (%d per super-epoch)\n",
           total_opt_steps, steps_per_super_epoch);

    // Largest subtree (by char count) paced first to surface OOM early.
    int largest_idx = 0;
    long long largest_chars = manifest.entries[0].total_edge_chars;
    for (int i = 1; i < manifest.n_subtrees; i++) {
        if (manifest.entries[i].total_edge_chars > largest_chars) {
            largest_chars = manifest.entries[i].total_edge_chars;
            largest_idx = i;
        }
    }
    printf("  largest subtree: rc=%d, %lld chars (peak per-subtree KV ≈ %.1f MB)\n",
           manifest.entries[largest_idx].root_child_id, largest_chars,
           largest_chars * cfg.d_model * 4.0 * 2 * cfg.n_layers / 1e6);

    // Lightning over per-subtree files: pre-sample root-child indices per SE,
    // weighted by each subtree's total_edge_chars (a proxy for corpus mass
    // flowing through that root-child; more principled would be to read each
    // subtree's edge_mass[0] but that requires a one-time scan). Each
    // root-child with ≥1 bucketed sample gets loaded once per SE and runs
    // that many Lightning samples within its local view.
    unsigned lightning_outer_rng = lightning.seed;
    double* lightning_rc_weights = NULL;
    double lightning_rc_total_weight = 0.0;
    if (lightning_active) {
        lightning_rc_weights = (double*)malloc(manifest.n_subtrees * sizeof(double));
        for (int i = 0; i < manifest.n_subtrees; i++) {
            lightning_rc_weights[i] = (double)manifest.entries[i].total_edge_chars;
            lightning_rc_total_weight += lightning_rc_weights[i];
        }
        if (lightning_rc_total_weight <= 0.0) lightning_rc_total_weight = 1.0;
    }

    for (int ep = 0; ep < super_epochs; ep++) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        double super_loss_sum = 0.0;
        long long super_nodes_trained = 0;
        int subtrees_done = 0;

        // ---- Lightning path ----
        if (lightning_active) {
            // Bucket this SE's lightning.steps samples by root-child index,
            // weighted by total_edge_chars.
            int* bucket = (int*)calloc(manifest.n_subtrees, sizeof(int));
            for (int s = 0; s < lightning.steps; s++) {
                double u = (double)xorshift_float01(&lightning_outer_rng) * lightning_rc_total_weight;
                double acc = 0.0;
                int pick = manifest.n_subtrees - 1;
                for (int i = 0; i < manifest.n_subtrees; i++) {
                    acc += lightning_rc_weights[i];
                    if (u <= acc) { pick = i; break; }
                }
                bucket[pick]++;
            }
            int touched = 0, max_bucket = 0;
            for (int i = 0; i < manifest.n_subtrees; i++) {
                if (bucket[i] > 0) touched++;
                if (bucket[i] > max_bucket) max_bucket = bucket[i];
            }
            printf("  SE %d lightning buckets: %d root-children touched, max %d samples/rc\n",
                   ep + 1, touched, max_bucket);

            for (int i = 0; i < manifest.n_subtrees; i++) {
                if (bucket[i] == 0) continue;
                SubtreeData s = load_subtree(manifest, i);
                RadixView view = subtree_to_radix_view(s);

                TrainPersistence persist;
                persist.h_adam_m_io = h_adam_m;
                persist.h_adam_v_io = h_adam_v;
                persist.adam_t_io = &adam_t;
                persist.quiet = true;
                persist.total_opt_steps_override = total_opt_steps;
                persist.warmup_steps_override = warmup_steps;

                // Build a per-call Lightning config with steps scoped to this
                // root-child's bucket. Reuse the same sampler/p_stop/mass-lr
                // settings but derive the inner RNG seed from seed+ep+rc so
                // runs are reproducible and bucket sequences diverge across SEs.
                LightningConfig inner = lightning;
                inner.steps = bucket[i];
                inner.seed  = lightning.seed ^ (unsigned)(0x9E3779B9u * (ep + 1)) ^ (unsigned)(0x85EBCA6Bu * (i + 1));

                run_radix_training(cfg, wo, h_weights, view.t,
                                   /*epochs=*/1, entropy_lambda, mass_weight, subtree_splits, partition_depth, accumulate,
                                   /*single_subtree=*/true, intermediate_weight,
                                   optimizer, momentum_beta, rmsprop_beta,
                                   lr_schedule, warmup_super_epochs,
                                   weight_decay, grad_clip_norm, /*save_every=*/0,
                                   curriculum, /*save_path=*/NULL,
                                   inner, &persist);

                super_nodes_trained += s.n_nodes;
                subtrees_done++;

                free_radix_view(view);
                free_subtree(s);
            }
            free(bucket);
        } else {
        // ---- Deterministic per-root-child path (existing) ----
        for (int ii = 0; ii < manifest.n_subtrees; ii++) {
            int i = ii;
            if (ep == 0 && ii == 0) i = largest_idx;
            else if (ep == 0 && ii == largest_idx) i = 0;

            SubtreeData s = load_subtree(manifest, i);
            RadixView view = subtree_to_radix_view(s);

            TrainPersistence persist;
            persist.h_adam_m_io = h_adam_m;
            persist.h_adam_v_io = h_adam_v;
            persist.adam_t_io = &adam_t;
            persist.quiet = true;
            persist.total_opt_steps_override = total_opt_steps;
            persist.warmup_steps_override = warmup_steps;

            // One subtree, one Adam/RMSProp step (single_subtree semantics per file).
            // Save path is deferred to the super-epoch level below.
            run_radix_training(cfg, wo, h_weights, view.t,
                               /*epochs=*/1, entropy_lambda, mass_weight, subtree_splits, partition_depth, accumulate,
                               /*single_subtree=*/true, intermediate_weight,
                               optimizer, momentum_beta, rmsprop_beta,
                               lr_schedule, warmup_super_epochs,
                               weight_decay, grad_clip_norm, /*save_every=*/0,
                               curriculum, /*save_path=*/NULL,
                               /*lightning=*/LightningConfig{},
                               &persist);

            super_nodes_trained += s.n_nodes;
            subtrees_done++;

            free_radix_view(view);
            free_subtree(s);
        }
        }

        clock_gettime(CLOCK_MONOTONIC, &t1);
        double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        printf("Super-epoch %d: %d subtrees, %lld radix nodes  (%.1f sec, adam_t=%d)\n",
               ep + 1, subtrees_done, super_nodes_trained, elapsed, adam_t);
        (void)super_loss_sum;  // loss is printed by run_radix_training when !quiet

        if (save_every > 0 && save_path && (ep + 1) % save_every == 0) {
            char ck_path[2048];
            snprintf(ck_path, sizeof(ck_path), "%s.ep%d", save_path, ep + 1);
            save_model_weights(ck_path, cfg, h_weights, wo);
            printf("  checkpoint: %s\n", ck_path);
        }
    }

    if (save_path) {
        save_model_weights(save_path, cfg, h_weights, wo);
        printf("Saved to %s\n", save_path);
    }
    free(h_adam_m); free(h_adam_v);
    if (lightning_rc_weights) free(lightning_rc_weights);
    printf("Done.\n");
    return 0;
}

// ============================================================================
// CLI + main
// ============================================================================

int main(int argc, char** argv) {
    const char* model_path = NULL;
    const char* trie_dir = NULL;
    const char* save_path = NULL;
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
    OptimizerKind optimizer = OptimizerKind::Adam;
    float momentum_beta = 0.9f;   // used by momentum + (via β₁) adam
    float rmsprop_beta = 0.999f;  // used by rmsprop + (via β₂) adam
    LRSchedule lr_schedule = LRSchedule::Constant;
    int warmup_epochs = 0;
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
        else if (strcmp(argv[i], "--optimizer") == 0 && i + 1 < argc) {
            const char* o = argv[++i];
            if      (strcmp(o, "adam")     == 0) optimizer = OptimizerKind::Adam;
            else if (strcmp(o, "sgd")      == 0) optimizer = OptimizerKind::SGD;
            else if (strcmp(o, "momentum") == 0) optimizer = OptimizerKind::Momentum;
            else if (strcmp(o, "rmsprop")  == 0) optimizer = OptimizerKind::RMSProp;
            else { fprintf(stderr, "Unknown optimizer '%s' (adam|sgd|momentum|rmsprop)\n", o); return 1; }
        }
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
            else { fprintf(stderr, "Unknown --lightning-sampler '%s' (l1|l2|l3)\n", s); return 1; }
        }
        else if (strcmp(argv[i], "--lightning-p-stop") == 0 && i + 1 < argc) lightning.p_stop = atof(argv[++i]);
        else if (strcmp(argv[i], "--lightning-seed") == 0 && i + 1 < argc) lightning.seed = (unsigned)strtoul(argv[++i], NULL, 0);
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
                        "  [--lightning-sampler l1|l2|l3] — sampler variant. Default l3 (mass-walk).\n"
                        "                                l1 = uniform over all radix nodes.\n"
                        "                                l2 = uniform over depth-1 root-children.\n"
                        "                                l3 = mass-weighted top-down walk with p_stop.\n"
                        "  [--lightning-p-stop F]      — L3 stop probability at each level (default 0.3).\n"
                        "  [--lightning-seed N]        — sampler RNG seed (default 0x5c115e1).\n"
                        "  [--lightning-mass-lr [off|log|sqrt|linear]] — per-sample LR scaling by\n"
                        "                                subtree mass. Bare flag = log (safest).\n"
                        "                                Each sample's step_lr is multiplied by\n"
                        "                                compress(subtree_mass[s]) / mean(compress).\n"
                        "                                Mean-normalized so average weight = 1.0.\n"
                        "                                linear can blow up RMSProp with a single\n"
                        "                                high-mass sample dominating; log is the\n"
                        "                                stable default. off = no scaling.\n"
                        "  [--save <path>]\n");
        return 1;
    }

    printf("AGPT CUDA Training Engine\n");

    // Load model
    Config cfg;
    cfg.lr = lr;
    cfg.chunk_queries = chunk_queries;
    float* h_weights = load_model_weights(model_path, &cfg);
    WeightOffsets wo = compute_offsets(cfg);

    // Detect trie format
    int format = detect_trie_format(trie_dir);
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
        return rc;
    }
    if (format == 1) {
        printf("Loading radix trie from %s...\n", trie_dir);
        RadixTrieData radix_trie = load_radix_trie(trie_dir);

        return run_radix_training(cfg, wo, h_weights, radix_trie, epochs, entropy_lambda, mass_weight, subtree_splits, partition_depth, accumulate, single_subtree, intermediate_weight, optimizer, momentum_beta, rmsprop_beta, lr_schedule, warmup_epochs, weight_decay, grad_clip_norm, save_every, curriculum, save_path, lightning);
    }

    // Load leveled trie
    printf("Loading trie from %s...\n", trie_dir);
    TrieData trie = load_trie(trie_dir);

    // Allocate GPU state
    printf("Allocating GPU memory...\n");
    TrainState state = allocate_train_state(cfg, trie, wo);

    // Upload weights
    CUDA_CHECK(cudaMemcpy(state.d_weights, h_weights, wo.total_floats * sizeof(float), cudaMemcpyHostToDevice));

    // cuBLAS handle
    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));

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
    printf("Done.\n");
    return 0;
}
