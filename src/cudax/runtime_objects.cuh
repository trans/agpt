#ifndef AGPT_V2_RUNTIME_OBJECTS_CUH
#define AGPT_V2_RUNTIME_OBJECTS_CUH

#include <cstdlib>
#include <cstring>

#include "cuda_support.cuh"
#include "kernels_v2.cuh"
#include "runtime_contracts.cuh"

namespace agpt_v2 {

static inline cublasMath_t read_cublas_math_mode_v2() {
    const char* env = std::getenv("AGPT_V2_CUBLAS_MATH");
    if (!env || !env[0]) return CUBLAS_TF32_TENSOR_OP_MATH;
    if (std::strcmp(env, "fp32") == 0 || std::strcmp(env, "default") == 0) {
        return CUBLAS_DEFAULT_MATH;
    }
    if (std::strcmp(env, "tf32") == 0) {
        return CUBLAS_TF32_TENSOR_OP_MATH;
    }
    return CUBLAS_TF32_TENSOR_OP_MATH;
}

struct CacheRuntimeV2 {
    CacheRuntimeContract contract;
    int* d_compact_slot = nullptr;
    __nv_bfloat16** d_k_layers = nullptr;
    __nv_bfloat16** d_v_layers = nullptr;
};

struct ChunkRuntimeV2 {
    ChunkRuntimeContract contract;
    float* d_query_state = nullptr;
    float* d_packed_attention = nullptr;
    float* d_saved_activations = nullptr;
    float* d_logits_and_loss = nullptr;
};

struct TrainerRuntimeV2 {
    TrainerRuntimeContract contract;
    CacheRuntimeV2 cache;
    ChunkRuntimeV2 chunk;
    cublasHandle_t cublas = nullptr;
    float* d_rope_cos = nullptr;
    float* d_rope_sin = nullptr;
    float* d_weights = nullptr;
    float* d_grads = nullptr;
    float* d_opt_m = nullptr;
    float* d_opt_v = nullptr;
};

struct UnitAncGradRuntimeV2 {
    bool enabled = false;
    int subtree_compact_chars = 0;
    int* d_compact_to_subtree_idx = nullptr;
    int* d_subtree_real_pos = nullptr;
    float** d_dkv_subtree_k = nullptr;
    float** d_dkv_subtree_v = nullptr;
    float** d_h_subtree = nullptr;
};

static inline void init_unit_anc_grad_runtime_v2(UnitAncGradRuntimeV2& runtime,
                                                 const TrainerRuntimeContract& contract,
                                                 const TrainerConfig& cfg,
                                                 const TrainingUnit& unit,
                                                 const RadixTrieStructure& trie) {
    runtime.enabled = cfg.anc_grad;
    if (!runtime.enabled) return;

    int compact_cap = (int)contract.cache.compact_char_capacity;
    int H = contract.shape.n_heads;
    int D = contract.shape.d_model;
    int L = contract.shape.n_layers;
    int seq_len = contract.shape.seq_len;

    int* h_subtree_pos = (int*)std::malloc((size_t)((unit.compact_char_count > 0 ? unit.compact_char_count : 1) * H) * sizeof(int));
    int* h_subtree_slots = (int*)std::malloc((size_t)(unit.compact_char_count > 0 ? unit.compact_char_count : 1) * sizeof(int));

    int n_sub = 0;
    for (int i = 0; i < unit.node_count; i++) {
        int r = unit.radix_ids[i];
        if (trie.edge_mass[r] == 1) continue;
        int start_pos = trie.edge_starts[r];
        int len = trie.edge_lens[r];
        for (int c = 0; c < len; c++) {
            int char_pos = start_pos + c;
            int slot = trie.compact_slot[char_pos];
            if (slot < 0) continue;
            h_subtree_slots[n_sub] = slot;
            int pos = trie.real_pos_of_char[char_pos];
            if (pos < 0) pos = 0;
            if (pos >= seq_len) pos = seq_len - 1;
            for (int h = 0; h < H; h++) {
                h_subtree_pos[n_sub * H + h] = pos;
            }
            n_sub++;
        }
    }

    runtime.subtree_compact_chars = n_sub;
    AGPT_V2_CUDA_CHECK(cudaMalloc(&runtime.d_compact_to_subtree_idx, (size_t)(compact_cap > 0 ? compact_cap : 1) * sizeof(int)));
    AGPT_V2_CUDA_CHECK(cudaMemset(runtime.d_compact_to_subtree_idx, 0xff,
                                  (size_t)(compact_cap > 0 ? compact_cap : 1) * sizeof(int)));

    if (n_sub > 0) {
        int* d_subtree_slots = nullptr;
        AGPT_V2_CUDA_CHECK(cudaMalloc(&d_subtree_slots, (size_t)n_sub * sizeof(int)));
        AGPT_V2_CUDA_CHECK(cudaMemcpy(d_subtree_slots, h_subtree_slots,
                                      (size_t)n_sub * sizeof(int),
                                      cudaMemcpyHostToDevice));
        launch_set_compact_to_subtree_v2(runtime.d_compact_to_subtree_idx, d_subtree_slots, n_sub);
        AGPT_V2_CUDA_CHECK(cudaFree(d_subtree_slots));

        AGPT_V2_CUDA_CHECK(cudaMalloc(&runtime.d_subtree_real_pos, (size_t)n_sub * H * sizeof(int)));
        AGPT_V2_CUDA_CHECK(cudaMemcpy(runtime.d_subtree_real_pos, h_subtree_pos,
                                      (size_t)n_sub * H * sizeof(int),
                                      cudaMemcpyHostToDevice));
    }

    runtime.d_dkv_subtree_k = (float**)std::calloc((size_t)L, sizeof(float*));
    runtime.d_dkv_subtree_v = (float**)std::calloc((size_t)L, sizeof(float*));
    runtime.d_h_subtree = (float**)std::calloc((size_t)L, sizeof(float*));
    if (n_sub > 0) {
        size_t per_layer_bytes = (size_t)n_sub * D * sizeof(float);
        for (int l = 0; l < L; l++) {
            AGPT_V2_CUDA_CHECK(cudaMalloc(&runtime.d_dkv_subtree_k[l], per_layer_bytes));
            AGPT_V2_CUDA_CHECK(cudaMalloc(&runtime.d_dkv_subtree_v[l], per_layer_bytes));
            AGPT_V2_CUDA_CHECK(cudaMalloc(&runtime.d_h_subtree[l], per_layer_bytes));
        }
    }

    std::free(h_subtree_pos);
    std::free(h_subtree_slots);
}

static inline void zero_unit_anc_grad_runtime_v2(UnitAncGradRuntimeV2& runtime,
                                                 const TrainerRuntimeContract& contract) {
    if (!runtime.enabled || runtime.subtree_compact_chars <= 0) return;
    size_t fire_bytes = (size_t)runtime.subtree_compact_chars * contract.shape.d_model * sizeof(float);
    for (int l = 0; l < contract.shape.n_layers; l++) {
        AGPT_V2_CUDA_CHECK(cudaMemset(runtime.d_dkv_subtree_k[l], 0, fire_bytes));
        AGPT_V2_CUDA_CHECK(cudaMemset(runtime.d_dkv_subtree_v[l], 0, fire_bytes));
        AGPT_V2_CUDA_CHECK(cudaMemset(runtime.d_h_subtree[l], 0, fire_bytes));
    }
}

static inline void free_unit_anc_grad_runtime_v2(UnitAncGradRuntimeV2& runtime,
                                                 const TrainerRuntimeContract& contract) {
    if (runtime.d_dkv_subtree_k) {
        for (int l = 0; l < contract.shape.n_layers; l++) {
            if (runtime.d_dkv_subtree_k[l]) cudaFree(runtime.d_dkv_subtree_k[l]);
        }
        std::free(runtime.d_dkv_subtree_k);
    }
    if (runtime.d_dkv_subtree_v) {
        for (int l = 0; l < contract.shape.n_layers; l++) {
            if (runtime.d_dkv_subtree_v[l]) cudaFree(runtime.d_dkv_subtree_v[l]);
        }
        std::free(runtime.d_dkv_subtree_v);
    }
    if (runtime.d_h_subtree) {
        for (int l = 0; l < contract.shape.n_layers; l++) {
            if (runtime.d_h_subtree[l]) cudaFree(runtime.d_h_subtree[l]);
        }
        std::free(runtime.d_h_subtree);
    }
    if (runtime.d_subtree_real_pos) cudaFree(runtime.d_subtree_real_pos);
    if (runtime.d_compact_to_subtree_idx) cudaFree(runtime.d_compact_to_subtree_idx);
    runtime = UnitAncGradRuntimeV2{};
}

static inline void init_cache_runtime_v2(CacheRuntimeV2& runtime,
                                         const CacheRuntimeContract& contract,
                                         const RadixTrieStructure& trie) {
    runtime.contract = contract;
    AGPT_V2_CUDA_CHECK(cudaMalloc(&runtime.d_compact_slot, (size_t)trie.total_edge_chars * sizeof(int)));
    AGPT_V2_CUDA_CHECK(cudaMemcpy(runtime.d_compact_slot, trie.compact_slot,
                                  (size_t)trie.total_edge_chars * sizeof(int), cudaMemcpyHostToDevice));
    runtime.d_k_layers = (__nv_bfloat16**)std::calloc(contract.layer_count, sizeof(__nv_bfloat16*));
    runtime.d_v_layers = (__nv_bfloat16**)std::calloc(contract.layer_count, sizeof(__nv_bfloat16*));
    for (int l = 0; l < contract.layer_count; l++) {
        if (contract.uses_managed_memory) {
            AGPT_V2_CUDA_CHECK(cudaMallocManaged(&runtime.d_k_layers[l], contract.per_layer_k_bytes));
            AGPT_V2_CUDA_CHECK(cudaMallocManaged(&runtime.d_v_layers[l], contract.per_layer_v_bytes));
        } else {
            AGPT_V2_CUDA_CHECK(cudaMalloc(&runtime.d_k_layers[l], contract.per_layer_k_bytes));
            AGPT_V2_CUDA_CHECK(cudaMalloc(&runtime.d_v_layers[l], contract.per_layer_v_bytes));
        }
    }
}

static inline void zero_cache_runtime_v2(const CacheRuntimeV2& runtime) {
    for (int l = 0; l < runtime.contract.layer_count; l++) {
        AGPT_V2_CUDA_CHECK(cudaMemset(runtime.d_k_layers[l], 0, runtime.contract.per_layer_k_bytes));
        AGPT_V2_CUDA_CHECK(cudaMemset(runtime.d_v_layers[l], 0, runtime.contract.per_layer_v_bytes));
    }
}

static inline void free_cache_runtime_v2(CacheRuntimeV2& runtime) {
    if (runtime.d_compact_slot) cudaFree(runtime.d_compact_slot);
    if (runtime.d_k_layers) {
        for (int l = 0; l < runtime.contract.layer_count; l++) {
            if (runtime.d_k_layers[l]) cudaFree(runtime.d_k_layers[l]);
        }
        std::free(runtime.d_k_layers);
    }
    if (runtime.d_v_layers) {
        for (int l = 0; l < runtime.contract.layer_count; l++) {
            if (runtime.d_v_layers[l]) cudaFree(runtime.d_v_layers[l]);
        }
        std::free(runtime.d_v_layers);
    }
    runtime = CacheRuntimeV2{};
}

static inline void init_chunk_runtime_v2(ChunkRuntimeV2& runtime,
                                         const ChunkRuntimeContract& contract) {
    runtime.contract = contract;
    AGPT_V2_CUDA_CHECK(cudaMalloc(&runtime.d_query_state, contract.query_state_bytes > 0 ? contract.query_state_bytes : sizeof(float)));
    AGPT_V2_CUDA_CHECK(cudaMalloc(&runtime.d_packed_attention, contract.packed_attention_bytes > 0 ? contract.packed_attention_bytes : sizeof(float)));
    AGPT_V2_CUDA_CHECK(cudaMalloc(&runtime.d_saved_activations, contract.saved_activation_bytes > 0 ? contract.saved_activation_bytes : sizeof(float)));
    AGPT_V2_CUDA_CHECK(cudaMalloc(&runtime.d_logits_and_loss, contract.loss_and_logits_bytes > 0 ? contract.loss_and_logits_bytes : sizeof(float)));
}

static inline void free_chunk_runtime_v2(ChunkRuntimeV2& runtime) {
    if (runtime.d_query_state) cudaFree(runtime.d_query_state);
    if (runtime.d_packed_attention) cudaFree(runtime.d_packed_attention);
    if (runtime.d_saved_activations) cudaFree(runtime.d_saved_activations);
    if (runtime.d_logits_and_loss) cudaFree(runtime.d_logits_and_loss);
    runtime = ChunkRuntimeV2{};
}

static inline void init_trainer_runtime_v2(TrainerRuntimeV2& runtime,
                                           const TrainerRuntimeContract& contract,
                                           const RadixTrieStructure& trie) {
    runtime.contract = contract;
    init_cache_runtime_v2(runtime.cache, contract.cache, trie);
    init_chunk_runtime_v2(runtime.chunk, contract.chunk);
    AGPT_V2_CUDA_CHECK(cudaMalloc(&runtime.d_weights, contract.weight_and_grad_bytes / 2));
    AGPT_V2_CUDA_CHECK(cudaMalloc(&runtime.d_grads, contract.weight_and_grad_bytes / 2));
    AGPT_V2_CUDA_CHECK(cudaMalloc(&runtime.d_opt_m, contract.optimizer_state_bytes / 2));
    AGPT_V2_CUDA_CHECK(cudaMalloc(&runtime.d_opt_v, contract.optimizer_state_bytes / 2));
    build_rope_cache_v2(&runtime.d_rope_cos, &runtime.d_rope_sin,
                        contract.shape.seq_len, contract.shape.head_dim);
    AGPT_V2_CUBLAS_CHECK(cublasCreate(&runtime.cublas));
    AGPT_V2_CUBLAS_CHECK(cublasSetMathMode(runtime.cublas, read_cublas_math_mode_v2()));
}

static inline void free_trainer_runtime_v2(TrainerRuntimeV2& runtime) {
    free_cache_runtime_v2(runtime.cache);
    free_chunk_runtime_v2(runtime.chunk);
    if (runtime.d_rope_cos) cudaFree(runtime.d_rope_cos);
    if (runtime.d_rope_sin) cudaFree(runtime.d_rope_sin);
    if (runtime.cublas) cublasDestroy(runtime.cublas);
    if (runtime.d_weights) cudaFree(runtime.d_weights);
    if (runtime.d_grads) cudaFree(runtime.d_grads);
    if (runtime.d_opt_m) cudaFree(runtime.d_opt_m);
    if (runtime.d_opt_v) cudaFree(runtime.d_opt_v);
    runtime = TrainerRuntimeV2{};
}

}  // namespace agpt_v2

#endif
