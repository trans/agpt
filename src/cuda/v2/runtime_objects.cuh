#ifndef AGPT_V2_RUNTIME_OBJECTS_CUH
#define AGPT_V2_RUNTIME_OBJECTS_CUH

#include <cstdlib>

#include "cuda_support.cuh"
#include "kernels_v2.cuh"
#include "runtime_contracts.cuh"

namespace agpt_v2 {

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
