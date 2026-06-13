#ifndef AGPT_V2_RUNTIME_CONTRACTS_CUH
#define AGPT_V2_RUNTIME_CONTRACTS_CUH

#include <cstddef>

#include "execution_plan.cuh"

namespace agpt_v2 {

struct CacheRuntimeContract {
    long long compact_char_capacity = 0;
    int layer_count = 0;
    int d_model = 0;
    std::size_t per_layer_k_bytes = 0;
    std::size_t per_layer_v_bytes = 0;
    std::size_t total_bytes = 0;
    KCoordinateSpace k_space = KCoordinateSpace::PostRope;
    bool v_is_rope_free = true;
    bool uses_managed_memory = true;
};

struct ChunkRuntimeContract {
    int node_capacity = 0;
    int query_capacity = 0;
    long long kv_capacity = 0;
    int max_kv_len = 0;

    std::size_t query_state_bytes = 0;
    std::size_t packed_attention_bytes = 0;
    std::size_t saved_activation_bytes = 0;
    std::size_t loss_and_logits_bytes = 0;
    std::size_t total_bytes = 0;
};

struct TrainerRuntimeContract {
    RuntimeShape shape;
    int rope_seq_len = 0;
    CacheRuntimeContract cache;
    ChunkRuntimeContract chunk;
    std::size_t optimizer_state_bytes = 0;
    std::size_t weight_and_grad_bytes = 0;
    std::size_t total_bytes = 0;
};

static inline ChunkRuntimeContract build_chunk_runtime_contract(const RuntimeShape& shape,
                                                               const ChunkPlanList& chunks) {
    ChunkRuntimeContract contract;
    contract.query_capacity = 0;
    contract.node_capacity = 0;
    contract.kv_capacity = 0;
    contract.max_kv_len = 0;

    for (int i = 0; i < chunks.chunk_count; i++) {
        const ChunkPlan& chunk = chunks.chunks[i];
        if (chunk.query_count > contract.query_capacity) contract.query_capacity = (int)chunk.query_count;
        if (chunk.node_count > contract.node_capacity) contract.node_capacity = chunk.node_count;
        if (chunk.kv_count > contract.kv_capacity) contract.kv_capacity = chunk.kv_count;
        if (chunk.max_kv_len > contract.max_kv_len) contract.max_kv_len = chunk.max_kv_len;
    }

    const std::size_t f32 = sizeof(float);
    const std::size_t i32 = sizeof(int);
    const int D = shape.d_model;
    const int F = shape.d_ff;
    const int V = shape.vocab_size;
    const int L = shape.n_layers;

    long long Tq = contract.query_capacity;
    long long Tkv = contract.kv_capacity;
    long long N = contract.node_capacity;

    // Baseline ownership targets for the current v2 milestone:
    // query_state_bytes: x/ln/q/k/v/attn/ff scratch + integer query metadata
    contract.query_state_bytes =
        (std::size_t)(Tq * (6LL * D + 2LL * F) * f32) +
        (std::size_t)(Tq * 3LL * i32) +
        (std::size_t)(N * 3LL * i32);

    // packed_attention_bytes: packed K/V, packed dQ/dK/dV, attention output/weights
    contract.packed_attention_bytes =
        (std::size_t)(Tkv * 4LL * D * f32) +
        (std::size_t)(Tq * 2LL * D * f32) +
        (std::size_t)(Tq * (long long)shape.n_heads * (long long)contract.max_kv_len * f32);

    // saved_activation_bytes: per-layer saved ln/attn/ff activations used by backward
    contract.saved_activation_bytes =
        (std::size_t)L * (
            (std::size_t)(Tq * 8LL * D * f32) +
            (std::size_t)(Tq * 2LL * F * f32) +
            (std::size_t)(Tq * 2LL * D * f32) +
            (std::size_t)(Tq * 2LL * f32) +
            (std::size_t)(Tq * (long long)shape.n_heads * (long long)contract.max_kv_len * f32)
        );

    contract.loss_and_logits_bytes =
        (std::size_t)(Tq * (2LL * V + 2LL * D + 2LL) * f32);

    contract.total_bytes =
        contract.query_state_bytes +
        contract.packed_attention_bytes +
        contract.saved_activation_bytes +
        contract.loss_and_logits_bytes;

    return contract;
}

static inline CacheRuntimeContract build_cache_runtime_contract(const RuntimeShape& shape,
                                                               long long compact_chars,
                                                               int layer_count) {
    CacheRuntimeContract contract;
    contract.compact_char_capacity = compact_chars;
    contract.layer_count = layer_count;
    contract.d_model = shape.d_model;
    contract.per_layer_k_bytes = (std::size_t)(compact_chars * (long long)shape.d_model * sizeof(unsigned short));
    contract.per_layer_v_bytes = (std::size_t)(compact_chars * (long long)shape.d_model * sizeof(unsigned short));
    contract.total_bytes = (contract.per_layer_k_bytes + contract.per_layer_v_bytes) * (std::size_t)layer_count;
    return contract;
}

static inline TrainerRuntimeContract build_trainer_runtime_contract(const RuntimeShape& shape,
                                                                   const CacheLayout& cache_layout,
                                                                   const ExecutionPlan& plan,
                                                                   const ChunkPlanList& largest_chunks,
                                                                   long long compact_slot_capacity = 0) {
    TrainerRuntimeContract contract;
    contract.shape = shape;
    contract.rope_seq_len = shape.rope_seq_len > 0 ? shape.rope_seq_len : shape.seq_len;
    long long cache_capacity = plan.total_compact_char_count;
    if (cache_layout.compact_slot_indexed && compact_slot_capacity > 0) {
        cache_capacity = compact_slot_capacity;
    } else if (compact_slot_capacity > cache_capacity) {
        cache_capacity = compact_slot_capacity;
    }
    contract.cache = build_cache_runtime_contract(shape, cache_capacity, shape.n_layers);
    contract.cache.k_space = cache_layout.k_space;
    contract.cache.v_is_rope_free = cache_layout.v_is_rope_free;
    contract.cache.uses_managed_memory = true;
    contract.chunk = build_chunk_runtime_contract(shape, largest_chunks);

    const std::size_t f32 = sizeof(float);
    long long param_count = 0;
    param_count += (long long)shape.vocab_size * shape.d_model;
    for (int l = 0; l < shape.n_layers; l++) {
        param_count += 4LL * shape.d_model * shape.d_model;
        param_count += 4LL * shape.d_model;
        param_count += 2LL * shape.d_model;
        param_count += (long long)shape.d_model * shape.d_ff;
        param_count += shape.d_ff;
        param_count += (long long)shape.d_ff * shape.d_model;
        param_count += shape.d_model;
        param_count += 2LL * shape.d_model;
    }
    param_count += 2LL * shape.d_model;
    param_count += (long long)shape.d_model * shape.vocab_size + shape.vocab_size;

    contract.weight_and_grad_bytes = (std::size_t)(param_count * 2LL * f32);
    contract.optimizer_state_bytes = (std::size_t)(param_count * 2LL * f32);
    contract.total_bytes =
        contract.cache.total_bytes +
        contract.chunk.total_bytes +
        contract.weight_and_grad_bytes +
        contract.optimizer_state_bytes;

    return contract;
}

}  // namespace agpt_v2

#endif
