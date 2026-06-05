#ifndef AGPT_V2_CHUNK_PLAN_CUH
#define AGPT_V2_CHUNK_PLAN_CUH

#include <cstdlib>

#include "successor_prefix_v2.cuh"
#include "training_unit.cuh"

namespace agpt_v2 {

struct ChunkPlan {
    int chunk_index = 0;
    int start_node_index = 0;
    int end_node_index = 0;
    int node_count = 0;
    long long query_count = 0;
    long long kv_count = 0;
    long long compact_char_count = 0;
    int max_kv_len = 0;
};

struct ChunkPlanList {
    int chunk_count = 0;
    ChunkPlan* chunks = nullptr;
};

static inline void free_chunk_plan_list(ChunkPlanList& list) {
    std::free(list.chunks);
    list = ChunkPlanList{};
}

static inline ChunkPlanList build_chunk_plan_for_unit(const RadixTrieStructure& trie,
                                                      const TrainingUnit& unit,
                                                      int chunk_queries,
                                                      const SuccessorPrefixTableV2* successor_table = nullptr) {
    if (chunk_queries <= 0) chunk_queries = 50000;

    ChunkPlanList list;
    list.chunks = (ChunkPlan*)std::calloc(unit.node_count > 0 ? unit.node_count : 1, sizeof(ChunkPlan));

    int start = 0;
    while (start < unit.node_count) {
        long long q_sum = 0;
        long long kv_sum = 0;
        long long compact_sum = 0;
        int max_kv_len = 0;
        int end = start;
        while (end < unit.node_count) {
            int r = unit.radix_ids[end];
            int succ_len = successor_prefix_path_len_v2(trie, successor_table, r);
            int q_next = trie.edge_lens[r] + succ_len;
            if (end > start && q_sum + q_next > chunk_queries) break;
            q_sum += q_next;
            int kv_next = trie.edge_first_char_depths[r] + trie.edge_lens[r] - 1 + succ_len;
            kv_sum += kv_next;
            if (kv_next > max_kv_len) max_kv_len = kv_next;
            if (trie.edge_mass[r] > 1) compact_sum += trie.edge_lens[r];
            end++;
        }

        ChunkPlan& chunk = list.chunks[list.chunk_count];
        chunk.chunk_index = list.chunk_count;
        chunk.start_node_index = start;
        chunk.end_node_index = end;
        chunk.node_count = end - start;
        chunk.query_count = q_sum;
        chunk.kv_count = kv_sum;
        chunk.compact_char_count = compact_sum;
        chunk.max_kv_len = max_kv_len;
        list.chunk_count++;
        start = end;
    }

    return list;
}

}  // namespace agpt_v2

#endif
