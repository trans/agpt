#ifndef AGPT_V2_EXECUTION_PLAN_CUH
#define AGPT_V2_EXECUTION_PLAN_CUH

#include "chunk_plan.cuh"

namespace agpt_v2 {

struct ExecutionPlan {
    int training_unit_count = 0;
    long long total_node_count = 0;
    long long total_query_count = 0;
    long long total_compact_char_count = 0;
    long long estimated_chunk_count = 0;

    TrainingUnit* largest_by_queries = nullptr;
    TrainingUnit* largest_by_compact_chars = nullptr;
};

static inline ExecutionPlan build_execution_plan(const RadixTrieStructure& trie,
                                                 const TrainingPlan& training_plan,
                                                 int chunk_queries) {
    ExecutionPlan plan;
    plan.training_unit_count = training_plan.unit_count;

    for (int i = 0; i < training_plan.unit_count; i++) {
        TrainingUnit& unit = training_plan.units[i];
        plan.total_node_count += unit.node_count;
        plan.total_query_count += unit.query_count;
        plan.total_compact_char_count += unit.compact_char_count;

        ChunkPlanList chunks = build_chunk_plan_for_unit(trie, unit, chunk_queries);
        plan.estimated_chunk_count += chunks.chunk_count;

        if (!plan.largest_by_queries || unit.query_count > plan.largest_by_queries->query_count) {
            plan.largest_by_queries = &unit;
        }
        if (!plan.largest_by_compact_chars || unit.compact_char_count > plan.largest_by_compact_chars->compact_char_count) {
            plan.largest_by_compact_chars = &unit;
        }

        free_chunk_plan_list(chunks);
    }

    return plan;
}

}  // namespace agpt_v2

#endif
