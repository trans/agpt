#ifndef AGPT_V2_TRAINING_UNIT_CUH
#define AGPT_V2_TRAINING_UNIT_CUH

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "io.cuh"

namespace agpt_v2 {

enum class TrainingUnitKind {
    RootChildSubtree,
    PartitionGroup,
    LightningSample,
};

struct TrainingUnit {
    TrainingUnitKind kind = TrainingUnitKind::RootChildSubtree;
    int unit_index = -1;
    int root_child_id = -1;
    int node_count = 0;
    long long query_count = 0;
    long long compact_char_count = 0;
    int max_endpoint_depth = 0;
    int prepended_ancestor_count = 0;
    int* radix_ids = nullptr;
};

struct TrainingPlan {
    int unit_count = 0;
    TrainingUnit* units = nullptr;
};

static inline int find_root_child(const RadixTrieStructure& trie, int r) {
    int cur = r;
    while (cur > 0 && trie.parents[cur] > 0) {
        cur = trie.parents[cur];
    }
    return cur;
}

static inline void free_training_plan(TrainingPlan& plan) {
    if (plan.units) {
        for (int i = 0; i < plan.unit_count; i++) {
            std::free(plan.units[i].radix_ids);
            plan.units[i].radix_ids = nullptr;
        }
        std::free(plan.units);
    }
    plan = TrainingPlan{};
}

static inline void sort_training_unit_radix_ids_by_endpoint_depth(const RadixTrieStructure& trie,
                                                                  TrainingUnit& unit) {
    if (unit.node_count <= 1 || unit.radix_ids == nullptr) return;

    std::vector<int> order(unit.radix_ids, unit.radix_ids + unit.node_count);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        int ea = trie.edge_first_char_depths[a] + trie.edge_lens[a] - 1;
        int eb = trie.edge_first_char_depths[b] + trie.edge_lens[b] - 1;
        if (ea != eb) return ea < eb;
        return a < b;
    });
    std::memcpy(unit.radix_ids, order.data(), (size_t)unit.node_count * sizeof(int));
}

static inline TrainingPlan build_pd1_training_plan(const RadixTrieStructure& trie) {
    TrainingPlan plan;

    int* counts = (int*)std::calloc(trie.radix_count, sizeof(int));
    char* seen = (char*)std::calloc(trie.radix_count, 1);
    for (int r = 1; r < trie.radix_count; r++) {
        int rc = find_root_child(trie, r);
        counts[rc]++;
        if (!seen[rc]) {
            seen[rc] = 1;
            plan.unit_count++;
        }
    }

    plan.units = (TrainingUnit*)std::calloc(plan.unit_count, sizeof(TrainingUnit));
    int* rc_to_unit = (int*)std::malloc(trie.radix_count * sizeof(int));
    for (int i = 0; i < trie.radix_count; i++) rc_to_unit[i] = -1;

    int unit_fill = 0;
    for (int rc = 1; rc < trie.radix_count; rc++) {
        if (!seen[rc]) continue;
        TrainingUnit& unit = plan.units[unit_fill];
        unit.kind = TrainingUnitKind::RootChildSubtree;
        unit.unit_index = unit_fill;
        unit.root_child_id = rc;
        unit.node_count = counts[rc];
        unit.radix_ids = (int*)std::malloc(unit.node_count * sizeof(int));
        rc_to_unit[rc] = unit_fill;
        unit_fill++;
    }

    int* fills = (int*)std::calloc(plan.unit_count, sizeof(int));
    for (int r = 1; r < trie.radix_count; r++) {
        int rc = find_root_child(trie, r);
        int unit_idx = rc_to_unit[rc];
        TrainingUnit& unit = plan.units[unit_idx];
        unit.radix_ids[fills[unit_idx]++] = r;
        unit.query_count += trie.edge_lens[r];
        if (trie.edge_mass[r] > 1) unit.compact_char_count += trie.edge_lens[r];
        int endpoint_depth = trie.edge_first_char_depths[r] + trie.edge_lens[r] - 1;
        if (endpoint_depth > unit.max_endpoint_depth) unit.max_endpoint_depth = endpoint_depth;
    }

    for (int i = 0; i < plan.unit_count; i++) {
        sort_training_unit_radix_ids_by_endpoint_depth(trie, plan.units[i]);
    }

    std::free(counts);
    std::free(seen);
    std::free(rc_to_unit);
    std::free(fills);
    return plan;
}

}  // namespace agpt_v2

#endif
