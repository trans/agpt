#ifndef AGPT_V2_TRAINING_UNIT_CUH
#define AGPT_V2_TRAINING_UNIT_CUH

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <utility>
#include <vector>

#include "io.cuh"
#include "types.cuh"

namespace agpt_v2 {

enum class TrainingUnitKind {
    WholeTrie,
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
    unsigned char* context_only = nullptr;
};

struct TrainingPlan {
    int unit_count = 0;
    TrainingUnit* units = nullptr;
};

static inline void free_training_unit(TrainingUnit& unit) {
    std::free(unit.radix_ids);
    std::free(unit.context_only);
    unit = TrainingUnit{};
}

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
            free_training_unit(plan.units[i]);
        }
        std::free(plan.units);
    }
    plan = TrainingPlan{};
}

static inline void sort_training_unit_radix_ids_by_endpoint_depth(const RadixTrieStructure& trie,
                                                                  TrainingUnit& unit) {
    if (unit.node_count <= 1 || unit.radix_ids == nullptr) return;

    std::vector<std::pair<int, unsigned char>> order;
    order.reserve((size_t)unit.node_count);
    for (int i = 0; i < unit.node_count; i++) {
        unsigned char flag = unit.context_only ? unit.context_only[i] : 0;
        order.push_back({unit.radix_ids[i], flag});
    }
    std::stable_sort(order.begin(), order.end(), [&](const auto& a, const auto& b) {
        int aid = a.first;
        int bid = b.first;
        int ea = trie.edge_first_char_depths[aid] + trie.edge_lens[aid] - 1;
        int eb = trie.edge_first_char_depths[bid] + trie.edge_lens[bid] - 1;
        if (ea != eb) return ea < eb;
        return aid < bid;
    });
    for (int i = 0; i < unit.node_count; i++) {
        unit.radix_ids[i] = order[(size_t)i].first;
        if (unit.context_only) unit.context_only[i] = order[(size_t)i].second;
    }
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

    std::vector<int> root_children;
    root_children.reserve((size_t)plan.unit_count);
    for (int rc = 1; rc < trie.radix_count; rc++) {
        if (seen[rc]) root_children.push_back(rc);
    }
    std::stable_sort(root_children.begin(), root_children.end(), [&](int a, int b) {
        int ta = trie.edge_lens[a] > 0 ? trie.edge_tokens_flat[trie.edge_starts[a]] : -1;
        int tb = trie.edge_lens[b] > 0 ? trie.edge_tokens_flat[trie.edge_starts[b]] : -1;
        if (ta != tb) return ta < tb;
        return a < b;
    });

    plan.units = (TrainingUnit*)std::calloc(plan.unit_count, sizeof(TrainingUnit));
    int* rc_to_unit = (int*)std::malloc(trie.radix_count * sizeof(int));
    for (int i = 0; i < trie.radix_count; i++) rc_to_unit[i] = -1;

    for (int unit_fill = 0; unit_fill < (int)root_children.size(); unit_fill++) {
        int rc = root_children[unit_fill];
        TrainingUnit& unit = plan.units[unit_fill];
        unit.kind = TrainingUnitKind::RootChildSubtree;
        unit.unit_index = unit_fill;
        unit.root_child_id = rc;
        unit.node_count = counts[rc];
        unit.radix_ids = (int*)std::malloc(unit.node_count * sizeof(int));
        rc_to_unit[rc] = unit_fill;
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

static inline TrainingPlan build_pd0_training_plan(const RadixTrieStructure& trie) {
    TrainingPlan plan;
    if (trie.radix_count <= 1) return plan;

    plan.unit_count = 1;
    plan.units = (TrainingUnit*)std::calloc(1, sizeof(TrainingUnit));

    TrainingUnit& unit = plan.units[0];
    unit.kind = TrainingUnitKind::WholeTrie;
    unit.unit_index = 0;
    unit.root_child_id = 0;
    unit.node_count = trie.radix_count - 1;
    unit.radix_ids = (int*)std::malloc(unit.node_count * sizeof(int));

    int fill = 0;
    for (int r = 1; r < trie.radix_count; r++) {
        unit.radix_ids[fill++] = r;
        unit.query_count += trie.edge_lens[r];
        if (trie.edge_mass[r] > 1) unit.compact_char_count += trie.edge_lens[r];
        int endpoint_depth = trie.edge_first_char_depths[r] + trie.edge_lens[r] - 1;
        if (endpoint_depth > unit.max_endpoint_depth) unit.max_endpoint_depth = endpoint_depth;
    }
    sort_training_unit_radix_ids_by_endpoint_depth(trie, unit);

    return plan;
}

struct LightningRngV2 {
    uint64_t state = 1;

    explicit LightningRngV2(uint64_t seed) {
        state = seed ? seed : 1;
    }

    uint64_t next_u64() {
        uint64_t x = state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        state = x;
        return x * 2685821657736338717ULL;
    }

    int uniform_int(int n) {
        if (n <= 1) return 0;
        return (int)(next_u64() % (uint64_t)n);
    }

    float uniform_float() {
        return (float)((next_u64() >> 40) & 0xFFFFFFULL) / (float)0x1000000ULL;
    }
};

struct LightningChildIndexV2 {
    std::vector<int> offsets;
    std::vector<int> children;
};

static inline LightningChildIndexV2 build_lightning_child_index_v2(const RadixTrieStructure& trie) {
    LightningChildIndexV2 idx;
    idx.offsets.assign((size_t)trie.radix_count + 1, 0);
    for (int r = 1; r < trie.radix_count; r++) {
        int p = trie.parents[r];
        if (p >= 0 && p < trie.radix_count) idx.offsets[(size_t)p + 1]++;
    }
    for (int i = 1; i <= trie.radix_count; i++) idx.offsets[(size_t)i] += idx.offsets[(size_t)i - 1];
    idx.children.assign((size_t)idx.offsets[(size_t)trie.radix_count], 0);
    std::vector<int> fill = idx.offsets;
    for (int r = 1; r < trie.radix_count; r++) {
        int p = trie.parents[r];
        if (p >= 0 && p < trie.radix_count) {
            idx.children[(size_t)fill[(size_t)p]++] = r;
        }
    }
    for (int r = 0; r < trie.radix_count; r++) {
        int b = idx.offsets[(size_t)r];
        int e = idx.offsets[(size_t)r + 1];
        std::stable_sort(idx.children.begin() + b, idx.children.begin() + e, [&](int a, int bnode) {
            int da = trie.edge_first_char_depths[a] + trie.edge_lens[a] - 1;
            int db = trie.edge_first_char_depths[bnode] + trie.edge_lens[bnode] - 1;
            if (da != db) return da < db;
            return a < bnode;
        });
    }
    return idx;
}

static inline int lightning_child_count_v2(const LightningChildIndexV2& idx, int r) {
    return idx.offsets[(size_t)r + 1] - idx.offsets[(size_t)r];
}

static inline int lightning_child_at_v2(const LightningChildIndexV2& idx, int r, int child_idx) {
    return idx.children[(size_t)idx.offsets[(size_t)r] + (size_t)child_idx];
}

static inline void lightning_add_node_v2(const RadixTrieStructure& trie,
                                         int r,
                                         std::vector<unsigned char>& node_kind,
                                         std::vector<int>& nodes,
                                         long long& query_count,
                                         long long& compact_char_count,
                                         int& max_endpoint_depth,
                                         bool context_only = false) {
    if (r <= 0 || r >= trie.radix_count) return;
    unsigned char new_kind = context_only ? 2 : 1;
    unsigned char& old_kind = node_kind[(size_t)r];
    if (old_kind) {
        if (!context_only && old_kind == 2) old_kind = 1;
        return;
    }
    old_kind = new_kind;
    nodes.push_back(r);
    query_count += trie.edge_lens[r];
    if (trie.edge_mass[r] > 1) compact_char_count += trie.edge_lens[r];
    int endpoint_depth = trie.edge_first_char_depths[r] + trie.edge_lens[r] - 1;
    if (endpoint_depth > max_endpoint_depth) max_endpoint_depth = endpoint_depth;
}

static inline void lightning_add_ancestor_closure_v2(const RadixTrieStructure& trie,
                                                     int r,
                                                     std::vector<unsigned char>& node_kind,
                                                     std::vector<int>& nodes,
                                                     long long& query_count,
                                                     long long& compact_char_count,
                                                     int& max_endpoint_depth,
                                                     bool context_only = false,
                                                     bool include_self = true) {
    std::vector<int> path;
    int cur = r;
    while (cur > 0 && cur < trie.radix_count) {
        path.push_back(cur);
        cur = trie.parents[cur];
    }
    int last = include_self ? 0 : 1;
    for (int i = (int)path.size() - 1; i >= last; i--) {
        lightning_add_node_v2(trie, path[(size_t)i], node_kind, nodes,
                              query_count, compact_char_count, max_endpoint_depth,
                              context_only);
    }
}

static inline int lightning_sample_anchor_v2(const RadixTrieStructure& trie,
                                             const LightningChildIndexV2& child_index,
                                             LightningRngV2& rng,
                                             float stop_p) {
    int root_children = lightning_child_count_v2(child_index, 0);
    if (root_children <= 0) return trie.radix_count > 1 ? 1 : 0;
    int cur = lightning_child_at_v2(child_index, 0, rng.uniform_int(root_children));
    while (cur > 0 && cur < trie.radix_count) {
        int child_count = lightning_child_count_v2(child_index, cur);
        if (child_count <= 0) break;
        if (rng.uniform_float() < stop_p) break;
        cur = lightning_child_at_v2(child_index, cur, rng.uniform_int(child_count));
    }
    return cur;
}

static inline int lightning_sample_descendant_path_v2(const RadixTrieStructure& trie,
                                                      const LightningChildIndexV2& child_index,
                                                      int anchor,
                                                      LightningRngV2& rng,
                                                      std::vector<unsigned char>& node_kind,
                                                      std::vector<int>& nodes,
                                                      long long& query_count,
                                                      long long& compact_char_count,
                                                      int& max_endpoint_depth) {
    lightning_add_ancestor_closure_v2(trie, anchor, node_kind, nodes,
                                      query_count, compact_char_count, max_endpoint_depth,
                                      /*context_only=*/true, /*include_self=*/false);
    lightning_add_node_v2(trie, anchor, node_kind, nodes,
                          query_count, compact_char_count, max_endpoint_depth,
                          /*context_only=*/false);
    int cur = anchor;
    while (cur > 0 && cur < trie.radix_count) {
        int endpoint_depth = trie.edge_first_char_depths[cur] + trie.edge_lens[cur] - 1;
        if (endpoint_depth >= trie.depth_file_count - 1) break;
        int child_count = lightning_child_count_v2(child_index, cur);
        if (child_count <= 0) break;
        cur = lightning_child_at_v2(child_index, cur, rng.uniform_int(child_count));
        lightning_add_node_v2(trie, cur, node_kind, nodes,
                              query_count, compact_char_count, max_endpoint_depth,
                              /*context_only=*/false);
    }
    return cur;
}

static inline void lightning_add_descendant_subtree_v2(const RadixTrieStructure& trie,
                                                       const LightningChildIndexV2& child_index,
                                                       int anchor,
                                                       std::vector<unsigned char>& node_kind,
                                                       std::vector<int>& nodes,
                                                       long long& query_count,
                                                       long long& compact_char_count,
                                                       int& max_endpoint_depth) {
    std::vector<int> stack;
    stack.push_back(anchor);
    while (!stack.empty()) {
        int cur = stack.back();
        stack.pop_back();
        lightning_add_node_v2(trie, cur, node_kind, nodes,
                              query_count, compact_char_count, max_endpoint_depth,
                              /*context_only=*/false);

        int child_count = lightning_child_count_v2(child_index, cur);
        for (int i = child_count - 1; i >= 0; i--) {
            stack.push_back(lightning_child_at_v2(child_index, cur, i));
        }
    }
}

static inline void append_training_node_v2(const RadixTrieStructure& trie,
                                           int r,
                                           std::vector<int>& nodes,
                                           std::vector<unsigned char>& context_flags,
                                           long long& query_count,
                                           long long& compact_char_count,
                                           int& max_endpoint_depth,
                                           bool context_only = false) {
    if (r <= 0 || r >= trie.radix_count) return;
    nodes.push_back(r);
    context_flags.push_back(context_only ? 1 : 0);
    query_count += trie.edge_lens[r];
    if (trie.edge_mass[r] > 1) compact_char_count += trie.edge_lens[r];
    int endpoint_depth = trie.edge_first_char_depths[r] + trie.edge_lens[r] - 1;
    if (endpoint_depth > max_endpoint_depth) max_endpoint_depth = endpoint_depth;
}

static inline TrainingUnit build_descendant_partition_unit_v2(const RadixTrieStructure& trie,
                                                              const LightningChildIndexV2& child_index,
                                                              int anchor,
                                                              int unit_index) {
    TrainingUnit unit;
    unit.kind = TrainingUnitKind::PartitionGroup;
    unit.unit_index = unit_index;
    unit.root_child_id = find_root_child(trie, anchor);

    std::vector<int> nodes;
    std::vector<unsigned char> context_flags;
    std::vector<int> path;
    int cur = anchor;
    while (cur > 0 && cur < trie.radix_count) {
        path.push_back(cur);
        cur = trie.parents[cur];
    }

    nodes.reserve(path.size() + 16);
    context_flags.reserve(path.size() + 16);
    for (int i = (int)path.size() - 1; i >= 1; i--) {
        append_training_node_v2(trie, path[(size_t)i], nodes, context_flags,
                                unit.query_count, unit.compact_char_count,
                                unit.max_endpoint_depth,
                                /*context_only=*/true);
    }

    std::vector<int> stack;
    stack.push_back(anchor);
    while (!stack.empty()) {
        cur = stack.back();
        stack.pop_back();
        append_training_node_v2(trie, cur, nodes, context_flags,
                                unit.query_count, unit.compact_char_count,
                                unit.max_endpoint_depth,
                                /*context_only=*/false);

        int child_count = lightning_child_count_v2(child_index, cur);
        for (int i = child_count - 1; i >= 0; i--) {
            stack.push_back(lightning_child_at_v2(child_index, cur, i));
        }
    }

    unit.node_count = (int)nodes.size();
    unit.radix_ids = (int*)std::malloc((size_t)(unit.node_count > 0 ? unit.node_count : 1) * sizeof(int));
    unit.context_only = (unsigned char*)std::malloc((size_t)(unit.node_count > 0 ? unit.node_count : 1) * sizeof(unsigned char));
    if (unit.node_count > 0) {
        std::memcpy(unit.radix_ids, nodes.data(), (size_t)unit.node_count * sizeof(int));
        std::memcpy(unit.context_only, context_flags.data(), (size_t)unit.node_count * sizeof(unsigned char));
        sort_training_unit_radix_ids_by_endpoint_depth(trie, unit);
    }
    return unit;
}

static inline TrainingPlan build_pdn_descendant_training_plan_v2(const RadixTrieStructure& trie,
                                                                 int partition_depth) {
    TrainingPlan plan;
    if (partition_depth <= 1) return build_pd1_training_plan(trie);

    LightningChildIndexV2 child_index = build_lightning_child_index_v2(trie);
    std::vector<int> anchors;
    anchors.reserve((size_t)trie.radix_count / 4);
    for (int r = 1; r < trie.radix_count; r++) {
        int first_depth = trie.edge_first_char_depths[r];
        int endpoint_depth = first_depth + trie.edge_lens[r] - 1;
        if (first_depth <= partition_depth && endpoint_depth >= partition_depth) {
            anchors.push_back(r);
        }
    }

    plan.unit_count = (int)anchors.size();
    plan.units = (TrainingUnit*)std::calloc((size_t)(plan.unit_count > 0 ? plan.unit_count : 1),
                                            sizeof(TrainingUnit));
    for (int i = 0; i < plan.unit_count; i++) {
        plan.units[i] = build_descendant_partition_unit_v2(trie, child_index, anchors[(size_t)i], i);
    }
    return plan;
}

static inline TrainingPlan build_training_plan_for_partition_depth(const RadixTrieStructure& trie,
                                                                   int partition_depth) {
    if (partition_depth == 0) return build_pd0_training_plan(trie);
    if (partition_depth == 1) return build_pd1_training_plan(trie);
    return build_pdn_descendant_training_plan_v2(trie, partition_depth);
}

static inline TrainingUnit build_lightning_sample_unit_v2(const RadixTrieStructure& trie,
                                                          const LightningChildIndexV2& child_index,
                                                          const TrainerConfig& cfg,
                                                          int unit_index) {
    TrainingUnit unit;
    unit.kind = TrainingUnitKind::LightningSample;
    unit.unit_index = unit_index;
    unit.root_child_id = 0;

    int query_budget = cfg.lightning_query_budget > 0 ? cfg.lightning_query_budget : cfg.chunk_queries;
    if (query_budget <= 0) query_budget = 50000;
    int sample_fanout = cfg.lightning_sample_fanout > 0 ? cfg.lightning_sample_fanout : 32;
    int anchors_per_step = cfg.lightning_anchors_per_step > 0 ? cfg.lightning_anchors_per_step : 1;

    LightningRngV2 rng(((uint64_t)cfg.lightning_seed << 32) ^ (uint64_t)(unit_index + 1) * 0x9E3779B97F4A7C15ULL);
    std::vector<unsigned char> node_kind((size_t)trie.radix_count, 0);
    std::vector<int> nodes;
    nodes.reserve((size_t)(query_budget > 0 ? query_budget : 1));

    if (cfg.lightning_anchor_mode == LightningAnchorModeV2::RandomDescendants) {
        int anchor = trie.radix_count > 1 ? 1 + rng.uniform_int(trie.radix_count - 1) : 0;
        unit.root_child_id = find_root_child(trie, anchor);
        lightning_add_ancestor_closure_v2(trie, anchor, node_kind, nodes,
                                          unit.query_count, unit.compact_char_count,
                                          unit.max_endpoint_depth,
                                          /*context_only=*/true, /*include_self=*/false);
        lightning_add_descendant_subtree_v2(trie, child_index, anchor, node_kind, nodes,
                                            unit.query_count, unit.compact_char_count,
                                            unit.max_endpoint_depth);
    } else {
        int guard = 0;
        while ((unit.query_count < query_budget || nodes.empty()) && guard < 10000) {
            guard++;
            for (int a = 0; a < anchors_per_step && (unit.query_count < query_budget || nodes.empty()); a++) {
                int anchor = lightning_sample_anchor_v2(trie, child_index, rng, cfg.lightning_stop_p);
                if (unit.root_child_id == 0) unit.root_child_id = find_root_child(trie, anchor);
                for (int p = 0; p < sample_fanout && (unit.query_count < query_budget || nodes.empty()); p++) {
                    lightning_sample_descendant_path_v2(trie, child_index, anchor, rng, node_kind, nodes,
                                                        unit.query_count, unit.compact_char_count,
                                                        unit.max_endpoint_depth);
                }
            }
        }

        if (nodes.empty() && trie.radix_count > 1) {
            int fallback = 1 + rng.uniform_int(trie.radix_count - 1);
            unit.root_child_id = find_root_child(trie, fallback);
            lightning_add_ancestor_closure_v2(trie, fallback, node_kind, nodes,
                                              unit.query_count, unit.compact_char_count,
                                              unit.max_endpoint_depth);
        }
    }

    unit.node_count = (int)nodes.size();
    unit.radix_ids = (int*)std::malloc((size_t)(unit.node_count > 0 ? unit.node_count : 1) * sizeof(int));
    unit.context_only = (unsigned char*)std::malloc((size_t)(unit.node_count > 0 ? unit.node_count : 1) * sizeof(unsigned char));
    if (unit.node_count > 0) {
        std::memcpy(unit.radix_ids, nodes.data(), (size_t)unit.node_count * sizeof(int));
        for (int i = 0; i < unit.node_count; i++) {
            int r = nodes[(size_t)i];
            unit.context_only[i] = node_kind[(size_t)r] == 2 ? 1 : 0;
        }
        sort_training_unit_radix_ids_by_endpoint_depth(trie, unit);
    }
    return unit;
}

static inline TrainingPlan build_lightning_training_plan_v2(const RadixTrieStructure& trie,
                                                            const TrainerConfig& cfg) {
    TrainingPlan plan;
    int updates = cfg.lightning_updates > 0 ? cfg.lightning_updates : (cfg.epochs > 0 ? cfg.epochs : 1);
    if (updates < 1) updates = 1;
    plan.unit_count = updates;
    plan.units = (TrainingUnit*)std::calloc((size_t)updates, sizeof(TrainingUnit));

    LightningChildIndexV2 child_index = build_lightning_child_index_v2(trie);
    for (int i = 0; i < updates; i++) {
        plan.units[i] = build_lightning_sample_unit_v2(trie, child_index, cfg, i);
    }
    return plan;
}

}  // namespace agpt_v2

#endif
