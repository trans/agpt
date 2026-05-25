#ifndef AGPT_V2_GROWTH_TRIE_V2_CUH
#define AGPT_V2_GROWTH_TRIE_V2_CUH

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "io.cuh"

namespace agpt_v2 {

struct GrowthNodeV2 {
    int parent = -1;
    int token = -1;
    int depth = 0;
    std::map<int, int> children;
    std::map<int, int> counts;
};

struct GrowthTrieStateV2 {
    std::vector<int> tokens;
    std::vector<GrowthNodeV2> nodes;
    int max_depth = 0;
    int ingested_starts = 0;
};

struct GrowthIncrementalRadixNodeV2 {
    std::map<int, int> children;
    std::map<int, int> count_index;
};

struct GrowthIncrementalRadixStateV2 {
    std::vector<int> tokens;
    int max_depth = 0;
    int ingested_starts = 0;
    std::vector<GrowthIncrementalRadixNodeV2> nodes;
    std::vector<int> parents;
    std::vector<int> edge_starts;
    std::vector<int> edge_lens;
    std::vector<int> edge_first_char_depths;
    std::vector<int> edge_mass;
    std::vector<int> edge_tokens_flat;
    std::vector<int> real_pos_of_char;
    std::vector<int> compact_slot;
    long long compact_slot_capacity = 0;
    std::vector<int> ancestor_char_offsets;
    std::vector<int> ancestor_char_ids;
    std::vector<int> counts_offset;
    std::vector<int> counts_len;
    std::vector<int> counts_tok;
    std::vector<int> counts_val;
};

static inline int utf8_next_codepoint_v2(const std::string& s, size_t& i) {
    unsigned char c = (unsigned char)s[i++];
    if (c < 0x80) return (int)c;
    if ((c >> 5) == 0x6 && i < s.size()) {
        int cp = ((int)(c & 0x1f) << 6) | ((int)((unsigned char)s[i++] & 0x3f));
        return cp;
    }
    if ((c >> 4) == 0xe && i + 1 < s.size()) {
        int cp = ((int)(c & 0x0f) << 12) |
                 ((int)((unsigned char)s[i++] & 0x3f) << 6) |
                 ((int)((unsigned char)s[i++] & 0x3f));
        return cp;
    }
    if ((c >> 3) == 0x1e && i + 2 < s.size()) {
        int cp = ((int)(c & 0x07) << 18) |
                 ((int)((unsigned char)s[i++] & 0x3f) << 12) |
                 ((int)((unsigned char)s[i++] & 0x3f) << 6) |
                 ((int)((unsigned char)s[i++] & 0x3f));
        return cp;
    }
    return 0xfffd;
}

static inline std::string read_text_file_v2(const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) {
        std::fprintf(stderr, "agpt_train_v2: cannot open corpus: %s\n", path);
        std::exit(1);
    }
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::string out;
    if (n > 0) {
        out.resize((size_t)n);
        if (std::fread(&out[0], 1, (size_t)n, f) != (size_t)n) {
            std::fprintf(stderr, "agpt_train_v2: failed reading corpus: %s\n", path);
            std::exit(1);
        }
    }
    std::fclose(f);
    return out;
}

static inline std::vector<int> tokenize_corpus_sorted_unique_utf8_v2(const char* path, int* vocab_size_out) {
    std::string text = read_text_file_v2(path);
    std::vector<int> cps;
    cps.reserve(text.size());
    for (size_t i = 0; i < text.size();) {
        cps.push_back(utf8_next_codepoint_v2(text, i));
    }
    std::vector<int> vocab = cps;
    std::sort(vocab.begin(), vocab.end());
    vocab.erase(std::unique(vocab.begin(), vocab.end()), vocab.end());
    std::map<int, int> id_of;
    for (int i = 0; i < (int)vocab.size(); i++) id_of[vocab[i]] = i;
    std::vector<int> tokens;
    tokens.reserve(cps.size());
    for (int cp : cps) tokens.push_back(id_of[cp]);
    if (vocab_size_out) *vocab_size_out = (int)vocab.size();
    return tokens;
}

static inline GrowthTrieStateV2 make_growth_trie_state_v2(std::vector<int>&& tokens, int max_depth) {
    GrowthTrieStateV2 st;
    st.tokens = std::move(tokens);
    st.max_depth = max_depth;
    st.nodes.push_back(GrowthNodeV2{});  // root
    st.nodes[0].parent = -1;
    st.nodes[0].token = -1;
    st.nodes[0].depth = 0;
    return st;
}

static inline GrowthIncrementalRadixStateV2 make_growth_incremental_radix_state_v2(std::vector<int>&& tokens,
                                                                                  int max_depth) {
    GrowthIncrementalRadixStateV2 st;
    st.tokens = std::move(tokens);
    st.max_depth = max_depth;
    st.nodes.push_back(GrowthIncrementalRadixNodeV2{});
    st.parents.push_back(0);
    st.edge_starts.push_back(0);
    st.edge_lens.push_back(0);
    st.edge_first_char_depths.push_back(0);
    st.edge_mass.push_back(0);
    st.ancestor_char_offsets.push_back(0);
    st.ancestor_char_offsets.push_back(0);
    st.counts_offset.push_back(0);
    st.counts_len.push_back(0);
    return st;
}

static inline int growth_get_or_create_child_v2(GrowthTrieStateV2& st, int parent, int token) {
    auto it = st.nodes[parent].children.find(token);
    if (it != st.nodes[parent].children.end()) return it->second;
    int id = (int)st.nodes.size();
    GrowthNodeV2 node;
    node.parent = parent;
    node.token = token;
    node.depth = st.nodes[parent].depth + 1;
    st.nodes.push_back(std::move(node));
    st.nodes[parent].children[token] = id;
    return id;
}

static inline void growth_ingest_start_v2(GrowthTrieStateV2& st, int start) {
    int n = (int)st.tokens.size();
    if (start < 0 || start >= n - 1) return;
    int parent = 0;
    int max_d = st.max_depth;
    if (start + max_d >= n) max_d = n - start - 1;
    for (int d = 1; d <= max_d; d++) {
        int tok = st.tokens[start + d - 1];
        int node = growth_get_or_create_child_v2(st, parent, tok);
        int target = st.tokens[start + d];
        st.nodes[node].counts[target] += 1;
        parent = node;
    }
}

static inline void growth_ingest_until_v2(GrowthTrieStateV2& st, int frontier_starts) {
    int max_starts = (int)st.tokens.size() - 1;
    if (frontier_starts > max_starts) frontier_starts = max_starts;
    if (frontier_starts < 0) frontier_starts = 0;
    while (st.ingested_starts < frontier_starts) {
        growth_ingest_start_v2(st, st.ingested_starts);
        st.ingested_starts++;
    }
}

static inline void growth_incremental_append_ancestors_for_new_node_v2(GrowthIncrementalRadixStateV2& st,
                                                                       int parent) {
    int parent_anc_start = st.ancestor_char_offsets[parent];
    int parent_anc_end = st.ancestor_char_offsets[parent + 1];
    for (int i = parent_anc_start; i < parent_anc_end; i++) {
        st.ancestor_char_ids.push_back(st.ancestor_char_ids[i]);
    }
    int parent_edge_start = st.edge_starts[parent];
    int parent_edge_len = st.edge_lens[parent];
    for (int i = 0; i < parent_edge_len; i++) {
        st.ancestor_char_ids.push_back(parent_edge_start + i);
    }
    st.ancestor_char_offsets.push_back((int)st.ancestor_char_ids.size());
}

static inline void growth_incremental_assign_compact_slots_v2(GrowthIncrementalRadixStateV2& st,
                                                              int radix_id) {
    int start = st.edge_starts[radix_id];
    int len = st.edge_lens[radix_id];
    for (int i = 0; i < len; i++) {
        int pos = start + i;
        if (pos >= 0 && pos < (int)st.compact_slot.size() && st.compact_slot[pos] < 0) {
            st.compact_slot[pos] = (int)st.compact_slot_capacity++;
        }
    }
}

static inline void growth_incremental_increment_edge_mass_v2(GrowthIncrementalRadixStateV2& st,
                                                             int radix_id) {
    int old_mass = st.edge_mass[radix_id];
    st.edge_mass[radix_id] = old_mass + 1;
    if (old_mass <= 1 && st.edge_mass[radix_id] > 1) {
        growth_incremental_assign_compact_slots_v2(st, radix_id);
    }
}

static inline void growth_incremental_rewrite_count_range_at_end_v2(GrowthIncrementalRadixStateV2& st,
                                                                    int radix_id) {
    int old_start = st.counts_offset[radix_id];
    int old_len = st.counts_len[radix_id];
    int new_start = (int)st.counts_tok.size();
    std::map<int, int> new_index;
    for (int i = 0; i < old_len; i++) {
        int src = old_start + i;
        int dst = (int)st.counts_tok.size();
        st.counts_tok.push_back(st.counts_tok[src]);
        st.counts_val.push_back(st.counts_val[src]);
        new_index[st.counts_tok[dst]] = dst;
    }
    st.counts_offset[radix_id] = new_start;
    st.nodes[radix_id].count_index = std::move(new_index);
}

static inline void growth_incremental_increment_count_v2(GrowthIncrementalRadixStateV2& st,
                                                         int radix_id,
                                                         int token,
                                                         int delta) {
    auto it = st.nodes[radix_id].count_index.find(token);
    if (it != st.nodes[radix_id].count_index.end()) {
        st.counts_val[it->second] += delta;
        return;
    }

    int start = st.counts_offset[radix_id];
    int len = st.counts_len[radix_id];
    if (len > 0 && start + len != (int)st.counts_tok.size()) {
        growth_incremental_rewrite_count_range_at_end_v2(st, radix_id);
        start = st.counts_offset[radix_id];
        len = st.counts_len[radix_id];
    } else if (len == 0) {
        st.counts_offset[radix_id] = (int)st.counts_tok.size();
    }

    int idx = (int)st.counts_tok.size();
    st.counts_tok.push_back(token);
    st.counts_val.push_back(delta);
    st.nodes[radix_id].count_index[token] = idx;
    st.counts_len[radix_id] = len + 1;
}

static inline void growth_incremental_set_single_count_v2(GrowthIncrementalRadixStateV2& st,
                                                          int radix_id,
                                                          int token,
                                                          int value) {
    int idx = (int)st.counts_tok.size();
    st.counts_offset[radix_id] = idx;
    st.counts_len[radix_id] = 1;
    st.counts_tok.push_back(token);
    st.counts_val.push_back(value);
    st.nodes[radix_id].count_index.clear();
    st.nodes[radix_id].count_index[token] = idx;
}

static inline int growth_incremental_create_edge_v2(GrowthIncrementalRadixStateV2& st,
                                                    int parent,
                                                    int first_char_depth,
                                                    int token_start,
                                                    int edge_len,
                                                    int endpoint_target) {
    int id = (int)st.nodes.size();
    int edge_start = (int)st.edge_tokens_flat.size();
    st.nodes.push_back(GrowthIncrementalRadixNodeV2{});
    st.parents.push_back(parent);
    st.edge_starts.push_back(edge_start);
    st.edge_lens.push_back(edge_len);
    st.edge_first_char_depths.push_back(first_char_depth);
    st.edge_mass.push_back(1);
    st.counts_offset.push_back((int)st.counts_tok.size());
    st.counts_len.push_back(0);

    for (int i = 0; i < edge_len; i++) {
        st.edge_tokens_flat.push_back(st.tokens[token_start + i]);
        int pos = first_char_depth + i - 1;
        st.real_pos_of_char.push_back(pos < 0 ? 0 : pos);
        st.compact_slot.push_back(-1);
    }

    growth_incremental_append_ancestors_for_new_node_v2(st, parent);
    st.nodes[parent].children[st.tokens[token_start]] = id;
    growth_incremental_set_single_count_v2(st, id, endpoint_target, 1);
    return id;
}

static inline int growth_incremental_split_edge_v2(GrowthIncrementalRadixStateV2& st,
                                                   int radix_id,
                                                   int split_len) {
    int old_start = st.edge_starts[radix_id];
    int old_len = st.edge_lens[radix_id];
    int old_mass = st.edge_mass[radix_id];
    int suffix_first_token = st.edge_tokens_flat[old_start + split_len];
    int suffix_id = (int)st.nodes.size();

    std::map<int, int> suffix_children = std::move(st.nodes[radix_id].children);
    std::map<int, int> suffix_count_index = std::move(st.nodes[radix_id].count_index);
    int suffix_counts_offset = st.counts_offset[radix_id];
    int suffix_counts_len = st.counts_len[radix_id];

    st.nodes.push_back(GrowthIncrementalRadixNodeV2{});
    st.nodes[suffix_id].children = std::move(suffix_children);
    st.nodes[suffix_id].count_index = std::move(suffix_count_index);
    for (const auto& kv : st.nodes[suffix_id].children) {
        st.parents[kv.second] = suffix_id;
    }

    st.parents.push_back(radix_id);
    st.edge_starts.push_back(old_start + split_len);
    st.edge_lens.push_back(old_len - split_len);
    st.edge_first_char_depths.push_back(st.edge_first_char_depths[radix_id] + split_len);
    st.edge_mass.push_back(old_mass);
    st.counts_offset.push_back(suffix_counts_offset);
    st.counts_len.push_back(suffix_counts_len);

    st.edge_lens[radix_id] = split_len;
    st.nodes[radix_id].children.clear();
    st.nodes[radix_id].children[suffix_first_token] = suffix_id;
    growth_incremental_set_single_count_v2(st, radix_id, suffix_first_token, old_mass);
    growth_incremental_append_ancestors_for_new_node_v2(st, radix_id);
    return suffix_id;
}

static inline void growth_incremental_ingest_start_v2(GrowthIncrementalRadixStateV2& st,
                                                      int start) {
    int n = (int)st.tokens.size();
    if (start < 0 || start >= n - 1) return;
    int max_d = st.max_depth;
    if (start + max_d >= n) max_d = n - start - 1;
    if (max_d <= 0) return;

    int parent = 0;
    int depth = 1;
    while (depth <= max_d) {
        int first_tok = st.tokens[start + depth - 1];
        auto child_it = st.nodes[parent].children.find(first_tok);
        if (child_it == st.nodes[parent].children.end()) {
            growth_incremental_create_edge_v2(st, parent, depth, start + depth - 1,
                                              max_d - depth + 1, st.tokens[start + max_d]);
            return;
        }

        int child = child_it->second;
        int edge_start = st.edge_starts[child];
        int edge_len = st.edge_lens[child];
        int max_match = max_d - depth + 1;
        int lcp = 0;
        while (lcp < edge_len && lcp < max_match &&
               st.edge_tokens_flat[edge_start + lcp] == st.tokens[start + depth - 1 + lcp]) {
            lcp++;
        }

        if (lcp < edge_len && lcp < max_match) {
            growth_incremental_split_edge_v2(st, child, lcp);
            growth_incremental_increment_edge_mass_v2(st, child);
            int diverge_depth = depth + lcp;
            int diverge_tok = st.tokens[start + diverge_depth - 1];
            growth_incremental_increment_count_v2(st, child, diverge_tok, 1);
            growth_incremental_create_edge_v2(st, child, diverge_depth, start + diverge_depth - 1,
                                              max_d - diverge_depth + 1, st.tokens[start + max_d]);
            return;
        }

        growth_incremental_increment_edge_mass_v2(st, child);
        if (lcp < edge_len) {
            return;
        }

        int endpoint_depth = depth + edge_len - 1;
        growth_incremental_increment_count_v2(st, child, st.tokens[start + endpoint_depth], 1);
        depth += edge_len;
        parent = child;
    }
}

static inline void growth_incremental_ingest_until_v2(GrowthIncrementalRadixStateV2& st,
                                                      int frontier_starts) {
    int max_starts = (int)st.tokens.size() - 1;
    if (frontier_starts > max_starts) frontier_starts = max_starts;
    if (frontier_starts < 0) frontier_starts = 0;
    while (st.ingested_starts < frontier_starts) {
        growth_incremental_ingest_start_v2(st, st.ingested_starts);
        st.ingested_starts++;
    }
}

static inline RadixTrieStructure growth_incremental_radix_view_v2(GrowthIncrementalRadixStateV2& st) {
    RadixTrieStructure trie;
    trie.radix_count = (int)st.nodes.size();
    trie.depth_file_count = st.max_depth + 1;
    trie.total_edge_chars = (long long)st.edge_tokens_flat.size();
    trie.total_counts = (int)st.counts_tok.size();
    trie.parents = st.parents.data();
    trie.edge_starts = st.edge_starts.data();
    trie.edge_lens = st.edge_lens.data();
    trie.edge_first_char_depths = st.edge_first_char_depths.data();
    trie.edge_mass = st.edge_mass.data();
    trie.edge_tokens_flat = st.edge_tokens_flat.empty() ? nullptr : st.edge_tokens_flat.data();
    trie.ancestor_char_offsets = st.ancestor_char_offsets.data();
    trie.ancestor_char_ids = st.ancestor_char_ids.empty() ? nullptr : st.ancestor_char_ids.data();
    trie.total_ancestor_chars = (long long)st.ancestor_char_ids.size();
    trie.real_pos_of_char = st.real_pos_of_char.empty() ? nullptr : st.real_pos_of_char.data();
    trie.compact_slot = st.compact_slot.empty() ? nullptr : st.compact_slot.data();
    trie.compact_slot_capacity = st.compact_slot_capacity;
    trie.counts_offset = st.counts_offset.data();
    trie.counts_len = st.counts_len.data();
    trie.counts_tok = st.counts_tok.empty() ? nullptr : st.counts_tok.data();
    trie.counts_val = st.counts_val.empty() ? nullptr : st.counts_val.data();
    return trie;
}

static inline int growth_count_sum_v2(const std::map<int, int>& counts) {
    int sum = 0;
    for (const auto& kv : counts) sum += kv.second;
    return sum;
}

struct GrowthRadixRecordV2 {
    int parent = 0;
    int first_char_depth = 1;
    std::vector<int> edge;
    int edge_mass = 0;
    std::map<int, int> counts;
};

static inline void finalize_radix_aux_v2(RadixTrieStructure& trie) {
    trie.ancestor_char_offsets = (int*)std::malloc((trie.radix_count + 1) * sizeof(int));
    long long* anc_lens = (long long*)std::calloc(trie.radix_count > 0 ? trie.radix_count : 1, sizeof(long long));
    long long total_anc_chars = 0;
    for (int r = 1; r < trie.radix_count; r++) {
        int p = trie.parents[r];
        anc_lens[r] = (p >= 0 && p < trie.radix_count) ? anc_lens[p] + trie.edge_lens[p] : 0;
        total_anc_chars += anc_lens[r];
    }
    trie.total_ancestor_chars = total_anc_chars;
    trie.ancestor_char_ids = (int*)std::malloc((total_anc_chars > 0 ? total_anc_chars : 1) * sizeof(int));
    trie.ancestor_char_offsets[0] = 0;
    for (int r = 0; r < trie.radix_count; r++) {
        trie.ancestor_char_offsets[r + 1] = trie.ancestor_char_offsets[r] + (int)anc_lens[r];
    }
    for (int r = 1; r < trie.radix_count; r++) {
        int p = trie.parents[r];
        if (p < 0 || p >= trie.radix_count) continue;
        int out = trie.ancestor_char_offsets[r];
        int parent_anc_off = trie.ancestor_char_offsets[p];
        int parent_anc_len = (int)anc_lens[p];
        std::memcpy(&trie.ancestor_char_ids[out],
                    &trie.ancestor_char_ids[parent_anc_off],
                    (size_t)parent_anc_len * sizeof(int));
        int parent_edge_start = trie.edge_starts[p];
        int parent_edge_len = trie.edge_lens[p];
        for (int e = 0; e < parent_edge_len; e++) {
            trie.ancestor_char_ids[out + parent_anc_len + e] = parent_edge_start + e;
        }
    }
    std::free(anc_lens);

    trie.compact_slot = (int*)std::malloc((trie.total_edge_chars > 0 ? trie.total_edge_chars : 1) * sizeof(int));
    long long compact_fill = 0;
    for (int r = 0; r < trie.radix_count; r++) {
        int start = trie.edge_starts[r];
        int len = trie.edge_lens[r];
        if (trie.edge_mass[r] <= 1) {
            for (int e = 0; e < len; e++) trie.compact_slot[start + e] = -1;
        } else {
            for (int e = 0; e < len; e++) trie.compact_slot[start + e] = (int)compact_fill++;
        }
    }
    trie.compact_slot_capacity = compact_fill;
}

static inline RadixTrieStructure growth_build_radix_view_v2(const GrowthTrieStateV2& st) {
    struct FrontierItem { int parent_radix; int original_id; int start_depth; };
    std::vector<GrowthRadixRecordV2> records;
    std::vector<FrontierItem> frontier;
    frontier.push_back({0, 0, 1});

    size_t frontier_pos = 0;
    while (frontier_pos < frontier.size()) {
        FrontierItem item = frontier[frontier_pos++];
        if (item.start_depth > st.max_depth) continue;
        const GrowthNodeV2& parent_node = st.nodes[item.original_id];
        for (const auto& child_kv : parent_node.children) {
            int child_id = child_kv.second;
            const GrowthNodeV2* current = &st.nodes[child_id];
            int current_id = child_id;
            int current_depth = item.start_depth;
            std::vector<int> edge;
            edge.push_back(current->token);
            int edge_mass = growth_count_sum_v2(current->counts);

            while (current->counts.size() == 1 &&
                   current->children.size() == 1 &&
                   current_depth + 1 <= st.max_depth) {
                current_id = current->children.begin()->second;
                current = &st.nodes[current_id];
                current_depth++;
                edge.push_back(current->token);
            }

            if (current->counts.empty()) continue;

            int radix_id = (int)records.size() + 1;
            GrowthRadixRecordV2 rec;
            rec.parent = item.parent_radix;
            rec.first_char_depth = item.start_depth;
            rec.edge = std::move(edge);
            rec.edge_mass = edge_mass;
            rec.counts = current->counts;
            records.push_back(std::move(rec));

            if (!current->children.empty() && current_depth + 1 <= st.max_depth) {
                frontier.push_back({radix_id, current_id, current_depth + 1});
            }
        }
    }

    RadixTrieStructure trie;
    trie.radix_count = (int)records.size() + 1;
    trie.depth_file_count = st.max_depth + 1;
    trie.total_edge_chars = 0;
    trie.total_counts = 0;
    for (const GrowthRadixRecordV2& r : records) {
        trie.total_edge_chars += (long long)r.edge.size();
        trie.total_counts += (int)r.counts.size();
    }

    trie.parents = (int*)std::calloc(trie.radix_count, sizeof(int));
    trie.edge_starts = (int*)std::calloc(trie.radix_count, sizeof(int));
    trie.edge_lens = (int*)std::calloc(trie.radix_count, sizeof(int));
    trie.edge_first_char_depths = (int*)std::calloc(trie.radix_count, sizeof(int));
    trie.edge_mass = (int*)std::calloc(trie.radix_count, sizeof(int));
    trie.edge_tokens_flat = (int*)std::calloc(trie.total_edge_chars > 0 ? trie.total_edge_chars : 1, sizeof(int));
    trie.real_pos_of_char = (int*)std::calloc(trie.total_edge_chars > 0 ? trie.total_edge_chars : 1, sizeof(int));
    trie.counts_offset = (int*)std::malloc((size_t)(trie.radix_count + 1) * sizeof(int));
    trie.counts_len = (int*)std::calloc((size_t)(trie.radix_count > 0 ? trie.radix_count : 1), sizeof(int));
    trie.counts_tok = (int*)std::malloc((size_t)(trie.total_counts > 0 ? trie.total_counts : 1) * sizeof(int));
    trie.counts_val = (int*)std::malloc((size_t)(trie.total_counts > 0 ? trie.total_counts : 1) * sizeof(int));

    trie.counts_offset[0] = 0;
    long long edge_fill = 0;
    int count_fill = 0;
    for (int i = 0; i < (int)records.size(); i++) {
        int rid = i + 1;
        const GrowthRadixRecordV2& r = records[i];
        trie.parents[rid] = r.parent;
        trie.edge_starts[rid] = (int)edge_fill;
        trie.edge_lens[rid] = (int)r.edge.size();
        trie.edge_first_char_depths[rid] = r.first_char_depth;
        trie.edge_mass[rid] = r.edge_mass;
        for (int e = 0; e < (int)r.edge.size(); e++) {
            trie.edge_tokens_flat[edge_fill + e] = r.edge[e];
            int pos = r.first_char_depth + e - 1;
            trie.real_pos_of_char[edge_fill + e] = pos < 0 ? 0 : pos;
        }
        edge_fill += (long long)r.edge.size();
        trie.counts_offset[rid] = count_fill;
        trie.counts_len[rid] = (int)r.counts.size();
        for (const auto& kv : r.counts) {
            trie.counts_tok[count_fill] = kv.first;
            trie.counts_val[count_fill] = kv.second;
            count_fill++;
        }
    }
    trie.counts_offset[trie.radix_count] = count_fill;

    finalize_radix_aux_v2(trie);
    return trie;
}

static inline std::vector<int> parse_growth_frontiers_v2(const char* text, int full_starts) {
    std::vector<int> out;
    if (!text || !text[0]) return out;
    const char* p = text;
    while (*p) {
        while (*p == ',' || *p == ' ' || *p == '\t') p++;
        if (!*p) break;
        if (std::strncmp(p, "full", 4) == 0) {
            out.push_back(full_starts);
            p += 4;
        } else {
            char* end = nullptr;
            long v = std::strtol(p, &end, 10);
            if (end == p) {
                std::fprintf(stderr, "agpt_train_v2: bad --growth-frontiers near '%s'\n", p);
                std::exit(1);
            }
            if (v < 0) v = 0;
            if (v > full_starts) v = full_starts;
            out.push_back((int)v);
            p = end;
        }
        while (*p && *p != ',') p++;
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

}  // namespace agpt_v2

#endif
