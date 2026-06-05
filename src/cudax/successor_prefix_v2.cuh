#ifndef AGPT_V2_SUCCESSOR_PREFIX_CUH
#define AGPT_V2_SUCCESSOR_PREFIX_CUH

#include <cstdio>
#include <cstdlib>

#include "io.cuh"

namespace agpt_v2 {

static constexpr unsigned SUCCESSOR_PREFIX_MAGIC_V2 = 0x43555341u; // 'ASUC'

struct SuccessorPrefixTableV2 {
    bool enabled = false;
    int radix_count = 0;
    int d_max = 0;
    int mode = 0; // 1=end, 2=head
    int mass_one_only = 0;
    unsigned long long deterministic_count = 0;
    unsigned long long skipped_fanout_count = 0;
    int* successor = nullptr;
};

static inline void free_successor_prefix_table_v2(SuccessorPrefixTableV2& table) {
    std::free(table.successor);
    table = SuccessorPrefixTableV2{};
}

static inline SuccessorPrefixTableV2 load_successor_prefix_table_v2(const char* path,
                                                                    int expected_radix_count,
                                                                    int expected_d_max) {
    SuccessorPrefixTableV2 table;
    if (!path || path[0] == '\0') return table;

    FILE* f = std::fopen(path, "rb");
    if (!f) {
        std::fprintf(stderr, "agpt_train_v2: cannot open successor prefix table: %s\n", path);
        std::exit(1);
    }

    unsigned magic = read_u32(f);
    if (magic != SUCCESSOR_PREFIX_MAGIC_V2) {
        std::fprintf(stderr, "agpt_train_v2: bad successor prefix table magic in %s\n", path);
        std::exit(1);
    }
    unsigned version = read_u32(f);
    if (version != 1u) {
        std::fprintf(stderr, "agpt_train_v2: unsupported successor prefix table v%u in %s\n", version, path);
        std::exit(1);
    }
    table.radix_count = (int)read_u32(f);
    table.d_max = (int)read_u32(f);
    table.mode = (int)read_u32(f);
    table.mass_one_only = (int)read_u32(f);
    table.deterministic_count = read_u64(f);
    table.skipped_fanout_count = read_u64(f);
    if (table.radix_count != expected_radix_count) {
        std::fprintf(stderr, "agpt_train_v2: successor table radix_count mismatch: table=%d trie=%d\n",
                     table.radix_count, expected_radix_count);
        std::exit(1);
    }
    if (table.d_max != expected_d_max) {
        std::fprintf(stderr, "agpt_train_v2: successor table d_max mismatch: table=%d trie=%d\n",
                     table.d_max, expected_d_max);
        std::exit(1);
    }
    table.successor = (int*)std::malloc((size_t)table.radix_count * sizeof(int));
    if (std::fread(table.successor, sizeof(int), (size_t)table.radix_count, f) != (size_t)table.radix_count) {
        std::fprintf(stderr, "agpt_train_v2: truncated successor prefix table: %s\n", path);
        std::exit(1);
    }
    std::fclose(f);
    table.enabled = true;
    return table;
}

static inline int successor_prefix_successor_v2(const SuccessorPrefixTableV2* table, int radix_id) {
    if (!table || !table->enabled || !table->successor) return -1;
    if (radix_id < 0 || radix_id >= table->radix_count) return -1;
    return table->successor[radix_id];
}

static inline int successor_prefix_path_len_v2(const RadixTrieStructure& trie,
                                               const SuccessorPrefixTableV2* table,
                                               int radix_id) {
    int succ = successor_prefix_successor_v2(table, radix_id);
    if (succ <= 0 || succ >= trie.radix_count) return 0;
    return (trie.ancestor_char_offsets[succ + 1] - trie.ancestor_char_offsets[succ]) +
           trie.edge_lens[succ];
}

static inline int successor_prefix_rope_base_v2(const RadixTrieStructure& trie,
                                                const SuccessorPrefixTableV2* table,
                                                int radix_id) {
    if (!table || !table->enabled) return 0;
    if (table->mode == 2) {
        int base = trie.edge_first_char_depths[radix_id] - 1;
        return base < 0 ? 0 : base;
    }
    return table->d_max;
}

}  // namespace agpt_v2

#endif
