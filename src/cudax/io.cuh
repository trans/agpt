#ifndef AGPT_V2_IO_CUH
#define AGPT_V2_IO_CUH

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "types.cuh"

namespace agpt_v2 {

static constexpr unsigned MGPT_MAGIC = 0x4D475054u;
static constexpr unsigned RADIX_MAGIC = 0x52445841u;

static inline int read_i32(FILE* f) {
    int v = 0;
    if (fread(&v, 4, 1, f) != 1) {
        std::fprintf(stderr, "agpt_train_v2: failed to read i32\n");
        std::exit(1);
    }
    return v;
}

static inline int read_i16(FILE* f) {
    short v = 0;
    if (fread(&v, 2, 1, f) != 1) {
        std::fprintf(stderr, "agpt_train_v2: failed to read i16\n");
        std::exit(1);
    }
    return (int)v;
}

static inline unsigned read_u32(FILE* f) {
    unsigned v = 0;
    if (fread(&v, 4, 1, f) != 1) {
        std::fprintf(stderr, "agpt_train_v2: failed to read u32\n");
        std::exit(1);
    }
    return v;
}

static inline unsigned long long read_u64(FILE* f) {
    unsigned long long v = 0;
    if (fread(&v, 8, 1, f) != 1) {
        std::fprintf(stderr, "agpt_train_v2: failed to read u64\n");
        std::exit(1);
    }
    return v;
}

struct RadixTrieStructure {
    int radix_count = 0;
    int depth_file_count = 0;
    long long total_edge_chars = 0;
    int total_counts = 0;

    int* parents = nullptr;
    int* edge_starts = nullptr;
    int* edge_lens = nullptr;
    int* edge_first_char_depths = nullptr;
    int* edge_mass = nullptr;
    int* edge_tokens_flat = nullptr;
    int* ancestor_char_offsets = nullptr;
    int* ancestor_char_ids = nullptr;
    long long total_ancestor_chars = 0;
    int* real_pos_of_char = nullptr;
    int* compact_slot = nullptr;
    int* counts_offset = nullptr;
    int* counts_tok = nullptr;
    int* counts_val = nullptr;
};

static inline void free_radix_trie_structure(RadixTrieStructure& trie) {
    std::free(trie.parents);
    std::free(trie.edge_starts);
    std::free(trie.edge_lens);
    std::free(trie.edge_first_char_depths);
    std::free(trie.edge_mass);
    std::free(trie.edge_tokens_flat);
    std::free(trie.ancestor_char_offsets);
    std::free(trie.ancestor_char_ids);
    std::free(trie.real_pos_of_char);
    std::free(trie.compact_slot);
    std::free(trie.counts_offset);
    std::free(trie.counts_tok);
    std::free(trie.counts_val);
    trie = RadixTrieStructure{};
}

static inline ModelHeader load_model_header(const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) {
        std::fprintf(stderr, "agpt_train_v2: cannot open model: %s\n", path);
        std::exit(1);
    }

    unsigned magic = read_u32(f);
    if (magic != MGPT_MAGIC) {
        std::fprintf(stderr, "agpt_train_v2: bad model magic in %s\n", path);
        std::exit(1);
    }

    ModelHeader header;
    header.shape.d_model = read_i32(f);
    header.shape.n_heads = read_i32(f);
    header.shape.n_layers = read_i32(f);
    header.shape.d_ff = read_i32(f);
    header.shape.vocab_size = read_i32(f);
    header.shape.seq_len = read_i32(f);
    header.shape.head_dim = header.shape.d_model / header.shape.n_heads;

    std::fclose(f);
    return header;
}

static inline RadixTrieStructure load_radix_structure_minimal(const char* dir) {
    RadixTrieStructure trie;
    char path[1024];

    std::snprintf(path, sizeof(path), "%s/meta.bin", dir);
    FILE* f = std::fopen(path, "rb");
    if (!f) {
        std::fprintf(stderr, "agpt_train_v2: cannot open trie meta: %s\n", path);
        std::exit(1);
    }

    unsigned magic = read_u32(f);
    if (magic != RADIX_MAGIC) {
        std::fprintf(stderr, "agpt_train_v2: bad radix magic in %s\n", path);
        std::exit(1);
    }
    int version = read_i32(f);
    if (version != 2 && version != 3) {
        std::fprintf(stderr, "agpt_train_v2: unsupported radix format v%d in %s\n", version, path);
        std::exit(1);
    }
    const bool narrow_tokens = (version >= 3);
    const int counts_entry_bytes = narrow_tokens ? 6 : 8;

    trie.radix_count = read_i32(f);
    trie.depth_file_count = read_i32(f);
    trie.total_edge_chars = (long long)read_u64(f);
    read_i32(f);     // corpus_token_count
    read_i32(f);     // vocab_size
    read_u64(f);     // corpus_hash
    int tokenizer_tag_len = read_i32(f);
    std::fseek(f, tokenizer_tag_len, SEEK_CUR);
    std::fclose(f);

    trie.parents = (int*)std::calloc(trie.radix_count, sizeof(int));
    trie.edge_starts = (int*)std::calloc(trie.radix_count, sizeof(int));
    trie.edge_lens = (int*)std::calloc(trie.radix_count, sizeof(int));
    trie.edge_first_char_depths = (int*)std::calloc(trie.radix_count, sizeof(int));
    trie.edge_mass = (int*)std::calloc(trie.radix_count, sizeof(int));
    trie.edge_tokens_flat = (int*)std::calloc(trie.total_edge_chars > 0 ? trie.total_edge_chars : 1, sizeof(int));
    trie.real_pos_of_char = (int*)std::calloc(trie.total_edge_chars > 0 ? trie.total_edge_chars : 1, sizeof(int));
    trie.compact_slot = (int*)std::malloc((trie.total_edge_chars > 0 ? trie.total_edge_chars : 1) * sizeof(int));
    int* count_lens = (int*)std::calloc(trie.radix_count > 0 ? trie.radix_count : 1, sizeof(int));

    long long edge_fill_pos = 0;
    for (int d = 0; d < trie.depth_file_count; d++) {
        std::snprintf(path, sizeof(path), "%s/radix_depth_%03d.bin", dir, d);
        f = std::fopen(path, "rb");
        if (!f) continue;

        unsigned dm = read_u32(f);
        if (dm != RADIX_MAGIC) {
            std::fprintf(stderr, "agpt_train_v2: bad radix depth magic in %s\n", path);
            std::exit(1);
        }
        int stored_depth = read_i32(f);
        if (stored_depth != d) {
            std::fprintf(stderr, "agpt_train_v2: radix depth mismatch in %s\n", path);
            std::exit(1);
        }
        int n = read_i32(f);
        for (int i = 0; i < n; i++) {
            int rid = read_i32(f);
            trie.parents[rid] = read_i32(f);
            trie.edge_first_char_depths[rid] = read_i32(f);
            trie.edge_lens[rid] = read_i32(f);
            trie.edge_starts[rid] = (int)edge_fill_pos;
            int max_real_pos = trie.depth_file_count >= 2 ? trie.depth_file_count - 2 : 0;
            for (int e = 0; e < trie.edge_lens[rid]; e++) {
                trie.edge_tokens_flat[edge_fill_pos + e] = narrow_tokens ? read_i16(f) : read_i32(f);
                trie.real_pos_of_char[edge_fill_pos + e] = trie.edge_first_char_depths[rid] + e - 1;
                if (trie.real_pos_of_char[edge_fill_pos + e] < 0) trie.real_pos_of_char[edge_fill_pos + e] = 0;
                if (trie.real_pos_of_char[edge_fill_pos + e] > max_real_pos) {
                    trie.real_pos_of_char[edge_fill_pos + e] = max_real_pos;
                }
            }
            edge_fill_pos += trie.edge_lens[rid];
            trie.edge_mass[rid] = read_i32(f);
            int ec = read_i32(f);
            count_lens[rid] = ec;
            trie.total_counts += ec;
            std::fseek(f, ec * counts_entry_bytes, SEEK_CUR);
        }
        std::fclose(f);
    }

    trie.counts_offset = (int*)std::malloc((trie.radix_count + 1) * sizeof(int));
    trie.counts_offset[0] = 0;
    for (int r = 0; r < trie.radix_count; r++) {
        trie.counts_offset[r + 1] = trie.counts_offset[r] + count_lens[r];
    }
    trie.counts_tok = (int*)std::malloc((trie.total_counts > 0 ? trie.total_counts : 1) * sizeof(int));
    trie.counts_val = (int*)std::malloc((trie.total_counts > 0 ? trie.total_counts : 1) * sizeof(int));
    int* count_fill = (int*)std::malloc((trie.radix_count > 0 ? trie.radix_count : 1) * sizeof(int));
    std::memcpy(count_fill, trie.counts_offset, trie.radix_count * sizeof(int));

    for (int d = 0; d < trie.depth_file_count; d++) {
        std::snprintf(path, sizeof(path), "%s/radix_depth_%03d.bin", dir, d);
        f = std::fopen(path, "rb");
        if (!f) continue;

        unsigned dm = read_u32(f);
        if (dm != RADIX_MAGIC) {
            std::fprintf(stderr, "agpt_train_v2: bad radix depth magic in %s\n", path);
            std::exit(1);
        }
        int stored_depth = read_i32(f);
        if (stored_depth != d) {
            std::fprintf(stderr, "agpt_train_v2: radix depth mismatch in %s\n", path);
            std::exit(1);
        }
        int n = read_i32(f);
        for (int i = 0; i < n; i++) {
            int rid = read_i32(f);
            read_i32(f);  // parent
            read_i32(f);  // first_char_depth
            int edge_len = read_i32(f);
            std::fseek(f, edge_len * (narrow_tokens ? 2 : 4), SEEK_CUR);
            read_i32(f);  // edge_mass
            int ec = read_i32(f);
            int fill = count_fill[rid];
            for (int e = 0; e < ec; e++) {
                trie.counts_tok[fill + e] = narrow_tokens ? read_i16(f) : read_i32(f);
                trie.counts_val[fill + e] = read_i32(f);
            }
            count_fill[rid] += ec;
        }
        std::fclose(f);
    }
    std::free(count_fill);
    std::free(count_lens);

    trie.ancestor_char_offsets = (int*)std::malloc((trie.radix_count + 1) * sizeof(int));
    long long* anc_lens = (long long*)std::calloc(trie.radix_count > 0 ? trie.radix_count : 1, sizeof(long long));
    anc_lens[0] = 0;
    long long total_anc_chars = 0;
    for (int r = 1; r < trie.radix_count; r++) {
        int p = trie.parents[r];
        if (p < 0 || p >= trie.radix_count) {
            anc_lens[r] = 0;
        } else {
            anc_lens[r] = anc_lens[p] + trie.edge_lens[p];
        }
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
                    parent_anc_len * sizeof(int));
        int parent_edge_start = trie.edge_starts[p];
        int parent_edge_len = trie.edge_lens[p];
        for (int e = 0; e < parent_edge_len; e++) {
            trie.ancestor_char_ids[out + parent_anc_len + e] = parent_edge_start + e;
        }
    }
    std::free(anc_lens);

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

    return trie;
}

}  // namespace agpt_v2

#endif
