#ifndef AGPT_V2_POSITION_SAMPLING_V2_CUH
#define AGPT_V2_POSITION_SAMPLING_V2_CUH

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "io.cuh"

namespace agpt_v2 {

struct PosBinV2 {
    uint16_t pos = 0;
    uint32_t count = 0;
};

struct PositionTableV2 {
    int window_size = 0;
    int substring_count = 0;
    int64_t total_bins = 0;
    std::vector<int32_t> pos_offsets;
    std::vector<PosBinV2> pos_bins;
};

struct PositionSamplingDataV2 {
    PositionTableV2 prefix_table;
    std::unordered_map<std::string, int> substring_id_by_tokens;
};

struct PositionSamplingStageV2 {
    const PositionSamplingDataV2* data = nullptr;
    std::vector<int> substring_id_by_radix;
    unsigned seed = 1;
};

static inline uint8_t pos_read_u8_v2(FILE* f) {
    uint8_t v = 0;
    if (std::fread(&v, 1, 1, f) != 1) {
        std::fprintf(stderr, "agpt_train_v2: failed reading u8 from position data\n");
        std::exit(1);
    }
    return v;
}

static inline uint16_t pos_read_u16_v2(FILE* f) {
    uint16_t v = 0;
    if (std::fread(&v, 2, 1, f) != 1) {
        std::fprintf(stderr, "agpt_train_v2: failed reading u16 from position data\n");
        std::exit(1);
    }
    return v;
}

static inline uint32_t pos_read_u32_v2(FILE* f) {
    uint32_t v = 0;
    if (std::fread(&v, 4, 1, f) != 1) {
        std::fprintf(stderr, "agpt_train_v2: failed reading u32 from position data\n");
        std::exit(1);
    }
    return v;
}

static inline uint64_t pos_read_u64_v2(FILE* f) {
    uint64_t v = 0;
    if (std::fread(&v, 8, 1, f) != 1) {
        std::fprintf(stderr, "agpt_train_v2: failed reading u64 from position data\n");
        std::exit(1);
    }
    return v;
}

static inline bool pos_check_magic_v2(FILE* f, const char* expected) {
    char magic[4] = {0, 0, 0, 0};
    if (std::fread(magic, 1, 4, f) != 4) return false;
    return std::memcmp(magic, expected, 4) == 0;
}

static inline std::string pos_token_key_v2(const std::vector<int>& tokens) {
    std::string key;
    key.reserve(tokens.size() * sizeof(uint32_t));
    for (int token : tokens) {
        uint32_t v = (uint32_t)token;
        key.push_back((char)(v & 0xffu));
        key.push_back((char)((v >> 8) & 0xffu));
        key.push_back((char)((v >> 16) & 0xffu));
        key.push_back((char)((v >> 24) & 0xffu));
    }
    return key;
}

static inline std::unordered_map<std::string, int> load_substring_catalog_map_v2(const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) {
        std::fprintf(stderr, "agpt_train_v2: cannot open substring catalog: %s\n", path);
        std::exit(1);
    }
    if (!pos_check_magic_v2(f, "ASUB")) {
        std::fprintf(stderr, "agpt_train_v2: bad substring catalog magic: %s\n", path);
        std::exit(1);
    }
    uint32_t count = pos_read_u32_v2(f);
    std::unordered_map<std::string, int> out;
    out.reserve((size_t)count * 2);
    std::vector<int> tokens;
    for (uint32_t sid = 0; sid < count; sid++) {
        uint8_t len = pos_read_u8_v2(f);
        tokens.clear();
        tokens.reserve(len);
        for (uint8_t i = 0; i < len; i++) {
            tokens.push_back((int)pos_read_u8_v2(f));
        }
        out.emplace(pos_token_key_v2(tokens), (int)sid);
    }
    std::fclose(f);
    return out;
}

static inline PositionTableV2 load_prefix_position_table_v2(const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) {
        std::fprintf(stderr, "agpt_train_v2: cannot open position table: %s\n", path);
        std::exit(1);
    }
    if (!pos_check_magic_v2(f, "APOS")) {
        std::fprintf(stderr, "agpt_train_v2: bad position table magic: %s\n", path);
        std::exit(1);
    }

    PositionTableV2 out;
    (void)pos_read_u8_v2(f);  // regime
    out.window_size = (int)pos_read_u16_v2(f);
    (void)pos_read_u8_v2(f);  // reserved
    out.substring_count = (int)pos_read_u32_v2(f);
    out.total_bins = (int64_t)pos_read_u64_v2(f);
    out.pos_offsets.resize((size_t)out.substring_count + 1);
    if (std::fread(out.pos_offsets.data(), sizeof(int32_t), (size_t)out.substring_count + 1, f) !=
        (size_t)out.substring_count + 1) {
        std::fprintf(stderr, "agpt_train_v2: failed reading position offsets: %s\n", path);
        std::exit(1);
    }
    out.pos_bins.resize((size_t)out.total_bins);
    for (int64_t i = 0; i < out.total_bins; i++) {
        out.pos_bins[(size_t)i].pos = pos_read_u16_v2(f);
        out.pos_bins[(size_t)i].count = pos_read_u32_v2(f);
    }
    std::fclose(f);
    return out;
}

static inline PositionSamplingDataV2 load_position_sampling_data_v2(const char* dir) {
    char path[1024];
    PositionSamplingDataV2 out;
    std::snprintf(path, sizeof(path), "%s/substrings.bin", dir);
    out.substring_id_by_tokens = load_substring_catalog_map_v2(path);
    std::snprintf(path, sizeof(path), "%s/prefix_position_table.bin", dir);
    out.prefix_table = load_prefix_position_table_v2(path);
    return out;
}

static inline uint32_t mix_u32_v2(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

static inline int sample_prefix_start_bin_v2(const PositionSamplingStageV2* stage,
                                             int radix_id,
                                             int epoch_index,
                                             int unit_root_child_id,
                                             int optimizer_step_index) {
    if (!stage || !stage->data) return -1;
    if (radix_id < 0 || radix_id >= (int)stage->substring_id_by_radix.size()) return -1;
    int sid = stage->substring_id_by_radix[(size_t)radix_id];
    const PositionTableV2& table = stage->data->prefix_table;
    if (sid < 0 || sid >= table.substring_count) return -1;
    int start = table.pos_offsets[(size_t)sid];
    int end = table.pos_offsets[(size_t)sid + 1];
    if (end <= start) return -1;

    uint64_t total = 0;
    for (int i = start; i < end; i++) total += (uint64_t)table.pos_bins[(size_t)i].count;
    if (total == 0) return -1;

    uint32_t h = stage->seed;
    h = mix_u32_v2(h ^ (uint32_t)radix_id);
    h = mix_u32_v2(h ^ ((uint32_t)epoch_index * 0x9e3779b9u));
    h = mix_u32_v2(h ^ ((uint32_t)unit_root_child_id * 0x85ebca6bu));
    h = mix_u32_v2(h ^ ((uint32_t)optimizer_step_index * 0xc2b2ae35u));
    uint64_t ticket = (uint64_t)h % total;
    for (int i = start; i < end; i++) {
        uint32_t count = table.pos_bins[(size_t)i].count;
        if (ticket < count) return (int)table.pos_bins[(size_t)i].pos;
        ticket -= count;
    }
    return (int)table.pos_bins[(size_t)end - 1].pos;
}

static inline int sampled_rope_pos_from_start_v2(int sampled_start, int depth_zero_based, int window_size) {
    if (sampled_start < 0 || window_size <= 0) return -1;
    int pos = (sampled_start + depth_zero_based) % window_size;
    if (pos < 0) pos += window_size;
    return pos;
}

static inline PositionSamplingStageV2 build_position_sampling_stage_v2(const PositionSamplingDataV2& data,
                                                                       const RadixTrieStructure& trie,
                                                                       unsigned seed) {
    PositionSamplingStageV2 out;
    out.data = &data;
    out.seed = seed;
    out.substring_id_by_radix.assign((size_t)trie.radix_count, -1);
    std::vector<int> tokens;
    for (int r = 1; r < trie.radix_count; r++) {
        tokens.clear();
        int anc_start = trie.ancestor_char_offsets[r];
        int anc_end = trie.ancestor_char_offsets[r + 1];
        tokens.reserve((size_t)(anc_end - anc_start + trie.edge_lens[r]));
        for (int i = anc_start; i < anc_end; i++) {
            int char_pos = trie.ancestor_char_ids[i];
            tokens.push_back(trie.edge_tokens_flat[char_pos]);
        }
        int edge_start = trie.edge_starts[r];
        int edge_len = trie.edge_lens[r];
        for (int i = 0; i < edge_len; i++) {
            tokens.push_back(trie.edge_tokens_flat[edge_start + i]);
        }
        auto it = data.substring_id_by_tokens.find(pos_token_key_v2(tokens));
        if (it != data.substring_id_by_tokens.end()) out.substring_id_by_radix[(size_t)r] = it->second;
    }
    return out;
}

static inline int count_position_sampling_matches_v2(const PositionSamplingStageV2& stage) {
    int n = 0;
    for (int sid : stage.substring_id_by_radix) {
        if (sid >= 0) n++;
    }
    return n;
}

}  // namespace agpt_v2

#endif
