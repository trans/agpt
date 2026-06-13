#ifndef AGPT_V2_TARGET_SIDECAR_V2_CUH
#define AGPT_V2_TARGET_SIDECAR_V2_CUH

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace agpt_v2 {

struct TargetSidecarEntryV2 {
    uint16_t token = 0;
    uint32_t count = 0;
};

struct TargetSidecarTableV2 {
    uint16_t version = 0;
    uint32_t scale = 0;
    uint32_t substring_count = 0;
    uint64_t total_entries = 0;
    std::vector<int32_t> offsets;
    std::vector<TargetSidecarEntryV2> entries;
};

static inline uint16_t target_sidecar_read_u16_v2(FILE* f) {
    uint16_t v = 0;
    if (std::fread(&v, sizeof(v), 1, f) != 1) {
        std::fprintf(stderr, "agpt_train_v2: failed reading target sidecar u16\n");
        std::exit(1);
    }
    return v;
}

static inline uint32_t target_sidecar_read_u32_v2(FILE* f) {
    uint32_t v = 0;
    if (std::fread(&v, sizeof(v), 1, f) != 1) {
        std::fprintf(stderr, "agpt_train_v2: failed reading target sidecar u32\n");
        std::exit(1);
    }
    return v;
}

static inline uint64_t target_sidecar_read_u64_v2(FILE* f) {
    uint64_t v = 0;
    if (std::fread(&v, sizeof(v), 1, f) != 1) {
        std::fprintf(stderr, "agpt_train_v2: failed reading target sidecar u64\n");
        std::exit(1);
    }
    return v;
}

static inline TargetSidecarTableV2 load_target_sidecar_table_v2(const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) {
        std::fprintf(stderr, "agpt_train_v2: cannot open target sidecar: %s\n", path);
        std::exit(1);
    }
    char magic[4] = {0, 0, 0, 0};
    if (std::fread(magic, 1, 4, f) != 4 ||
        magic[0] != 'A' || magic[1] != 'G' || magic[2] != 'T' || magic[3] != 'S') {
        std::fprintf(stderr, "agpt_train_v2: bad target sidecar magic: %s\n", path);
        std::exit(1);
    }
    TargetSidecarTableV2 out;
    out.version = target_sidecar_read_u16_v2(f);
    out.scale = target_sidecar_read_u32_v2(f);
    out.substring_count = target_sidecar_read_u32_v2(f);
    out.total_entries = target_sidecar_read_u64_v2(f);
    if (out.version != 1 || out.scale == 0) {
        std::fprintf(stderr, "agpt_train_v2: unsupported target sidecar header: %s\n", path);
        std::exit(1);
    }
    out.offsets.resize((size_t)out.substring_count + 1);
    if (std::fread(out.offsets.data(), sizeof(int32_t), (size_t)out.substring_count + 1, f) !=
        (size_t)out.substring_count + 1) {
        std::fprintf(stderr, "agpt_train_v2: failed reading target sidecar offsets: %s\n", path);
        std::exit(1);
    }
    out.entries.resize((size_t)out.total_entries);
    for (uint64_t i = 0; i < out.total_entries; i++) {
        out.entries[(size_t)i].token = target_sidecar_read_u16_v2(f);
        out.entries[(size_t)i].count = target_sidecar_read_u32_v2(f);
    }
    std::fclose(f);
    return out;
}

}  // namespace agpt_v2

#endif
