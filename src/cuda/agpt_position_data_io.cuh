// Loaders for per-substring position-distribution side-tables emitted by
// `bin/agpt_build_position_table`. Mirrors src/agpt/{substring_catalog,
// position_table,radix_to_substring}.cr on the C++ side so the trainer
// can consume them.
//
// Used by --position-data <dir> + --pos-encoder {expected|dist-rope}.
//
// Layout (all little-endian):
//   substrings.bin            — magic "ASUB" + count + per-substring length+tokens
//   prefix_radix_to_substring.bin — magic "PRTS" + radix_count + int32[radix_count]
//   suffix_radix_to_substring.bin — magic "SRTS" + radix_count + int32[radix_count]
//   prefix_position_table.bin     — magic "APOS" + regime + window_size + reserved
//                                   + substring_count + total_bins
//                                   + pos_offsets[substring_count+1]
//                                   + pos_bins[total_bins] (u16 pos + u32 count)
//   suffix_position_table.bin     — same layout as prefix
#ifndef AGPT_POSITION_DATA_IO_CUH
#define AGPT_POSITION_DATA_IO_CUH

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>

struct RadixToSubstring {
    enum class Side : uint8_t { Prefix = 0, Suffix = 1 };
    Side side = Side::Prefix;
    int radix_count = 0;
    std::vector<int32_t> ids;  // ids[radix_id] = substring_id (or -1)
};

struct PosBin {
    uint16_t pos;
    uint32_t count;
};

struct PositionTable {
    enum class Regime : uint8_t { Aligned = 0, Sliding = 1 };
    Regime regime = Regime::Sliding;
    int window_size = 0;
    int substring_count = 0;
    int64_t total_bins = 0;
    std::vector<int32_t> pos_offsets;  // [substring_count + 1]
    std::vector<PosBin> pos_bins;      // [total_bins]

    // Returns slice [pos_offsets[sid] .. pos_offsets[sid+1]) into pos_bins.
    const PosBin* bins_for(int substring_id, int& out_count) const {
        int start = pos_offsets[substring_id];
        int end   = pos_offsets[substring_id + 1];
        out_count = end - start;
        return pos_bins.data() + start;
    }

    // Returns the expected position E[p] = Σ p · count(p) / Σ count(p).
    // 0.0 if substring has no bins.
    float expected_pos(int substring_id) const {
        uint64_t sum_pc = 0;
        uint64_t sum_c  = 0;
        int n = 0;
        const PosBin* bins = bins_for(substring_id, n);
        for (int i = 0; i < n; i++) {
            sum_pc += (uint64_t)bins[i].pos * (uint64_t)bins[i].count;
            sum_c  += (uint64_t)bins[i].count;
        }
        if (sum_c == 0) return 0.0f;
        return (float)((double)sum_pc / (double)sum_c);
    }
};

// --- File readers ---

static int8_t pd_read_i8(FILE* f) {
    int8_t v; fread(&v, 1, 1, f); return v;
}
static uint8_t pd_read_u8(FILE* f) {
    uint8_t v; fread(&v, 1, 1, f); return v;
}
static uint16_t pd_read_u16(FILE* f) {
    uint16_t v; fread(&v, 2, 1, f); return v;
}
static int32_t pd_read_i32(FILE* f) {
    int32_t v; fread(&v, 4, 1, f); return v;
}
static uint32_t pd_read_u32(FILE* f) {
    uint32_t v; fread(&v, 4, 1, f); return v;
}
static uint64_t pd_read_u64(FILE* f) {
    uint64_t v; fread(&v, 8, 1, f); return v;
}
static bool pd_check_magic(FILE* f, const char* expected) {
    char buf[5] = {0,0,0,0,0};
    fread(buf, 1, 4, f);
    return memcmp(buf, expected, 4) == 0;
}

static RadixToSubstring load_radix_to_substring(const char* path) {
    RadixToSubstring r;
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", path); exit(1); }
    char magic[5] = {0,0,0,0,0};
    fread(magic, 1, 4, f);
    if (memcmp(magic, "PRTS", 4) == 0) {
        r.side = RadixToSubstring::Side::Prefix;
    } else if (memcmp(magic, "SRTS", 4) == 0) {
        r.side = RadixToSubstring::Side::Suffix;
    } else {
        fprintf(stderr, "ERROR: bad magic in %s (expected PRTS or SRTS, got %s)\n", path, magic);
        exit(1);
    }
    r.radix_count = (int)pd_read_u32(f);
    r.ids.resize(r.radix_count);
    fread(r.ids.data(), sizeof(int32_t), r.radix_count, f);
    fclose(f);
    return r;
}

static PositionTable load_position_table(const char* path) {
    PositionTable t;
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", path); exit(1); }
    if (!pd_check_magic(f, "APOS")) {
        fprintf(stderr, "ERROR: bad magic in %s (expected APOS)\n", path);
        exit(1);
    }
    t.regime      = (PositionTable::Regime)pd_read_u8(f);
    t.window_size = (int)pd_read_u16(f);
    pd_read_u8(f);  // reserved padding
    t.substring_count = (int)pd_read_u32(f);
    t.total_bins      = (int64_t)pd_read_u64(f);

    t.pos_offsets.resize(t.substring_count + 1);
    fread(t.pos_offsets.data(), sizeof(int32_t), t.substring_count + 1, f);
    t.pos_bins.resize(t.total_bins);
    for (int64_t i = 0; i < t.total_bins; i++) {
        t.pos_bins[i].pos   = pd_read_u16(f);
        t.pos_bins[i].count = pd_read_u32(f);
    }
    fclose(f);
    return t;
}

// --- Precompute eff_cos / eff_sin tables ---
//
// Two modes:
//   mode=expected  → use scalar expected_pos[sid] as the position.
//                    eff_cos[sid][i] = cos(E[p] / base^(2i/HD))
//                    eff_sin[sid][i] = sin(E[p] / base^(2i/HD))
//
//   mode=dist_rope → encode the full distribution.
//                    eff_cos[sid][i] = Σ_p (count[p]/total) · cos(p / base^(2i/HD))
//                    eff_sin[sid][i] = Σ_p (count[p]/total) · sin(p / base^(2i/HD))
//
// Output: row-major [substring_count, HD] for each of cos and sin.
// Convention matches the existing cos_cache / sin_cache layout the RoPE
// kernel reads (`cos_cache[pos * dim + j]`).
//
// Substrings with zero bins (suffix-only entries in a prefix-keyed lookup,
// or vice versa) get cos=1, sin=0 (identity rotation, position-0).
enum class PosEncoderMode { Default = 0, Expected = 1, DistRope = 2 };

static void compute_eff_rope_caches(const PositionTable& table,
                                    int head_dim,
                                    float rope_base,
                                    PosEncoderMode mode,
                                    std::vector<float>& out_cos,
                                    std::vector<float>& out_sin) {
    const int sc = table.substring_count;
    const int hd = head_dim;
    out_cos.assign((size_t)sc * hd, 1.0f);
    out_sin.assign((size_t)sc * hd, 0.0f);

    // Precompute inverse-frequencies: inv_freq[i] = 1 / base^(2i/HD) for i in [0, HD/2)
    std::vector<double> inv_freq(hd / 2);
    for (int i = 0; i < hd / 2; i++) {
        inv_freq[i] = 1.0 / std::pow((double)rope_base, (2.0 * i) / (double)hd);
    }

    for (int sid = 0; sid < sc; sid++) {
        int n;
        const PosBin* bins = table.bins_for(sid, n);
        if (n == 0) continue;  // keep identity

        if (mode == PosEncoderMode::Expected) {
            // Compute expected position
            double sum_pc = 0.0, sum_c = 0.0;
            for (int i = 0; i < n; i++) {
                sum_pc += (double)bins[i].pos * (double)bins[i].count;
                sum_c  += (double)bins[i].count;
            }
            double e_p = sum_pc / sum_c;
            for (int i = 0; i < hd / 2; i++) {
                double angle = e_p * inv_freq[i];
                float c = (float)std::cos(angle);
                float s = (float)std::sin(angle);
                // RoPE kernel reads cos_cache[pos*dim + j0] with j0 = 2*i,
                // and the build_rope_cache helper populates both 2i and 2i+1
                // with the same value (so both pair-elements look up the
                // same rotation). Mirror that layout here.
                out_cos[(size_t)sid * hd + 2 * i]     = c;
                out_cos[(size_t)sid * hd + 2 * i + 1] = c;
                out_sin[(size_t)sid * hd + 2 * i]     = s;
                out_sin[(size_t)sid * hd + 2 * i + 1] = s;
            }
        } else if (mode == PosEncoderMode::DistRope) {
            // Convolve cos/sin against the position distribution
            double total = 0.0;
            for (int i = 0; i < n; i++) total += (double)bins[i].count;
            for (int i = 0; i < hd / 2; i++) {
                double acc_c = 0.0, acc_s = 0.0;
                for (int b = 0; b < n; b++) {
                    double w = (double)bins[b].count / total;
                    double angle = (double)bins[b].pos * inv_freq[i];
                    acc_c += w * std::cos(angle);
                    acc_s += w * std::sin(angle);
                }
                float c = (float)acc_c;
                float s = (float)acc_s;
                out_cos[(size_t)sid * hd + 2 * i]     = c;
                out_cos[(size_t)sid * hd + 2 * i + 1] = c;
                out_sin[(size_t)sid * hd + 2 * i]     = s;
                out_sin[(size_t)sid * hd + 2 * i + 1] = s;
            }
        }
    }
}

#endif  // AGPT_POSITION_DATA_IO_CUH
