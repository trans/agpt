#ifndef AGPT_V1_BACKOFF_TABLE_CUH
#define AGPT_V1_BACKOFF_TABLE_CUH

// Sidecar loader + reverse-lookup builder for the slot-selection Step 0
// backoff slots experiment. See notes/seq-len-extension/slot-selection.md.
//
// Sidecar binary format (produced by bin/agpt_build_backoff_table):
//   magic     u32 = 'BKOF' (0x464F4B42)
//   version   u32 = 1
//   n_radix   u32   number of radix nodes (must match the trie)
//   B         u32   backoff levels per node
//   case2     u64   total case-2 sentinels (diagnostic)
//   shallow   u64   total shallow-K sentinels (diagnostic)
//   sidecar   u32[n_radix * B]
//                   sidecar[k*B + i] = K_back_(i+1).id, or 0xFFFFFFFF
//                   (BackoffTable::SENTINEL_ID) if no such node exists.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

struct BackoffTable {
    static constexpr uint32_t MAGIC = 0x464F4B42u;        // 'BKOF' (LE)
    static constexpr uint32_t VERSION = 1u;
    static constexpr uint32_t SENTINEL_ID = 0xFFFFFFFFu;

    uint32_t n_radix = 0;
    uint32_t b = 0;
    uint64_t case2_count = 0;
    uint64_t shallow_count = 0;

    // Forward map: sidecar[k * b + i] = K_back_(i+1).id or SENTINEL_ID.
    // Sized n_radix * b. Host memory.
    std::vector<uint32_t> sidecar;

    // Reverse lookup: rev_offsets[M] .. rev_offsets[M+1] is a contiguous
    // slice of rev_entries holding all (K.id, i) pairs that name M as
    // their K_back_(i+1). Built once from `sidecar`.
    //   rev_offsets : [n_radix + 1]
    //   rev_entries : packed [K_id (uint32), backoff_level_i (uint32)] pairs
    //                 (i = 0..B-1 corresponding to backoff level i+1)
    std::vector<uint32_t> rev_offsets;
    std::vector<uint32_t> rev_entries;  // 2 entries per pair: [K, i, K, i, ...]

    bool load(const char* path, int expected_n_radix) {
        FILE* f = std::fopen(path, "rb");
        if (!f) {
            std::fprintf(stderr, "agpt_train: failed to open backoff sidecar: %s\n", path);
            return false;
        }
        struct Header {
            uint32_t magic;
            uint32_t version;
            uint32_t n_radix;
            uint32_t b;
            uint64_t case2_count;
            uint64_t shallow_count;
        } hdr{};
        if (std::fread(&hdr, sizeof(hdr), 1, f) != 1) {
            std::fprintf(stderr, "agpt_train: backoff sidecar header read failed: %s\n", path);
            std::fclose(f);
            return false;
        }
        if (hdr.magic != MAGIC) {
            std::fprintf(stderr,
                         "agpt_train: backoff sidecar bad magic 0x%08x (expected 0x%08x) in %s\n",
                         hdr.magic, MAGIC, path);
            std::fclose(f);
            return false;
        }
        if (hdr.version != VERSION) {
            std::fprintf(stderr,
                         "agpt_train: backoff sidecar version %u unsupported (this build expects %u): %s\n",
                         hdr.version, VERSION, path);
            std::fclose(f);
            return false;
        }
        if ((int)hdr.n_radix != expected_n_radix) {
            std::fprintf(stderr,
                         "agpt_train: backoff sidecar n_radix=%u does not match loaded trie's %d: %s\n",
                         hdr.n_radix, expected_n_radix, path);
            std::fclose(f);
            return false;
        }
        n_radix = hdr.n_radix;
        b = hdr.b;
        case2_count = hdr.case2_count;
        shallow_count = hdr.shallow_count;

        size_t total = (size_t)n_radix * (size_t)b;
        sidecar.resize(total);
        if (std::fread(sidecar.data(), sizeof(uint32_t), total, f) != total) {
            std::fprintf(stderr, "agpt_train: backoff sidecar payload short-read: %s\n", path);
            std::fclose(f);
            return false;
        }
        std::fclose(f);

        std::fprintf(stderr,
                     "agpt_train: loaded backoff sidecar %s (n_radix=%u, B=%u, "
                     "case2=%llu, shallow=%llu)\n",
                     path, n_radix, b,
                     (unsigned long long)case2_count,
                     (unsigned long long)shallow_count);
        return true;
    }

    // Build the reverse lookup: for each radix node M, which (K, i) pairs
    // named M as their K_back_(i+1)? Used at runtime so that when M's
    // hidden state h_M is computed, we can scatter it into every stash
    // slot that wants it. Counting + offset + scatter, two passes.
    void build_rev_lookup() {
        if (n_radix == 0 || b == 0) return;
        rev_offsets.assign((size_t)n_radix + 1, 0u);

        // Pass 1: count entries per target.
        size_t total = (size_t)n_radix * (size_t)b;
        for (size_t s = 0; s < total; ++s) {
            uint32_t target = sidecar[s];
            if (target == SENTINEL_ID) continue;
            // bounds-check: a corrupted sidecar with target >= n_radix
            // would silently scribble the cumulative-sum array; surface it.
            if (target >= n_radix) continue;
            ++rev_offsets[target + 1];
        }
        // Cumulative sum to convert counts to start offsets.
        for (size_t m = 1; m <= n_radix; ++m) {
            rev_offsets[m] += rev_offsets[m - 1];
        }
        size_t pair_count = rev_offsets[n_radix];
        rev_entries.assign(pair_count * 2, 0u);

        // Pass 2: scatter (K, i) pairs.
        // We use a temporary write-cursor per target.
        std::vector<uint32_t> cursor = rev_offsets;
        for (uint32_t k = 0; k < n_radix; ++k) {
            for (uint32_t i = 0; i < b; ++i) {
                uint32_t target = sidecar[(size_t)k * b + i];
                if (target == SENTINEL_ID) continue;
                if (target >= n_radix) continue;
                size_t pos = cursor[target]++;
                rev_entries[pos * 2 + 0] = k;
                rev_entries[pos * 2 + 1] = i;
            }
        }
        // Re-verify final cursor == next offset (sanity).
        for (uint32_t m = 0; m < n_radix; ++m) {
            if (cursor[m] != rev_offsets[m + 1]) {
                std::fprintf(stderr,
                             "agpt_train: backoff rev_lookup sanity check failed at M=%u "
                             "(cursor=%u vs offset=%u)\n",
                             m, cursor[m], rev_offsets[m + 1]);
            }
        }

        // Diagnostics: how many M's have non-empty rev_lookup?
        size_t M_with_lookup = 0;
        size_t max_per_M = 0;
        for (uint32_t m = 0; m < n_radix; ++m) {
            size_t cnt = rev_offsets[m + 1] - rev_offsets[m];
            if (cnt > 0) {
                ++M_with_lookup;
                if (cnt > max_per_M) max_per_M = cnt;
            }
        }
        std::fprintf(stderr,
                     "agpt_train: built backoff rev_lookup: %zu (K,i) pairs, "
                     "%zu/%u nodes have rev_lookup entries (max %zu per node)\n",
                     pair_count, M_with_lookup, n_radix, max_per_M);
    }
};

#endif  // AGPT_V1_BACKOFF_TABLE_CUH
