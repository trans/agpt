#ifndef AGPT_V1_PRECONDITION_SIDECAR_CUH
#define AGPT_V1_PRECONDITION_SIDECAR_CUH

// Sidecar loader for the precondition strand experiment.
// See notes/seq-len-extension/precondition.md.
//
// Sidecar binary format (produced by bin/agpt_build_precondition_sidecar):
//   magic         u32 = 'PREC' (0x43455250)
//   version       u32 = 2
//   n_radix       u32   must match the loaded trie's radix_count
//   d_pre         u32   precondition prefix length in tokens
//   n_instances   u64   total instances stored across all nodes
//   skipped       u64   instances skipped at build time (diagnostic)
//   offsets       u32[n_radix + 1]
//                       offsets[k]..offsets[k+1] is K's slice into the
//                       instance arrays (in units of *instances*)
//   inst_tokens   i32[n_instances * d_pre]
//                       per-instance d_pre tokens in forward (corpus) order
//   inst_positions u32[n_instances]
//                       per-instance K-start corpus position (v2 addition;
//                       not used by the encoder, kept for future
//                       corpus-order-traversal extensions)

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

struct PreconditionSidecar {
    static constexpr uint32_t MAGIC = 0x43455250u;  // 'PREC' (LE)
    static constexpr uint32_t VERSION = 2u;

    uint32_t n_radix = 0;
    uint32_t d_pre = 0;
    uint64_t n_instances = 0;
    uint64_t skipped_count = 0;  // diagnostic

    // offsets[k]..offsets[k+1] = K's instance slice (in units of *instances*)
    std::vector<uint32_t> offsets;          // [n_radix + 1]

    // Flat tokens: inst_tokens[slot * d_pre + j] = j-th token of slot-th instance
    std::vector<int32_t>  inst_tokens;      // [n_instances * d_pre]

    // Per-instance corpus position (v2 addition).
    std::vector<uint32_t> inst_positions;   // [n_instances]

    bool load(const char* path, int expected_n_radix, int expected_d_pre) {
        FILE* f = std::fopen(path, "rb");
        if (!f) {
            std::fprintf(stderr, "agpt_train: failed to open precondition sidecar: %s\n", path);
            return false;
        }
        struct Header {
            uint32_t magic;
            uint32_t version;
            uint32_t n_radix;
            uint32_t d_pre;
            uint64_t n_instances;
            uint64_t skipped;
        } hdr{};
        if (std::fread(&hdr, sizeof(hdr), 1, f) != 1) {
            std::fprintf(stderr, "agpt_train: precondition sidecar header short-read: %s\n", path);
            std::fclose(f);
            return false;
        }
        if (hdr.magic != MAGIC) {
            std::fprintf(stderr,
                         "agpt_train: precondition sidecar bad magic 0x%08x (expected 0x%08x) in %s\n",
                         hdr.magic, MAGIC, path);
            std::fclose(f);
            return false;
        }
        if (hdr.version != VERSION) {
            std::fprintf(stderr,
                         "agpt_train: precondition sidecar version %u unsupported (this build expects %u): %s\n",
                         hdr.version, VERSION, path);
            std::fclose(f);
            return false;
        }
        if ((int)hdr.n_radix != expected_n_radix) {
            std::fprintf(stderr,
                         "agpt_train: precondition sidecar n_radix=%u does not match loaded trie's %d: %s\n",
                         hdr.n_radix, expected_n_radix, path);
            std::fclose(f);
            return false;
        }
        if ((int)hdr.d_pre != expected_d_pre) {
            std::fprintf(stderr,
                         "agpt_train: precondition sidecar d_pre=%u does not match config %d: %s\n",
                         hdr.d_pre, expected_d_pre, path);
            std::fclose(f);
            return false;
        }
        n_radix = hdr.n_radix;
        d_pre = hdr.d_pre;
        n_instances = hdr.n_instances;
        skipped_count = hdr.skipped;

        // offsets[n_radix + 1]
        offsets.resize((size_t)n_radix + 1);
        if (std::fread(offsets.data(), sizeof(uint32_t), offsets.size(), f) != offsets.size()) {
            std::fprintf(stderr, "agpt_train: precondition sidecar offsets short-read: %s\n", path);
            std::fclose(f);
            return false;
        }

        // inst_tokens[n_instances * d_pre]
        size_t tok_count = (size_t)n_instances * (size_t)d_pre;
        inst_tokens.resize(tok_count);
        if (std::fread(inst_tokens.data(), sizeof(int32_t), tok_count, f) != tok_count) {
            std::fprintf(stderr, "agpt_train: precondition sidecar inst_tokens short-read: %s\n", path);
            std::fclose(f);
            return false;
        }

        // inst_positions[n_instances]
        inst_positions.resize((size_t)n_instances);
        if (std::fread(inst_positions.data(), sizeof(uint32_t), inst_positions.size(), f) != inst_positions.size()) {
            std::fprintf(stderr, "agpt_train: precondition sidecar inst_positions short-read: %s\n", path);
            std::fclose(f);
            return false;
        }

        std::fclose(f);

        // Sanity: offsets[n_radix] should equal n_instances.
        if ((uint64_t)offsets[n_radix] != n_instances) {
            std::fprintf(stderr,
                         "agpt_train: precondition sidecar internal consistency check failed "
                         "(offsets[n_radix]=%u vs n_instances=%llu)\n",
                         offsets[n_radix], (unsigned long long)n_instances);
            return false;
        }

        // Diagnostic summary.
        size_t nodes_with_at_least_one = 0;
        uint32_t max_per_node = 0;
        for (uint32_t r = 0; r < n_radix; r++) {
            uint32_t cnt = offsets[r + 1] - offsets[r];
            if (cnt > 0) {
                ++nodes_with_at_least_one;
                if (cnt > max_per_node) max_per_node = cnt;
            }
        }
        std::fprintf(stderr,
                     "agpt_train: loaded precondition sidecar %s "
                     "(n_radix=%u d_pre=%u n_instances=%llu skipped=%llu, "
                     "%zu/%u nodes have >=1 instance, max %u per node)\n",
                     path, n_radix, d_pre,
                     (unsigned long long)n_instances,
                     (unsigned long long)skipped_count,
                     nodes_with_at_least_one, n_radix, max_per_node);
        double bytes = (double)sizeof(Header)
                     + (double)offsets.size() * sizeof(uint32_t)
                     + (double)inst_tokens.size() * sizeof(int32_t)
                     + (double)inst_positions.size() * sizeof(uint32_t);
        std::fprintf(stderr, "agpt_train: precondition sidecar in-memory: %.1f MB\n",
                     bytes / 1024.0 / 1024.0);
        return true;
    }
};

#endif  // AGPT_V1_PRECONDITION_SIDECAR_CUH
