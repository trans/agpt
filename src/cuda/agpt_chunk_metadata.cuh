#ifndef AGPT_CHUNK_METADATA_CUH
#define AGPT_CHUNK_METADATA_CUH

// RoPE position source. Depth (default) uses the sequential char depth
// within the trie; Mass uses the radix node's edge_mass (= count of
// strings sharing this prefix). Mass-RoPE probe: trie nodes that occur
// more often get a different positional signature than rare nodes.
// Within a radix edge, all chars share the same node, so they share
// the same mass — positional info within edges is collapsed.
enum class RopeMode { Depth = 0, Mass = 1, LogMass = 2, Off = 3, PermDepth = 4, Swap = 5, Split = 6 };

// Secondary signal for split mode: depth-heads always use depth, but the
// other heads use one of these signals as their RoPE position.
enum class SplitSecondary {
    Mass = 0,         // edge_mass (count of strings sharing prefix)
    LogMass = 1,      // floor(log2(edge_mass))
    Branching = 2,    // k = distinct continuations at the radix node
    LogBranching = 3, // floor(log2(k+1))
    CorpusPos = 4,    // canonical corpus position mod W (where this prefix
                      // first occurred in the corpus, wrapped to a window).
                      // Genuinely orthogonal to depth — doesn't measure
                      // anything about the prefix itself, but where in the
                      // document it tends to live.
};

struct ChunkBuildContext {
    const Config& cfg;
    const RadixTrieData& trie;
    const int* radix_list;
    int subtree_offset;
    int n_at_depth;
    int chunk_start;
    int T_q_cap;
    int N_cap;
    int H;
    const char* branch_drop_mask;  // nullable
    int chunk_cycle_shift;
    const int* real_pos_of_char;
    long long T_kv_max;
    RopeMode rope_mode = RopeMode::Depth;
    const int* rope_perm = nullptr;  // size cfg.seq_len; used only in PermDepth mode
    // Split mode: heads [0, split_depth_heads) use depth-positions; heads
    // [split_depth_heads, H) use split_secondary as their position. Lets
    // the model attend on two complementary signals per query.
    int split_depth_heads = 0;
    SplitSecondary split_secondary = SplitSecondary::Mass;
    int corpus_window = 128;  // W for SplitSecondary::CorpusPos (mod window)
};

struct ChunkMetadata {
    int next_chunk_start = 0;
    int chunk_end = 0;
    int N = 0;
    int T_q = 0;
    int T_kv = 0;
    int max_kv_len = 0;
    int T_anc = 0;

    int* h_radix_ids = nullptr;
    int* h_query_offsets = nullptr;
    int* h_kv_offsets = nullptr;
    int* h_kv_lengths = nullptr;
    int* h_query_to_node = nullptr;
    int* h_token_ids = nullptr;
    int* h_rope_positions = nullptr;
    int* h_char_pos = nullptr;
    int* h_query_depth = nullptr;
    int* h_query_d_split = nullptr;
    int* h_anc_ids = nullptr;
    int* h_anc_offsets = nullptr;
    int* h_anc_lengths = nullptr;
    int* h_own_lengths = nullptr;
    int* h_read_pos_flat = nullptr;
};

static void free_chunk_metadata(ChunkMetadata& m) {
    free(m.h_radix_ids);      m.h_radix_ids = nullptr;
    free(m.h_query_offsets);  m.h_query_offsets = nullptr;
    free(m.h_kv_offsets);     m.h_kv_offsets = nullptr;
    free(m.h_kv_lengths);     m.h_kv_lengths = nullptr;
    free(m.h_query_to_node);  m.h_query_to_node = nullptr;
    free(m.h_token_ids);      m.h_token_ids = nullptr;
    free(m.h_rope_positions); m.h_rope_positions = nullptr;
    free(m.h_char_pos);       m.h_char_pos = nullptr;
    free(m.h_query_depth);    m.h_query_depth = nullptr;
    free(m.h_query_d_split);  m.h_query_d_split = nullptr;
    free(m.h_anc_ids);        m.h_anc_ids = nullptr;
    free(m.h_anc_offsets);    m.h_anc_offsets = nullptr;
    free(m.h_anc_lengths);    m.h_anc_lengths = nullptr;
    free(m.h_own_lengths);    m.h_own_lengths = nullptr;
    free(m.h_read_pos_flat);  m.h_read_pos_flat = nullptr;
    m = ChunkMetadata{};
}

static bool build_chunk_metadata(const ChunkBuildContext& ctx, ChunkMetadata& out) {
    out.next_chunk_start = ctx.chunk_start;

    int chunk_end = ctx.chunk_start;
    int T_q = 0;
    while (chunk_end < ctx.n_at_depth) {
        int r = ctx.radix_list[ctx.subtree_offset + chunk_end];
        if (ctx.branch_drop_mask && ctx.branch_drop_mask[r]) {
            chunk_end++;
            continue;
        }
        int L = ctx.trie.edge_lens[r];
        if (T_q + L > ctx.T_q_cap || chunk_end - ctx.chunk_start >= ctx.N_cap) break;
        T_q += L;
        chunk_end++;
    }
    if (chunk_end == ctx.chunk_start) {
        out.next_chunk_start = ctx.chunk_start + 1;
        return false;
    }

    int N;
    if (ctx.branch_drop_mask) {
        N = 0;
        for (int k = ctx.chunk_start; k < chunk_end; k++) {
            int r = ctx.radix_list[ctx.subtree_offset + k];
            if (!ctx.branch_drop_mask[r]) N++;
        }
        if (N == 0) {
            out.next_chunk_start = chunk_end;
            return false;
        }
    } else {
        N = chunk_end - ctx.chunk_start;
    }

    out.h_radix_ids      = (int*)malloc(N * sizeof(int));
    out.h_query_offsets  = (int*)malloc((N + 1) * sizeof(int));
    out.h_kv_offsets     = (int*)malloc((N + 1) * sizeof(int));
    out.h_kv_lengths     = (int*)malloc(N * sizeof(int));
    out.h_query_to_node  = (int*)malloc(T_q * sizeof(int));
    out.h_token_ids      = (int*)malloc(T_q * sizeof(int));
    out.h_rope_positions = (int*)malloc((long long)T_q * ctx.H * sizeof(int));
    out.h_char_pos       = (int*)malloc(T_q * sizeof(int));
    out.h_query_depth    = (int*)malloc(T_q * sizeof(int));
    out.h_query_d_split  = (int*)malloc(T_q * sizeof(int));

    int q_fill = 0;
    int kv_fill = 0;
    int i = 0;
    int k_iter = ctx.chunk_start;
    while (i < N) {
        int r = ctx.radix_list[ctx.subtree_offset + k_iter];
        k_iter++;
        if (ctx.branch_drop_mask && ctx.branch_drop_mask[r]) continue;
        out.h_radix_ids[i] = r;
        int L = ctx.trie.edge_lens[r];
        int anc_len = ctx.trie.ancestor_char_offsets[r + 1] - ctx.trie.ancestor_char_offsets[r];
        int K_i = anc_len + L;
        out.h_query_offsets[i] = q_fill;
        out.h_kv_offsets[i] = kv_fill;
        out.h_kv_lengths[i] = K_i;

        int edge_start = ctx.trie.edge_starts[r];
        int fcd = ctx.trie.edge_first_char_depths[r];
        int node_d_split = ctx.trie.d_split ? ctx.trie.d_split[r] : INT_MAX;
        int mass_pos = 0;
        if (ctx.rope_mode == RopeMode::Mass || ctx.rope_mode == RopeMode::LogMass) {
            int em = ctx.trie.edge_mass[r];
            if (em < 1) em = 1;
            if (ctx.rope_mode == RopeMode::LogMass) {
                // floor(log2(em)) — compresses 170k mass range to ~[0, 17],
                // roughly the same span as depth. Tests whether the *range*
                // of position values matters or only the monotonic ordering.
                int lg = 0;
                int v = em;
                while (v > 1) { v >>= 1; lg++; }
                mass_pos = lg;
            } else {
                mass_pos = em;
            }
        }
        // Precompute depth pos once per (j); split-mode secondary position
        // is per-radix-node (constant across the edge) for most signals,
        // EXCEPT corpus-pos which varies per char-in-edge.
        int em_pos = (ctx.rope_mode == RopeMode::Split) ? ctx.trie.edge_mass[r] : 0;
        if (em_pos < 0) em_pos = 0;
        int split_secondary_pos = 0;
        bool secondary_is_per_char = false;
        if (ctx.rope_mode == RopeMode::Split) {
            switch (ctx.split_secondary) {
                case SplitSecondary::Mass: {
                    split_secondary_pos = em_pos;
                    break;
                }
                case SplitSecondary::LogMass: {
                    int v = em_pos < 1 ? 1 : em_pos;
                    int lg = 0; while (v > 1) { v >>= 1; lg++; }
                    split_secondary_pos = lg;
                    break;
                }
                case SplitSecondary::Branching: {
                    int k = ctx.trie.counts_offset[r + 1] - ctx.trie.counts_offset[r];
                    if (k < 0) k = 0;
                    split_secondary_pos = k;
                    break;
                }
                case SplitSecondary::LogBranching: {
                    int k = ctx.trie.counts_offset[r + 1] - ctx.trie.counts_offset[r];
                    int v = k < 1 ? 1 : k;
                    int lg = 0; while (v > 1) { v >>= 1; lg++; }
                    split_secondary_pos = lg;
                    break;
                }
                case SplitSecondary::CorpusPos: {
                    // Computed per j below; placeholder here.
                    secondary_is_per_char = true;
                    break;
                }
            }
        }
        for (int j = 0; j < L; j++) {
            out.h_query_to_node[q_fill + j] = i;
            out.h_token_ids[q_fill + j] = ctx.trie.edge_tokens_flat[edge_start + j];
            int pos;
            if (ctx.rope_mode == RopeMode::Off) {
                // pos=0 → cos=1, sin=0 → RoPE rotation = identity.
                // Effectively disables RoPE without code changes elsewhere.
                pos = 0;
            } else if (ctx.rope_mode == RopeMode::Mass || ctx.rope_mode == RopeMode::LogMass) {
                pos = mass_pos;
            } else if (ctx.rope_mode == RopeMode::PermDepth || ctx.rope_mode == RopeMode::Swap) {
                // PermDepth: full random permutation of depth indices.
                // Swap: identity except for one transposition (set via
                // --rope-swap A,B). Both routed via rope_perm.
                int d_raw = fcd + j - 1;
                if (d_raw < 0) d_raw = 0;
                if (d_raw >= ctx.cfg.seq_len) d_raw = ctx.cfg.seq_len - 1;
                pos = ctx.rope_perm ? ctx.rope_perm[d_raw] : d_raw;
            } else {
                pos = fcd + j - 1;
                if (pos < 0) pos = 0;
                if (pos >= ctx.cfg.seq_len) pos = ctx.cfg.seq_len - 1;
            }
            if (ctx.rope_mode == RopeMode::Split) {
                // Dispatch by head: depth-heads get clamped depth position;
                // secondary-heads get split_secondary_pos. The rope_cache
                // must cover both ranges (handled in run_radix_training).
                int d_clamp = fcd + j - 1;
                if (d_clamp < 0) d_clamp = 0;
                if (d_clamp >= ctx.cfg.seq_len) d_clamp = ctx.cfg.seq_len - 1;
                int sec = split_secondary_pos;
                if (secondary_is_per_char && ctx.real_pos_of_char) {
                    int cp = ctx.real_pos_of_char[edge_start + j];
                    if (cp < 0) cp = 0;
                    int W = ctx.corpus_window > 0 ? ctx.corpus_window : 128;
                    sec = cp % W;
                }
                for (int h = 0; h < ctx.H; h++) {
                    int p = (h < ctx.split_depth_heads) ? d_clamp : sec;
                    out.h_rope_positions[(q_fill + j) * ctx.H + h] = p;
                }
            } else {
                for (int h = 0; h < ctx.H; h++) {
                    out.h_rope_positions[(q_fill + j) * ctx.H + h] = pos;
                }
            }
            out.h_char_pos[q_fill + j] = edge_start + j;
            out.h_query_depth[q_fill + j] = fcd + j;
            out.h_query_d_split[q_fill + j] = node_d_split;
        }
        q_fill += L;
        kv_fill += K_i;
        i++;
    }
    out.h_query_offsets[N] = q_fill;
    out.h_kv_offsets[N] = kv_fill;
    out.T_kv = kv_fill;
    if ((long long)out.T_kv > ctx.T_kv_max) {
        fprintf(stderr, "Chunk T_kv=%d exceeds T_kv_max=%lld; skip\n", out.T_kv, ctx.T_kv_max);
        out.next_chunk_start = chunk_end;
        free_chunk_metadata(out);
        return false;
    }

    int max_kv_len = 0;
    for (int idx = 0; idx < N; idx++) {
        if (out.h_kv_lengths[idx] > max_kv_len) max_kv_len = out.h_kv_lengths[idx];
    }

    int T_anc = 0;
    for (int idx = 0; idx < N; idx++) {
        int r = out.h_radix_ids[idx];
        T_anc += ctx.trie.ancestor_char_offsets[r + 1] - ctx.trie.ancestor_char_offsets[r];
    }
    out.h_anc_ids       = (int*)malloc((T_anc > 0 ? T_anc : 1) * sizeof(int));
    out.h_anc_offsets   = (int*)malloc((N + 1) * sizeof(int));
    out.h_anc_lengths   = (int*)malloc(N * sizeof(int));
    out.h_own_lengths   = (int*)malloc(N * sizeof(int));
    out.h_read_pos_flat = (int*)malloc((T_anc > 0 ? T_anc : 1) * sizeof(int));
    {
        int fill = 0;
        for (int idx = 0; idx < N; idx++) {
            out.h_anc_offsets[idx] = fill;
            int r = out.h_radix_ids[idx];
            int anc_off = ctx.trie.ancestor_char_offsets[r];
            int anc_len = ctx.trie.ancestor_char_offsets[r + 1] - anc_off;
            for (int a = 0; a < anc_len; a++) {
                int char_pos = ctx.trie.ancestor_char_ids[anc_off + a];
                out.h_anc_ids[fill] = char_pos;
                int vr = ctx.real_pos_of_char[char_pos] + ctx.chunk_cycle_shift;
                if (vr >= ctx.cfg.seq_len) vr = ctx.cfg.seq_len - 1;
                out.h_read_pos_flat[fill] = vr;
                fill++;
            }
            out.h_anc_lengths[idx] = anc_len;
            out.h_own_lengths[idx] = ctx.trie.edge_lens[r];
        }
        out.h_anc_offsets[N] = fill;
    }

    out.next_chunk_start = chunk_end;
    out.chunk_end = chunk_end;
    out.N = N;
    out.T_q = q_fill;
    out.max_kv_len = max_kv_len;
    out.T_anc = T_anc;
    return true;
}

#endif
