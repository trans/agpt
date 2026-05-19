#ifndef AGPT_CHUNK_METADATA_CUH
#define AGPT_CHUNK_METADATA_CUH

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
        for (int j = 0; j < L; j++) {
            out.h_query_to_node[q_fill + j] = i;
            out.h_token_ids[q_fill + j] = ctx.trie.edge_tokens_flat[edge_start + j];
            int pos = fcd + j - 1;
            if (pos < 0) pos = 0;
            if (pos >= ctx.cfg.seq_len) pos = ctx.cfg.seq_len - 1;
            for (int h = 0; h < ctx.H; h++) {
                out.h_rope_positions[(q_fill + j) * ctx.H + h] = pos;
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
