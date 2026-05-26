#ifndef AGPT_V2_CHUNK_METADATA_V2_CUH
#define AGPT_V2_CHUNK_METADATA_V2_CUH

#include <cstdlib>
#include <cstring>

#include "chunk_plan.cuh"
#include "position_sampling_v2.cuh"
#include "types.cuh"

namespace agpt_v2 {

struct ChunkMetadataV2 {
    int N = 0;
    int T_q = 0;
    int T_kv = 0;
    int T_anc = 0;
    int max_kv_len = 0;
    double total_query_weight = 0.0;

    int* h_radix_ids = nullptr;
    int* h_query_offsets = nullptr;
    int* h_kv_offsets = nullptr;
    int* h_kv_lengths = nullptr;
    int* h_query_to_node = nullptr;
    int* h_token_ids = nullptr;
    float* h_query_weights = nullptr;
    int* h_rope_positions = nullptr;
    int* h_char_pos = nullptr;
    int* h_query_depth = nullptr;
    int* h_anc_ids = nullptr;
    int* h_anc_offsets = nullptr;
    int* h_anc_lengths = nullptr;
    int* h_own_lengths = nullptr;
    int* h_read_pos_flat = nullptr;
};

static inline void free_chunk_metadata_v2(ChunkMetadataV2& m) {
    std::free(m.h_radix_ids);
    std::free(m.h_query_offsets);
    std::free(m.h_kv_offsets);
    std::free(m.h_kv_lengths);
    std::free(m.h_query_to_node);
    std::free(m.h_token_ids);
    std::free(m.h_query_weights);
    std::free(m.h_rope_positions);
    std::free(m.h_char_pos);
    std::free(m.h_query_depth);
    std::free(m.h_anc_ids);
    std::free(m.h_anc_offsets);
    std::free(m.h_anc_lengths);
    std::free(m.h_own_lengths);
    std::free(m.h_read_pos_flat);
    m = ChunkMetadataV2{};
}

static inline ChunkMetadataV2 build_chunk_metadata_v2(const TrainerConfig& cfg,
                                                      const RuntimeShape& shape,
                                                      const RadixTrieStructure& trie,
                                                      const TrainingUnit& unit,
                                                      const ChunkPlan& chunk,
                                                      const PositionSamplingStageV2* pos_stage = nullptr,
                                                      int epoch_index = 0,
                                                      int optimizer_step_index = 0) {
    ChunkMetadataV2 out;
    out.N = chunk.node_count;
    out.T_q = (int)chunk.query_count;
    out.T_kv = (int)chunk.kv_count;
    out.max_kv_len = chunk.max_kv_len;

    out.h_radix_ids = (int*)std::malloc(out.N * sizeof(int));
    out.h_query_offsets = (int*)std::malloc((out.N + 1) * sizeof(int));
    out.h_kv_offsets = (int*)std::malloc((out.N + 1) * sizeof(int));
    out.h_kv_lengths = (int*)std::malloc(out.N * sizeof(int));
    out.h_query_to_node = (int*)std::malloc(out.T_q * sizeof(int));
    out.h_token_ids = (int*)std::malloc(out.T_q * sizeof(int));
    out.h_query_weights = (float*)std::malloc(out.T_q * sizeof(float));
    out.h_rope_positions = (int*)std::malloc((long long)out.T_q * shape.n_heads * sizeof(int));
    out.h_char_pos = (int*)std::malloc(out.T_q * sizeof(int));
    out.h_query_depth = (int*)std::malloc(out.T_q * sizeof(int));

    int q_fill = 0;
    int kv_fill = 0;
    int t_anc = 0;
    for (int i = 0; i < out.N; i++) {
        int r = unit.radix_ids[chunk.start_node_index + i];
        out.h_radix_ids[i] = r;
        int anc_len = trie.ancestor_char_offsets[r + 1] - trie.ancestor_char_offsets[r];
        int own_len = trie.edge_lens[r];
        int edge_mass = trie.edge_mass[r] > 0 ? trie.edge_mass[r] : 1;
        int edge_start = trie.edge_starts[r];
        int fcd = trie.edge_first_char_depths[r];
        int sampled_start = -1;
        if (cfg.rope_position_mode == RopePositionModeV2::SampledBin) {
            sampled_start = sample_prefix_start_bin_v2(pos_stage, r, epoch_index,
                                                       unit.root_child_id, optimizer_step_index);
        }

        out.h_query_offsets[i] = q_fill;
        out.h_kv_offsets[i] = kv_fill;
        out.h_kv_lengths[i] = anc_len + own_len;
        t_anc += anc_len;

        for (int j = 0; j < own_len; j++) {
            out.h_query_to_node[q_fill + j] = i;
            out.h_token_ids[q_fill + j] = trie.edge_tokens_flat[edge_start + j];
            out.h_query_weights[q_fill + j] = (float)edge_mass;
            out.total_query_weight += (double)edge_mass;
            out.h_char_pos[q_fill + j] = edge_start + j;
            out.h_query_depth[q_fill + j] = fcd + j;
            int pos = fcd + j - 1;
            if (sampled_start >= 0) {
                int sampled_pos = sampled_rope_pos_from_start_v2(
                    sampled_start, fcd + j - 1, pos_stage->data->prefix_table.window_size);
                if (sampled_pos >= 0) pos = sampled_pos;
            }
            if (pos < 0) pos = 0;
            if (pos >= cfg.seq_len) pos = cfg.seq_len - 1;
            for (int h = 0; h < shape.n_heads; h++) {
                out.h_rope_positions[(q_fill + j) * shape.n_heads + h] = pos;
            }
        }
        q_fill += own_len;
        kv_fill += anc_len + own_len;
    }
    out.h_query_offsets[out.N] = q_fill;
    out.h_kv_offsets[out.N] = kv_fill;
    out.T_anc = t_anc;

    out.h_anc_ids = (int*)std::malloc((out.T_anc > 0 ? out.T_anc : 1) * sizeof(int));
    out.h_anc_offsets = (int*)std::malloc((out.N + 1) * sizeof(int));
    out.h_anc_lengths = (int*)std::malloc(out.N * sizeof(int));
    out.h_own_lengths = (int*)std::malloc(out.N * sizeof(int));
    out.h_read_pos_flat = (int*)std::malloc((out.T_anc > 0 ? out.T_anc : 1) * sizeof(int));

    int anc_fill = 0;
    for (int i = 0; i < out.N; i++) {
        int r = out.h_radix_ids[i];
        int sampled_start = -1;
        if (cfg.rope_position_mode == RopePositionModeV2::SampledBin) {
            sampled_start = sample_prefix_start_bin_v2(pos_stage, r, epoch_index,
                                                       unit.root_child_id, optimizer_step_index);
        }
        int anc_off = trie.ancestor_char_offsets[r];
        int anc_len = trie.ancestor_char_offsets[r + 1] - anc_off;
        out.h_anc_offsets[i] = anc_fill;
        out.h_anc_lengths[i] = anc_len;
        out.h_own_lengths[i] = trie.edge_lens[r];
        for (int a = 0; a < anc_len; a++) {
            int char_pos = trie.ancestor_char_ids[anc_off + a];
            out.h_anc_ids[anc_fill] = char_pos;
            int read_pos = trie.real_pos_of_char[char_pos];
            if (sampled_start >= 0) {
                int sampled_pos = sampled_rope_pos_from_start_v2(
                    sampled_start, read_pos, pos_stage->data->prefix_table.window_size);
                if (sampled_pos >= 0) read_pos = sampled_pos;
            }
            if (read_pos >= cfg.seq_len) read_pos = cfg.seq_len - 1;
            out.h_read_pos_flat[anc_fill] = read_pos;
            anc_fill++;
        }
    }
    out.h_anc_offsets[out.N] = anc_fill;

    return out;
}

}  // namespace agpt_v2

#endif
