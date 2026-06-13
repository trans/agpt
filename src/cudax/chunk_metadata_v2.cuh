#ifndef AGPT_V2_CHUNK_METADATA_V2_CUH
#define AGPT_V2_CHUNK_METADATA_V2_CUH

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "chunk_plan.cuh"
#include "position_sampling_v2.cuh"
#include "successor_prefix_v2.cuh"
#include "target_sidecar_v2.cuh"
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
    float* h_query_grad_scales = nullptr;
    int* h_rope_positions = nullptr;
    int* h_char_pos = nullptr;
    int* h_query_depth = nullptr;
    int* h_anc_ids = nullptr;
    int* h_anc_offsets = nullptr;
    int* h_anc_lengths = nullptr;
    int* h_own_lengths = nullptr;
    int* h_read_pos_flat = nullptr;
    int* h_target_counts_offset = nullptr;
    int* h_target_counts_len = nullptr;
    int* h_target_counts_tok = nullptr;
    int* h_target_counts_val = nullptr;
    int target_counts_total = 0;
    long long phase_target_nodes = 0;
    long long phase_target_zero_nodes = 0;
    long long phase_target_singleton_nodes = 0;
    long long phase_target_prefix_mass = 0;
    long long phase_target_global_mass = 0;
    long long phase_target_local_mass = 0;
    double phase_target_global_entropy_mass = 0.0;
    double phase_target_local_entropy_mass = 0.0;
};

static inline void free_chunk_metadata_v2(ChunkMetadataV2& m) {
    std::free(m.h_radix_ids);
    std::free(m.h_query_offsets);
    std::free(m.h_kv_offsets);
    std::free(m.h_kv_lengths);
    std::free(m.h_query_to_node);
    std::free(m.h_token_ids);
    std::free(m.h_query_weights);
    std::free(m.h_query_grad_scales);
    std::free(m.h_rope_positions);
    std::free(m.h_char_pos);
    std::free(m.h_query_depth);
    std::free(m.h_anc_ids);
    std::free(m.h_anc_offsets);
    std::free(m.h_anc_lengths);
    std::free(m.h_own_lengths);
    std::free(m.h_read_pos_flat);
    std::free(m.h_target_counts_offset);
    std::free(m.h_target_counts_len);
    std::free(m.h_target_counts_tok);
    std::free(m.h_target_counts_val);
    m = ChunkMetadataV2{};
}

static inline float query_mass_weight_from_count_v2(const TrainerConfig& cfg, float count) {
    if (count <= 0.0f) return 0.0f;
    switch (cfg.mass_weight) {
        case MassWeightModeV2::Off: return 1.0f;
        case MassWeightModeV2::Linear: return count;
        case MassWeightModeV2::Sqrt: return sqrtf(count);
        case MassWeightModeV2::Log: return logf(1.0f + count);
        case MassWeightModeV2::InvLog: return 1.0f / logf(2.0f + count);
        case MassWeightModeV2::InvLinear: return 1.0f / (1.0f + count);
    }
    return count;
}

static inline float query_mass_weight_v2(const TrainerConfig& cfg, int edge_mass) {
    float count = (float)(edge_mass > 0 ? edge_mass : 1);
    return query_mass_weight_from_count_v2(cfg, count);
}

static inline uint64_t dropout_mix64_v2(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

static inline bool dropout_keep_node_v2(const TrainerConfig& cfg,
                                        const TrainingUnit& unit,
                                        int radix_id,
                                        int epoch_index,
                                        int optimizer_step_index) {
    if (cfg.dropout_node_keep_prob >= 1.0f) return true;
    uint64_t h = (uint64_t)(cfg.dropout_seed ? cfg.dropout_seed : 1U);
    h ^= (uint64_t)(unsigned)(unit.root_child_id + 0x9e3779b9U) * 0xbf58476d1ce4e5b9ULL;
    h ^= (uint64_t)(unsigned)(radix_id + 0x85ebca6bU) * 0x94d049bb133111ebULL;
    h ^= (uint64_t)(unsigned)(epoch_index + 1) * 0x9e3779b97f4a7c15ULL;
    h ^= (uint64_t)(unsigned)(optimizer_step_index + 1) * 0xd2b74407b1ce6e93ULL;
    h = dropout_mix64_v2(h);
    float u = (float)((h >> 40) & 0xFFFFFFULL) / (float)0x1000000ULL;
    return u < cfg.dropout_node_keep_prob;
}

static inline float endpoint_entropy_norm_v2(const RadixTrieStructure& trie,
                                             int radix_id,
                                             int vocab_size) {
    int start = trie.counts_offset[radix_id];
    int end = start + trie.counts_len[radix_id];
    long long total = 0;
    for (int e = start; e < end; e++) total += trie.counts_val[e];
    if (total <= 0 || vocab_size <= 1) return 0.0f;
    double h = 0.0;
    double inv_total = 1.0 / (double)total;
    for (int e = start; e < end; e++) {
        int cnt = trie.counts_val[e];
        if (cnt <= 0) continue;
        double p = (double)cnt * inv_total;
        h -= p * std::log(p);
    }
    double denom = std::log((double)vocab_size);
    if (denom <= 0.0) return 0.0f;
    double norm = h / denom;
    if (norm < 0.0) norm = 0.0;
    if (norm > 1.0) norm = 1.0;
    return (float)norm;
}

static inline float entropy_gate_scale_v2(const TrainerConfig& cfg,
                                          float entropy_norm) {
    if (cfg.entropy_gate_min_scale >= 1.0f) return 1.0f;
    if (entropy_norm < 0.0f) entropy_norm = 0.0f;
    if (entropy_norm > 1.0f) entropy_norm = 1.0f;
    return cfg.entropy_gate_min_scale +
           (1.0f - cfg.entropy_gate_min_scale) * entropy_norm;
}

static inline float entropy_grad_scale_v2(const TrainerConfig& cfg,
                                          float entropy_norm) {
    if (cfg.entropy_grad_min_scale >= 1.0f) return 1.0f;
    if (entropy_norm < 0.0f) entropy_norm = 0.0f;
    if (entropy_norm > 1.0f) entropy_norm = 1.0f;
    return cfg.entropy_grad_min_scale +
           (1.0f - cfg.entropy_grad_min_scale) * entropy_norm;
}

static inline double count_distribution_entropy_v2(const std::vector<int>& counts,
                                                   size_t start,
                                                   size_t end,
                                                   long long total) {
    if (total <= 0) return 0.0;
    double inv_total = 1.0 / (double)total;
    double h = 0.0;
    for (size_t i = start; i < end; i++) {
        if (counts[i] <= 0) continue;
        double p = (double)counts[i] * inv_total;
        h -= p * std::log(p);
    }
    return h;
}

static inline ChunkMetadataV2 build_chunk_metadata_v2(const TrainerConfig& cfg,
                                                      const RuntimeShape& shape,
                                                      const RadixTrieStructure& trie,
                                                      const TrainingUnit& unit,
                                                      const ChunkPlan& chunk,
                                                      const PositionSamplingStageV2* pos_stage = nullptr,
                                                      int epoch_index = 0,
                                                      int optimizer_step_index = 0,
                                                      const SuccessorPrefixTableV2* successor_table = nullptr,
                                                      const TargetSidecarTableV2* target_sidecar = nullptr) {
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
    out.h_query_grad_scales = (float*)std::malloc(out.T_q * sizeof(float));
    out.h_rope_positions = (int*)std::malloc((long long)out.T_q * shape.n_heads * sizeof(int));
    out.h_char_pos = (int*)std::malloc(out.T_q * sizeof(int));
    out.h_query_depth = (int*)std::malloc(out.T_q * sizeof(int));

    int q_fill = 0;
    int kv_fill = 0;
    int t_anc = 0;
    int rope_seq_len = cfg.rope_seq_len > 0 ? cfg.rope_seq_len : cfg.seq_len;
    bool phase_conditioned_targets =
        cfg.rope_position_mode == RopePositionModeV2::PhaseConditioned &&
        pos_stage && pos_stage->data;
    bool sidecar_targets = target_sidecar && pos_stage && pos_stage->data;
    bool local_soft_targets = phase_conditioned_targets || sidecar_targets;
    std::vector<int> prefix_tokens;
    std::vector<int> target_toks;
    std::vector<int> target_vals;
    if (local_soft_targets) {
        out.h_target_counts_offset = (int*)std::malloc((out.N + 1) * sizeof(int));
        out.h_target_counts_len = (int*)std::calloc(out.N > 0 ? out.N : 1, sizeof(int));
    }
    for (int i = 0; i < out.N; i++) {
        int unit_node_index = chunk.start_node_index + i;
        int r = unit.radix_ids[unit_node_index];
        bool context_only_node = unit.context_only && unit.context_only[unit_node_index];
        out.h_radix_ids[i] = r;
        int anc_len = trie.ancestor_char_offsets[r + 1] - trie.ancestor_char_offsets[r];
        int own_len = trie.edge_lens[r];
        int edge_mass = trie.edge_mass[r] > 0 ? trie.edge_mass[r] : 1;
        int edge_start = trie.edge_starts[r];
        int fcd = trie.edge_first_char_depths[r];
        int succ = successor_prefix_successor_v2(successor_table, r);
        int succ_anc_len = 0;
        int succ_own_len = 0;
        int succ_len = 0;
        int succ_rope_base = 0;
        if (succ > 0 && succ < trie.radix_count) {
            succ_anc_len = trie.ancestor_char_offsets[succ + 1] - trie.ancestor_char_offsets[succ];
            succ_own_len = trie.edge_lens[succ];
            succ_len = succ_anc_len + succ_own_len;
            succ_rope_base = successor_prefix_rope_base_v2(trie, successor_table, r);
        }
        int sampled_start = -1;
        int effective_mass = edge_mass;
        bool keep_dropout_node = dropout_keep_node_v2(cfg, unit, r, epoch_index, optimizer_step_index);
        float endpoint_entropy_norm = endpoint_entropy_norm_v2(trie, r, shape.vocab_size);
        if (cfg.rope_position_mode == RopePositionModeV2::SampledBin) {
            sampled_start = sample_prefix_start_bin_v2(pos_stage, r, epoch_index,
                                                       unit.root_child_id, optimizer_step_index);
        } else if (cfg.rope_position_mode == RopePositionModeV2::PhaseSweep ||
                   cfg.rope_position_mode == RopePositionModeV2::PhaseWeighted ||
                   cfg.rope_position_mode == RopePositionModeV2::PhaseConditioned) {
            sampled_start = sample_prefix_start_unit_phase_v2(pos_stage, trie, r, epoch_index,
                                                              unit.root_child_id,
                                                              cfg.rope_position_offset,
                                                              cfg.rope_phase_shuffle,
                                                              cfg.rope_phase_shuffle_seed);
            if (cfg.rope_position_mode == RopePositionModeV2::PhaseWeighted ||
                cfg.rope_position_mode == RopePositionModeV2::PhaseConditioned) {
                effective_mass = prefix_position_mass_for_presentation_start_v2(
                    pos_stage, trie, r, sampled_start);
            }
        }
        float query_weight = query_mass_weight_from_count_v2(cfg, (float)effective_mass);

        out.h_query_offsets[i] = q_fill;
        out.h_kv_offsets[i] = kv_fill;
        out.h_kv_lengths[i] = anc_len + own_len + succ_len;
        t_anc += anc_len;

        for (int j = 0; j < own_len; j++) {
            out.h_query_to_node[q_fill + j] = i;
            out.h_token_ids[q_fill + j] = trie.edge_tokens_flat[edge_start + j];
            int query_depth = fcd + j;
            float depth_query_weight = query_weight;
            if (cfg.loss_depth_min > 0 &&
                (query_depth < cfg.loss_depth_min || query_depth > cfg.loss_depth_max)) {
                depth_query_weight = 0.0f;
            }
            if (context_only_node) depth_query_weight = 0.0f;
            if (!keep_dropout_node) depth_query_weight = 0.0f;
            float entropy_norm = (j + 1 == own_len) ? endpoint_entropy_norm : 0.0f;
            depth_query_weight *= entropy_gate_scale_v2(cfg, entropy_norm);
            out.h_query_weights[q_fill + j] = depth_query_weight;
            out.h_query_grad_scales[q_fill + j] =
                entropy_grad_scale_v2(cfg, endpoint_entropy_norm);
            out.total_query_weight += (double)depth_query_weight;
            out.h_char_pos[q_fill + j] = edge_start + j;
            out.h_query_depth[q_fill + j] = query_depth;
            int pos = fcd + j - 1;
            if (sampled_start >= 0) {
                int sampled_pos = (cfg.rope_position_mode == RopePositionModeV2::SampledBin)
                    ? sampled_rope_pos_from_start_v2(
                        sampled_start, fcd + j - 1, pos_stage->data->prefix_table.window_size)
                    : presentation_rope_pos_from_start_v2(
                        sampled_start, fcd + j - 1, pos_stage->data->prefix_table.window_size);
                if (sampled_pos >= 0) pos = sampled_pos;
            }
            if (pos < 0) pos = 0;
            if (pos >= rope_seq_len) pos = rope_seq_len - 1;
            for (int h = 0; h < shape.n_heads; h++) {
                out.h_rope_positions[(q_fill + j) * shape.n_heads + h] = pos;
            }
        }
        if (succ_len > 0) {
            int out_j = 0;
            int succ_anc_off = trie.ancestor_char_offsets[succ];
            for (int a = 0; a < succ_anc_len; a++, out_j++) {
                int char_pos = trie.ancestor_char_ids[succ_anc_off + a];
                out.h_query_to_node[q_fill + own_len + out_j] = i;
                out.h_token_ids[q_fill + own_len + out_j] = trie.edge_tokens_flat[char_pos];
                out.h_query_weights[q_fill + own_len + out_j] = 0.0f;
                out.h_query_grad_scales[q_fill + own_len + out_j] = 1.0f;
                out.h_char_pos[q_fill + own_len + out_j] = -1;
                out.h_query_depth[q_fill + own_len + out_j] = succ_rope_base + out_j + 1;
                int pos = succ_rope_base + out_j;
                if (pos < 0) pos = 0;
                if (pos >= rope_seq_len) pos = rope_seq_len - 1;
                for (int h = 0; h < shape.n_heads; h++) {
                    out.h_rope_positions[(q_fill + own_len + out_j) * shape.n_heads + h] = pos;
                }
            }
            int succ_edge_start = trie.edge_starts[succ];
            for (int a = 0; a < succ_own_len; a++, out_j++) {
                int char_pos = succ_edge_start + a;
                out.h_query_to_node[q_fill + own_len + out_j] = i;
                out.h_token_ids[q_fill + own_len + out_j] = trie.edge_tokens_flat[char_pos];
                out.h_query_weights[q_fill + own_len + out_j] = 0.0f;
                out.h_query_grad_scales[q_fill + own_len + out_j] = 1.0f;
                out.h_char_pos[q_fill + own_len + out_j] = -1;
                out.h_query_depth[q_fill + own_len + out_j] = succ_rope_base + out_j + 1;
                int pos = succ_rope_base + out_j;
                if (pos < 0) pos = 0;
                if (pos >= rope_seq_len) pos = rope_seq_len - 1;
                for (int h = 0; h < shape.n_heads; h++) {
                    out.h_rope_positions[(q_fill + own_len + out_j) * shape.n_heads + h] = pos;
                }
            }
        }
        q_fill += own_len + succ_len;
        kv_fill += anc_len + own_len + succ_len;

        if (local_soft_targets) {
            out.h_target_counts_offset[i] = (int)target_toks.size();
            int endpoint_depth_zero_based = fcd + own_len - 2;
            if (endpoint_depth_zero_based < 0) endpoint_depth_zero_based = 0;
            int count_start = trie.counts_offset[r];
            int count_end = count_start + trie.counts_len[r];
            std::vector<int> global_vals;
            global_vals.reserve((size_t)(count_end - count_start));
            long long global_total = 0;
            long long local_total = 0;
            size_t local_start = target_vals.size();
            for (int e = count_start; e < count_end; e++) {
                int global_cnt = trie.counts_val[e];
                if (global_cnt > 0) {
                    global_vals.push_back(global_cnt);
                    global_total += (long long)global_cnt;
                }
            }
            if (sidecar_targets) {
                int sid = (r >= 0 && r < (int)pos_stage->substring_id_by_radix.size())
                    ? pos_stage->substring_id_by_radix[(size_t)r]
                    : -1;
                int side_start = 0;
                int side_end = 0;
                long long side_total = 0;
                if (sid >= 0 && sid < (int)target_sidecar->substring_count) {
                    side_start = target_sidecar->offsets[(size_t)sid];
                    side_end = target_sidecar->offsets[(size_t)sid + 1];
                    for (int e = side_start; e < side_end; e++) {
                        side_total += (long long)target_sidecar->entries[(size_t)e].count;
                    }
                }
                float mix = cfg.target_sidecar_mix;
                if (mix < 0.0f) mix = 0.0f;
                if (mix > 1.0f) mix = 1.0f;
                if (mix >= 0.999999f || global_total <= 0) {
                    for (int e = side_start; e < side_end; e++) {
                        const TargetSidecarEntryV2& entry = target_sidecar->entries[(size_t)e];
                        if (entry.count > 0) {
                            target_toks.push_back((int)entry.token);
                            target_vals.push_back((int)entry.count);
                            local_total += (long long)entry.count;
                        }
                    }
                } else {
                    std::vector<double> mixed((size_t)shape.vocab_size, 0.0);
                    double raw_weight = 1.0 - (double)mix;
                    double side_weight = side_total > 0 ? (double)mix : 0.0;
                    double weight_total = raw_weight + side_weight;
                    if (weight_total <= 0.0) weight_total = 1.0;
                    raw_weight /= weight_total;
                    side_weight /= weight_total;
                    double inv_global = global_total > 0 ? 1.0 / (double)global_total : 0.0;
                    for (int e = count_start; e < count_end; e++) {
                        int tok = trie.counts_tok[e];
                        int cnt = trie.counts_val[e];
                        if (tok >= 0 && tok < shape.vocab_size && cnt > 0) {
                            mixed[(size_t)tok] += raw_weight * (double)cnt * inv_global;
                        }
                    }
                    double inv_side = side_total > 0 ? 1.0 / (double)side_total : 0.0;
                    for (int e = side_start; e < side_end; e++) {
                        const TargetSidecarEntryV2& entry = target_sidecar->entries[(size_t)e];
                        int tok = (int)entry.token;
                        if (tok >= 0 && tok < shape.vocab_size && entry.count > 0) {
                            mixed[(size_t)tok] += side_weight * (double)entry.count * inv_side;
                        }
                    }
                    int scale = target_sidecar->scale > 0 ? (int)target_sidecar->scale : 1000000;
                    int q_total = 0;
                    int best_tok = -1;
                    double best_p = -1.0;
                    size_t quant_start = target_vals.size();
                    for (int tok = 0; tok < shape.vocab_size; tok++) {
                        double p = mixed[(size_t)tok];
                        if (p > best_p) {
                            best_p = p;
                            best_tok = tok;
                        }
                        int q = (int)llround(p * (double)scale);
                        if (q > 0) {
                            target_toks.push_back(tok);
                            target_vals.push_back(q);
                            q_total += q;
                        }
                    }
                    if (q_total <= 0 && best_tok >= 0) {
                        target_toks.push_back(best_tok);
                        target_vals.push_back(scale);
                        q_total = scale;
                    } else if (q_total != scale && best_tok >= 0) {
                        bool adjusted = false;
                        for (size_t p = quant_start; p < target_vals.size(); p++) {
                            if (target_toks[p] == best_tok) {
                                target_vals[p] += scale - q_total;
                                adjusted = true;
                                break;
                            }
                        }
                        if (!adjusted) {
                            target_toks.push_back(best_tok);
                            target_vals.push_back(scale - q_total);
                        }
                    }
                    local_total += (long long)scale;
                }
            } else {
                int target_endpoint_phase = presentation_rope_pos_from_start_v2(
                    sampled_start, endpoint_depth_zero_based + 1,
                    pos_stage->data->prefix_table.window_size);
                bool use_direct_targets = has_prefix_phase_targets_v2(pos_stage);
                if (!use_direct_targets) radix_prefix_tokens_v2(trie, r, prefix_tokens);
                for (int e = count_start; e < count_end; e++) {
                    int tok = trie.counts_tok[e];
                    int cnt = 0;
                    if (use_direct_targets) {
                        cnt = prefix_phase_target_count_v2(pos_stage, r, sampled_start, tok);
                    } else {
                        prefix_tokens.push_back(tok);
                        auto it = pos_stage->data->substring_id_by_tokens.find(pos_token_key_v2(prefix_tokens));
                        if (it != pos_stage->data->substring_id_by_tokens.end()) {
                            cnt = substring_position_endpoint_phase_bin_count_v2(
                                pos_stage, it->second, endpoint_depth_zero_based + 1, target_endpoint_phase);
                        }
                        prefix_tokens.pop_back();
                    }
                    if (cnt > 0) {
                        target_toks.push_back(tok);
                        target_vals.push_back(cnt);
                        local_total += (long long)cnt;
                    }
                }
            }
            out.h_target_counts_len[i] =
                (int)target_toks.size() - out.h_target_counts_offset[i];
            out.phase_target_nodes++;
            if (out.h_target_counts_len[i] == 0) out.phase_target_zero_nodes++;
            if (out.h_target_counts_len[i] == 1) out.phase_target_singleton_nodes++;
            out.phase_target_prefix_mass += (long long)effective_mass;
            out.phase_target_global_mass += global_total;
            out.phase_target_local_mass += local_total;
            out.phase_target_global_entropy_mass +=
                (double)global_total * count_distribution_entropy_v2(global_vals, 0, global_vals.size(), global_total);
            out.phase_target_local_entropy_mass +=
                (double)local_total * count_distribution_entropy_v2(target_vals, local_start, target_vals.size(), local_total);
        }
    }
    out.h_query_offsets[out.N] = q_fill;
    out.h_kv_offsets[out.N] = kv_fill;
    if (local_soft_targets) {
        out.h_target_counts_offset[out.N] = (int)target_toks.size();
        out.target_counts_total = (int)target_toks.size();
        out.h_target_counts_tok = (int*)std::malloc((out.target_counts_total > 0 ? out.target_counts_total : 1) * sizeof(int));
        out.h_target_counts_val = (int*)std::malloc((out.target_counts_total > 0 ? out.target_counts_total : 1) * sizeof(int));
        if (out.target_counts_total > 0) {
            std::memcpy(out.h_target_counts_tok, target_toks.data(), (size_t)out.target_counts_total * sizeof(int));
            std::memcpy(out.h_target_counts_val, target_vals.data(), (size_t)out.target_counts_total * sizeof(int));
        }
    }
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
        } else if (cfg.rope_position_mode == RopePositionModeV2::PhaseSweep ||
                   cfg.rope_position_mode == RopePositionModeV2::PhaseWeighted ||
                   cfg.rope_position_mode == RopePositionModeV2::PhaseConditioned) {
            sampled_start = sample_prefix_start_unit_phase_v2(pos_stage, trie, r, epoch_index,
                                                              unit.root_child_id,
                                                              cfg.rope_position_offset,
                                                              cfg.rope_phase_shuffle,
                                                              cfg.rope_phase_shuffle_seed);
        }
        int anc_off = trie.ancestor_char_offsets[r];
        int anc_len = trie.ancestor_char_offsets[r + 1] - anc_off;
        out.h_anc_offsets[i] = anc_fill;
        out.h_anc_lengths[i] = anc_len;
        out.h_own_lengths[i] = trie.edge_lens[r] + successor_prefix_path_len_v2(trie, successor_table, r);
        for (int a = 0; a < anc_len; a++) {
            int char_pos = trie.ancestor_char_ids[anc_off + a];
            out.h_anc_ids[anc_fill] = char_pos;
            int read_pos = trie.real_pos_of_char[char_pos];
            if (sampled_start >= 0) {
                int sampled_pos = (cfg.rope_position_mode == RopePositionModeV2::SampledBin)
                    ? sampled_rope_pos_from_start_v2(
                        sampled_start, read_pos, pos_stage->data->prefix_table.window_size)
                    : presentation_rope_pos_from_start_v2(
                        sampled_start, read_pos, pos_stage->data->prefix_table.window_size);
                if (sampled_pos >= 0) read_pos = sampled_pos;
            }
            if (read_pos >= rope_seq_len) read_pos = rope_seq_len - 1;
            out.h_read_pos_flat[anc_fill] = read_pos;
            anc_fill++;
        }
    }
    out.h_anc_offsets[out.N] = anc_fill;

    return out;
}

}  // namespace agpt_v2

#endif
