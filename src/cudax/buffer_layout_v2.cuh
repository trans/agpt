#ifndef AGPT_V2_BUFFER_LAYOUT_CUH
#define AGPT_V2_BUFFER_LAYOUT_CUH

#include "runtime_objects.cuh"

namespace agpt_v2 {

struct QueryScratchLayoutV2 {
    float* x = nullptr;
    float* ln_out = nullptr;
    float* q = nullptr;
    float* k = nullptr;
    float* v = nullptr;
    float* attn_out = nullptr;
    float* ff_h = nullptr;
    float* ff_out = nullptr;
};

struct OutputHeadLayoutV2 {
    float* final_out = nullptr;
    float* final_norm = nullptr;
    float* final_std_inv = nullptr;
    float* logits = nullptr;
    float* d_logits = nullptr;
    float* loss = nullptr;
};

struct PackedAttentionLayoutV2 {
    float* kv_pack_k = nullptr;
    float* kv_pack_v = nullptr;
    float* d_dk_pack = nullptr;
    float* d_dv_pack = nullptr;
    float* d_dq_pack = nullptr;
    float* attn_weights = nullptr;
};

struct SavedLayerStateLayoutV2 {
    float* x_res1 = nullptr;
    float* ln1_norm = nullptr;
    float* ln1_std_inv = nullptr;
    float* ln1_out = nullptr;
    float* x_res2 = nullptr;
    float* ln2_norm = nullptr;
    float* ln2_std_inv = nullptr;
    float* ln2_out = nullptr;
    float* ff_h = nullptr;
    float* ff_mask = nullptr;
    float* attn_out = nullptr;
    float* q = nullptr;
    float* k = nullptr;
    float* v = nullptr;
    float* attn_weights = nullptr;
};

struct ChunkBufferLayoutV2 {
    QueryScratchLayoutV2 query;
    OutputHeadLayoutV2 output;
    PackedAttentionLayoutV2 packed;
};

static inline ChunkBufferLayoutV2 make_chunk_buffer_layout_v2(const ChunkRuntimeV2& runtime,
                                                              int T_q,
                                                              int D,
                                                              int F,
                                                              int V) {
    ChunkBufferLayoutV2 layout;
    long long block = (long long)T_q * D;
    layout.query.x = runtime.d_query_state;
    layout.query.ln_out = layout.query.x + block;
    layout.query.q = layout.query.ln_out + block;
    layout.query.k = layout.query.q + block;
    layout.query.v = layout.query.k + block;
    layout.query.attn_out = layout.query.v + block;
    layout.query.ff_h = layout.query.attn_out + block;
    layout.query.ff_out = layout.query.ff_h + (long long)T_q * F;

    layout.output.final_out = runtime.d_logits_and_loss;
    layout.output.final_norm = layout.output.final_out + block;
    layout.output.final_std_inv = layout.output.final_norm + block;
    layout.output.logits = layout.output.final_std_inv + T_q;
    layout.output.d_logits = layout.output.logits + (long long)T_q * V;
    layout.output.loss = layout.output.d_logits + (long long)T_q * V;

    layout.packed.kv_pack_k = runtime.d_packed_attention;
    layout.packed.kv_pack_v = layout.packed.kv_pack_k + (long long)runtime.contract.kv_capacity * D;
    layout.packed.d_dk_pack = layout.packed.kv_pack_v + (long long)runtime.contract.kv_capacity * D;
    layout.packed.d_dv_pack = layout.packed.d_dk_pack + (long long)runtime.contract.kv_capacity * D;
    layout.packed.d_dq_pack = layout.packed.d_dv_pack + (long long)runtime.contract.kv_capacity * D;
    layout.packed.attn_weights =
        runtime.d_packed_attention +
        (long long)runtime.contract.kv_capacity * 4 * D +
        (long long)runtime.contract.query_capacity * 2 * D;
    return layout;
}

static inline SavedLayerStateLayoutV2 make_layer_saved_state_v2(const ChunkRuntimeV2& runtime,
                                                                int layer,
                                                                int T_q,
                                                                int D,
                                                                int F,
                                                                int H,
                                                                int max_kv_len) {
    SavedLayerStateLayoutV2 s;
    long long blockD = (long long)T_q * D;
    long long blockF = (long long)T_q * F;
    long long blockStd = (long long)T_q;
    long long blockAttn = (long long)T_q * H * max_kv_len;
    long long per_layer =
        10LL * blockD +
        2LL * blockF +
        2LL * blockStd +
        blockAttn;
    float* base = runtime.d_saved_activations + (long long)layer * per_layer;
    s.x_res1 = base;                   base += blockD;
    s.ln1_norm = base;                 base += blockD;
    s.ln1_std_inv = base;              base += blockStd;
    s.ln1_out = base;                  base += blockD;
    s.x_res2 = base;                   base += blockD;
    s.ln2_norm = base;                 base += blockD;
    s.ln2_std_inv = base;              base += blockStd;
    s.ln2_out = base;                  base += blockD;
    s.ff_h = base;                     base += blockF;
    s.ff_mask = base;                  base += blockF;
    s.attn_out = base;                 base += blockD;
    s.q = base;                        base += blockD;
    s.k = base;                        base += blockD;
    s.v = base;                        base += blockD;
    s.attn_weights = base;
    return s;
}

}  // namespace agpt_v2

#endif
