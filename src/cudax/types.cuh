#ifndef AGPT_V2_TYPES_CUH
#define AGPT_V2_TYPES_CUH

namespace agpt_v2 {

enum class LrSchedule {
    Constant = 0,
    WarmupCosine = 1,
};

struct TrainerConfig {
    int d_model = 0;
    int n_heads = 0;
    int n_layers = 0;
    int d_ff = 0;
    int vocab_size = 0;
    int seq_len = 0;

    int epochs = 0;
    int partition_depth = 1;
    int chunk_queries = 0;

    float lr = 0.0f;
    float grad_clip_norm = 0.0f;
    LrSchedule lr_schedule = LrSchedule::Constant;
    int warmup_epochs = 0;

    bool anc_grad = false;
    bool accumulate = false;
    bool quiet = false;
};

struct RuntimeShape {
    int d_model = 0;
    int n_heads = 0;
    int head_dim = 0;
    int n_layers = 0;
    int d_ff = 0;
    int vocab_size = 0;
    int seq_len = 0;
};

struct ModelHeader {
    RuntimeShape shape;
};

struct TrainerStatus {
    bool ok = true;
    const char* message = "uninitialized";
};

}  // namespace agpt_v2

#endif
