#ifndef AGPT_V2_TYPES_CUH
#define AGPT_V2_TYPES_CUH

namespace agpt_v2 {

enum class LrSchedule {
    Constant = 0,
    WarmupCosine = 1,
};

enum class OptimizerKind {
    Adam = 0,
    SGD = 1,
    Momentum = 2,
    RMSProp = 3,
};

enum class RopePositionModeV2 {
    Depth = 0,
    SampledBin = 1,
    PhaseSweep = 2,
    PhaseWeighted = 3,
    PhaseConditioned = 4,
};

enum class MassWeightModeV2 {
    Off = 0,
    Linear = 1,
    Sqrt = 2,
    Log = 3,
    InvLog = 4,
    InvLinear = 5,
};

struct TrainerConfig {
    int d_model = 0;
    int n_heads = 0;
    int n_layers = 0;
    int d_ff = 0;
    int vocab_size = 0;
    int seq_len = 0;
    int rope_seq_len = 0;

    int epochs = 0;
    int partition_depth = 1;
    int chunk_queries = 0;

    float lr = 0.0f;
    float grad_clip_norm = 0.0f;
    float momentum_beta = 0.9f;
    float rmsprop_beta = 0.999f;
    float optimizer_eps = 1e-8f;
    float weight_decay = 0.0f;
    LrSchedule lr_schedule = LrSchedule::Constant;
    OptimizerKind optimizer = OptimizerKind::RMSProp;
    int warmup_epochs = 0;

    bool anc_grad = false;
    bool accumulate = false;
    bool quiet = false;
    RopePositionModeV2 rope_position_mode = RopePositionModeV2::Depth;
    MassWeightModeV2 mass_weight = MassWeightModeV2::Linear;
    unsigned pos_sample_seed = 1;
    int rope_position_offset = -1;
    bool rope_phase_shuffle = false;
    unsigned rope_phase_shuffle_seed = 1;
    int loss_depth_min = -1;
    int loss_depth_max = -1;
};

struct RuntimeShape {
    int d_model = 0;
    int n_heads = 0;
    int head_dim = 0;
    int n_layers = 0;
    int d_ff = 0;
    int vocab_size = 0;
    int seq_len = 0;
    int rope_seq_len = 0;
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
