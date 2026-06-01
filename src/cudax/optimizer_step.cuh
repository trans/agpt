#ifndef AGPT_V2_OPTIMIZER_STEP_CUH
#define AGPT_V2_OPTIMIZER_STEP_CUH

#include "kernels_v2.cuh"
#include "types.cuh"

namespace agpt_v2 {

struct OptimizerStepResult {
    bool ok = true;
    const char* message = "optimizer step applied";
};

static inline OptimizerStepResult run_optimizer_step_sgd(float lr,
                                                         float* d_weights,
                                                         float* d_grads,
                                                         int total_floats) {
    OptimizerStepResult result;
    cuda_sgd_bulk(d_weights, d_grads, lr, total_floats);
    AGPT_V2_CUDA_CHECK(cudaDeviceSynchronize());
    result.message = "single SGD step applied";
    return result;
}

static inline OptimizerStepResult run_optimizer_step_sgd(const TrainerConfig& cfg,
                                                         float* d_weights,
                                                         float* d_grads,
                                                         int total_floats) {
    return run_optimizer_step_sgd(cfg.lr, d_weights, d_grads, total_floats);
}

static inline OptimizerStepResult run_optimizer_step_rmsprop(float lr,
                                                             float* d_weights,
                                                             float* d_grads,
                                                             float* d_opt_v,
                                                             int total_floats,
                                                             float beta = 0.999f,
                                                             float eps = 1e-8f) {
    OptimizerStepResult result;
    AGPT_V2_CUDA_CHECK(cudaMemset(d_opt_v, 0, (size_t)total_floats * sizeof(float)));
    cuda_rmsprop_bulk(d_weights, d_grads, d_opt_v, lr, beta, eps, total_floats);
    AGPT_V2_CUDA_CHECK(cudaDeviceSynchronize());
    result.message = "single RMSProp step applied";
    return result;
}

static inline OptimizerStepResult run_optimizer_step_rmsprop(const TrainerConfig& cfg,
                                                             float* d_weights,
                                                             float* d_grads,
                                                             float* d_opt_v,
                                                             int total_floats,
                                                             float beta = 0.999f,
                                                             float eps = 1e-8f) {
    return run_optimizer_step_rmsprop(cfg.lr, d_weights, d_grads, d_opt_v, total_floats, beta, eps);
}

static inline OptimizerStepResult run_optimizer_step_rmsprop_stateful(float lr,
                                                                      float* d_weights,
                                                                      float* d_grads,
                                                                      float* d_opt_v,
                                                                      int total_floats,
                                                                      float beta = 0.999f,
                                                                      float eps = 1e-8f) {
    OptimizerStepResult result;
    cuda_rmsprop_bulk(d_weights, d_grads, d_opt_v, lr, beta, eps, total_floats);
    AGPT_V2_CUDA_CHECK(cudaDeviceSynchronize());
    result.message = "stateful RMSProp step applied";
    return result;
}

static inline OptimizerStepResult run_optimizer_step_rmsprop_stateful(const TrainerConfig& cfg,
                                                                      float* d_weights,
                                                                      float* d_grads,
                                                                      float* d_opt_v,
                                                                      int total_floats,
                                                                      float beta = 0.999f,
                                                                      float eps = 1e-8f) {
    return run_optimizer_step_rmsprop_stateful(cfg.lr, d_weights, d_grads, d_opt_v, total_floats, beta, eps);
}

static inline OptimizerStepResult run_optimizer_step_momentum_stateful(float lr,
                                                                       float* d_weights,
                                                                       float* d_grads,
                                                                       float* d_opt_m,
                                                                       int total_floats,
                                                                       float beta = 0.9f) {
    OptimizerStepResult result;
    cuda_momentum_bulk(d_weights, d_grads, d_opt_m, lr, beta, total_floats);
    AGPT_V2_CUDA_CHECK(cudaDeviceSynchronize());
    result.message = "stateful Momentum step applied";
    return result;
}

static inline OptimizerStepResult run_optimizer_step_adam_stateful(float lr,
                                                                   float* d_weights,
                                                                   float* d_grads,
                                                                   float* d_opt_m,
                                                                   float* d_opt_v,
                                                                   int total_floats,
                                                                   int t,
                                                                   float beta1 = 0.9f,
                                                                   float beta2 = 0.999f,
                                                                   float eps = 1e-8f) {
    OptimizerStepResult result;
    cuda_adam_bulk(d_weights, d_grads, d_opt_m, d_opt_v, lr, beta1, beta2, eps, t, total_floats);
    AGPT_V2_CUDA_CHECK(cudaDeviceSynchronize());
    result.message = "stateful Adam step applied";
    return result;
}

static inline OptimizerStepResult run_optimizer_step_stateful(const TrainerConfig& cfg,
                                                              float lr,
                                                              float* d_weights,
                                                              float* d_grads,
                                                              float* d_opt_m,
                                                              float* d_opt_v,
                                                              int total_floats,
                                                              int optimizer_step_index) {
    OptimizerStepResult result;
    switch (cfg.optimizer) {
        case OptimizerKind::Adam:
            result = run_optimizer_step_adam_stateful(lr, d_weights, d_grads, d_opt_m, d_opt_v,
                                                      total_floats, optimizer_step_index,
                                                      cfg.momentum_beta, cfg.rmsprop_beta,
                                                      cfg.optimizer_eps);
            break;
        case OptimizerKind::SGD:
            result = run_optimizer_step_sgd(lr, d_weights, d_grads, total_floats);
            break;
        case OptimizerKind::Momentum:
            result = run_optimizer_step_momentum_stateful(lr, d_weights, d_grads, d_opt_m,
                                                          total_floats, cfg.momentum_beta);
            break;
        case OptimizerKind::RMSProp:
            result = run_optimizer_step_rmsprop_stateful(lr, d_weights, d_grads, d_opt_v,
                                                         total_floats, cfg.rmsprop_beta,
                                                         cfg.optimizer_eps);
            break;
        default:
            result.ok = false;
            result.message = "unknown optimizer";
            return result;
    }
    if (result.ok && cfg.weight_decay > 0.0f) {
        cuda_weight_decay(d_weights, lr, cfg.weight_decay, total_floats);
        AGPT_V2_CUDA_CHECK(cudaDeviceSynchronize());
        result.message = "optimizer step + decoupled weight decay applied";
    }
    return result;
}

}  // namespace agpt_v2

#endif
