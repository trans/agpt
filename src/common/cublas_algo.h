#ifndef AGPT_COMMON_CUBLAS_ALGO_H
#define AGPT_COMMON_CUBLAS_ALGO_H

// Shared cuBLAS-algo override for v1 (src/cuda/agpt_train.cu) and v2
// (src/cudax/agpt_train_v2.cu) determinism probes.
//
// Env var AGPT_CUBLAS_GEMM_ALGO ∈ {algo0, algo1} forces every Sgemm
// call to route through cublasGemmEx with an explicit
// CUBLAS_GEMM_ALGO[0|1] (or _TENSOR_OP variant when TF32 is on).
// Unset → real cublasSgemm with the heuristic (current default).
// Same env var name for both binaries — runtime decides which binary
// is running; there's no need for V1/V2 prefixes since the var only
// affects whichever binary is invoked.
//
// Usage in a .cu file:
//     #include "../common/cublas_algo.h"
//     // ... after the include, all cublasSgemm() calls in this TU
//     // are redirected through agpt_cublas::sgemm.
//
// IMPORTANT: the `#define cublasSgemm agpt_cublas::sgemm` at the end
// of this header MUST come after the wrapper body so the wrapper's
// own internal cublasSgemm call resolves to the real cuBLAS function,
// not recursively to itself. Includer's call sites get redirected.

#include <cstdlib>
#include <cstring>
#include <cublas_v2.h>

namespace agpt_cublas {

enum class GemmAlgoMode { Heuristic = 0, Algo0 = 1, Algo1 = 2 };

inline GemmAlgoMode read_mode() {
    const char* env = std::getenv("AGPT_CUBLAS_GEMM_ALGO");
    if (!env || !env[0]) return GemmAlgoMode::Heuristic;
    if (std::strcmp(env, "algo0") == 0) return GemmAlgoMode::Algo0;
    if (std::strcmp(env, "algo1") == 0) return GemmAlgoMode::Algo1;
    return GemmAlgoMode::Heuristic;
}

inline cublasGemmAlgo_t resolve_algo(cublasHandle_t handle, GemmAlgoMode mode) {
    cublasMath_t math_mode = CUBLAS_DEFAULT_MATH;
    (void)cublasGetMathMode(handle, &math_mode);
    const bool tensor_ops = (math_mode != CUBLAS_DEFAULT_MATH);
    if (mode == GemmAlgoMode::Algo1) {
        return tensor_ops ? CUBLAS_GEMM_ALGO1_TENSOR_OP : CUBLAS_GEMM_ALGO1;
    }
    return tensor_ops ? CUBLAS_GEMM_ALGO0_TENSOR_OP : CUBLAS_GEMM_ALGO0;
}

inline cublasStatus_t sgemm(cublasHandle_t handle,
                            cublasOperation_t transa,
                            cublasOperation_t transb,
                            int m, int n, int k,
                            const float* alpha,
                            const float* A, int lda,
                            const float* B, int ldb,
                            const float* beta,
                            float* C, int ldc) {
    const GemmAlgoMode mode = read_mode();
    if (mode == GemmAlgoMode::Heuristic) {
        return cublasSgemm(handle, transa, transb, m, n, k,
                           alpha, A, lda, B, ldb, beta, C, ldc);
    }
    const cublasGemmAlgo_t algo = resolve_algo(handle, mode);
    return cublasGemmEx(handle, transa, transb, m, n, k,
                        alpha,
                        A, CUDA_R_32F, lda,
                        B, CUDA_R_32F, ldb,
                        beta,
                        C, CUDA_R_32F, ldc,
                        CUBLAS_COMPUTE_32F, algo);
}

}  // namespace agpt_cublas

#ifdef cublasSgemm
#undef cublasSgemm
#endif
#define cublasSgemm agpt_cublas::sgemm

#endif
