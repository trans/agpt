#ifndef AGPT_V2_CUDA_SUPPORT_CUH
#define AGPT_V2_CUDA_SUPPORT_CUH

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#define AGPT_V2_CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        std::fprintf(stderr, "agpt_train_v2: CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                     cudaGetErrorString(err__)); \
        std::exit(1); \
    } \
} while(0)

#define AGPT_V2_CUBLAS_CHECK(call) do { \
    cublasStatus_t st__ = (call); \
    if (st__ != CUBLAS_STATUS_SUCCESS) { \
        std::fprintf(stderr, "agpt_train_v2: cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, (int)st__); \
        std::exit(1); \
    } \
} while(0)

enum class AgptV2GemmAlgoMode {
    Heuristic = 0,
    Algo0 = 1,
    Algo1 = 2,
};

static inline AgptV2GemmAlgoMode read_cublas_gemm_algo_mode_v2() {
    const char* env = std::getenv("AGPT_V2_CUBLAS_GEMM_ALGO");
    if (!env || !env[0]) return AgptV2GemmAlgoMode::Heuristic;
    if (std::strcmp(env, "algo0") == 0) return AgptV2GemmAlgoMode::Algo0;
    if (std::strcmp(env, "algo1") == 0) return AgptV2GemmAlgoMode::Algo1;
    return AgptV2GemmAlgoMode::Heuristic;
}

static inline cublasGemmAlgo_t resolve_cublas_gemm_algo_v2(cublasHandle_t handle,
                                                            AgptV2GemmAlgoMode mode) {
    cublasMath_t math_mode = CUBLAS_DEFAULT_MATH;
    (void)cublasGetMathMode(handle, &math_mode);
    const bool tensor_ops = (math_mode != CUBLAS_DEFAULT_MATH);

    if (mode == AgptV2GemmAlgoMode::Algo1) {
        return tensor_ops ? CUBLAS_GEMM_ALGO1_TENSOR_OP : CUBLAS_GEMM_ALGO1;
    }
    return tensor_ops ? CUBLAS_GEMM_ALGO0_TENSOR_OP : CUBLAS_GEMM_ALGO0;
}

static inline cublasStatus_t agpt_v2_cublas_sgemm(cublasHandle_t handle,
                                                   cublasOperation_t transa,
                                                   cublasOperation_t transb,
                                                   int m, int n, int k,
                                                   const float* alpha,
                                                   const float* A, int lda,
                                                   const float* B, int ldb,
                                                   const float* beta,
                                                   float* C, int ldc) {
    const AgptV2GemmAlgoMode mode = read_cublas_gemm_algo_mode_v2();
    if (mode == AgptV2GemmAlgoMode::Heuristic) {
        return cublasSgemm(handle, transa, transb, m, n, k,
                           alpha, A, lda, B, ldb, beta, C, ldc);
    }
    const cublasGemmAlgo_t algo = resolve_cublas_gemm_algo_v2(handle, mode);
    return cublasGemmEx(handle, transa, transb, m, n, k,
                        alpha,
                        A, CUDA_R_32F, lda,
                        B, CUDA_R_32F, ldb,
                        beta,
                        C, CUDA_R_32F, ldc,
                        CUBLAS_COMPUTE_32F, algo);
}

#ifdef cublasSgemm
#undef cublasSgemm
#endif
#define cublasSgemm agpt_v2_cublas_sgemm

#endif
