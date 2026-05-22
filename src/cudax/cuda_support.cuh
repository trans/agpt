#ifndef AGPT_V2_CUDA_SUPPORT_CUH
#define AGPT_V2_CUDA_SUPPORT_CUH

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#include "../common/cublas_algo.h"

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

#endif
