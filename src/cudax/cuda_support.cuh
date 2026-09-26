#ifndef AGPT_V2_CUDA_SUPPORT_CUH
#define AGPT_V2_CUDA_SUPPORT_CUH

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#include "../common/cublas_algo.h"

// Element type of the ancestor K/V cache. bf16 by default; build with
// -DAGPT_KV_CACHE_FP32 for an fp32 cache (gradient-parity diagnostics,
// rnd/gradient-population Exp 7). The byte size lives in runtime_contracts.cuh.
#ifdef AGPT_KV_CACHE_FP32
typedef float agpt_kv_t;
__device__ __forceinline__ agpt_kv_t agpt_kv_from_float(float x) { return x; }
__device__ __forceinline__ float agpt_kv_to_float(agpt_kv_t x) { return x; }
#else
typedef __nv_bfloat16 agpt_kv_t;
__device__ __forceinline__ agpt_kv_t agpt_kv_from_float(float x) { return __float2bfloat16(x); }
__device__ __forceinline__ float agpt_kv_to_float(agpt_kv_t x) { return __bfloat162float(x); }
#endif

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
