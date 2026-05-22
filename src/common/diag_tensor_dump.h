#ifndef AGPT_COMMON_DIAG_TENSOR_DUMP_H
#define AGPT_COMMON_DIAG_TENSOR_DUMP_H

// Element-wise binary dump helpers for the v1↔v2 forward-parity probe.
// Both trainers use this header so the on-disk file format
// (.f32 / .i32, name = <point>_e<E>_r<R>_c<C>_l<L>) stays in lockstep —
// cmp/sha256sum/numpy.fromfile can compare dumps from either side
// without coordinating format.
//
// No dependency on v1's CUDA_CHECK or v2's AGPT_V2_CUDA_CHECK; uses
// cudaMemcpy directly so the header is includable from any .cu in
// either trainer.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

namespace agpt_diag {

// Float32 device-buffer dump. Silent no-op when dir is NULL or
// n_floats <= 0, so callers can guard with a single env-var check.
inline void emit_tensor_bin(const char* dir,
                            int epoch, int root_id, int chunk_idx, int layer,
                            const char* point,
                            const float* d_buf, int n_floats) {
    if (!dir || n_floats <= 0) return;
    float* scratch = (float*)std::malloc((size_t)n_floats * sizeof(float));
    if (!scratch) return;
    cudaError_t err = cudaMemcpy(scratch, d_buf,
                                 (size_t)n_floats * sizeof(float),
                                 cudaMemcpyDeviceToHost);
    if (err == cudaSuccess) {
        char fname[512];
        std::snprintf(fname, sizeof(fname), "%s/%s_e%d_r%d_c%d_l%d.f32",
                      dir, point, epoch, root_id, chunk_idx, layer);
        FILE* f = std::fopen(fname, "wb");
        if (f) {
            std::fwrite(scratch, sizeof(float), n_floats, f);
            std::fclose(f);
        }
    }
    std::free(scratch);
}

// Int32 device-buffer dump. Same convention as the float variant.
inline void emit_tensor_int_bin(const char* dir,
                                int epoch, int root_id, int chunk_idx, int layer,
                                const char* point,
                                const int* d_buf, int n_ints) {
    if (!dir || n_ints <= 0) return;
    int* scratch = (int*)std::malloc((size_t)n_ints * sizeof(int));
    if (!scratch) return;
    cudaError_t err = cudaMemcpy(scratch, d_buf,
                                 (size_t)n_ints * sizeof(int),
                                 cudaMemcpyDeviceToHost);
    if (err == cudaSuccess) {
        char fname[512];
        std::snprintf(fname, sizeof(fname), "%s/%s_e%d_r%d_c%d_l%d.i32",
                      dir, point, epoch, root_id, chunk_idx, layer);
        FILE* f = std::fopen(fname, "wb");
        if (f) {
            std::fwrite(scratch, sizeof(int), n_ints, f);
            std::fclose(f);
        }
    }
    std::free(scratch);
}

}  // namespace agpt_diag

#endif
