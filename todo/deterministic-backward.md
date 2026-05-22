# Deterministic backward (atomicAdd-free)

## Status

Open. Forward determinism shipped 2026-05-22 (ef3a8e9, shared
build/kernels_gpu.o + no --use_fast_math). Backward is still
non-deterministic by construction. This blocks fine-grained
parity work (v1 vs v2 algorithmic comparison, before/after fix
comparison at the bit level, etc.).

## What's non-deterministic

`grep -n "atomicAdd" src/cuda/kernels.cu` and src/cuda/agpt_train.cu:

  src/cuda/kernels.cu:
    line ~310-311   LayerNorm dgamma, dbeta accumulation
    line ~611       token embedding gradient (multiple positions →
                    same token, atomicAdd into d_token_emb)
    line ~682       loss_out reduction
    line ~1429+     attention backward dk[p,d], dv[p,d] (shared K
                    across queries scatter-add)

  src/cuda/agpt_train.cu:
    line ~2598      anc-grad dkv_subtree scatter-add
    line ~2821      global_dkv ancestor scatter-add

Each atomicAdd produces different bit-level results depending on
thread scheduling. Same input + same kernel → different output
across runs.

## Empirical noise floor

Measured today on v1 against itself, same input file, probe envs
(TF32 off, algo0 pinned, CUBLAS_WORKSPACE_CONFIG=:4096:8,
CUDA_LAUNCH_BLOCKING=1): single seed, 10 SE Shakespeare 1M d=16,
two consecutive runs:
  run A PPL: 9.1138
  run B PPL: 9.5709
  drift: 0.36 PPL

That's the "v1 self-noise" we have to clear before any v1-vs-v2 or
A-vs-B parity question is sharper than ~0.5 PPL.

## Why we don't have to fix it today

For routine research (running an experiment, comparing to baseline
across multiple seeds), the noise floor is fine. Distribution-level
comparison absorbs the atomicAdd variance.

The fix matters when we want POINTWISE parity:
  - "did this code change produce bit-identical output?"
  - "does v1 match v2 algorithmically at this specific configuration?"
  - "is this gradient computation correct down to fp rounding?"

## Fix outline

Replace atomicAdd-based scatter with deterministic reductions.
Options:

1. **Per-thread accumulation + tree reduction**: each thread
   computes its contribution to a small buffer, then a structured
   reduction tree combines them. No atomics. Slower per kernel
   launch (extra memory traffic) but bit-deterministic.

2. **Sort-and-segmented-reduce**: sort gradient contributions by
   destination index, then segmented reduce. Reproducible but
   adds a sort step.

3. **Replicate-and-sum**: write into per-thread or per-warp slots
   in shared memory, then sum across the block. Bit-deterministic
   within a block. Needs care for cross-block aggregation.

Pick per-kernel based on memory pressure and access pattern.
LayerNorm dgamma/dbeta is the easiest (per-element accumulation
across rows; replace with a separate reduction kernel). Attention
backward dk/dv is the hardest (sparse scatter pattern).

## Estimated effort

Half day to a day for the kernels in agpt_train.cu (anc-grad,
global dkv). Possibly more for the kernels.cu functions if we
want to keep the same API.

## Pick up when

- v1-vs-v2 algorithmic parity becomes an open research question
  again (e.g., when one of them is being deprecated, or when a
  cross-trainer audit is needed)
- A subtle gradient bug surfaces that we want to bit-trace
- Future code changes need before/after bit-comparison
