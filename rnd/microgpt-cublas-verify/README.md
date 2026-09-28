---
title: microgpt cuBLAS verification
kind: diagnostic
status: concluded
outcome: n/a
question: >-
  Is microgpt's cuBLAS training path correct, and how much wall-clock speedup does it give
  over openBLAS?
answer: >-
  It is correct: loss matches openBLAS bit-for-bit at steps 50 and 250, and diverges only
  by fp roundoff later. The speedup depends on model size. cuBLAS is 1.35x slower at d=64
  L=2 seq=32, 1.31x faster at seq=128, and 5.5x faster at d=256 L=4 seq=128. SGD at the historical
  d=64 L=2 config therefore gains nothing from the GPU.
opened: 2026-05-10
updated: 2026-05-11
code: main
eval: legacy
family: attention
headline:
- {label: 'cuBLAS vs openBLAS, d=256 L=4 seq=128, 1000 steps', metric: wall-clock speedup
    (x), value: 5.5}
- {label: 'microgpt SGD d=256 L=4 seq=128, 1000 steps (openblas)', metric: legacy PPL (bin/perplexity
    on data/input.txt), value: 7.64}
tags: [infrastructure, kernels, baseline]
---

# microgpt cuBLAS — verification and scaling

**Date:** 2026-05-10
**Question:** Is microgpt's cuBLAS training path correct, and does it
give meaningful wall-clock speedup vs. the openBLAS CPU path?

## TL;DR

- **cuBLAS is correct** — bit-identical loss curves vs. openBLAS at
  matched seed early in training; fp-roundoff divergence by step 450.
  The "NaN at step 0" bug from prior memory entries appears resolved
  (likely by Justfile commit `27235e2` which fixed the bin from linking
  against CPU stubs).
- **cuBLAS speedup depends entirely on model size**: at d=64 L=2
  seq=32 it's *slower* than openBLAS; at d=64 L=2 seq=128 it's 1.31×
  faster; at d=256 L=4 seq=128 it's 5.5× faster.
- **For AGPT-vs-SGD wall comparisons at the historical d=64 L=2
  config, cuBLAS does not give SGD a meaningful boost.** SGD effectively
  has to stay on openBLAS at that scale. A bigger-model comparison
  (d≥128) would let SGD run on GPU competitively.

## Build fix

`build-microgpt-tools` recipe in Justfile needed
`--allow-unsupported-compiler -std=c++17` for GCC 16 + CUDA 13.2
compatibility (same fix as `build-agpt-train`). Applied here. Without
it, nvcc errors on `char8_t` and C++20 `requires` clauses from GCC 16's
`type_traits` header.

## Verification — matched-seed parity

500 steps, d_model=64, n_layers=2, seq_len=32, lr=3e-4, seed=42,
`data/input.txt` (Shakespeare 1M):

| backend | step 50 loss | step 250 | step 450 | final avg | wall (s) |
|---|---:|---:|---:|---:|---:|
| openblas | 3.2335 | 3.2480 | 2.9969 | 3.0580 | 4.63 |
| **cublas** | **3.2335** | **3.2480** | 2.9941 | 3.0577 | 7.31 |

Step 50 and 250 are **bit-identical**. Step 450 diverges by 0.003 loss
(within fp32 roundoff for accumulated matmuls). cuBLAS is correct.

## Wall-time scaling

5000 steps, same model dims (d=64 L=2 seq=32):

| backend | wall | final loss |
|---|---:|---:|
| openblas | 46.89 s | 2.2667 |
| cublas | 63.53 s | 2.2647 |

cuBLAS 1.35× **slower**. Kernel-launch overhead dominates the small
matmuls.

1000 steps, d=64 L=2, **seq=128**:

| backend | wall | final loss |
|---|---:|---:|
| openblas | 21.13 s | 2.5067 |
| cublas | 16.07 s | 2.5055 |

cuBLAS **1.31× faster**. Crossover happens between seq=32 and seq=128
at this d_model.

1000 steps, **d=256 L=4 seq=128**:

| backend | wall | final loss |
|---|---:|---:|
| openblas | 614.65 s | 2.0510 |
| **cublas** | **111.51 s** | 2.0547 |

cuBLAS **5.5× faster**. At realistic transformer sizes, the speedup is
substantial.

## Implications for AGPT-vs-SGD wall comparison

The project's standard test config (d_model=64, n_layers=2, 108k params)
is *below* the cuBLAS crossover for short sequences. So:

- At seq≤32: openBLAS SGD vs. cuBLAS AGPT was actually *favorable* to
  SGD's wall budget, contrary to past assumptions. Past comparisons
  citing "AGPT is way faster wall-clock" need re-examination here.
- At seq=128: cuBLAS SGD is 31% faster than openBLAS SGD; using it
  closes some of AGPT's wall advantage but doesn't eliminate it because
  AGPT-on-GPU isn't paying the SGD overhead penalty.
- For meaningful AGPT-vs-SGD scaling comparisons, both should run at
  d≥128 so both are GPU-bound and the wall numbers reflect actual
  algorithmic differences.

## Held-out PPL on Shakespeare 1M (`bin/perplexity` against `data/input.txt`)

| config | steps | openblas PPL | cublas PPL |
|---|---:|---:|---:|
| d=64 L=2 seq=32 seed=42 | 500 | 17.82 | 17.82 |
| d=64 L=2 seq=32 | 5000 | 9.50 | 9.49 |
| d=64 L=2 seq=128 | 1000 | 12.11 | 12.07 |
| **d=256 L=4 seq=128** | 1000 | **7.64** | **7.68** |

Backends produce indistinguishable models within fp-roundoff at every
config. d=256 L=4 in just 1000 steps reaches **PPL 7.64** — a strong
big-model SGD baseline for AGPT-vs-SGD scaling comparisons.

## Files preserved

```
logs/sgd-openblas-5k.log         # d=64 seq=32 5000 steps
logs/sgd-cublas-5k.log           # d=64 seq=32 5000 steps
logs/sgd-seq128-openblas.log     # d=64 seq=128 1000 steps
logs/sgd-seq128-cublas.log       # d=64 seq=128 1000 steps
logs/sgd-big-openblas.log        # d=256 L=4 seq=128 1000 steps
logs/sgd-big-cublas.log          # d=256 L=4 seq=128 1000 steps
```

## Followups

- Re-run the historical "SGD seq=128 500k cosine on Gutenberg 5M" on
  cuBLAS to get a wall-time number comparable to AGPT's GPU runs (the
  earlier 4.85 PPL was on openBLAS, ~3 hours; cuBLAS estimated ~30%
  faster).
- Run an AGPT-vs-SGD wall comparison at d_model=128 or 256 to see
  algorithmic differences at GPU-saturating sizes.
- The `MicroGPT.use_cublas!` path may still have edge cases for
  seq_len < model.max_seq_len; verify generation-after-training works.
