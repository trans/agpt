---
title: Exact-build speed
kind: diagnostic
status: concluded
outcome: positive
question: >-
  How much slower does v2 train in full precision (fp32 K/V cache, no TF32 matmuls), the
  build whose gradients match float64 to rounding, than in its normal build (bf16 K/V cache,
  TF32 matmuls)?
answer: >-
  Not measurably, at the d64/L2 model on depth-16 Shakespeare. Full precision takes 29.3 s
  against 28.3 s for 4 passes at pd=1 (+3.7%) and 54.1 s against 54.4 s for a pass at pd=3
  (-0.6%), inside the 1-3 s spread between repeats of the same setting. The two halves (fp32
  cache alone, fp32 matmuls alone) are within the same spread. Scores don't separate from run
  to run either at two repeats. At this size the GPU's time goes to the custom fp32 kernels,
  not the cuBLAS matmuls or the cache, so neither precision choice buys speed here. Full
  precision is v2's default build since fe3a4c8, checked to match the timed full-precision
  setting to rounding.
opened: 2026-11-18
updated: 2026-11-19
code: main
eval: canonical
family: attention
headline:
- label: 'AGPT pd=3 + crown, 1 pass, full precision (fp32 cache, NVIDIA_TF32_OVERRIDE=0), repeat 1'
  metric: train_wall_seconds
  value: 54.4
  run: 20261118T232218-pd3-fp32-fp32-r1
- label: 'Same, the normal build (bf16 cache, TF32), repeat 1'
  metric: train_wall_seconds
  value: 54.9
  run: 20261118T231810-pd3-bf16-tf32-r1
tags: [precision, performance, trainer, v2]
related: [one-pass-gap, gradient-population, partition-depth-rerun]
---

# Exact-build speed

`rnd/one-pass-gap` follow-up 5 showed v2's gradients match a float64
PyTorch transformer to rounding in full precision, and that the normal
build's bf16 K/V cache and TF32 matmuls put rounding errors of 1–4% into a
typical unit's gradient. Making full precision the default needed its speed
measured first. The user, 2026-11-18: "want to do that".

## Setup

- **Two of `rnd/one-pass-gap`'s runs**, unchanged but for their names
  (`make_configs.py`): `agpt-e4-lr0.012` (AGPT pd=1, 4 passes, Adam 0.012)
  and `pd3cs-e1-lr0.0015` (pd=3 + crown units, shuffled, 1 pass, the 7.81).
  The d64/L2 model, its depth-16 Shakespeare trie, the canonical split,
  seed 42, the exact ancestor gradient (`anc_grad_exact`).
- **Four precision settings**, each run twice:

  | setting | K/V cache | matmuls | `--trainer` | env |
  |---|---|---|---|---|
  | bf16-tf32 (normal build) | bf16 | TF32 | `v2` | – |
  | bf16-fp32 | bf16 | fp32 | `v2` | `NVIDIA_TF32_OVERRIDE=0` |
  | fp32-tf32 | fp32 | TF32 | `v2-kvfp32` | – |
  | fp32-fp32 (full precision) | fp32 | fp32 | `v2-kvfp32` | `NVIDIA_TF32_OVERRIDE=0` |

  `v2-kvfp32` runs `bin/agpt_train_v2_kvfp32` (`just
  build-agpt-train-v2-kvfp32`, `-DAGPT_KV_CACHE_FP32`), and the runner now
  records `NVIDIA_TF32_OVERRIDE` in `meta.json` (`18284a0`).
  `NVIDIA_TF32_OVERRIDE=0` is follow-up 5's full-precision setting.
  Since the switch (below), `--trainer v2-kvfp32` and its recipe are
  gone: check out `18c104c` to rerun the grid as recorded.
- **Order** (`run.sh`, slot 97, 2026-11-18, 23:14–23:32 UTC, log
  `grid.out`): all of repeat 1, then repeat 2, so drift in the GPU's clocks
  falls on every setting. Binaries: `agpt_train_v2` a535286c…,
  `agpt_train_v2_kvfp32` 81ef94f7… (sha256 in `grid.out`). Twelve runs
  record `git_sha` eff0ed6, the first four 18284a0: eff0ed6 landed during
  the grid and changed only `rnd/refined-trie-counts`'s README.
- **Timing** is `result.json`'s `train_wall_seconds`: the trainer's
  process, trie loading (about 1 s) included.

## Results

`python3 rnd/exact-build-speed/summarize.py`. Training seconds (repeat 1,
repeat 2; mean, against the normal build), rolling byte PPL (canonical,
the 10 Shakespeare held-out chunks) and fixed-window PPL:

**pd=1, 4 passes**

| setting | training seconds | rolling | fixed |
|---|---|---|---|
| bf16-tf32 (normal) | 28.9, 27.7; 28.3 | 10.493, 10.559 | 10.273, 10.371 |
| bf16-fp32 | 31.9, 28.5; 30.2 (+6.6%) | 10.490, 10.517 | 10.248, 10.243 |
| fp32-tf32 | 29.5, 27.8; 28.6 (+1.1%) | 10.378, 10.519 | 10.119, 10.256 |
| fp32-fp32 (full) | 30.3, 28.4; 29.3 (+3.7%) | 10.578, 10.601 | 10.306, 10.401 |

**pd=3 + crown, 1 pass**

| setting | training seconds | rolling | fixed |
|---|---|---|---|
| bf16-tf32 (normal) | 54.9, 54.0; 54.4 | 7.807, 7.880 | 7.180, 7.271 |
| bf16-fp32 | 58.4, 55.2; 56.8 (+4.3%) | 7.844, 7.767 | 7.233, 7.166 |
| fp32-tf32 | 55.4, 54.3; 54.8 (+0.7%) | 7.930, 7.847 | 7.348, 7.240 |
| fp32-fp32 (full) | 54.4, 53.9; 54.1 (−0.6%) | 7.762, 7.833 | 7.119, 7.231 |

The normal build reproduces `rnd/one-pass-gap`: 27.1 s and 10.510 there
for pd=1, 53.3 s and 7.813 for pd=3. The exact ancestor pass, which the
trainer times, takes 1.17–1.19 s a pass in the normal build and 1.05–1.06 s
in full precision (pd=1, repeat 1).

## Reading

- **Full precision costs nothing measurable here.** +3.7% at pd=1 and
  −0.6% at pd=3, where repeats of one setting differ by 1–3 s (repeat 1's
  first run, 28.9 s, is the slowest of its setting). The slowest setting in
  both cases is bf16-fp32 (+4–7%), the mix nobody would use, with the
  largest spread between its repeats too.
- **Why, probably:** an Nsight Systems profile of v2 at pd=1 on this model
  and trie (2026-09-28, not in the repo) put the cuBLAS matmuls, the only
  part TF32 touches, at about 12% of GPU time, and their matrices are small
  (width 64). The tree-attention and elementwise kernels, about two-thirds
  of the time, are scalar fp32 in every build. Doubling the cache's width
  didn't slow the runs, so reading it is not what limits them. Peak memory
  was not measured.
- **Scores don't separate at two repeats.** Repeats of one setting differ by
  up to 0.14 PPL at pd=1 and 0.08 at pd=3 (GPU atomics make runs
  nondeterministic). Full precision is the highest of the four at pd=1 in
  both repeats (10.578 and 10.601, 0.006 nats above the normal build's
  mean), and the best and the middle at pd=3. Follow-up 5 found the same
  for one pass (7.80 against 7.81).
- **This holds at this size.** At a wider model the matmuls take a larger
  share and TF32 would start to matter, and a deep trie's cache would make
  its width matter for memory. Measure again before scaling up.

## The switch to full precision (2026-11-19)

The user chose full precision as the default (2026-11-18: "I agree,
fp32"). Since `fe3a4c8`, `just build-agpt-train-v2` builds an fp32 K/V
cache and runs cuBLAS at `CUBLAS_DEFAULT_MATH`, with no env var needed.
`just build-agpt-train-v2-reduced` (`-DAGPT_V2_REDUCED_PRECISION`,
`bin/agpt_train_v2_reduced`, `--trainer v2-reduced`) builds the old
default, bf16 cache and TF32. `AGPT_V2_CUBLAS_MATH=fp32|tf32` still sets
the math mode in either build. v1 still uses TF32 (`AGPT_V1_DISABLE_TF32=1`
turns it off), so match precision before comparing v1 with v2.

`switch_check.sh` (slot 103, 2026-11-19, 02:10–02:18 UTC, log
`switch_check.out`) checked the new builds against the two binaries the
grid timed, kept aside before rebuilding. New binaries: `agpt_train_v2`
3abef619…, `agpt_train_v2_reduced` 3607b24a… (sha256 in the log).

**Gradients at fixed weights.** Frozen-weight dumps
(`AGPT_GRAD_DUMP_DIR`: one pd=1 pass, 65 units, no step taken) at
pd1-fp32-fp32-r1's checkpoint, compared by `compare_dumps.py`
(`switch_dumps.json`). Relative error of the first against the second:

| pair | whole pass | unit median | unit max |
|---|---|---|---|
| new v2, one dump against another | 1.1e-7 | 2.6e-7 | 3.6e-7 |
| new v2 against the old fp32 cache with `NVIDIA_TF32_OVERRIDE=0` | 1.2e-7 | 2.5e-7 | 3.7e-7 |
| reduced, one dump against another | 2.6e-7 | 6.8e-7 | 6.9e-6 |
| reduced against the old default | 2.3e-7 | 6.7e-7 | 6.9e-6 |
| reduced against new v2 | 3.1e-3 | 4.2e-3 | 1.9e-2 |

Each new build differs from the old one it replaces by no more than two
dumps of itself differ (GPU atomics), so the new default is the grid's
full-precision setting and the reduced build is the old default. Between
the two precisions, at this point, the whole pass differs by 0.3%, the
median unit by 0.4%, the worst by 1.9%. That is less than follow-up 5
found at its weight points (1–4% per unit; 8% on the whole pass, trained);
why it is less here was not looked into.

**Finite differences** (`src/tools/agpt_grad_check.py`, `switch_fd.out`)
on new v2 with no env var set, along the ancestor gradient: ratio 0.9998,
1.0000 and 1.0001 at steps of 1e-2, 3e-3 and 1e-3.

**Runs**, one of each case with each new build, no env var:

| run | build | training s | rolling | fixed |
|---|---|---|---|---|
| 20261119T021352-pd1-v2fp32-switch | `v2` (full) | 28.0 | 10.494 | 10.288 |
| 20261119T021444-pd1-v2reduced-switch | `v2-reduced` | 26.9 | 10.569 | 10.363 |
| 20261119T021531-pd3-v2fp32-switch | `v2` (full) | 53.3 | 7.785 | 7.183 |
| 20261119T021645-pd3-v2reduced-switch | `v2-reduced` | 53.0 | 7.871 | 7.241 |

Both builds ran about 1 s faster than their settings' grid means, and
each score is within the grid's spread between repeats (0.14 at pd=1,
0.08 at pd=3) of its setting's two grid runs. Over three runs each, full
precision averages 10.558 against 10.540 at pd=1 and 7.793 against 7.853
at pd=3: no consistent direction at this count. `summarize.py` leaves
these runs out.
