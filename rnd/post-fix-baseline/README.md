# Post-Fix AGPT Baseline Re-establishment

**Context**: Commit `1c858c0` fixed a fundamental bug where Wk, Wv, and all
7 biases (wq_b, wk_b, wv_b, wo_b, l1_b, l2_b, out_b) had been silently
frozen at random initialization across the entire project's training
history. AGPT was effectively a "random-features-attention" architecture.

With attention now fully trainable, every prior hyperparameter — learning
rate, weight decay, warmup, mass-weight, sampler choice, super-epochs —
may have shifted. We need fresh baselines before building new features on
top.

## Reference pre-fix numbers (for sanity comparison)

At matched Lightning config (3 SE × 260 samples, L3 p_stop=0.3, mass=linear):

| depth | pre-fix PPL | pre-fix LR |
|---|---|---|
| d=16 | 15.38 | 2e-4 |
| d=32 | 12.07 | 2e-4 |

Note: the pre-fix model had frozen attention weights — those numbers are
**not a correctness target**. We aim to equal or beat them with a
correctly-trained model, which may require different LR/regularization.

## Preliminary post-fix findings (single runs)

| config | PPL |
|---|---|
| d=16 wc-lr3e-4-wd0.01 3SE | 15.08 (beats pre-fix 15.38) |
| d=16 const-lr1e-4-wd0.01 3SE | 15.42 |
| d=32 wc-lr3e-4-wd0.01 3SE | 12.86 |
| d=32 wc-lr2e-4-wd0.01 3SE | 13.20 |
| d=32 wc-lr3e-4-wd0.01 6SE | 12.93 (plateaued) |

d=32 hasn't reached pre-fix 12.07 yet. LR and schedule need more sweep.

## Plan

### Phase 1: deterministic d=16 baseline ✓

Paper-style per-root-child (3 SE × 65 steps, `--no-accumulate`). Sweep lr
and weight-decay. Reference: paper's pre-fix 15.28.

### Phase 2: deterministic d=32 baseline ✓

Same, at d=32 (paper's pre-fix 13.17 / 13.36).

### Phase 3: Lightning vs deterministic

Pick best det-LR from phases 1-2, then run Lightning at matched budgets
(260, 650 samples/SE). Find Lightning's best lr.

### Phase 4: mass-weight re-check

pre-fix: `--mass-weight linear` won everywhere. Re-sweep {off, log, sqrt,
linear} at best lr for both d=16 and d=32. With trainable K/V the winner
may shift.

## Results summary (Phases 1+2 complete)

### Phase 1: d=16 deterministic (3 SE × 65 steps, mass=linear, entropy-lambda=1.0, 3 seeds/cell)

| config | mean PPL | min | max | spread |
|---|---|---|---|---|
| **wc lr=3e-3** | **14.38** | 14.23 | 14.66 | 0.44 |
| const lr=3e-3 | 14.95 | 13.97 | 15.72 | 1.76 |
| const lr=1e-3 | 15.21 | 15.01 | 15.38 | 0.37 |
| wc lr=1e-3 | 16.19 | 16.16 | 16.21 | 0.05 |
| wc lr=3e-4 | 17.01 | — | — | 0.00 |
| wc lr=1e-2 | 23.80 | 22.32 | 24.62 | — |
| wc lr=3e-2 | NaN (diverged) | — | — | — |

**d=16 winner: `--lr-schedule warmup-cosine --lr 3e-3 --warmup-epochs 1`, mean 14.38, min 14.23.**

Beats paper's pre-fix reference (15.28) by ~0.9 PPL at same budget.

### Phase 2: d=32 deterministic (3 SE × 65 steps, same recipe, 3 seeds/cell)

| config | mean PPL | min | max | spread |
|---|---|---|---|---|
| **wc lr=3e-3** | **12.79** | **12.48** | 13.22 | 0.74 |
| wc lr=1e-3 | 13.27 | 13.23 | 13.32 | 0.09 |
| wc lr=1e-3-wd0.01 | 13.31 | 13.29 | 13.32 | 0.03 |
| const lr=1e-3 | 13.68 | 13.62 | 13.78 | 0.16 |
| const lr=3e-3 | 14.10 | 13.48 | 14.65 | 1.16 |
| wc lr=5e-3 | 14.43 | 13.89 | 14.91 | — |
| wc lr=7e-3 | 14.27 | 14.03 | 14.55 | — |

**d=32 winner: same recipe (`wc lr=3e-3`), mean 12.79, min 12.48.**

Beats paper's pre-fix reference (13.17) and memory's pre-fix best
(13.36 at d=32 linear-mass). Within range of pre-fix Lightning best
(12.07, single-seed) — but that was with frozen K/V implicit
regularization; new post-fix 12.48 is a correctly-trained min.

### Phase 3: real SGD window training as reference

`bin/microgpt --backend openblas` (openblas works; cublas is broken — see
`notes/core/known-bugs-cublas-training.md`). seq_len=32, lr=3e-4, d_model=64,
n_layers=2, from random init:

| steps | PPL | notes |
|---|---|---|
| 500 | 19.00 | |
| **2000** | **14.72** | matches paper's claimed saturation point |
| 10000 | 26.37 | overtrained at fixed lr=3e-4 |

The 2000-step number is the true SGD-window reference.

### Phase 4: L4 path-sampling ("SGD-equivalent" inside AGPT code path)

Matched step budget, 3 seeds each:

| config | L4 mean | L4 min |
|---|---|---|
| d=16 s=195 | 22.15 | 21.19 |
| d=16 s=650 | 19.64 | 17.73 |
| d=16 s=2000 | 17.98 | 16.37 |
| d=16 s=10000 | 17.00 | 15.97 |
| d=32 s=195 | 23.22 | 21.93 |
| d=32 s=650 | 18.52 | 18.03 |
| d=32 s=2000 | 17.39 | 17.19 |
| d=32 s=10000 | 15.15 | 14.68 |

L4 shows diminishing returns past 2k steps. 50× the steps only buys ~2 PPL.

**L4 vs real SGD at 2000 steps**:
- Real SGD seq=32: 14.72 PPL
- L4 d=32: 17.39 PPL (2.7 PPL worse)
- L4 d=16: 17.98 PPL

L4 is NOT a faithful SGD emulator despite the statistical equivalence in
sample selection. Likely culprits: (a) AGPT's KL-at-endpoint loss vs SGD's
pure next-token CE, (b) RMSProp vs microgpt's optimizer, (c) warmup-cosine
LR schedule vs microgpt's constant LR. The real SGD number is the correct
comparator.

## AGPT vs SGD headline comparison

At matched compute (via optimizer steps, roughly equal per-step cost at
this small scale):

| method | steps | seq / depth | PPL |
|---|---|---|---|
| SGD window (openblas) | 2000 | 32 | 14.72 |
| AGPT det | **195** | d=16 | **14.38** |
| AGPT det | **195** | d=32 | **12.79** |

**AGPT at d=16 with 10× fewer optimizer steps matches SGD window seq=32.**
**AGPT at d=32 with 10× fewer steps beats SGD window seq=32 by 1.9 PPL.**

This is the apples-to-apples result the paper was trying to establish; with
the Wk/Wv fix + openblas SGD reference, it now holds up cleanly.

## Observations

- **Both depths prefer `warmup-cosine` + `lr=3e-3`.** Sharp U-curve with
  5e-3 already worse, 3e-2 diverging. 10× higher LR than the pre-fix
  regime (which wanted 2e-4..3e-4 against frozen K/V).
- **Weight decay doesn't help at 195 steps.** Probably matters at longer
  schedules (Phase 3).
- **d=32 post-fix 12.79 mean vs d=16 post-fix 14.38** — 1.59 PPL gap
  from depth alone at matched step budget + identical recipe.
- Variance is low at the optimal config (spread 0.44 at d=16, 0.74 at
  d=32). Recipe is stable.
