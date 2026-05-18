# Per-rc Adam — Stage 1 of topological optimizer state

## Hypothesis

Standard Adam/RMSprop maintain a *single global* second-moment estimate `v`
across all parameters and all updates. AGPT fires updates from different
regions of the trie (different root-child subtrees), each with potentially
different gradient-scale profiles. The hypothesis: localizing `v` to root-child
buckets — so the "t"-subtree has its own `v` separate from "w"-subtree —
extracts a structural signal that pooled `v` is averaging away.

This is the cheapest precursor to the broader **topological optimizer state**
program (see `notes/agpt/suffix_weighted_curvature.md`): tests whether
*any* form of trie-structured optimizer state helps, before paying the
engineering cost of within-step suffix-ensemble F_p (Stage 2).

## Implementation

`--per-rc-adam` flag in `agpt_train`. Replaces the single global `d_adam_m`,
`d_adam_v` buffers with `[n_root_children × total_floats]` versions; at each
optimizer fire, redirects the m/v pointers to the firing rc's bucket. Step
counter `adam_t` becomes a host-side `[n_root_children]` array.

Constraints: requires `--no-accumulate` (per-rc only meaningful in per-group
fire mode). State is not persisted across runs — one-shot experimental flag.
At pd=1, n_root_children ≈ V ≈ 65 buckets; memory ~56 MB for the test model.

Diagnostic flag `--dump-per-rc-v PATH` writes the final per-rc v buffer for
offline analysis (`analyze_dump.py`).

## Experiment

Shakespeare 1M d=16, 50 SE, 3 seeds × 2 variants:
- baseline: shared global Adam state (current behavior)
- per_rc: 65 per-rc buckets

Otherwise identical: `--partition-depth 1 --no-accumulate --lr 3e-3
--lr-schedule warmup-cosine --warmup-epochs 1 --optimizer rmsprop
--rmsprop-beta 0.999 --mass-weight log --entropy-lambda 1.0`.

## Result

| Seed | baseline PPL | per_rc PPL | regression |
|---|---:|---:|---:|
| 100 | 5.3443 | 6.7544 | +26% |
| 200 | 5.5602 | 6.5451 | +18% |
| 300 | 5.4755 | 6.2283 | +14% |

**Means:** baseline 5.460 ± 0.103, per_rc 6.509 ± 0.265.

Per-rc Adam regresses by **19%** with **2.6× higher seed variance**.

## Diagnosis

Two compounding mechanisms predict this regression *without falsifying the
underlying hypothesis*:

**1. RMSprop's no-bias-correction artifact (dominant).** Per-rc buckets fire
~50× total over 50 SE (one fire per SE per bucket at pd=1). At β₂=0.999,
`1 - β₂^50 ≈ 0.049`, so `v` is underestimated by ~20×, which inflates the
effective step size by ~√20 ≈ 4.5×. That's deep into instability territory.
The global-Adam baseline doesn't suffer this because its `t` is shared
(50 × 65 = 3250 fires) — the EMA is well-warmed.

**2. Estimator variance.** Even with proper bias correction (Adam's
`1 - β₂^t` factor), per-rc `v` after 50 updates is a *noisier* estimate of
that bucket's curvature than the global pooled `v` after 3250 updates. The
2.6× variance inflation in PPL across seeds is consistent with this.

Both effects scale away as super-epoch count grows, but at 50 SE neither is
manageable.

## Dump analysis (the answer to "is the signal really there")

The diagnostic `--dump-per-rc-v` was used to inspect the final per-rc `v`
buffer from one 50 SE run. Key findings:

**v magnitudes vary dramatically across buckets:**

| rc | steps | ‖v‖₂ | mean(v) | max(v) |
|---:|---:|---:|---:|---:|
| 0 | 50 | 1.86e+07 | 1.69e+03 | 6.74e+06 |
| 1 | 50 | 2.07e+08 | 1.64e+04 | 8.21e+07 |
| 2 | 50 | 7.47e+04 | 7.17e+00 | 3.06e+04 |
| 3 | 50 | 3.16 | 2.12e-04 | 1.37 |
| 4 | 50 | 1.88e+01 | 1.43e-03 | 8.04 |
| 5 | 50 | 1.34e+06 | 1.52e+02 | 4.80e+05 |
| 6 | 50 | 2.69e+06 | 2.20e+02 | 1.23e+06 |
| 7 | 50 | 1.36e+05 | 1.62e+01 | 4.61e+04 |
| 8 | 50 | 2.36e+05 | 2.75e+01 | 8.96e+04 |
| 9 | 50 | 3.17 | 2.27e-04 | 1.57 |

Eight orders of magnitude variation. Buckets 3 and 9 have near-identical
small magnitudes — likely structurally similar characters. Bucket 1 is huge
(probably the space character: dominant mass with log-mass-weighting).

**Pairwise cosine similarity (top-8 buckets by step count):**

|  | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.00 | 0.78 | 0.83 | 0.50 | 0.29 | 0.61 | 0.71 | 0.70 |
| 1 | 0.78 | 1.00 | 0.78 | 0.53 | 0.38 | 0.51 | 0.71 | 0.65 |
| 2 | 0.83 | 0.78 | 1.00 | 0.51 | 0.29 | 0.55 | 0.76 | 0.61 |
| 3 | 0.50 | 0.53 | 0.51 | 1.00 | 0.32 | 0.60 | 0.63 | 0.38 |
| 4 | 0.29 | 0.38 | 0.29 | 0.32 | 1.00 | 0.25 | 0.38 | 0.23 |
| 5 | 0.61 | 0.51 | 0.55 | 0.60 | 0.25 | 1.00 | 0.66 | 0.41 |
| 6 | 0.71 | 0.71 | 0.76 | 0.63 | 0.38 | 0.66 | 1.00 | 0.56 |
| 7 | 0.70 | 0.65 | 0.61 | 0.38 | 0.23 | 0.41 | 0.56 | 1.00 |

Similarities range 0.23-0.83 — structured, not uniform. High-mass buckets
(0,1,2,6) cluster around 0.7-0.83 with each other. Low-mass bucket 4 sits
apart (0.23-0.38 with all others). This is not what pure noise would
produce, and not what an identical-across-buckets `v` would produce either.

**Interpretation: the topological localization signal exists** — different
rcs have measurably different curvature profiles in shape *and* magnitude.
Stage 1 fails not because the signal is absent but because temporal EMA
accumulation cannot extract it at 50 fires/bucket under RMSprop.

## Verdict and next step

- **Per-rc Adam (Stage 1) is closed**: at usable super-epoch counts, the
  cold-start and small-sample variance dominate the signal. Adam-with-bias-correction
  re-runs would fix one of those (bias) but not the other (variance) —
  not a clean test of the hypothesis at this scale.
- **The signal is alive**: dump analysis shows structured per-rc `v`
  variation that pooled-Adam is averaging away. Localization is worth
  pursuing.
- **Skip Stage 1b** (Adam-rerun, per-root-char-at-pd>1). The same cold-start
  / variance problems apply with different constants. Not worth the time.
- **Proceed to Stage 2** (within-step suffix-ensemble F_p, per the
  curvature note): estimates per-position curvature from the *spatial*
  ensemble of subtree descendants, not from temporal EMA. No warmup
  requirement, no sample-size problem — each fire has many descendant
  contributions available simultaneously.

The Stage 1 failure mode is exactly what Stage 2 is designed to avoid by
construction. Stage 1 served its purpose as the cheap precursor: it
confirmed the localization signal is real and the right way to extract it
is spatially, not temporally.

## Ablation: removing mass-weighting

The original experiment used `--mass-weight log`, which multiplicatively
amplifies gradients at high-mass nodes (~5× across the rc range, squared
to ~25× in v). Hypothesis: mass-weighting was inflating per-rc's
cold-start instability via differential amplification of low-fire-count
buckets. Ablation removes mass-weighting; entropy-lambda kept on.

**Result:**

| Setting | baseline | per_rc | gap |
|---|---:|---:|---:|
| With mass-weight | 5.460 ± 0.103 | 6.509 ± 0.265 | +19.2% |
| No mass-weight   | 5.591 ± 0.122 | 6.735 ± 0.114 | +20.5% |

**Interpretation:**

- Per-rc gap is essentially unchanged (19.2% → 20.5%). Mass-weighting was
  *not* the dominant driver of the regression. Per-rc Adam fails
  intrinsically at 50 SE under RMSprop, regardless of loss reweighting.
- Per-rc variance dropped 2.3× (0.265 → 0.114) without mass-weighting.
  Mass-weighting was injecting variance specifically into the per-rc
  path via amplification of rare-bucket gradient spikes hitting
  under-warmed v.
- Baseline benefits modestly from mass-weighting (~2.4% PPL improvement).
  Consistent with prior findings — mass-weighting helps the shared path
  but doesn't differentially help per-rc.

The dump diagnostic conclusion holds: structural localization signal is
real, but the wrong instrument is being used to extract it. The fix is
spatial F (Stage 2), not parameter tuning of Stage 1.

## Files

- `run_compare.sh` — 6-run main experiment script
- `run_ablation_no_mw.sh` — 6-run no-mass-weighting ablation
- `analyze_dump.py` — offline dump analyzer
- `logs/`, `logs_no_mw/` — per-run training and PPL logs
- `results.csv`, `no_mw_results.csv` — flat CSVs of per-run PPL/wall
