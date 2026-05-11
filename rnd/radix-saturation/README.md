# Radix Saturation vs. PPL

**Status**: in progress (branch: `main`).

**Trainer note**: pre-fix, needs reassessment.

This experiment's core question still stands, but the AGPT-side quantitative
results were framed before the `Wk` / `Wv` / bias gradient fix in commit
`1c858c0` (2026-04-23). Reassess the reported depth/PPL curve on the corrected
trainer before treating the old numbers as current evidence.

## Hypothesis

The radix trie has a hard asymptote: at d=32, 99.99% of cap endpoints are
singletons (virtually every 32-gram in the Shakespeare corpus is unique).
Adding more depth can't create many new endpoints — the corpus has a
finite number of distinct substrings, period.

If loss is fundamentally a "fingerprint of the corpus's conditional
distributions," then **held-out PPL at convergence should track the radix
saturation curve**. Going d=8 → d=16 → d=32 should show diminishing PPL
gains, and d=64 should give essentially no improvement over d=32.

## Motivation for from-scratch

Prior mass-weight sweep seeded from `data/input.agpt.model` — which was
already trained to PPL ≈ 13.74 — so the delta we measured was
*fine-tuning*, not convergence. This experiment re-runs from
`data/input.random.model` (seed=42, PPL ≈ 166) so the curve reflects
actual training-from-scratch behaviour at each depth.

## Configs

All runs share architecture (d_model=64, n_heads=4, n_layers=2, d_ff=256,
vocab=65) and training recipe (RMSProp β=0.999, warmup-cosine,
entropy-lambda=1.0, single-subtree).

| Label | Tool | Depth | Weighting | Epochs | N runs |
|---|---|---|---|---|---|
| sat-d8-linear-ep3 | agpt_train | d=8 | linear | 3 | 3 |
| sat-d16-linear-ep3 | agpt_train | d=16 | linear | 3 | 3 |
| sat-d32-linear-ep3 | agpt_train | d=32 | linear | 3 | 3 |
| sat-sgd-s16-50k | microgpt | seq_len=16 | (n/a) | 50000 steps | 3 |
| sat-sgd-s32-50k | microgpt | seq_len=32 | (n/a) | 50000 steps | 3 |

`--mass-weight linear` picked from the prior sweep as the best-performing
mode; other modes can be added later if curve shape warrants.

Epoch budget = 3 chosen from a d=16 pilot (see `logs/pilot_d16.log`):
from random init, best PPL hits at ep2 and then monotonically overfits
(ep1=16.02, ep2=14.36, ep3=14.61, ... ep10=18.66). 3 epochs covers
the peak with one epoch of headroom and wastes no compute on divergence.

## Files

- `run.sh` — runs the matrix from `data/input.random.model`.
- `logs/` — per-run logs and JSON sidecars.
- `results/summary.md` — condensed PPL table and curve notes.
