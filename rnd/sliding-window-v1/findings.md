# Sliding-Window AGPT v1.0 — Logit Pooling Results

**Date:** 2026-05-11
**Tool:** `bin/agpt_sliding_window_perplexity`
**Model:** `/tmp/agpt-gut-d16-pd1.model` (Gutenberg 5M, d=16 pd=1, 30 SE,
mass-weight=off; baseline PPL@16 = 8.01)
**Eval:** `data/gutenberg_5m.txt`, openBLAS backend

## v1.0 — Logit-Pool Sanity Check (matched-N comparison at 2048 positions)

For each target position i, compute d=16 contributor predictions
(windows starting at i-16..i-1) and pool their log-prob predictions
of token[i]. This is the simplest pooling test — operates at the
output (logit) level, no architectural change beyond running
the model d× per target.

| pool mode | PPL@2048 | NLL | delta vs deep_only |
|---|---:|---:|---:|
| **deep_only** (= PPL@d=16 baseline) | **8.628** | 2.155 | — |
| depth_w (w_k ∝ k+1) | 9.514 | 2.253 | +10.3% PPL |
| uniform | 10.510 | 2.352 | +21.8% PPL |

## v1.0 — Sanity check at N=4096

Confirms tool correctness: `deep_only` at N=4096 reproduces the
established PPL@16 baseline of 8.01 (NLL 2.0811), exactly. The pooling
infrastructure adds no numerical drift when restricted to one
contributor.

```
Pool mode:  deep_only
d_window:   16
Positions:  4096
Mean NLL:   2.081087
Perplexity: 8.0132  ✓ matches prior PPL@16 = 8.01
```

## Interpretation

**Logit pooling does not work at the output layer.** Both uniform and
depth-weighted pools strictly *regress* against the deep-only baseline.
This was the predicted outcome from the design discussion: the d
contributors are *not equally informative*. They predict the same
target token but with varying amounts of preceding context:

- Contributor j=0 (window starts at i-1): model sees just chars[i-1]
- Contributor j=8 (window starts at i-9): model sees chars[i-9..i-1]
- Contributor j=15 (window starts at i-16): model sees chars[i-16..i-1]

The j=15 contributor has full d-1 chars of context — the most-informed
prediction. The j=0 contributor has zero prior context — essentially
just a unigram prediction. Averaging d such predictions dilutes the
best-informed one with noise from the others.

The depth-weighted variant (w_k ∝ k+1) confirms this monotonically:
favoring deeper-context contributors recovers some of the gap, but
the limiting case (weight only on the deepest, = `deep_only`) is still
best. There's no useful diversity here — every contributor is just a
worse view of the same target.

## What this rules out and what it doesn't

**Ruled out:** the simplest form of sliding-window pooling at the
output layer. Logit averaging is a strict regression. No matter how
you weight the contributors, the deepest-context one is the best
single predictor.

**Not ruled out:** the actual sliding-window AGPT design from
`notes/agpt/sliding_window_agpt.md`. That design pools *residuals*
(intermediate activations), not logits, and then runs a *sequence-
level attention layer* over the pooled sequence. The hypothesis there
is fundamentally different:

- Residuals encode position p with j chars of preceding context — each
  is a distinct representation of p
- A subsequent attention pass lets the model *recombine* these
  representations across the seq_len attention window
- The attention reads multiple positions, each with their pooled
  context-aware representation

This is structurally different from "average the model's output". The
v1.0 logit pool tested ONLY the output-side ensembling, not the
architectural shift.

## Next: v1.1 — Activation Pooling + Sequence-Level Attention

The natural next test: pool the *post-final-block residuals* across
contributing windows at each global position, then run an attention
pass (and final_norm + output) on the pooled sequence.

Implementation considerations:

1. **The sequence-level attention layer needs weights.** Options:
   - Reuse the model's last transformer block's W_K, W_Q, W_V
   - Add a fresh attention head with new weights (requires init)
   - Substitute pooled residuals as input to the existing model and
     re-run through the upper blocks
2. **Sequence-length still bounded.** Even with sliding-window
   pooling, if we reuse the trained model's last block, that block
   was trained at seq_len=d=16. Running it on seq_len=32 inputs has
   the same Phase 1A failure mode (out-of-distribution).

This suggests v1.1 inference might also fail without retraining,
just at a different point in the pipeline. The actual proof of the
sliding-window-AGPT design likely requires **end-to-end training**
with the pooling step in the loop — which is the v2 commitment.

The v1.0 result IS still useful: it confirms that the architectural
work is necessary. Naive ensembling at the output is not the cheap
win we'd hoped for.

## Performance note

The logit-pool variants run at 67-70 pos/sec, vs deep_only at
1170 pos/sec — a 17× slowdown from running the model d=16× per target.
This matches expectation (one extra forward pass per contributor). For
full eval at the corpus size (5M positions × 17×) the wall budget is
~21 hours; we should rely on subsampled evals (max-positions cap) for
v1.x prototypes.

## Files

```
rnd/sliding-window-v1/
├── README.md                      — experiment plan
├── findings.md                    — this file
├── logs/
│   ├── v1_deep_only_n4096.log    — baseline sanity (PPL 8.01 ✓)
│   ├── v1_deep_only_n2048.log    — N-matched baseline (PPL 8.63)
│   ├── v1_uniform_n2048.log      — uniform pool (PPL 10.51)
│   └── v1_depth_w_n2048.log      — depth-weighted pool (PPL 9.51)
```
