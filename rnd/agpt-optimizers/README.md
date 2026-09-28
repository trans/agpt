---
title: AGPT optimizers
kind: experiment
status: concluded
outcome: positive
question: >-
  Does AGPT subtree training need an adaptive optimizer, or can plain SGD or momentum match
  RMSProp?
answer: >-
  It needs one. Without clipping, SGD and momentum blow up almost at once. With grad-clip-norm
  1 they finish but stall far above RMSProp: best momentum 18.50 and best SGD 19.51 against
  RMSProp 12.79 (legacy PPL, d=32 Shakespeare, 3 SE).
opened: 2026-04-25
updated: 2026-04-25
code: main
eval: legacy
family: attention
headline:
- {label: RMSProp (beta=0.999) reference, metric: legacy PPL (protocol not recorded), value: 12.79}
- {label: 'Momentum lr=3e-2, clip=1', metric: legacy PPL (protocol not recorded), value: 18.5}
- {label: 'SGD lr=3e-1, clip=1', metric: legacy PPL (protocol not recorded), value: 19.51}
tags: [optimizer]
related: [per-rc-adam-v1, gradient-population]
---

# AGPT Optimizers

**Status**: closed

**Trainer note**: post-fix.

**Code**: commit `5b1014b` (2026-04-25).

## Hypothesis

AGPT subtree training aggregates gradients across millions of token positions
per optimizer step. That may create a much wider per-parameter gradient-scale
distribution than ordinary window training. If so, AGPT may require an
adaptive optimizer to work well; plain SGD or momentum might be intrinsically
mismatched to the subtree-aggregation regime.

## Setup

- trainer: `bin/agpt_train`
- corpus/trie: `d=32` Shakespeare radix
- recipe: warmup-cosine, mass-weight `linear`, entropy-lambda `1.0`
- optimizer sweep:
  - SGD
  - momentum
  - with and without `grad-clip-norm=1`
- reference baseline:
  post-fix RMSProp deterministic subtree run, mean PPL `12.79`

## What was run

The log matrix under `logs/` covers SGD and momentum at several learning rates:

- SGD: `1e-3` to `1.0`
- momentum: `3e-4` to `1e-1`
- clipped and unclipped variants

The unclipped runs commonly collapsed into clearly broken states (`loss=0`
after an explosive first epoch). The clipped runs stayed numerically stable
enough to finish, but still underperformed badly.

## Results

Headline numbers from commit `5b1014b`:

| optimizer | best reported PPL | gap vs RMSProp |
|---|---:|---:|
| RMSProp (`β=0.999`) | **12.79** | baseline |
| momentum (`lr=3e-2`, clip=1) | 18.50 | +5.71 |
| momentum (`lr=1e-1`, clip=1) | 19.06 | +6.27 |
| SGD (`lr=3e-1`, clip=1) | 19.51 | +6.72 |
| SGD (`lr=1.0`, clip=1) | 19.71 | +6.92 |

Patterns visible in the logs:

- without clipping, SGD/momentum often blow up almost immediately
- clipping stabilizes training but does not make it competitive
- even the best clipped runs remain far above the RMSProp reference

## Conclusion

The conclusion of this sweep was strong:

- AGPT subtree aggregation and **adaptive optimization** are a package deal
- plain SGD / momentum are a poor fit for subtree training
- the main issue is not just numerical blowup; even stabilized SGD-family runs
  plateau far above RMSProp

Interpretation from the commit:

- subtree aggregation creates very uneven gradient magnitudes across
  parameters
- RMSProp's second-moment normalization handles that heterogeneity
- a single global LR leaves some parameters under-trained and others unstable

## Reproduction note

No dedicated `run.sh` was preserved in this directory. The logs themselves and
commit `5b1014b` are the main record. If this sweep needs to be rerun, rebuild
it from the optimizer / LR matrix encoded in the filenames.
