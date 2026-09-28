---
title: Dual-view consistency (forward + backward models)
kind: experiment
status: concluded
outcome: inconclusive
question: >-
  Does a stop-gradient KL consistency loss between a forward (prefix) model and a backward
  (reversed-suffix) model shrink their divergence and improve forward-only PPL?
answer: >-
  Only the mechanism was tested, at 10k positions with the Crystal trainer. Coupling shrinks
  the F-vs-B KL gap (0.66 nats uncoupled, 0.54 at beta 0.1, 0.17 at beta 1.0) at little CE
  cost, and a shuffled-suffix control does not shrink it (0.69). Forward-only PPL (Tier 2)
  and the ensemble (Tier 3) were never measured, and no 50k or CUDA results were recorded.
opened: 2026-05-05
updated: 2026-05-05
code: main
eval: none
family: attention
headline:
- {label: 'beta 1.0, aligned suffix, 10k positions', metric: 'F-vs-B symmetric KL gap (nats,
    training)', value: 0.17}
- {label: 'beta 0 (uncoupled), 10k positions', metric: 'F-vs-B symmetric KL gap (nats, training)',
  value: 0.66}
- {label: 'beta 0.1, shuffled-suffix control, 10k positions', metric: 'F-vs-B symmetric KL
    gap (nats, training)', value: 0.69}
tags: [targets]
related: [cap-folding, virtual-tree]
---

# Dual-view consistency (forward + backward models)

> Stub README (2026-09-28): this directory predates the README convention. The
> summary below was reconstructed from its files; see them for detail.

**Question.** Does a stop-gradient KL consistency loss between a forward (prefix) model and a backward (reversed-suffix) model shrink their divergence and improve forward-only PPL?

**Answer.** Only the mechanism was tested, at 10k positions with the Crystal trainer. Coupling shrinks the F-vs-B KL gap (0.66 nats uncoupled, 0.54 at beta 0.1, 0.17 at beta 1.0) at little CE cost, and a shuffled-suffix control does not shrink it (0.69). Forward-only PPL (Tier 2) and the ensemble (Tier 3) were never measured, and no 50k or CUDA results were recorded.

- beta 1.0, aligned suffix, 10k positions: F-vs-B symmetric KL gap (nats, training) = 0.17
- beta 0 (uncoupled), 10k positions: F-vs-B symmetric KL gap (nats, training) = 0.66
- beta 0.1, shuffled-suffix control, 10k positions: F-vs-B symmetric KL gap (nats, training) = 0.69

**Sources.** PRELIMINARY_FINDINGS.md results table and tier readout; PLAN.md for the hypothesis and tiers.

**Caveats.** There is no README; the dir holds PLAN.md, PLAN_REVIEW_1.md and PRELIMINARY_FINDINGS.md. PLAN.md names a branch dual-model-fold that no longer exists, but the trainer (src/tools/agpt_dual_train.cr) is on main. PLAN.md itself says this is per-position one-hot CE, not trie-aggregated AGPT. The headline values are KL at 10k positions, not PPL. The 'Tier 4 confirmed' claim rests on single runs.
