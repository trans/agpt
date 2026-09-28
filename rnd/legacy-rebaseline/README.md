---
title: Legacy v1 rebaseline
kind: baseline
status: concluded
outcome: n/a
question: >-
  What held-out PPL does the legacy v1 trainer reach on Shakespeare 1M (d=16, 10 SE, anc-grad
  off) from the shared seed models, as a parity reference for v2?
answer: >-
  Seed 1 reached 9.58 and seed 2 reached 8.89 (legacy sliding-window PPL, d=16, 10k positions
  of a 50k-char held-out file). The landing commit notes that v2 beat legacy by about 0.9
  PPL at this config.
opened: 2026-05-21
updated: 2026-05-21
code: main
eval: legacy
family: attention
headline:
- {label: 'v1 trainer, 10 SE, anc-grad off, seed 1', metric: 'legacy sliding-window held-out
    PPL (d=16, 10k positions)', value: 9.5843}
- {label: 'v1 trainer, 10 SE, anc-grad off, seed 2', metric: 'legacy sliding-window held-out
    PPL (d=16, 10k positions)', value: 8.8919}
tags: [baseline, trainer, reproducibility]
related: [cudax-anc-grad-parity, anc-grad, v2-compare]
---

# Legacy v1 rebaseline

> Stub README (2026-09-28): this directory predates the README convention. The
> summary below was reconstructed from its files; see them for detail.

**Question.** What held-out PPL does the legacy v1 trainer reach on Shakespeare 1M (d=16, 10 SE, anc-grad off) from the shared seed models, as a parity reference for v2?

**Answer.** Seed 1 reached 9.58 and seed 2 reached 8.89 (legacy sliding-window PPL, d=16, 10k positions of a 50k-char held-out file). The landing commit notes that v2 beat legacy by about 0.9 PPL at this config.

- v1 trainer, 10 SE, anc-grad off, seed 1: legacy sliding-window held-out PPL (d=16, 10k positions) = 9.5843
- v1 trainer, 10 SE, anc-grad off, seed 2: legacy sliding-window held-out PPL (d=16, 10k positions) = 8.8919

**Sources.** Numbers come from shakespeare/seed{1,2}_off/heldout.log. The purpose comes from the message of commit 2b913be ('fresh legacy baseline runs against /tmp/seed{1,2,3}.model for the parity reference').

**Caveats.** The commit mentions seeds 1-3, but only seeds 1 and 2 are present. RMSProp lr=3e-3 warmup-cosine, pd=1. The held-out file is 50k tokens (probably the tail of data/input.txt, and possibly inside the training trie (unverified). rnd/TRIAGE.md lists this dir as affected by the 2026-05-26 loss fix.
