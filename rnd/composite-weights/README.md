---
title: Composite mass × entropy weights
kind: experiment
status: concluded
outcome: negative
question: >-
  Does multiplying mass and entropy per-event loss weights together carry over to Gutenberg,
  where the single-axis Shakespeare wins did not?
answer: >-
  No. On Gutenberg 5M d=16 (10 SE, 36 cells, 3 seeds each) the best composite, mass-log ×
  entropy-up with events normalization, reached 9.28 ± 0.31 legacy held-out sliding-window
  PPL against a 9.24 baseline. Every mass-linear cell was 4-5% worse. Per-event loss weighting
  was closed as a direction on 2026-05-23.
opened: 2026-05-21
updated: 2026-05-22
code: main
eval: legacy
family: attention
tags: [gradient, data]
related: [depth-weight, per-fire-norm, gutenberg-anc-sweep, beta2-diagnostic]
---

# Composite mass × entropy weights

> Stub README (2026-09-28): this directory predates the README convention. The
> summary below was reconstructed from its files; see them for detail.

**Question.** Does multiplying mass and entropy per-event loss weights together carry over to Gutenberg, where the single-axis Shakespeare wins did not?

**Answer.** No. On Gutenberg 5M d=16 (10 SE, 36 cells, 3 seeds each) the best composite, mass-log × entropy-up with events normalization, reached 9.28 ± 0.31 legacy held-out sliding-window PPL against a 9.24 baseline. Every mass-linear cell was 4-5% worse. Per-event loss weighting was closed as a direction on 2026-05-23.

**Sources.** run_gutenberg.sh header (hypothesis, 36-cell design, baseline 9.24); results only in project memory (microgpt memory project_composite_weights_gutenberg.md, project_weighting_arc_closed.md).

**Caveats.** The directory holds only the two run scripts; the result outputs (gutenberg/, se-sweep/) were never committed, so the answer rests on the memory note. No result was found anywhere for the second script, run_shakespeare_se_sweep.sh (a step-count test of the mass-linear+events regime flip). rnd/TRIAGE.md marks the directory 'directly relevant — redo' and suggests retiring it rather than re-running. No headline, because the numbers are not in any directory file.
