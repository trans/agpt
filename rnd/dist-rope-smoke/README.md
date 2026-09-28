---
title: dist-rope smoke test
kind: experiment
status: concluded
outcome: negative
question: >-
  Does replacing chunk-local RoPE positions with per-substring position-distribution summaries
  (dist-rope, or the scalar expected position) help AGPT training?
answer: >-
  No. After 100 SE on Shakespeare (d=16 trie, d64/L2), final training loss was 1.489 with
  default RoPE, 1.753 with dist-rope (+18%) and 1.929 with expected position (+30%). Per-substring
  position summaries break RoPE's relative-position semantics.
opened: 2026-05-25
updated: 2026-05-25
code: main
eval: none
family: attention
headline:
- {label: 'Default RoPE, 100 SE', metric: 'training loss (nats), epoch 100', value: 1.489}
- {label: 'dist-rope, 100 SE', metric: 'training loss (nats), epoch 100', value: 1.753}
- {label: 'expected-position RoPE, 100 SE', metric: 'training loss (nats), epoch 100', value: 1.929}
tags: [position, rope]
related: [rope-position-substitution, harmonic-filter-diagnostic, harmonic-bias-prototype]
---

# dist-rope smoke test

> Stub README (2026-09-28): this directory predates the README convention. The
> summary below was reconstructed from its files; see them for detail.

**Question.** Does replacing chunk-local RoPE positions with per-substring position-distribution summaries (dist-rope, or the scalar expected position) help AGPT training?

**Answer.** No. After 100 SE on Shakespeare (d=16 trie, d64/L2), final training loss was 1.489 with default RoPE, 1.753 with dist-rope (+18%) and 1.929 with expected position (+30%). Per-substring position summaries break RoPE's relative-position semantics.

- Default RoPE, 100 SE: training loss (nats), epoch 100 = 1.489
- dist-rope, 100 SE: training loss (nats), epoch 100 = 1.753
- expected-position RoPE, 100 SE: training loss (nats), epoch 100 = 1.929

**Sources.** The 'Epoch 100: loss=' lines in default.log / distrope.log / expected.log, the message of commit 32c3a0c, and notes/seq-len-extension/position-distributions-plan.md ('ruled out', +18% / +30%).

**Caveats.** The only metric is training loss (no PPL eval). v1 trainer, RMSProp lr=3e-3, anc-grad on. rnd/TRIAGE.md lists this dir under KEEP ('regression was decisive').
