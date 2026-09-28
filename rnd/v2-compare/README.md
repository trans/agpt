---
title: v2 trainer comparison run
kind: diagnostic
status: concluded
outcome: n/a
question: >-
  How does the new v2 (CUDAX) trainer train on a standard pd=1 run, for comparison with the
  v1 trainer?
answer: >-
  Only the v2 side is recorded. Ten epochs at pd=1 (d=64 L=2, depth-16 Shakespeare trie, RMSprop
  warmup-cosine, lr 3e-3) took mean training loss from 3.81 after epoch 1 to 2.165 after epoch
  10. The directory holds no v1 counterpart and no PPL evaluation.
opened: 2026-05-20
updated: 2026-05-20
code: main
eval: none
family: attention
tags: [trainer]
related: [v1-vs-v2-comparison]
---

# v2 trainer comparison run

> Stub README (2026-09-28): this directory predates the README convention. The
> summary below was reconstructed from its files; see them for detail.

**Question.** How does the new v2 (CUDAX) trainer train on a standard pd=1 run, for comparison with the v1 trainer?

**Answer.** Only the v2 side is recorded. Ten epochs at pd=1 (d=64 L=2, depth-16 Shakespeare trie, RMSprop warmup-cosine, lr 3e-3) took mean training loss from 3.81 after epoch 1 to 2.165 after epoch 10. The directory holds no v1 counterpart and no PPL evaluation.

**Sources.** v2-train.log (config header and per-epoch 'summary ... mean_loss' lines); commit 4a5e479 message ('saved model + train log from a v2 trainer comparison run'); rnd/TRIAGE.md ('one-off Codex v1-vs-v2 comparison').

**Caveats.** rnd/TRIAGE.md lists it under REMOVE. The log contradicts itself: the config line says accumulate=false, while the train-epoch line says accumulate=true. The related link to v1-vs-v2-comparison is inferred from the topic only. The directory contains a 434 KB model checkpoint (v2-run.model).
