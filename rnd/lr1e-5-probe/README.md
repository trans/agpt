---
title: LR 1e-5 probe
kind: diagnostic
status: concluded
outcome: n/a
question: >-
  How does 10-epoch RMSProp training on the Shakespeare d=16 radix trie compare at lr 1e-5
  (with and without branching-endpoint entropy icing) against lr 3e-3 (constant and warmup-cosine)?
answer: >-
  On training loss only: lr 1e-5 barely trains in 10 epochs (loss 3.393 with icing, 3.349
  without), while lr 3e-3 reaches 2.199 with a constant schedule and 2.063 with warmup-cosine.
  No held-out PPL was measured.
opened: 2026-05-20
updated: 2026-05-20
code: main
eval: none
family: attention
headline:
- {label: 'lr 1e-5, entropy icing on, 10 epochs', metric: 'train loss (nats, training trie)',
  value: 3.393104}
- {label: 'lr 3e-3 warmup-cosine, 10 epochs', metric: 'train loss (nats, training trie)',
  value: 2.063377}
tags: [optimizer]
---

# LR 1e-5 probe

> Stub README (2026-09-28): this directory predates the README convention. The
> summary below was reconstructed from its files; see them for detail.

**Question.** How does 10-epoch RMSProp training on the Shakespeare d=16 radix trie compare at lr 1e-5 (with and without branching-endpoint entropy icing) against lr 3e-3 (constant and warmup-cosine)?

**Answer.** On training loss only: lr 1e-5 barely trains in 10 epochs (loss 3.393 with icing, 3.349 without), while lr 3e-3 reaches 2.199 with a constant schedule and 2.063 with warmup-cosine. No held-out PPL was measured.

- lr 1e-5, entropy icing on, 10 epochs: train loss (nats, training trie) = 3.393104
- lr 3e-3 warmup-cosine, 10 epochs: train loss (nats, training trie) = 2.063377

**Sources.** Epoch loss lines and header settings in train.log, lr3e-3/train.log, lr3e-3-wc/train.log and no-icing/train.log; commit 4a5e479 message calls these LR probe runs.

**Caveats.** rnd/TRIAGE.md lists it under REMOVE as a one-off LR probe. v1 CUDA trainer, before the 2026-05-26 loss fix. The headline values come from train.log, not from a README or result.json.
