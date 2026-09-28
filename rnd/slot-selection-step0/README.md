---
title: Backoff slot selection
kind: experiment
status: concluded
outcome: negative
question: >-
  Does giving attention extra K/V slots for suffix-backoff trie nodes (B=4, gradient through
  the anc-grad path) lower held-out PPL at Shakespeare d=16?
answer: >-
  No. At L=2/pd=0/25 ep rmsprop, B=4 raised byte PPL (3-seed means: 19.25 same-as-k, 18.71
  sentinel, vs 17.10 baseline). It was null at L=4 (23.23 vs 23.24) and in a 100-epoch Adam
  run (11.67 vs 11.56, one seed). The design needs pd=0, so it cannot be tested at pd=1, where
  AGPT performs.
opened: 2026-05-30
updated: 2026-05-31
code: {branch: slot-selection, tag: exp/slot-selection}
eval: canonical
family: attention
headline:
- {label: 'baseline B=0, L=2 pd=0 25 ep rmsprop', metric: 'rolling byte PPL (tail-heldout),
    3-seed mean', value: 17.1}
- {label: 'backoff B=4 same-as-k, L=2 pd=0 25 ep', metric: 'rolling byte PPL (tail-heldout),
    3-seed mean', value: 19.25}
- {label: 'backoff B=4 sentinel, L=2 pd=0 25 ep', metric: 'rolling byte PPL (tail-heldout),
    3-seed mean', value: 18.71}
tags: [context-length, attention, partitioning]
related: [cap-recurrence, precondition, anc-grad]
---

# Backoff slot selection

Slot selection (Step 0) added B=4 extra K/V slots to attention for each query. The slots
hold the suffix-backoff trie nodes (the node's path minus leading characters, as in
Kneser-Ney backoff), with gradient through the existing `--anc-grad` closed-form path.
It was built in the v1 CUDA trainer and sign-checked: the K/V gradient helps, and the harm
comes from diluting the attention softmax. At d=16, L=2, pd=0, 25 ep rmsprop, both variants
raised byte PPL over the 17.10 baseline (same-as-k 19.25, sentinel 18.71; 3 seeds). At L=4
and in a 100-epoch Adam run the effect was null. Closed 2026-05-31: the mechanism needs
pd=0 because `h_subtree` is fire-scoped, so it cannot run at pd=1, where AGPT reaches
useful PPL.

Code: branch `slot-selection`, tag `exp/slot-selection`. Key files are
`notes/seq-len-extension/slot-selection.md` (branch version), `src/cuda/agpt_backoff_kernels.cuh`,
`src/cuda/agpt_backoff_table.cuh`, `src/cuda/agpt_train.cu`, `src/cuda/kernels.cu` and
`src/tools/agpt_build_backoff_table.cr` (YAML `experimental.backoff_slots`,
`backoff_position`). Run configs are in `rnd/slot-selection-step0/configs/`.
