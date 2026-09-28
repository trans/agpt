---
title: Precondition encoder
kind: experiment
status: concluded
outcome: negative
question: >-
  Does a GRU encoder over the 16 characters before a node's prefix, residual-injected at layer
  0, extend AGPT's effective context and lower held-out PPL?
answer: >-
  No. At d=64/L=2 (25 ep, 3 seeds) the GRU tied the baseline (byte PPL 22.687 vs 22.687),
  and a mean-pool encoder was worse (22.941). At d=128/L=6/pd=1/128 ep the GRU regressed slightly
  (byte PPL 6.474 vs 6.366) with tied train loss and 7.2x the wall time. The line is also
  closed on principle: the injection sits outside AGPT's aggregation graph.
opened: 2026-06-01
updated: 2026-06-02
code: {branch: precondition, tag: exp/precondition}
eval: canonical
headline:
- {label: baseline d128 L6 pd=1 128 ep, metric: rolling byte PPL (tail-heldout), value: 6.366,
  run: 20260602T050004-baseline-d128l6-seed1}
- {label: 'precondition GRU d_pre=16, d128 L6 pd=1 128 ep', metric: rolling byte PPL (tail-heldout),
  value: 6.474, run: 20260602T060555-precondition-d128l6-seed1}
tags: [context-length, recurrence]
related: [cap-recurrence, slot-selection-step0]
---

# Precondition encoder

The precondition strand gave AGPT context from before a node's prefix. A GRU encoder
ran over d_pre=16 characters preceding the prefix, sampled as one corpus instance per
node per fire from a sidecar file, and its state was residual-injected through `W_pre`
at layer 0 (before or after LN1). Tests used the v1 trainer via `bin/agpt_experiment`.
At d=64/L=2 (25 ep, 3 seeds, 12 runs) the GRU tied the baseline (byte PPL 22.687) and
mean-pooling was worse (22.941). With the d=128/L=6/pd=1/128 ep recipe the GRU
regressed slightly (6.474 vs 6.366), with tied train loss and 7.2x the wall time.
Closed 2026-06-02 as null. The design adds a term from outside AGPT's aggregation
graph, so it would not extend AGPT even if it had helped.

Code: branch `precondition`, tag `exp/precondition`. Key files are
`notes/seq-len-extension/precondition.md`, `src/cuda/agpt_precondition_kernels.cuh`,
`src/cuda/agpt_precondition_sidecar.cuh`, `src/tools/agpt_build_precondition_sidecar.cr`
and `src/cuda/agpt_train.cu` (YAML `experimental.precondition.d_pre`). Run dirs are
`rnd/precondition-step1/` and `rnd/precondition-d128L6/` on the branch.
