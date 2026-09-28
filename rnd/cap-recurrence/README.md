---
title: Cap recurrence
kind: experiment
status: concluded
outcome: negative
question: >-
  Does feeding a trie node's predecessors' deepest hidden state (h_cap, mass-weighted over
  corpus predecessors) into the node as a side input h_in extend AGPT's context and lower
  loss at Shakespeare d=16?
answer: >-
  No. Direct add, a learnable additive W_inject and a learnable K/V injection slot were all
  flat to slightly worse on training loss; an oracle h_in that encodes the answer did lower
  training loss (1.716 to 1.513 at 25 ep), so the wiring works. The re-test with lm-eval byte
  PPL at 25 ep, lr=1e-2 gave 8.53 (mass-weighted h_in) and 8.64 (random h_in) against 8.29
  for the baseline (3 interleaved pairs). Averaging over predecessors leaves no usable signal.
opened: 2026-05-27
updated: 2026-05-30
code: {branch: agpt-cap-recurrence, tag: exp/cap-recurrence}
eval: canonical
headline:
- {label: 'baseline, 25 ep lr=1e-2', metric: 'lm-eval rolling byte PPL (agpt_lm_eval.py, tail
    heldout), 3-pair mean', value: 8.29}
- {label: 'kv-mass h_in injection, 25 ep lr=1e-2', metric: 'lm-eval rolling byte PPL (agpt_lm_eval.py,
    tail heldout), 3-pair mean', value: 8.53}
- {label: 'kv-random h_in control, 25 ep lr=1e-2', metric: 'lm-eval rolling byte PPL (agpt_lm_eval.py,
    tail heldout), 3-pair mean', value: 8.64}
tags: [recurrence, context-length, attention]
related: [slot-selection-step0, precondition]
---

# Cap recurrence

Cap recurrence fed the deepest hidden state `h_cap` of a trie node's corpus predecessors
(averaged by mass) into the node as a side input `h_in`, to give AGPT context from before
the node's prefix. Three injection forms were tried on the v1 CUDA trainer at Shakespeare
d=16 (d_model=64, L=2): direct add, a learnable additive `W_inject`, and a learnable K/V
slot in attention. All were flat to slightly worse. An oracle `h_in` that encodes the
answer did lower loss, which rules out a wiring bug. A re-test with lm-eval byte PPL
(25 ep, lr=1e-2) gave 8.53 for mass-weighted `h_in` and 8.64 for random `h_in`, against
8.29 for the baseline. Closed as negative: averaging over predecessors, which the radix
factorization forces, removes the per-instance signal.

Code: branch `agpt-cap-recurrence`, tag `exp/cap-recurrence`. Key files are
`src/cuda/agpt_cap_capture.cuh`, `src/cuda/agpt_train.cu` (env vars `AGPT_CAPTURE_H_CAPS`,
`AGPT_CAP_KV_INJECT`, `AGPT_CAP_H_IN_WEIGHT`), `src/tools/agpt_build_predecessor_table.cr`
and `notes/seq-len-extension/cap-recurrence-design.md`. Run write-ups are in
`rnd/cap-recurrence/2026*/README.md` on the branch.
