---
title: The one-pass gap (AGPT pd=1 vs ordinary minibatch training, 1–8 passes)
kind: experiment
status: planned
outcome: n/a
question: >-
  AGPT's measuring stick is "one epoch and done". How close does AGPT pd=1 get to a conventional
  training run in one pass over Tiny Shakespeare, and in 2, 4 and 8 passes, against ordinary
  minibatch Adam training of the same d64/L2 model from the same weights, at matched passes and
  wall time? And is AGPT's deficit just its 65 updates per pass?
answer: >-
  Not run yet. Planned 2026-11-14: 44 short runs through the orchestrator (AGPT pd=1, minibatch 128,
  and minibatch 16,302 = 65 updates per pass), each budget its own cosine schedule, with a
  learning-rate sweep per budget.
opened: 2026-11-14
updated: 2026-11-14
code: main
eval: canonical
family: attention
tags: [one-pass, baseline, cadence, optimizer]
related: [gradient-population, context-tree-orientation, vanilla-attn-torch, partition-depth]
---

# one-pass-gap

**Status:** planned (2026-11-14). The user asked for it after we found the
work had drifted off AGPT's main path. It is item 04 of the site's next
tests ("Measure the one-pass gap"), never done before.

## Question

AGPT's goal is a model that learns enough from one pass over a corpus to
match a conventional training run, in less time. The record has AGPT's
best d64/L2 result after 625 passes (rolling byte PPL 4.764,
`gradient-population` `adam-pd1-exact-625ep`), and an ordinary minibatch
run of the same model at 4.45 after 10 passes (June, `vanilla-attn-torch`,
no `result.json`). Nobody has measured either method at one pass, at
matched cost, under the canonical evaluation.

Three questions, in order:

1. **The gap at one pass:** AGPT pd=1 after one trie traversal against
   minibatch training after one epoch, and against the conventional run.
2. **How many passes AGPT needs** to reach what minibatch training reaches
   in one, and how many seconds that costs.
3. **Is it the update count?** AGPT pd=1 takes 65 Adam steps per pass;
   minibatch training at 128 windows takes 8,279. The third arm is
   minibatch training with 16,302 windows per step, 65 steps per pass.
   If it tracks AGPT, AGPT's deficit is its cadence, as the
   context-tree orientation's no-trie control suggested (a tie with
   shuffled minibatches at equal steps). If AGPT beats it, the trie's
   grouping is worth something on its own.

## Design

- **Model and data:** d64 L2 h4 ff256, seq_len 16, Tiny Shakespeare
  canonical split `4fa9aec1db6b3aea`; every run starts from the seed
  weights the AGPT record uses
  (`rnd/cudax-d16-linear-mass-rerun/seeds/shake-d64L2-h4-dff256-d16-seed42.model`,
  near-random: PPL 66). Seed 42.
- **One pass covers the same events in each arm.** A depth-16 trie holds
  every prefix of length 1–16 with its next character; minibatch training
  visits every window start once per epoch with 16 targets per window.
- **AGPT arm (`agpt-*`):** `bin/agpt_train_v2`, depth-16 trie, pd=1, exact
  ancestor gradient: the 625-pass record config with only the budget and
  learning rate changed.
- **Minibatch arms (`mb-*`, `mbL-*`):** `src/tools/agpt_minibatch_train.py`
  (new, a port of the June trainer to an orchestrator custom trainer).
  `mb`: 128 windows per step. `mbL`: 16,302 windows per step,
  accumulated over micro-batches of 2,048 (same mean gradient).
- **Same optimizer and schedule everywhere:** Adam (0.9, 0.999, eps 1e-8,
  no weight decay, the same as v2's kernel), warmup-cosine per optimizer
  step with no warmup, down to zero. The minibatch trainer reimplements
  v2's `scheduled_lr` step for step.
- **Each budget is its own run** (1, 2, 4, 8 passes), so every run
  finishes its own schedule; intermediate checkpoints of a longer cosine
  run would not be the same thing.
- **Learning rates:** AGPT and `mbL` at 0.003, 0.006, 0.012, 0.024; `mb`
  at 0.0015, 0.003, 0.006. A best value at the edge of a grid means
  extending it before reading the result.
- **Evaluation:** the orchestrator's canonical lm-eval rolling byte PPL on
  the multi-chunk held-out split, plus its fixed-window PPL. Training
  seconds come from each `result.json`.
- **Cost:** 44 runs, about an hour of GPU in one slot, most of it the 45
  epochs of the `mb` arm. `make_configs.py` writes `configs/`;
  `run_grid.sh` rebuilds the binaries and runs them, one-pass runs first.

## Checks done before the run (2026-11-14, CPU)

- The trainer's model writer reproduces the seed file byte for byte after
  a load/save round trip.
- A 2-epoch smoke run through `bin/agpt_experiment` produced a
  `result.json` with a scored epoch checkpoint and its training seconds;
  the learning rate at the end of epoch 1 of 2 was half the peak, as the
  cosine requires.
- Micro-batch accumulation: 512 windows in one pass vs four of 128 gave
  the same training loss to 1e-5 and weights within 7e-4 (Adam amplifies
  rounding in near-zero gradients; the total change was 0.064).

## Caveats known in advance

- One seed. The seed weights are the AGPT record's; in June they slowed
  minibatch training compared with random init.
- AGPT's 65 pd=1 units differ in size (by root character); `mbL`'s 65
  steps are equal random batches.
- Equal passes is not equal compute: training seconds are reported for
  each run so the comparison can be read either way.
