---
title: The one-pass gap (AGPT pd=1 vs ordinary minibatch training, 1–8 passes)
kind: experiment
status: concluded
outcome: negative
question: >-
  AGPT's measuring stick is "one epoch and done". How close does AGPT pd=1 get to a conventional
  training run in one pass over Tiny Shakespeare, and in 2, 4 and 8 passes, against ordinary
  minibatch Adam training of the same d64/L2 model from the same weights, at matched passes and
  wall time? And is AGPT's deficit just its 65 updates per pass?
answer: >-
  Far off. After one pass AGPT pd=1 scores rolling byte PPL 15.73 (best of four learning rates);
  ordinary minibatch training of the same model from the same weights scores 4.83 in 31 s, which
  already beats AGPT's 100-pass run (5.225), and 4.66 after two passes, better than AGPT's best
  on record (4.764 after 625 passes). After 8 passes AGPT reaches 8.71. Most of the gap is the
  update count (65 steps per pass against 8,279): random equal batches with AGPT's 65 steps per pass
  score 12.56 after one pass. But AGPT trails that control at every budget, by a widening margin
  (5.67 against 8.71 after 8 passes), so AGPT's steps are worth about a third of random-batch
  steps here. One seed; 48 runs.
opened: 2026-11-14
updated: 2026-11-14
code: main
eval: canonical
family: attention
tags: [one-pass, baseline, cadence, optimizer]
related: [gradient-population, context-tree-orientation, vanilla-attn-torch, partition-depth]
---

# one-pass-gap

**Status:** concluded (2026-11-14). The user asked for it after we found the
work had drifted off AGPT's main path. It is item 04 of the site's next
tests ("Measure the one-pass gap"), never done before. 48 runs in slot 30
(08:32–09:38 UTC), 64 minutes in all.

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
  at 0.0015, 0.003, 0.006, and 0.012 added after 0.006 came out best at
  every budget. Every best value below is inside its grid.
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

## Results

Canonical rolling byte PPL on the multi-chunk held-out split (fixed-window
PPL in brackets), best learning rate per arm and budget, and training
seconds from `result.json`. `python3 rnd/one-pass-gap/summarize.py
--markdown` prints every run.

| Passes | AGPT pd=1 (65 steps/pass) | minibatch 16,302 (65 steps/pass) | minibatch 128 (8,279 steps/pass) |
|---:|---:|---:|---:|
| 1 | 15.734 (15.657), lr 0.012, 8 s | 12.562 (12.496), lr 0.006, 11 s | **4.826** (4.293), lr 0.006, 31 s |
| 2 | 12.307 (12.181), lr 0.012, 14 s | 9.242 (8.939), lr 0.012, 19 s | **4.659** (4.113), lr 0.006, 60 s |
| 4 | 10.335 (10.178), lr 0.006, 27 s | 7.015 (6.560), lr 0.012, 36 s | **4.582** (4.029), lr 0.006, 116 s |
| 8 | 8.710 (8.462), lr 0.006, 54 s | 5.669 (5.169), lr 0.012, 71 s | **4.526** (3.973), lr 0.006, 266 s |

For reference, from the record on the same split and model: AGPT pd=1
with the exact gradient, lr 0.0015, 100 passes 5.225 (654 s) and 625
passes 4.764 (`gradient-population`); minibatch training from random init
at a constant rate, 4.45 after 10 passes (June, no `result.json`).

## Reading

- **AGPT is nowhere near one epoch and done at pd=1.** One pass gives
  15.7, against 4.83 for one pass of minibatch training. Minibatch
  training beats AGPT's 100-pass result in one pass and its best result
  ever (625 passes, about 70 minutes) in two passes (60 s).
- **Equal wall time doesn't rescue it.** An AGPT pass is about 4× cheaper
  here (7 s against 31 s), but AGPT after 54 s (8 passes) is at 8.71,
  where minibatch training was at 4.83 after 31 s.
- **The update count explains most of the gap.** With AGPT's 65 steps per
  pass, random equal batches score 12.6 after one pass, much nearer to
  AGPT's 15.7 than to minibatch training's 4.8. Each step there averages
  16,302 windows, so it is close to the exact full-batch gradient, and
  65 such steps can't train the model in one pass however exact they
  are.
- **But AGPT's 65 steps are worth less than 65 random batches,** and
  the shortfall grows with training: 15.7 against 12.6 after one pass,
  8.71 against 5.67 after eight. AGPT after 8 passes (520 steps) sits
  between the control's 2 and 4 passes (130 and 260 steps), so its steps
  are worth about a third of the control's. Three things differ and are
  not yet separated:
  1. **Unit sizes:** AGPT's 65 units are the root characters' subtrees,
     from the space-rooted one down to a handful of events, and Adam
     takes a full-size step on each. The control's batches are equal.
  2. **Order:** v2 trains the units in the same fixed order every pass
     (`experimental.unit_order_seed` defaults to −1), so each pass ends on
     the same rare letters at the bottom of the cosine. The control
     shuffles.
  3. **Grouping:** each unit holds the contexts that start with one
     character. The context-tree orientation control found that grouping
     costs nothing for a GRU against random groups of the same unequal
     sizes in shuffled order, so it is the least likely of the three, but
     it is untested with attention.
- **Learning rates:** AGPT's best rate falls from 0.012 to 0.006 as the
  budget grows, and 0.006 is best for minibatch training at every budget
  (the 625-pass record used 0.0015). The control's best is 0.006–0.012.

## Next tests (not run)

1. **What makes AGPT's steps weaker** (about 15 minutes of GPU): the
   minibatch trainer grouping windows by their first character, which is
   AGPT pd=1's units in minibatch form, in v2's fixed order and shuffled;
   random groups with the units' sizes; and v2 with
   `experimental.unit_order_seed` set. The fixed-order first-character
   arm should reproduce AGPT's numbers, which would also check that AGPT
   pd=1 is grouped SGD in practice.
2. **AGPT at minibatch cadence:** pd=3 has about 11,500 units per pass,
   more than minibatch training's 8,279, so one pass of it is the direct
   test of one epoch and done with exact subtree steps. pd≥2 plans never
   train targets shallower than pd (`gradient-population` Exp 2 reading),
   which ruins the rolling metric, so the shallow targets need training
   first (for example one pd=1-style pass over the depth-<pd stubs).

## Caveats

- One seed. The seed weights are the AGPT record's; in June they slowed
  minibatch training compared with random init.
- AGPT's 65 pd=1 units differ in size (by root character); `mbL`'s 65
  steps are equal random batches.
- Equal passes is not equal compute: training seconds are reported for
  each run so the comparison can be read either way. They time the
  trainer process, start-up included: about 1 s for AGPT (loading the
  trie; a pass is about 6.6 s) and about 2 s for the minibatch trainer
  (starting PyTorch, reading the corpus; a pass is 28.6 s).
- Each run's `meta.json` says `git_dirty: true`: the trainer and the
  configs were committed (`84054bc`) before the runs, but untracked files
  (this README's summary script, the grid's output) and, for the four
  lr 0.012 runs, the edited `make_configs.py` were in the tree.
