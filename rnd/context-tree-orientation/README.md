---
title: Context-tree orientation
kind: experiment
status: concluded
outcome: positive
question: >-
  Does the orientation of AGPT's context tree matter for training, the prefix trie (oldest
  character at the root, as AGPT has always built it) or the suffix tree (newest at the root,
  each subtree one target's prediction problem), and if so, why?
answer: >-
  Yes, and the prefix trie is the right one, for a reason nobody had seen (a minor breakthrough,
  with a proviso to double-check). A pd=1 step uses one root-child subtree. Prefix-trie units
  group contexts by their oldest, least predictive character, so their targets are well mixed and
  they train as well as shuffled minibatches with the same number of steps. Suffix-tree units
  group by the newest, most predictive character, so each step is a target-sorted batch. With the
  tree removed and one GRU, that grouping alone costs about two thirds of the suffix tree's pd=1
  gap; the newest-first reading costs the rest. GRU d64 depth 8, diagnostic held-out PPL, mostly
  one seed.
opened: 2026-10-03
updated: 2026-11-10
code: main
eval: legacy
family: recurrent
headline:
- label: 'pd=1 prefix-trie units, oldest-first GRU, d64 depth 8, 500 epochs (2 seeds)'
  metric: diagnostic held-out PPL (context <= 8, carved split, not canonical)
  value: 5.093
- label: 'Same GRU and steps, suffix-tree units (grouping control, no tree)'
  metric: diagnostic held-out PPL (context <= 8, carved split, not canonical)
  value: 5.805
- label: 'pd=1 suffix tree, newest-first GRU, 500 epochs'
  metric: diagnostic held-out PPL (context <= 8, carved split, not canonical)
  value: 6.286
tags: [trie-structure, recurrence, coherence, partitioning, minor-breakthrough]
related: [gradient-population, linear-recurrence, count-backoff-gate, slot-selection-step0]
---

# Context-tree orientation

Design and motivation: `todo/context-tree-orientation.md`. In short, the
prefix trie groups contexts by their oldest characters. The suffix tree
groups them by their newest. Both have the same nodes, counts and loss, but
in the suffix tree a node's whole subtree predicts the same target as the
node, and the parent is the backoff context. The derivation predicted that
the trie's gradient sums (`G_p = g_p + Σ J^T G_child`) would be more coherent
in the suffix tree.

## Conclusion (2026-11-10): a minor breakthrough

The user called the strand on 2026-11-10, labelled it a minor breakthrough,
and attached a proviso to double-check and polish it (the Proviso section,
at the end).

AGPT has used the prefix trie from the start, and its partitioned training
(pd≥1, one optimizer step per root-child subtree) has worked. It worked for a
reason nobody had identified. A prefix-trie pd=1 unit is every context that
starts with one character. The oldest character of a context says the least
about the next one, so the units come out well mixed in their targets, and
stepping through them behaves like stepping through shuffled minibatches. The
transposed tree, the suffix tree, groups contexts by their newest character,
which says the most. Its units are target-sorted batches, and partitioned
training on them is clearly worse. The user had long expected AGPT's training
to show exactly this sorting problem, and never understood why it didn't show
more clearly. It didn't because AGPT, without anyone choosing it for this
reason, never grouped that way.

The evidence (GRU d64, depth 8, Adam at 3e-3, 65 steps per epoch, diagnostic
held-out NLL):

- **Target skew of the units** (mean KL of a unit's targets from the corpus's):
  suffix-tree 0.861 nats, prefix-trie 0.090, shuffled minibatches 0.0002.
- **The grouping causes most of the gap.** In a no-trie control, the same GRU
  takes the same steps and only the grouping changes. At 500 epochs:
  - oldest-first GRU: suffix-tree grouping 1.759, prefix-trie grouping 1.628;
  - newest-first GRU: 1.838 and 1.681.

  The grouping accounts for about two thirds of the October pd=1 gap, and the
  newest-first reading for the rest.
- **Prefix-trie units are as good as shuffled minibatches.** With equal
  numbers of equal-sized steps they agree on train loss to 0.001 at epoch 100,
  under a constant and a cosine schedule. Held-out differences are about 0.01 and change sign with the
  epoch: 1.628 and 1.632 (two seeds) against 1.632 for start-labelled random
  batches at 500 epochs; 1.704 against 1.714 for shuffled windows under
  cosine at 100.

The user suspected the prefix units' remaining correlation (skew 0.09 against
~0) explained part of AGPT's gap to vanilla minibatch training
(`rnd/vanilla-attn-torch`). At an equal number of equal-sized steps, this
control finds no such cost, so that gap more likely comes from elsewhere. The
plainest candidate is step count: vanilla training takes many more, smaller
steps per epoch. Their reading (2026-11-10) is that getting further needs
something past the tree, though they suspect something is still being missed.

## Setup

`src/tools/agpt_context_tree_coherence.py`. Carved Shakespeare split
`data/.splits/4fa9aec1db6b3aea` (train slice). The node table is every
context of length 1–8 with its next-character counts: 1,495,879 nodes and
8,477,036 events, shared by both trees. There are 1,401 depth-2 units, as in
`rnd/gradient-population` Experiment 1.

| model | tree | node state reads | readout |
|---|---|---|---|
| fwd | prefix trie, parent = s[:-1] | newest char (left-to-right GRU) | W h_s + b |
| rev | suffix tree, parent = s[1:] | oldest char (newest-first GRU) | W h_s + b |
| rev-res | suffix tree | oldest char | z_parent + W h_s + b |

All three use d_model 64, GRUCell and the same initial weights (seed 1). Each
takes 2,000 full-batch Adam steps at lr 3e-3 on the exact gradient of the
same objective, with no partitioning, so the update schedule is identical.
The gradient is computed one depth at a time, leaves first, with the
children's summed adjoint injected. It matches plain autograd to 1e-16 in
float64 for all three models, and the depth-2 unit gradients plus the
depth-1 own losses sum to the full gradient to 1e-15.

Held-out is a diagnostic: mean NLL of each next character with up to 8
characters of context, within each of the 10 held-out chunks. It is not the
canonical lm-eval protocol.

Coherence, as defined in the tool docstring:
- **node**: at node p, the parts are the readout adjoint at `h_p` and each
  child's `J^T G_child`. The kept fraction is `Σ_p ‖G_p‖ / Σ_p Σ‖parts‖`, per
  depth.
- **unit**: parameter-space gradients of the depth-2 subtrees, as in
  Experiment 1. ρ global is what a pd=0 step keeps of them, ρ per-root is what
  a pd=1 step keeps (event-weighted average over roots), and pd1 is ρ over the
  65 whole root subtrees.

Run: `20261003T051146-gru-d64-depth8-fullbatch-adam3e-3-2000/` (`run.log`,
`summary.json`). The tool was uncommitted at launch on top of cac6c86.
About 8 minutes of training per model.

## Results

| model | train loss @2000 | held-out NLL | held-out PPL |
|---|---:|---:|---:|
| fwd | 1.7428 | 1.5582 | 4.750 |
| rev | 1.7858 | 1.6280 | 5.094 |
| rev-res | 1.7681 | 1.6007 | 4.956 |

fwd is ahead at every checkpoint after step 0. Train loss by depth at step
2000 shows where the gap opens:

| depth | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fwd | 2.469 | 1.993 | 1.711 | 1.596 | 1.553 | 1.541 | 1.539 | 1.541 |
| rev | 2.471 | 1.996 | 1.729 | 1.641 | 1.615 | 1.610 | 1.611 | 1.613 |
| rev-res | 2.488 | 2.030 | 1.757 | 1.622 | 1.565 | 1.549 | 1.555 | 1.579 |

Node coherence (kept fraction) at depths 1 / 2 / 3:

| step | fwd | rev | rev-res |
|---:|---|---|---|
| 0 | 0.73 / 0.72 / 0.71 | 0.80 / 0.86 / 0.91 | 0.83 / 0.89 / 0.92 |
| 100 | 0.54 / 0.61 / 0.67 | 0.24 / 0.66 / 0.85 | 0.26 / 0.59 / 0.81 |
| 300 | 0.33 / 0.50 / 0.59 | 0.14 / 0.39 / 0.73 | 0.17 / 0.38 / 0.67 |
| 2000 | 0.26 / 0.41 / 0.50 | 0.34 / 0.27 / 0.52 | 0.43 / 0.35 / 0.53 |

At roughly matched train loss (fwd step 1000 at 1.778, rev step 2000 at
1.786, rev-res step 2000 at 1.768), node coherence is fwd 0.25 / 0.43 / 0.52,
rev 0.34 / 0.27 / 0.52 and rev-res 0.43 / 0.35 / 0.53. Unit coherence at the
same points (ρ global, ρ per-root, pd1) is fwd 0.06 / 0.28 / 0.20, rev
0.10 / 0.32 / 0.30 and rev-res 0.15 / 0.37 / 0.41. The unit numbers move
non-monotonically between checkpoints (fwd pd1 0.13, 0.20, 0.47 at steps
300, 1000, 2000), so read them as noisy.

## Reading

1. **The prediction held at initialization only.** At step 0 the suffix-tree
   sums keep more of their parts at every depth, and more the deeper the node
   (0.80 / 0.86 / 0.91 against a flat ~0.72). After training, a gain remains
   at depth 1, depth 2 is lower, and depth 3 ties.
2. **Under full-batch training the tree is bookkeeping.** The exact gradient
   of a given f_θ is the same whatever tree computes it (SGD-equivalence). So
   this run compares **reading directions** of the GRU, which is the todo's
   experiment 1 more than its experiment 2. The newest-first GRU loses at
   depth 3 and beyond. That is consistent with recency inversion: it must
   carry the newest character through every older step. The residual readout
   recovers about 40% of the held-out gap.
3. **Coherence of an exact sum is not lost gradient.** At a full-batch
   stationary point the summed gradient is zero by definition, so falling
   global coherence partly just means convergence. This also qualifies
   Experiment 1's reading ("cancellation grows with training"). Coherence
   matters for partitioned updates, where units step in sequence, not for
   the exact full-batch step.

## pd=1: one Adam step per root-child subtree

Run `20261003T062907-gru-d64-depth8-pd1-adam3e-3-500ep/`. Each step uses one of
the 65 root subtrees, with its event-mean gradient (as `bin/agpt_train_v2`
fires it). Unit order is reshuffled each epoch, with the same sequence for all
three models. The subtree gradients sum to the full gradient to 1e-16
(float64 check). The units differ by orientation: "contexts starting with c"
in the prefix trie, "contexts ending in c" in the suffix tree. The root masses
are the same.

Held-out NLL at equal epochs (pd=0 rows from the first run):

| epochs | fwd pd0 | fwd pd1 | rev pd0 | rev pd1 | rev-res pd0 | rev-res pd1 |
|---:|---:|---:|---:|---:|---:|---:|
| 30 | 2.708 | 1.797 | 3.174 | 2.213 | 2.845 | 2.562 |
| 100 | 2.222 | 1.677 | 2.435 | 2.009 | 2.203 | 2.326 |
| 300 | 1.844 | 1.640 | 2.001 | 1.902 | 1.820 | 2.214 |
| 500 | | 1.628 | | 1.838 | | 2.222 |
| 1000 | 1.607 | | 1.696 | | 1.642 | |
| 2000 | 1.558 | | 1.628 | | 1.601 | |

Train loss by depth at pd=1 epoch 500:

| depth | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fwd | 2.498 | 2.047 | 1.767 | 1.665 | 1.628 | 1.614 | 1.613 | 1.614 |
| rev | 2.500 | 2.079 | 1.882 | 1.837 | 1.825 | 1.825 | 1.827 | 1.831 |
| rev-res | 2.578 | 2.174 | 1.974 | 1.910 | 1.923 | 1.982 | 2.073 | 2.187 |

1. **Partitioning helps the prefix trie about twice as much.** At epoch 300,
   pd=1 improves held-out NLL by 0.20 for fwd and 0.10 for rev. With the
   residual readout it *hurts* after ~100 epochs: 2.214 against 1.820 at
   pd=0. Its loss rises with depth beyond 4, because the summed corrections
   compound.
2. **Why: each suffix-tree unit is one prediction problem.** Every node under
   root c predicts the character after c, so a pd=1 step is a target-sorted
   minibatch: its targets are skewed toward what follows c (mean KL to the
   corpus target distribution 0.86 nats, against 0.09 for prefix-trie units;
   see the grouping control below). It pulls the shared weights toward one
   unit's targets and the next step undoes it. (The 2026-10-03 version of this
   note said "one target class", which overstates it: the targets vary, but
   they are heavily skewed.) The depth-2 unit gradients show the structure.
   For rev, cosine within a root is 8–30× the cosine across roots from epoch
   10 on (for example 0.103 vs 0.013 at epoch 300). For fwd it is about 1.8×
   (0.015 vs 0.008). Grouping by the *least* predictive character is what
   makes prefix-trie units well mixed.
3. **This is a property of the tree, not the GRU.** Any f_θ that shares
   computation in the suffix tree gets these units. So an attention f_θ would
   face the same pd=1 penalty. With the GRU, the grouping control below puts
   the units' share of the penalty at about two thirds. Only an attention
   model's pd=0 behaviour (reading direction without recency inversion) is
   open.

pd=1 epochs are about 0.55 s in this Python tool against 0.23 s at pd=0
(per-unit overhead). The node work per epoch is the same.

## Grouping control: the units, not the reading direction (2026-11-10)

The pd=1 comparison above changes two things at once, the units and the GRU's
reading direction. `src/tools/agpt_unit_grouping_control.py` takes the tree
away. Every context in the same node table (1,495,879 nodes, same counts) is an
example whose GRU state is recomputed from its own characters, so any grouping
can be paired with either reading direction. Steps are as at pd=1: 65 groups
per epoch in a fresh random order (the orientation tool's RNG sequence), an
event-mean gradient, Adam at 3e-3, 500 epochs, seed 1.

| grouping | one step holds | which is | target skew |
|---|---|---|---:|
| first | the contexts with the same oldest char | a prefix-trie pd=1 unit | 0.0899 |
| last | the contexts with the same newest char | a suffix-tree pd=1 unit | 0.8606 |
| window | corpus windows, assigned at random each epoch | a shuffled minibatch | 0.0002 |
| start | windows grouped by a random label on their start position | prefix-unit clumping without the char sort | 0.0002 |

Target skew is the event-weighted mean over groups of KL(group's target
distribution ‖ corpus target distribution), in nats. The random groupings draw
group sizes from the prefix units' event shares, so step sizes match in
distribution. A window is one event: context `train[i:i+L]`, target
`train[i+L]`, 1 ≤ L ≤ 8.

**Checks.** The fwd/first and rev/last cells are the October pd=1 runs
recomputed without the tree. They match to four decimals through epoch 10
(`repro-fwd-first.log`, `repro-rev-last.log`; stopped there to save GPU time).
The event (window) code path, with each window labelled by its oldest char,
reproduces fwd/first exactly at epoch 3 (2.3147). A first random control
assigned whole nodes to groups. It was lumpy, because the 65 depth-1 nodes carry
an eighth of all events, so a small group could be a single heavy context
(skew 0.034). It was stopped at epoch 30 (`aborted-fwd-random-node/`), and the
window-level control replaced it.

Run: `20261110T094042-gru-d64-depth8-grouping-control-500ep/`, one
subdirectory per cell, each with `run.log` and `summary.json`. fwd/first and
rev/last at 500 epochs are the October run. fwd/first seed 2 is in
`fwd-first-seed2/`. The cosine runs are in
`20261110T102850-gru-d64-depth8-grouping-cosine-100ep/`. The tool was
uncommitted at launch on top of `a6f3c84` and is committed with this README.
The GPU was shared with another project's job, so the timings are not
comparable.

Held-out NLL (diagnostic, as above):

| epochs | fwd/first | fwd/first s2 | fwd/window | fwd/start | fwd/last | rev/first | rev/last |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 2.3147 | 2.2967 | 2.1829 | 2.1835 | 2.7774 | 2.5526 | 3.1216 |
| 10 | 2.0165 | 2.0139 | 1.9627 | 1.9601 | 2.3290 | 2.1970 | 2.5254 |
| 30 | 1.7967 | 1.7930 | 1.8013 | 1.7890 | 2.0977 | 1.9187 | 2.2127 |
| 100 | 1.6771 | 1.6824 | 1.6936 | 1.6915 | 1.8896 | 1.7616 | 2.0092 |
| 300 | 1.6402 | 1.6382 | 1.7380 | 1.6311 | 1.8135 | 1.7010 | 1.9020 |
| 500 | 1.6278 | 1.6320 | 1.6656 | 1.6322 | 1.7587 | 1.6812 | 1.8383 |
| PPL @500 | 5.093 | 5.114 | 5.289 | 5.115 | 5.805 | 5.372 | 6.286 |

fwd = oldest-first GRU (the prefix trie's), rev = newest-first (the suffix
tree's). fwd/first and rev/last at 500 epochs are the October runs.

Train loss by depth at epoch 500:

| depth | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fwd/first | 2.498 | 2.047 | 1.767 | 1.665 | 1.628 | 1.614 | 1.613 | 1.614 | 1.8058 |
| fwd/first s2 | 2.504 | 2.059 | 1.786 | 1.674 | 1.634 | 1.623 | 1.620 | 1.621 | 1.8151 |
| fwd/window | 2.500 | 2.055 | 1.789 | 1.693 | 1.665 | 1.658 | 1.658 | 1.659 | 1.8347 |
| fwd/start | 2.494 | 2.034 | 1.761 | 1.657 | 1.626 | 1.619 | 1.618 | 1.619 | 1.8035 |
| fwd/last | 2.496 | 2.082 | 1.856 | 1.774 | 1.752 | 1.748 | 1.747 | 1.748 | 1.9003 |
| rev/first | 2.498 | 2.064 | 1.770 | 1.687 | 1.670 | 1.666 | 1.669 | 1.673 | 1.8371 |
| rev/last | 2.500 | 2.079 | 1.882 | 1.837 | 1.825 | 1.825 | 1.827 | 1.831 | 1.9508 |

With a cosine schedule (2-epoch warmup to 3e-3, cosine to 0 at epoch 100;
`20261110T102850-gru-d64-depth8-grouping-cosine-100ep/`), oldest-first GRU:

| epochs | first: train | first: held-out | window: train | window: held-out | last: train | last: held-out |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 2.1315 | 2.0476 | 2.0842 | 1.9904 | 2.3792 | 2.3322 |
| 30 | 1.9348 | 1.8061 | 1.9372 | 1.8116 | 2.1389 | 2.0620 |
| 50 | 1.8820 | 1.7382 | 1.8892 | 1.7546 | 2.0525 | 1.9529 |
| 100 | 1.8547 | 1.7038 | 1.8539 | 1.7137 | 1.9933 | 1.8797 |

1. **The grouping causes most of the suffix tree's pd=1 penalty.** At 500
   epochs suffix grouping costs 0.131 nats with the oldest-first GRU (1.759
   against 1.628) and 0.157 with the newest-first GRU (1.838 against 1.681).
   Reading newest-first costs 0.053 with prefix grouping and 0.080 with suffix
   grouping. Of the October gap (0.211), the units account for about 0.14 and
   the reading direction for about 0.07, and the two compound slightly. With
   either GRU, suffix grouping raises the train loss at every depth from 3 on,
   by 0.09–0.16.
2. **Prefix-trie units tie shuffled minibatches.** With equal numbers of
   equal-sized steps, train loss agrees to 0.001 at epoch 100 under both
   schedules: 1.8446 against 1.8455 constant, 1.8547 against 1.8539 cosine.
   The held-out differences are about 0.01 and change sign:
   - at epoch 100, prefix units lead (1.677 and 1.682 against 1.694 and 1.692;
     cosine 1.704 against 1.714);
   - at epochs 300 and 500, start-labelled random batches tie or lead (1.631
     against 1.640 and 1.638; 1.632 against 1.628 and 1.632).

   The prefix units' leftover skew (0.09 against 0.0002) costs nothing
   measurable at this scale. Shuffled batches are ahead only in the first ~10
   epochs (2.183 against 2.315 at epoch 3).
3. **Clumping by start position costs nothing.** Keeping together all the
   windows that start at one position (start) tracks fully shuffled windows
   (window) to within about 0.01 through epoch 100. So neither part of a prefix
   unit's structure costs anything measurable: neither the clumping by start
   position nor the sort by oldest character.
4. **Shuffled windows at a constant 3e-3 hit an instability.** The window
   cell's train loss rose after epoch 100 (1.846, then 1.893 at 300) and only
   partly recovered (1.835 at 500, held-out 1.666). Start-labelled random
   batches at the same LR did not do this (train 1.799 at 300, 1.804 at 500).
   Both have near-zero skew, and each has one seed, so this is unexplained.
   The cosine runs avoid it, and the 500-epoch window number should not be
   read as a property of shuffled batches.

## Proviso: double-check before building on this

The user called this strand on 2026-11-10 as a minor breakthrough, with a
proviso to double-check and polish it. These are the checks, roughly in order
of how much each could change the reading:

1. **Seeds.** Only fwd/first has two seeds (5.093 and 5.114). The random
   cells, the 2×2 cells and the cosine runs have one each.
2. **An attention f_θ.** The grouping belongs to the tree, so any f_θ trained
   on suffix-tree units gets target-sorted steps. How much that costs an
   attention model is unmeasured. A cheaper test than a suffix-tree trainer
   is this control with an attention f_θ over the window: the grouping needs
   no tree. The v2 trainer would also need suffix-tree targets (todo, "Trainer
   targets").
3. **The CUDA trainers' unit order.** v2 visits units in the same order every
   epoch unless `experimental.unit_order_seed` is set
   (`src/cudax/yaml_config_v2.cuh:22`, default −1). These Python runs
   reshuffle every epoch. v1's `--shuffle-order` gained about 2% at pd>1
   (`rnd/cap-folding`), which fits this result: a fixed order is a mild sort.
   Measure it on v2 at pd=1.
4. **Depth 16, pd=2/3, and a longer warmup-cosine LR.** Everything here is
   depth 8, and mostly at a constant 3e-3. The cosine check runs only 100
   epochs. At pd=2, prefix units share their two oldest characters,
   which still say little about the target. Suffix units would share the two
   newest, which say a lot.
5. **Canonical eval.** Every number here is the diagnostic held-out NLL.
   Anything cited elsewhere needs `bin/agpt_experiment`.
6. **Interference between unit steps** (todo, step 1) was not measured. With
   this control it is less needed.
