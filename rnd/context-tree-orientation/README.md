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
  checked with attention on 2026-11-17). A pd=1 step uses one root-child subtree. Prefix-trie
  units group contexts by their oldest, least predictive character, so their targets are well
  mixed. Suffix-tree units group by the newest, most predictive character, so each step is a
  target-sorted batch (skew 0.86 nats at any depth, the mutual information of adjacent
  characters). With the tree removed and the d64/L2 attention model at depth 16, suffix-tree
  units trail prefix-trie units by 0.91 nats of canonical rolling byte PPL after one pass and
  0.59 after 16 passes of 65 steps; a GRU at depth 8 showed the same effect, smaller (diagnostic
  eval). Prefix units trail random windows in batches of their sizes by up to 0.28 nats in the
  first four passes and are level with them from pass 16.
opened: 2026-10-03
updated: 2026-11-17
code: main
eval: canonical
family: attention
headline:
- label: 'Prefix-trie pd=1 units (first), d64/L2 attention, depth 16, 16 passes, Adam 0.003 constant'
  metric: rolling byte PPL (canonical, carved split's held-out chunks)
  value: 6.719
- label: 'Same steps, suffix-tree pd=1 units (last), best rate 0.0015'
  metric: rolling byte PPL (canonical, carved split's held-out chunks)
  value: 12.131
- label: 'Same steps, random windows in batches of the prefix units'' sizes (window)'
  metric: rolling byte PPL (canonical, carved split's held-out chunks)
  value: 6.863
tags: [trie-structure, recurrence, attention, coherence, partitioning, minor-breakthrough]
related: [gradient-population, linear-recurrence, count-backoff-gate, slot-selection-step0, one-pass-gap]
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
once training is past its first passes, stepping through them behaves like
stepping through shuffled minibatches. (Before that the leftover mixing costs
something, for both models tested: see the attention control below.) The
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
- **For the GRU, prefix-trie units become as good as shuffled minibatches.**
  With equal numbers of equal-sized steps they agree on train loss to 0.001 at
  epoch 100, under a constant and a cosine schedule. Held-out differences are
  about 0.01 and change sign with the epoch: 1.628 and 1.632 (two seeds)
  against 1.632 for start-labelled random batches at 500 epochs; 1.704 against
  1.714 for shuffled windows under cosine at 100. Early on the shuffled
  batches lead: by 0.13 nats at epoch 3 (2.315 against 2.183) and 0.05 at
  epoch 10. They tie from epoch 30.

The user suspected the prefix units' remaining correlation (skew 0.09 against
~0) explained part of AGPT's gap to vanilla minibatch training
(`rnd/vanilla-attn-torch`). At an equal number of equal-sized steps, this
control finds no such cost after epoch 30, so that gap more likely comes from
elsewhere. The plainest candidate is step count: vanilla training takes many
more, smaller steps per epoch. Their reading (2026-11-10) is that getting
further needs something past the tree, though they suspect something is still
being missed.

**Later evidence from attention (2026-11-14/15, `rnd/one-pass-gap`).** The
d64/L2 attention model, after one pass at pd=1's 65 steps, does pay for the
prefix grouping. AGPT's prefix units score rolling byte PPL 15.73, against
13.54 for random windows in batches of the same sizes: 0.150 nats (follow-up
6). At pd=3, grouping by the three oldest characters costs 0.21 nats (7.81
against 6.33, follow-up 1). The GRU paid the same way in its first epochs.
So the user's suspicion holds early in training, which is where one-pass
training lives, and for the GRU it fades with epochs.

**The attention control (2026-11-17) confirms both halves.** It removes the
tree and trains the d64/L2 attention model at depth 16 on the three
groupings, with canonical eval and two order seeds. Suffix-tree units trail
prefix-trie units by 0.91 nats after one pass and 0.59 after 16 (each arm at
its best rate), more than they cost the GRU. Prefix units trail random
windows in batches of their sizes by 0.09–0.28 nats in the first four
passes. From pass 16 the two are level, under a constant rate and under
cosine. So for attention too, the prefix trie is the right tree, and the
cost of its grouping is an early-training cost. Details: Attention grouping
control, below.

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
   open. (Measured 2026-11-17: the attention model pays 0.59–0.91 nats for
   the suffix units; see Attention grouping control.)

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
   measurable from about epoch 30. Before that, shuffled batches are ahead
   (2.183 against 2.315 at epoch 3, 0.05 nats at epoch 10). The attention
   control finds the same early cost, larger (its reading 2).
3. **Clumping by start position costs nothing.** Keeping together all the
   windows that start at one position (start) tracks fully shuffled windows
   (window) to within about 0.01 through epoch 100. So past the first epochs,
   neither part of a prefix unit's structure costs anything measurable:
   neither the clumping by start position nor the sort by oldest character.
4. **Shuffled windows at a constant 3e-3 hit an instability.** The window
   cell's train loss rose after epoch 100 (1.846, then 1.893 at 300) and only
   partly recovered (1.835 at 500, held-out 1.666). Start-labelled random
   batches at the same LR did not do this (train 1.799 at 300, 1.804 at 500).
   Both have near-zero skew, and each has one seed, so this is unexplained.
   The cosine runs avoid it, and the 500-epoch window number should not be
   read as a property of shuffled batches.

## Attention grouping control (2026-11-17)

Proviso check 2: the grouping control with an attention f_θ, at depth 16.
`src/tools/agpt_minibatch_train.py` with `experimental.trie_plan: group`
(commit `16f73e7`) trains the d64/L2 attention model of `rnd/one-pass-gap`
(RoPE, 4 heads, d_ff 256, the same seed weights) on exactly the events a
depth-16 trie holds: every window start of the training split, read past the
end from the front as the cached tries do, and every context length 1–16
(16.95M events). A pass is 65 steps, one per character, in a new random order
each pass. Each step takes the event-mean gradient of its events, and Adam
(β1 0.9, β2 0.999) runs at a constant rate, as in the GRU control. The
groupings are the GRU control's:

| arm | one step holds | which is | target skew (depth 16) |
|---|---|---|---:|
| first | the events whose context starts with one char | AGPT's prefix-trie pd=1 unit (the same partition as pd depth 1) | 0.0325 |
| last | the events whose context ends with one char | the suffix tree's pd=1 unit | 0.8606 |
| window | whole windows, assigned at random each pass to groups of first's sizes | a size-matched shuffled minibatch | 0.0001 |

With the wrapped tail, all three take steps of exactly the same sizes
(coefficient of variation 1.62, as AGPT pd=1). Checks on the CPU: at depth 8
the three skews are the GRU control's (0.0899 / 0.8606 / 0.0002); every
`last` step holds exactly its character's events; at lr 0 all three arms and
pd depth 1 give the same epoch loss.

Every run is canonical (`bin/agpt_experiment`, rolling byte PPL on the carved
split's 10 held-out chunks). Each is evaluated at passes 1, 2, 4, 8 and 16,
and also 32 for first and window. Rates 0.0015, 0.003 and 0.006, and a second
order seed (`trie_plan_seed` 2) at 0.003. Configs from
`attention/make_configs.py`, run by `attention/run.sh` (slot 66: four runs)
and `attention/run2.sh` (slot 68: the other eight); `attention/summarize.py`
makes the tables.

**What the suffix grouping costs to compute.** A model that reads oldest
first can share a step's computation only across contexts with a common
start. Grouping by the newest character breaks that. A `last` pass runs 147M
window tokens against 17M for `first` (117M when each window is cut at its
deepest event), and on a trie it would need 48.5M positions against the
prefix trie's 8.9M. A `last` pass took about 7 times as long.

### Results

Rolling byte PPL (fixed-window in brackets), constant rates. s2 is the second
order seed:

| arm | lr | seed | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| first | 0.0015 | 1 | 15.818 (16.007) | 12.665 (12.806) | 11.016 (10.870) | 8.317 (7.976) | 6.980 (6.533) | 5.952 (5.411) |
| first | 0.003 | 1 | 13.626 (13.741) | 11.833 (11.823) | 10.397 (10.034) | 8.072 (7.623) | 6.719 (6.229) | 5.853 (5.269) |
| first | 0.003 | s2 | 13.813 (13.820) | 11.335 (11.278) | 9.506 (9.222) | 8.142 (7.764) | 6.761 (6.265) | 6.270 (5.719) |
| first | 0.006 | 1 | 12.899 (12.993) | 11.888 (11.800) | 10.950 (10.573) | 8.274 (7.915) | 7.080 (6.596) | 6.062 (5.456) |
| window | 0.0015 | 1 | 12.889 (12.977) | 10.179 (10.088) | 8.898 (8.688) | 7.654 (7.292) | 6.973 (6.501) | 6.216 (5.725) |
| window | 0.003 | 1 | 12.046 (12.021) | 9.389 (9.195) | 8.575 (8.244) | 7.605 (7.185) | 6.863 (6.342) | 6.338 (5.873) |
| window | 0.003 | s2 | 11.950 (11.881) | 9.949 (9.798) | 8.359 (8.068) | 7.680 (7.236) | 6.652 (6.191) | 6.274 (5.789) |
| window | 0.006 | 1 | 11.740 (11.633) | 8.979 (8.701) | 8.779 (8.445) | 7.664 (7.263) | 7.454 (7.076) | 6.517 (6.076) |
| last | 0.0015 | 1 | 35.738 (36.124) | 28.043 (28.240) | 20.406 (20.315) | 15.464 (15.303) | 12.131 (11.729) | |
| last | 0.003 | 1 | 32.107 (32.350) | 25.981 (26.136) | 20.304 (20.442) | 14.629 (14.495) | 13.160 (12.780) | |
| last | 0.003 | s2 | 38.472 (38.988) | 31.154 (31.102) | 24.253 (24.678) | 15.116 (14.726) | 12.886 (12.441) | |
| last | 0.006 | 1 | 33.843 (34.014) | 27.581 (28.095) | 23.418 (23.460) | 16.668 (16.635) | 14.133 (13.980) | |

The gaps in nats (log ratio of rolling byte PPL), at each arm's best rate for
seed 1 and as two-seed means at 0.003:

| pass | last − first, best | last − first, seeds | first − window, best | first − window, seeds |
|---:|---:|---:|---:|---:|
| 1 | +0.912 | +0.941 | +0.094 | +0.134 |
| 2 | +0.786 | +0.899 | +0.276 | +0.181 |
| 4 | +0.669 | +0.803 | +0.193 | +0.161 |
| 8 | +0.595 | +0.607 | +0.060 | +0.059 |
| 16 | +0.591 | +0.659 | −0.021 | −0.002 |
| 32 | | | −0.060 | −0.040 |

The two order seeds differ by up to 0.09 nats for first, 0.06 for window and
0.18 for last (in its first four passes).

### Reading

1. **The orientation result holds for attention, more strongly than for the
   GRU.** Suffix-tree units trail prefix-trie units by 0.91 nats after one
   pass and 0.59 after 16, each at its best rate (0.94 and 0.66 in the
   two-seed means). The GRU's gap was 0.46 nats at epoch 3, 0.31 at 10 and
   0.13 at 500 (diagnostic eval, so the sizes compare only roughly). The gap
   narrows with passes, but slowly: from pass 8 to 16 it barely moves.
2. **Prefix units against size-matched random windows: the GRU's pattern.**
   Random windows lead by 0.09–0.28 nats in the first four passes and by 0.06
   at pass 8. From pass 16 the two are level: −0.02 and −0.06 at the best
   rates, −0.00 and −0.04 in the seed means, inside the seeds' spread. The
   GRU led by 0.13 at epoch 3 and 0.05 at 10, and tied from 30. So the
   user's suspicion that the prefix units' leftover correlation costs
   something holds for attention too: early in training, where one-pass
   training sits, and fading with passes. AGPT pd=1's one-pass deficit
   against random batches of its unit sizes (0.150 nats, `rnd/one-pass-gap`
   follow-up 6) is this early cost.
3. **Rates.** first and window do best at 0.003 (0.006 after one pass; 0.0015
   for window at 32). last does best at 0.003 through pass 8 and at 0.0015 at
   pass 16, the bottom of the grid, so its best rate there may be lower still.
   0.0015 beat 0.003 there by 0.08 nats; the gap to first is 0.59.
4. **Train-loss rises follow the step order, not the rate.** Random windows'
   pass-mean train loss rises more often than the prefix units' (5–7 rises
   above 0.02 in each 32-pass run, against 0–3). Under order seed 1 the rises
   fall on the same passes (8, 11, 17, 21, 24) at all three rates, and under
   seed 2 on others. They do not show in held-out: window at 0.003 improves
   from 8.575 at pass 4 to 7.605 at pass 8, the pass of its largest rise. So
   this is not the instability the GRU's shuffled-window cell showed, whose
   held-out NLL got worse. The pass-mean train loss is taken during the pass,
   so with steps of unequal sizes it depends on where in the pass the big
   ones fall. Whether that explains the rises, or why random windows show
   more of them, was not checked.
5. **The cosine check agrees: level by pass 16.** first and window for 16
   passes under warmup-cosine (one pass of warmup, then cosine to 0), peak
   rates 0.003 and 0.006, evaluated at passes 4, 8 and 16:

   | arm | peak lr | 4 | 8 | 16 |
   |---|---:|---:|---:|---:|
   | first | 0.003 | 10.987 (10.728) | 8.161 (7.808) | 7.281 (6.906) |
   | first | 0.006 | 10.852 (10.478) | 8.063 (7.669) | 6.968 (6.544) |
   | window | 0.003 | 8.836 (8.629) | 7.361 (6.959) | 6.762 (6.311) |
   | window | 0.006 | 8.697 (8.370) | 7.795 (7.431) | 6.897 (6.442) |

   At each arm's best peak, window ends 0.030 nats ahead (6.762 against
   6.968). Under a constant rate first was 0.021 ahead at pass 16. Both are
   inside the seeds' spread. first's best peak is the top of this grid, so a
   higher peak might help it a little. Over 16 passes, cosine is worse than
   a constant rate for both arms here (first 6.968 against 6.719), so the
   constant-rate runs are the better-trained ones. Random windows' train loss
   rises under cosine too (2.144 at pass 4, 2.200 at pass 8, peak 0.006), so
   the rises in item 4 are not caused by the constant rate.

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

### Status (2026-11-17)

| # | Check | Status |
|---|---|---|
| 1 | Seeds | Attention: two order seeds per arm at 0.003. They differ by up to 0.09 nats (first), 0.06 (window) and 0.18 (last, early passes), against a suffix gap of 0.6–0.9 nats. GRU cells: still one seed each, except fwd/first. |
| 2 | Attention f_θ | **Done (2026-11-17), holds.** Suffix-tree units cost the d64/L2 attention model 0.91 nats after one pass and 0.59 after 16; prefix units are level with size-matched random windows from pass 16 (Attention grouping control, above). |
| 3 | v2's fixed unit order | Measured at pd=3 only (`rnd/one-pass-gap`, follow-up 1): radix-id order 8.00 against shuffled 7.81 after one pass, 0.024 nats. Not measured at pd=1. |
| 4 | Depth 16, pd=2/3, cosine | Depth 16: done with attention (above). pd=3 prefix units with attention trail size-matched random batches by 0.21 nats after one pass (`rnd/one-pass-gap`). Suffix units at pd=2/3 not run. Cosine: the attention control's 16-pass check (above). |
| 5 | Canonical eval | Done for the attention control (`bin/agpt_experiment`, rolling byte PPL). The GRU numbers remain diagnostic. |
| 6 | Interference between unit steps | Not measured. |

**Skew at depth 16** (`trie_plan group`, commit `16f73e7`, the full training
split with the trie's tail wrapping): prefix units 0.0325 nats, suffix units
0.8606, size-matched random windows 0.0001. At depth 8 the same code gives
the GRU control's 0.0899 / 0.8606 / 0.0002. The suffix units' skew does not
depend on depth. Each of their events pairs a context's newest character with
the character after it, so a unit's targets are always that character's
successor distribution, and the skew is the mutual information between
adjacent characters. A prefix unit's targets mix the characters 1 to T
places after its character, and the farther ones depend on it less, so
deeper tries mix their prefix units better.
