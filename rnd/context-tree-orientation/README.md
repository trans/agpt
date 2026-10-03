---
title: Context-tree orientation
kind: experiment
status: concluded
outcome: negative
question: >-
  Does organizing contexts newest-character-first (the suffix tree, each subtree one target's
  prediction problem) train a better model than the prefix trie, under full-batch (pd=0) or
  partitioned (pd=1) Adam?
answer: >-
  No, for a GRU at depth 8, and partitioning widens the gap. At pd=0 (2000 epochs) diagnostic
  held-out PPL is 4.750 prefix vs 5.094 suffix (4.956 with a residual readout); at pd=1 (500
  epochs) it is 5.093 vs 6.286 (9.222 residual). Suffix-tree pd=1 units each hold one target
  class, so sequential steps act like class-sorted minibatches; prefix-trie units, grouped by
  the least predictive character, are well mixed. Attention f_theta untested.
opened: 2026-10-03
updated: 2026-10-03
code: main
eval: legacy
family: recurrent
headline:
- label: 'pd=1, prefix trie, left-to-right GRU, d64 depth 8, 500 epochs'
  metric: diagnostic held-out PPL (context <= 8, carved split, not canonical)
  value: 5.093
- label: 'pd=1, suffix tree, newest-first GRU, same'
  metric: diagnostic held-out PPL (context <= 8, carved split, not canonical)
  value: 6.286
- label: 'pd=1, suffix tree, newest-first GRU, residual readout, same'
  metric: diagnostic held-out PPL (context <= 8, carved split, not canonical)
  value: 9.222
tags: [trie-structure, recurrence, coherence]
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
2. **Why: suffix-tree units are one target class each.** Every node under
   root c predicts the character after c, so a pd=1 step is a class-sorted
   minibatch. It pulls the shared weights toward one class and the next step
   undoes it. The depth-2 unit gradients show the structure. For rev, cosine
   within a root is 8–30× the cosine across roots from epoch 10 on (for example 0.103 vs
   0.013 at epoch 300). For fwd it is about 1.8× (0.015 vs 0.008). Grouping
   by the *least* predictive character is what makes prefix-trie units well
   mixed.
3. **This is a property of the tree, not the GRU.** Any f_θ that shares
   computation in the suffix tree gets these units. So an attention f_θ would
   face the same pd=1 penalty. Only its pd=0 behaviour (reading direction
   without recency inversion) is open.

pd=1 epochs are about 0.55 s in this Python tool against 0.23 s at pd=0
(per-unit overhead). The node work per epoch is the same.

## Not tested yet

- **An f_θ without recency inversion,** for example attention over the path,
  where reading order does not set what is remembered.
- **Longer training.** All three were still descending at step 2000.
