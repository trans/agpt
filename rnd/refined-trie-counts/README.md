---
title: Refined trie counts
kind: experiment
status: concluded
outcome: positive
question: >-
  Can AGPT keep a small corpus's trie (its nodes, so its cost) and take the trie's child counts
  from a larger corpus that contains it, as mobius refines a small corpus's count table with a
  larger corpus's counts, and how does that compare with training on the larger corpus's own
  trie or on its text?
answer: >-
  Yes. AGPT on a 1M-character PG-19 slice's trie, with each context's next-character
  distribution counted over the 20M or 100M text that contains the slice, beats the same trie
  with its own counts by 0.051-0.057 nats of canonical rolling byte PPL after 8 passes (three
  pairs, two order seeds), at the same steps and nodes. The 4.75M text's counts give nothing.
  It is a better objective, not a speed-up: in minibatch training the 100M counts add 0.058
  nats over the slice's own counts as soft targets, and the slice with the 100M counts beats
  minibatch on the 4.75M text itself. Keeping the small trie beats growing it: AGPT on the
  4.75M text's own trie trails the slice's. The gain holds at plan depths 0, 1 and 6 too
  (0.033-0.073 nats on five more pairs), full-batch training included. AGPT's schedules still
  trail minibatch: the best here, full-batch pd=0 with the 100M counts, reaches 6.62 after 512
  passes, where minibatch with them reaches 6.50 after one epoch.
opened: 2026-11-18
updated: 2026-11-18
code: main
eval: canonical
family: attention
headline:
- label: 'AGPT pd=3 replay, the 1M slice''s trie with the 100M text''s counts, 8 passes, order seed 1'
  metric: rolling byte PPL (canonical, the 50 PG-19 held-out chunks)
  value: 9.4551
  run: 20261118T051418-agpt-a1m-ref100m
- label: 'Same steps and trie, the slice''s own counts (baseline)'
  metric: rolling byte PPL (canonical, the 50 PG-19 held-out chunks)
  value: 10.0056
  run: 20261118T045845-agpt-a1m
- label: 'Minibatch, the 1M slice with the 100M text''s counts as soft targets, 8 epochs'
  metric: rolling byte PPL (canonical, the 50 PG-19 held-out chunks)
  value: 6.1502
  run: 20261118T053638-mb-a1m-ref100m
tags: [trie-structure, count-prior, data, pg19, scaling]
related: [prior-scaling-plain-1g, count-backoff-gate, one-pass-gap, context-tree-orientation]
---

# Refined trie counts

The user, 2026-11-17: most of the useful data lives in the shallow tree, and
mobius uses a larger corpus to refine the counts of a smaller one's tree;
could AGPT use that?

## Why it fits AGPT

AGPT's loss reads the text only through the trie's counts n(c, a), and its
cost is one evaluation per trie node however often the node occurs. So the
counts can come from a larger corpus at no cost per step: the trie keeps its
nodes, and each node's targets get the larger corpus's evidence.

The shallow trie is where this pays. Counted on the first 32M characters of
mobius's PG-19 text (`imm65-200m.txt`), distinct contexts grow much more
slowly than the corpus up to depth 8, and deep contexts mostly occur once
(sharing = positions × depth / nodes, the events each node evaluation covers):

| chars | nodes, depth ≤ 8 | sharing | nodes, depth ≤ 16 | sharing |
|---:|---:|---:|---:|---:|
| 1M | 1.19M | 6.7× | 7.84M | 2.0× |
| 4M | 3.60M | 8.9× | 28.5M | 2.2× |
| 16M | 8.73M | 14.7× | 97.1M | 2.6× |
| 32M | 13.4M | 19.2× | 176.9M | 2.9× |

Up to depth 8 the nodes grow about as corpus^0.7, and the sharing is still
rising at 32M. Depth 16 stays near Shakespeare's 1.9×.

## Setup

- **Data:** PG-19 as mobius normalizes it (96 symbols), linked under
  `data/pg19/` (see `data/pg19/README.md`). A is the first 1,000,000
  characters (40 pieces of 25,000) of the 4.75M split's training text. The
  4.75M, 20M and 100M training texts each contain the one before. Held out:
  the 50 chunks of 5,000 characters every PG-19 split shares.
- **Model:** the d64/L2 attention model (h4, d_ff 256) from one random init
  (seed 42), contexts of up to 8 characters, Adam at a constant 0.003.
- **AGPT arms** replay v2's pd=3 + crown plan through
  `src/tools/agpt_minibatch_train.py` (`trie_plan pd`): 19,999 steps a pass
  on A, a new order each pass. With `trie_targets_corpus` each event's
  target is its context's next-token distribution over the larger text. A
  plan step holds every event of each of its contexts, so with the corpus's
  own counts the soft-target steps are the hard-target steps (checked: the
  first steps agree to 1e-6, and the trajectories then drift apart at the
  rate two hard-target runs on 1 and 2 CPU threads do).
- **Trie sizes up to depth 8:** A has 1,539,197 nodes (8.0M events, 5.2×
  sharing; 1,032,007 contexts seen once). Every one of A's contexts occurs in
  the larger texts; the median count per context is 2 in the 4.75M text and
  7 in the 20M. The 4.75M text's own trie has 4,538,285 nodes (38.0M
  events, 8.4×), 2.95× A's.

| run | trie / text | counts | schedule | passes |
|---|---|---|---|---:|
| agpt-a1m (+ s2) | A | A | AGPT pd=3 | 8 |
| agpt-a1m-ref5m | A | 4.75M | AGPT pd=3 | 8 |
| agpt-a1m-ref20m | A | 20M | AGPT pd=3 | 8 |
| agpt-a1m-ref100m (+ s2) | A | 100M | AGPT pd=3 | 8 |
| agpt-5m | 4.75M | 4.75M | AGPT pd=3 | 3 |
| mb-a1m | A | A | minibatch, 128 windows | 8 |
| mb-a1m-tt | A | A, as soft targets | minibatch, 128 windows | 8 |
| mb-a1m-ref100m | A | 100M | minibatch, 128 windows | 8 |
| mb-5m | 4.75M | 4.75M | minibatch, 128 windows | 2 |

Configs: `make_configs.py`; runs: `run.sh` (`mb-a1m-tt` was added after the
grid, see Results). Each context keeps its weight in A (one event per
occurrence in A); only its next-character distribution comes from the larger
text. Weighting contexts by their count in the larger text was not tried.

## Results

Slot 86, 2026-11-18, 04:58–06:12 UTC: the 10 runs of `run.sh`, then
`mb-a1m-tt`, added once the minibatch pair was in (it separates the soft
targets' own effect from the larger text's counts). Canonical eval on the 50
held-out chunks (`multi-chunk-heldout`). Tables: `summarize.py`. The train
losses and step counts come from the runs' `train.log`, which is not
committed, and are kept in `runs.json`.

Rolling byte PPL (fixed-window PPL in brackets) by pass:

| run | 1 | 2 | 3 | 4 | 8 |
|---|---:|---:|---:|---:|---:|
| agpt-a1m | 12.407 (11.213) | 11.776 (10.451) |  | 10.752 (9.202) | 10.006 (8.327) |
| agpt-a1m-s2 | 13.624 (12.562) | 12.410 (11.383) |  | 11.291 (10.038) | 10.683 (9.033) |
| agpt-a1m-ref5m | 12.996 (11.505) | 12.340 (10.401) |  | 10.543 (9.284) | 10.052 (8.406) |
| agpt-a1m-ref20m | 12.209 (10.998) | 11.547 (9.746) |  | 9.884 (8.646) | 9.469 (7.872) |
| agpt-a1m-ref100m | 12.682 (11.617) | 11.727 (10.099) |  | 10.148 (8.678) | **9.455** (7.799) |
| agpt-a1m-ref100m-s2 | 12.458 (11.286) | 11.218 (10.505) |  | 10.627 (9.694) | 10.155 (8.332) |
| agpt-5m | 12.186 (10.756) | 11.667 (9.754) | 11.146 (9.110) |  |  |
| mb-a1m | 6.990 (5.855) | 6.786 (5.615) |  | 6.655 (5.470) | 6.571 (5.368) |
| mb-a1m-tt | 6.715 (5.552) | 6.592 (5.409) |  | 6.518 (5.309) | 6.516 (5.304) |
| mb-a1m-ref100m | 6.503 (5.315) | 6.349 (5.138) |  | 6.233 (4.992) | **6.150** (4.901) |
| mb-5m | 6.395 (5.189) | 6.314 (5.081) |  |  |  |

Paired gaps in nats of rolling byte PPL (negative = the first run is better;
AGPT pairs share their order seed):

| run | against | 1 | 2 | 4 | 8 |
|---|---|---:|---:|---:|---:|
| agpt-a1m-ref5m | agpt-a1m | +0.046 | +0.047 | −0.020 | +0.005 |
| agpt-a1m-ref20m | agpt-a1m | −0.016 | −0.020 | −0.084 | −0.055 |
| agpt-a1m-ref100m | agpt-a1m | +0.022 | −0.004 | −0.058 | −0.057 |
| agpt-a1m-ref100m-s2 | agpt-a1m-s2 | −0.089 | −0.101 | −0.061 | −0.051 |
| mb-a1m-tt | mb-a1m | −0.040 | −0.029 | −0.021 | −0.008 |
| mb-a1m-ref100m | mb-a1m-tt | −0.032 | −0.038 | −0.045 | −0.058 |
| mb-a1m-ref100m | mb-a1m | −0.072 | −0.067 | −0.066 | −0.066 |

Steps a pass: 19,999 for the AGPT arms on A (1,576,699 trie positions),
32,236 for `agpt-5m` (4,599,493 positions; units average 1,179 events against
A's 400), 7,813 for minibatch on A, 37,110 for `mb-5m`.

### Reading

- **AGPT can use a larger text's counts, and they pay.** On A's trie, at A's
  steps, the counts of the 20M or 100M text put AGPT 0.051–0.057 nats ahead
  of A's own counts after 8 passes on all three pairs, and 0.058–0.084 ahead
  at pass 4. On seed 1 the 100M arm at pass 4 (10.148) is where the baseline
  gets at about pass 7. On seed 2 it is ahead at pass 4 of the baseline at
  pass 8.
- **The counts need enough text.** The 4.75M text gives nothing (+0.005 at
  pass 8). In it, A's contexts occur a median of 2 times, and 602,160 of A's
  1,539,197 contexts (39%) still occur only once; at 20M 347,009 (23%), at
  100M 185,231 (12%). The 20M and 100M texts give the same gain at pass 8,
  so at depth 8, for this model, the useful evidence is in by 20× A.
- **It is a better objective, not a speed-up.** In AGPT's steps a
  corpus's own counts are exactly the hard targets (a step holds every event
  of each of its contexts), so the AGPT gain is the change of counts and
  nothing else. In minibatch training soft targets also cut gradient noise.
  A's own counts as soft targets (`mb-a1m-tt`) are worth 0.040 nats after
  one epoch, fading to 0.008 by eight as both runs approach the same optimum
  (one-pass-gap's follow-up 3 again, larger here because depth-8 contexts
  repeat more). On top of that the 100M counts add 0.032 nats after one
  epoch, growing to 0.058 by eight, the size of AGPT's gain. The train
  losses show the change of objective directly. Measured on A's own one-hot
  targets after 8 epochs, `mb-a1m-tt` fits A best (1.702 nats against the
  100M arm's 1.729) and generalizes worse (6.516 against 6.150).
- **For AGPT, keeping the small trie beats growing it.** AGPT on the 4.75M
  text's own trie (2.95× A's nodes) ends at 11.146 after 3 passes, 96,708
  steps. A's own trie is ahead of that by pass 4, 79,996 steps (10.752 and
  11.291 on the two seeds), and A with the 100M counts more so (10.148,
  10.627). The bigger trie's units are bigger, so each event gets fewer
  steps: one-pass-gap's "more sharing kept = worse". Refined counts bring
  the larger text's evidence without its nodes.
- **In minibatch training too, the counts beat more text.** The 1M slice
  with the 100M counts (6.150 after 62,504 steps) beats 2 epochs of the
  4.75M text itself (6.314 after 74,220 steps) by 0.026 nats. Not tested:
  minibatch on the 20M or 100M text for the same steps (the trainer
  budgets in whole epochs only).
- **AGPT's schedule is still the gap.** Minibatch is far ahead at every
  point (6.15–6.57 against 9.46–10.68 after 8 passes). The refined counts
  move both by about 0.06 nats and leave the one-pass gap where it was.
- **Seeds.** The baseline's two order seeds differ by 0.065 nats at pass 8,
  as much as the effect. The order seed moves both arms of a pair together,
  and the paired gaps on the two seeds agree within 0.006 at passes 4 and 8.
  Gaps at passes 1–2 are noisy (+0.022 and −0.089 for the same arms on the
  two seeds).
- **Cost.** Counting A's contexts in the larger text took 7 s (4.75M),
  11 s (20M) and 57 s (100M; 105 s with two other runs on the machine), and
  each trainer stayed under 1.4 GB resident. The 100M table holds 8.9M
  (context, next) pairs, against 2.2M for A's own counts. In this Python
  replay, soft-target steps take about 1.5× as long as one-hot ones (dense
  rows of 96), and in minibatch about 1.15×. In v2 a node's loss already
  sums over its children's counts, so other counts would change the
  numbers in a node's target, not its cost.

## Follow-up: plan depths

The user, 2026-11-18, asked for pd=1 and pd=0 as well; pd=6 is the deepest
plan on record and its best score (`partition-depth`, legacy eval). Same
pair, A's own counts against the 100M text's, at three more plan depths:

- **pd=0:** the whole trie is one unit, so one step a pass is the full-batch
  gradient. `trie_plan.pd_unit_tags` gained k = 0 for it (`0485dce`); on the
  CPU it reproduces minibatch with every window in one batch (the same train
  losses, weights within 8.8e-7). 512 passes, so 512 Adam steps; one run
  each, since a single step a pass has no order.
- **pd=1:** 89 units a pass (median 7,792 events). 128 passes, two order
  seeds.
- **pd=6:** 379,312 units a pass (median 5 events). 2 passes, at the
  record's pd=6 rate (Adam 0.001).

Slot 88, 2026-11-18, 10:13–13:35 UTC, three streams on the GPU
(`run2.sh`, log `grid2.out`). Canonical eval as above.

Rolling byte PPL (fixed-window PPL in brackets) by pass:

| run | 1 | 2 | 8 | 32 | 64 | 128 | 256 | 512 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| agpt-a1m-pd0 |  |  |  | 14.018 (13.975) |  | 8.502 (7.669) | 7.465 (6.478) | 6.841 (5.690) |
| agpt-a1m-pd0-ref100m |  |  |  | 14.132 (14.140) |  | 8.553 (7.743) | 7.358 (6.330) | **6.621** (5.488) |
| agpt-a1m-pd1 |  |  | 12.297 (11.775) | 9.644 (8.878) | 8.873 (7.986) | 8.364 (7.484) |  |  |
| agpt-a1m-pd1-ref100m |  |  | 12.116 (11.484) | 9.477 (8.697) | 8.604 (7.589) | 7.772 (6.768) |  |  |
| agpt-a1m-pd1-s2 |  |  | 11.892 (11.416) | 9.799 (9.095) | 8.585 (7.668) | 7.929 (6.925) |  |  |
| agpt-a1m-pd1-ref100m-s2 |  |  | 11.933 (11.377) | 9.538 (8.777) | 8.386 (7.442) | 7.675 (6.645) |  |  |
| agpt-a1m-pd6 | 9.688 (7.745) | 9.459 (7.172) |  |  |  |  |  |  |
| agpt-a1m-pd6-ref100m | 9.086 (7.141) | 8.982 (6.613) |  |  |  |  |  |  |

Paired gaps, 100M counts against own (nats; negative = the 100M counts
better):

| plan | order seed | 1 | 2 | 8 | 32 | 64 | 128 | 256 | 512 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pd=0 | – |  |  |  | +0.008 |  | +0.006 | −0.014 | −0.033 |
| pd=1 | 1 |  |  | −0.015 | −0.017 | −0.031 | −0.073 |  |  |
| pd=1 | 2 |  |  | +0.003 | −0.027 | −0.023 | −0.033 |  |  |
| pd=6 | 1 | −0.064 | −0.052 |  |  |  |  |  |  |

At the last pass of every plan depth:

| plan | steps a pass | passes | own counts | 100M counts | gap |
|---|---:|---:|---:|---:|---:|
| pd=0 | 1 | 512 | 6.841 | 6.621 | −0.033 |
| pd=1, seed 1 / 2 | 89 | 128 | 8.364 / 7.929 | 7.772 / 7.675 | −0.073 / −0.033 |
| pd=3, seed 1 / 2 | 19,999 | 8 | 10.006 / 10.683 | 9.455 / 10.155 | −0.057 / −0.051 |
| pd=6 | 379,312 | 2 | 9.459 | 8.982 | −0.052 |

### Reading

- **The gain holds at every plan depth.** At the last pass the 100M counts
  are 0.033–0.073 nats ahead on all five new pairs, around pd=3's 0.054.
  The plan sets which events share a step, not the objective, so a change
  of objective should show at every depth, and it does. pd=0 is the
  cleanest case: the exact full gradient, no order and no step noise, and
  the 100M counts still win.
- **When it appears differs.** In full-batch training the gain is absent
  through 128 steps (+0.006) and appears from 256 on, still growing at 512.
  pd=1 shows it from pass 32 (2,848 steps) and pd=6 after one pass
  (−0.064). One reading, not tested: the 100M counts change mostly the rare
  deep contexts' distributions, which full-batch GD, weighting contexts by
  their events, reaches only late; pd=6's small units fit a few deep
  contexts per step from the start.
- **Plan depth, at equal passes:** the deeper the plan, the further ahead
  early. pd=6 after 2 passes (9.459) is ahead of pd=3 after 8 (10.006 and
  10.683), and with the 100M counts (8.982) of pd=3 with them (9.455). This
  agrees with the record's pd=6 best.
- **Plan depth, at equal steps:** the order reverses. pd=0's 512 steps reach
  6.841; pd=1's 11,392 reach 7.93–8.36; pd=6's 758,624 reach 9.459. pd=0 with
  the 100M counts (6.621) is the best AGPT result here, and it was still
  falling (0.09 and 0.11 nats from pass 256 to 512).
- **Minibatch is still ahead.** Minibatch with the same 100M counts reaches
  6.503 after one epoch, and without them reaches pd=0's 6.841 after about
  1.7 epochs (13.8M positions). pd=0 needed 512 passes for it: 788M node
  evaluations even counting only the trie's 1.54M nodes a pass, as v2 would
  (this replay evaluates every window, 8.0M positions a pass), about 57×
  minibatch's compute.
- **Cost here.** pd=0 took 8–9 s a pass (steps chunked into 131,072
  tokens), pd=1 8–12 s, pd=6 about 30 minutes a pass with own counts and 99
  minutes with the 100M counts (11,890 s against 3,624 s for 2 passes, both
  sharing the GPU with the other streams). At pd=6's tiny steps the Python
  replay's per-step costs dominate, the soft-target build among them.

### Next

- Refined counts in v2 (the CUDA trainer): the trie's child counts from a
  reference text. This is the cheap way to get the result into AGPT proper.
- Deeper tries: at depth 16 most of A's contexts are rare even at 100M, so
  the gain should shrink with depth; worth measuring before scaling up.
- Minibatch streaming the larger text for the same steps, the control the
  epoch-only budget left out.
