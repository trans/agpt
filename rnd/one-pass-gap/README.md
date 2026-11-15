---
title: The one-pass gap (AGPT pd=1–4 vs ordinary minibatch training)
kind: experiment
status: concluded
outcome: negative
question: >-
  AGPT's measuring stick is "one epoch and done". How close does AGPT get to a conventional
  training run in one pass over Tiny Shakespeare, and in a few more, against ordinary minibatch
  Adam training of the same d64/L2 model from the same weights? Is AGPT's deficit its update
  count: 65 per pass at pd=1, and more at pd=2–4 when the shallow targets are trained too?
  How does AGPT compare with conventional epochs, and how much of the trie's prefix sharing can
  a plan keep before the grouping it needs costs more than the sharing saves?
answer: >-
  Far off, and the update count is only part of it. After one pass AGPT pd=1 scores rolling byte
  PPL 15.73; minibatch training scores 4.83 in 31 s, already better than AGPT's 100-pass run, and
  4.66 after two passes, better than AGPT's best on record (4.764 after 625). More updates help
  AGPT a lot: one pass at pd=2 (1,463 updates) scores 9.65, pd=3 (12,675) 7.81 and pd=4 (58,721)
  6.74. But random equal batches with the same update counts score 12.56, 5.18 and 4.83, and pd=4
  takes 185 s a pass. At pd=3 the steps also match in average size; half the gap (0.27 of 0.48
  nats) is the units' unequal sizes (random windows in batches of pd=3's sizes score 6.33), and
  half is the trie's grouping of each step by a shared prefix. Weighting units by mass
  (fire_norm sum) is worse (8.53). A conventional epoch, each character a target once (518
  updates, 2 s), scores 6.94, better than a pd=3 pass, which computes 8.4 times as many
  positions; 8 conventional epochs, about a pd=3 pass's compute, score 5.12. In equal steps, the
  more of the trie's sharing a plan keeps, the worse it scores: 4.83 with 12% (random batches),
  5.39 with 65%, 5.86 with 88%, 6.34 with 97%. Keeping 65% costs 0.11 nats, more than all of the
  sharing could save at this context length (about 0.05 nats at equal compute). One seed; 124
  runs.
opened: 2026-11-14
updated: 2026-11-15
code: main
eval: canonical
family: attention
tags: [one-pass, baseline, cadence, optimizer, partition-depth]
related: [gradient-population, context-tree-orientation, vanilla-attn-torch, partition-depth]
---

# one-pass-gap

**Status:** concluded (2026-11-14). The user asked for the experiment
after we found the work had drifted off AGPT's main path. It is item 04 of
the site's next tests ("Measure the one-pass gap"), never done before. The
first grid was 48 runs in slot 30 (08:32–09:38 UTC). The follow-up, AGPT at
minibatch cadence, was 47 runs in slot 33 (17:50–18:54 UTC). Follow-up 2,
conventional epochs and AGPT's plans in the minibatch trainer, was 29 runs
in slot 39 (2026-11-15, 01:29–02:07 UTC).

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
2. **AGPT at minibatch cadence:** done, in the follow-up below.

## Follow-up: AGPT at minibatch cadence (2026-11-14)

The user chose next test 2 ("Sure do pd=3. The main issue is the number of
updates."). One pass of pd=1 is 65 updates. pd=3 cuts the trie into one
unit per depth-3 node, which gives more updates per pass than minibatch
training takes. 47 runs in slot 33 (17:50–18:54 UTC).

**The fix it needs.** A pd≥2 plan trains each anchor's subtree, with the
anchor's ancestors as context only, so the targets of every node whose
edge ends above depth pd were never trained. v2 now prints a
`plan-coverage` line. On the canonical trie, without the fix:

- pd=2 leaves 62 nodes untrained, 6.25% of the events;
- pd=3 leaves 1,194 nodes, 12.43%;
- pd=4 leaves 8,738 nodes, 18.41%.

The new option `experimental.train_crown` (`b1019b3`) makes each such
crown node a unit of its own, with its ancestors as context. With it,
every one of the 1,528,495 radix nodes is trained exactly once per pass at
pd=2, 3 and 4, as at pd=1 (checked on the CPU with `--mode plan`, and in
each run's `train.log`).

| pd | Anchored units | Crown units | Updates per pass |
|---:|---:|---:|---:|
| 1 | 65 | 0 | 65 |
| 2 | 1,401 | 62 | 1,463 |
| 3 | 11,481 | 1,194 | 12,675 |
| 4 | 49,983 | 8,738 | 58,721 |

**Arms.** Same model, weights, schedule and evaluation as above; configs
from `make_configs.py`:

- `pd2c`, `pd3c`, `pd4c`: AGPT at pd=2, 3 and 4 with crown units, in
  radix-id order (v2's default; crown units fall among the anchored ones).
- `pd3cs`: pd=3, shuffled each pass (`experimental.unit_order_seed: 1`),
  at 1, 2 and 4 passes.
- `mb724`, `mb84`: minibatch training with as many updates per pass as
  pd=2 (724 windows, 1,464 updates) and pd=3 (84 windows, 12,615
  updates). `mb84`'s batches hold 1,344 events, and pd=3's units 1,338 on
  average.
- `mbu3`: random windows in batches with pd=3's 12,675 unit sizes, in a
  new order each pass (`experimental.step_events_file`, `21ec506`). The
  sizes are each unit's trained events from the `pd3cs` lr 0.0015 log
  (`pd3_unit_events.txt`); a step takes max(1, round(events / 16))
  windows.
- `pd1s`, `pd3css`: pd=1, and shuffled pd=3, with `train.fire_norm: sum`
  (`e59b10c`). v2's default divides each unit's gradient by its own
  event count, so the updates of a pass add up to Σ_u mean_u(g) and weight
  every unit equally, whatever its size. `sum` divides every unit by one
  constant, so a pass weights every event equally, as the corpus loss
  does.

pd=3's best one-pass rate came out at 0.0015, the bottom of its first grid
(0.0015–0.012), so the grid was extended down to 0.000375 before the rest
ran. All one-pass bests below are inside their grids except `pd1s`.

### Results

One pass, best learning rate per arm, rolling byte PPL (fixed-window PPL
in brackets) and training seconds:

| Updates per pass | AGPT | Random equal batches |
|---:|---|---|
| 65 | pd=1: 15.73 (15.66), 8 s | 16,302 windows: 12.56 (12.50), 11 s |
| ~1,460 | pd=2: 9.65 (9.22), lr 0.003, 15 s | 724 windows: 5.18 (4.66), lr 0.006, 11 s |
| 8,279 | | 128 windows: **4.83** (4.29), 31 s |
| ~12,650 | pd=3 shuffled: 7.81 (7.20), lr 0.0015, 53 s; radix-id order: 8.00 (7.31), 54 s | 84 windows: **4.83** (4.29), lr 0.006, 45 s |
| 58,721 | pd=4: 6.74 (5.90), lr 0.00075, 185 s | |

What makes pd=3's steps weaker. All four arms take about 12,650 steps per
pass, with about 1,340 events per step on average:

| Arm | Step sizes | What a step holds | Rolling (fixed) |
|---|---|---|---:|
| `mb84` | equal, 84 windows | random windows | 4.83 (4.29) |
| `mbu3` | pd=3's unit sizes | random windows | 6.33 (5.87) |
| `pd3cs` | pd=3's unit sizes | one depth-3 subtree (or crown node) | 7.81 (7.20) |
| `pd3css` | pd=3's, summed, not averaged | one depth-3 subtree | 8.53 (8.16) |

More passes of shuffled pd=3: 7.81 after one pass, 6.98 after two (lr
0.0015, 108 s) and 6.29 after four (lr 0.0015, 216 s). Minibatch training
with 128 windows scores 4.83, 4.66 and 4.58 after the same numbers of
passes. With `fire_norm: sum`, pd=1 scores 23.37 after one pass against
15.73 (the four rates gave 23.4–28.0, not in order of rate). After eight
passes it scores 8.85 against 8.71.

### Reading

- **More updates help AGPT a lot, but they don't close the gap.** One pass
  falls from 15.7 at 65 updates to 9.65 at 1,463, 7.81 at 12,675 and 6.74
  at 58,721. Random equal batches reach 5.18 with 1,464 updates and 4.83
  from 8,279 on. AGPT is far behind at every matched count. pd=4, with 40
  times `mb724`'s updates, still trails it (6.74 against 5.18).
- **At pd=3 the gap is in what each step holds.** The step count is
  matched, and so is the average step size, so the gap of 0.48 nats
  (ln 7.81 − ln 4.83) comes from the steps' contents. It splits about in
  half:
  - **Unequal sizes cost 0.27 nats** (4.83 → 6.33, random windows in
    batches of pd=3's sizes). pd=3's median unit holds 168 events, 2,005
    units hold fewer than 24, and the largest holds 212,674, 1.25% of a
    pass. Under the event-mean rule, Adam gives every one of them a
    step of about the same size.
  - **The trie's grouping costs another 0.21 nats** (6.33 → 7.81), against
    random windows in batches of the same sizes. This is the cost of
    giving each step only contexts that share their first three
    characters. The context-tree orientation control found no such cost
    at pd=1 with a GRU (first-character groups against random groups of
    the same sizes); at pd=3 with attention the cost is there.
- **Weighting the units by their mass makes it worse.** `fire_norm: sum`
  scores 8.53 against 7.81 at pd=3. Under Adam, summed gradients give the
  few big units most of the movement and the thousands of small ones
  almost none, which cuts the number of updates that count. The size
  problem calls for units of equal size, not reweighted gradients.
- **Order matters little:** shuffled 7.81, radix-id order 8.00.
- **Deeper partitions also cost time.** With the exact ancestor gradient, a
  pd=3 pass takes 53 s and a pd=4 pass 185 s, against 31 s (128 windows)
  and 45 s (84 windows) for a minibatch pass. So at these depths AGPT is
  slower per pass as well as worse.
- **Answer to the user's point:** the number of updates is necessary. AGPT
  improves steadily with it, and at pd=1 it is most of the gap. But
  beyond about 1,500 updates per pass, the composition of each update
  matters more than how many there are.

### Next

1. **Equal-mass units:** done in follow-up 2 (the packed plans).
2. **The grouping cost against the compute saved:** done in follow-up 2,
   for three piece sizes.

## Follow-up 2: conventional epochs, and AGPT's plans in the minibatch trainer (2026-11-15)

Two questions the record had never asked. Every "epoch" above, in v2 and in
the minibatch trainer, is every window start with 16 targets each, so each
character is a target 16 times per pass. A conventional epoch makes each
character a target once. How does AGPT compare with that? And the trie
saves compute only where a step's contexts share prefixes, which means
grouping them. How much of the sharing can a plan keep before the grouping
costs more than the sharing saves? 29 runs in slot 39 (2026-11-15,
01:29–02:07 UTC), code `bf32854`.

**Arms.** Same model, seed weights, Adam, schedule and evaluation as above,
all in `src/tools/agpt_minibatch_train.py`.

- `cv`: conventional epochs (`experimental.window_stride: 16`): 66,226
  non-overlapping windows from a random offset each epoch, 1,059,616
  targets, 128 windows a step, 518 steps per epoch. Run for 1, 2, 4, 8 and
  16 epochs. Sixteen epochs compute as many positions as one pass of `mb`.
- `tp3`: AGPT's pd=3 + crown plan, replayed event by event
  (`experimental.trie_plan: pd`, new module `src/tools/trie_plan.py`). Each
  unit is one Adam step on its own events, picked out of ordinary windows
  by a loss mask, and the units are shuffled each pass. A unit's gradient
  is the sum of its events' gradients, so this gives the updates v2 gives,
  computed by a different program. It is the control for everything
  between v2 and the PyTorch trainer.
- `tq1338`, `tq335`, `tq84`: packed plans (`trie_plan: packed`), each with
  12,675 steps of equal size and 1,338 events per step on average, as at
  pd=3. A window's deep events go to a piece: the subtree of the shallowest
  prefix that holds at most P events (P = 1,338, 335 or 84). The heavier
  nodes' own events are split by occurrence into chunks of at most P.
  Pieces and chunks are packed at random into the steps, repacked each
  pass. The smaller P is, the shallower the shared prefixes and the less
  sharing is left.

**Trie positions per pass** count what a trie trainer would compute for a
plan: in each step, every distinct context its events read. The replay
trainer itself computes whole windows, so its times don't show the
sharing. Every pass has 16,953,872 events.

**Checks done before the run (CPU):**

- With no trie plan the trainer writes a byte-identical model to before.
- The pd=3 replay has v2's 12,675 units, and 99.5% of the sorted sizes are
  identical to v2's. The total differs by 272 events, from the trie's wrap
  at the corpus end.
- Every event is trained exactly once per pass in every plan.
- Windows cut at their deepest event give the full-window gradient to
  within 8e-16 (float64).

### Results

Conventional epochs, best rate 0.006 at every budget, rolling byte PPL
(fixed-window PPL in brackets) and the trainer's own training seconds:

| Epochs | Steps | Positions | Rolling (fixed) | Train s |
|---:|---:|---:|---:|---:|
| 1 | 518 | 1.06M | 6.940 (6.509) | 2 |
| 2 | 1,036 | 2.12M | 6.036 (5.558) | 4 |
| 4 | 2,072 | 4.24M | 5.476 (4.972) | 7 |
| 8 | 4,144 | 8.48M | 5.125 (4.587) | 15 |
| 16 | 8,288 | 16.95M | 4.854 (4.319) | 29 |

One pass of each plan, all with 12,675 steps of 1,338 events on average.
"Sharing kept" is the share of the trie's saving (16.95M events against
pd=3's 8.93M positions) the plan keeps:

| Plan | Step size CV | Positions per pass | Sharing kept | Rolling (fixed), best lr |
|---|---:|---:|---:|---:|
| `tp3`, pd=3 replay | 3.83 | 8.93M | 100% | 7.762 (7.179), 0.0015 |
| `tq1338` | 0.23 | 9.18M | 97% | 6.341 (5.836), 0.0015 |
| `tq335` | 0.06 | 9.90M | 88% | 5.864 (5.377), 0.003 |
| `tq84` | 0.017 | 11.71M | 65% | 5.387 (4.859), 0.003 |
| `mb84`, random windows (above) | 0 | 15.97M | 12% | 4.827 (4.290), 0.006 |

Median piece depth: 5, 6 and 7 characters for P = 1,338, 335 and 84.

### Reading

- **The control holds.** The pd=3 replay scores 7.76 against v2's 7.81, at
  the same best rate (0.0015; the neighbours are 7.94 and 8.32, against
  v2's 8.13 and 8.01). So the pd=3 gap is in the plan, not in v2.
- **One conventional epoch beats a pass of AGPT pd=3.** 518 steps and 1.06M
  positions (2 s) give 6.94. AGPT's pd=3 pass takes 24 times the steps and
  8.4 times the positions and gives 7.76–7.81. Only pd=4 (6.74, 185 s) is
  better than one conventional epoch.
- **At equal compute the gap is wide.** Eight conventional epochs compute
  about as many positions as a pd=3 pass (8.48M against 8.93M) and score
  5.12 against 7.81.
- **The every-start objective is worth what it costs, no more.** Sixteen
  conventional epochs score 4.85 and one pass over every window start
  4.83, with the same positions and nearly the same steps (8,288 against
  8,279). Training each target at all 16 context lengths is as good as
  seeing the corpus 16 times.
- **The more sharing a plan keeps, the worse it scores.** With equal steps
  and equal mean size, keeping 12% of the sharing gives 4.83, 65% 5.39,
  88% 5.86 and 97% 6.34.
- **Equal sizes fix the size half and leave the grouping half.** At equal
  sizes, grouping costs 0.27 nats (4.83 → 6.34). pd=3's unequal sizes on
  top of that grouping cost 0.20 more (6.34 → 7.76). In follow-up 1, in the
  other order, the two costs were 0.27 and 0.21 nats, so they roughly add.
  Unequal sizes without grouping (`mbu3`, 6.33) cost the same as grouping
  at equal sizes (`tq1338`, 6.34).
- **No point on the curve pays.** `tq84` computes 11.71M positions a pass
  in a trie trainer and scores 5.39. Conventional epochs with that many
  positions score about 5.0, interpolated between 8 and 16 epochs, and
  that is before the trie trainer's per-position cost (v2 is about 2.2×
  slower per position than the PyTorch trainer). At the other end, if the
  full sharing cost nothing in step quality, pd=3's 8.93M positions would
  buy random-batch quality (4.83) where conventional training at that
  compute scores about 5.10. That is a gain of about 0.05 nats. Keeping
  even 65% of the sharing costs 0.11 nats (5.39 against 4.83).
- **Why the gain is that small.** The trie's saving is a fixed ~8.2M
  positions, about 7.7 per character, all from contexts shorter than about
  12 characters: past that depth almost every context in the corpus is
  distinct. So a step can share only where its contexts have a short
  prefix in common. At longer context lengths the saving is a smaller
  share of the cost (1.32× at T=32, 1.06× at T=128), so the 0.05-nat
  ceiling shrinks.

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
- Follow-up runs are dirty for the same reasons: untracked weight files
  elsewhere in `rnd/`, the grid's logs, and configs written between
  commits. The trainer code is the commit each `meta.json` names: each
  restart of the grid rebuilt both binaries after a commit, and the one
  build that came before later edits (18:02:32 UTC, for the runs naming
  `b1019b3`) finished a minute before the first of them.
- Follow-up: one seed. The 2- and 4-pass pd=3 runs tried two rates each,
  and 0.0015, the higher, won both, so more passes may do somewhat better
  at a higher rate. `pd1s`'s best one-pass rate is the bottom of its grid.
- `mbu3` matches pd=3's step sizes, not its mix of depths: a window gives
  one target at each depth 1–16, while a pd=3 unit's targets sit at the
  depths of its subtree. Rounding to whole windows gives 1,060,418 windows
  per pass, 801 more than one epoch; the extra ones come from a second
  random permutation.
- Follow-up 2: one seed. `tq1338`'s best rate is the bottom of its grid.
  The 16-epoch `cv` run tried only 0.006, the best rate at every other
  budget.
- The plan replays' training seconds in `result.json` include building the
  plan (6–17 s). The replay trainer computes whole windows for every
  step, so its times say nothing about what a trie trainer would take.
  Positions per pass are the cost measure for the plans.
- Follow-up 2 runs say `git_dirty: true` because of untracked files only
  (weight files elsewhere in `rnd/`, the grid logs). The tracked tree was
  `bf32854` exactly.
