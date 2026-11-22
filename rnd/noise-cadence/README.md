---
title: Noise cadence (split-half test, random-order growth)
kind: experiment
status: concluded
outcome: positive
question: >-
  The user (2026-11-21/22): can the training rhythm follow a more nuanced signal than the
  number of count changes? A split-half noise test decides when to step: the window starts
  are sampled in random order and alternate between two tries; step when the halves agree
  (noise at most theta x the gradient's signal), else add windows, aggregating only their
  gradient at the same weights. Does it close the gap to minibatch training at equal
  compute? And does random-order growth help the fixed-cadence arms of trie-growth?
answer: >-
  With a fresh sample after every step (the user's resampling variant) the noise cadence
  is level with minibatch at equal compute, within seed noise: 4.567 against 4.552 at 136M
  trie positions (means of 3 seeds each; minibatch's seeds span 4.526-4.570) and 4.535
  against 4.522 at 271M (2 seeds each; AGPT 4.520-4.550, minibatch 4.511-4.534), with
  5.8-8.7x fewer optimizer steps. 0.8 PPL better than every earlier AGPT arm (pd=3
  doubling 5.51 at 54M). The apparent 0.035-0.04 gap against seed 42 alone was mostly
  minibatch's seed spread (0.08 at 1 epoch, 0.044 at 8). v2's gradient matches PyTorch's
  to 1e-6 and v2 trained like minibatch (128 windows a step through tries) matches the
  PyTorch trainer over 3 seeds. A no-trie control, the same cadence on plain windows in
  PyTorch, is level with it at equal compute (4.578 against 4.567 at 136M, 3 seeds each;
  4.536 against 4.535 at 271M): the cadence (adaptive batch) carries the result, and the
  trie adds a 15-18% compute discount worth at most 0.01 here. The cumulative sample
  overfits (6.12); constant LR, sample caps and depth-stratified sampling do not help. Random-order growth helps pd=1 doubling by 0.9 PPL, not pd=3.
opened: 2026-11-21
updated: 2026-11-22
code: main
eval: canonical
family: attention
tags: [cadence, growth, noise-scale, one-pass-gap]
related: [trie-growth, one-pass-gap, gradient-population]
---

# Noise cadence

**Status:** concluded, positive: parity with minibatch at equal compute, within seed
noise; the no-trie control is level too, so the gain is the adaptive batch, not the trie
(2026-11-22, GPU slots 150, 153, 156, 160 and 166). Code `0a6f4de`, `7bf11e0`, `6a435d6`,
`e88bf88`.
Configs `make_configs.py`, runs `run.sh`, table `summarize.py` (writes `runs.json`).
Same setup as `trie-growth` and `one-pass-gap`: Tiny Shakespeare split
`4fa9aec1db6b3aea`, d64/L2, depth 16, the same seed weights, the trie's own counts.

## Question

After `trie-growth`, the user asked whether the model could learn its training
rhythm from more nuanced signals than the number of count changes. The noise
scale (diagnostic, 2026-11-21, exact PyTorch gradients on training text) grows
from 13 windows at the seed weights to about 1,500 at the growth runs' end and
70k / 330k after 1 / 8 minibatch epochs. Progressive-sampling methods (Byrd,
Chin, Nocedal and Wu 2012; Bollapragada, Byrd and Nocedal 2017) grow the
sample until a norm test passes. This experiment builds that test into AGPT.

## Code changes (`0a6f4de`)

- `experimental.growth_random_order` / `growth_order_seed`: window starts in
  a seeded random order, so each stage's trie is a random sample of the text.
  The full trie is the same as in corpus order (1,528,489 nodes, 8,902,781
  positions, 2,563,750 counts).
- `experimental.growth_noise_theta` (pd=0): sampled starts alternate between
  tries A and B. At fixed weights the gradient is linear in the counts, so
  g = (G_A + G_B) / (E_A + E_B), and the halves' disagreement estimates the
  sampling noise in g:
  noise^2 = |G_A/E_A - G_B/E_B|^2 n_A n_B / n^2 (1 - n/N),
  signal^2 = |g|^2 - noise^2. Step when noise <= theta x signal. Otherwise add
  windows (the norm test's own target, clamped to [1.05 n, 2 n]) and add only
  their gradient, at the same weights, to G_A and G_B; then test again. After
  each step both tries are recomputed whole at the new weights. The LR
  schedule (warmup-cosine, no warmup) runs over a budget in trie positions
  (`growth_noise_budget_positions`, every gradient pass counted).
- `experimental.growth_noise_resample`: after every step the tries are
  dropped and refilled with fresh windows, as many as the test last passed
  with, the next ones in the shuffled order (reshuffled each pass through the
  text). Without it the sample is cumulative.
- `AGPT_NOISE_CHECK=1`: after each growth, trie A recomputed whole against
  the incremental aggregate. Events identical, mean loss to 7 digits, gradient
  within about 1e-6; outliers to 5e-4 moved to different checks on each of
  three reruns (float32 atomics, not structure).

## Results

Rolling byte PPL (fixed-token PPL in `runs.json`), canonical evaluation on
the held-out chunks. Positions are trie positions; a minibatch epoch is 16.95M
(`one-pass-gap`, 128 windows a step, Adam lr 0.006, best of 0.0015-0.012 at
every epoch count).

### Fresh sample each step (`f0`), against minibatch

| compute | theta | best lr | steps | rolling | minibatch at this compute |
|---|---:|---:|---:|---:|---:|
| 54M (3.2 epochs) | 1 | 0.012 (0.006 not run) | 1,662 | 4.86 | 4.61 (interpolated, 2 ep 4.659 / 4 ep 4.582) |
| | 2 | 0.006 | 3,653 | 4.74 | |
| | 4 | 0.006 | 5,863 | 4.72 | |
| | 8 | 0.006 (0.003 not run) | 6,381 | **4.70** | |
| 68M (4 epochs) | 2 | 0.006 | 4,619 | 4.68 | 4.582 (8,279 steps an epoch: 33,116) |
| | 4 | 0.006 | 7,547 | **4.66** | |
| 136M (8 epochs) | 2 | 0.006 | 8,221 | 4.57 | 4.526 (66,232 steps) |
| | 4 | 0.006 | 11,453 | **4.565** | |

- Bracketed: theta 2 and 4 at 54M (0.003 and 0.012 both worse). At 68M and
  136M only 0.006 and 0.012 ran; 0.003 was worse at 54M.
- The gap to minibatch: 0.10 / 0.08 / 0.04 PPL at 54 / 68 / 136M (0.020 /
  0.017 / 0.009 nats a character).
- Train seconds: 105 / 137 / 230 for theta 4 at 54 / 68 / 136M, against
  minibatch's 116 / 266 at 4 / 8 epochs (a different trainer, PyTorch).
- Data: the theta 4 runs drew 3.8 / 4.8 / 9.6 epochs of windows. The trie
  merged 15-16% of their positions (13-18% across the runs), so the same
  steps without the trie would cost about a fifth more (18-22%).
- Samples: a step used 128-1,460 windows from 10% to 90% of the steps
  (128-1,060 at theta 4 and 8); the
  last 2% of the steps took 26-32% of the budget on samples of 30k-156k
  windows, as the signal shrank late in the cosine schedule. The schedule
  anneals twice (lower LR and bigger samples).

### Cumulative sample (`n0`)

| theta | lr 0.003 | 0.006 | 0.012 | 0.024 | steps (best) |
|---:|---:|---:|---:|---:|---:|
| 1 | | **6.12** | 6.42 | 6.64 | 351 |
| 2 | 6.73 | 6.50 | 6.31 | 6.69 | 325 |
| 4 | | 6.18 | 6.58 | 6.98 | 335 |

All at 54-55M positions. The test held the sample at one size for 215-238
steps (9,650 windows at theta 1, 11,904 at theta 4), about 30M positions,
while the loss on it fell from 2.26 to 1.69-1.74; fresh windows then scored
1.83, 0.09-0.14 nats worse. Both halves are trained on, so overfitting the
sample does not make them disagree: the test measures noise, not this bias.

### Random-order growth at fixed cadences (`r`), against trie-growth's corpus order

| arm | lr | steps | positions | random order | corpus order |
|---|---:|---:|---:|---:|---:|
| pd=1, one stage (the full trie) | 0.012 | 65 | 8.9M | 15.725 | 15.724 (static 15.719) |
| pd=1 doubling, 1 pass a stage | 0.006 | 738 | 27.4M | **8.64** | 9.51 |
| pd=1 doubling, 2 passes a stage | 0.006 | 1,476 | 54.9M | **7.24** | 8.15 |
| pd=3 + crown doubling, 1 pass a stage | 0.0015 | 80,054 | 27.4M | 6.12 | 5.96 |
| (same, lr 0.003) | 0.003 | 80,054 | 27.4M | 6.69 | |
| pd=0, trie x1.1 a stage | 0.006 | 75 | 116.5M | 12.29 | 12.74 |

pd=3's order noise at one pass is about 0.11 (trie-growth), so random order
does not help pd=3. pd=1 doubling at 2 passes a stage and the 1% arm were not
rerun in random order (dropped for slot time).

### Follow-up (slot 156): does the gap cross?

The user (2026-11-22), on "if it doesn't cross I'd call the line done": "This
is the first time we got this close!" Ten runs, theta 4 and lr 0.006 unless
noted; a new minibatch bar at 16 epochs (`mb-e16`, one-pass-gap's recipe,
run here).

| run | positions | steps | rolling |
|---|---:|---:|---:|
| fresh sample, 16 epochs' budget | 272.7M | 15,166 | 4.550 |
| fresh sample, theta 8, 16 epochs' budget | 271.3M | 15,425 | **4.546** |
| minibatch, 16 epochs, lr 0.006 / 0.003 | 271.2M | 132,464 | **4.511** / 4.514 |
| fresh sample, 136M, seed 43 (the same init, another window order) | 136.7M | 9,872 | 4.568 (seed 42: 4.565) |
| fresh sample, theta 8, 136M | 136.2M | 11,660 | 4.578 |
| constant LR 0.006 / 0.003, 136M | 135.6M | 13,250 / 16,614 | 4.650 / 4.657 |
| constant LR 0.006 / 0.003, 54M | 54.2M | 8,569 / 7,901 | 4.817 / 4.818 |

- The gap: 0.10 / 0.08 / 0.04 / 0.035 PPL at 54 / 68 / 136 / 271M. It stops
  narrowing after 136M; both sides gain about 0.015 from 8 to 16 epochs.
- Two seeds at 136M agree to 0.003. (That was read at the time as "the gap is
  real"; it measured AGPT's spread only. Minibatch's seeds spread far more: see
  the free-rein section below.)
- The 16-epoch budget bought only 3,700 more steps than the 8-epoch one: the
  extra compute went into late samples of 130k-250k windows.
- Constant LR: the sample stops growing near 600-1,300 windows and training
  never settles; the cosine tail's large, expensive steps pay for themselves.

### Free rein (slot 160): the trainer control, seeds, caps

The user (2026-11-22): if these runs show no progress, free rein for one idea.
Three checks, in order.

**Depth-stratified sampling (CPU diagnostic, closed).** The trie truncated at
depth k is small (65 / 1,466 / 12,947 / 62,930 positions at k = 1-4), so its
gradient is the exact full-corpus gradient of every window's first k targets;
sampling only the deeper targets would remove the shallow noise. Split by
depth band on four checkpoints (exact PyTorch gradients, training text;
`tools/depth_noise.py`, output `tools/depth_noise.json`; the gradient check is `tools/grad_check.py`), the noise is spread evenly over depth: exact
depths 1 / 1-2 / 1-3 / 1-4 / 1-8 remove 2-5% / 10-15% / 17-24% / 26-29% /
46-53% of B_noise. Depths 1-3 cost ~13k positions a step for ~20% less noise,
more than a mid-training sample costs. Not built.

**The trainer control.** `growth_noise_max_windows` (`7bf11e0`) caps the
sample; with growth_start = cap = 128, v2 trains like minibatch through tries
(`m0-w128`: 128 fresh windows a step, the trie merging 5.1%). Seed 42 put it
0.044 / 0.079 behind PyTorch at 1 / 2 epochs, so the gradient was checked:
`AGPT_NOISE_DUMP` (`6a435d6`) writes v2's sample gradient, and PyTorch's on
the same 128 windows at the same weights (float64) matches it, rel diff 2.1e-7
at the seed weights and 1.3e-6 at m0-w128-e2's final weights, per parameter
and overall, loss to 8 digits. Adam (bias-corrected, eps 1e-8) and the cosine
are the same code shape in both. Then seeds:

| rolling PPL, seeds 42 / 43 / 44 | mean |
|---|---:|
| 1 epoch, v2 through tries (m0-w128) 4.870 / 4.825 / 4.830 | **4.842** |
| 1 epoch, PyTorch minibatch 4.826 / 4.867 / 4.904 | 4.866 |
| 8 epochs, fresh-sample cadence theta 4 4.565 / 4.568 / 4.569 (slot 166) | 4.567 |
| 8 epochs, PyTorch minibatch 4.526 / 4.570 / 4.559 | **4.552** |
| 16 epochs, fresh-sample cadence theta 4 4.550 / **4.520** / - | 4.535 |
| 16 epochs, PyTorch minibatch 4.511 / 4.534 / - | **4.522** |

(seed = window order; the init is the same seed file for every run.)

- The trainers agree: v2 at 128 windows is level with PyTorch (0.024 ahead on
  the mean, inside the spread).
- Minibatch's seed spread is 0.078 at 1 epoch, 0.044 at 8, 0.023 at 16.
  Seed 42, the one-pass-gap bar used everywhere above, is its best seed at 1
  and 8 epochs.
- The cadence against minibatch: +0.015 at 136M, +0.013 at 271M on the means,
  inside both spreads; AGPT seed 43 at 271M (4.520) beats minibatch seed 43
  (4.534). Level, not ahead; more seeds would be needed to resolve 0.01.
- Steps: 9,872-11,453 against 66,232 at 136M (5.8-6.7x fewer), 15,166-15,895
  against 132,464 at 271M (8.3-8.7x fewer).
- The 54M / 68M gaps (0.10 / 0.08 above) are against seed 42 only; no
  minibatch seeds were run at 2-4 epochs.

**Caps on the late samples (no help).** theta 4 at 136M capped at 2,048 /
8,192 windows: 4.586 / 4.581 against 4.565 uncapped. With the constant-LR
result, the large late samples do the annealing and earn their cost.

### The no-trie control (slot 166): the same cadence on plain windows

The user (2026-11-22) asked for it. `agpt_minibatch_train.py`
`experimental.noise_theta` (`e88bf88`) runs `run_noise_cadence_v2`'s
fresh-sample cadence in PyTorch on plain windows: the same halves, noise test,
growth rule, resample size and cosine over the budget, with each window costing
16 positions per gradient (the minibatch meter). Since the trie's gradient is
the windows' gradient exactly (free-rein section), the trie's only part is
compute: here it spends 13.1-13.6 positions a window (15-18% fewer). `fl` arms:
theta 4, lr 0.006, at AGPT's budgets (equal compute, so ~16% fewer windows) and
at 163M (about the windows AGPT's 136M runs computed, 10.3M against 10.1M:
equal data). A third AGPT seed at 136M (`f0-t4-b136-s44`) came with it.

| rolling PPL, seeds 42 / 43 / 44 | mean | windows | steps | train s |
|---|---:|---:|---:|---:|
| 136M, AGPT cadence (trie) 4.565 / 4.568 / 4.569 | **4.567** | 10.1M | 9,872-11,453 | 205-230 |
| 136M, flat cadence 4.574 / 4.564 / 4.595 | 4.578 | 8.5-8.6M | 10,347-10,885 | 111-112 |
| 163M, flat cadence 4.561 / 4.553 / - | 4.557 | 10.3M | 11,032-11,638 | 126-132 |
| 136M, PyTorch minibatch 4.526 / 4.570 / 4.559 | 4.552 | 17.0M | 66,232 | 268-275 |
| 271M, AGPT cadence (trie) 4.550 / 4.520 / - | 4.535 | 20.6-20.8M | 15,166-15,895 | 378-412 |
| 271M, flat cadence 4.543 / 4.530 / - | 4.536 | 17.0-17.2M | 13,926-14,492 | 184-186 |
| 271M, PyTorch minibatch 4.511 / 4.534 / - | 4.522 | 33.9M | 132,464 | 449-465 |

(The seeds draw different windows in the two trainers: v2 shuffles with
mt19937, PyTorch with torch.randperm, so rows don't pair by seed.)

- At equal compute the trie's discount is worth 0.010 at 136M and 0.001 at
  271M, inside the flat side's spread (0.030 at 136M; without its seed 44
  the means are 4.569 and 4.567).
- At equal data (flat 163M against AGPT 136M) the two implementations land
  level, 4.557 against 4.567: the same algorithm gives the same result.
- Both cadences against minibatch, all runs pooled by budget: 4.572 (6 runs)
  against 4.552 at 136M, 4.536 (4) against 4.522 at 271M, 0.014-0.021 behind
  on the means, about 1-1.5 standard errors. Level within noise, possibly a
  little behind; not ahead.
- Wall clock: the flat cadence trains in half v2's time (111 s against
  205-230 s at 136M) and well under minibatch's (268-275 s), with 6-9x fewer
  steps than minibatch.

## Reading

- **The cadence works once the sample is fresh.** With fresh samples the
  noise test gives 4.70 at 54M positions, where every earlier AGPT arm was at
  5.51 or worse (pd=3 doubling), and the cumulative sample at 6.12. Its steps
  are good: 11,453 of them reach 4.565, which minibatch reaches in about 41k steps
  (interpolated: 4 epochs, 33,116 steps, give 4.582; 8 epochs, 66,232, give 4.526).
- **It is level with minibatch at 8-16 epochs' compute.** Against seed 42 alone
  the gap looked like 0.10 / 0.04 / 0.035 at 54 / 136 / 271M; with seeds it is
  0.015 / 0.013 at 136 / 271M, inside the spread (free-rein section). At
  1 epoch, v2 trained like minibatch is level with PyTorch, so the trainers are
  not the difference either.
- **The gain is the cadence, not the trie (measured, slot 166).** The same
  cadence on plain windows in PyTorch is level with the trie version at equal
  compute (4.578 / 4.536 against 4.567 / 4.535 at 136M / 271M) and at equal
  data. The trie saves 15-18% of the positions at these sample sizes, worth
  at most 0.01 here. The fresh-sample cadence is adaptive-batch minibatch;
  the trie merges its batch.
- **The expensive end of the schedule pays.** A quarter to a third of the
  budget goes to the last 2% of the steps, on samples of 30k-360k windows; a
  constant LR (the test left to anneal) and caps of 2,048 / 8,192 windows are
  both worse, so those samples are the annealing, not waste.
- **Random order is a pd=1 fix.** It removes the corpus-order curriculum,
  which cost pd=1 doubling 0.9 PPL; pd=3 units already mix the text.

## Next (not decided by the user)

- More seeds (4-5 a side) at 136M and 271M to resolve a 0.01 difference, and
  minibatch seeds at 2-4 epochs for the 54M / 68M points.
- The steps advantage (6-9x fewer at equal compute and quality) is what
  adaptive batching buys in data-parallel training; it needs no trie (slot 166).
- Where the trie could still add: a setting where the batch merges far more
  (a shallower trie, or samples that are a large share of the text), or the
  refined-counts objective (`refined-trie-counts`, a larger text's next-char
  counts on the small trie's nodes), which so far helps minibatch as much as
  AGPT.
- Closed here: constant LR, sample caps, depth-stratified sampling, the
  no-trie control.
