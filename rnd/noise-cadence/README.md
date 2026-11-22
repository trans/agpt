---
title: Noise cadence (split-half test, random-order growth)
kind: experiment
status: concluded
outcome: mixed
question: >-
  The user (2026-11-21/22): can the training rhythm follow a more nuanced signal than the
  number of count changes? A split-half noise test decides when to step: the window starts
  are sampled in random order and alternate between two tries; step when the halves agree
  (noise at most theta x the gradient's signal), else add windows, aggregating only their
  gradient at the same weights. Does it close the gap to minibatch training at equal
  compute? And does random-order growth help the fixed-cadence arms of trie-growth?
answer: >-
  With a fresh sample after every step (the user's resampling variant) the noise cadence
  reaches rolling byte PPL 4.70 at 54M trie positions, 4.66 at 68M and 4.565 at 136M,
  against minibatch 4.61 (interpolated), 4.58 and 4.53: 0.8 PPL better than every earlier
  AGPT arm (best 5.51), within 0.10-0.04 of minibatch, the gap shrinking with compute, but
  not past it. It takes 5.8x fewer steps than minibatch for that (11,453 against 66,232 at
  136M). With the sample kept cumulative it overfits its sample and stalls at 6.12. The
  trie merges 13-18% of the positions; the rest of the gain is the cadence. Random-order
  growth helps pd=1 doubling by 0.9 PPL and does not help pd=3.
opened: 2026-11-21
updated: 2026-11-22
code: main
eval: canonical
family: attention
tags: [cadence, growth, noise-scale, one-pass-gap]
related: [trie-growth, one-pass-gap, gradient-population]
---

# Noise cadence

**Status:** concluded, mixed (2026-11-22, GPU slots 150 and 153). Code `0a6f4de`.
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

## Reading

- **The cadence works once the sample is fresh.** With fresh samples the
  noise test gives 4.70 at 54M positions, where every earlier AGPT arm was at
  5.51 or worse (pd=3 doubling), and the cumulative sample at 6.12. Its steps
  are good: 11,453 of them reach 4.565, which minibatch reaches in about 41k steps
  (interpolated: 4 epochs, 33,116 steps, give 4.582; 8 epochs, 66,232, give 4.526).
- **It does not beat minibatch yet.** The gap narrows with compute, from 0.10
  PPL at 54M to 0.04 at 136M. That fits the noise-scale reading (the trie's
  big batch pays late in training) but is not past minibatch at any budget
  run.
- **Most of the gain is the cadence, not the trie.** The trie merges 13-18%
  of the positions at these sample sizes; a flat trainer with the same
  adaptive batch would take the same steps at about a fifth more compute. The
  fresh-sample cadence is adaptive-batch minibatch with the batch merged in a
  trie.
- **The end of the schedule is wasteful.** The test grows the sample as the
  gradient's signal shrinks, while the cosine schedule also cuts the LR; a
  quarter to a third of the budget goes to the last 2% of the steps.
- **Random order is a pd=1 fix.** It removes the corpus-order curriculum,
  which cost pd=1 doubling 0.9 PPL; pd=3 units already mix the text.

## Next (not decided by the user)

- Constant LR with the noise test doing the annealing (Smith et al. 2018,
  "don't decay the learning rate, increase the batch size"), or a cap on the
  sample, to spend the last third of the budget on steps that move.
- Larger budgets (16 epochs' worth) to see whether the gap crosses zero.
- lr 0.003 at 68M and 136M, theta 8 at 136M.
