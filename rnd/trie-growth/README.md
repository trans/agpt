---
title: Trie growth (cadence in time)
kind: experiment
status: concluded
outcome: mixed
question: >-
  The user (2026-11-20): train while the prefix trie grows, so early updates come from small,
  cheap tries ("cadence in time, not space"). With the current v2 (fp32, exact ancestor
  gradient), does growth beat full-trie AGPT at equal compute, and how close does it get to
  minibatch training? Arms: pd=1 and pd=3 units with the trie doubled each stage, and pd=0
  with one update each time the trie grows by 10% or 1% (the user's "N changes, then update").
answer: >-
  Growth beats full-trie AGPT at equal compute at pd=1 and pd=3, and at about equal wall
  time: pd=3 doubling reaches rolling byte PPL 5.96 in 27M trie positions against static
  6.28 in 36M, and 5.51 in 54M against 5.77 in 71M. The gain is cheap steps (small early
  tries), not better ones. It does not reach minibatch training: 5.51 at 3.2 epochs' worth of
  positions against minibatch 4.66 at 2 epochs and 4.58 at 4, though the gap at pd=3 narrows
  from 2.9 at one pass to 0.9. pd=0 with one update per 1% of growth (the user's "N changes,
  then update", whole trie recomputed after each) makes the best steps (697 steps give 7.02,
  which pd=1 doubling needs 2,776 steps to reach) but costs 110 full-trie passes. Static
  pd=3's rate is unbracketed above at 2-8 passes. Found and fixed on
  the way: v2 growth mode had never applied the exact ancestor gradient (03f3a49).
opened: 2026-11-20
updated: 2026-11-20
code: main
eval: canonical
family: attention
tags: [cadence, growth, one-pass-gap]
related: [streaming-agpt-v1, progressive-growth-sgd-comparison, one-pass-gap, gradient-population]
---

# Trie growth

**Status:** concluded, mixed (2026-11-20, GPU slot 132, 14:56-17:04 UTC). Configs
`make_configs.py`, runs `run.sh`, table `summarize.py` (writes `runs.json`).

## Question

The user (2026-11-20): "training the model as the prefix tree grows. It helps
b/c the early trie are much smaller, and thus faster, and the updates are more
gradual... So we get cadence in time, not space." Between updates the trie's
gradient is aggregated as counts arrive (exact at fixed weights); after each
update it is recomputed at the new weights.

The record: `streaming-agpt-v1` (May 16-18, legacy eval, before the event
weighting, exact-gradient, race and fp32 fixes) beat full-trie AGPT at equal
passes, and was never compared with minibatch training at equal compute
(`progressive-growth-sgd-comparison` is inconclusive). The bar here is
`one-pass-gap`'s minibatch runs on the same split, seed weights and model:
rolling byte PPL 4.83 / 4.66 / 4.58 / 4.53 after 1 / 2 / 4 / 8 epochs (lr 0.006).

## Code changes for this test

v2's growth mode (`train-growth`) carries weights, Adam moments and one LR
schedule across stages, and has grown the trie in place since `55904b2`. Added:

- `experimental.growth_start` / `growth_ratio`: geometric stage frontiers.
- `experimental.growth_exact_schedule`: count the optimizer steps before
  training. v2 otherwise assumes stages × epochs × vocab_size steps, right only
  at pd=1, so warmup-cosine would end at the wrong step at pd=0 and pd=3.
- `experimental.train_crown` and `unit_order_seed` now work in growth mode
  too (they were static-only). Without crown units, growth at pd=3 never trains
  targets shallower than depth 3: likely the window-start problem
  `progressive-growth-sgd-comparison` traced its rolling-PPL regression to.
- `03f3a49`: growth mode now runs the exact ancestor backward
  (`anc_grad_exact`, default on since September). Until then train-growth
  returned before train-epoch's check of that key, so it accepted
  `anc_grad_exact: true` and silently trained on the truncated ancestor
  gradient, which the September checks found off by 2-6%. Every earlier v2
  growth result used the truncated gradient. Found by the first check below:
  the one-stage growth run's second unit had mean loss 4.5288 against the
  static run's 4.3176; it is 4.3176 after the fix.
- `e4ed3c0`: a growth block with only `growth_ratio` (no divisions) failed the
  trainer's required-argument check.

Checks: `g1-full-e1` and `g3-full-e1` (one growth stage, the full trie) must
reproduce the static `s1-e1` and `s3-e1`.

## Arms

See `make_configs.py`. Compute is counted in trie positions (each stage's
edge characters × its passes); a full-trie pass is 8.90M positions and a
minibatch epoch 16.95M. Doubling from 1,000 starts ends with two near-full
stages (1,024,000 and 1,059,633 starts), so E passes a stage cost about 3E
full-trie passes, not the 2E first planned.

## Results

### Checks: one growth stage against the static trie

Rolling byte PPL / fixed-token PPL, one pass:

| | static | static, order seed 2 | static, order seed 3 | growth, one stage |
|---|---:|---:|---:|---:|
| pd=1, lr 0.012 | 15.719 / 15.650 | | | 15.725 / 15.654 |
| pd=3 + crown, lr 0.0015 | 7.749 / 7.160 | 7.837 / 7.253 | 7.857 / 7.236 | 7.692 / 7.064 |

- pd=1 reproduces. The units come in the same order, and per-unit mean losses
  agree within 0.0035 nats (median 0.0005) over the 65 units.
- pd=3 holds the same units in a different order. The incremental trie numbers
  its nodes differently, so the same shuffle seed visits them differently. The
  one-stage growth run is one more draw from the order noise, which spans 0.11
  across three static seeds at one pass.
- The growth trie has 106 fewer query positions than the stored trie (8,902,781
  of 8,902,887), all in 19 units near the end of the corpus.
- Both growth runs took exactly the steps the exact schedule counted: 65 and
  12,675.

### Growth against the full trie and minibatch training

Rolling byte PPL (fixed-token PPL), best learning rate per row, with optimizer
steps, compute in trie positions (minibatch epochs' worth in brackets) and
training seconds from `result.json`. Every learning rate run is in `runs.json`.

| arm | lr | steps | positions | train s | rolling (fixed) |
|---|---:|---:|---:|---:|---:|
| **pd=1** static, 1 pass | 0.012 | 65 | 8.9M (0.5) | 8 | 15.72 (15.65) |
| static, 2 passes | 0.012 | 130 | 17.8M (1.1) | 14 | 12.26 (12.10) |
| static, 4 passes | 0.006 | 260 | 35.6M (2.1) | 28 | 10.27 (10.17) |
| static, 8 passes | 0.006 | 520 | 71.2M (4.2) | 55 | 8.66 (8.42) |
| doubling, 1 pass a stage | 0.006 | 694 | 27.1M (1.6) | 30 | **9.51** (9.19) |
| doubling, 2 passes a stage | 0.006 | 1,388 | 54.2M (3.2) | 53 | **8.15** (7.75) |
| doubling, 4 passes a stage | 0.006 | 2,776 | 108.3M (6.4) | 101 | **7.00** (6.59) |
| **pd=3** + crown, shuffled: static, 1 pass | 0.0015 | 12,675 | 8.9M (0.5) | 53 | 7.75-7.86 (7.16-7.25), 3 orders |
| static, 2 passes | 0.0015 | 25,350 | 17.8M (1.1) | 105 | 6.91 (6.22) |
| static, 4 passes | 0.0015 | 50,700 | 35.6M (2.1) | 209 | 6.28 (5.54) |
| static, 8 passes | 0.0015 (only) | 101,400 | 71.2M (4.2) | 416 | 5.77 (5.00) |
| doubling, 1 pass a stage | 0.0015 | 72,962 | 27.1M (1.6) | 234 | **5.96** (5.24) |
| doubling, 2 passes a stage | 0.0015 | 145,924 | 54.2M (3.2) | 461 | **5.51** (4.78) |
| **pd=0**, one update a stage: trie ×1.1 a stage | 0.006 | 75 | 113.9M (6.7) | 92 | 12.74 (12.56) |
| trie ×1.01 a stage | 0.012 (highest) | 697 | 974.7M (57.5) | 742 | 7.02 (6.58) |
| **minibatch**, 128 windows (`one-pass-gap`), 1 / 2 / 4 / 8 epochs | 0.006 | 8,279 an epoch | 17-136M (1-8) | 31 / 60 / 116 / 266 | 4.83 / 4.66 / 4.58 / 4.53 |

Notes on the table:

- **Which best rates are bracketed** (a worse rate run on both sides):
  - Bracketed: every growth arm except 1%; static pd=1 at 4 and 8 passes.
  - Static at one pass ran one rate only. `one-pass-gap`'s grid found it best:
    0.012 at pd=1, and 0.0015 at pd=3, where 0.003 was 0.20 worse and 0.00075
    0.31 worse.
  - Not bracketed above: static pd=1 at 2 passes (0.012); static pd=3 at 2 and
    4 passes (0.0015 the higher of two); static pd=3 at 8 passes (0.0015 only).
    At 4 passes 0.00075 was only 0.08 worse than 0.0015, so static pd=3 may
    gain somewhat at 0.003. That would narrow growth's 0.3 lead at pd=3 by an
    unknown amount.
  - The 1% arm (7.02 at 0.012, 7.05 at 0.006) is also unbracketed above.
- The 10% arm: 12.74 at 0.006, 12.76 at 0.012, 13.52 at 0.003.
- A minibatch epoch is 16.95M token positions, and the trie shares prefixes.
  Equal positions is therefore roughly equal forward compute, not equal data.
  The minibatch trainer is a different program (PyTorch), so wall times
  compare only loosely.
- Growth ingests window starts in corpus order (`growth_trie_v2.cuh`, start
  *i* = text position *i*). The early stages train only on the opening 1,000,
  2,000, 4,000 ... characters of the play, and the last 3.4% of the text
  appears only in the final stage.

## Reading

- **Growth beats the full trie.** It does so at pd=1 and pd=3, at less
  compute and at about equal wall time:
  - pd=1: 9.51 in 27M positions against static's 10.27 in 36M, and 8.15 in
    54M against 8.66 in 71M.
  - pd=3: 5.96 in 27M against 6.28 in 36M, and 5.51 in 54M (461 s) against
    5.77 in 71M (416 s). Static pd=3 was not tried above lr 0.0015 at 2-8
    passes, so this lead may shrink.

  This confirms the record's May result (`streaming-agpt-v1`, equal passes,
  legacy evaluation) at equal positions, with the exact gradient, fp32 and the
  canonical evaluation.
- **The gain is cheap steps, not better ones.** This is the user's reason: "the
  early trie are much smaller, and thus faster". Per step, the full trie is
  better: at pd=1, 520 static steps give 8.66 and 694 growth steps give 9.51.
  But the growth steps cost 39k positions each against 137k.
- **Growth doesn't reach minibatch training.**
  - The best growth run, pd=3 at 54M positions (3.2 epochs' worth), reaches
    5.51. Minibatch reaches 4.66 at 2 epochs and 4.58 at 4.
  - At pd=3 the gap narrows from 2.9 at one pass (7.75 against 4.83) to 0.9.
  - The remaining gap is not the number of updates. pd=3 doubling takes
    145,924 steps, 4.4 times minibatch's 33,116 over 4 epochs, and still
    trails. `one-pass-gap`'s finding remains: AGPT's unit steps are worth less
    than equal random batches. Its suspects were unequal unit sizes (Adam takes
    a full step on a 1-event unit) and grouping by context. Here the
    corpus-order curriculum is a third.
- **The user's literal scheme (pd=0) makes the best steps but costs the most
  compute.** Here N(trie) is the growth between updates, and the whole trie is
  recomputed after each update.
  - Its steps are the best of any arm: 697 one-update stages give 7.02, about
    what pd=1 doubling reaches in 2,776 steps (7.00).
  - Every step is the exact gradient of the whole current trie, which costs
    about (1+α)/α full-trie passes: 12.8 measured at α = 10% and 109.5 at 1%.
  - So at equal steps it is the best AGPT arm. At equal compute it is the
    worst: 57 minibatch epochs' worth for 7.02.

## Next (not decided by the user)

- Static pd=3 at lr 0.003 for 4 and 8 passes (about 11 min), to bracket the
  rate behind growth's 0.3 lead at pd=3.

- Growth with window starts in random order, so each stage's trie is a
  random sample of the text rather than its opening. That is the
  minibatch-like version, and it separates the curriculum from the size.
- Growth at pd=3 with equal-size units (`one-pass-gap`'s first suspect), or
  with fewer, larger units on the small early tries.
- pd=3 doubling at 4 passes a stage (about 108M positions, 15 min), to see
  whether it keeps closing on minibatch or flattens.
