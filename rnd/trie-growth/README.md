---
title: Trie growth (cadence in time)
kind: experiment
status: active
outcome: n/a
question: >-
  The user (2026-11-20): train while the prefix trie grows, so early updates come from small,
  cheap tries ("cadence in time, not space"). With the current v2 (fp32, exact ancestor
  gradient), does growth beat full-trie AGPT at equal compute, and how close does it get to
  minibatch training? Arms: pd=1 and pd=3 units with the trie doubled each stage, and pd=0
  with one update each time the trie grows by 10% or 1% (the user's "N changes, then update").
answer: >-
  Not yet answered: runs pending (GPU slot 132).
opened: 2026-11-20
updated: 2026-11-20
code: main
eval: canonical
family: attention
tags: [cadence, growth, one-pass-gap]
related: [streaming-agpt-v1, progressive-growth-sgd-comparison, one-pass-gap, gradient-population]
---

# Trie growth

**Status:** active (2026-11-20). Configs `make_configs.py`, runs `run.sh`.

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
