---
title: Doped counts at depth 16
kind: experiment
status: concluded
outcome: mixed
question: >-
  The user (2026-11-22), after refined-trie-counts' depth-8 result: does a small corpus doped
  with a large text's next-character counts still come out ahead at depth 16, where most of
  the small corpus's contexts are rare even in 100M characters? And do 1G and 10G characters
  of counts fill in the deep contexts (the user expects the relative gain to shrink at 100M
  and the larger texts to be needed)?
answer: >-
  Doping still pays at depth 16 and grows with the count text: the 1M slice with the 100M /
  1G / 10G text's next-character counts beats hard targets by 0.053 / 0.064 / 0.072 nats per
  character after 8 epochs (two seeds each, tight), about 0.01 more per tenfold text, as the
  deep contexts fill in (at depth 16, 21% / 39% / 59% of the slice's contexts were seen twice
  or more). Against the slice's own counts as soft targets the 100M gain is about depth 8's
  (0.053 against 0.058). But the doped slice no longer comes out ahead of the bigger text:
  minibatch on the 100M text itself, same steps, is 0.086 nats ahead of the slice, 0.014 ahead
  of the slice with 10G counts (5.397 against 5.471), and still improving (5.278 at 3x the
  steps); at depth 8 the slice with 100M counts beat it (6.150 against 6.329). At depth 16 the
  slice runs short of contexts, not only of targets: every slice arm gains under 0.01 nats from
  epoch 4 to 8, and doping refines the targets of contexts the slice has but adds none. A first 10G build
  held two held-out pieces whole (other editions of their test books); found and removed
  before any 10G count was used.
opened: 2026-11-22
updated: 2026-11-22
code: main
eval: canonical
family: attention
tags: [refined-counts, doping, trie-as-storage]
related: [refined-trie-counts, dynamic-eval, noise-cadence]
---

# Doped counts at depth 16

**Status:** concluded, mixed (2026-11-22, GPU slot 170). Configs `make_configs.py`, tables `count.sh`, runs
`run.sh` (order `order.txt`).

## Setup

- **Training text:** refined-trie-counts' 1M PG-19 slice A (`data/pg19/a1m`), the
  same held-out chunks and vocabulary (96 symbols).
- **Count texts:** the 100M split, and a 10G text that continues it
  (`data/pg19/imm-split-10g-dedup`, built by `tools/build_pg19_big.py`: the 20M
  split's pieces, then 25,390 whole PG-19 train books in mobius's seeded order,
  normalized as mobius's `pg19-normal.cr`). Its first 100,000,000 characters are
  byte for byte the 100M split (the build checks the SHA-256), so the 100M, 1G and
  10G texts are nested prefixes of one file.
- **Held-out leak, found and removed:** the first build (all train books but the
  pieces') held the held-out pieces 20 and 23 whole and piece 6 at 95% (of their
  64-character strings; other editions of their test books, e.g. train/3891 for
  test/3890), 10 pieces above 1%. mobius's own overlap check had already left these
  books out of what it counts. The text now takes, past the 100M split, only
  books mobius counts for the 20M split, and leaves out 54 more found by
  `tools/heldout_overlap.py` (`tools/excluded_books.txt`, 1,339 books in all).
  Rescanned (`tools/heldout_overlap_10g_dedup.json`): the worst piece has 0.6% of
  its 64-character strings in the 10G text (mean 0.02%; 0.3% at 1G, 0.2% at
  100M), quotations: two books near the end quote ~220 of piece 33's. That bounds
  the leak to about 0.01% of held-out positions; those books were kept.
- **Tables:** `src/tools/doped_counts.py` streams the 10G text once and writes A's
  depth-16 contexts (8,585,209) with their next-character counts at 100M, 1G and
  10G. It walks each position leafward and stops at the first context A lacks.
  Checked equal to `trie_plan.context_targets_from` (the in-memory counter the
  trainer uses for `trie_targets_corpus`) on a 100k slice at two marks.
- **Training:** `src/tools/agpt_minibatch_train.py`, refined-trie-counts' minibatch
  recipe at seq_len 16: d64/L2 from one init, 128 windows a step, Adam 0.003
  constant, 8 epochs; train seeds 42 and 43. The table arms use
  `experimental.trie_targets_table`.
- **Leak check:** `tools/heldout_overlap.py`, the share of each held-out chunk's
  32- and 64-character strings found in the count texts.

## Results

Slot 170, 2026-11-22, 17:28-19:13 UTC: the 12 runs of `order.txt` (the table arms
after the tables, `count.sh`, 11.7 minutes on the GPU for 10G). Canonical rolling
byte PPL on the 50 held-out pieces (`multi-chunk-heldout`); table: `summarize.py`
(writes `runs.json`).

| arm, seeds 42 / 43 | epoch 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| slice, hard targets | 6.081 / 6.076 | 5.969 / 6.004 | 5.897 / 5.918 | 5.895 / 5.864 |
| slice, its own counts (soft) | 5.983 / 6.053 | 5.896 / 5.959 | 5.880 / 5.893 | 5.861 / 5.904 |
| slice + 100M counts | 5.813 / 5.846 | 5.697 / 5.705 | 5.607 / 5.609 | 5.586 / 5.567 |
| slice + 1G counts | 5.765 / 5.801 | 5.633 / 5.660 | 5.552 / 5.562 | 5.516 / 5.510 |
| slice + 10G counts | 5.728 / 5.759 | 5.615 / 5.618 | 5.526 / 5.509 | **5.476 / 5.465** |
| the 100M text itself, same steps | | | | **5.384 / 5.410** |

The 100M text at 16 / 25 epochs (2x / 3x the steps): 5.303 / 5.337, 5.269 / 5.287.

Mean log PPL against hard targets at epoch 8 (nats per character): own counts
+0.001; 100M counts −0.053; 1G −0.064; 10G −0.072; the 100M text −0.086.

Coverage at depth 16 (`doped_counts.py`'s table meta): the share of the slice's
depth-16 contexts seen at least twice / 10 / 100 times in the count text is
20.5% / 4.4% / 0.6% at 100M, 39.1% / 15.6% / 3.8% at 1G, 58.9% / 34.5% / 14.5% at
10G. At depth 8: 90% / 73% / 44% at 100M, 98% / 96% / 88% at 10G.

## Reading

- **The counts still help at depth 16, and more text helps more.** Each tenfold
  of count text adds about 0.01 nats (0.053, 0.064, 0.072), the deep contexts
  filling in as the user expected. Seeds agree to 0.02 PPL on every doped arm.
- **Against own-count soft targets, the 100M gain is about depth 8's** (0.053
  against 0.058). The own counts alone are worth nothing at depth 16 by epoch 8
  (+0.001; a 0.010 speed-up at epochs 1-2), as at depth 8 (fading to −0.008).
- **But the doped slice no longer beats the bigger text.** At depth 8 the slice
  with 100M counts beat minibatch on the 100M text by 0.029 nats at equal steps
  (6.150 against 6.329); at depth 16 the 100M text leads the slice with 100M counts
  by 0.033 and with 10G counts by 0.014, and keeps improving with more steps
  (−0.022 more by 3x). Every slice arm is nearly flat from epoch 4 to 8 (hard
  targets −0.005 nats, 10G counts −0.008): at depth 16 the slice is short of
  contexts, the inputs, not only of their targets. A 1M text has
  ~1M distinct depth-16 contexts; the 100M text has ~100x as many. Doping refines
  the targets of the contexts the slice has and adds none.
- **So doping is a complement to more training text at depth 16, not a
  substitute.** The test that would show it stacking: a larger training text
  doped with 10G counts. Its table can't be dense ([contexts, 96] int32 is 3 GiB
  for the 1M slice's 8.6M contexts; a 10M text's would be ~30 GiB), so that needs
  the counts kept sparsely, in a trie: the trie-as-storage problem itself.

## Prior record at 20M (missed when this was written)

The "bigger doped text" test proposed above was already run, on imm's 20M
split (2026-11-18, air slot 90, air `5b007ee`, runs
`experiments/prior_scaling/runs/distill_s1.84m_d24_w*.json`): a 1.84M char
transformer (block 256) trained toward mobius's gated depth-24 rows counted
with 10G of PG-19 books as soft targets, scored with no prior. Held out
1.8359 / 1.8496 / 1.8890 bits/char at soft-target weight 0.25 / 0.5 / 0.75,
against 1.835 trained normally: no transfer. With the prior kept at test time
(input and logits), the 1 GB prior beat training on the 1 GB by 0.025
bits/char ([prior-scaling-plain-1g](../prior-scaling-plain-1g/), one seed;
a tie at 200M, [prior-scaling-plain-220m](../prior-scaling-plain-220m/)).

Soft targets are not the only training-only test there. Networks trained with
the prior and prior dropout 0.5 (half the sequences without it), then scored
with no prior, against the plain network (one seed each; air `bef9142`,
`aae6754`, air `.ai/memory/todo.md`; imm `.ai/memory/measurements.md`): air's
1.84M with the prior on its logits 3.4975 against plain 3.5689, **0.029
bits/char better** (0.036 with a fresh output layer); air's 1.84M with the prior
in its input 3.6274 (+0.023 worse); air's 10.8M logits 3.0489 against 3.0386
(level); air's 10.8M input 3.1057 (+0.031 worse); imm's h256 3.6177 with its
own head; mobius's carried-table 10.8M at block 1,024 (2026-11-22,
`rnd/carried-network`) 1.577 bits/char with no prior against the no-prior
network's 1.557 (+0.020 worse). So one route moved knowledge into the weights (prior on the logits
in training, at 1.84M); soft targets did not.

Read together: doping helps a ~100k-parameter model on a 1M text at depth 8 and
less so at depth 16, where the 100M text overtakes it, and nothing carries at
20M with a 1.84M model. One difference not tested: air's rows condition on at
most 24 characters while its network reads 256, so its soft targets are
for a shorter context than the model's input; here the counts' context is the
model's whole input.
