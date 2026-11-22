---
title: Doped counts at depth 16
kind: experiment
status: active
outcome: n/a
question: >-
  The user (2026-11-22), after refined-trie-counts' depth-8 result: does a small corpus doped
  with a large text's next-character counts still come out ahead at depth 16, where most of
  the small corpus's contexts are rare even in 100M characters? And do 1G and 10G characters
  of counts fill in the deep contexts (the user expects the relative gain to shrink at 100M
  and the larger texts to be needed)?
answer: >-
  Not yet run.
opened: 2026-11-22
updated: 2026-11-22
code: main
eval: canonical
family: attention
tags: [refined-counts, doping, trie-as-storage]
related: [refined-trie-counts, dynamic-eval, noise-cadence]
---

# Doped counts at depth 16

**Status:** active (2026-11-22). Configs `make_configs.py`, tables `count.sh`, runs
`run.sh` (order `order.txt`).

## Setup

- **Training text:** refined-trie-counts' 1M PG-19 slice A (`data/pg19/a1m`), the
  same held-out chunks and vocabulary (96 symbols).
- **Count texts:** the 100M split, and a 10G text that continues it
  (`data/pg19/imm-split-10g`, built by `tools/build_pg19_big.py`: the 20M split's
  pieces, then 25,024 whole PG-19 train books in mobius's seeded order, none a
  piece's book, normalized as mobius's `pg19-normal.cr`). Its first 100,000,000
  characters are byte for byte the 100M split (the build checks the SHA-256), so
  the 100M, 1G and 10G texts are nested prefixes of one file.
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
