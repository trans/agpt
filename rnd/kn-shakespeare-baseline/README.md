---
title: KN Shakespeare multi-chunk baseline
kind: baseline
status: concluded
outcome: n/a
question: >-
  What char-level KenLM KN PPL does Shakespeare reach by n-gram order on a 10-chunk random
  95/5 disjoint held-out split?
answer: >-
  The KN curve plateaus at order 7 (4.114). Orders 6-12 all land between 4.11 and 4.14, order
  5 gives 4.34 and order 3 gives 7.25. The directory also defines the reproducible multi-chunk
  split (k=10, 5%, seed 42) that the sweep used.
opened: 2026-05-28
updated: 2026-05-28
code: main
eval: legacy
headline:
- {label: KenLM KN order 7, metric: 'KenLM per-token PPL (multi-chunk held-out, seed 42)',
  value: 4.1143}
- {label: KenLM KN order 6, metric: 'KenLM per-token PPL (multi-chunk held-out, seed 42)',
  value: 4.1444}
- {label: KenLM KN order 8, metric: 'KenLM per-token PPL (multi-chunk held-out, seed 42)',
  value: 4.1196}
tags: [baseline, evaluation, data]
related: [kenlm-baseline, scale-vs-kn]
---

# KN Shakespeare multi-chunk baseline

> Stub README (2026-09-28): this directory predates the README convention. The
> summary below was reconstructed from its files; see them for detail.

**Question.** What char-level KenLM KN PPL does Shakespeare reach by n-gram order on a 10-chunk random 95/5 disjoint held-out split?

**Answer.** The KN curve plateaus at order 7 (4.114). Orders 6-12 all land between 4.11 and 4.14, order 5 gives 4.34 and order 3 gives 7.25. The directory also defines the reproducible multi-chunk split (k=10, 5%, seed 42) that the sweep used.

- KenLM KN order 7: KenLM per-token PPL (multi-chunk held-out, seed 42) = 4.1143
- KenLM KN order 6: KenLM per-token PPL (multi-chunk held-out, seed 42) = 4.1444
- KenLM KN order 8: KenLM per-token PPL (multi-chunk held-out, seed 42) = 4.1196

**Sources.** kn_sweep.txt (order/PPL table with split description and SHAs), manifest.json and build_multi_chunk_split.py docstring.

**Caveats.** Tokens scored (53,681) is less than the held-out size (55,760 chars), so newline/space handling differs from byte PPL; not comparable to canonical numbers. The train SHA (ab2abad...) does not match data/.splits/4fa9aec1db6b3aea (a4eb0b7...), which later canonical runs use. Memory's KN order-7 4.19 figure refers to that other split (rnd/kn-4fa9aec, which no longer exists), not this one. train_corpus.txt and heldout_corpus.txt exist locally but are untracked.
