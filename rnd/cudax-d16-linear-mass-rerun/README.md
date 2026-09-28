---
title: CUDAX d16 linear-mass rerun
kind: baseline
status: concluded
outcome: n/a
question: >-
  What tail-heldout PPL does the v2 trainer's d=16 trie, d64/L2 static-epoch baseline reach
  with paper-correct linear mass weighting at 25, 100 and 250 epochs?
answer: >-
  100 epochs was best (rolling byte PPL 6.98). 250 epochs was slightly worse (7.15) and 25
  epochs worse (8.44). Two later reruns of the 100-epoch config under the migrated config
  schema gave 7.27 and 7.12. No conclusion was written.
opened: 2026-05-28
updated: 2026-05-29
code: main
eval: canonical
headline:
- {label: static 100 epochs, metric: rolling byte PPL (tail-heldout), value: 6.9773, run: 20260528T170108-d16-d64l2-static100}
- {label: static 250 epochs, metric: rolling byte PPL (tail-heldout), value: 7.1471, run: 20260528T171523-d16-d64l2-static250}
- {label: static 25 epochs, metric: rolling byte PPL (tail-heldout), value: 8.439, run: 20260528T165653-d16-d64l2-static25}
tags: [baseline, trainer]
related: [v1-vs-v2-comparison, cudax-d124-probe, cudax-growth-heldout-rerun]
---

# cudax-d16-linear-mass-rerun

**Status:** active

## Hypothesis

(fill in)

## Scope

(fill in)

## Results

<!-- agpt-experiment-table:start -->
| Run ID | byte_perplexity | bits/byte | train (s) | total (s) |
|--------|----------------:|----------:|----------:|----------:|
| `20260528T165653-d16-d64l2-static25` | — | — | 152.0 | 185.0 |
| `20260528T170108-d16-d64l2-static100` | — | — | 634.0 | 663.0 |
| `20260528T171523-d16-d64l2-static250` | — | — | 1510.0 | 1543.0 |
| `20260528T230843-d16-d64l2-static100` | — | — | 552.0 | 580.0 |
| `20260529T001452-d16-d64l2-static100` | — | — | 599.0 | 628.0 |
<!-- agpt-experiment-table:end -->

## Conclusion

(fill in once enough runs have landed)