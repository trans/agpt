---
title: CUDAX Section 2 progressive reruns
kind: baseline
status: concluded
outcome: n/a
question: >-
  With the Section 2 event-count weighting restored in the CUDAX (v2) trainer, what held-out
  PPL do progressive-growth schedules (divisions × epochs) and static full-prefix training
  reach on Shakespeare?
answer: >-
  Re-baselined on the tail-heldout split. d64 L2 runs range from 8.25 rolling byte PPL (16x1to6)
  to 7.05 (16x25). At 6 epochs per stage, 64, 128 and 256 divisions (7.50, 7.35, 7.39) beat
  16 divisions (7.98). The best run is d128 L6 static at 200 epochs: 6.16 rolling byte PPL
  (5.65 fixed-window). The README never received a written conclusion.
opened: 2026-05-26
updated: 2026-05-27
code: main
eval: canonical
family: attention
headline:
- {label: 'd128 L6, static full-prefix, 200 epochs', metric: 'rolling byte PPL (lm-eval, tail-heldout)',
  value: 6.1636, run: 20260527T225128-section2-d128l6-static200}
- {label: 'd64 L2, 16 divisions × 25 epochs', metric: 'rolling byte PPL (lm-eval, tail-heldout)',
  value: 7.0509, run: 20260527T055358-section2-16x25}
- {label: 'd64 L2, 16 divisions × 6 epochs', metric: 'rolling byte PPL (lm-eval, tail-heldout)',
  value: 7.9789, run: 20260526T224149-section2-16x6}
tags: [baseline, trainer, scaling]
related: [progressive-growth-sgd-comparison, cudax-growth]
---

# cudax-section2-progressive

**Status:** concluded, n/a (reviewed 2026-09-28; this line previously read "active"). The answer is in the front matter above.

## Hypothesis

Re-run progressive-growth CUDAX benchmarks after restoring the Section 2
event-count weighting term. The previous progressive table is not comparable
because compressed radix queries were effectively weighted as unique rows
instead of corpus events.

## Results

<!-- agpt-experiment-table:start -->
| Run ID | fixed_token_ppl | rolling_byte_ppl | bits/byte | train (s) | total (s) |
|--------|----------------:|-----------------:|----------:|----------:|----------:|
| `20260526T224149-section2-16x6` | 7.567 | 7.9789 | 2.9962 | 313.0 | 345.0 |
| `20260526T224750-section2-64x6` | 7.0247 | 7.5008 | 2.9071 | 1179.0 | 1208.0 |
| `20260526T231511-section2-256x6` | 6.9358 | 7.3926 | 2.8861 | 4750.0 | 4777.0 |
| `20260527T025333-section2-16x10` | 7.0592 | 7.5558 | 2.9176 | 523.0 | 554.0 |
| `20260527T030528-section2-64x10` | 6.9576 | 7.4577 | 2.8987 | 1983.0 | 2014.0 |
| `20260527T034618-section2-16x3to10` | 7.293 | 7.7103 | 2.9468 | 378.0 | 408.0 |
| `20260527T035317-section2-64x3to10` | 6.8859 | 7.3954 | 2.8866 | 1423.0 | 1453.0 |
| `20260527T043319-section2-16x1to6` | 7.8375 | 8.2522 | 3.0448 | 213.0 | 243.0 |
| `20260527T044738-section2-16x1to10` | 7.6878 | 8.1091 | 3.0195 | 347.0 | 378.0 |
| `20260527T050713-section2-128x6` | 6.8494 | 7.3506 | 2.8779 | 2345.0 | 2374.0 |
| `20260527T055358-section2-16x25` | 6.5925 | 7.0509 | 2.8178 | 1265.0 | 1294.0 |
| `20260527T072703-section2-d128l6-16x25` | 6.1003 | 6.4484 | 2.6889 | 7621.0 | 7669.0 |
| `20260527T142250-section2-d128l6-static100` | 5.8981 | 6.4163 | 2.6818 | 4029.0 | 4080.0 |
| `20260527T192634-section2-d128l8-static200-cq25k` | 5.9332 | 6.4422 | 2.6876 | 10550.0 | 10605.0 |
| `20260527T225128-section2-d128l6-static200` | 5.6529 | 6.1636 | 2.6238 | 8162.0 | 8212.0 |
<!-- agpt-experiment-table:end -->

## Notes

Initial rerun set: `16x6`, `64x6`, `256x6`.
