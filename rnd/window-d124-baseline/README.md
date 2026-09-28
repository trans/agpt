---
title: Window baseline at seq_len 124
kind: baseline
status: concluded
outcome: n/a
question: >-
  What held-out PPL does a standard sliding-window microgpt model reach at seq_len 124 (d64
  L2, about 25 corpus epochs), as a reference for depth-124 AGPT?
answer: >-
  Rolling byte PPL 5.915 and fixed-token PPL 5.838 on the 5% tail heldout after 214,000 steps
  at constant lr 3e-4.
opened: 2026-05-28
updated: 2026-05-29
code: main
eval: canonical
family: attention
headline:
- {label: 'microgpt window d64/L2 seq 124, 214k steps', metric: rolling byte PPL (tail-heldout
    5%), value: 5.915, run: 20260528T094311-window-adam-d64l2-s124-25ep}
- {label: 'microgpt window d64/L2 seq 124, 214k steps', metric: fixed-window PPL (tail-heldout
    5%), value: 5.8381, run: 20260528T094311-window-adam-d64l2-s124-25ep}
tags: [baseline, context-length]
related: [cudax-d124-probe, radix-depth124, shake-sgd-baseline]
---

# window-d124-baseline

**Status:** concluded, n/a (reviewed 2026-09-28; this line previously read "active"). The answer is in the front matter above.

## Hypothesis

(fill in)

## Scope

(fill in)

## Results

<!-- agpt-experiment-table:start -->
| Run ID | fixed_token_ppl | rolling_byte_ppl | bits/byte | train (s) | total (s) |
|--------|----------------:|-----------------:|----------:|----------:|----------:|
| `20260528T094311-window-adam-d64l2-s124-25ep` | 5.8381 | 5.915 | 2.5644 | 4292.0 | 4330.0 |
<!-- agpt-experiment-table:end -->

## Conclusion

(fill in once enough runs have landed)