---
title: CUDAX exposure fix
kind: diagnostic
status: concluded
outcome: n/a
question: >-
  Does CUDAX static full-prefix training behave correctly after restoring radix exposure and
  switching to Section 2 (paper) event-count weighting?
answer: >-
  No conclusion is recorded. The intermediate per-chunk-normalized patch run (static-25ep)
  is marked invalid for Section 2 parity. The Section 2 event-weighted runs scored rolling
  byte PPL 10.28 (10 ep), 8.80 (25 ep) and 7.06 (100 ep) on the 5% tail split. The weighting
  fix landed as commit 816f7d0.
opened: 2026-05-26
updated: 2026-05-26
code: main
eval: canonical
headline:
- {label: 'Static full-prefix, Section 2 weighting, 100 epochs', metric: rolling byte PPL,
  value: 7.0634, run: 20260526T192559-static-100ep-section2}
- {label: 'Static full-prefix, Section 2 weighting, 100 epochs', metric: fixed-window PPL,
  value: 6.6623, run: 20260526T192559-static-100ep-section2}
- {label: 'Static full-prefix, Section 2 weighting, 25 epochs', metric: rolling byte PPL,
  value: 8.7956, run: 20260526T191259-static-25ep-section2}
tags: [trainer, gradient]
related: [cudax-static-epochs, cudax-section2-progressive, v1-vs-v2-comparison]
---

# cudax-exposure-fix

**Status:** concluded, n/a (reviewed 2026-09-28; this line previously read "active"). The answer is in the front matter above.

## Hypothesis

(fill in)

## Scope

(fill in)

## Results

Note: `20260526T183923-static-25ep` was produced by the intermediate
per-chunk-normalized exposure patch. It is invalid for Section 2 parity
and should not be used as evidence.

<!-- agpt-experiment-table:start -->
| Run ID | fixed_token_ppl | rolling_byte_ppl | bits/byte | train (s) | total (s) |
|--------|----------------:|-----------------:|----------:|----------:|----------:|
| `20260526T183923-static-25ep` | 8.4962 | 8.9954 | 3.1692 | 157.0 | 186.0 |
| `20260526T191259-static-25ep-section2` | 8.5035 | 8.7956 | 3.1368 | 154.0 | 184.0 |
| `20260526T191714-static-10ep-section2` | 10.0784 | 10.2799 | 3.3618 | 66.0 | 93.0 |
| `20260526T192559-static-100ep-section2` | 6.6623 | 7.0634 | 2.8204 | 600.0 | 627.0 |
<!-- agpt-experiment-table:end -->

## Conclusion

(fill in once enough runs have landed)
