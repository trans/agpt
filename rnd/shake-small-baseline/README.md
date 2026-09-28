---
title: Small Shakespeare baseline
kind: baseline
status: concluded
outcome: n/a
question: >-
  What held-out byte perplexity does the standard small AGPT recipe (d64 L2, depth-16 trie,
  10 static epochs, rmsprop 3e-3, anc-grad) reach on Shakespeare's 5% tail?
answer: >-
  Rolling byte PPL 10.435 (3.38 bits/byte) on the tail-heldout split, in 86 s of wall time.
  This was the first held-out run under the orchestrator.
opened: 2026-05-25
updated: 2026-05-26
code: main
eval: canonical
headline:
- {label: 'd64 L2, depth 16, 10 static epochs', metric: rolling byte PPL, value: 10.4353,
  run: 20260526T033625-d64l2-d16-10ep}
tags: [baseline]
related: [cudax-d124-probe]
---

# shake-small-baseline

**Status:** active

## Hypothesis

(fill in)

## Scope

(fill in)

## Results

<!-- agpt-experiment-table:start -->
| Run ID | byte_ppl | bits/byte | word_ppl | wall (s) |
|--------|---------:|----------:|---------:|---------:|
| `20260526T033625-d64l2-d16-10ep` | 10.4353 | 3.3834 | 494799.75 | 86.0 |
<!-- agpt-experiment-table:end -->

## Conclusion

(fill in once enough runs have landed)