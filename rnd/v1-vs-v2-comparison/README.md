---
title: v1 vs v2 trainer comparison
kind: diagnostic
status: concluded
outcome: inconclusive
question: >-
  How does the v1 trainer compare with v2 under the canonical evaluator, and how does v1 respond
  to mass weighting (linear/sqrt/log)?
answer: >-
  Never finished. The README has no hypothesis or conclusion, and only v1 runs landed (external
  held-out). Per rnd/TRIAGE.md these runs exposed the pre-fix per-prefix loss weighting (unweighted
  v1 100 SE: fixed-window PPL 6.36 at context 16 but 45.55 at context 1). That led to the
  2026-05-26 fix. The mw-linear runs used the wrong fire normalizer.
opened: 2026-05-26
updated: 2026-05-29
code: main
eval: canonical
headline:
- {label: 'v1, 100 SE, unweighted', metric: rolling byte PPL, value: 7.7672, run: 20260526T182401-v1-100se}
- {label: 'v1, 100 SE, mass-weight linear', metric: rolling byte PPL, value: 8.3198, run: 20260526T220612-v1-100se-mwlinear}
- {label: 'v1, 25 SE, unweighted', metric: rolling byte PPL, value: 9.2848, run: 20260526T182115-v1-25se}
tags: [trainer, gradient]
related: [cudax-static-epochs, cudax-exposure-fix, v2-compare, legacy-rebaseline]
---

# v1-vs-v2-comparison

**Status:** active

## Hypothesis

(fill in)

## Scope

(fill in)

## Results

<!-- agpt-experiment-table:start -->
| Run ID | fixed_token_ppl | rolling_byte_ppl | bits/byte | train (s) | total (s) |
|--------|----------------:|-----------------:|----------:|----------:|----------:|
| `20260526T182115-v1-25se` | 8.2815 | 9.2848 | 3.2149 | 141.0 | 165.0 |
| `20260526T182401-v1-100se` | 6.3645 | 7.7672 | 2.9574 | 556.0 | 578.0 |
| `20260526T184410-v1-25se-mwlinear` | 10.4075 | 10.6104 | 3.4074 | 144.0 | 168.0 |
| `20260526T184657-v1-25se-mwsqrt` | 8.3709 | 9.3515 | 3.2252 | 140.0 | 164.0 |
| `20260526T184941-v1-25se-mwlog` | 8.2955 | 9.1782 | 3.1982 | 141.0 | 163.0 |
| `20260526T220612-v1-100se-mwlinear` | 8.0097 | 8.3198 | 3.0566 | 572.0 | 598.0 |
<!-- agpt-experiment-table:end -->

## Conclusion

(fill in once enough runs have landed)