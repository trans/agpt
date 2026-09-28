---
title: CUDAX depth-124 probe
kind: experiment
status: concluded
outcome: inconclusive
question: >-
  Can the CUDAX v2 trainer train a d64 L2 model on the full depth-124 Shakespeare trie, and
  how does held-out PPL move with the number of static epochs?
answer: >-
  It trains. Held-out rolling byte PPL falls monotonically from 16.21 at 1 epoch to 7.47 at
  25 epochs (5226 s of training) and is still falling at the end. The README never recorded
  a hypothesis or conclusion, there is no depth-16 control in the directory, and the probe
  was not followed up here.
opened: 2026-05-28
updated: 2026-05-29
code: main
eval: canonical
headline:
- {label: 'depth 124, 1 static epoch', metric: rolling byte PPL, value: 16.2114, run: 20260528T060729-d124-d64l2-static1}
- {label: 'depth 124, 10 static epochs', metric: rolling byte PPL, value: 8.8797, run: 20260528T071446-d124-d64l2-static10}
- {label: 'depth 124, 25 static epochs', metric: rolling byte PPL, value: 7.4666, run: 20260528T081405-d124-d64l2-static25}
tags: [context-length, trie-structure, scaling]
related: [radix-depth124, window-d124-baseline, shake-small-baseline]
---

# cudax-d124-probe

**Status:** concluded, inconclusive (reviewed 2026-09-28; this line previously read "active"). The answer is in the front matter above.

## Hypothesis

(fill in)

## Scope

(fill in)

## Results

<!-- agpt-experiment-table:start -->
| Run ID | fixed_token_ppl | rolling_byte_ppl | bits/byte | train (s) | total (s) |
|--------|----------------:|-----------------:|----------:|----------:|----------:|
| `20260528T060729-d124-d64l2-static1` | 15.97 | 16.2114 | 4.0189 | 169.0 | 212.0 |
| `20260528T061312-d124-d64l2-static3` | 11.0397 | 11.2072 | 3.4864 | 851.0 | 897.0 |
| `20260528T064713-d124-d64l2-static6` | 9.5774 | 9.7179 | 3.2806 | 971.0 | 1021.0 |
| `20260528T071446-d124-d64l2-static10` | 8.6956 | 8.8797 | 3.1505 | 2051.0 | 2107.0 |
| `20260528T081405-d124-d64l2-static25` | 7.3041 | 7.4666 | 2.9005 | 5226.0 | 5289.0 |
<!-- agpt-experiment-table:end -->

## Conclusion

(fill in once enough runs have landed)