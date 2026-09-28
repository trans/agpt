---
title: CUDAX anc-grad parity
kind: diagnostic
status: concluded
outcome: n/a
question: >-
  Does the v2 (CUDAX) trainer's descendant-to-ancestor Wk/Wv gradient (anc-grad) reproduce
  the legacy trainer's held-out improvement over anc-grad off?
answer: >-
  Yes, directionally. With anc-grad on, legacy sliding-window PPL was lower in all three seeds
  (seed1 10.14 -> 9.06, seed2 9.00 -> 8.87, seed3 9.20 -> 8.46; 10 epochs RMSprop pd=1, d=16
  trie).
opened: 2026-05-21
updated: 2026-05-21
code: main
eval: legacy
family: attention
headline:
- {label: 'seed1, anc-grad off', metric: 'legacy sliding-window PPL (d=16, 10k positions)',
  value: 10.1351}
- {label: 'seed1, anc-grad on', metric: 'legacy sliding-window PPL (d=16, 10k positions)',
  value: 9.0585}
tags: [gradient, trainer, attention]
related: [anc-grad, per-fire-norm, legacy-rebaseline]
---

# CUDAX anc-grad parity

> Stub README (2026-09-28): this directory predates the README convention. The
> summary below was reconstructed from its files; see them for detail.

**Question.** Does the v2 (CUDAX) trainer's descendant-to-ancestor Wk/Wv gradient (anc-grad) reproduce the legacy trainer's held-out improvement over anc-grad off?

**Answer.** Yes, directionally. With anc-grad on, legacy sliding-window PPL was lower in all three seeds (seed1 10.14 -> 9.06, seed2 9.00 -> 8.87, seed3 9.20 -> 8.46; 10 epochs RMSprop pd=1, d=16 trie).

- seed1, anc-grad off: legacy sliding-window PPL (d=16, 10k positions) = 10.1351
- seed1, anc-grad on: legacy sliding-window PPL (d=16, 10k positions) = 9.0585

**Sources.** The 'Perplexity:' line of shakespeare/seed{1,2,3}_{on,off}/heldout.log, the 'anc-grad: enabled' banner in the train.log files, and the commit message of 2b913be ('parity probe data ... from codex-agpt's anc-grad alignment work').

**Caveats.** I inferred the purpose from logs and the commit message. The eval file is not named in the logs (50k tokens, vocab from data/input.txt); it is probably /tmp/shake_holdout.txt, which is the training tail (in-distribution). TRIAGE.md lists the dir as affected by the pre-2026-05-26 loss bug. The 2026-09-25 addendum in todo/descendant-ancestor-scatter.md shows the anc-grad scatter was itself truncated. anc-grad is on in these runs, so the truncated-ancestor-gradient caveat applies.
