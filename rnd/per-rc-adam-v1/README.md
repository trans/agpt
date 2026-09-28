---
title: Per-root-child Adam state
kind: experiment
status: concluded
outcome: negative
question: >-
  Does localizing the optimizer second-moment state to root-child subtree buckets (per-rc
  Adam/RMSprop) improve AGPT training over one global state?
answer: >-
  No. Per-rc state regressed PPL about 19%: mean 5.46 -> 6.51 over 3 seeds (legacy bin/perplexity
  on the training corpus, 4096 positions; Shakespeare d=16, 50 SE, pd=1 RMSprop). Seed variance
  was 2.6x higher, and the gap stayed at about 20% without mass weighting. A dump of per-rc
  v showed real structured differences, so the next step proposed was spatial curvature estimation
  (Stage 2).
opened: 2026-05-18
updated: 2026-05-18
code: main
eval: legacy
family: attention
headline:
- {label: 'Global RMSprop state, 3-seed mean', metric: 'legacy PPL (bin/perplexity on training
    corpus, 4096 positions)', value: 5.46}
- {label: 'Per-rc RMSprop state, 3-seed mean', metric: 'legacy PPL (bin/perplexity on training
    corpus, 4096 positions)', value: 6.509}
tags: [optimizer, curvature, partitioning]
related: [agpt-optimizers]
---

# Per-root-child Adam state

> Stub README (2026-09-28): this directory predates the README convention. The
> summary below was reconstructed from its files; see them for detail.

**Question.** Does localizing the optimizer second-moment state to root-child subtree buckets (per-rc Adam/RMSprop) improve AGPT training over one global state?

**Answer.** No. Per-rc state regressed PPL about 19%: mean 5.46 -> 6.51 over 3 seeds (legacy bin/perplexity on the training corpus, 4096 positions; Shakespeare d=16, 50 SE, pd=1 RMSprop). Seed variance was 2.6x higher, and the gap stayed at about 20% without mass weighting. A dump of per-rc v showed real structured differences, so the next step proposed was spatial curvature estimation (Stage 2).

- Global RMSprop state, 3-seed mean: legacy PPL (bin/perplexity on training corpus, 4096 positions) = 5.46
- Per-rc RMSprop state, 3-seed mean: legacy PPL (bin/perplexity on training corpus, 4096 positions) = 6.509

**Sources.** findings.md (the Result, Verdict and Ablation sections), plus results.csv and no_mw_results.csv.

**Caveats.** findings.md cites notes/optimization/suffix_weighted_curvature.md for Stage 2. Pre-2026-05-26 loss fix (TRIAGE: affected). The updated date (05-28) is a notes-reorganization commit; the last substantive change was 05-18.
