---
title: Streaming AGPT on Gutenberg (overnight run)
kind: experiment
status: concluded
outcome: positive
question: >-
  Does streaming AGPT (100 stages x 5 SE) beat single-stage 500-SE training on Gutenberg 5M
  at depth 16, as it did on Shakespeare?
answer: >-
  Yes. Mean legacy PPL is 4.083 vs 4.365 (-6.46%, 3 seeds each, Welch t = -3.08), with every
  seed winning. That is about 3x the Shakespeare margin (-2.1%), with higher seed variance.
opened: 2026-05-18
updated: 2026-05-18
code: main
eval: legacy
headline:
- {label: 'streaming 100 x 5 SE, mean of 3 seeds', metric: 'legacy PPL (bin/perplexity, 4096
    positions, seq 16, Gutenberg 5M)', value: 4.0831}
- {label: 'single-stage 500 SE baseline, mean of 3 seeds', metric: 'legacy PPL (bin/perplexity,
    4096 positions, seq 16, Gutenberg 5M)', value: 4.3652}
- {label: 'streaming, best seed (300)', metric: 'legacy PPL (bin/perplexity, 4096 positions,
    seq 16, Gutenberg 5M)', value: 3.9878}
tags: [trainer, data]
related: [streaming-agpt-v1, gutenberg-5m, heldout-tree-vs-model]
---

# Streaming AGPT on Gutenberg (overnight run)

> Stub README (2026-09-28): this directory predates the README convention. The
> summary below was reconstructed from its files; see them for detail.

**Question.** Does streaming AGPT (100 stages x 5 SE) beat single-stage 500-SE training on Gutenberg 5M at depth 16, as it did on Shakespeare?

**Answer.** Yes. Mean legacy PPL is 4.083 vs 4.365 (-6.46%, 3 seeds each, Welch t = -3.08), with every seed winning. That is about 3x the Shakespeare margin (-2.1%), with higher seed variance.

- streaming 100 x 5 SE, mean of 3 seeds: legacy PPL (bin/perplexity, 4096 positions, seq 16, Gutenberg 5M) = 4.0831
- single-stage 500 SE baseline, mean of 3 seeds: legacy PPL (bin/perplexity, 4096 positions, seq 16, Gutenberg 5M) = 4.3652
- streaming, best seed (300): legacy PPL (bin/perplexity, 4096 positions, seq 16, Gutenberg 5M) = 3.9878

**Sources.** findings.md TL;DR and per-seed table; run.log per-seed PPLs; run.sh/summarize.sh show bin/perplexity scoring.

**Caveats.** findings.md names rnd/streaming-agpt-v1/findings.md as the canonical writeup, so this dir could be folded into it. PPL was scored with bin/perplexity on data/gutenberg_5m.txt, the corpus the trie was built from (in-sample, not held-out). heldout-tree-vs-model's claim that this is a generalization win is an inference. Baseline seeds 200/300 ran on RunPod 2026-05-17 and seed 100 on the laptop. findings.md says 'no wall savings', but run.log shows the laptop baseline seed at 10771 s vs ~6312 s per streaming seed. The streaming std is ±0.1353 in findings.md vs ±0.1105 in run.log (sample vs population std).
