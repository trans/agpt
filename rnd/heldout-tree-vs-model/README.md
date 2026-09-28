---
title: Held-out trie vs trained model
kind: experiment
status: concluded
outcome: negative
question: >-
  Does the trie alone (count lookup with backoff) predict held-out text as well as a trained
  AGPT model, i.e. is the trie doing the real work?
answer: >-
  No. On a 10% contiguous held-out tail of Gutenberg 5M, the 100-SE d64 L2 model scores legacy
  PPL 5.03 against 170.39 for naive-backoff trie lookup. Naive backoff overstates the 34x
  ratio, but the ordering is robust. A pd=1 shuffle ablation gave 5.29 (single seed, not distinguishable).
opened: 2026-05-18
updated: 2026-05-18
code: main
eval: legacy
headline:
- {label: 'trained model, 100 SE, pd=1', metric: 'legacy held-out PPL (4096 positions, seq
    16, Gutenberg 10% tail)', value: 5.03}
- {label: 'trie alone, naive backoff', metric: 'legacy held-out PPL (4096 positions, seq 16,
    Gutenberg 10% tail)', value: 170.39}
- {label: 'trained model, pd=1 with shuffle', metric: 'legacy held-out PPL (4096 positions,
    seq 16, Gutenberg 10% tail)', value: 5.29}
tags: [baseline, trie-structure, evaluation]
related: [streaming-agpt-v1, count-backoff-gate]
---

# Held-out trie vs trained model

> Stub README (2026-09-28): this directory predates the README convention. The
> summary below was reconstructed from its files; see them for detail.

**Question.** Does the trie alone (count lookup with backoff) predict held-out text as well as a trained AGPT model, i.e. is the trie doing the real work?

**Answer.** No. On a 10% contiguous held-out tail of Gutenberg 5M, the 100-SE d64 L2 model scores legacy PPL 5.03 against 170.39 for naive-backoff trie lookup. Naive backoff overstates the 34x ratio, but the ordering is robust. A pd=1 shuffle ablation gave 5.29 (single seed, not distinguishable).

- trained model, 100 SE, pd=1: legacy held-out PPL (4096 positions, seq 16, Gutenberg 10% tail) = 5.03
- trie alone, naive backoff: legacy held-out PPL (4096 positions, seq 16, Gutenberg 10% tail) = 170.39
- trained model, pd=1 with shuffle: legacy held-out PPL (4096 positions, seq 16, Gutenberg 10% tail) = 5.29

**Sources.** findings.md (Question, Result table, Interpretation, Caveats, Shuffle ablation); logs/model_ppl_heldout.log (5.0274) and logs/trie_ppl_heldout.log (170.3934).

**Caveats.** md is the writeup and a stub is needed. Outcome 'negative' refers to the tested hypothesis 'the trie does the work'. The finding that the trie does not generalize holds only for the naive-backoff evaluator, as the doc's own caveat 4 says. The later count-backoff-gate learned gate reaches heldout fixed PPL ~3.9 on Shakespeare. Model and trie artifacts lived in /tmp.
