---
title: Trie path-probability convergence
kind: diagnostic
status: concluded
outcome: n/a
question: >-
  Do path-probability products from tries built on two independent halves of a corpus agree,
  and how does agreement change with trie depth?
answer: >-
  Only ordinally. Spearman rank correlation between the halves stays at 0.80-0.91 through
  depth 16 on Shakespeare 1.1M and Gutenberg 5M (unfiltered paths), and this holds across
  split strategies. Absolute log-divergence grows roughly as d^1.5-d^2, not the predicted
  sqrt(d). The share of probability mass agreeing within 0.1 log-units falls from 27.9% at
  depth 4 to 0% at depth 16 (count>=5 filter).
opened: 2026-04-18
updated: 2026-04-21
code: main
eval: none
family: n/a
headline:
- {label: 'Shakespeare 1.1M, depth 16, count>=1, 20-block split', metric: 'Spearman rho, log
    path probability, half A vs half B', value: 0.89}
- {label: 'Shakespeare 1.1M, depth 8, count>=1, 20-block split', metric: 'Spearman rho, log
    path probability, half A vs half B', value: 0.803}
- {label: 'Gutenberg 5M, depth 8, count>=1, 20-block split', metric: 'Spearman rho, log path
    probability, half A vs half B', value: 0.799}
tags: [trie-structure, data]
---

# Trie path-probability convergence

> Stub README (2026-09-28): this directory predates the README convention. The
> summary below was reconstructed from its files; see them for detail.

**Question.** Do path-probability products from tries built on two independent halves of a corpus agree, and how does agreement change with trie depth?

**Answer.** Only ordinally. Spearman rank correlation between the halves stays at 0.80-0.91 through depth 16 on Shakespeare 1.1M and Gutenberg 5M (unfiltered paths), and this holds across split strategies. Absolute log-divergence grows roughly as d^1.5-d^2, not the predicted sqrt(d). The share of probability mass agreeing within 0.1 log-units falls from 27.9% at depth 4 to 0% at depth 16 (count>=5 filter).

- Shakespeare 1.1M, depth 16, count>=1, 20-block split: Spearman rho, log path probability, half A vs half B = 0.89
- Shakespeare 1.1M, depth 8, count>=1, 20-block split: Spearman rho, log path probability, half A vs half B = 0.803
- Gutenberg 5M, depth 8, count>=1, 20-block split: Spearman rho, log path probability, half A vs half B = 0.799

**Sources.** results-extended.md (Summary, Tables 1-3, section 7 Implications) and results-section.md (coverage-collapse table); experiment-notes.md is the original plan with the sqrt(d) and >90%-coverage predictions.

**Caveats.** Classified as diagnostic/n-a per the vocabulary (no training runs), but it did test explicit predictions: sqrt(d) error growth and >90% coverage were refuted, and the ordinal-fingerprint result held, so a reviewer could reasonably call it experiment/mixed. results-extended.md cites bin/convergence and rnd/convergence_analysis.py, but the script here is analysis.py. The *_mc1 and b2/b4/b100/b500 subdirs are untracked. The directory also contains a .docx paper draft.
