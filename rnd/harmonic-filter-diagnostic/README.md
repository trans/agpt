---
title: Harmonic filter diagnostic
kind: diagnostic
status: concluded
outcome: n/a
question: >-
  Do position-distribution 'chord' features of trie substrings separate on-path from off-path
  query/key pairs, and which operator and frequency set separates them best?
answer: >-
  Pooled over all pairs, symmetric chord-vs-chord scores (E3/E4) separate weakly or not at
  all. Stratified by key mass, the signal sits at low-mass keys. The asymmetric K-at-p_Q operator
  with DFT frequencies separates best: at key mass 2-9 (HD=48, W=64) the gap is 1.60 pooled-IQR
  units on Shakespeare and 1.68 on Gutenberg, falling to about 0.04 at mass ≥1000.
opened: 2026-05-25
updated: 2026-05-25
code: main
eval: none
family: n/a
headline:
- {label: 'ASYM DFT operator, HD=48 W=64, Shakespeare, key mass 2-9', metric: on/off-path
    separation (pooled-IQR units), value: 1.6}
- {label: 'ASYM DFT operator, HD=48 W=64, Gutenberg, key mass 2-9', metric: on/off-path separation
    (pooled-IQR units), value: 1.68}
tags: [position, attention, trie-structure]
related: [harmonic-bias-prototype]
---

# Harmonic filter diagnostic

> Stub README (2026-09-28): this directory predates the README convention. The
> summary below was reconstructed from its files; see them for detail.

**Question.** Do position-distribution 'chord' features of trie substrings separate on-path from off-path query/key pairs, and which operator and frequency set separates them best?

**Answer.** Pooled over all pairs, symmetric chord-vs-chord scores (E3/E4) separate weakly or not at all. Stratified by key mass, the signal sits at low-mass keys. The asymmetric K-at-p_Q operator with DFT frequencies separates best: at key mass 2-9 (HD=48, W=64) the gap is 1.60 pooled-IQR units on Shakespeare and 1.68 on Gutenberg, falling to about 0.04 at mass ≥1000.

- ASYM DFT operator, HD=48 W=64, Shakespeare, key mass 2-9: on/off-path separation (pooled-IQR units) = 1.6
- ASYM DFT operator, HD=48 W=64, Gutenberg, key mass 2-9: on/off-path separation (pooled-IQR units) = 1.68

**Sources.** notes/seq-len-extension/harmonic-filter-asymmetric.md ('diagnostic-validated 2026-05-25', stratified tables); stratified/shake_dft_HD48_W64.txt and gut_dft_HD48_W64.txt 'ASYM aggregate score' rows; shakespeare*/summary.md E3/E4 all-pairs separations (-0.03 to +0.23).

**Caveats.** The headline values are in stratified/*.The training follow-up (harmonic-bias-prototype) is listed in rnd/TRIAGE.md as a null result. The design notes live in notes/seq-len-extension/harmonic-filter-*.md.
