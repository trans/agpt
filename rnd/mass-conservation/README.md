---
title: Mass conservation and depth cap
kind: diagnostic
status: concluded
outcome: n/a
question: >-
  Does the trie depth cap distort path-probability convergence beyond the ordinary noise of
  mass-1 paths?
answer: >-
  No. In the Shakespeare d=16 shared-path data, max-depth paths with mass ≥2 converge about
  as well as shallow paths with mass ≥2 (Spearman ρ 0.9095 vs 0.9211). Deep paths look noisy
  only because most of them are mass-1. The notes also give node mass conservation: count
  = child-edge counts + terminal count + cutoff count.
opened: 2026-04-21
updated: 2026-04-21
code: main
eval: none
headline:
- {label: 'max-depth paths (d=16), mass ≥ 2', metric: Spearman ρ of log path probability between
    paired tries, value: 0.9095}
- {label: 'shallow paths (depth ≤ 8), mass ≥ 2', metric: Spearman ρ of log path probability
    between paired tries, value: 0.9211}
tags: [trie-structure, data]
related: [convergence]
---

# Mass conservation and depth cap

> Stub README (2026-09-28): this directory predates the README convention. The
> summary below was reconstructed from its files; see them for detail.

**Question.** Does the trie depth cap distort path-probability convergence beyond the ordinary noise of mass-1 paths?

**Answer.** No. In the Shakespeare d=16 shared-path data, max-depth paths with mass ≥2 converge about as well as shallow paths with mass ≥2 (Spearman ρ 0.9095 vs 0.9211). Deep paths look noisy only because most of them are mass-1. The notes also give node mass conservation: count = child-edge counts + terminal count + cutoff count.

- max-depth paths (d=16), mass ≥ 2: Spearman ρ of log path probability between paired tries = 0.9095
- shallow paths (depth ≤ 8), mass ≥ 2: Spearman ρ of log path probability between paired tries = 0.9211

**Sources.** findings.md question/answer, results table and interpretation; notes.md conservation-law derivation; mass_filter_experiment.py reads rnd/convergence shared-path CSVs.

**Caveats.** md. rnd/TRIAGE.md lists this directory under AFFECTED-BY-BUG as 'directly relevant — redo'. But the analysis uses trie/corpus statistics only, with no trained model, and rnd/README.md calls it 'not obviously trainer-dependent', so the loss bug should not apply. The uniform-mass follow-up was not run. findings.md's reproduce command points at rnd/mass_filter_experiment.py, but the script is in rnd/mass-conservation/.
