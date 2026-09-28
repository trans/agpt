---
title: RMSprop beta2 vs training length
kind: experiment
status: concluded
outcome: negative
question: >-
  Is RMSprop's slow beta2 transient the bottleneck in short AGPT runs, so that beta2=0.99
  at 10 SE matches beta2=0.999 at 100 SE?
answer: >-
  No. Training length dominates: 10 -> 100 SE lowers legacy sliding-window PPL by about 3.0
  (beta2=0.999: 9.245 -> 6.250, 3 seeds), while beta2=0.99 gains only 0.29 at 10 SE and 0.12
  at 100 SE. No escalation to 1000 SE was needed.
opened: 2026-05-21
updated: 2026-05-22
code: main
eval: legacy
family: attention
headline:
- {label: 'beta2=0.999, 100 SE (3-seed mean)', metric: 'legacy sliding-window PPL (d=16, 10k
    positions)', value: 6.25}
- {label: 'beta2=0.99, 100 SE (3-seed mean)', metric: 'legacy sliding-window PPL (d=16, 10k
    positions)', value: 6.134}
- {label: 'beta2=0.999, 10 SE (3-seed mean)', metric: 'legacy sliding-window PPL (d=16, 10k
    positions)', value: 9.245}
tags: [optimizer, cadence]
related: [depth-weight, runpod]
---

# RMSprop beta2 vs training length

> Stub README (2026-09-28): this directory predates the README convention. The
> summary below was reconstructed from its files; see them for detail.

**Question.** Is RMSprop's slow beta2 transient the bottleneck in short AGPT runs, so that beta2=0.99 at 10 SE matches beta2=0.999 at 100 SE?

**Answer.** No. Training length dominates: 10 -> 100 SE lowers legacy sliding-window PPL by about 3.0 (beta2=0.999: 9.245 -> 6.250, 3 seeds), while beta2=0.99 gains only 0.29 at 10 SE and 0.12 at 100 SE. No escalation to 1000 SE was needed.

- beta2=0.999, 100 SE (3-seed mean): legacy sliding-window PPL (d=16, 10k positions) = 6.25
- beta2=0.99, 100 SE (3-seed mean): legacy sliding-window PPL (d=16, 10k positions) = 6.134
- beta2=0.999, 10 SE (3-seed mean): legacy sliding-window PPL (d=16, 10k positions) = 9.245

**Sources.** summary.txt (per-cell means) and the commit message of 6f31595 ('SE dominates, beta2 marginal'); hypotheses H1/H2 are spelled out in run.sh.

**Caveats.** results.txt has empty heldout_ppl= fields (grep bug); the per-cell heldout_ppl.txt files and summary.txt hold the numbers. The eval file /tmp/shake_holdout.txt was later found to be the tail of the training corpus (microgpt memory project_heldout_methodology), so 'held-out' here is really in-distribution PPL. TRIAGE.md lists the dir as affected by the pre-2026-05-26 loss-normalization bug. 'd=16' in summary.txt is the trie depth; the model is d_model=64 L=2.
