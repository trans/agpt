---
title: Window baseline, wall-matched
kind: baseline
status: concluded
outcome: n/a
question: >-
  What tail-heldout PPL does a standard window-trained transformer (d128 L6, seq_len=16, 180k
  steps) reach at roughly the wall time of CUDAX d128/L6 static200?
answer: >-
  Rolling byte PPL 6.96 and fixed-window 6.38. The comparison note places it behind CUDAX
  d128/L6 at tree depth 16 (static200: 6.16 rolling, 5.65 fixed), and both trail KenLM 8-gram
  on the same split (5.24 rolling).
opened: 2026-05-28
updated: 2026-05-28
code: main
eval: canonical
family: attention
headline:
- {label: 'window Adam d128 L6 seq16, 180k steps', metric: rolling byte PPL (tail-heldout),
  value: 6.9604, run: 20260528T014059-window-adam-d128l6-s16-180k}
- {label: 'window Adam d128 L6 seq16, 180k steps', metric: fixed-window PPL (tail-heldout),
  value: 6.3818, run: 20260528T014059-window-adam-d128l6-s16-180k}
tags: [baseline]
related: [cudax-section2-progressive, window-d124-baseline, kn-shakespeare-baseline]
---

# Window baseline, wall-matched

> Stub README (2026-09-28): this directory predates the README convention. The
> summary below was reconstructed from its files; see them for detail.

**Question.** What tail-heldout PPL does a standard window-trained transformer (d128 L6, seq_len=16, 180k steps) reach at roughly the wall time of CUDAX d128/L6 static200?

**Answer.** Rolling byte PPL 6.96 and fixed-window 6.38. The comparison note places it behind CUDAX d128/L6 at tree depth 16 (static200: 6.16 rolling, 5.65 fixed), and both trail KenLM 8-gram on the same split (5.24 rolling).

- window Adam d128 L6 seq16, 180k steps: rolling byte PPL (tail-heldout) = 6.9604
- window Adam d128 L6 seq16, 180k steps: fixed-window PPL (tail-heldout) = 6.3818

**Sources.** The run's eval_recovered.json metrics and config.yml description; the comparison and recovery story are in notes/trainer/window-baseline-vs-cudax-section2.md.

**Caveats.** The run dir has no result.json, so by CLAUDE.md it is not canonical; the numbers come from a recovered lm-eval rerun. notes/trainer/window-baseline-vs-cudax-section2.md says to treat this as a window-transformer baseline, not a clean SGD baseline (the config says adam lr 3e-4 constant, trainer microgpt). The KenLM 5.24 reference in the answer comes from that note and uses the tail split, not kn-shakespeare-baseline's multi-chunk split. The comparison CUDAX numbers live in rnd/cudax-section2-progressive.
