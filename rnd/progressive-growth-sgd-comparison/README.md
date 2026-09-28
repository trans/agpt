---
title: Progressive growth vs SGD
kind: experiment
status: concluded
outcome: inconclusive
question: >-
  How does CUDAX progressive-growth AGPT at several division × epoch schedules compare with
  a µGPT sliding-window SGD baseline of the same model size and context length?
answer: >-
  Every CUDAX schedule beat the preliminary SGD baseline (seq 16, 10k steps) on tail-heldout:
  fixed-window PPL 6.72-9.27 vs 10.06, rolling byte PPL 8.27-9.96 vs 10.23. But the baseline
  was not matched on any agreed budget, and the CUDAX runs predate the Section 2 event-weighting
  fix. At 64+ divisions rolling byte PPL got worse as fixed-window PPL improved; dense-causal-profile.md
  traces this to poor predictions at the start of each 16-token window.
opened: 2026-05-26
updated: 2026-05-30
code: main
eval: canonical
headline:
- {label: CUDAX 16 divisions × 6 epochs, metric: 'rolling byte PPL (lm-eval, tail-heldout)',
  value: 8.2664, run: 20260526T133412-cudax-16x6}
- {label: CUDAX 64 divisions × 6 epochs, metric: fixed-window PPL (tail-heldout), value: 6.7244,
  run: 20260526T064328-cudax-64x6}
- {label: 'µGPT SGD seq=16, 10k steps', metric: 'rolling byte PPL (lm-eval, tail-heldout)',
  value: 10.2344, run: 20260526T153756-sgd-s16-10k}
tags: [baseline, evaluation, trainer]
related: [cudax-section2-progressive, sgd-sanity-check]
superseded_by: [cudax-section2-progressive]
---

# Progressive Growth vs SGD Baseline

Status: initial run set complete

## Question

Compare CUDAX progressive growth at several division/epoch schedules against a
standard Crystal µGPT sliding-window baseline at the same model size and context
length.

## Protocol

Each run is a subdirectory under this directory. The copied `config.yml`,
`resolved_config.json`, `meta.json`, `eval_raw.json`, and `result.json` in each
run directory are the committed canonical record for that run. Raw logs and
checkpoints remain local debugging artifacts.

## Runs

| Run | Purpose |
|---|---|
| `cudax-16x1` | 16 progressive divisions, 1 epoch per stage |
| `cudax-16x3` | 16 progressive divisions, 3 epochs per stage |
| `cudax-16x6` | 16 progressive divisions, 6 epochs per stage |
| `cudax-64x1` | 64 progressive divisions, 1 epoch per stage |
| `cudax-64x3` | 64 progressive divisions, 3 epochs per stage |
| `cudax-64x6` | 64 progressive divisions, 6 epochs per stage |
| `cudax-256x6` | 256 progressive divisions, 6 epochs per stage |
| `sgd-s16-10k` | Crystal µGPT SGD baseline, seq_len=16, 10k steps |

## Caveats

- `train (s)` is the primary timing column. For CUDAX runs it is the sum of
  trainer-reported `growth-stage-timing` totals. For the µGPT run it is
  approximated from the train log close time because the old µGPT binary did
  not report elapsed training time.
- `total (s)` is full harness time: split, training, HF conversion, rolling
  evaluation, fixed-token evaluation, and aggregation.
- Some runs overlapped with other GPU/CPU work, so timing is noisy. Treat PPL
  as the primary result from this batch.
- The SGD baseline is preliminary. A fair baseline needs explicit agreement on
  matching criterion: wall time, optimizer steps, target-token exposures, or
  some combination of these.

## Results

<!-- agpt-experiment-table:start -->
| Run ID | fixed_token_ppl | rolling_byte_ppl | bits/byte | train (s) | total (s) |
|--------|----------------:|-----------------:|----------:|----------:|----------:|
| `20260526T055319-cudax-64x1` | 7.2606 | 8.6681 | 3.1157 | 216 | 249 |
| `20260526T055745-cudax-64x3` | 6.8696 | 8.9280 | 3.1583 | 769 | 802 |
| `20260526T064328-cudax-64x6` | 6.7244 | 9.0170 | 3.1727 | 1681 | 1711 |
| `20260526T073057-cudax-16x1` | 9.2729 | 9.9573 | 3.3158 | 57 | 86 |
| `20260526T083945-cudax-16x3` | 7.4665 | 8.6196 | 3.1076 | 158 | 184 |
| `20260526T133412-cudax-16x6` | 6.8595 | 8.2664 | 3.0473 | 306 | 334 |
| `20260526T135924-cudax-256x6` | 6.7331 | 9.2635 | 3.2116 | 4604 | 4634 |
| `20260526T153756-sgd-s16-10k` | 10.0631 | 10.2344 | 3.3553 | 66 | 88 |
<!-- agpt-experiment-table:end -->
