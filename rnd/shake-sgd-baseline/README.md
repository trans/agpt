---
title: Shakespeare SGD window baseline
kind: baseline
status: concluded
outcome: n/a
question: >-
  What held-out PPL does a standard sliding-window microgpt model (d64 L2, seq_len 16, 10k
  steps) reach under the AGPT experiment-harness split and evaluator?
answer: >-
  Rolling byte PPL 10.23 and fixed-token PPL 10.06 on the 5% tail heldout after 10,000 steps
  at constant lr 3e-4.
opened: 2026-05-30
updated: 2026-05-30
code: main
eval: canonical
family: attention
headline:
- {label: 'microgpt window d64/L2 seq 16, 10k steps', metric: rolling byte PPL (tail-heldout
    5%), value: 10.2344, run: 20260528T181839-d64l2-s16-10k}
- {label: 'microgpt window d64/L2 seq 16, 10k steps', metric: fixed-window PPL (tail-heldout
    5%), value: 10.0631, run: 20260528T181839-d64l2-s16-10k}
tags: [baseline]
related: [window-d124-baseline]
---

# shake-sgd-baseline

Status: initial baseline landed

## Question

Record a standard Crystal microgpt sliding-window baseline for the Shakespeare
small-model setup, using the same AGPT experiment harness split and evaluator.

## Protocol

- Trainer: `/home/trans/Projects/microgpt/bin/microgpt`
- Mode: standard sliding-window SGD
- Corpus: `data/input.txt`
- Training split: prefix 95%
- Evaluation split: held-out tail 5%
- Model init: `data/input.model`
- Window: `seq_len=16`
- Model: `d_model=64`, `n_layers=2`, `d_ff=256`
- Optimizer: SGD, `lr=0.0003`, constant schedule
- Budget: 10,000 steps

## Caveats

This run predates the stricter raw-log policy and did not originally write a
`result.json`; the committed `result.json` was reconstructed from `eval_raw.json`
and `meta.json`.

## Results

| Run ID | fixed_token_ppl | rolling_byte_ppl | bits/byte |
|--------|----------------:|-----------------:|----------:|
| `20260528T181839-d64l2-s16-10k` | 10.0631 | 10.2344 | 3.3553 |
