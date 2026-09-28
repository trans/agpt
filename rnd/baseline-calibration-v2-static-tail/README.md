---
title: v2 static baseline, tail split
kind: baseline
status: concluded
outcome: n/a
question: >-
  How does the static v2 d64/L2 calibration setup score on a contiguous 5% tail held-out split,
  compared with the sampled multi-chunk split?
answer: >-
  The tail split scores worse. At 128 epochs: rolling byte PPL 6.46 and fixed-window PPL 5.94,
  against 5.22 and 4.67 for the matching sampled multi-chunk run.
opened: 2026-05-30
updated: 2026-05-30
code: main
eval: canonical
headline:
- {label: 'd64/L2 pd=1 Adam lr=0.0015, 128 epochs, tail split', metric: rolling byte PPL,
  value: 6.4617, run: 20260530T211622-d64l2-depth16-pd1-adam-lr0015-tail-128ep}
- {label: 'd64/L2 pd=1 Adam lr=0.0015, 128 epochs, tail split', metric: fixed-window PPL,
  value: 5.9387, run: 20260530T211622-d64l2-depth16-pd1-adam-lr0015-tail-128ep}
tags: [baseline, evaluation]
related: [baseline-calibration-v2-static-sampled]
---

# baseline-calibration-v2-static-tail

Status: tail comparison run complete

## Question

Measure the same static v2 calibration setup against a contiguous tail heldout
split, so the sampled multi-chunk baseline can be compared against the older
tail-only scoring convention.

## Protocol

Shared with `rnd/baseline-calibration-v2-static-sampled`: `bin/agpt_train_v2`,
static full-trie training, `data/input.txt`, init
`data/seeds/shake-d64L2-h4-dff256-s128-seed42.model`, model architecture
`d_model=64`, `n_layers=2`, `n_heads=4`, `d_ff=256`, effective AGPT
depth/window 16, warmup-cosine with no warmup, `anc_grad=true`,
`chunk_queries=50000`, checkpoints at powers of two.

The varied evaluation split is `corpus.carve.mode=tail`, ratio 5%.

## Results

<!-- agpt-experiment-table:start -->
| Run ID | Config Delta | fixed_token_ppl | rolling_byte_ppl | bits/byte | train (s) | total (s) |
|--------|--------------|----------------:|-----------------:|----------:|----------:|----------:|
| `20260530T211622-d64l2-depth16-pd1-adam-lr0015-tail-128ep` | `pd=1`, Adam `lr=0.0015` `beta1=0.9` `beta2=0.999`, 128 epochs | 5.9387 | 6.4617 | 2.6919 | 723.0 | 946.0 |
<!-- agpt-experiment-table:end -->

Comparable sampled multi-chunk run:
`20260530T181858-d64l2-depth16-pd1-adam-lr0015-128ep` scored
fixed-token PPL 4.6727, rolling byte PPL 5.2191, bits/byte 2.3838.

## Checkpoints

### pd=1 Adam lr0.0015 tail

| Epoch | train (s) | fixed_token_ppl | rolling_byte_ppl | bits/byte |
|------:|----------:|----------------:|-----------------:|----------:|
| 1 | 5.9 | 17.5777 | 17.0488 | 4.0916 |
| 2 | 11.7 | 13.2203 | 13.0206 | 3.7027 |
| 4 | 23.2 | 10.6255 | 10.7009 | 3.4197 |
| 8 | 46.3 | 8.7885 | 9.0156 | 3.1724 |
| 16 | 92.7 | 7.5458 | 7.9707 | 2.9947 |
| 32 | 186.1 | 6.8417 | 7.3170 | 2.8713 |
| 64 | 369.5 | 6.1092 | 6.6547 | 2.7344 |
| 128 | 721.9 | 5.9387 | 6.4617 | 2.6919 |
