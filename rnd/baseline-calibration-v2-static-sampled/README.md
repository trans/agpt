# baseline-calibration-v2-static-sampled

Status: first calibration run complete

## Question

Measure a static v2 baseline as one calibration run with epoch checkpoints,
using the standard sampled heldout carve.

## Protocol

- Trainer: `bin/agpt_train_v2`
- Mode: static full-trie training
- Corpus source: `data/input.txt`
- Heldout: sampled 5%, 10 chunks, seed 42
- Model init: `data/seeds/shake-d64L2-h4-dff256-s128-seed42.model`
- Depth/window: 16
- Partition depth: 0
- Optimizer: RMSProp, `lr=0.003`, `beta=0.999`
- Schedule: warmup-cosine, no warmup
- Checkpoints: epochs 1, 2, 4, 8, 16, 32, 64, 128

## Results

<!-- agpt-experiment-table:start -->
| Run ID | byte_perplexity | bits/byte | train (s) | total (s) |
|--------|----------------:|----------:|----------:|----------:|
| `20260530T084221-d64l2-depth16-pd0-128ep` | 10.094 | 3.3354 | 727.0 | 950.0 |
<!-- agpt-experiment-table:end -->

## Checkpoint Trajectory

This is a `partition_depth=0` single-fire run: one optimizer update per full
trie epoch. It is a pd=0 calibration record, not an optimizer-step-budget match
for the older pd=1 static baselines.

| Epoch | fixed_token_ppl | rolling_byte_ppl | bits/byte |
|------:|----------------:|-----------------:|----------:|
| 1 | 86.9399 | 87.2196 | 6.4466 |
| 2 | 1148.6918 | 1135.4918 | 10.1491 |
| 4 | 168.6218 | 166.9190 | 7.3830 |
| 8 | 27.1075 | 26.8082 | 4.7446 |
| 16 | 18.1265 | 18.1461 | 4.1816 |
| 32 | 13.2884 | 13.3642 | 3.7403 |
| 64 | 10.6293 | 10.7694 | 3.4289 |
| 128 | 9.8714 | 10.0940 | 3.3354 |
