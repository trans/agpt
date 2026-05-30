# baseline-calibration-v2-static-sampled

Status: first calibration run complete

## Question

Measure a static v2 baseline as one calibration run with epoch checkpoints,
using the standard sampled heldout carve.

## Protocol

Shared settings: `bin/agpt_train_v2`, static full-trie training, `data/input.txt`,
sampled heldout 5% x 10 chunks seed 42, init
`data/seeds/shake-d64L2-h4-dff256-s128-seed42.model`, `d64 L2 h4 ff256`,
depth/window 16, RMSProp `lr=0.003` `beta=0.999`, warmup-cosine with no
warmup, `anc_grad=true`, `chunk_queries=50000`.

`partition_depth=0` means single-fire training: one optimizer update per full
trie epoch. These numbers are pd=0 calibration records, not
optimizer-step-budget matches for older pd=1 static baselines.

## Results

<!-- agpt-experiment-table:start -->
| Run ID | Config Delta | fixed_token_ppl | rolling_byte_ppl | bits/byte | train (s) | total (s) |
|--------|--------------|----------------:|-----------------:|----------:|----------:|----------:|
| `20260530T084221-d64l2-depth16-pd0-128ep` | `pd=0`, 128 epochs, checkpoints at powers of two | 9.8714 | 10.0940 | 3.3354 | 727.0 | 950.0 |
<!-- agpt-experiment-table:end -->

## Checkpoint Trajectory

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
