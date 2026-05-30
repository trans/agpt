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
