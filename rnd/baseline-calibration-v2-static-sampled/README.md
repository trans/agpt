# baseline-calibration-v2-static-sampled

Status: pd=0 and pd=1 calibration runs complete

## Question

Measure static v2 calibration runs with epoch checkpoints, using the standard
sampled heldout carve.

## Protocol

Shared settings: `bin/agpt_train_v2`, static full-trie training, `data/input.txt`,
sampled heldout 5% x 10 chunks seed 42, init
`data/seeds/shake-d64L2-h4-dff256-s128-seed42.model`, model architecture
`d_model=64`, `n_layers=2`, `n_heads=4`, `d_ff=256`, effective AGPT
depth/window 16, warmup-cosine with no warmup, `anc_grad=true`,
`chunk_queries=50000`, checkpoints at powers of two.

The seed filename's `s128` is the checkpoint header sequence length. These
runs reconcile that seed checkpoint to effective `train.max_depth=16` for AGPT
training and evaluation.

Varied settings are `partition_depth` and optimizer. `pd=0` is single-fire
training: one optimizer update per full trie epoch. `pd=1` fires once per
root-child group.

## Results

<!-- agpt-experiment-table:start -->
| Run ID | Config Delta | fixed_token_ppl | rolling_byte_ppl | bits/byte | train (s) | total (s) |
|--------|--------------|----------------:|-----------------:|----------:|----------:|----------:|
| `20260530T084221-d64l2-depth16-pd0-128ep` | `pd=0`, RMSProp `lr=0.003` `beta=0.999`, 128 epochs | 9.8714 | 10.0940 | 3.3354 | 727.0 | 950.0 |
| `20260530T164733-d64l2-depth16-pd1-128ep` | `pd=1`, RMSProp `lr=0.003` `beta=0.999`, 128 epochs | 5.1504 | 5.6884 | 2.5080 | 731.0 | 964.0 |
| `20260530T172139-d64l2-depth16-pd1-adam-128ep` | `pd=1`, Adam `lr=0.001` `beta1=0.9` `beta2=0.999`, 128 epochs | 4.8116 | 5.3522 | 2.4201 | 717.0 | 933.0 |
<!-- agpt-experiment-table:end -->

## Checkpoints

### pd=0

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

### pd=1 RMSProp

| Epoch | train (s) | fixed_token_ppl | rolling_byte_ppl | bits/byte |
|------:|----------:|----------------:|-----------------:|----------:|
| 1 | 5.8 | 16.4266 | 16.2453 | 4.0220 |
| 2 | 11.3 | 12.9021 | 12.9013 | 3.6894 |
| 4 | 22.3 | 11.3412 | 11.4398 | 3.5160 |
| 8 | 44.9 | 9.6562 | 9.9629 | 3.3166 |
| 16 | 89.7 | 8.4472 | 8.8159 | 3.1401 |
| 32 | 183.9 | 6.7333 | 7.2279 | 2.8536 |
| 64 | 370.6 | 5.6474 | 6.1836 | 2.6285 |
| 128 | 729.7 | 5.1504 | 5.6884 | 2.5080 |

### pd=1 Adam

| Epoch | train (s) | fixed_token_ppl | rolling_byte_ppl | bits/byte |
|------:|----------:|----------------:|-----------------:|----------:|
| 1 | 5.7 | 20.0170 | 19.4602 | 4.2825 |
| 2 | 11.2 | 14.7111 | 14.5636 | 3.8643 |
| 4 | 22.7 | 11.4704 | 11.5534 | 3.5302 |
| 8 | 45.4 | 9.1182 | 9.3240 | 3.2209 |
| 16 | 91.1 | 7.2226 | 7.5667 | 2.9197 |
| 32 | 182.0 | 5.9041 | 6.4115 | 2.6807 |
| 64 | 361.3 | 5.1128 | 5.6664 | 2.5024 |
| 128 | 715.6 | 4.8116 | 5.3522 | 2.4201 |
