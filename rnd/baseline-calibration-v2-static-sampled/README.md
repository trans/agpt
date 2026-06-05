# baseline-calibration-v2-static-sampled

Status: closed

## Question

Measure static v2 calibration runs with epoch checkpoints, using the standard
sampled heldout carve. This line ended as a calibration/optimizer sweep for
the current CUDAX depth-16 AGPT trainer.

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

Early runs vary `partition_depth` and optimizer. `pd=0` is single-fire
training: one optimizer update per full trie epoch. `pd=1` fires once per
root-child group.

Later d128/L6 runs use init
`rnd/cudax-section2-progressive/seeds/shake-d128L6-h8-dff512-d16-seed42.model`
with `d_model=128`, `n_layers=6`, `n_heads=8`, `d_ff=512`, `pd=1`,
`anc_grad=true`, sampled heldout, wrapped trie, and the same depth/window 16.

## Results

<!-- agpt-experiment-table:start -->
| Run ID | Config Delta | fixed_token_ppl | rolling_byte_ppl | bits/byte | train (s) | total (s) |
|--------|--------------|----------------:|-----------------:|----------:|----------:|----------:|
| `20260530T084221-d64l2-depth16-pd0-128ep` | `pd=0`, RMSProp `lr=0.003` `beta=0.999`, 128 epochs | 9.8714 | 10.0940 | 3.3354 | 727.0 | 950.0 |
| `20260530T164733-d64l2-depth16-pd1-128ep` | `pd=1`, RMSProp `lr=0.003` `beta=0.999`, 128 epochs | 5.1504 | 5.6884 | 2.5080 | 731.0 | 964.0 |
| `20260530T172139-d64l2-depth16-pd1-adam-128ep` | `pd=1`, Adam `lr=0.001` `beta1=0.9` `beta2=0.999`, 128 epochs | 4.8116 | 5.3522 | 2.4201 | 717.0 | 933.0 |
| `20260530T181858-d64l2-depth16-pd1-adam-lr0015-128ep` | `pd=1`, Adam `lr=0.0015` `beta1=0.9` `beta2=0.999`, 128 epochs | 4.6727 | 5.2191 | 2.3838 | 719.0 | 937.0 |
| `20260530T215135-d64l2-depth16-pd1-adam-lr0015-256ep` | `pd=1`, Adam `lr=0.0015` `beta1=0.9` `beta2=0.999`, 256 epochs | 4.3913 | 4.9552 | 2.3089 | 1459.0 | 1715.0 |
| `20260530T225246-d64l2-depth16-pd1-adam-lr0015-256ep-wrap` | `pd=1`, Adam `lr=0.0015` `beta1=0.9` `beta2=0.999`, 256 epochs, wrapped trie | 4.3868 | 4.9377 | 2.3038 | 1448.0 | 1697.0 |
| `20260603T192234-d64l2-depth16-pd1-adam-lr0015-1024ep-wrap` | `pd=1`, Adam `lr=0.0015` `beta1=0.9` `beta2=0.999`, 1024 epochs, wrapped trie | 4.2380 | 4.7991 | 2.2628 | 5787.6 | 6062.7 |
<!-- agpt-experiment-table:end -->

## d64/L2 1024-Epoch Followup

The 1024-epoch d64/L2 wrapped run was added after the phase-position runs
showed continued improvement at 512 epochs. It clarifies that the standard
d64/L2 line also had not reached its heldout floor at 512.

| Epoch | train (s) | fixed_token_ppl | rolling_byte_ppl | bits/byte |
|------:|----------:|----------------:|-----------------:|----------:|
| 128 | 725.8 | 4.6914 | 5.3249 | 2.4128 |
| 256 | 1462.4 | 4.4598 | 5.0975 | 2.3498 |
| 512 | 2893.4 | 4.2741 | 4.8762 | 2.2858 |
| 768 | 4322.9 | 4.2290 | 4.8000 | 2.2630 |
| 1024 | 5786.4 | 4.2380 | 4.7991 | 2.2628 |

The fixed-token PPL bottom in this run was epoch 768 (`4.2290`). Rolling byte
PPL and bits/byte improved slightly again at epoch 1024, but fixed-token PPL
regressed by about `0.0090`, so the practical d64/L2 bottom appears to be near
768-1024 epochs for this schedule.

## d128/L6 1024-Epoch Attempt

Run `20260603T212104-d128l6-depth16-pd1-adam-lr0010-1024ep-cq25k-wrap`
revisited the larger d128/L6 linear-mass setup with `chunk_queries=25000` and
planned checkpoints through 1024 epochs. Training failed inside epoch 506 before
the epoch-512 checkpoint:

`forward pass failed at epoch=506 unit=44/65 rc=627374 chunk=28/32:
non-finite forward loss`.

The partial run was evaluated with `bin/agpt_experiment --eval-run-dir`, which
skips missing future checkpoints. It produced checkpoint metrics through epoch
384 only.

| Epoch | train (s) | fixed_token_ppl | rolling_byte_ppl | bits/byte |
|------:|----------:|----------------:|-----------------:|----------:|
| 1 | 32.4 | 15.7264 | 15.6494 | 3.9680 |
| 2 | 64.8 | 11.6199 | 11.6944 | 3.5477 |
| 4 | 130.0 | 9.0332 | 9.2926 | 3.2161 |
| 8 | 260.2 | 6.8465 | 7.1993 | 2.8478 |
| 16 | 520.8 | 5.1968 | 5.7735 | 2.5295 |
| 32 | 1037.0 | 4.3978 | 5.0062 | 2.3237 |
| 64 | 2038.8 | 4.2976 | 4.8963 | 2.2917 |
| 96 | 3040.1 | 4.2808 | 4.8392 | 2.2748 |
| 112 | 3540.5 | 4.4488 | 4.9677 | 2.3126 |
| 128 | 4041.0 | 4.2040 | 4.7586 | 2.2505 |
| 144 | 4541.5 | 4.7223 | 5.1136 | 2.3543 |
| 160 | 5042.2 | 4.8792 | 5.2211 | 2.3844 |
| 192 | 6080.3 | 4.5810 | 4.9645 | 2.3116 |
| 256 | 8091.7 | 5.2111 | 5.3233 | 2.4123 |
| 384 | 12159.5 | 6.0552 | 5.7446 | 2.5222 |

Although train loss continued drifting down into the low `1.21` range by epoch
505, heldout PPL had already degraded sharply after the epoch-128 checkpoint.
This reinforces the earlier d128/L6 result: the useful checkpoint is early, not
late, and the old `4.1705` fixed-token PPL at epoch 128 remains the best result
in this line.

## Closeout

The best result in this line was the larger d128/L6 linear-mass run at epoch
128:

| Run ID | Checkpoint | Settings | fixed_token_ppl | rolling_byte_ppl | bits/byte |
|--------|-----------:|----------|----------------:|-----------------:|----------:|
| `20260601T064609-d128l6-depth16-pd1-adam-lr0010-512ep-wrap` | 128 | Adam `lr=0.0010`, linear mass | 4.1705 | 4.7086 | 2.2353 |

That run continued to 512 epochs but overfit badly after the epoch-128
checkpoint (`fixed_token_ppl=8.3467` at 512). The earlier high-LR d128/L6 run
with `lr=0.0015` also reached the same neighborhood at epoch 128
(`fixed_token_ppl=4.1827`) before degrading and later hitting the finite-loss
guard failure that motivated the optimizer hardening below.

Log-mass and Adam-stability followups were useful diagnostics but did not move
the floor:

| Run ID | Best checkpoint | Settings delta | fixed_token_ppl | rolling_byte_ppl | bits/byte |
|--------|----------------:|----------------|----------------:|-----------------:|----------:|
| `20260601T135307-d128l6-depth16-pd1-adam-lr0010-logmass-32ep-wrap` | 32 | log mass | 4.3866 | 5.8679 | 2.5529 |
| `20260601T144234-d128l6-depth16-pd1-adam-lr0010-eps1e4-logmass-128ep-wrap` | 32 | log mass, Adam `eps=1e-4` | 4.3193 | 5.7365 | 2.5202 |
| `20260601T163352-d128l6-depth16-pd1-adam-lr0005-eps1e4-logmass-128ep-cq25k-wr` | 64 | log mass, Adam `lr=0.0005`, `eps=1e-4`, `chunk_queries=25000` | 4.3753 | 5.8886 | 2.5579 |
| `20260601T180021-d128l6-depth16-pd1-adam-lr0010-eps1e4-wd001-logmass-128ep-wr` | 32 | log mass, Adam `eps=1e-4`, `weight_decay=0.01` | 4.2874 | 5.6025 | 2.4861 |

Interpretation:

- `pd=1` is the viable static CUDAX training mode for this baseline; `pd=0`
  converges much worse.
- Adam beats RMSProp for this setup, and `lr=0.0015` was best for d64/L2.
- Wrapping the corpus tail was mildly positive and removed the skewed tail
  construction issue.
- Progressive growth can approach the static d64/L2 result faster, but did not
  beat the best static run.
- d128/L6 reaches the best fixed-token PPL near epoch 128, but further training
  overfits sampled heldout.
- Log mass improves some early fixed-token checkpoints but substantially hurts
  byte/rolling PPL and does not beat the linear-mass d128/L6 result.
- Adam `eps` and decoupled `weight_decay` support are useful stability knobs,
  but neither overcame the depth-16 context ceiling in this sweep.

The line is closed. The next likely gains are structural rather than another
optimizer sweep: larger/effective context, position-aware RoPE using corpus
position histograms, or backoff/smoothing behavior closer to KN.

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

### pd=1 Adam lr0.0015

| Epoch | train (s) | fixed_token_ppl | rolling_byte_ppl | bits/byte |
|------:|----------:|----------------:|-----------------:|----------:|
| 1 | 5.9 | 17.2236 | 16.9480 | 4.0830 |
| 2 | 11.7 | 12.9081 | 12.9626 | 3.6963 |
| 4 | 23.3 | 10.3697 | 10.5094 | 3.3936 |
| 8 | 46.4 | 8.3754 | 8.6282 | 3.1091 |
| 16 | 91.9 | 6.7340 | 7.1651 | 2.8410 |
| 32 | 182.3 | 5.6562 | 6.1967 | 2.6315 |
| 64 | 362.6 | 4.9869 | 5.5415 | 2.4703 |
| 128 | 717.0 | 4.6727 | 5.2191 | 2.3838 |

### pd=1 Adam lr0.0015 256 epochs

| Epoch | train (s) | fixed_token_ppl | rolling_byte_ppl | bits/byte |
|------:|----------:|----------------:|-----------------:|----------:|
| 1 | 5.8 | 17.2128 | 16.9406 | 4.0824 |
| 2 | 11.6 | 12.8940 | 12.9548 | 3.6954 |
| 4 | 23.1 | 10.3828 | 10.5128 | 3.3941 |
| 8 | 46.3 | 8.3307 | 8.5728 | 3.0998 |
| 16 | 92.2 | 6.7242 | 7.1729 | 2.8426 |
| 32 | 184.4 | 5.6620 | 6.1997 | 2.6322 |
| 64 | 365.2 | 5.0817 | 5.6915 | 2.5088 |
| 128 | 728.5 | 4.6157 | 5.2144 | 2.3825 |
| 256 | 1457.9 | 4.3913 | 4.9552 | 2.3089 |

### pd=1 Adam lr0.0015 256 epochs wrapped trie

| Epoch | train (s) | fixed_token_ppl | rolling_byte_ppl | bits/byte |
|------:|----------:|----------------:|-----------------:|----------:|
| 1 | 6.1 | 17.2163 | 16.9440 | 4.0827 |
| 2 | 11.8 | 12.8970 | 12.9572 | 3.6957 |
| 4 | 23.3 | 10.3713 | 10.5089 | 3.3935 |
| 8 | 45.8 | 8.3080 | 8.5694 | 3.0992 |
| 16 | 90.9 | 6.7499 | 7.1564 | 2.8392 |
| 32 | 181.3 | 5.6472 | 6.1781 | 2.6272 |
| 64 | 361.1 | 5.0072 | 5.5873 | 2.4821 |
| 128 | 717.9 | 4.6037 | 5.1937 | 2.3768 |
| 256 | 1446.1 | 4.3868 | 4.9377 | 2.3038 |
