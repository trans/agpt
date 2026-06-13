# Stochastic AGPT

Status: active.

Purpose: verify the CUDA v2 attention-based AGPT path before running a larger
attention experiment.

Canonical smoke:

```text
d_model=64
n_layers=2
n_heads=4
d_ff=256
seq_len=16
max_depth=16
partition_depth=1
epochs=1
optimizer=Adam
lr=0.0015
chunk_queries=50000
```

Run:

```bash
bin/agpt_experiment \
  --config rnd/stochastic-agpt/d64L2-depth16-pd1-1ep.yml \
  --trainer v2
```

Notes:

- This uses `bin/agpt_train_v2`, the attention AGPT trainer.
- The init checkpoint must have `seq_len=16`; using the older `s128` seed is
  rejected by the trainer header check.
- If running under a sandbox, CUDA may be hidden. In that case run outside the
  sandbox or on the GPU pod.

## Runs

### 20260611T155913-d64l2-depth16-pd1-1ep

Result: pass.

```text
run dir: rnd/stochastic-agpt/20260611T155913-d64l2-depth16-pd1-1ep
train wall: 7.1s
total wall: 54.0s
lm_eval_rolling_byte_perplexity: 21.3199
lm_eval_rolling_bits_per_byte: 4.4141
agpt_fixed_token_perplexity: 21.3901
```

This validates the d64/L2 attention `f_theta` path at depth/seq_len 16 with
`pd=1`, including CUDA training, checkpoint save, HF conversion, lm-eval
rolling byte PPL, and AGPT fixed-token PPL.

### 20260611T160456-d64l2-depth16-pd1-100ep

Result: pass.

```text
run dir: rnd/stochastic-agpt/20260611T160456-d64l2-depth16-pd1-100ep
train wall: 582.1s
total wall: 821.3s
final lm_eval_rolling_byte_perplexity: 5.3359
final lm_eval_rolling_bits_per_byte: 2.4157
final agpt_fixed_token_perplexity: 4.7929
```

Checkpoint curve:

| epoch | train loss | train wall | rolling byte PPL | fixed-token PPL |
|------:|-----------:|-----------:|-----------------:|----------------:|
| 10 | 2.1013 | 57.7s | 8.0352 | 7.6297 |
| 20 | 1.9141 | 115.8s | 6.7435 | 6.2399 |
| 30 | 1.8251 | 173.8s | 6.2396 | 5.6777 |
| 40 | 1.7732 | 232.0s | 5.9379 | 5.3691 |
| 50 | 1.7331 | 290.0s | 5.7229 | 5.1300 |
| 60 | 1.7086 | 348.1s | 5.5780 | 5.0118 |
| 70 | 1.6833 | 406.3s | 5.4450 | 4.8974 |
| 80 | 1.6705 | 464.5s | 5.3737 | 4.8274 |
| 90 | 1.6631 | 522.6s | 5.3466 | 4.8017 |
| 100 | 1.6600 | 580.7s | 5.3359 | 4.7929 |

The curve is still improving at 100 epochs, but the marginal gain from 80 to
100 is small. This is the pd1 baseline for the matching pd0 run.

### 20260611T162130-d64l2-depth16-pd0-100ep

Result: pass.

```text
run dir: rnd/stochastic-agpt/20260611T162130-d64l2-depth16-pd0-100ep
train wall: 583.9s
total wall: 861.2s
final lm_eval_rolling_byte_perplexity: 12.0540
final lm_eval_rolling_bits_per_byte: 3.5914
final agpt_fixed_token_perplexity: 12.1095
```

Checkpoint curve:

| epoch | train loss | train wall | rolling byte PPL | fixed-token PPL |
|------:|-----------:|-----------:|-----------------:|----------------:|
| 10 | 3.2422 | 56.2s | 24.3748 | 24.7057 |
| 20 | 2.9805 | 112.8s | 19.1189 | 19.4395 |
| 30 | 2.8021 | 170.5s | 16.1471 | 16.3316 |
| 40 | 2.6765 | 229.3s | 14.3620 | 14.4748 |
| 50 | 2.5976 | 287.9s | 13.3405 | 13.4290 |
| 60 | 2.5467 | 346.8s | 12.7111 | 12.7997 |
| 70 | 2.5151 | 405.7s | 12.3394 | 12.4125 |
| 80 | 2.4975 | 464.7s | 12.1434 | 12.2039 |
| 90 | 2.4902 | 523.6s | 12.0664 | 12.1226 |
| 100 | 2.4888 | 582.6s | 12.0540 | 12.1095 |

## pd0 vs pd1 at 100 Epochs

| partition depth | train wall | optimizer steps | train loss | rolling byte PPL | fixed-token PPL |
|----------------:|-----------:|----------------:|-----------:|-----------------:|----------------:|
| pd0 | 583.9s | 100 | 2.4888 | 12.0540 | 12.1095 |
| pd1 | 582.1s | ~6,500 | 1.6600 | 5.3359 | 4.7929 |

The wall time is essentially the same, but the optimizer cadence is not. With
pd0, Adam only gets one update per full-tree epoch; with pd1, it gets one update
per unigram subtree, about 65 updates per epoch. This run is strong evidence
that plain Adam is a poor fit for full-tree pd0 training at this scale unless
the optimizer/update rule is changed.

## pd6 Descendant Sweep Control

Motivation: compare stochastic full-subtree sampling against a deterministic
fine-grained sweep over all depth-6 anchors.

Implementation note: v2 now supports `partition_depth > 1` by enumerating radix
nodes whose edge covers the requested depth, then training that anchor plus its
full descendant subtree. Ancestor-prefix rows are included only to seed/read the
K/V cache and are marked context-only, so their query weights are zero.

| run | ancestor treatment | train wall | trained query passes | trained events | rolling byte PPL | fixed-token PPL |
|---|---|---:|---:|---:|---:|---:|
| `20260612T182037-d64l2-depth16-pd6-desc-1ep-cosfloor10` | trainable | 252.0s | 10,079,007 | 15,155,457,723 | 14.9605 | 14.3885 |
| `20260612T183010-d64l2-depth16-pd6-desc-loss6-1ep-cosfloor10` | depth filter 6..16 | 252.0s | n/a | n/a | 6.4771 | 5.2039 |
| `20260612T195303-d64l2-depth16-pd6-desc-contextonly-1ep-cosfloor10` | context-only node flags | 258.9s | 8,813,779 | 12,027,251 | 6.3355 | 5.1450 |

The first row is invalid as a clean pd6 comparison because it repeatedly trains
the shallow ancestor closure across 275k units. The context-only row is the clean
control, and it is still far behind pd1 and Lightning random-descendants. A
single deterministic pd6 pass produces many optimizer updates but too little
effective target mass per update for this v2 attention setup.

## Fixed-Depth Loss Control

Motivation: test whether the rolling-vs-fixed eval gap is partly caused by
mixing all query depths during training. The control keeps the same depth-16
tree, `seq_len=16`, `pd=1`, Adam settings, and checkpoint cadence, but only one
query depth contributes direct cross-entropy loss.

Important: the first depth-filter attempts below were run before
`bin/agpt_train_v2` was rebuilt with `experimental.loss_depth_min/max` support.
Those runs ignored the depth filter and must not be treated as fixed-depth
results. After rebuilding, a one-unit check showed the filter is active:
`loss_depth=1` trained 1 query row in the first unit, while `loss_depth=7`
trained 9,521 query rows in the same unit.

The first attempted `loss_depth=2` run was stopped after epoch 30 so the first
single-depth probe could target the more plausible middle depth. That run is
also invalid for fixed-depth analysis because it used the stale trainer.

### 20260611T173031-d64l2-depth16-pd1-100ep-lossdepth7

Result: invalid for fixed-depth analysis; stale trainer ignored
`experimental.loss_depth_min/max`.

```text
run dir: rnd/stochastic-agpt/20260611T173031-d64l2-depth16-pd1-100ep-lossdepth7
loss depth: 7 only
train wall: 610.3s
total wall: 886.9s
final train loss: 1.6587
final lm_eval_rolling_byte_perplexity: 5.3115
final lm_eval_rolling_bits_per_byte: 2.4091
final agpt_fixed_token_perplexity: 4.7700
```

Checkpoint curve:

| epoch | train loss | train wall | rolling byte PPL | fixed-token PPL |
|------:|-----------:|-----------:|-----------------:|----------------:|
| 10 | 2.1058 | 57.7s | 7.9839 | 7.5755 |
| 20 | 1.9103 | 121.8s | 6.7217 | 6.2057 |
| 30 | 1.8284 | 184.4s | 6.2329 | 5.6944 |
| 40 | 1.7697 | 248.3s | 5.8744 | 5.3263 |
| 50 | 1.7359 | 305.8s | 5.6803 | 5.1088 |
| 60 | 1.7032 | 362.3s | 5.5289 | 4.9582 |
| 70 | 1.6838 | 420.7s | 5.4326 | 4.8804 |
| 80 | 1.6693 | 482.8s | 5.3466 | 4.8046 |
| 90 | 1.6620 | 546.3s | 5.3190 | 4.7746 |
| 100 | 1.6587 | 609.1s | 5.3115 | 4.7700 |

Against the normal pd1 baseline, this stale-binary run is slightly better on
rolling byte PPL at every checkpoint and slightly better on fixed-token PPL
except epoch 30, but it is not evidence about depth-7-only training:

| epoch | baseline rolling | depth-7 rolling | baseline fixed | depth-7 fixed |
|------:|--------------:|----------------:|------------:|--------------:|
| 10 | 8.0352 | 7.9839 | 7.6297 | 7.5755 |
| 20 | 6.7435 | 6.7217 | 6.2399 | 6.2057 |
| 30 | 6.2396 | 6.2329 | 5.6777 | 5.6944 |
| 40 | 5.9379 | 5.8744 | 5.3691 | 5.3263 |
| 50 | 5.7229 | 5.6803 | 5.1300 | 5.1088 |
| 60 | 5.5780 | 5.5289 | 5.0118 | 4.9582 |
| 70 | 5.4450 | 5.4326 | 4.8974 | 4.8804 |
| 80 | 5.3737 | 5.3466 | 4.8274 | 4.8046 |
| 90 | 5.3466 | 5.3190 | 4.8017 | 4.7746 |
| 100 | 5.3359 | 5.3115 | 4.7929 | 4.7700 |

This is best interpreted as another baseline-like run, not as a valid
fixed-depth control. The fixed-depth sweep must use the rebuilt trainer.

### Corrected Masked Sweep

After rebuilding `bin/agpt_train_v2`, `experimental.loss_depth_min/max` is
honored. `bin/agpt_experiment` now also preflights v2 configs with the actual
trainer binary before launching training, so stale binaries should fail before
spending GPU time.

All rows below use the same d64/L2 depth-16 pd1 setup as the baseline, but only
one query depth has nonzero direct CE loss.

| loss depth | active rows/epoch | train loss | train wall | rolling byte PPL | fixed-token PPL |
|-----------:|------------------:|-----------:|-----------:|-----------------:|----------------:|
| 1 | 65 | 2.4654 | 546.7s | 34.3168 | 45.7976 |
| 2 | 1,401 | 1.9972 | 566.7s | 28.7651 | 44.5523 |
| 3 | 11,481 | 1.8161 | 572.7s | 26.6527 | 46.5172 |
| 4 | 49,983 | 1.7922 | 529.4s | 20.4347 | 38.8269 |
| 5 | 137,995 | 1.7731 | 531.6s | 17.0156 | 33.6806 |
| 6 | 275,409 | 1.7440 | 576.4s | 13.6779 | 27.8135 |
| 7 | 432,531 | 1.6924 | 557.9s | 8.9847 | 14.8901 |
| 8 | 587,028 | 1.6676 | 558.7s | 9.5679 | 14.9244 |
| 9 | 720,048 | 1.6584 | 572.2s | 8.5202 | 10.9882 |
| 10 | 821,780 | 1.6430 | 609.5s | 7.6776 | 8.5076 |
| 11 | 894,836 | 1.6405 | 606.5s | 8.0391 | 8.1519 |
| 12 | 945,106 | 1.6243 | 583.3s | 7.6428 | 7.0596 |
| 13 | 978,815 | 1.6213 | 602.9s | 8.4137 | 6.9120 |
| 14 | 1,001,507 | 1.6349 | 584.0s | 8.0327 | 5.5658 |
| 15 | 1,017,109 | 1.6223 | 622.5s | 8.0394 | 5.2973 |
| 16 | 1,027,793 | 1.6133 | 590.6s | 8.7932 | 5.2318 |

Baseline comparison:

| objective | rolling byte PPL | fixed-token PPL |
|----------:|-----------------:|----------------:|
| normal pd1 baseline, all depths | 5.3359 | 4.7929 |
| best single-depth rolling, depth 12 | 7.6428 | 7.0596 |
| best single-depth fixed, depth 16 | 8.7932 | 5.2318 |

Findings:

- Single shallow depths are correctly terrible once the depth mask is actually
  honored. The stale depth-1 result was a binary/version problem, not a model
  result.
- The best single-depth rolling-byte score is depth 12, not the 6-8 range.
- The best fixed-token score is depth 16, which matches the fixed eval's
  16-token context.
- No single-depth objective beats the normal pd1 baseline. Training all depths
  is still materially better for both held-out metrics at 100 epochs.
