---
title: RNN AGPT
kind: experiment
status: concluded
outcome: negative
question: >-
  Can a small recurrent f_θ (tanh recurrence) inside AGPT, later extended with a neural history
  residual on top of a count/backoff prior, give a strong character model on Tiny Shakespeare?
answer: >-
  No, not as posed. Plain tanh recurrence (depth 16, d=64, pd=1) reached held-out rolling
  PPL 6.50 at 300 epochs (fitted asymptote about 6.44), far behind a learned count/backoff
  prior (about 3.79, fixed skip-depth). A fixed-lag neural history residual on the live count
  prior improved it only from 3.8032 to 3.8017. The thread was closed on 2026-06-14 and moved
  to a separate prior-residual project.
opened: 2026-06-12
updated: 2026-06-14
code: main
eval: legacy
headline:
- {label: 'tanh recurrence d=64, depth 16, pd=1, 300 epochs', metric: 'held-out rolling PPL
    (bin/agpt_recur_perplexity, 8192 positions)', value: 6.5003}
- {label: 'learned count/backoff prior, simple standardized features', metric: full held-out
    fixed skip-depth PPL, value: 3.7914}
- {label: live count prior + fixed-lag history residual, metric: 'full legal held-out PPL
    (history script, 54,736 positions)', value: 3.8017}
tags: [recurrence, priors]
related: [tanh-recurrence, count-backoff-gate]
---

# RNN AGPT

Status: closed.

Closed on 2026-06-14. This thread started as recurrent `f_theta` inside AGPT,
but the strongest results and open questions moved into a different research
problem: combining explicit priors with neural residual models. The AGPT tree
is no longer the central object in that formulation; it is one possible source
of an a priori distribution. See
[`../prior-residual-project-reference.md`](../prior-residual-project-reference.md)
for the extracted project brief.

No new experiments should be added here unless they are archival reruns or
verification of this exact line. New prior/residual work belongs in the separate
project.

## Closing Summary

This line should not continue as "RNN AGPT" without reframing. The useful
finding is not that a tanh/GRU recurrence is a better AGPT implementation. The
useful finding is:

```text
logits = log P_prior(x | context) + gated_residual_theta(context, memory, prior_stats)
```

is a separate project.

The count/backoff gate is already a strong prior on Tiny Shakespeare:

| system | eval | PPL |
|--------|------|----:|
| learned count/backoff prior | direct full heldout fixed skip-depth | ~3.7919 |
| live count prior in history script | full legal history positions | 3.8032 |
| live count prior + current fixed-lag residual | full legal history positions | 3.8017 |

The residual path can learn and can be forced awake, but the current fixed-lag
RNN-attention residual adds only a tiny improvement when the prior is clean.
Hard surprise filtering makes attention specialize, but overcorrects. The next
work should be done in a prior-residual project with a residual architecture
that has its own learned retrieval/projection space.

This line starts from the smallest useful recurrent `f_theta` inside AGPT:

```text
h_child = tanh(W_h h_parent + W_x emb[token] + b)
logits = W_o h_child + c_o
```

The point is to isolate the local tree-depth recurrent model before adding any
history attention, residual stream, suffix-side features, Fisher updates, or
stride-tree assistance.

## Square One

Use the existing recurrent trainer:

```text
bin/agpt_train_recur
```

and evaluate held-out rolling byte PPL with:

```text
bin/agpt_recur_perplexity
```

The initial baseline should be:

- train-only Tiny Shakespeare trie
- depth 16
- `d_model=64`
- plain tanh recurrence
- no RMSNorm
- no phase weights or phase embeddings
- no singleton backoff
- Adam first, because it is the implemented optimizer

Clean train/heldout split:

```text
train:   data/.splits/4fa9aec1db6b3aea/train_corpus.txt
heldout: data/.splits/4fa9aec1db6b3aea/heldout_corpus.txt
vocab:   data/input.txt
```

Clean depth-16 trie:

```text
data/.tries/49a0a5fc6d5a615c
```

This trie is built from the train split only.

## Variation Ledger

The results below mix several related but different systems. This ledger is the
short version of what was tried and what each result means.

| family | variation | neural history? | eval | best/representative PPL | result |
|--------|-----------|-----------------|------|------------------------:|--------|
| tanh AGPT recurrence | depth 16, d64, pd1 | no | heldout rolling, 8192 positions | 10.0785 at epoch 20 | works, but plain local recurrence is far from the count prior |
| tanh AGPT recurrence | depth 16, d64, pd0 | no | heldout rolling, 8192 positions | 30.9973 at epoch 20 | too few Adam updates; pd0 cadence is ineffective |
| count/backoff prior | Witten-Bell heuristic | no | full heldout fixed skip-depth | 7.2162 | useful baseline, far behind learned gate |
| count/backoff prior | learned sigmoid gate, original dense features | no | heldout fixed skip-depth, 8192 positions | 3.7934 | first strong learned-prior result |
| count/backoff prior | learned sigmoid gate, standardized core features | no | full heldout fixed skip-depth | 3.7917 | standardization holds up on full heldout |
| count/backoff prior | simple standardized features | no | full heldout fixed skip-depth | 3.7914 | best pure prior so far |
| count/backoff prior | all rate/slope features | no | full heldout fixed skip-depth | 3.7927 | more features overfit/noise slightly |
| count/backoff prior | simple without duplicate suffix mass | no | full heldout fixed skip-depth | 3.7919 | essentially tied; duplicate removed conceptually |
| count feature ablation | one feature only / leave-one-out | no | full heldout fixed skip-depth | reliability alone 3.8392 | many weak features combine; reliability and depth are load-bearing |
| compact prior feature sweep | selected reliability/depth/mass/branch/suffix features | no | full heldout fixed skip-depth | 3.7953 | compact set close but not better than full simple |
| local prior residual | tanh residual logits on frozen AGTS prior | no long history | heldout rolling, 8192 positions | 3.9157 at eval scale 0.5 | tiny gain only if residual is priced/scaled |
| fixed-lag history residual | random warmup chunks, k64 | yes | heldout rolling, 8192 positions | 3.8995 | tiny gain; repeated warmup wasted compute |
| fixed-lag history residual | learned alpha gate | yes | heldout rolling, 8192 positions | 3.8999 | alpha mostly suppresses residual |
| fixed-lag history residual | prior dropout | yes | heldout rolling, 8192 positions | 3.9002 at dropout 0.1 | wakes attention up but does not improve PPL |
| fixed-lag history residual | muted/uniform prior | yes | heldout rolling, 8192 positions | 14.1312 | residual path can learn, but is much weaker than count prior |
| fixed-lag history residual | bounded delta + KL trust region | yes | heldout rolling, 8192 positions | 3.8999 | correct qualitative shape, very small gain |
| ordered fixed-lag residual | one ordered corpus pass, no RoPE | yes | heldout rolling, 8192 positions | 3.8989 at eval scale 0.5 | avoids warmup waste; still small and scale-sensitive |
| ordered fixed-lag residual | one ordered corpus pass + RoPE char positions | yes | heldout rolling, 8192 positions | 3.8917 | RoPE gives the first non-noise history gain on sampled eval |
| ordered fixed-lag residual | RoPE, full heldout eval | yes | full legal heldout, 54,736 positions | 3.9409 | gain survives full eval, but AGTS prior baseline is weaker there |
| segment-memory residual | mass-to-1 segment endpoints | yes | full heldout, 55,624 positions | 3.9928 | attention specializes but residual overcorrects badly |
| fixed-lag history residual | learned alpha + prior summary features | yes | sampled 8192 positions | 3.8860 | best sampled history-residual result in this script |
| fixed-lag history residual | learned alpha + prior summary features | yes | full legal heldout, 54,736 positions | 3.9310 | full eval confirms gain over AGTS sidecar prior, but not over best count prior |
| fixed-lag history residual | learned alpha + prior summary features + refreshed simple prior sidecar | yes | full legal heldout, 54,736 positions | 3.8365 | stronger sidecar closes most of the gap; residual adds only 0.0010 PPL |
| fixed-lag history residual | learned alpha + prior summary features + live count prior | yes | full legal heldout, 54,736 positions | 3.8017 | apples-to-apples prior path; residual adds only 0.0015 PPL |
| fixed-lag history residual | live count prior + top-25% surprise training | yes | full legal heldout, 54,736 positions | 3.9679 | attention wakes up, but residual overcorrects badly |
| fixed-lag history residual | live count prior + top-50% surprise training | yes | full legal heldout, 54,736 positions | 3.8189 | gentler than 25%, but still worse than prior |

Important comparison boundary:

- The `3.79` results are pure learned count/backoff priors. They use local
  trie history up to depth 16 and suffix backoff, but no RNN state, no attention
  memory, no RoPE, and no learned long-history representation.
- The older `3.93` history-residual result used the first exported AGTS
  sidecar prior, whose own full-eval prior baseline was `3.9461`. The residual
  improved that sidecar to `3.9310`, but that run was not stacked on top of the
  later best count-gate feature set.
- A refreshed AGTS export from the current simple standardized count gate moves
  the sidecar prior to `3.8376` on the history script's full legal heldout
  protocol. The same history residual improves it only to `3.8365`. That says
  the prior-export mismatch mattered a lot, but the current fixed-lag neural
  history residual still adds only a very small correction.
- A live count-prior path avoids the position-data catalog coverage loss. On
  the exact same history-script full legal heldout protocol, the live prior is
  `3.8032` and the residual improves it to `3.8017`.

## First Smoke

```text
OPENBLAS_NUM_THREADS=1 bin/agpt_train_recur \
  --trie data/.tries/49a0a5fc6d5a615c \
  --d-model 64 \
  --epochs 1 \
  --lr 0.001 \
  --seed 1 \
  --partition-depth 1 \
  --save rnd/rnn-agpt/smoke_d16_d64_pd1_ep1.recur
```

Held-out check:

```text
bin/agpt_recur_perplexity \
  --checkpoint rnd/rnn-agpt/smoke_d16_d64_pd1_ep1.recur \
  --file data/.splits/4fa9aec1db6b3aea/heldout_corpus.txt \
  --vocab-file data/input.txt \
  --seq-len 16 \
  --max-positions 8192
```

Result:

| run | train PPL | heldout rolling PPL | bpc | wall |
|-----|----------:|--------------------:|----:|-----:|
| d16 d64 pd1 epoch 1 | 41.5200 | 28.0537 | 4.8101 | 53.25s train, 1.40s eval |
| d16 d64 pd1 epoch 20 | 11.1051 | 10.0785 | 3.3332 | ~18.3m train, 1.52s eval |
| d16 d64 pd0 epoch 20 | 38.3356 | 30.9973 | 4.9541 | ~16.1m train, 1.40s eval |

The pd0 result confirms that one optimizer step per full-tree epoch is not a
reasonable training cadence for Adam here. pd1 is doing the same full tree
work but gets 65 optimizer steps per epoch, and is dramatically better by epoch
20.

## Levelized Radix Execution

Implemented first pass:

```text
bin/agpt_train_recur_level
src/tools/agpt_train_recur_level.cr
```

This is currently narrow by design:

- plain tanh only
- Float32 weights/states/Adam
- compatible `.recur` checkpoint output for `bin/agpt_recur_perplexity`
- resume from plain tanh `.recur` checkpoints via `--load`
- `partition-depth` 0 or 1
- no RMSNorm, phase weights/embeddings, or singleton backoff yet

The old trainer is mathematically useful but computationally shaped wrong for
recurrent `f_theta`. It walks each radix edge and applies one small BLAS `dgemv`
per token transition:

```text
z_child = W_h h_parent + W_x emb[token] + b
h_child = tanh(z_child)
```

Only `emb @ W_x^T` is currently batched. The clean optimization is to expand
the radix trie into a logical transition plan and process it by character
depth:

```text
for depth in 1..D:
  rows = transitions ending at this depth
  H_parent = gather(H, rows.src_state)
  X_token  = gather(X_proj, rows.token)
  Z        = H_parent @ W_h^T + X_token + b
  H[rows.dst_state] = tanh(Z)
```

The level trainer preserves radix compression semantically by adding synthetic
logical states for mid-edge positions:

```text
radix endpoint states: use the existing radix record id
mid-edge states:       allocate synthetic ids after radix_count
```

For a radix record with edge `a b c`, the transition plan becomes:

```text
parent -> synthetic_1 --a
synthetic_1 -> synthetic_2 --b
synthetic_2 -> radix_record_id --c
```

This keeps the current endpoint-only loss semantics exactly: losses are still
applied only at radix record ids, not at synthetic mid-edge states.

Backward can be levelized in reverse:

```text
for depth in D..1:
  rows = transitions ending at this depth
  dz = dh_child * (1 - h_child^2)
  grad_b   += sum(dz)
  grad_W_h += dz^T @ H_parent
  dh_parent += dz @ W_h

  G_token[token] += dz
  grad_W_x += G_token^T @ emb
  grad_emb += G_token @ W_x
```

The output head can also be batched over endpoint rows:

```text
logits = H_endpoint @ W_o^T + c_o
```

Main tradeoff: exact levelized backward wants hidden states and adjoints for the
current partition's logical states, not just radix endpoints. At depth 16 on
Tiny Shakespeare, the full pd0 plan is about `total_edge_chars + 1 = 8.9M`
states. At `d_model=64`:

```text
Float64 H only:  ~4.6 GB
Float32 H only:  ~2.3 GB
H + dH doubles that
```

For pd1, plans are per root-child subtree, so the peak hidden-state slab is much
smaller than the partition-summed total:

```text
d16 d64 pd1 partition-summed H: 2173.54 MB
d16 d64 pd1 peak-partition H:   295.49 MB
```

The first implementation uses Float32/SGEMM and chunks each depth's transition
rows. It still keeps a partition's depth states in memory. Depth-file state
offload remains the next memory extension if pd0 or larger corpora make peak
partition memory too large.

## Level Trainer Smoke

Truncated parity, first 2,000 radix records, `d_model=16`, epoch 1:

| mode | trainer | pd | train PPL | wall |
|------|---------|---:|----------:|-----:|
| edge walk | `bin/agpt_train_recur` | 1 | 62.655531 | 0.430s |
| level batched | `bin/agpt_train_recur_level` | 1 | 62.655531 | 0.023s |
| edge walk | `bin/agpt_train_recur` | 0 | 65.098594 | 0.033s |
| level batched | `bin/agpt_train_recur_level` | 0 | 65.098594 | 0.007s |

Full depth-16, `d_model=64`, pd1, epoch 1:

| trainer | train PPL | heldout rolling PPL | wall |
|---------|----------:|--------------------:|-----:|
| edge walk | 41.519984 | 28.0537 | 53.25s |
| level batched | 41.519984 | 28.0537 | 18.85s |

Full depth-16, `d_model=64`, pd1, epoch 20:

| trainer | train PPL | heldout rolling PPL | train wall |
|---------|----------:|--------------------:|-----------:|
| edge walk | 11.105062 | 10.0785 | ~18.3m |
| level batched | 11.105062 | 10.0785 | ~7.0m |

The full pd1 level run is an exact loss/eval match to the existing trainer and
is about `2.6x` faster through epoch 20. This is not yet the final speed ceiling:
output-head loss is batched, recurrent transitions are batched, but state
gather/scatter is still Crystal-loop heavy.

Longer level-batched pd1 trajectory:

| epoch | train PPL | heldout rolling PPL | bpc |
|------:|----------:|--------------------:|----:|
| 20 | 11.1051 | 10.0785 | 3.3332 |
| 40 | 9.8261 | 8.6320 | 3.1097 |
| 60 | 9.1995 | 7.9071 | 2.9832 |
| 80 | 8.8522 | 7.5037 | 2.9076 |
| 100 | 8.6323 | 7.2185 | 2.8517 |
| 120 | 8.4818 | 7.0506 | 2.8177 |
| 140 | 8.3712 | 6.9033 | 2.7873 |
| 160 | 8.2947 | 6.7958 | 2.7647 |
| 180 | 8.2325 | 6.7010 | 2.7444 |
| 200 | 8.1806 | 6.6333 | 2.7297 |
| 220 | 8.1165 | 6.5850 | 2.7192 |
| 240 | 8.1010 | 6.5630 | 2.7144 |
| 260 | 8.0855 | 6.5404 | 2.7094 |
| 280 | 8.0708 | 6.5201 | 2.7049 |
| 300 | 8.0571 | 6.5003 | 2.7005 |

At epoch 200, lowering LR from `0.001` to `0.0003` removed the late train-PPL
wobble and continued improving heldout. A simple late-curve fit over epochs
200-300 estimates an asymptote near `6.44`, with epoch 400 around `6.46`. This
is only a trajectory estimate, but it strongly suggests plain tanh-only local
recurrence is a mid-6 PPL model, not a route to the low-4s by training longer.

## Prior Direction

The missing element is the learned count/backoff trust prior:

```text
q(root) = p_root
q(c) = w(c) * p_mle(next | c) + (1 - w(c)) * q(suffix(c))
w(c) = sigmoid(theta . phi(c))
```

where `phi(c)` includes reliability, KL gain, normalized entropy, normalized
depth, entropy delta, and suffix-side branch/reliability/entropy/KL/delta
features. `suffix_mass_norm` was removed from the public feature set because it
duplicates prefix-side `mass_norm` for the same node.

Feature standardization is now available in `src/tools/agpt_count_gate.py` via
`--standardize-features`. It computes train-fit-context mean/std for every
non-bias gate feature and feeds z-scored values to the sigmoid:

```text
z_i(c) = (phi_i(c) - mean_i) / std_i
w(c) = sigmoid(theta . z(c))
```

The bias remains a literal `1.0`. The saved JSON includes
`feature_standardization.mean` and `feature_standardization.std`, so evaluation
and sidecar export use the same transform. This should make optimization better
conditioned and make theta signs more interpretable: positive means an
above-average feature value increases local-count trust.

This prior is strange from a generic neural-model viewpoint, but natural from
the tree geometry viewpoint: it is an algebra over prefix evidence, suffix
projection, and learned trust. The RNN should likely learn residual/comprehension
state on top of this prior rather than trying to learn the whole local character
law from scratch.

First implementation step:

- `bin/agpt_train_recur_level` can train residual logits on top of a frozen
  AGTS sidecar:
  `logits = log(q_prior) * prior_scale + W_o h + c_o`
- The trainer also supports charging the residual a price:
  `--residual-scale` scales the residual contribution during training, and
  `--residual-l2` adds a per-node residual-logit gradient penalty.
- `bin/agpt_recur_perplexity` can evaluate the same combined model with
  `--position-data`, `--prior-sidecar`, `--prior-scale`, and
  `--residual-scale`.
- Heldout eval must use longest-suffix sidecar lookup. Exact full depth-16
  contexts are rare in heldout, but suffix backoff is the whole point of the
  prior. On the first 8192 rolling heldout positions below, only 296 contexts
  hit the exact depth-16 substring; 7896 used suffix backoff; 0 missed.

Current dense prior artifacts:

```text
position data: /tmp/agpt_rnn_prior_d16_position_data
dense AGTS:    /tmp/agpt_rnn_prior_d16_dense.agts
```

Dense count-prior diagnostic from `src/tools/agpt_count_gate.py`:

| split/metric | loss nats | PPL |
|--------------|----------:|----:|
| heldout rolling learned gate | 1.3628 | 3.9073 |
| heldout fixed skip-depth learned gate | 1.3333 | 3.7934 |

First residual-on-prior run:

```text
OPENBLAS_NUM_THREADS=1 bin/agpt_train_recur_level \
  --trie data/.tries/49a0a5fc6d5a615c \
  --d-model 64 \
  --epochs 10 \
  --lr 0.001 \
  --seed 1 \
  --partition-depth 1 \
  --batch-size 65536 \
  --position-data /tmp/agpt_rnn_prior_d16_position_data \
  --prior-sidecar /tmp/agpt_rnn_prior_d16_dense.agts \
  --prior-scale 1.0 \
  --output-init-scale 0.01 \
  --save rnd/rnn-agpt/prior_resid_d16_d64_pd1_ep10.recur \
  --checkpoint-every 5
```

Train PPL moved only slightly:

| epoch | train PPL | wall/epoch |
|------:|----------:|-----------:|
| 1 | 5.2074 | 20.0s |
| 5 | 5.2086 | 23.6s |
| 10 | 5.1903 | 26.3s |

Heldout rolling, same 8192 positions, prior-aware eval:

| checkpoint | residual scale | loss nats | PPL |
|------------|---------------:|----------:|----:|
| epoch 10 | 0.00 | 1.3657 | 3.9183 |
| epoch 10 | 0.10 | 1.3653 | 3.9170 |
| epoch 10 | 0.20 | 1.3652 | 3.9165 |
| epoch 10 | 0.25 | 1.3652 | 3.9165 |
| epoch 10 | 0.30 | 1.3653 | 3.9168 |
| epoch 10 | 0.50 | 1.3660 | 3.9197 |
| epoch 10 | 1.00 | 1.3713 | 3.9403 |

Interpretation: the frozen prior is carrying almost all of the result. A small
residual scale gives a tiny heldout gain on this early run, but the full
residual is overconfident and hurts. The next useful prior-residual experiment
should add an explicit residual trust/temperature parameter, regularize the
residual logits, or train the residual against a heldout-calibrated objective
rather than assuming scale 1.0 is correct.

Follow-up: make the residual pay for moving away from the prior.

Scale-only training did not solve the issue:

| run | train residual scale | residual L2 | epoch 10 train PPL | eval residual scale | heldout PPL |
|-----|---------------------:|------------:|-------------------:|--------------------:|------------:|
| prior only | 0.00 | 0.0 | n/a | 0.00 | 3.9183 |
| scale-only | 0.20 | 0.0 | 5.2000 | 0.20 | 3.9258 |

Adding a residual-logit price helped:

| run | train residual scale | residual L2 | epoch 10 train PPL | eval residual scale | heldout PPL |
|-----|---------------------:|------------:|-------------------:|--------------------:|------------:|
| unpriced | 1.00 | 0.0 | 5.1903 | 1.00 | 3.9403 |
| priced | 1.00 | 1.0 | 5.1881 | 1.00 | 3.9203 |
| priced | 1.00 | 1.0 | 5.1881 | 0.50 | 3.9157 |
| stronger price | 1.00 | 5.0 | 5.1958 | 1.00 | 3.9176 |
| stronger price | 1.00 | 5.0 | 5.1958 | 0.50 | 3.9169 |

The important observation is not the tiny absolute gain yet; it is the change in
behavior. With `residual_l2=5.0`, full-scale residual eval is no longer badly
overconfident and slightly beats the prior-only baseline on the same 8192
rolling heldout positions. This is the first clean sign that the residual can
add useful information if it has to pay for leaving the prior.

## Questions

- How strong is plain tanh at depth 16 before any residual history attention?
- Does the depth-16 recurrent path behave better or worse than the earlier
  clean depth-8 tanh baseline?
- How much of the later count-prior plus residual result can be attributed to
  the local recurrent tree state alone?
- Where should the end-cap hidden states be exported if we add a separate
  residual history-attention model?

## Next Objective: History-Attention Residual

The local prior/residual result says the prior is strong and the local residual
only helps if it pays a price. The next question is whether a residual with
information outside the 16-character prior window can help materially.

Square-one objective:

```text
local state:      h_t = RNN(x[t-d:t])
history memory:   M_t = [h_{t-d}, h_{t-2d}, ..., h_{t-Kd}]
attention:        a_t = Attn(q=h_t, k/v=M_t)
residual:         r_t = W_r a_t
prediction:       logits = log(q_prior(x[t-d:t])) + alpha * r_t
objective:        CE(logits, x[t]) + lambda * ||r_t||^2
```

Initial settings:

```text
d = 16
K = 64
d_model = 64
heads = 1
alpha = 1.0
prior = frozen dense AGTS learned count/backoff prior
optimizer = Adam
```

The acceptance test is deliberately narrow:

```text
history residual beats prior-only by more than the local residual did
```

The local priced residual only reached about `3.9176` PPL versus prior-only
`3.9183` on the same 8192 rolling heldout positions. A history-attention
residual needs to clear that by a convincing margin; otherwise the extra context
is not paying rent.

Required diagnostics:

- prior-only PPL on the exact same positions
- residual-on-prior PPL at full scale
- residual scale sweep
- residual norm versus prior NLL
- attention entropy
- top attended lags

The residual must work for the prior, not replace it. Strong prior positions
should have low residual norm; weak/ambiguous prior positions are where residual
capacity should be spent.

First prototype:

```text
src/tools/agpt_history_residual.py
```

This is a CPU PyTorch prototype, not the final Crystal/CUDA trainer. It loads
the frozen dense AGTS sidecar directly, computes the local `d`-token RNN state
for the current position and each history endpoint, applies one-head attention
over `K` previous states, and trains priced residual logits on top of the prior.

Smoke:

```text
K=8, d_model=32, steps=5, batch=16
```

Result: shape/file-format path works. Attention entropy is near `ln(8)`, as
expected at initialization.

First square-one run:

```text
K=64, d_model=64, steps=100, batch=64, residual_l2=5.0
out: rnd/rnn-agpt/history_resid_k64_d64_steps100_l2_5.json
```

Heldout on 8192 rolling positions:

| metric | value |
|--------|------:|
| prior-only PPL | 3.8341 |
| residual scale 1.0 PPL | 3.8343 |
| residual scale 0.5 PPL | 3.8342 |
| avg residual norm | 0.0545 |
| avg attention entropy | 4.1578 |
| `ln(64)` | 4.1589 |

Interpretation: the model is wired correctly, but after 100 CPU steps the
history attention is still essentially uniform and the residual is neutral to
slightly harmful. This is a smoke result, not a serious training result. The
next real test needs either more steps, a stronger residual signal, or a cheaper
implementation that can afford enough updates for attention to specialize.

Correction: the first prototype summarized each 16-character segment
independently. The intended model must carry RNN state through the stream. The
correct state convention is:

```text
h_i = RNN state after consuming x[i]
target at t = x[t]
fresh legal query = h_{t-1}
M_t = [h_{t-1}, h_{t-16}, h_{t-32}, ..., h_{t-1024}]
```

The stream prototype (`--mode stream`) uses contiguous chunks and skips early
positions until the full history window exists, so there are no negative-index
states.

Corrected stream run:

```text
K=64, d_model=64, steps=100, batch=64 contiguous targets, residual_l2=5.0
out: rnd/rnn-agpt/history_stream_k64_d64_steps100_l2_5.json
```

| metric | value |
|--------|------:|
| prior-only PPL | 3.9002 |
| residual scale 1.0 PPL | 3.8996 |
| residual scale 0.5 PPL | 3.8998 |
| avg residual norm | 0.0772 |
| avg attention entropy | 4.1736 |
| `ln(65)` | 4.1744 |

Longer stream run:

```text
K=64, d_model=64, steps=1000, batch=64 contiguous targets, residual_l2=5.0
out: rnd/rnn-agpt/history_stream_k64_d64_steps1000_l2_5.json
```

| metric | value |
|--------|------:|
| prior-only PPL | 3.9002 |
| residual scale 1.0 PPL | 3.8998 |
| residual scale 0.5 PPL | 3.8999 |
| avg residual norm | 0.0488 |
| avg attention entropy | 4.1691 |
| `ln(65)` | 4.1744 |

The stream version is a cleaner test and gives a tiny positive result, but 1000
CPU steps did not produce strong attention specialization. Entropy moved from
near-uniform to only slightly below uniform. More steps alone may not be the
best lever; likely next knobs are weaker residual price, larger batches/chunks,
attention dropout/temperature, or a training target that emphasizes cases where
the prior is wrong.

Learned residual gate:

```text
logits = log(q_prior) + alpha_t * r_theta(history)
alpha_t = sigmoid(W_alpha [query, attended] + b_alpha)
```

The alpha head is initialized with zero weights and bias `-2.0`, so initial
`alpha ~= 0.119`. The residual price is charged on the applied residual
`alpha_t * r_theta`, not on the raw residual.

```text
K=64, d_model=64, steps=1000, batch=64 contiguous targets, residual_l2=5.0
out: rnd/rnn-agpt/history_stream_k64_d64_steps1000_l2_5_alpha.json
```

| metric | value |
|--------|------:|
| prior-only PPL | 3.9002 |
| learned-alpha residual PPL | 3.8999 |
| fixed residual scale 0.5 PPL | 3.8995 |
| avg raw residual norm | 0.2832 |
| avg applied residual norm | 0.0204 |
| avg alpha | 0.0711 |
| avg attention entropy | 4.1450 |

This confirms that the residual has some usable signal, but the learned alpha
mostly learns to stay out of the prior's way. The count prior is strong enough
that the history residual does not receive much pressure to become useful.

Prior dropout test:

Training-only prior dropout replaces a fraction of training rows' detailed
`log(q_prior(context))` with the root prior. Evaluation still uses the full
prior. This asks whether the history path can learn when the local prior is not
always allowed to dominate.

```text
logits_train = dropped_log_prior + alpha_t * r_theta(history)
logits_eval  = log(q_prior)      + alpha_t * r_theta(history)
```

| run | prior dropout | PPL | avg alpha | avg applied residual norm | avg attention entropy |
|-----|--------------:|----:|----------:|--------------------------:|----------------------:|
| learned alpha | 0.0 | 3.8999 | 0.0711 | 0.0204 | 4.1450 |
| learned alpha | 0.1 | 3.9002 | 0.0724 | 0.1486 | 3.1702 |
| learned alpha | 0.5 | 3.9862 | 0.1994 | 0.8206 | 2.4583 |

Prior dropout clearly wakes up the history path: attention becomes much more
selective and the applied residual gets larger. But naive row dropout does not
improve full-prior heldout PPL. At `0.5`, it trains the residual to compensate
for an artificial missing-prior problem and hurts eval badly. At `0.1`, it is
nearly neutral on PPL while still making attention less uniform.

Conclusion: "the prior has it too easy" is a real optimization issue, but
simple prior-row dropout is only a diagnostic/regularizer so far. A better
objective should pressure the residual on cases where the prior is actually
weak or wrong, rather than randomly hiding a good prior.

Muted-prior residual-only test:

```text
K=64, d_model=64, steps=1000, batch=64 contiguous targets
prior_mode = uniform
residual_l2 = 0.0
out: rnd/rnn-agpt/history_stream_k64_d64_steps1000_uniform_prior.json
```

Here the prediction prior is replaced by a constant uniform distribution during
both train and eval:

```text
logits = -log(V) + r_theta(history)
```

The sidecar is still loaded only for comparable diagnostics. It does not
contribute information to the logits.

| metric | value |
|--------|------:|
| uniform baseline PPL | 65.0000 |
| residual-only heldout PPL | 14.1312 |
| residual-only heldout PPL, scale 0.5 | 18.5742 |
| final sampled train PPL | 12.1036 |
| avg residual norm | 24.6569 |
| avg attention entropy | 2.3040 |

This answers the diagnostic question: the history/RNN residual path can learn a
real standalone character-model signal when the count prior is fully muted. It
is much weaker than the learned count/backoff prior, but far better than
uniform. The residual path is therefore not inert; the hard part is calibrating
it so it contributes only where it has information the prior lacks.

Bounded trust-region residual:

The next correction is to treat the history residual as a small log-probability
update, not a replacement model. The prototype now supports:

```text
raw_delta = residual_scale * r_theta(history)
delta = max_delta * tanh(raw_delta / max_delta)
logits = log(q_prior) + delta
loss = CE(logits, target) + beta * KL(q_prior || softmax(logits))
```

This gives two direct controls:

- `--residual-max-delta`: per-logit correction cap, in nats
- `--trust-kl`: KL trust-region weight keeping the corrected distribution near
  the prior

Runs below use the full prior, no prior dropout, no learned alpha,
`trust_kl=10.0`, `residual_l2=0.0`, `K=64`, `d_model=64`, 1000 steps.

| max delta | prior PPL | corrected PPL | scale 0.5 PPL | avg applied residual norm | avg attention entropy |
|----------:|----------:|--------------:|--------------:|--------------------------:|----------------------:|
| 0.05 | 3.900171 | 3.900045 | 3.900072 | 0.1298 | 4.1709 |
| 0.10 | 3.900171 | 3.899935 | 3.900018 | 0.1621 | 4.1711 |

This is the right qualitative behavior: very small updates, tiny KL drift, and
heldout improvement rather than overconfident degradation. It is still a small
effect, and attention remains close to uniform, but this is a cleaner loss
shape than naive residual scaling or prior dropout.

## Ordered Stream Mode

The first stream prototype sampled random chunks shaped like:

```text
1024 warmup chars + 64 supervised target chars
```

That means each optimizer step processed 1088 RNN input chars, but only 64
positions had direct CE loss. Most compute was repeated warmup.

`src/tools/agpt_history_residual.py` now supports:

```text
--mode ordered
```

Ordered mode reads the train corpus straight through. It carries the GRU hidden
state and a detached rolling history-state buffer across chunks, and detaches at
chunk boundaries. The first legal update pays the initial history warmup once;
after that each step processes `batch_size` new chars and trains on
`batch_size` targets.

With `depth=16`, `K=64`, `batch_size=64`:

```text
step 1 processed_inputs = 1024  # one-time warmup to first full-history target
step N processed_inputs = 64    # after warmup
```

This is truncated BPTT: state carries forward, gradients do not cross chunk
boundaries.

Ordered smoke:

```text
K=8, d_model=32, batch=16, steps=5, prior_mode=uniform
out: rnd/rnn-agpt/history_stream_smoke_ordered_uniform_prior.json
```

The first update processed `128` chars to reach the first full-history target;
subsequent updates processed `16` chars each. This verified that ordered mode
does not repeatedly re-warm.

Ordered comparison runs:

```text
K=64, d_model=64, batch=64, steps=1000
```

| run | prior mode | loss controls | prior PPL | heldout PPL | final cursor | wall |
|-----|------------|---------------|----------:|------------:|-------------:|-----:|
| ordered bounded residual | full | `max_delta=0.10`, `trust_kl=10` | 3.900171 | 3.900422 | 64,961 | 46.8s |
| ordered muted prior | uniform | none | 65.0000 | 14.5276 | 64,961 | 46.4s |

Ordered mode is much more compute-efficient than random warmup chunks, but 1000
ordered steps only covers the first ~65k train characters. The muted-prior
result learns quickly on the training stream, but heldout is slightly worse than
the random-chunk muted-prior run (`14.53` vs `14.13`), likely because random
chunks cover the corpus more broadly at the same update count.

Conclusion: ordered streaming fixes the warmup waste, but raw step count is no
longer the right comparison unit. Future ordered runs should be measured in full
or fractional corpus passes, or use shuffled contiguous blocks to combine broad
coverage with no repeated 1024-character warmup.

One ordered corpus pass:

With this train split and `batch_size=64`, one complete ordered pass over all
full-history targets takes:

```text
1 + ceil((train_len - 1025) / 64) = 16542 optimizer steps
```

The first step trains only the first full-history target at position 1024; the
remaining steps train 64 targets each except the final partial step.

```text
out: rnd/rnn-agpt/history_ordered_k64_d64_1epoch_full_prior_delta010_kl10.json
```

| run | prior mode | loss controls | prior PPL | heldout PPL | wall |
|-----|------------|---------------|----------:|------------:|-----:|
| one ordered pass | full | `max_delta=0.10`, `trust_kl=10` | 3.900171 | 3.900198 | 597.8s |
| one ordered pass | full | unbounded, `trust_kl=10` | 3.900171 | 3.904481 | 597.8s |
| one ordered pass, eval scale 0.5 | full | unbounded, `trust_kl=10` | 3.900171 | 3.898888 | 597.8s |

This run produced much more selective attention than the short ordered run:

```text
avg attention entropy: 2.0640
top lags: 1024, 1008, 992, ...
```

But the heldout correction was still effectively neutral and slightly worse
than the prior. This suggests the residual is learning a real history behavior,
but the current bounded full-prior objective is not yet making that behavior
useful as a calibrated heldout correction.

Removing the hard bound while keeping `trust_kl=10` made the residual quieter
during most of training than expected, but full-scale eval overcorrected. The
same learned residual at eval scale `0.5` improved heldout to `3.898888`. This
is the strongest history-residual result in this set so far:

```text
out: rnd/rnn-agpt/history_ordered_k64_d64_1epoch_full_prior_unbounded_kl10.json
```

Interpretation: the residual is learning useful information, but its learned
scale is miscalibrated. The next cleaner objective is probably not a hard cap;
it is training with the residual scale/temperature that will be used at eval,
or learning a calibrated gate with enough pressure to use the residual without
letting full-scale logits overcorrect.

RoPE position signal:

The history attention initially had no explicit position signal. The memory
slots were ordered in code, but attention saw only hidden-state content:

```text
M_t = [h[t-1], h[t-16], h[t-32], ..., h[t-1024]]
```

`src/tools/agpt_history_residual.py` now supports:

```text
--position-encoding rope
```

This applies RoPE to attention q/k using actual corpus character positions:

```text
query position:  t - 1
memory positions: [t - 1, t - 16, t - 32, ..., t - 1024]
```

These are absolute character positions. The periodicity comes from RoPE's
sinusoidal frequency basis, not from a learned history-slot ordinal.

One ordered corpus pass, full prior, unbounded residual, `trust_kl=10`:

```text
out: rnd/rnn-agpt/history_ordered_k64_d64_1epoch_full_prior_unbounded_kl10_rope.json
```

| run | position signal | prior PPL | heldout PPL | eval scale | avg attention entropy |
|-----|-----------------|----------:|------------:|-----------:|----------------------:|
| unbounded, KL | none | 3.900171 | 3.904481 | 1.0 | 2.8720 |
| unbounded, KL | none | 3.900171 | 3.898888 | 0.5 | 2.8720 |
| unbounded, KL | RoPE char positions | 3.900171 | 3.891656 | 1.0 | 3.5685 |
| unbounded, KL | RoPE char positions | 3.900171 | 3.893267 | 0.5 | 3.5685 |

This is the first non-noise-sized history-residual gain in this thread. RoPE
made the full-scale residual useful instead of overcorrecting, and it improved
the same 8192-position heldout sample by about `0.0085` PPL relative to the
prior. The result still needs a larger/full heldout eval and seed checks, but
the direction is now technically meaningful rather than just neutral.

Full-heldout eval mode:

`src/tools/agpt_history_residual.py` now supports:

```text
--eval-all
```

For stream/ordered history attention, this evaluates every legal heldout target
with a full history window. With `depth=16`, `K=64`, the first legal target is
after `1024` warmup characters, so the carved heldout split gives:

```text
heldout length: 55760
legal history-residual eval positions: 54736
```

Prior-only full-heldout check:

```text
out: rnd/rnn-agpt/history_eval_all_prior_only_check.json
```

| eval | positions | prior PPL |
|------|----------:|----------:|
| 8192 sampled stream positions | 8192 | 3.900171 |
| all legal heldout positions | 54736 | 3.946054 |

This is an important correction: the 8192-position sample was optimistic for
our current dense AGTS prior. It also increases the mismatch with
`agpt-ultra`'s reported prior-only `3.8595`, so prior/eval alignment is now the
first debugging target.

Full-heldout RoPE one-pass result:

```text
out: rnd/rnn-agpt/history_ordered_k64_d64_1epoch_full_prior_unbounded_kl10_rope_evalall.json
```

| run | positions | prior PPL | residual PPL | eval scale |
|-----|----------:|----------:|-------------:|-----------:|
| RoPE, ordered one pass, unbounded, `trust_kl=10` | 54736 | 3.946054 | 3.941554 | 1.0 |
| same checkpoint | 54736 | 3.946054 | 3.940855 | 0.5 |

RoPE still helps under full evaluation, but the absolute level is worse than
the sampled result. The residual gain is about `0.0052` PPL at eval scale `0.5`;
useful, but the larger issue is that our prior is far behind the
`agpt-ultra` count prior on a full heldout protocol.

## Segment Rule From `agpt-ultra`

The `agpt-ultra` segment-memory model is not using fixed 16-character chunks
or fixed lag states. Its segment rule appears to be:

```text
cursor = 0
while cursor < corpus_len:
  walk the prefix trie from root using corpus[cursor...]
  stop when the current prefix mass becomes 1, or when max_depth is reached
  emit that token span as one segment
  cursor += segment_length
```

A quick local reconstruction with `max_depth=16` and `unique_threshold=1`
matches the reported `agpt-ultra` train segmentation exactly:

```text
train_len: 1,059,634
segments: 113,758
mean segment length: 9.3148
max segment length: 16
```

Length distribution:

| segment length | count |
|---------------:|------:|
| 8 | 17,044 |
| 9 | 15,760 |
| 7 | 15,493 |
| 10 | 12,867 |
| 6 | 11,286 |
| 11 | 9,827 |
| 12 | 7,009 |
| 16 | 6,351 |
| 5 | 5,416 |
| 13 | 4,946 |
| 14 | 3,459 |
| 15 | 2,367 |
| 4 | 1,692 |
| 3 | 225 |
| 2 | 16 |

This explains a major architecture difference. Our current prototype attends
over fixed character-lag states:

```text
[h[t-1], h[t-16], h[t-32], ..., h[t-1024]]
```

`agpt-ultra` attends over recent terminal states of these variable mass-to-1
segments. With `max_memory=16`, that is usually about `16 * 9.31 = 149`
characters of segment-terminal memory on average, not a fixed 1024-character
ladder. The memory keys are aligned to trie uniqueness boundaries rather than
uniform character offsets.

## Local Segment-Memory Diagnostic

`src/tools/agpt_history_residual.py` now supports:

```text
--memory-mode segment
```

Current limitations:

- only implemented for `--mode ordered`
- uses the last `K` completed mass-to-1 segment terminal states as attention
  memory
- still predicts with the existing residual head on the attended memory output
- does **not** yet feed cross-attention back into each GRU token state the way
  `agpt-ultra` does

Segment smoke:

```text
out: rnd/rnn-agpt/history_ordered_segment_smoke_uniform_prior.json
```

This reproduced the train segmentation:

```text
segments: 113,758
mean_len: 9.3148
max_len: 16
```

First full diagnostic, using settings closest to the `agpt-ultra` recipe that
fit this local prototype:

```text
--mode ordered
--memory-mode segment
--k-history 16
--d-model 64
--residual-scale 0.10
--residual-l2 0.02
--zero-output-head
--eval-all
```

Output:

```text
rnd/rnn-agpt/history_ordered_segment_k16_d64_1epoch_full_prior_scale010_l2_002.json
```

| run | positions | prior PPL | residual PPL | avg attention entropy |
|-----|----------:|----------:|-------------:|----------------------:|
| segment memory, 1 ordered pass | 55624 | 3.932617 | 3.992798 | 1.1647 |

The attention became very selective over the 16 segment memories, but the
simple residual interface overcorrected badly. This is a useful negative
diagnostic: segment-terminal memory alone is not enough in this local script.
The missing `agpt-ultra` ingredients are likely important:

- stronger depth-8 count prior (`3.8595` reported prior-only)
- cross-attention feedback into the GRU hidden sequence, not just a separate
  residual head over attended memory
- context-gated residual alpha
- possibly their exact segment-level update/eval harness

## Prior-Feature Alpha Gate

The history residual alpha gate can now receive cheap prior-summary features via
`--alpha-prior-features`:

```text
entropy_norm(log P_trie)
top_mass
top1 - top2 margin
matched_depth / D
```

The AGTS sidecar does not store raw count mass, so `log_count` is not available
in this path yet. The implementation keeps the existing neural alpha
`g(h)` and adds a learned prior-feature correction to its logit. This avoids
rewriting the residual pipeline while still letting the gate condition on prior
confidence.

Comparable sampled stream run:

```text
out: rnd/rnn-agpt/history_stream_k64_d64_steps1000_l2_5_alpha_priorfeatures.json
mode=stream, k=64, d_model=64, steps=1000, residual_l2=5.0,
learn_alpha=true, alpha_prior_features=true, eval_positions=8192
```

| run | prior PPL | residual PPL | avg alpha |
|-----|----------:|-------------:|----------:|
| learned alpha, no prior features | 3.900171 | 3.899910 | 0.0711 |
| learned alpha + prior features | 3.900171 | 3.886003 | 0.0300 |

The prior-feature gate made the residual much more selective: alpha dropped
while raw residual norms grew. This is a better expression of the prior +
residual split, though the number above is still sampled eval rather than
full heldout eval.

Full fixed-lag heldout eval:

```text
out: rnd/rnn-agpt/history_stream_k64_d64_steps1000_l2_5_alpha_priorfeatures_evalall.json
```

| run | positions | prior PPL | residual PPL | avg alpha |
|-----|----------:|----------:|-------------:|----------:|
| learned alpha + prior features, full eval | 54,736 | 3.946054 | 3.931027 | 0.0303 |

This confirms the sampled improvement was not just sampling noise.

Refreshed simple-prior sidecar:

```text
count gate:
  rnd/rnn-agpt/count_prior_d16_simple_standardized_sidecar.json
sidecar:
  /tmp/agpt_rnn_prior_d16_simple_standardized.agts
history residual:
  rnd/rnn-agpt/history_stream_k64_d64_steps1000_l2_5_alpha_priorfeatures_simple_prior_evalall.json
```

The sidecar was exported from the current simple standardized count-gate
feature set:

```text
core + mass_norm + branch_norm + entropy_delta + suffix_stats
```

Direct count-gate eval for this export:

| metric | positions | PPL |
|--------|----------:|----:|
| heldout rolling learned gate | 55,759 | 3.791926 |
| heldout fixed skip-depth learned gate | 55,744 | 3.791902 |

History-script full legal eval with the refreshed sidecar:

| run | positions | prior PPL | residual PPL | avg alpha | avg attention entropy |
|-----|----------:|----------:|-------------:|----------:|----------------------:|
| learned alpha + prior features, refreshed simple prior | 54,736 | 3.837559 | 3.836515 | 0.0301 | 3.9395 |

This closes most of the gap caused by using the older dense sidecar
(`3.9461 -> 3.8376` prior-only in the history script), but the neural history
residual itself still adds only a small improvement (`0.0010` PPL). The
remaining difference between the direct count-gate diagnostic (`~3.7919`) and
the AGTS sidecar path (`3.8376`) needs its own audit before treating the
history-residual result as fully apples-to-apples with the pure prior.

Sidecar audit:

```text
src/tools/agpt_sidecar_audit.py
out: rnd/rnn-agpt/sidecar_audit_simple_prior_history.json
```

The audit compares the live count gate and AGTS sidecar on the exact same
history-stream legal positions:

| path | positions | PPL |
|------|----------:|----:|
| live count gate on full depth-16 context | 54,736 | 3.803192 |
| live count gate on the suffix matched by the sidecar catalog | 54,736 | 3.837546 |
| AGTS sidecar logits | 54,736 | 3.837559 |

This isolates the mismatch. The sidecar export itself is effectively exact for
the context it stores: average L1 delta versus the live matched-suffix
distribution is only `1.7e-5`. The loss comes from lookup coverage. The current
sidecar catalog is the position-data substring catalog, not a complete catalog
of every context the live count model can use. Therefore, for most heldout
contexts, the sidecar backs off to the longest suffix present in the position
catalog even when the live count model has a longer usable count context.

Conclusion: to make the residual prior apples-to-apples with the direct count
gate, AGTS needs a catalog/export keyed by the count model's context table, or
the history script needs a live count-prior path. The current position-data
catalog is sufficient for AGPT node targets, but it is too sparse for arbitrary
rolling heldout context lookup.

Live count-prior path:

`src/tools/agpt_history_residual.py` now supports an additive option:

```text
--count-prior-result rnd/rnn-agpt/count_prior_d16_simple_standardized_sidecar.json
```

This rebuilds the learned count gate from the saved JSON and uses it directly
for prior distributions. The existing `--prior-sidecar` AGTS path is unchanged;
the script requires exactly one of `--prior-sidecar` or `--count-prior-result`.

Full fixed-lag residual run:

```text
out: rnd/rnn-agpt/history_stream_k64_d64_steps1000_l2_5_alpha_priorfeatures_live_prior_evalall.json
```

| run | positions | prior PPL | residual PPL | avg alpha | avg attention entropy |
|-----|----------:|----------:|-------------:|----------:|----------------------:|
| learned alpha + prior features, live count prior | 54,736 | 3.803192 | 3.801722 | 0.0309 | 3.8666 |

This is the cleanest result for the current fixed-lag history residual. The
residual still helps, but only slightly (`0.0015` PPL). The learned alpha keeps
the applied residual small, which is probably correct given how strong the
count prior is.

Surprise-filtered residual training:

`src/tools/agpt_history_residual.py` now supports:

```text
--surprise-top-frac 0.25
```

This trains the residual only on the top fraction of each batch by prior
surprise:

```text
surprise_t = -log p_prior(x_t | context_t)
```

Evaluation still scores every legal heldout position. First full run:

```text
out: rnd/rnn-agpt/history_stream_k64_d64_steps1000_l2_5_alpha_priorfeatures_live_prior_surprise25_evalall.json
```

| run | positions | prior PPL | residual PPL | avg alpha | avg attention entropy |
|-----|----------:|----------:|-------------:|----------:|----------------------:|
| live prior, top-25% surprise training | 54,736 | 3.803192 | 3.967901 | 0.1579 | 2.1724 |
| live prior, top-50% surprise training | 54,736 | 3.803192 | 3.818943 | 0.0847 | 3.0085 |

This confirms surprise is a strong training signal: attention entropy dropped
from `3.8666` in the all-row live-prior run to `2.1724` at top-25% and
`3.0085` at top-50%. Alpha also rose from `0.0309` to `0.1579` and `0.0847`.
But both hard filters overcorrect on full heldout. The trend is sensible:
top-50% is much less harmful than top-25%, but still worse than the prior.
The idea is useful as a diagnostic, but hard masking is not the right shape with
the current residual head and L2 price. Gentler variants would need a larger
kept fraction, soft surprise weighting, stronger residual pricing, or a better
residual projection space.
