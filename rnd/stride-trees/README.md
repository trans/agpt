# Stride Trees

Status: active.

This line tests whether AGPT can expose longer-range structure by building
prefix trees over strided corpus positions:

```text
stride 1: x[t], x[t+1], x[t+2], ...
stride 2: x[t], x[t+2], x[t+4], ...
```

For stride 2, separate phase trees are built:

```text
phase 0 starts: t mod 2 == 0
phase 1 starts: t mod 2 == 1
```

Depth still counts consumed tree tokens. A depth-8 stride-2 path consumes 8
tokens but spans 16 original corpus positions.

## Builder

`bin/agpt_build_radix_corpus` now supports:

```text
--stride N
--phase P
--target-offset N
```

Semantics:

```text
for each start i where i mod stride == phase:
  insert x[i], x[i+stride], x[i+2*stride], ..., x[i+max_depth*stride]
```

The final inserted token after `max_depth` stride steps is used to populate the
next-token distribution at the depth cap, matching the ordinary radix builder's
depth/target convention.

With `--target-offset N`, the node target is changed from the next stride child
to the original corpus token `N` positions after the prefix endpoint. For
stride-2, `--target-offset 1` creates a target-aligned "next real character"
head:

```text
prefix x[t], x[t+2], x[t+4], ... predicts x[endpoint+1]
```

## First Build

Clean train split:

```text
train: data/.splits/4fa9aec1db6b3aea/train_corpus.txt
vocab: data/input.txt
```

Commands:

```text
bin/agpt_build_radix_corpus \
  --corpus data/.splits/4fa9aec1db6b3aea/train_corpus.txt \
  --vocab-file data/input.txt \
  --max-depth 8 \
  --stride 2 \
  --phase 0 \
  --out /tmp/agpt_stride2_p0_clean_d8_radix

bin/agpt_build_radix_corpus \
  --corpus data/.splits/4fa9aec1db6b3aea/train_corpus.txt \
  --vocab-file data/input.txt \
  --max-depth 8 \
  --stride 2 \
  --phase 1 \
  --out /tmp/agpt_stride2_p1_clean_d8_radix
```

Build summaries:

| tree | radix nodes | total edge chars | max endpoint depth |
|------|------------:|-----------------:|-------------------:|
| adjacent d8 clean baseline | 814,760 | 1,495,893 | 8 |
| stride-2 phase 0 d8 | 699,371 | 1,870,049 | 8 |
| stride-2 phase 1 d8 | 699,695 | 1,867,969 | 8 |
| stride-4 phase 0 d8 | 349,295 | 1,213,884 | 8 |
| stride-4 phase 1 d8 | 349,024 | 1,214,197 | 8 |
| stride-4 phase 2 d8 | 348,726 | 1,215,297 | 8 |
| stride-4 phase 3 d8 | 348,744 | 1,214,759 | 8 |
| stride-16 phase avg d8 | 85,707 | 345,752 | 8 |

The stride trees have fewer radix storage nodes than the adjacent tree, but more
leveled nodes. That means the every-other-token paths branch less often and
compress into longer unary edges.

## Structure Profile

Depth-cap comparison:

| tree | cap nodes | cap total count | cap mean count | cap avg branch | cap avg edge len | cap singleton |
|------|----------:|----------------:|---------------:|---------------:|-----------------:|--------------:|
| adjacent d8 clean baseline | 587,028 | 1,059,634 | 1.81 | 1.23 | 2.08 | 87.72% |
| stride-2 phase 0 d8 | 508,959 | 529,817 | 1.04 | 1.02 | 3.24 | 98.58% |
| stride-2 phase 1 d8 | 508,983 | 529,817 | 1.04 | 1.02 | 3.23 | 98.58% |
| stride-4 phase 0 d8 | 264,724 | 264,909 | 1.00 | 1.00 | 4.24 | 99.96% |
| stride-4 phase 1 d8 | 264,735 | 264,909 | 1.00 | 1.00 | 4.25 | 99.96% |
| stride-4 phase 2 d8 | 264,730 | 264,908 | 1.00 | 1.00 | 4.25 | 99.96% |
| stride-4 phase 3 d8 | 264,752 | 264,908 | 1.00 | 1.00 | 4.25 | 99.97% |
| stride-16 phase avg d8 | 66,227 | 66,227 | 1.00 | 1.00 | - | 100.00% |

Interior shape:

| tree | depth-2 nodes | depth-2 avg branch | depth-4 nodes | depth-4 avg branch | compression |
|------|--------------:|-------------------:|--------------:|-------------------:|------------:|
| adjacent d8 clean baseline | 1,132 | 9.90 | 24,675 | 4.57 | 1.84x |
| stride-2 phase 0 d8 | 2,060 | 13.45 | 50,663 | 4.22 | 2.67x |
| stride-2 phase 1 d8 | 2,060 | 13.37 | 50,690 | 4.21 | 2.67x |
| stride-4 phase 0 d8 | 2,400 | 14.82 | 41,656 | 3.20 | 3.48x |
| stride-4 phase 1 d8 | 2,395 | 14.92 | 41,367 | 3.20 | 3.48x |
| stride-4 phase 2 d8 | 2,404 | 14.89 | 41,518 | 3.21 | 3.48x |
| stride-4 phase 3 d8 | 2,403 | 14.86 | 41,403 | 3.22 | 3.48x |
| stride-16 phase avg d8 | - | - | - | - | 4.03x |

Late-depth branching collapse:

| tree | depth-6 nodes | depth-6 avg branch | depth-7 nodes | depth-7 avg branch |
|------|--------------:|-------------------:|--------------:|-------------------:|
| stride-2 phase 0 d8 | 40,137 | 2.56 | 17,257 | 2.36 |
| stride-2 phase 1 d8 | 40,381 | 2.56 | 17,655 | 2.35 |
| stride-4 phase 0 d8 | 2,780 | 2.18 | 503 | 2.10 |
| stride-4 phase 1 d8 | 2,810 | 2.22 | 483 | 2.11 |
| stride-4 phase 2 d8 | 2,730 | 2.19 | 466 | 2.12 |
| stride-4 phase 3 d8 | 2,794 | 2.19 | 457 | 2.13 |
| stride-16 phase avg d8 | 62 | 2.02 | 4 | 2.00 |

Cap-edge start depth:

| tree | cap starts <= d3 | cap starts d4 | cap starts d5 | cap starts >= d6 |
|------|-----------------:|--------------:|--------------:|-----------------:|
| adjacent d8 clean baseline | 0.38% | 2.57% | 8.94% | 88.12% |
| stride-2 phase 0 d8 | 1.62% | 12.03% | 28.92% | 57.42% |
| stride-2 phase 1 d8 | 1.60% | 12.01% | 28.87% | 57.52% |
| stride-4 phase 0 d8 | 5.34% | 33.41% | 44.54% | 16.71% |
| stride-4 phase 1 d8 | 5.36% | 33.66% | 44.25% | 16.74% |
| stride-4 phase 2 d8 | 5.39% | 33.60% | 44.67% | 16.33% |
| stride-4 phase 3 d8 | 5.40% | 33.53% | 44.60% | 16.47% |
| stride-16 phase avg d8 | 20.53% | 53.17% | 23.30% | 2.99% |

Initial read:

- Phase 0 and phase 1 are structurally almost identical, which is good.
- Stride-2 has broader shallow branching than the adjacent tree.
- Stride-2 becomes singleton-dominated very quickly by the cap: `98.58%`
  singleton at depth 8.
- Stride-4 makes the collapse much stronger: by depth 7 only about 500
  branching endpoints remain per phase, and the cap is effectively all
  singleton (`99.96%` to `99.97%`).
- Stride-16 is essentially collapsed. The cap is 100% singleton, about 74% of
  cap edges start by depth 4, and each phase has only about four branching
  endpoints left at depth 7.
- The collapse is not immediate at depth 3. For stride-4, only about `5.4%` of
  cap edges start at or before depth 3. The main collapse happens in the middle:
  about one third of cap edges start at depth 4 and about 45% start at depth 5.
- The effective original span is longer, but the target distribution at the cap
  is much more often deterministic. That may make standalone stride-2 training
  easy to fit but less useful unless the learned state is fused back into an
  adjacent-token objective.

## Next Questions

- Train standalone stride-2 phase models and evaluate same-phase heldout PPL.
  This tests whether the strided recurrence learns, but it is not directly
  ordinary next-character PPL.
- Build depth-4 stride-2 trees to compare the same original span as adjacent
  depth 8, but with fewer consumed tokens.
- Design the useful integration: feed stride state into adjacent prediction,
  probably through an auxiliary term:

```text
h1_child = tanh(W_h h1_parent + W_x emb[x[t+1]] + W_aux h2_state + b)
```

or through the output head:

```text
logits = W_o h_stride1 + W_o2 h_stride2 + c
```

## Frozen Stride-State Fusion Plan

First integrated probe:

```text
1. Train stride-2 phase trees as standalone recurrent AGPT models.
2. Freeze those stride-2 models.
3. Materialize or recompute h2 states from the stride trees.
4. Train the adjacent stride-1 tanh model with a leak-free h2 feature.
```

Candidate transition:

```text
h1_child = tanh(W1_h h1_parent + W1_x emb[x[t+1]] + W_s2 h2_lookup[t+1] + b1)
```

Leak rule:

```text
h2_lookup[t+1] must end before x[t+1]

predict x[10] -> use even stride-2 state ending at x[8]
predict x[11] -> use odd stride-2 state ending at x[9]
```

The difficult part is not the recurrence; it is aggregation. An adjacent AGPT
prefix node can occur at many corpus positions and therefore multiple target
parities. A frozen stride feature must be phase-aware. Otherwise the model
averages away the same parity signal the stride tree was meant to expose.

Pragmatic first target:

```text
Train two standalone stride-2 d8/d64/pd1 models:
  /tmp/agpt_stride2_p0_clean_d8_d64_pd1_lr001_100.recur
  /tmp/agpt_stride2_p1_clean_d8_d64_pd1_lr001_100.recur
```

Then build the smallest useful fusion trainer/evaluator around those frozen
models.

Early source-model sanity check, epoch 20, same-phase heldout eval:

```text
bin/agpt_recur_perplexity \
  --checkpoint /tmp/agpt_stride2_p0_clean_d8_d64_pd1_lr001_100.epoch_000020.recur \
  --file data/.splits/4fa9aec1db6b3aea/heldout_corpus.txt \
  --vocab-file data/input.txt \
  --seq-len 8 \
  --stride 2 \
  --phase 0

bin/agpt_recur_perplexity \
  --checkpoint /tmp/agpt_stride2_p1_clean_d8_d64_pd1_lr001_100.epoch_000020.recur \
  --file data/.splits/4fa9aec1db6b3aea/heldout_corpus.txt \
  --vocab-file data/input.txt \
  --seq-len 8 \
  --stride 2 \
  --phase 1
```

| source model | epoch | train PPL | same-phase heldout PPL | bpc |
|--------------|------:|----------:|-----------------------:|----:|
| stride-2 phase 0 | 20 | 18.5174 | 17.9496 | 4.1659 |
| stride-2 phase 1 | 20 | 18.7754 | 18.5551 | 4.2137 |
| stride-2 phase 0 | 100 | 16.4385 | 15.3781 | 3.9428 |
| stride-2 phase 1 | 100 | 16.6033 | 15.6746 | 3.9704 |
| stride-2 phase 0 | 300 | 15.9274 | 14.8759 | 3.8949 |
| stride-2 phase 1 | 300 | 15.9794 | 15.0680 | 3.9134 |

This score is not ordinary next-character PPL. It only checks that the
standalone stride-2 source models are learning same-parity continuation before
we use their states as frozen auxiliary features.

Fixed probability-oracle mixture on ordinary heldout next-character PPL:

```text
p_mix = w1 * p_adjacent + w2 * p_stride2_same + w3 * p_stride2_other
```

| stride source epoch | adjacent PPL | best weights | mixture PPL |
|--------------------:|-------------:|--------------|------------:|
| 100 | 6.3104 | w1=0.99, same=0.01, other=0.00 | 6.3062 |
| 300 | 6.3104 | w1=0.99, same=0.01, other=0.00 | 6.3027 |

The same-phase stride oracle adds a tiny amount of complementary signal. The
opposite-phase oracle gets zero weight because, under the current training
convention, it naturally predicts the following opposite-phase token rather
than the current target.

## Target-Aligned Stride Heads

To avoid the opposite-phase target mismatch without exploding the vocabulary
into `65 * 65` bigram symbols, build two additional stride-2 trees with
`--target-offset 1`. These heads predict the next original character after the
stride endpoint:

```text
same-phase head:  context ending at x[t-2] predicts x[t]
target-next head: context ending at x[t-1] predicts x[t]
```

Build summaries:

| tree | radix nodes | total edge chars | max endpoint depth |
|------|------------:|-----------------:|-------------------:|
| stride-2 phase 0 target-offset=1 d8 | 575,853 | 1,558,580 | 8 |
| stride-2 phase 1 target-offset=1 d8 | 576,482 | 1,557,922 | 8 |

Training, d64/depth8/pd1/lr0.001/Adam/100 epochs:

| source model | epoch | train PPL | target-next heldout PPL | bpc |
|--------------|------:|----------:|------------------------:|----:|
| stride-2 phase 0 target-offset=1 | 20 | 11.6665 | 11.4455 | 3.5167 |
| stride-2 phase 1 target-offset=1 | 20 | 11.8752 | 11.5698 | 3.5323 |
| stride-2 phase 0 target-offset=1 | 100 | 9.7872 | 9.1685 | 3.1967 |
| stride-2 phase 1 target-offset=1 | 100 | 9.8181 | 9.1729 | 3.1974 |

Full heldout probability-oracle mixture, using adjacent d8/d64/pd1 epoch 500,
same-phase stride heads at epoch 300, and target-next heads at epoch 100:

```text
p_mix = w1 * p_adjacent + w2 * p_stride2_same + w3 * p_stride2_next
```

| mixture | adjacent PPL | stride same PPL | stride next PPL | best weights | mixture PPL |
|---------|-------------:|----------------:|----------------:|--------------|------------:|
| s1+same | 6.3104 | 14.9717 | - | w1=0.99, same=0.01 | 6.3027 |
| s1+same+next | 6.3104 | 14.9717 | 9.0418 | w1=0.93, same=0.00, next=0.07 | 6.2679 |

This is a real but bounded signal: target-aligned stride heads add more than the
old off-target opposite phase, but the fixed probability mixture still gives
most of the weight to the adjacent model. The result supports the idea that
strided views contain complementary context, while also confirming that a simple
late mixture is not yet a strong integration mechanism.

## Superseded Bigram Alternative

The single-char stride-2 tree splits the corpus into two parity streams. That
means only one phase is directly aligned to predict a given target character:

```text
predict x[10]:
  even stride context predicts x[10]
  odd stride context naturally predicts x[11]
```

A possible correction considered before `--target-offset` was to make the
stride-2 tree operate on adjacent bigram symbols:

```text
y[k] = (x[2k], x[2k+1])
```

Then a stride-2 transition over original characters becomes an adjacent
transition over bigram tokens:

```text
(x[0], x[1]) -> (x[2], x[3]) -> (x[4], x[5]) -> ...
```

This keeps both parities inside the same state. The source model predicts a
distribution over next bigrams:

```text
P((a, b) | previous bigram context)
```

For ordinary next-character prediction, the bigram oracle can be marginalized:

```text
P_next_char(a) = sum_b P((a, b) | context)
```

or, for the second character of the target pair:

```text
P_second_char(b) = sum_a P((a, b) | context)
```

This preserves cross-parity local structure while exposing a wider original
span, but it is not the preferred next step. The vocabulary becomes `V * V`,
which is barely tolerable for a 65-character alphabet and a poor scaling story
for BPE. Target-offset stride heads keep the original vocabulary and directly
fix the target-alignment problem.
