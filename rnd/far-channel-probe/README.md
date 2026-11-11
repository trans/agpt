---
title: Far-channel probe
kind: experiment
status: concluded
outcome: positive
question: >-
  Can the text older than a 16-character softmax window be read through a channel that is
  exact and recurrent but blurry in position (linear content, a smooth query-steered positional
  kernel), and how much of the gain of exact attention over the same offsets does it recover?
answer: >-
  Yes, partly. In layer 2 of a d64/L2 model, beside an exact 16-character window, a channel with
  linear content and a constant-Q positional kernel recovers 65% of the gain of exact softmax over
  the same offsets (16-127). It falls short of exact by 0.0104 +/- 0.0009 nats on every one of 4
  seeds. A recency-only kernel recovers 48%. The channel is exact and recurrent with a fixed
  state, but at this reach that state is no smaller than the keys and values it replaces.
opened: 2026-11-11
updated: 2026-11-11
code: main
eval: legacy
family: attention
tags: [position, attention, context-length, linear-attention]
headline:
- label: constant-Q far channel, layer 2, offsets 16-127, 4 seeds
  metric: fraction of exact attention's gain over the 16-window recovered (pooled)
  value: 0.65
  run: 20261111T002822-30k-cq-s42
- label: constant-Q far channel, layer 2, offsets 16-127, 4 seeds
  metric: diagnostic held-out NLL shortfall vs exact attention, nats (paired mean)
  value: 0.0104
  run: 20261111T002822-30k-cq-s42
related: [superposition-moment-order, superposed-tail-rope, vanilla-attn-torch, window-d124-baseline]
---

# far-channel-probe

**Status:** concluded (2026-11-11). Four arms × 4 seeds at 30k steps.

## Question

The softmax-limit note (`notes/seq-len-extension/superposition-softmax-limit.md`)
shows that a superposed state can't cheaply reproduce a sharp softmax over
occurrences. A channel that is exact and recurrent has to give something up.
The user's point (2026-11-10): a blurry position is all that's really needed,
as long as it stays ordinal. The note's §5 kernel is blurry in position. In
layer 1, though, its values are single characters, so it would return a
positionally weighted bag of characters. In layer 2 the keys depend on
context, and exactness there needs linear content. The user agreed to linear
content (2026-11-10).

So: put such a channel in layer 2, beside an exact 16-character window, and
compare it with exact softmax over the same offsets.

## Design

`src/tools/agpt_far_channel_probe.py`. No trie. A plain minibatch-Adam
transformer with the AGPT d64/L2 layout (4 heads, d_ff 256, pre-LN, RoPE with
interleaved pairs, base 10000, ReLU, untied head). Layer 1 always attends over
offsets 0–15. The arms differ only in layer 2:

| Arm | Layer 2 | Receptive field | Params |
|---|---|---|---|
| `w16` | softmax over offsets 0–15 (baseline) | 30 | 108,481 |
| `exact` | softmax over offsets 0–127 (upper bound) | 142 | 108,481 |
| `decay` | softmax over 0–15, plus far channel over 16–127, recency kernel | 142 | 118,885 |
| `cq` | the same, constant-Q kernel | 142 | 120,965 |

Spans compound across layers: layer 2 at t reads layer-1 states up to its
span back, and each of those reads 15 more. So `w16` already reaches 30
characters, and the other arms reach 142.

**The far channel**, per head, for the query at t and a key at p, Δ = t − p:

```
a[t, p] = φ(qf_t) · φ(kf_p) · w_t(Δ),        16 ≤ Δ < 128,  φ(z) = elu(z) + 1
far_t   = Σ_p a[t, p] v_p / Σ_p a[t, p]
out_t   = (softmax over offsets 0–15) + gate_h · far_t,     gate_h starts at 1
```

qf and kf are the channel's own projections, and v is the layer's value. The
content term has no position: position enters only through the kernel, which
the query steers.

- **`decay`:** w_t(Δ) = Σ_a π_{t,a} e^{−(Δ−16)/τ_a}, with π_t a softmax over 8
  atoms and τ = 2, 4, …, 256. It is completely monotone, so it expresses
  recency only.
- **`cq`:** w_t(Δ) = |Σ_j b_{t,j} λ_j^{Δ−16}|², with b_t complex and 8
  constant-Q atoms λ_j = e^{−1/P_j} e^{2πi/P_j}, P = 4, 8, …, 512. It is
  nonnegative, and it can put a bump at a chosen distance, with a width that
  grows with the distance.

Both kernels are finite sums of exponentials in Δ, so the channel is exactly
recurrent: one state Σ_p μ^{t−p} φ(kf_p) v_pᵀ (and its normalizer) per atom μ,
8 atoms for `decay` and 64 products λ_j conj(λ_l) for `cq`. Training uses the
parallel form, which is the same sum.

## Protocol

- **Training:** Adam, lr 1.5e-3, 500 warmup steps, then a cosine decay to
  10%. Each step is 16 blocks of 512 characters from the canonical split's
  `train_corpus.txt`, with every position scored (8,192 targets). 30k steps
  is about 246M targets, about 1.5× the ~170M at which vanilla seq-16
  attention plateaued (`rnd/vanilla-attn-torch`).
- **Evaluation (diagnostic, not lm-eval):** each of the 55,750 targets in the
  10 held-out chunks is scored once, with at least 144 preceding characters
  where the chunk has them. NLL is also reported by the target's position in
  its chunk: 1–30 (short context for every arm), 31–142 (full context for
  `w16` only), and 143 to the end (full context for every arm: the main
  comparison).
- **Order of runs:** the controls (`w16`, `exact`) first. The far arms are
  run only if `exact` clearly beats `w16` at this budget, because a control
  that can't show the known gain from context makes the test meaningless
  (`rnd/superposed-tail-rope`).

**Checks** (run at start-up in every run, 2026-11-11 smoke tests):
- Changing inputs more than 142 characters back changes no arm's output
  (max change 0.0).
- Changing inputs at offsets 31–142 leaves `w16` unchanged (0.0) and changes
  every other arm.
- The far channel's recurrence over atom states matches the parallel form
  to a relative 4e-7 (`decay`, `cq`; float32).
- Each held-out target is scored exactly once (55,750).

## Results

### Four arms × seeds 42–45, 30k steps (`20261111T002822-30k-*`)

Diagnostic held-out NLL in nats per character, from each run's
`metrics.json`, on targets 143 to the end of their chunk (full receptive
field for every arm, 97% of targets). Runs: `run_arms.sh`, one at a time.

| Arm | Seed 42 | Seed 43 | Seed 44 | Seed 45 | Mean | Seed SD | Train loss (mean) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `w16` | 1.3346 | 1.3518 | 1.3427 | 1.3434 | 1.3431 | 0.0070 | 1.216 |
| `decay` | 1.3308 | 1.3248 | 1.3346 | 1.3259 | 1.3290 | 0.0046 | 1.192 |
| `cq` | 1.3220 | 1.3245 | 1.3233 | 1.3260 | 1.3240 | 0.0017 | 1.185 |
| `exact` | 1.3112 | 1.3125 | 1.3155 | 1.3148 | 1.3135 | 0.0020 | 1.182 |

Paired by seed (every arm of a seed draws the same batches and the same
initial weights for the shared layers; the output head's initial weights
differ between the far arms and the others, because the far channel's
modules are created before it):

| Difference | Per seed (42, 43, 44, 45) | Mean ± SE | t |
|---|---|---|---:|
| `exact` − `w16` | −0.0234, −0.0393, −0.0272, −0.0285 | −0.0296 ± 0.0034 | −8.7 |
| `cq` − `w16` | −0.0126, −0.0273, −0.0194, −0.0174 | −0.0192 ± 0.0031 | −6.3 |
| `decay` − `w16` | −0.0038, −0.0270, −0.0081, −0.0175 | −0.0141 ± 0.0052 | −2.7 |
| `cq` − `exact` | +0.0108, +0.0120, +0.0078, +0.0111 | +0.0104 ± 0.0009 | +11.3 |
| `decay` − `exact` | +0.0197, +0.0122, +0.0191, +0.0110 | +0.0155 ± 0.0023 | +6.9 |
| `cq` − `decay` | −0.0088, −0.0002, −0.0113, +0.0001 | −0.0051 ± 0.0029 | −1.7 |

Pooled over seeds, `cq` recovers 65% of `exact`'s gain over `w16`
(per seed 54%, 69%, 71%, 61%), and `decay` 48% (16%, 69%, 30%, 61%). All
heads use the channel: final gates 0.25–0.40 (`cq`) and 0.27–0.44 (`decay`)
on every seed.

**The control holds.** `exact` beats `w16` on every seed, and its curve is
below `w16`'s at every evaluation from 4k steps on. All curves are flat from
about 20k steps, with held-out well above train loss, so the runs are
converged.

**Reading.**
- **The far past can be read through a blurry, exact, recurrent channel.**
  Both kernels beat the window on every seed. Older text helps, and a
  channel with linear content and only an ordinal, smooth sense of position
  delivers most of that help.
- **It is not exact attention.** `cq` falls short of `exact` by a steady
  0.010 nats (standard error 0.0009 over 4 seeds). That is the
  cost of linear content and a smooth kernel at this size.
- **Bumps may beat recency, but it isn't established.** `cq`, whose kernel
  can peak at a distance, beats `decay` by about 0.01 on two seeds and ties
  on two (t = −1.7).
- **The far channel steadies the model.** Its seed SD is 0.0017 (`cq`),
  close to `exact`'s 0.0020, while `w16`'s is 0.0070.
- **The short-context buckets are small.** Targets 1–30 of each chunk
  (300 in all) read worse in the far arms (1.30 against 1.27 for `w16`);
  targets 31–142 (1,120) read better (`cq` 1.290, `exact` 1.289, `w16`
  1.313). Neither bucket bears on the comparison.

### Cost at this reach

The far channel's point is a state that doesn't grow with the reach. At a
reach of 128 it saves nothing. Layer 2's keys and values for offsets 16–127
are 112 × 4 heads × 32 = 14,336 numbers. The channel's state is
8 × 4 × 272 = 8,704 real numbers for `decay` (8 atoms, each a 16 × 16 state
plus a 16-entry normalizer, per head). For `cq` it is 64 × 4 × 272 ≈ 70k real
numbers: the 64 products λ_j conj(λ_l) are 8 real atoms and 28 conjugate
pairs of complex ones. So `cq` pays for itself only past a reach of about
550 characters, and `decay` past about 70. In training, the parallel
form costs more than exact attention here: 34 ms (`decay`) and 42 ms (`cq`)
per step, against 21 ms.

## Caveats

- **Diagnostic evaluation**, not lm-eval byte PPL. It is the same for every
  arm, and its absolute values aren't comparable with canonical PPL.
- **One size and one reach:** d64/L2, a channel in layer 2 only, offsets
  16–127. The efficiency case, a reach where keeping the keys and values
  costs more than the fixed state, is untested.
- **Not on the trie.** This is a plain transformer. In AGPT, a trie node
  merges occurrences that share their last 16 characters but not their
  older text. A far channel makes each occurrence's prediction depend on
  that older text, so the node's occurrences no longer share a gradient.
  Putting the channel into AGPT would break the per-node aggregation unless
  the channel's input is shared per node, which is a separate design
  question.
- **The far arms have about 10% more parameters** (118,885 and 120,965
  against 108,481). The extra parameters see only offsets 16–127, so any gain
  they give comes from the older text. `exact`, with the fewest parameters,
  is still best.

## Next, if wanted

- A longer reach (512 or 1024), where the fixed state is smaller than the
  keys and values, to see whether the 0.010 shortfall holds.
- More seeds, or a sharper test, for `cq` against `decay`.
