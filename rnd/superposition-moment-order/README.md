---
title: Superposition moment order
kind: diagnostic
status: concluded
outcome: n/a
question: >-
  How many moments of the superposed occurrence cloud does a trained head's softmax over
  older occurrences need, compared with keeping the occurrences themselves?
answer: >-
  Far more than superposition saves, for any polynomial readout. In a regular d64/L2 model
  trained on 124-character windows, layer-1 heads are sharp (median window entropy 0.60 nats
  against 2.77 uniform, median largest older-offset score 14). Reproducing their attention
  distribution needs moment order 8 for p90 KL 0.012 nats, and order 16 for p99 KL below 1e-3
  with the better of two estimators (Chebyshev interpolation; centred Taylor needs 24). That is
  265,729 to about 3.9e7 combination frequencies per vocabulary symbol (about half of them
  independent complex numbers, shared by the layer's heads), where the occurrences it would
  replace number at most 108 for all symbols together. A nonlinear decoder can do far better than a polynomial, but by
  the compressed-sensing lower bound it can't use a state smaller than the occurrence list
  itself.
opened: 2026-11-10
updated: 2026-11-10
code: main
eval: none
family: attention
headline:
- label: regular d64/L2, 124-char windows; offsets 16–123 superposed
  metric: moment order for p99 attention KL < 1e-3 (best of centred Taylor and Chebyshev)
  value: 16
  run: 20261110T233226-window-s124-d64l2
tags: [position, rope, attention, context-length]
related: [superposed-tail-rope, window-d124-baseline, cudax-d124-probe, sampled-node-phase-rope]
---

# superposition-moment-order

**Status:** concluded (2026-11-10). This is a measurement on existing
checkpoints, with no training. The derivation it tests is in
`notes/seq-len-extension/superposition-softmax-limit.md`.

**Revised 2026-11-10.** The first version said superposition compresses only
the linear readout. That holds for polynomial readouts, the only kind measured
here. The sum itself loses no information, and the limit for a nonlinear
decoder is the compressed-sensing bound (note §4). The note's §5 adds an exact
alternative, a polynomial positional kernel. The `20261110T233226-*` runs add a
Chebyshev estimator and reproduce every number of the `20261110T210803-*` runs
exactly.

## Question

Summing a character's occurrences into one RoPE state keeps the *sum* of
their points φ(Δ) = (e^{iω_j Δ})_j on the RoPE torus. That is exact for a
linear readout. Softmax needs Z_c = Σ_p exp(s_p), which depends on the whole
cloud of points. With the cloud's moment tensors up to order m, the exact
quantities available are Σ_p s_p^k for k ≤ m. So how large must m be for a
real trained head, and what does that cost against simply keeping the
occurrences?

## Method

`src/tools/agpt_superposition_moments.py`, layer 1 only. That is the only
layer where queries and keys depend on the character alone, which makes
every score exactly enumerable.

- **Queries:** every 23rd position from 1,024 on, in each of the 10 held-out
  chunks of the canonical split: 1,980 queries × 4 heads.
- **Positions:** the window (offsets 0–15) stays exact. Older occurrences at
  offsets [16, max_offset) are grouped by character. The attention
  distribution covers the 16 window positions plus one group per character.
  Values depend only on the character, so this distribution is what the
  head's output depends on.
- **Estimators of Z_c from moments up to order m** (even m, so the Taylor
  polynomial stays positive). Moments up to order m give Σ_p P(s_p) for any
  degree-m polynomial P, so each estimator is a choice of P:
  - `taylor0`: Σ_p T_m(s_p), the Taylor series of exp at 0.
  - `centred`: e^{s̄_c} Σ_p T_m(s_p − s̄_c), where s̄_c is the group's mean
    score (order-1 information).
  - `chebyshev`: Σ_p C_m(s_p), where C_m interpolates exp at the Chebyshev
    nodes of [lo, hi]. That interval is the score's range over every offset
    in 16–123 for this query character, key character and head. It is known
    before any occurrence is seen, so C_m is a fixed polynomial.
    Chebyshev interpolation is near-best in uniform error, though not
    necessarily for KL.
- **Metric:** KL(exact ‖ approximate) of each (query, head) attention
  distribution.
- **Self-checks** in every run:
  - The factored score matched the head's actual RoPE logit to ≤ 2.5e-6
    (float32 rounding).
  - Σ_p s_p^k computed from the order-k moment tensor matched direct
    summation to ≤ 1.5e-16 for k = 2 and 3.
  - The order-24 Chebyshev interpolant matched exp at its nodes to ≤ 2.3e-15
    of the interval's maximum.

## Results

| Model (checkpoint) | Offsets superposed | Older attention mass, mean / p99 | Max older score, median | Window entropy, median (uniform 2.77) | Window max weight, median |
|---|---|---|---|---|---|
| Regular d64/L2, 124-char windows (`window-d124-baseline/…-s124-25ep`) | 16–123 (trained) | 0.042 / 0.389 | 14.1 | 0.597 | 0.842 |
| AGPT depth-124 static25 (`cudax-d124-probe/…-static25`) | 16–123 (trained) | 0.272 / 1.0 | 43.1 | 0.007 | 0.999 |
| AGPT depth-16, 625 ep (`gradient-population/…-exact-625ep`) | 16–63 (**extrapolated**) | 0.764 / 1.0 | 10.6 | 0.521 | 0.859 |

KL(exact ‖ order-m), p50 / p90 / p99, in nats:

| Model | Estimator | m = 2 | m = 4 | m = 8 | m = 12 | m = 16 | m = 24 |
|---|---|---|---|---|---|---|---|
| Regular, s124 | centred | 8.8e-3 / 0.13 / 0.99 | 2.9e-3 / 0.059 / 0.57 | 2.6e-4 / 0.012 / 0.15 | 1.2e-5 / 2.0e-3 / 0.037 | 1.3e-7 / 1.6e-4 / 8.5e-3 | 1.4e-13 / 2.2e-8 / 3.6e-5 |
| Regular, s124 | chebyshev | 0.018 / 0.33 / 3.2 | 0.074 / 0.92 / 5.9 | 6.0e-4 / 0.025 / 0.27 | 5.0e-7 / 1.9e-4 / 4.2e-3 | 1.4e-11 / 4.7e-7 / 6.6e-5 | 2.5e-17 / 9.8e-16 / 3.5e-10 |
| AGPT depth-124 | centred | 1.4e-3 / 0.95 / 11 | 3.3e-4 / 0.44 / 8.1 | 2.7e-5 / 0.094 / 3.9 | 1.4e-6 / 0.021 / 1.7 | 7.9e-8 / 4.9e-3 / 1.1 | 1.8e-11 / 1.4e-4 / 0.44 |
| AGPT depth-124 | chebyshev | 1.5e-3 / 0.21 / 470 | 7.7e-4 / 0.76 / 740 | 3.3e-5 / 0.10 / 730 | 5.5e-7 / 4.6e-3 / 740 | 3.1e-9 / 1.1e-4 / 59 | 3.1e-14 / 2.6e-8 / 2.4e-3 |

**Chebyshev against centred Taylor.** For the regular model, Chebyshev
reaches p99 KL below 1e-3 at order 16 instead of 24: about two thirds of the
order, and 3.9e7 combination frequencies per symbol instead of 8.7e8. Below
order 10 it is worse. Its interval is fixed in advance and wide, and a
uniform-error polynomial undershoots near the bottom of the interval, while
the centred estimator adapts to each group's mean. The p99 values of several hundred nats on the
depth-124 model can only come from estimates clamped at 1e-300, so there the
polynomial went non-positive for groups that carry real mass.

The `taylor0` estimator is unusable. For the regular model its median KL
*rises* from 0.11 at m = 2 to 2.3 at m = 12. Scores around −14 make T_m(s)
huge while exp(s) ≈ 1e-6, and the series only turns over once m ≳ e·|s| ≈ 38.
That is the predicted cost of sharpness. Full tables are in each run's
`metrics.json`.

**The cost of order m.** Order m needs the cloud's Fourier coefficients at
every signed combination ±ω_{j1} ± … ± ω_{jk} with k ≤ m. With 8 frequencies
per head, that is D(8, m) combination frequencies (a Delannoy number) for
each vocabulary symbol. The coefficient at −n is the conjugate of the one at
n, so about (D + 1)/2 of them are independent complex numbers. RoPE uses the same
frequencies in every head, so the layer's 4 heads share these states; only
the readout differs per head.

| m | 1 | 2 | 4 | 8 | 12 | 16 | 24 |
|---|---|---|---|---|---|---|---|
| D(8, m) per symbol | 17 | 145 | 3,649 | 265,729 | 4,673,345 | 39,490,049 | 872,893,441 |
| Independent complex numbers, whole layer (65 symbols) | 585 | 4,745 | 118,625 | 8.6e6 | 1.5e8 | 1.3e9 | 2.8e10 |

Offsets 16–123 hold at most 108 occurrences, of all symbols together.
Storing them, which is ordinary attention over a 124-character window, takes
108 entries. At order 16 the layer's state is about 10 GB in complex64, and
each query's readout runs over about 5e9 terms. Direct attention over the
108 occurrences is 6,912 multiply-adds (note §7).

## Reading

- **Trained heads are sharp, and AGPT's are no softer.** The regular model's
  layer-1 heads put a median 0.84 of their window weight on one position,
  and the AGPT depth-16 heads 0.86. Their scores against older occurrences
  reach a median of 14.
- **A polynomial readout of softmax over superposed occurrences needs order
  8–16.** That is 10^5 to 4×10^7 combination frequencies per vocabulary
  symbol, against at most 108 occurrences of all symbols together. At this
  sharpness, a polynomial readout expands the
  representation instead of compressing it. The only polynomial readout that
  compresses is the linear one (m = 1, 17 coefficients), and trained heads
  are far from linear: the m = 2 error is already 0.13 nats at p90.
- **A nonlinear decoder doesn't change the conclusion.** It can recover the
  occurrences from far fewer numbers than the moment tower. But by the
  compressed-sensing lower bound, it can't recover them stably from a state
  smaller than the occurrence list itself (note §4).
- **The older text matters to this model's layer 1, modestly:** a mean
  4.2% of attention, and 39% at p99, goes to offsets 16–123.
- **The AGPT depth-124 static25 model is pathologically sharp:** window
  entropy 0.007 nats and scores up to 43. It is weakly trained (rolling byte
  PPL 7.47) and predates the race fix, so it's a curiosity, not evidence.
- **The depth-16 model's older-offset numbers are extrapolation.** It never
  trained beyond offset 15, so its heads would put 76% of their weight
  there. Only its in-window numbers are meaningful.

## Caveats

- **The estimators aren't optimal for KL.** The best polynomial
  needs order 16 at most for p99 KL below 1e-3, and possibly less. Even
  order 8 costs 265,729 combination frequencies per symbol against 108
  occurrences, so the
  conclusion doesn't depend on which polynomial is optimal.
- Layer 1 only. In deeper layers, keys depend on context and can't be
  superposed exactly at any order.
- The measure is the attention distribution, not end-to-end loss. A head
  may tolerate a small KL, but the m = 2 and m = 4 errors are not small.
- There is one regular long-context model (d64/L2, 25 epochs). A larger
  model would likely be sharper still, not softer.
