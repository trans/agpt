---
title: Superposition moment order
kind: diagnostic
status: concluded
outcome: n/a
question: >-
  How many moments of the superposed occurrence cloud does a trained head's softmax over
  older occurrences need, compared with keeping the occurrences themselves?
answer: >-
  Far more than superposition saves. In a regular d64/L2 model trained on 124-character
  windows, layer-1 heads are sharp (median window entropy 0.60 nats against 2.77 uniform,
  median largest older-offset score 14). Reproducing their attention distribution needs moment
  order 8 for p90 KL 0.012 nats and order 24 for p99 KL below 1e-3. That is 265,729 to
  about 8.7e8 complex coefficients per character, where the occurrences it would replace number
  at most 108. Superposition compresses only the linear readout, which trained heads don't use.
opened: 2026-11-10
updated: 2026-11-10
code: main
eval: none
family: attention
headline:
- label: regular d64/L2, 124-char windows; offsets 16–123 superposed
  metric: moment order for p99 attention KL < 1e-3 (centred estimator)
  value: 24
  run: 20261110T210803-window-s124-d64l2
tags: [position, rope, attention, context-length]
related: [superposed-tail-rope, window-d124-baseline, cudax-d124-probe, sampled-node-phase-rope]
---

# superposition-moment-order

**Status:** concluded (2026-11-10). This is a measurement on existing
checkpoints, with no training. The derivation it tests is in
`notes/seq-len-extension/superposition-softmax-limit.md`.

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
  polynomial stays positive):
  - `taylor0`: Σ_p T_m(s_p), the Taylor series of exp at 0.
  - `centred`: e^{s̄_c} Σ_p T_m(s_p − s̄_c), where s̄_c is the group's mean
    score (order-1 information).
- **Metric:** KL(exact ‖ approximate) of each (query, head) attention
  distribution.
- **Self-checks** in every run:
  - The factored score matched the head's actual RoPE logit to ≤ 2.5e-6
    (float32 rounding).
  - Σ_p s_p^k computed from the order-k moment tensor matched direct
    summation to ≤ 1.5e-16 for k = 2 and 3.

## Results

| Model (checkpoint) | Offsets superposed | Older attention mass, mean / p99 | Max older score, median | Window entropy, median (uniform 2.77) | Window max weight, median |
|---|---|---|---|---|---|
| Regular d64/L2, 124-char windows (`window-d124-baseline/…-s124-25ep`) | 16–123 (trained) | 0.042 / 0.389 | 14.1 | 0.597 | 0.842 |
| AGPT depth-124 static25 (`cudax-d124-probe/…-static25`) | 16–123 (trained) | 0.272 / 1.0 | 43.1 | 0.007 | 0.999 |
| AGPT depth-16, 625 ep (`gradient-population/…-exact-625ep`) | 16–63 (**extrapolated**) | 0.764 / 1.0 | 10.6 | 0.521 | 0.859 |

KL(exact ‖ order-m), centred estimator, p50 / p90 / p99, in nats:

| Model | m = 2 | m = 4 | m = 8 | m = 12 | m = 16 | m = 24 |
|---|---|---|---|---|---|---|
| Regular, s124 | 8.8e-3 / 0.13 / 0.99 | 2.9e-3 / 0.059 / 0.57 | 2.6e-4 / 0.012 / 0.15 | 1.2e-5 / 2.0e-3 / 0.037 | 1.3e-7 / 1.6e-4 / 8.5e-3 | 1.4e-13 / 2.2e-8 / 3.6e-5 |
| AGPT depth-124 | 1.4e-3 / 0.95 / 11 | 3.3e-4 / 0.44 / 8.1 | 2.7e-5 / 0.094 / 3.9 | 1.4e-6 / 0.021 / 1.7 | 7.9e-8 / 4.9e-3 / 1.1 | 1.8e-11 / 1.4e-4 / 0.44 |

The `taylor0` estimator is unusable. For the regular model its median KL
*rises* from 0.11 at m = 2 to 2.3 at m = 12. Scores around −14 make T_m(s)
huge while exp(s) ≈ 1e-6, and the series only turns over once m ≳ e·|s| ≈ 38.
That is the predicted cost of sharpness. Full tables are in each run's
`metrics.json`.

**The cost of order m.** Order m needs the cloud's Fourier coefficients at
every signed combination ±ω_{j1} ± … ± ω_{jk} with k ≤ m. With 8 frequencies
per head that is the Delannoy number D(8, m) of complex coefficients per
character:

| m | 1 | 2 | 4 | 8 | 12 | 16 | 24 |
|---|---|---|---|---|---|---|---|
| D(8, m) | 17 | 145 | 3,649 | 265,729 | 4,673,345 | 39,490,049 | 872,893,441 |

Offsets 16–123 hold at most 108 occurrences in total. Storing them, which is
ordinary attention over a 124-character window, takes 108 entries.

## Reading

- **Trained heads are sharp, and AGPT's are no softer.** The regular model's
  layer-1 heads put a median 0.84 of their window weight on one position,
  and the AGPT depth-16 heads 0.86. Their scores against older occurrences
  reach a median of 14.
- **Softmax over superposed occurrences needs order 8–24.** That is 10^5 to
  10^9 coefficients per character, against at most 108 occurrences. At
  this sharpness, superposition expands the representation instead of
  compressing it. The only readout it compresses is the linear one (m = 1,
  17 coefficients), and trained heads are far from linear: the m = 2 error
  is already 0.13 nats at p90.
- **The older text matters to this model's layer 1, modestly:** a mean
  4.2% of attention, and 39% at p99, goes to offsets 16–123.
- **The AGPT depth-124 static25 model is pathologically sharp:** window
  entropy 0.007 nats and scores up to 43. It is weakly trained (rolling byte
  PPL 7.47) and predates the race fix, so it's a curiosity, not evidence.
- **The depth-16 model's older-offset numbers are extrapolation.** It never
  trained beyond offset 15, so its heads would put 76% of their weight
  there. Only its in-window numbers are meaningful.

## Caveats

- Layer 1 only. In deeper layers, keys depend on context and can't be
  superposed exactly at any order.
- The measure is the attention distribution, not end-to-end loss. A head
  may tolerate a small KL, but the m = 2 and m = 4 errors are not small.
- There is one regular long-context model (d64/L2, 25 epochs). A larger
  model would likely be sharper still, not softer.
