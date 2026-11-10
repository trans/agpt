---
title: Superposed-tail RoPE
kind: experiment
status: concluded
outcome: inconclusive
question: >-
  Does attention to history older than the 16-character window help when every character's
  older occurrences are superposed into one RoPE key, and does the timing (phase) in that
  superposition matter?
answer: >-
  Inconclusive, and not worth pursuing in this form. The linear channel stayed almost unused
  (gates near 0) and moved diagnostic held-out NLL by less than 0.012 nats on two seeds. The
  control that should have shown the known gain from 64 characters of context showed none,
  because the runs were undertrained (20k steps, about 20M scored targets, still improving at
  the end), so the test could not detect a gain at all. The user judged the test worthless
  (2026-11-10): ordinary language models are known to gain from 64 characters of context, and
  that had been tested before.
opened: 2026-11-10
updated: 2026-11-10
code: main
eval: legacy
family: attention
tags: [position, rope, context-length, attention]
related: [sampled-node-phase-rope, rope-position-substitution, harmonic-filter-diagnostic, vanilla-attn-torch]
---

# superposed-tail-rope

**Status:** concluded, inconclusive (2026-11-10). The test could not detect a gain: see [What went wrong](#what-went-wrong). The user judged it worthless and the line is not continued in this form.

"Tail" in this record means the text older than the 16-character window.
The mechanism is **not attention over that text.** It is unnormalized linear
attention with keys and values fixed by character identity: no softmax, no
competition with the window, and no per-position content (every 'e' carries
the same value).

## Question

AGPT's attention sees a 16-character window. Can it use what came before,
cheaply, if each character's earlier occurrences are summed into one
superposed RoPE key? And is that because of *when* the characters occurred
(the phase), or only because of *how recently* and *how often*?

This is step 1 of the multi-positional RoPE line that the user named on
2026-11-10. It tests the idea without the trie. If the tail helps here, step 2
puts it on the trie, where averaging the state over a node's occurrences gives
the node's co-occurrence profile (pair-superposed RoPE). The averaging is only
an approximation and would need its own test.

## Idea

The user's starting thought: with no resolution limit, training reduces to one
embedding per vocabulary symbol, each carrying the multi-position RoPE of all
its occurrences. The causal, finite-resolution version is a per-character
state, one complex number per frequency j:

```
S_s[c, j] = (1 − r_j) · Σ_{p < s, x_p = c} λ_j^(s − p),     λ_j = r_j · e^{iω_j}
S_s       = λ ⊙ S_{s−1} + (1 − r) ⊙ λ ⊙ onehot(x_{s−1})        (recurrent form)
```

In layer 1 a key depends only on its character (k = k_c), so RoPE's score
from a query at t = s + i to *all* older occurrences of c factors exactly:

```
Σ_{p < s, x_p = c} q·k (RoPE, at offset t − p)  =  Re Σ_j zq_j · λ_j^i · conj(zk_{c,j}) · S_s[c, j]
```

Each head adds `gate_h · Σ_c w[t, c] · v_c` to its softmax output over the
window. The gate starts at 0, so every arm starts as the baseline. The query
locates by interference. Steering its phases to an offset Δ makes occurrences
at Δ add in phase and the rest cancel. This is linear attention, so there is no
softmax sharpening. It also locates only by character and offset, never by
content. The general form, with context-dependent keys, is RetNet's retention.

## Arms

All arms use the same 8 atoms per head, the same state size and the same
cost per step. They differ only in (ω_j, τ_j), with r_j = exp(−1/τ_j).

| Arm | Frequencies ω_j | Decay τ_j |
|---|---|---|
| `none` | (no tail) | (baseline) |
| `fourier` | the head's RoPE frequencies, base^(−2j/16) | one shared τ = 512 |
| `wavelet` | RoPE frequencies | one wavelength each: 2π/ω_j = 6.3, 20, 63, 199, 628, 1,987, 6,283, 19,869 |
| `decay` | 0 (no phase) | the wavelet ladder |

`wavelet` against `decay` isolates the phase: the two arms have the same
timescales, with and without timing. `wavelet` against `fourier` isolates
constant-Q damping against a single damping.

## Protocol

- **Trainer:** `src/tools/agpt_tail_rope_probe.py`, with no trie. The model
  is the AGPT d64/L2 layout: 4 heads, d_ff 256, pre-LN, RoPE with interleaved
  pairs and base 10000, ReLU, untied head, 108,481 parameters without the
  tail. It uses N(0, 0.02) random init and Adam lr 1.5e-3, with 500 warmup
  steps and then a cosine decay to 10%. There are 20k steps of 1,024 windows.
- **Data:** the canonical split `data/.splits/4fa9aec1db6b3aea/`. Training
  cuts `train_corpus.txt` into documents of 5,568 characters, and the tail
  state resets at each document start.
- **Training targets:** one per window, the character after a full
  16-character window. The first smoke test scored every window position.
  That let the fastest atom (τ ≈ 6) learn to stand in for the missing
  in-window context of early positions. Training loss improved by 0.034 nats,
  but held-out got 0.012 nats worse, with gates near 0.6. Scoring only
  full-window targets matches evaluation exactly.
- **Evaluation (diagnostic, not lm-eval):** each of the 10 held-out chunks
  (5,576 characters) is its own document, so tail history starts at the chunk
  start. Every target is predicted from up to 16 preceding characters plus the
  tail. That gives 55,750 targets, with NLL also reported by position in the
  chunk (1–15, 16–255, 256–1023, 1024 to the end). The 1–15 bucket has short
  windows, which training never scores, so it reads high in every arm.
- **Checks** (2026-11-10, before any run): the fast state matched a
  brute-force sum over all older positions to a relative 1e-7 in all three arms,
  across document resets. The factored score matched the head's actual RoPE
  dot product at offsets 17, 100 and 999.
- **Sweep:** `run_sweep.sh`, with 4 arms × seeds 42 and 43, run one at a
  time. Both seeds draw the same batches in every arm.

## Results

### Sweep: 4 arms × 2 seeds, 20k steps (`20261110T182838-sweep20k/`)

The values are diagnostic held-out NLL in nats per character, from each run's
`metrics.json`. "t ≥ 256" pools the two late buckets: targets at least 256
characters into their chunk, 95% of all targets, where every arm has full
history. Gates are layer 1's four per-head tail gates at the end of training.

| Arm | Seed | All targets | t ≥ 256 | 16–255 | 256–1023 | 1024–end | Gates |
|---|---:|---:|---:|---:|---:|---:|---|
| `none` | 42 | 1.3583 | 1.3578 | 1.3399 | 1.3075 | 1.3663 | — |
| `none` | 43 | 1.3521 | 1.3523 | 1.3220 | 1.3026 | 1.3607 | — |
| `fourier` | 42 | 1.3598 | 1.3592 | 1.3474 | 1.3055 | 1.3682 | +0.02, −0.07, −0.04, −0.03 |
| `fourier` | 43 | 1.3526 | 1.3524 | 1.3250 | 1.2924 | 1.3625 | +0.01, −0.27, −0.07, −0.10 |
| `wavelet` | 42 | 1.3609 | 1.3604 | 1.3363 | 1.3098 | 1.3689 | +0.00, −0.09, −0.01, −0.01 |
| `wavelet` | 43 | 1.3559 | 1.3564 | 1.3196 | 1.2971 | 1.3664 | −0.01, −0.04, −0.02, +0.00 |
| `decay` | 42 | 1.3468 | 1.3463 | 1.3263 | 1.2893 | 1.3560 | −0.02, −0.85, −0.01, −0.03 |
| `decay` | 43 | 1.3545 | 1.3542 | 1.3347 | 1.3068 | 1.3621 | −0.03, −0.24, −0.03, +0.02 |

Change against `none` on the same seed (all targets): `fourier` +0.0016 and
+0.0005, `wavelet` +0.0026 and +0.0038, `decay` −0.0115 and +0.0024. The
baseline's own seed gap is 0.0062.

**Reading.**
- **No arm helps measurably.** Fourier and wavelet are within ±0.004 of the
  baseline on both seeds.
- **Decay is mixed.** It gains 0.012 on seed 42 and loses 0.002 on seed 43,
  while the seed gap is 0.006.
- **No gain grows with available history.** A useful tail should help most
  in the 1024–end bucket, and none does.
- **The model mostly declines the tail.** Most gates end within ±0.1 of 0.
  The one open gate (head 1 in `decay`, −0.85 and −0.24) carries the only
  hint of a gain.
- **The phase does not help.** `wavelet` (timing) is no better than `decay`
  (same timescales, no timing), and if anything slightly worse.

Absolute values are diagnostic, from every target having a full 16-character
window. They are not comparable with lm-eval byte PPL or with AGPT's
fixed-token PPL.

### Positive control: wider softmax windows (`20261110T190338-window-control20k/`)

This is the baseline (no tail) with plain softmax windows of 32 and 64
characters, the same protocol, 20k steps and seeds 42 and 43. Values are
diagnostic held-out NLL. "t ≥ 256" is the fair comparison across windows,
because for targets earlier than the window length, the first block of each
chunk scores them with short windows.

| Window | Seed | All targets | t ≥ 256 | 256–1023 | 1024–end | Train s |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 42 | 1.3583 | 1.3578 | 1.3075 | 1.3663 | 207 |
| 16 | 43 | 1.3521 | 1.3523 | 1.3026 | 1.3607 | 180 |
| 32 | 42 | 1.3468 | 1.3467 | 1.2968 | 1.3552 | 297 |
| 32 | 43 | 1.3511 | 1.3511 | 1.3006 | 1.3596 | 299 |
| 64 | 42 | 1.3589 | 1.3574 | 1.3082 | 1.3656 | 639 |
| 64 | 43 | 1.3518 | 1.3504 | 1.2873 | 1.3611 | 640 |

The change at t ≥ 256 against window 16 on the same seed is −0.0111 and
−0.0012 for window 32, and −0.0005 and −0.0019 for window 64. The seed gap is
0.006.

**Reading:** this control failed. Ordinary language models gain from 64
characters of context over 16. This setup showed no gain because the runs
were undertrained. Each step scores one target per window, so 20k steps are
about 20M targets. The vanilla trainer reached its plateau after about 170M
(`rnd/vanilla-attn-torch`), and every curve here was still falling at 20k
steps. A model that hasn't yet learned the short-range patterns has nothing
to gain from more context.

## What went wrong

- **The positive control did not reproduce a known effect,** and I read
  that as a property of the model instead of a fault in the setup. An
  earlier version of this README concluded that "this model can't use
  context beyond 16" and that context isn't AGPT's bottleneck at this size.
  Both claims are withdrawn: they rested on undertrained runs.
- **The channel was weak by construction:** linear, character-keyed,
  additive and in layer 1 only. Even at a budget with power, a null would
  say little about superposed RoPE in general.
- **The question was largely settled before:** that context beyond 16 helps
  an ordinary model, and the earlier superposition-style RoPE records
  (`dist-rope-smoke`, `harmonic-bias-prototype`). The record should have
  been checked with the user before building a new test.

The trainer (`src/tools/agpt_tail_rope_probe.py`) and its exactness checks
are kept, in case a properly budgeted comparison is wanted later.
