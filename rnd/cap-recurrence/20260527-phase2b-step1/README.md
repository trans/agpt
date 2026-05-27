# Phase 2B step 1 — direct h_in injection diagnostic (no learning)

**Date:** 2026-05-27.
**Status:** Negative/diagnostic. Confirms wiring; rules out scale-only injection.

## Setup

- Trie: Shakespeare 1M d=16 (`/tmp/shake_d16_radix`, 1,607,928 radix nodes)
- Predecessor table: `/tmp/shake_d16_predecessors.bin` (8,510,942 pairs)
- Warm-start h_caps: `rnd/cap-recurrence/20260527-smoke/h_caps.bin` (from Phase 1 smoke)
- Trainer: 1 SE, partition-depth 1, no-accumulate, lr=3e-3 warmup-cosine,
  rmsprop β=0.999, mass-weight log, entropy-λ 1.0.
- Injection mode: direct (no learnable projection); add `scale * h_in[i, :]`
  to `d_x[q_first_of_i, :]` after embedding gather.

## Results

| inject_scale | loss | Δ vs scale=0 |
|---:|---:|---:|
| 0.0 (baseline w/ pred table) | 2.496743 | — |
| 0.1 | 2.497588 | +0.0008 |
| 0.5 | 2.499799 | +0.0031 |
| 1.0 | 2.499184 | +0.0024 |
| 5.0 | 2.574409 | **+0.078 (+3.1%)** |
| 100.0 | 2.623666 | **+0.127 (+5.1%)** |

Baseline (no capture, no pred table, no injection) sits at 2.499-2.502
across 3 seeds — the 0.0/0.1/0.5/1.0 scales are all within that band.

## Findings

**Wiring is correct.** Large scales (5.0, 100.0) clearly degrade PPL,
proving the injection reaches the model.

**Small-scale direct injection is a wash.** Scales 0.1-1.0 are
indistinguishable from no-injection (within cuBLAS non-determinism).
The model cannot use h_in productively without a learnable projection
that aligns it with its embedding space.

This is exactly what was predicted in
`notes/seq-len-extension/cap-recurrence-design.md` Q1: option A
(additive bias) without per-pair learnability is too weak; we need
option B (extra K-token with Wq/Wk) or at minimum a learnable
`W_inject @ h_in` projection.

## Why small scales don't hurt

Two non-mutually-exclusive explanations:
1. h_in vectors are tightly clustered in norm (std/mean ~0.3) and
   likely in direction too — they may align with some "mean post-LN"
   direction that the model's LN layers normalize out.
2. Attention propagation gives later queries the option to attend
   AWAY from the injected position, effectively ignoring the bias.

Either way, the upshot is the same: direct addition is too weak. The
model needs a learnable transform to convert h_in from "post-LN
representation space" into "input-embedding-perturbation space" that
correlates with predictive error.

## Next step

Phase 2B step 2: add a learnable `W_inject` matrix (d_model × d_model)
and train it (simple SGD or piggyback on RMSProp). Project as
`d_x[q_first_of_i] += W_inject @ h_in[i]`. Persist W_inject across
runs in a sidecar file (so we don't rebuild the model file format).

If step 2 shows PPL improvement, that's the cap-recurrence signal we've
been hunting. If step 2 is also flat, we need to reconsider — either
the injection POINT (depth-0 only is too localized) or the architecture
(option A is too weak even with projection, need option B).

## Reproducibility

Build commit: 607d62a + this diagnostic kernel
(launch_inject_h_in_direct in agpt_cap_capture.cuh).

```sh
cp data/input.random.model /tmp/cap_inj.model
AGPT_CAPTURE_H_CAPS=1 \
AGPT_CAPTURE_H_CAPS_IN=rnd/cap-recurrence/20260527-smoke/h_caps.bin \
AGPT_CAPTURE_H_CAPS_OUT=/tmp/cap_inj_h_caps.bin \
AGPT_CAPTURE_H_CAPS_PRED_TABLE=/tmp/shake_d16_predecessors.bin \
AGPT_CAPTURE_H_CAPS_INJECT_SCALE=<scale> \
bin/agpt_train --model /tmp/cap_inj.model --trie-dir /tmp/shake_d16_radix \
    --save /tmp/cap_inj_out.model --epochs 1 ...
```
