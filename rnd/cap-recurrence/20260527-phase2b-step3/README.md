# Phase 2B step 3 — option B (learnable K/V-token injection)

**Date:** 2026-05-27 / 2026-05-28.
**Status:** NEGATIVE. Implementation works (forward/backward parity verified, gradient flows, W moves); the mechanism produces no useful signal at this scale. This closes Phase 2B (and likely cap-recurrence in this form).

## TL;DR

Three forms of cap-recurrence injection tested in sequence, each more capable than the last. All three give the same answer:

| Form | Description | Result vs baseline (5-ep, interleaved when applicable) |
|---|---|---|
| Step 1 — direct add | `d_x[q_first] += scale·h_in` (no learning) | Flat at small scales; degrades at large scales. Step-1 README. |
| Step 2 — learnable additive | `d_x[q_first] += W_inject·h_in` (W learned via own RMSProp) | Flat to slightly worse at 5-ep; the 1-ep dip was noise. |
| **Step 3 — option B (this run)** | `K_inject=W_k·h_in, V_inject=W_v·h_in` prepended as p=0 of every layer's KV; shared W across layers | **Flat. Interleaved mean Δ=+0.002. Inference-time ablation shows the trained model does not use the injection slot.** |

The implementation is correct (we verified forward parity, gradient flow, and end-to-end backward through the attention KV pack). The mechanism does not extract a signal because the input does not carry one in a form the model can generalize from.

## Setup

- Corpus: Shakespeare 1M chars, d=16.
- Trie: `/tmp/shake_d16_radix` (1,607,928 radix nodes).
- Predecessor table: `/tmp/shake_d16_predecessors.bin` (8.5M pairs).
- h_caps warm-start: `rnd/cap-recurrence/20260527-smoke/h_caps.bin` (from Phase 1 smoke).
- Init model: `data/input.model` (d_model=64, 2 layers, 4 heads, 108,481 weights).
- Trainer flags: `--epochs 5 --lr 3e-3 --lr-schedule warmup-cosine --optimizer rmsprop --rmsprop-beta 0.999 --partition-depth 1 --no-accumulate --mass-weight log --entropy-lambda 1.0`.
- Step-3 specific: `AGPT_CAP_KV_INJECT=1 AGPT_CAP_KV_INJECT_LR=1e-5` (also tested 1e-4 and 1e-12 frozen).

## Forward parity check (W=0, lr=1e-12)

W_k=W_v=0 makes K_inject=V_inject=0 in the injected slot. Attention still includes `exp(0)=1` in its softmax denominator, so this is NOT bit-parity with baseline — the output gets scaled by `S/(S+1)` where `S = Σ exp(score)`. At d=16 with short prefixes, the perturbation is bigger than the original "~1%" estimate.

Empirically: at 1 epoch, frozen-W KV-inject lands within the baseline noise band [2.565, 2.577]. Wiring confirmed correct.

## 5-epoch results

### Initial (non-interleaved) batch — looked positive

| condition | run-1 | run-2 | run-3 | mean | Δ vs THIS baseline |
|---|---|---|---|---|---|
| baseline | 1.9140 | 1.9326 | 1.9304 | 1.9257 | — |
| KV lr=1e-5 | 1.9231 | 1.9060 | 1.8849 | 1.9047 | −0.021 (−1.1%) |
| KV lr=1e-4 | 1.9001 | 1.9115 | 1.9183 | 1.9100 | −0.016 (−0.8%) |

Looks like a real ~1% improvement. **But this batch's baseline (1.9257) is 0.021 higher than the previous step-2 batch's baseline (1.9045) — same model, same setup, just run hours apart.** Cross-time baseline drift is the same magnitude as the apparent effect. Need interleaving.

### Interleaved 5-pair batch — null

```
pair    baseline    KV lr=1e-5    Δ (kv − base)
 1      1.876       1.883         +0.007
 2      1.897       1.906         +0.008
 3      1.886       1.903         +0.016
 4      1.926       1.907         −0.019
 5      1.919       1.918         −0.002
                                  ──────
mean    1.901       1.903         +0.002
```

Mean Δ = +0.002, within-pair range −0.019 to +0.016. 3/5 pairs favor baseline, 2/5 favor KV. **No signal.** The "1.1% improvement" from the first batch was cross-time baseline drift.

## Inference-time ablation

To distinguish "model learned to ignore KV" from "model learned to rely on KV (false narrative)" from "model genuinely uses KV":

Protocol — single seed each (uncontrolled run-to-run noise on B vs C):
1. Train *kvt*: 5ep with KV active, lr=1e-5.
2. Train *baset*: 5ep baseline.
3. Eval A: kvt model, KV active at eval (lr=1e-12 so weights barely shift).
4. Eval B: kvt model, KV disabled at eval.
5. Eval C: baset model, KV disabled (sanity).

```
                                          loss
Train kvt                               1.879
Train baset                             1.939

Eval A (kvt + KV active)                1.879
Eval B (kvt + KV disabled)              1.881
Eval C (baset + KV disabled)            1.939

‖W_k‖_F=0.073, ‖W_v‖_F=0.067, max|W_k|=0.006 (matrix moved but to a non-useful direction)
```

**The controlled comparison is A vs B** (same model, only difference is whether KV is active at eval). Δ = +0.002 — the KV-trained model gets no benefit at eval from the mechanism it was trained with. Attention learned to route around the inject slot: W_k makes K_inject orthogonal to typical queries, so attention weight on the slot ≈ 0, so V_inject contributes nothing to the output.

The B vs C gap of 0.058 is *uncontrolled* (single shots, 5-ep run-to-run spread is ~0.05) — the interleaved batch already showed Δ(kvt − baset) = +0.002. Most parsimonious reading: sampling noise.

## Three-state logical analysis (Thomas's framing)

For any candidate signal in `h_in`:

| If h_in carries… | Then training should produce… | Observed |
|---|---|---|
| Generalizable useful signal | A < B by some real amount | A ≈ B (Δ=+0.002) — falsified |
| False narrative the model latches onto | B > C (KV-trained without KV underperforms baseline) | B ≤ C — falsified |
| Redundant or noise | A ≈ B, kvt ≈ baset in interleaved test | exactly what we see |

Either way, the mechanism doesn't matter at this scale.

## Why this happened — theoretical reading

**1. Training→inference distribution mismatch on the input.** `h_in[K]` at training is a mass-weighted *average* of `h_cap[K'_i]` over K's corpus predecessors. At inference, the predecessor is one specific concrete prefix — no averaging. For high-mass K (many predecessors) the averaging smears across diverse contexts. For low-mass K (few or single predecessor), `h_in` is essentially the cap of one specific corpus context, frozen — a "ghost" not reproducible at inference.

**2. Attention can extract structure, not idiosyncrasy.** The W_k/W_v matrices have D² parameters; they can only extract patterns that exist *across the K population*. For idiosyncratic per-K cap content, there's nothing to extract — attention correctly (and successfully) learned to ignore the slot.

**3. Compounding redundancy at this scale.** At d=16 on a 1MB char corpus, almost no context exceeds 16 chars. The model's native attention already sees the full prefix. `h_in` largely encodes information the model already has access to in-window. Cap-recurrence is meant to extend effective context *beyond* d; at d=16 on this corpus there is nothing past d to extend. The mechanism is being tested in a regime where it has nothing to offer.

The design doc's Q2-D (persona clustering) was an attempt to address (1) by surfacing only the structural component of `h_cap`. We did not test it — Q2-A (mass-weighted averaging) was the cheaper first cut. Q2-D might extract more, but (3) — in-window redundancy at this scale — would still apply.

## Implementation status

All on branch `agpt-cap-recurrence`. Unmerged. Builds clean (`just build-agpt-train`).

Phase 0-2A (capture + h_in compute + predecessor table): committed `4b00449`..`607d62a`.
Phase 2B step 1 (direct add): committed `9fb9ae2`.
Phase 2B step 2 (learnable additive) and step 3 (option B / K-V token): **uncommitted** as of this writeup. The implementation is correct (forward parity, gradient flow, W moves) but the mechanism produces no measurable benefit — the disciplined call is to keep it on the branch as a record but not merge to main.

Reproducibility:
```sh
# Train kvt
AGPT_CAPTURE_H_CAPS=1 \
AGPT_CAPTURE_H_CAPS_IN=rnd/cap-recurrence/20260527-smoke/h_caps.bin \
AGPT_CAPTURE_H_CAPS_OUT=/tmp/kvt.hcaps.bin \
AGPT_CAPTURE_H_CAPS_PRED_TABLE=/tmp/shake_d16_predecessors.bin \
AGPT_CAP_KV_INJECT=1 AGPT_CAP_KV_INJECT_LR=1e-5 \
AGPT_CAP_KV_INJECT_OUT=/tmp/kvt.W.bin \
bin/agpt_train --model data/input.model --trie-dir /tmp/shake_d16_radix \
  --save /tmp/kvt.model --epochs 5 --lr 3e-3 --lr-schedule warmup-cosine \
  --optimizer rmsprop --rmsprop-beta 0.999 --partition-depth 1 --no-accumulate \
  --mass-weight log --entropy-lambda 1.0
```

## What would change the picture

A genuine test of cap-recurrence requires a regime where it has something to add:

1. **`d` much smaller than the corpus's actual dependency range.** At Shakespeare with d=4 or d=8 (so the model can't see whole words), `h_in` from longer-range context might actually carry information the model lacks. (The corpus is small, so the effect would still be modest.)
2. **A corpus with strong long-range structure** — narrative coherence, episode-level patterns — where 16+ char context is materially incomplete. Even then, point (1) of the theoretical reading still applies: `h_in` would need to carry generalizable structure, not idiosyncratic memorized state.
3. **Q2-D persona clustering** instead of Q2-A averaging — explicitly extracting only the structural component of `h_cap` before injection. Addresses theory point (1) but not point (3).

None of these is a small experiment, and the prior from this work is that the basic mechanism (loop hidden state back, train on it) has a deeper false-narrative problem that simpler aggregation tricks won't solve. The persona path is the most principled — it's effectively a learned codebook over discourse states. If anyone returns to cap-recurrence, start there, not here.
