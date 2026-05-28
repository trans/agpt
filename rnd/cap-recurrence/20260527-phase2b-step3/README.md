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

## Bug check — is the signal even getting through?

The ablation showed `Δ(eval-with-KV − eval-without-KV) ≈ 0`, which is consistent with either "model successfully ignored the slot" or "wiring bug stopped the signal from reaching the model." Distinguishing these matters for the writeup.

**Probe:** load a synthetic sidecar with `W_k`, `W_v` set to a known scale, run forward-only (`lr=1e-12`), measure loss. If forward is invariant under W changes → wiring bug. If forward responds → wiring works and the eval result is honest.

Result (forward-only loss on `data/input.model` seed, 1 epoch, lr=1e-12):

| sidecar | ‖W‖_F | K_inject[0] norm (dumped) | loss |
|---|---|---|---|
| no KV (baseline, slot absent) | — | — | 3.464 |
| W = 0 (slot present, content zero) | 0 | 0.00 | 3.448 |
| W = random × 0.1 | 6 | 0.5 | 3.446 |
| W = random × 1.0 | 64 | 4.7–7.0 | 3.448 |
| W = random × 10.0 | 640 | 47–70 | **3.920** |

With `‖W‖=640`, K_inject magnitude exceeds real K's (~12), the INJ slot dominates softmax, V_inject swamps the attention output, and loss explodes by +0.47. **The signal is reaching the model.** With `‖W‖=64`, K_inject is comparable to real K's (~5 vs ~12), INJ gets ~3% softmax weight, V contribution ~2% of output — that perturbation is real but lost in 1-ep run-to-run noise (~0.012). At the trained `‖W‖≈0.07`, K_inject magnitude is ~0.005, INJ slot's softmax weight is essentially zero, and the slot contributes nothing measurable to the output.

**So the implementation is correct.** And then *why does W stay at ~0.07 instead of growing?* Because the gradient through W requires a consistent loss-reducing direction, and the loss surface is flat in W's directions for this h_in content. The model isn't *deciding* to keep W small — the optimizer just has no signal to grow it. That IS the operational definition of "redundant": no improvement available from extracting anything from `h_in`, so the gradient pressure is ≈ 0, so W random-walks at small magnitude. The eval ablation's `Δ ≈ 0` is then a direct consequence: at that tiny `‖W‖`, the slot's softmax weight is negligible regardless of content.

## Three-state logical analysis (Thomas's framing)

For any candidate signal in `h_in`:

| If h_in carries… | Then training should produce… | Observed |
|---|---|---|
| Generalizable useful signal | A < B by some real amount | A ≈ B (Δ=+0.002) — falsified |
| False narrative the model latches onto | B > C (KV-trained without KV underperforms baseline) | B ≤ C — falsified |
| Redundant or noise | A ≈ B, kvt ≈ baset in interleaved test | exactly what we see |

Either way, the mechanism doesn't matter at this scale.

## What `h_in[K]` actually is — and why it can't carry usable signal

For a radix node K and each corpus position s where K appears, the predecessor
table records `K_prev` = the trie node matching `corpus[s−d .. s−1]` (the
d chars *immediately before* K starts). `h_cap[K_prev]` is the model's
`d_final_out` (post-final-LN, D=64) at K_prev's endpoint — the model's
about-to-predict state at corpus position `s−1` having seen those d chars.

Then:

```
                     Σᵢ count_i · h_cap[K_prev_i]
h_in[K]  =        ─────────────────────────────────
                          Σᵢ count_i
```

aggregated over the distinct K_prev's that occur immediately before K's
corpus appearances.

So h_in *does* carry out-of-window content — the d chars BEFORE K start
are not in K's own attention window. (An earlier draft of this README
called the failure "in-window redundancy" — that was wrong; correcting.)
It's not that h_in is redundant with what attention already sees; it's
that the form of h_in we deliver cannot be used.

### Why this aggregation can't carry usable signal — three compounding reasons

**1. Aggregation is forced by the radix factorization itself.** The whole
point of AGPT is "process each unique trie node K *once per fire*, not
once per corpus occurrence" — so for K's training step we need one
`h_in[K]` value even though K has many distinct K_prev's across the
corpus. Aggregation is the only way to compress a multi-valued predecessor
signal into one per-K input. The Q2-C alternative (per-(K, K_prev) pair
training) would un-factorize the trie back to corpus-length, destroying
the framework's efficiency. So **averaging is the cost of keeping the
factorization** — not a design choice we could swap for free.

**2. Mass-stratified mismatch — the regime with information has no
gradient, the regime with gradient has no information.** Predecessor
count for K scales with K's corpus mass, not depth per se:

| K's mass | K_prev variety | h_in[K] character | Gradient mass during training |
|---|---|---|---|
| 1 (deep, rare) | one predecessor | specific, real, useful | tiny (K fires once per epoch) |
| many (shallow, common) | many predecessors | heavily averaged → smeared | strong (K fires many times) |

For deep mass-1 K, h_in is genuinely a specific predecessor's cap (no
real averaging), but the model barely sees K — gradient through W_k/W_v
from this K is negligible. For shallow high-mass K, the model sees the K
many times (strong gradient), but h_in is averaged across thousands of
diverse contexts into something close to a corpus-wide centroid — no
discriminative per-K signal left to extract. **Neither regime gives the
model what it would need to learn an extraction.**

**3. Training→inference distribution mismatch.** Even if (2) were
somehow overcome, the training-time h_in is an average of many
predecessor caps. At inference, the predecessor is one specific concrete
prefix. The model can't be trained on averaged ghost predecessors and
expected to use single concrete ones at generation.

The design doc's Q2-D (persona clustering — codebook over discourse
states) attempts to address (1) by representing K's predecessor
distribution as a mixture over a small global vocabulary of "discourse
states." That would preserve more of the cross-K structural signal than
centroid-averaging. But (2) and (3) still apply: low-mass K still gives
weak gradient, and inference still uses one specific predecessor not a
mixture. Q2-D is a partial mitigation, not a fix.

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

## Cross-check at 10-ep with higher lr — real h_in at oracle-comparable W magnitudes

Once oracle showed lr=1e-3 reaches ‖W_v‖≈15 and produces a real −0.019
benefit at 5ep, the natural follow-up was: does real (mass-weighted)
h_in show *any* benefit if we give W the same room to grow? Maybe the
prior lr=1e-5 nulls were just "W never had a chance."

Interleaved 3-triple 10-ep (baseline / kv-mass lr=1e-4 / kv-mass lr=1e-3),
plus 3 single shots of kv-none at 10ep:

| Condition | mean | spread | Δ vs baseline | ‖W_v‖ |
|---|---|---|---|---|
| baseline | 1.8156 | 0.028 | — | — |
| kv-none | 1.8182 | 0.044 | +0.003 | 7e-9 (frozen) |
| kv-mass lr=1e-4 | 1.8433 | 0.030 | **+0.028** | ~1.0 |
| kv-mass lr=1e-3 | 1.8132 | 0.014 | −0.002 | ~8.4 |

Three sharp observations:

1. **kv-none ≈ baseline at 10ep** (Δ=+0.003, within noise). The earlier
   "softmax-stealing +0.008" at 5ep was sampling noise — it disappears
   with more training. Slot-presence alone genuinely contributes
   nothing.
2. **kv-mass at lr=1e-4 (W~1) is meaningfully WORSE** (+0.028, outside
   noise). At moderate W the model can't ignore the inject slot, but
   its content carries no predictive info — so the perturbation
   actively hurts.
3. **kv-mass at lr=1e-3 (W~8) returns to baseline** (Δ ≈ 0). At large
   W, the W/V matrices have trained into a configuration that
   effectively routes around the noisy content. Model adapts to the
   meaningless perturbation without extracting benefit.

Compare directly to oracle at the same lr regime:

| | content | lr | ‖W_v‖ | Δ vs baseline |
|---|---|---|---|---|
| kv-mass | real (noise) | 1e-3 | ~8 | ≈ 0 |
| kv-oracle | predictive | 1e-3 | ~15 | **−0.019** |
| kv-oracle | predictive | 1e-2 | ~100 | **−0.096** |

Same architecture, same wiring, same lr scale, similar W magnitudes —
oracle content drives loss down; real-h_in content doesn't. The
mechanism distinguishes signal from noise correctly. The non-monotonic
loss-vs-‖W‖ pattern for real h_in (tiny=neutral, moderate=hurts,
large=adapts) is the exact signature of "noise injection that the
model neither benefits from nor can entirely ignore."

**Final conclusion locked: there is no extractable predictive signal
in mass-weighted aggregated predecessor caps for this corpus / model
/ d combination, regardless of W magnitude or epoch count.**

## Definitive bug-vs-design test: ORACLE h_in (does the path work AT ALL?)

After kv-mass, kv-inv, kv-random, kv-constant, and kv-none all gave the
same flat result, the natural worry was: maybe the signal isn't reaching
the training gradient at all (i.e., a bug we haven't found). Testing
this directly:

**`AGPT_CAP_H_IN_WEIGHT=oracle`** fills `h_in[K]` with a deterministic
hash-derived vector seeded by K's *modal next-token* (the most common
token following K, looked up from the trie's per-node count table).
So h_in literally encodes the answer: same modal token → identical
h_in; different modal token → different h_in. The model has a perfect
predictive shortcut available *if* it can use h_in's content at all.

Sweeping the W_k/W_v learning rate (5-ep × 3 seeds per lr):

| condition | lr | n | mean loss | Δ vs baseline | SE-dist | ‖W_v‖ |
|---|---|---|---|---|---|---|
| baseline | — | 5 | 1.906 | — | — | — |
| kv-mass | 1e-5 | 5 | 1.914 | +0.008 | <1 | 0.07 |
| kv-oracle | 1e-5 | 5 | 1.901 | −0.005 | <1 | 0.12 |
| kv-oracle | 1e-4 | 3 | 1.903 | −0.003 | <1 | 1.4 |
| kv-oracle | 1e-3 | 3 | 1.887 | −0.018 | ~1.4 | 15 |
| **kv-oracle** | **1e-2** | 3 | **1.810** | **−0.096** | **~5-6** | 103 |

Honest read: only the lr=1e-2 result is statistically significant on
its own (≈5-6 SE from zero, n=3). lr=1e-3 is suggestive but underpowered
(~1.4 SE, p ≈ 0.2). The lower lrs are within noise.

What gives me *some* additional confidence beyond the lr=1e-2 point: the
four oracle Δs are monotone in lr with no sign flips. P(monotone | random) =
1/24 ≈ 4%, so the trend itself carries weight — but that's weaker than
"each point is individually real."

**The lr=1e-2 result alone is enough** to lock the wiring conclusion: loss
drops by 5% (~10× the within-condition spread), ‖W_v‖ reaches 100+ as
the model learns to route attention heavily through the INJ slot. Compare to `kv-mass` at lr=1e-2 from the
step-1 era, which DEGRADED loss to 2.654 (W exploded but in a
non-useful direction because real h_in is noise). Same wiring, same
lr, same architecture; the *only* difference is whether h_in's content
is real (noise) or oracle (signal).

This **rules out any wiring or signal-flow bug**. The path —
h_in → K_inject/V_inject → attention output → loss → backward → dW —
is fully functional. The optimizer finds useful directions when they
exist. The model uses h_in's content when that content is predictive.

So **every prior null result was an honest measurement.** kv-mass /
kv-inv / kv-random / kv-constant / kv-none all gave flat results not
because the mechanism is broken but because **real aggregated
predecessor caps carry no extractable predictive signal about K's
continuations**, while a synthetic oracle on the same path does. The
"redundant" framing has direct empirical support: real h_in is
operationally noise from the model's perspective; oracle h_in is
operationally signal; the mechanism distinguishes correctly.

What this *sharpens* about the failure: it's not the aggregation
function (kv-inv/uniform/random all null), not the gradient path (oracle
exploits it), not lr (oracle works at lr=1e-3 onward). It's specifically
that **the h_in we deliver — averages or per-K random or constant —
contains no information correlated with K's next-token distribution
beyond what the model already extracts from K's own prefix**. Q2-D
persona clustering or any other aggregation reform only helps if the
result is more oracle-like than the current centroid-average.

## Decomposition: is the +0.008 of kv-mass content or slot-presence? (Tested: slot-presence.)

The interleaved A/B (next section) showed `kv-mass` landing ~+0.008 above
baseline. To isolate whether that came from the learned `K_inject`/`V_inject`
content vs the slot merely existing in attention (softmax-stealing
perturbation), we added a `kv-none` condition: `AGPT_CAP_KV_INJECT=1`
with `AGPT_CAP_KV_INJECT_LR=1e-12` so W stays frozen at zero. The slot
participates in softmax but its K and V are zero — content adds nothing
to the attention output, only the slot's presence does (via the
`exp(0)=1` term in the softmax denominator).

5-ep × 5 single shots (same obsolete heuristic for cross-batch
comparability with the other conditions):

| condition | mean (5 runs) | Δ vs baseline | what it tests |
|---|---|---|---|
| baseline | 1.906 | — | no inject slot at all |
| kv-none | **1.914** | **+0.008** | slot exists, K=V=0 (softmax-stealing only) |
| kv-mass | 1.914 | +0.008 | slot + mass-weighted h_in content, lr=1e-5 |
| kv-inv | 1.904 | −0.003 | slot + inverse-weighted h_in content, lr=1e-5 |

W norms in kv-none stayed at ~6e-9 (truly frozen at zero, as intended).

**kv-none ≡ kv-mass.** The +0.008 attributed to KV-inject is *entirely*
the slot-presence perturbation; the learned W_k/W_v contribute zero
measurable signal on top. And the slot-presence effect itself sits at
one SE of zero — small and likely noise at n=5. kv-inv differs from
kv-none by −0.010 which is well inside the within-pair spread of ±0.05.

Three nested nulls — slot vs no-slot, content-vs-empty-slot,
aggregation-function-vs-aggregation-function — each independently
confirms that there is no extractable predictive signal in aggregated
predecessor caps in this setup.

## Cross-check: does the aggregation *function* matter? (Tested: no.)

A natural question is whether the null is specific to mass-weighting, or
intrinsic to aggregation. We added `AGPT_CAP_H_IN_WEIGHT={mass|uniform|inverse}`
(env-var-gated, default `mass` preserves prior behavior) — `inverse` weights
predecessors by `1/count_j`, the TF-IDF / importance-sampling-toward-uniform
choice that should be the most discriminative if any aggregation is going to work.

Interleaved 5-pair 5-ep A/B (baseline / KV-inject mass / KV-inject inverse):

```
pair    baseline   kv-mass    kv-inv    Δ(mass)   Δ(inv)
 1      1.918      1.919      1.918     +0.001    -0.000
 2      1.907      1.907      1.898     +0.000    -0.009
 3      1.906      1.912      1.925     +0.006    +0.019
 4      1.872      1.927      1.910     +0.055    +0.038
 5      1.928      1.906      1.870     -0.022    -0.058
                                        ──────    ──────
mean    1.906      1.914      1.904     +0.008    -0.003
```

‖W_k‖ across inverse runs: 0.05–0.09 (same magnitude band as mass-weighted).

**Inverse weighting moves nothing of substance.** Mean Δ vs baseline of −0.003
sits well inside the ±0.05 per-pair spread, exactly like mass-weighted's +0.008.
Swapping the aggregation function for the most principled alternative does not
change the null. This confirms the diagnosis: the problem is **aggregation
itself**, combined with the mass-stratified gradient mismatch and the
train→inference distribution shift — not the choice of which weighting rule
defines the aggregate. Any single-value-per-K aggregation has the same fate.

## What would change the picture

Given the three-way diagnosis (forced aggregation + mass-stratified mismatch + train-inference distribution shift), any "fix" has to address all three at once. None of the following alone is sufficient:

1. **Persona clustering (Q2-D) instead of mass-weighted averaging.** Preserves more structural signal per K than centroid-averaging. Helps with point (1) but doesn't change the mass-stratified mismatch (point 2) or the inference distribution gap (point 3).
2. **Larger d / longer-range corpus.** Gives the recurrence more out-of-window context to carry. But the aggregation problem is independent of d — averaging over many predecessors smears regardless.
3. **Per-(K, K_prev) treatment (Q2-C).** Solves the aggregation problem (each predecessor instance gets its own training signal). But un-factorizes the radix and gives back O(corpus_length) training cost — defeating AGPT's reason to exist.
4. **A genuinely structural state instead of raw h_cap.** Replace per-trie-node EMA caps with something explicitly designed to be aggregation-stable (e.g., a small clustered discourse-state embedding chosen by routing K's caps into a learned codebook). Combined with (3)-like per-instance routing at inference, this might thread the needle — but it's a substantial new design, not an extension of what's here.

The honest read of this work: cap-recurrence in the form we tested (and likely in any form that preserves the radix factorization) is fundamentally limited by the aggregation/factorization tension that step-2 and step-3 surface. Anyone returning to this should think carefully about whether they're escaping that tension, not just changing the injection mechanism.
