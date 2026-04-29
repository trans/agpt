# Trie-as-Attention Framing — Final Findings

> **Status (2026-04-28): closed.** Descriptive predictions confirmed;
> prescriptive operationalizations all neutral-to-marginally-negative
> under tested recipes. The strict architectural realization remains
> unbuilt.

## TL;DR

The radix trie's structural decomposition (decision zone d≤d* + identity
zone d>d* up to leaf) maps cleanly onto Q/K/V attention, with K=decision
and V=identity. We confirmed three predictions about trie statistics and
d-curve shape, but none of the gradient-routing or loss-filtering
operationalizations of the framing translated into a training-time
improvement over the existing AGPT baseline. The framing remains
descriptively useful and predictively sound; it didn't yield a
prescriptive training-engine win at any flag-level operationalization.

## What worked: descriptive predictions

### d* scales as log₂(N)/H across 50× corpus-size range

For each mass-1 prefix-tree leaf, computed d* = depth at which path becomes
unique (mass first drops to 1) walking root→leaf. Per-corpus distributions
are bell-shaped; means are tabulated below.

| Corpus | log₂(N)/2 predicted | Observed mean d* | Δ |
|---|---:|---:|---:|
| Shakespeare 100k | 8.31 | 7.94 | −0.37 |
| Shakespeare 1M   | 10.04 | 9.71 | −0.33 |
| Gutenberg 5M     | 11.15 | 11.23 | +0.08 |

Slope between Shakespeare 100k → 1M: observed 0.53 char-per-doubling, vs
predicted 0.50. The two Shakespeare points sit ~0.35 below the H=2
prediction line, consistent with Shakespeare's per-char entropy ≈ 2.07
(dialogue formatting, character-name repetition). Gutenberg falls right on
the H=2 line.

### d=32 is the sweet spot for English at 1-5M corpus size

AGPT d-sweep on Shakespeare 1M (single run per d, recipe: rmsprop wc
lr=3e-3 entropy-λ=1.0 mass-weight=linear no-accumulate 3SE):

| d | PPL | Δ vs d=32 |
|---|---:|---:|
| 16 | 15.37 | +2.38 |
| 32 | 12.99 | — |
| 48 | 12.94 | −0.05 |

Three predictions, three matches:
1. **d=16 deficient**: identity zone only 6 chars (16 − d* of 9.71),
   insufficient → big PPL hit (+18% relative).
2. **d=32 near-optimal**: matches predicted d_optimal of ~31
   (= log₂(1M)/H + 21).
3. **d=48 saturated**: identity zone already filled at d=32; extra trie
   depth doesn't help → 0.05 PPL is noise.

Capacity-only explanations don't predict the d=32 → d=48 flatness.

### Decision events carry ~97% of the learning at 8% of the events

Decision-only ablation (zero CE loss + gradient at queries past d_split):

| buf | events kept | PPL | nats (CE) | Δ vs baseline |
|---|---:|---:|---:|---:|
| baseline | 100% | 13.25 | 2.584 | — |
| buf=15 | 69.4% | 13.33 | 2.590 | +0.006 |
| buf=5 (3 runs) | 28.8% | 13.54 | 2.605 | +0.021 |
| buf=2 | 16.5% | 14.12 | 2.647 | +0.063 |
| buf=0 (strict) | 8.2% | 13.93 | 2.634 | +0.050 |

Reference: untrained char-LM has CE ≈ ln(65) ≈ 4.17 nats. Baseline learns
1.586 nats relative to random; decision-only at 8.2% events still learns
1.536 nats — **96.8% of baseline's total learning, achieved with 12× less
data**. PPL is exp(CE), so small PPL differences correspond to tiny nats
differences; the apparent "5% PPL hit" is really +0.05 nats out of 1.59
total.

This is the framing's strongest empirical support: the decisions *do*
carry the bulk of the learning signal. The 92% of events we dropped
provide the last 3% of marginal value, not the bulk.

## What didn't work: prescriptive operationalizations

### Static depth-routing (AGPT_DEPTH_ROUTE_K)

Hard binary mask at integer threshold k. Queries at depth ≤ k feed Wk-grad
only; queries at depth > k feed Wv-grad only.

| k | runs | PPL mean (range) |
|---|---:|---:|
| baseline | 3 | 13.57 (0.26) |
| k=5 | 1 | 14.07 |
| k=7 | 3 | 13.71 (1.08 — first run 13.01 was lucky outlier) |
| k=9 | 1 | 13.86 |
| k=11 (framing-predicted) | 3 | 13.78 (0.25) |
| k=20 | 1 | 15.12 |

k=11 within noise, all others worse. The k=7 first-run win didn't
replicate; range across replications was 4× wider than baseline,
suggesting the routing inflates variance.

### Per-leaf d* routing (AGPT_DEPTH_ROUTE_PERLEAF=1)

Replaced static k with per-radix-node d_split (depth at which the node's
path first becomes mass=1). Multi-mass intermediate nodes get
d_split=INT_MAX (all queries route to Wk).

Combined across two sessions:

| Setting | runs | mean PPL | std |
|---|---:|---:|---:|
| baseline | 7 | 13.39 | 0.27 |
| per-leaf d* | 6 | 13.10 | 0.25 |
| Gap | | 0.29 PPL (2.1%) | t≈2.0, p≈0.07 |

Within-session-only: gap drops to 0.16 PPL (1.2%), p≈0.37 — not
significant. Initial 3-run "strict ordering" of per-leaf < baseline was a
small-sample artifact; new baseline runs (12.95, 13.11) are *better* than
several per-leaf runs.

Honest read: per-leaf d* is neutral-to-marginally-positive. Better than
static k=11 (neutral-to-marginally-negative) but not a clean win.

### Decision-only loss (AGPT_DECISION_ONLY=1)

See diminishing-returns table above. None of the buffer configs beat
baseline at 3 epochs. Matched-compute test (9 epochs decision-only vs
9 epochs baseline) confounded by recipe overfitting at extended epochs:
both settings get *worse* at 9 epochs with the warmup-cosine schedule
because the LR trajectory is fundamentally different at different epoch
counts.

| Setting | epochs | PPL |
|---|---:|---:|
| baseline 3ep | 3 | 13.25 |
| baseline 9ep | 9 | 16.92 (overfit) |
| decision-only 9ep (3 runs) | 9 | 14.75-17.71, mean 15.82 |
| decision-only 27ep | 27 | 20.04 |

Decision-only 9ep beat baseline 9ep on the mean (15.82 vs 16.92), but
both setups overfit, so the matched-compute hypothesis isn't cleanly
testable with this recipe. Would need constant LR + weight decay to
isolate the question.

### Microgpt SGD parallel

Same routing flag (`--depth-route-k`) added to standard window-attention
SGD trainer.

| Setup | baseline CE (3 runs) | k=11 CE (3 runs) | gap |
|---|---:|---:|---:|
| AGPT d=32 | 13.57 PPL (range 0.26) | 13.78 PPL | +1.5% (within noise) |
| SGD seq_len=128 | 2.164 nats (range 0.047) | 2.275 nats | +11.6% PPL (clear) |
| SGD seq_len=32 | 2.282 nats (range 0.032) | 2.353 nats | +7.3% PPL (clear) |

Pattern: harm scales with context length. The longer the window, the
fewer events feed Wk relative to Wv, and the worse the routing
performs. Even at the framing's predicted regime (seq_len=32, k=11, the
11/21 split that matches AGPT d=32), SGD with routing is 7% worse.

## Diagnosis: why naive routing can't realize the framing

The framing says K should encode decisions, V should encode identity.
Standard CE-on-next-char training gives V near-zero loss on
deterministic-tail events (mass=1 unary chains where the next char is
fully determined). Routing those events' gradient to Wv doesn't teach Wv
to "name the leaf" — it just trains Wv to do trivial next-char
prediction at deep positions, which is a different job than the framing
prescribes.

For V to actually realize "identity = which leaf am I in," the *loss*
needs to change, not just the gradient flow. Three options remain
unexplored:

1. **Predict a leaf-ID directly** (representation learning): V's job is
   "output the unique fingerprint"; supervised against learned leaf
   embeddings.
2. **Predict the full 21-char tail in one shot**: a sequence-output head
   that takes V and emits the cap; sequence loss instead of per-position
   CE.
3. **V as corpus-lookup, not learned projection**: drop Wv entirely;
   V_b = embedding of the first char of the unary tail at b. Train
   Wq/Wk/Wo only. Decision-paced inference: at generation, after attention
   selects branch b*, emit corpus tail directly.

Option 3 is the strict architectural realization of the framing. It's
the actual unsettled question — none of the flag-level operationalizations
can substitute for it.

## Lessons

1. **PPL is exp(CE) — small PPL gaps mean tiny nats gaps.** A 5% PPL hit
   is +0.05 nats out of ~1.6 nats of total learning. The decision-only
   ablation looked discouraging in PPL terms (13.93 vs 13.25) but is
   actually striking in nats terms (96.8% of total learning at 8% of
   events). Always convert before drawing conclusions about training
   signal quality.

2. **Hard mask is too sharp; soft anneal might behave differently.** Every
   hard-mask operationalization either matched baseline (per-leaf,
   marginally) or lost. Soft weighting (sigmoid around the threshold)
   preserves all training events while still biasing the K/V projections
   toward their roles. Untested.

3. **Recipe-test interaction matters.** Three-epoch warmup-cosine tuned
   for the standard recipe doesn't extend cleanly to 9 or 27 epochs. The
   matched-compute experiment was confounded by this; needs
   constant-LR + weight-decay to isolate.

4. **Descriptive ≠ prescriptive.** The framing predicted three corpus
   statistics correctly (d* scaling, d=32 sweet spot, decision-event
   information density). It still didn't yield a training-time
   improvement. A correct *description* of how a system organizes itself
   doesn't automatically tell you how to *prescribe* its training.

## Artifacts

### Code (uncommitted at time of close)

In `src/cuda/agpt_train.cu`:
- `mask_grad_by_query_depth_kernel` + launcher (static depth routing)
- `mask_grad_by_query_dsplit_kernel` + launcher (per-leaf d* routing)
- `mask_loss_decision_only_kernel` + launcher (decision-only with buffer)
- `d_split` field in `RadixTrieData`, computed at end of `load_radix_trie`
- Per-query depth and d_split host arrays in chunk loop
- Env vars: `AGPT_DEPTH_ROUTE_K`, `AGPT_DEPTH_ROUTE_PERLEAF`,
  `AGPT_DECISION_ONLY`, `AGPT_DECISION_BUFFER`

In microgpt `src/microgpt/`:
- `MicroGPT.depth_route_k` class property (`backend.cr`)
- Per-row mask in `MultiHeadAttention.backward` (`micro_gpt.cr`)
- `--depth-route-k N` CLI flag (`main.cr`)

### Data

- `/home/trans/agpt-tries/shakespeare_d32_radix_corpus/` — built for d* analysis
- `/home/trans/agpt-tries/shakespeare_100k_d32_radix_corpus/` — built for d* analysis
- `/home/trans/agpt-tries/gutenberg_5m_d32_radix_corpus/` — pre-existing, d* analysis ran on it

### Logs

`/tmp/dr_*.log` files from the live session were lost on reboot. The
result tables in this document and the README are the surviving record.
