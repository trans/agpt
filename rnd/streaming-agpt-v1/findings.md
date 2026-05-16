# Streaming AGPT v1 — Findings

**Date:** 2026-05-16
**Tool:** `bash rnd/streaming-agpt-v1/run.sh`
**Corpus:** Shakespeare 1M, d=16, pd=1, rmsprop, constant LR=3e-3,
mass-weight=log, entropy-lambda=1.0

## Headline

| variant | total wall | train wall | PPL@16 |
|---|---:|---:|---:|
| Streaming 5×20 SE | 380s | 355s | **5.88** |
| Baseline 100 SE | 595s | 588s | **5.30** |

**Streaming underperforms baseline by 10.9% PPL at matched SE budget.**

## Per-stage trajectory

| stage | radix nodes | loss e1 | loss e20 | PPL@16 |
|---|---:|---:|---:|---:|
| 20% | 323k | 3.42 | 2.01 | 6.66 |
| 40% | 640k | 3.60 | 1.92 | 6.31 |
| 60% | 963k | 3.47 | 1.87 | 6.13 |
| 80% | 1.29M | 3.34 | 1.83 | 5.85 |
| 100% | 1.61M | 3.28 | 1.81 | 5.88 |

Stage 80%'s PPL (5.85) slightly *beat* stage 100% (5.88) — adding the
final 20% of corpus and 20 SE of training mildly regressed. Within
noise band but suggestive.

## Confound 1: Optimizer cold restart

Each stage starts with a fresh RMSprop second-moment buffer (the
`save_model_weights` function doesn't persist optimizer state). The
e1 loss of each stage jumps **dramatically** up from where the
previous stage ended:

```
End of stage 20%:   loss = 2.01
Start of stage 40%: loss = 3.60   ← +1.6 nats regression
```

Pattern repeats at every transition. RMSprop with β₂=0.999 needs
~1000 steps to warm its second moment; AGPT at pd=1 fires 65
steps/SE × 20 SE = 1300 steps per stage. So each stage barely
finishes warming before it's over.

Net effect: maybe 50-70% of each stage's training is *recovering* to
where the previous stage left off, not making new progress.

## Confound 2: SE budget split

Streaming spent 100 SE across stages, but only 20 SE on the final
(full-corpus) trie. PPL is evaluated against the full corpus, so the
gradient signal from the partial-corpus stages doesn't fully transfer.

Compared on a per-event basis:

- Baseline: 100 SE × 65 fires/SE × full-trie-events = 100% events
- Streaming: 20 × 65 × (0.2 + 0.4 + 0.6 + 0.8 + 1.0)-frac-events
  ≈ 60% of baseline's total events

Streaming actually used ~60% of baseline's compute (training events).
That it got within 10.9% of baseline's PPL on 60% of the compute is
arguably efficient — but it's not the "free lunch" the hypothesis
predicted.

## Matched-wall-time comparison

| metric | streaming | baseline at matched train-wall |
|---|---:|---:|
| Wall time (training) | 355s | ~353s (≈ baseline e60) |
| Training loss | 1.81 | ~1.83 |

At equal training wall time, streaming and baseline reach similar
training loss (within ~1%). The headline PPL gap comes from baseline
*continuing past* this point — it kept training another 235s on the
full trie and reached PPL 5.30.

## What this rules out and what it doesn't

**Rules out:** "streaming as designed beats baseline at matched SE."
At matched SE, streaming loses by ~11%.

**Does NOT rule out:** the core hypothesis that AGPT's no-forgetting
property enables efficient streaming training. The two confounds
(optimizer cold restart, partial-trie compute share) prevented a
clean test.

To test the hypothesis cleanly, we'd want:

1. **Optimizer state persistence** — fix `save_model_weights` to also
   save Adam/RMSprop moments. This is a moderate code change but
   would eliminate the e1 loss jumps.

2. **Comparison framing.** With persistent optimizer state, the
   "fair" comparison becomes:
   - Streaming: build trie incrementally, train continuously
   - Baseline: same total SE but all on full trie
   The streaming hypothesis predicts streaming wins (or matches) at
   matched SE only if (a) the no-forgetting property is real, AND
   (b) curriculum-of-growing-tries is informative-not-noise.

3. **Alternative budget allocation.** Spend less time on early
   (smaller-trie) stages, more on late stages. E.g., 5/10/20/25/40 SE
   instead of 20/20/20/20/20. This biases compute toward the full
   trie where it matters for eval.

## Recommended next steps

In priority order:

1. **Fix optimizer state save/load** (moderate work, ~half a day).
   This is the blocker for a clean streaming test. Once fixed, re-run.

2. **Re-run streaming with persistent optimizer state.** Expected
   outcome: e1 of each stage matches end of previous stage, no loss
   jumps. PPL should improve substantially over current 5.88.

3. **Investigate weighted-SE-budget variants.** Once (1) and (2) are
   done, try non-uniform SE allocation favoring later stages.

4. **Move to sliding-tree experiment** (separate hypothesis,
   orthogonal mechanism). The two ideas combine cleanly only after
   we understand each in isolation.

## Files

```
rnd/streaming-agpt-v1/
├── README.md                              — plan
├── findings.md                            — this file
├── run.sh                                 — orchestration
├── models/                                — gitignored
└── logs/
    ├── orchestration.log                  — summary output
    ├── baseline_build.log                 — full-trie build
    ├── baseline_train.log                 — 100 SE training
    ├── baseline_ppl.log                   — PPL 5.30
    ├── streaming_stage{20,40,60,80,100}_build.log
    ├── streaming_stage{20,40,60,80,100}_train.log
    └── streaming_stage{20,40,60,80,100}_ppl.log
```
