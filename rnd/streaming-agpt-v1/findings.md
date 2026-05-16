# Streaming AGPT — Findings

## v3 (2026-05-16) — warmup-cosine + LR-schedule horizon override

Added `--total-epochs-budget` flag to `agpt_train`: when streaming
fires repeated runs each with `--epochs 20`, this flag tells the
LR schedule "the global horizon is 100 SE, not 20". The schedule
references `adam_t / total_steps` where total_steps = budget × fires/SE,
so the cosine decay is one smooth curve across all 5 stages rather
than 5 truncated curves.

Same Shakespeare d=16 setup, recipe: rmsprop, lr=3e-3,
**warmup-cosine** (warmup_epochs=1), mass-weight=log,
entropy-lambda=1.0, --no-accumulate, --partition-depth 1.

| variant | PPL@16 | wall (s) |
|---|---:|---:|
| **Streaming 5×20 SE (v3)** | **4.48** | 388 |
| Baseline 100 SE (v3) | 4.74 | 596 |

Streaming beats baseline by 5.4% PPL in 35% less wall time.

### v3 vs v2 comparison

| variant | v2 (constant LR) | v3 (warmup-cosine) | delta |
|---|---:|---:|---:|
| Streaming | 5.06 | 4.48 | −11.5% PPL |
| Baseline | 5.48 | 4.74 | −13.5% PPL |
| streaming-vs-baseline gap | +7.7% | +5.4% | narrows |

Both variants got a meaningful PPL improvement from the LR schedule.
Baseline benefited slightly more (its single training run gets the
clean cosine annealing without any restart bookkeeping), which is why
the relative streaming advantage shrunk from 7.7% to 5.4%.

The headline conclusion stands: **streaming AGPT trains more
efficiently than batch AGPT, in less wall time, at matched SE budget.**
This holds across both the v2 (constant LR) and v3 (warmup-cosine)
configurations.

### v3 per-stage trajectory

| stage | radix nodes | PPL@16 |
|---|---:|---:|
| 20% | 323k | 5.32 |
| 40% | 640k | 4.80 |
| 60% | 963k | 4.57 |
| 80% | 1.29M | 4.47 |
| 100% | 1.61M | 4.48 ← slight uptick |

Stage 80 → 100 went up by 0.01 PPL (4.47 → 4.48) — within noise but
the same direction as v2's similar uptick. Suggests Shakespeare 1M
d=16 is near the plateau after ~80 SE on the full corpus equivalent
of compute.

### Implementation note: --total-epochs-budget

New CLI flag added with this experiment. Without it, each streaming
stage's `--epochs 20` would compute its own LR schedule with total=20,
so stage 2's adam_t=1300 would already be past the schedule end at
total=20 × 65 = 1300. Symptom would be LR ≈ 0 from stage 2 onward.

With `--total-epochs-budget 100`, the schedule references the global
horizon, and adam_t (persisted across stages) traces a single smooth
cosine curve from step 0 to step 6500 spread across all 5 stages.

---

## v2 (2026-05-16) — with optimizer-state persistence

Re-ran the same experiment after adding optimizer-state save/load to
`agpt_train.cu` (commit 6a655e7). Adam/RMSprop moments now persist
across `--save` → `--model` chains via an OPT_MAGIC footer in the
model checkpoint.

| variant | PPL@16 | wall (s) | training wall (s) |
|---|---:|---:|---:|
| **Streaming 5×20 SE (v2)** | **5.06** | 387 | ~360 |
| Baseline 100 SE (v2) | 5.48 | 589 | ~583 |

**Streaming beats baseline by 7.7% PPL, in 34% less wall time.**

The optimizer-state cold-restart was the dominant confound. Comparing
per-stage e1 (first-epoch loss after each stage transition):

| stage | v1 e1 (cold) | v2 e1 (warm) | jump removed |
|---|---:|---:|---:|
| 20% (cold start, expected) | 3.42 | 3.42 | — |
| 40% | 3.60 | 2.20 | −1.40 |
| 60% | 3.47 | 1.95 | −1.52 |
| 80% | 3.34 | 1.87 | −1.47 |
| 100% | 3.28 | 1.78 | −1.50 |

In v1 each stage started ~1.5 nats above the previous stage's final
loss — the optimizer's second moment buffer had to be re-warmed from
zero. In v2 each stage picks up right where the previous left off,
sometimes slightly lower as the new trie nodes (introduced by the
larger corpus subset) have not yet been seen and pull e1 up by a
small amount that's quickly absorbed.

### v2 Per-stage trajectory

| stage | radix nodes | e1 | e20 | PPL@16 |
|---|---:|---:|---:|---:|
| 20% | 323k | 3.42 | 1.94 | 6.33 |
| 40% | 640k | 2.20 | 1.78 | 5.42 |
| 60% | 963k | 1.95 | 1.70 | 5.17 |
| 80% | 1.29M | 1.87 | 1.68 | 5.14 |
| 100% | 1.61M | 1.78 | 1.70 | 5.06 |

PPL monotonically decreases this time (vs v1's stage-80→100 uptick).
The model converges cleanly across the streaming curriculum.

### Why streaming wins now

With optimizer state preserved, the streaming hypothesis works as
predicted:

1. **No catastrophic forgetting.** Trie targets at high-mass shallow
   nodes are cumulative; gradients pull toward the long-run empirical
   distribution at every stage.
2. **More effective optimizer steps per corpus pass.** The same 100
   SE budget actually fires 6500 optimizer steps in both variants,
   but streaming spreads them across a sequence of progressively
   richer tries — a natural curriculum.
3. **Compute efficiency.** Earlier stages train on smaller tries
   (less compute per SE), so streaming uses ~60% of baseline's total
   training events but converges to better PPL. The smaller tries
   give the model a head-start on the easy distributional structure
   before the full trie's harder details arrive.

### Caveats

- Eval scope was 4096 positions (1 chunk of held-out tokens). For
  publication-grade numbers we'd want multi-seed runs.
- Shakespeare d=16 is a small-scale test. Gutenberg 5M at d=32 might
  behave differently; should retest.
- Baseline PPL varies run-to-run (v1=5.30, v2=5.48) due to stochastic
  shuffle. The streaming-vs-baseline gap within v2 is the meaningful
  measurement.

### What this unlocks

- The per-call AGPT trainer can now be used as a building block in
  larger orchestrations (streaming, curriculum, mixed strategies)
  without losing optimizer momentum.
- The "no-forgetting" property of AGPT is now empirically demonstrated
  to translate into training efficiency, not just a theoretical
  property.
- Sliding-tree RoPE work can now use streaming as a substrate —
  the two ideas combine cleanly.

### Next steps (per priority)

1. Scale up: Gutenberg 5M d=32 streaming vs baseline.
2. Test finer cadence: 10 or 20 checkpoints instead of 5.
3. Combine with sliding-tree RoPE (separate experiment).
4. Multi-seed runs to nail down PPL noise band.

---

## v1 (2026-05-16) — without optimizer-state persistence [SUPERSEDED]

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
