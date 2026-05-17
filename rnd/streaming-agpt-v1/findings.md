# Streaming AGPT — Findings

## Cadence sweep — 100×1, 250×2, 100×5 vs baseline (2026-05-17)

Three multi-seed cadence comparisons at varying per-stage budgets.

| variant | total SE | mean PPL ± std | wall (s) | p vs baseline |
|---|---:|---|---:|---:|
| Baseline 500 SE (3-seed) | 500 | 4.265 ± 0.018 | 2899 | — |
| Streaming 50 × 10 SE (3-seed) | 500 | 4.228 ± 0.095 | 1571 | p > 0.5 (n.s.) |
| **Streaming 100 × 5 SE (3-seed)** | 500 | **4.175 ± 0.025** | 1675 | **p < 0.01** |
| Streaming 250 × 2 SE (3-seed) | 500 | 4.149 ± 0.083 | 1763 | p ≈ 0.07 (marginal) |
| Streaming 100 × 1 SE (3-seed) | 100 | 4.558 ± 0.076 | 455 | — |

Key observations:

1. **100 × 5 SE is the headline winner.** Significantly better than
   baseline (p < 0.01) AND tight variance comparable to baseline.

2. **Mean PPL improves with finer cadence,** at constant 500 SE
   budget: 50×10 → 100×5 → 250×2 = 4.228 → 4.175 → 4.149. But
   variance varies non-monotonically.

3. **Per-stage budget hits a variance sweet spot at ~325 fires.**

| cadence | fires/stage | std |
|---|---:|---:|
| 50 × 10 | 650 | 0.095 (wide) |
| **100 × 5** | **325** | **0.025 (tight)** |
| 250 × 2 | 130 | 0.083 (wide) |
| 100 × 1 (different budget) | 65 | 0.076 (wide) |

Hypothesis: 325 fires/stage is enough for RMSprop second-moment
stabilization within a stage AND short enough that the global cosine
schedule dominates over per-stage trajectory drift. 50×10 has too
much within-stage drift (long enough for divergent training paths);
250×2 has too little within-stage settling. 100×5 is a balance.

4. **100×1 (100 SE budget) confirms streaming wins even at minimum
   per-stage budget.** PPL 4.558 ± 0.076 vs single-seed baseline 4.74
   = 3.8% improvement. Worse than 20×5's 4.33 at the same budget,
   but still beats baseline.

---

## 100 × 5 SE multi-seed — decisive streaming win

**Result (2026-05-17, 3 seeds with seeded inits):**

| variant | mean PPL ± std | wall (s) |
|---|---|---:|
| Baseline 500 SE (3 seeds) | 4.265 ± 0.018 | 2899 ± 17 |
| Streaming 50 × 10 SE (3 seeds) | 4.228 ± 0.095 | 1571 ± 7 |
| **Streaming 100 × 5 SE (3 seeds)** | **4.175 ± 0.025** | 1675 ± 25 |

**Welch's t-test (100×5 vs baseline): t ≈ 5.1, p < 0.01** — decisively
significant.

100×5 beats baseline by **2.1% PPL** in **42% less wall time**, with
**tight variance comparable to baseline** (0.025 vs 0.018).

100×5 also beats 50×10 by 1.3% PPL and has 4× lower variance, despite
having half the per-stage budget (325 fires/stage vs 650).

### What this overturns

- My earlier "needs ≥500 fires per stage" guess was wrong. 100×5 has
  325 fires/stage and works better than 50×10's 650.
- 50×2's failure (130 fires/stage) is a separate effect (very small
  per-stage budget specifically), not a smooth-threshold curve.
- The 50×10 multi-seed std of 0.095 may have been a 3-seed sampling
  fluke — 100×5 reaches 0.025 with the same 3 seeds.

### Streaming-vs-baseline status across all matched comparisons

| budget | baseline | best streaming | gap |
|---:|---:|---:|---:|
| 100 SE (single-seed) | 4.74 | 4.33 (20×5) | -8.6% |
| 500 SE (single-seed) | 4.26 | 3.996 (50×10) | -6.2% |
| 500 SE (3-seed) | 4.265 ± 0.018 | 4.175 ± 0.025 (100×5) | **-2.1%, p<0.01** |
| 500 SE (3-seed) | 4.265 ± 0.018 | 4.228 ± 0.095 (50×10) | -0.9%, n.s. |

The single-seed runs were lucky — true mean gap is ~2%, not the 6-9%
the lucky single-seeds suggested. But it's a *real* gap, and it's
statistically clean at the 100×5 cadence.

---

## Extended cadence — 200/500 SE budget (2026-05-16)

Question: how low can streaming push PPL with more compute at the same
fine cadence?

| variant | total SE | PPL@16 | wall (s) | improvement over baseline |
|---|---:|---:|---:|---:|
| 20 × 10 SE | 200 | 4.21 | 663 | −11.2% |
| **50 × 10 SE** | **500** | **3.996** | 1607 | **−15.7%** |

### Headline

**Streaming 50 × 10 SE breaks below PPL 4.0 on Shakespeare d=16** at
500 SE total budget — PPL 3.996 in 27 min vs **matched-compute 500
SE baseline at PPL 4.26 in 48 min** (added 2026-05-16 after request).

| total SE | baseline | streaming (best) | streaming advantage |
|---:|---:|---:|---:|
| 100 | 4.74 | 4.33 (20 × 5) | −8.6% PPL |
| 500 | **4.26** | **3.996** (50 × 10) | **−6.2% PPL, 45% less wall** |

At matched-compute 500 SE, streaming wins by 6.2% PPL in 45% less
wall time. Both higher budgets improve over their 100 SE counterparts;
streaming retains the advantage at every budget tested.

The relative streaming gap *narrows* with budget (8.6% → 6.2%) —
baseline benefits more per added SE because streaming was already
extracting more per unit. But streaming still wins absolutely.

### Compute-scaling trajectory

| total SE | best variant | PPL@16 |
|---:|---|---:|
| 100 | 20 × 5 SE | 4.33 |
| 200 | 20 × 10 SE | 4.21 |
| 500 | 50 × 10 SE | 4.00 |

Compute scaling on Shakespeare d=16:
- 100 → 200 SE (2× compute): PPL 4.33 → 4.21 (−2.8%)
- 200 → 500 SE (2.5× compute): PPL 4.21 → 4.00 (−5.0%)
- 100 → 500 SE (5× compute): PPL 4.33 → 4.00 (−7.6%)

Diminishing returns, but not flat — there's still room to push by
investing more compute. The model is approaching but not at its
Shakespeare d=16 ceiling.

### Cadence × budget interaction

The 50-stage cadence at 100 SE budget (50 × 2 SE) underperformed
baseline (PPL 4.80 vs 4.74). At 500 SE budget (50 × 10 SE) the same
50-stage cadence excels (PPL 4.00). The difference: per-stage budget.
- 50 × 2 SE = 130 optimizer steps per stage — below RMSprop's
  β₂=0.999 second-moment warmup horizon (~1000 steps)
- 50 × 10 SE = 650 optimizer steps per stage — sufficient

So the cadence ceiling at 100 SE total budget (around 20 stages) is
about *per-stage compute*, not about diminishing returns from finer
curriculum. At higher total budgets we can use finer cadence.

### Compute efficiency

Streaming gets ~2× lower wall-time-per-SE than baseline because early
stages train on smaller tries (less data per SE):

| variant | total SE | wall (s) | sec/SE |
|---|---:|---:|---:|
| Baseline 100 SE | 100 | 596 | 5.96 |
| Streaming 50 × 10 SE | 500 | 1607 | 3.21 |

This is "free" compute efficiency on top of the algorithmic win.

### What we lost (and recovered)

The 50 × 10 SE first attempt was cut off at stage 25 by a `/tmp` full
condition (RAM-backed tmpfs on CachyOS). After clearing /tmp,
re-ran cleanly. Captured the recovery procedure in
`feedback_bash_outage.md`.

---

---

## Cadence sweep (2026-05-16) — finer cadence helps, but only up to a point

Question: does using more, smaller stages (finer cadence) at constant
100-SE budget improve PPL further?

Sweep across N_STAGES ∈ {5, 10, 20, 50}, per_stage_SE = 100/N_STAGES.
Same recipe as v3 (warmup-cosine, mass-weight=log, pd=1).

| cadence | PPL@16 | wall (s) | vs baseline (4.74) |
|---|---:|---:|---:|
| 5 × 20 SE | 4.47 | 366 | −5.7% |
| 10 × 10 SE | 4.52 | 353 | −4.6% |
| **20 × 5 SE** | **4.33** | 361 | **−8.6%** ← best |
| 50 × 2 SE | 4.80 | 401 | +1.3% (worse) |

### Headline

**20 stages × 5 SE each** is the best cadence at this scale.
PPL 4.33 vs baseline 4.74 = **8.6% improvement** in **39% less wall
time**. Beats the 5×20 cadence (4.47) by an additional 3%.

### Observations

1. **Curve is non-monotonic.** PPL improves from 5 → 20 stages, then
   regresses sharply at 50. There's an optimum, not "more is always
   better".

2. **50 × 2 SE underperforms baseline.** Likely two-part cause:
   - First stages train on tiny tries (2-4% of corpus = ~22k-44k
     chars). Too little data to give useful gradient signal.
   - Per-stage budget of 2 SE = 130 optimizer steps is below
     RMSprop's β₂=0.999 second-moment warmup horizon (~1000 steps).
     The optimizer never fully calibrates within a single stage.

3. **Wall time is roughly constant** across cadences (350-400s).
   Trie-build overhead grows linearly with stages, but per-stage
   training shrinks. Bisecting cadences is essentially free
   compute-wise.

4. **5 × 20 reproduction matches v3's separate run** (4.47 vs 4.48).
   Run-to-run variation at this scale is < 1% PPL, so the
   20-stage result is real signal.

### Why 20 × 5 SE wins

Hypothesis: with optimizer state and LR schedule both persistent
across stages, finer cadence is essentially "more curriculum
checkpoints with smooth optimization". More stages = more chances
to integrate growing trie information into the model. The optimum
is bounded below by per-stage minimum compute (need enough SE for
gradients to be useful) and above by transition overhead.

5 SE per stage (=325 optimizer steps) is enough to make meaningful
progress within a stage; 2 SE (=130 steps) is not.

### Followup: optimal cadence may depend on scale

Tested on Shakespeare 1M d=16. At larger scales:
- Gutenberg 5M d=32: per-SE compute is 4-5× larger, so per-stage
  budgets can be smaller in SE terms. Optimum might shift to 30-50
  stages.
- Larger models: per-stage warmup needs scale with model size.
- Worth re-testing at scale before generalizing.

---

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
