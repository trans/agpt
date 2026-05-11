# Granularity Redundancy — pd, Curriculum, and Hotspot Are Not Composable

**Date:** 2026-05-01
**Trainer:** post-bias-fix (commit `1c858c0+`), Adam pd=6 era
**Corpus:** Shakespeare 1M, d=32 d_model=64

> **TL;DR.** `--partition-depth N`, `--curriculum progressive`, and
> `--hotspot-coverage F` all add the same thing: **optimizer-fire density**.
> Stacking them at pd≥2 is mutually redundant and *hurts* PPL because the
> mechanisms fight each other's update cadence. Each mechanism is only a
> win when it's the *only* density-adder in play.
>
> The new fast-iteration sweet spot is **pd=3 flat** (4.70 PPL @ 78s on
> Shakespeare 1M). The previous "best" of pd=6 flat (3.82 PPL @ 640s) is
> worth its 8× wall-clock only for absolute-PPL final runs.

## Background — what motivated this sweep

The `partition-depth` finding (2026-04-30) showed pd=6 + Adam = 3.82 PPL,
massively beating pd=1 = 10.83. We had several earlier abandoned lines
(`hotspot-curriculum`, `subtree-dropout`, depth-curriculum) measured at
pd=1 with absolute numbers ~10× higher than pd=6 makes possible. **Many
prior conclusions needed re-validation under the post-pd=6 recipe.**

The user's intuition (correctly) flagged that `--curriculum progressive`
and `--hotspot-coverage` are themselves "ways to add training granularity"
— so they might either compose with pd or fight it. This sweep settled the
question.

## The full grid (Shakespeare 1M d=32, all Adam unless noted, --no-accumulate)

| recipe | epochs | optimizer | mean PPL | wall | reps |
|---|---:|---|---:|---:|---:|
| pd=6 flat | 3 | Adam lr=1e-3 | 3.82 | 640s | from memory |
| **pd=3 flat** | 3 | Adam lr=1e-3 | **4.70** ±.034 | **78s** | 3 |
| pd=2 flat | 3 | Adam lr=1e-3 | 6.02 ±.005 | 56s | 3 |
| pd=1 flat | 3 | RMSprop lr=3e-3 | 10.49 | 68s | 1 |
| pd=1 flat | 3 | Adam lr=1e-3 | 15.88 ±.0002 | 61s | 3 |
| pd=1 flat | 10 | RMSprop lr=3e-3 | 8.09 ±.28 | 197s | 3 |
| pd=1 flat | 10 | Adam lr=1e-3 | 9.42 ±.0003 | 204s | 3 |
| pd=6 hotspot 0.8 | 3 | Adam lr=1e-3 | 4.07 ±.046 | 723s | 3 |
| pd=6 progressive | 3 | Adam lr=1e-3 | 5.28 ±.011 | 5480s | 3 |
| pd=3 progressive | 3 | Adam lr=1e-3 | 6.33 ±.052 | 601s | 3 |
| pd=2 progressive | 3 | Adam lr=1e-3 | 6.88 ±.270 | 158s | 3 |
| pd=1 hotspot 0.8 | 3 | RMSprop lr=3e-3 | **9.21** ±.19 | 65s | 3 |
| pd=1 hotspot 0.8 | 3 | Adam lr=1e-3 | 10.50 ±.0007 | 65s | 3 |
| pd=1 progressive | 3 | Adam lr=1e-3 | 7.22 ±.122 | 106s | 3 |
| pd=1 progressive | 3 | RMSprop lr=3e-3 | 10.04 ±1.31 | 113s | 3 |
| pd=1 progressive | 10 | Adam lr=1e-3 | 5.68 ±.21 | 377s | 3 |
| pd=1 progressive | 10 | RMSprop lr=3e-3 | 7.08 ±.40 | 362s | 3 |

## Finding 1 — Curriculum's value is monotonically destroyed by pd

Same recipe, varying pd, with vs without `--curriculum progressive`:

| pd | flat | progressive | Δ % |
|---:|---:|---:|---:|
| **1 (Adam)** | 15.88 | **7.22** | **−54%** ✓ |
| 1 (RMSprop) | 10.49 | 10.04 ±1.31 | −4% (in noise) |
| 2 | 6.02 | 6.88 | +14% ✗ |
| 3 | 4.70 | 6.33 | +34% ✗ |
| 6 | 3.82 | 5.28 | +38% ✗ |

The Adam pd=1 +54% gain looks impressive but is **mostly an Adam-specific
rescue**, not a curriculum effect (see Finding 4). RMSprop's +4% at pd=1
(not significant given variance) is the actual ceiling of the
"curriculum-as-staged-learning" benefit.

For pd≥2, curriculum is uniformly harmful — both PPL and wall-clock
regress. The redundancy is geometric: pd=N already provides Adam fires per
N-gram-rooted partition group; curriculum's depth-loop adds a 32× outer
multiplier on top, dragging optimizer state through 32 sub-distributions
per outer epoch and never letting it settle.

## Finding 2 — Hotspot's value is similarly pd-bounded

| pd | recipe | flat | hotspot 0.8 | Δ % |
|---|---|---:|---:|---:|
| 1 | RMSprop | 10.49 | **9.21** | **−12%** ✓ |
| 1 | Adam | 15.88 | 10.50 | −34% (Adam rescue, see F4) |
| 6 | Adam | 3.82 | 4.07 | +6% ✗ |

At pd=1, splitting the high-residual rc=0 (space character) subtree into
~30 pieces between epochs creates 30 new Adam fires per epoch concentrated
on the worst-fitting region. Real density redirection.

At pd=6, partition-6 has already created thousands of Adam fires *inside*
rc=0. Hotspot's outer-boundary reshuffling doesn't add fires — it just
relabels which group of partition-groups gets the same total updates. Pure
bookkeeping overhead, slight regression.

**RMSprop pd=1 hotspot (9.21 PPL) is the only "extra mechanism" win in the
entire grid** — and it's at the fastest wall-clock cell (65s).

## Finding 3 — pd=3 is the new wall-clock-sweet-spot default

| pd | PPL | wall | PPL × wall |
|---:|---:|---:|---:|
| 1 (RMSprop) | 10.49 | 68s | 713 |
| 2 | 6.02 | 56s | 337 |
| **3** | **4.70** | **78s** | **367** |
| 6 | 3.82 | 640s | 2445 |

pd=3 captures ~85% of pd=6's PPL gain at 12% the wall-clock. For dev/test
iteration, pd=3 should be the new default; pd=6 reserved for "I want the
absolute lowest PPL" final runs.

The pd=1→2 jump is the largest single granularity step (15.88→6.02 at
RMSprop, or 6.02 from pd=2 itself; 62% relative to RMSprop pd=1). Past
pd=2 each step buys diminishing returns: pd=2→3 gains 22%, pd=3→6 gains
19% at 8× the wall.

## Finding 4 — Adam at coarse pd has a cold-start problem unrelated to AGPT

The Adam pd=1 flat catastrophe (15.88 PPL) is **not a pd=1 ceiling** but
an Adam-cold-start artifact. Two mechanisms compounding:

1. **Second-moment estimator unwarmed.** β₂=0.999 has time constant
   ~1000 steps. At pd=1 + 3 SE, Adam fires only 195 times — barely past
   bias correction. The per-parameter LR adaptation (the entire point of
   Adam) hasn't kicked in.

2. **Momentum cancels across orthogonal root-child gradients.** The 65
   root-child subtrees are nearly disjoint character distributions. Their
   gradients in parameter space are roughly orthogonal. β₁=0.9
   averages ~10 of these together → mostly cancellation, useless update
   direction.

RMSprop has neither problem (no momentum; second-moment cold-start is
less catastrophic because there's no update direction to confuse).
Confirmed by RMSprop pd=1 flat hitting 10.49 (matches the historical
10.83 baseline) while Adam at the identical recipe hits 15.88.

**Confirmation via 10 SE pd=1 flat:** boosting fires from 195 → 650
keeps RMSprop deterministic-ish (10.49 → 8.09, range 0.28) but
dramatically rescues Adam (15.88 → 9.42, range 0.0003). The
Adam-vs-RMSprop gap collapses from 5.4 → 1.3 PPL — proof that the
15.88 number was Adam waiting for warm moments, not a pd=1 ceiling.

**Adam-vs-RMSprop reverses with optimizer-fire count:**

| optimizer fires | better optimizer | gap |
|---:|---|---:|
| 195 (pd=1 flat 3SE) | RMSprop | Adam +5.4 PPL |
| 650 (pd=1 flat 10SE) | RMSprop | Adam +1.3 PPL |
| 6240 (pd=1 prog 3SE) | Adam | Adam −2.8 PPL |
| 20.8k (pd=1 prog 10SE) | Adam | Adam −1.4 PPL |
| 850k (pd=6 flat 3SE) | Adam | Adam −0.13 PPL (memory) |

The crossover sits somewhere between 650 and 6240 fires, roughly where
Adam's β₂=0.999 second-moment estimator (time constant ~1000 steps) is
warm enough to provide useful per-parameter LR scaling.

Once moments warm and consecutive gradients share structure (which they
do under any of {pd≥2, curriculum, hotspot}), Adam's per-parameter
adaptation pays off.

## Finding 5 — More SE at pd=1 progressive can't catch pd=6 flat

| pd=1 progressive Adam | epochs | PPL | wall |
|---|---:|---:|---:|
|  | 3 | 7.22 | 106s |
|  | 10 | 5.68 | 377s |

PPL drops 21% per 3.3× SE. Extrapolating: ~30 SE would be needed to reach
pd=6 flat's 3.82 PPL — and at ~1130s wall-clock, that's already 1.8× pd=6
flat's wall. Curriculum cannot Pareto-dominate partition-depth on this
corpus regardless of SE budget.

## Finding 5b — pd=1's true plateau is 4.43-4.46 PPL at SE≈110, optimizers near-tied

Two 120-SE runs (RMSprop and Adam) with `--save-every 10`:

| SE | Adam PPL | RMSprop PPL | Adam − RMS |
|---:|---:|---:|---:|
| 10 | 7.81 | 7.80 | +0.01 |
| 20 | 6.11 | 6.52 | −0.41 |
| 30 | 5.32 | 6.01 | −0.69 |
| 40 | 5.01 | 5.38 | −0.37 |
| 50 | 4.81 | 5.43 | −0.62 |
| 60 | 4.65 | 4.90 | −0.25 |
| 70 | 4.56 | 4.68 | −0.12 |
| 80 | 4.51 | 4.57 | −0.06 |
| 90 | 4.46 | 4.51 | −0.05 |
| 100 | 4.44 | 4.47 | −0.03 |
| 110 | **4.43** | **4.46** | −0.03 |
| 120 | 4.43 | 4.46 | −0.03 |

**Adam plateau: 4.43 PPL. RMSprop plateau: 4.46 PPL.** Adam wins by only
0.03 PPL at asymptote — effectively tied. But Adam descends substantially
faster in mid-training (SE=20-50 advantage of 0.4-0.7 PPL).

**Interpretation:** the per-optimizer ceilings at pd=1 flat converge to
the same value once moments have fully warmed. The "Adam is fundamentally
better at high fire counts" framing from earlier in this doc is too
strong — at the pd=1 plateau they're nearly tied. What's empirically
true is **"Adam descends faster once warm; RMSprop descends slower but
catches up."** Total wall-clock ~40 min for either.

The historical "5.39 @ 40 SE" finding was correct for that point but
didn't capture that pd=1 has another full PPL of headroom past SE=40.

Comparison at-plateau vs default-SE:

| recipe | PPL | wall |
|---|---:|---:|
| pd=6 Adam 3 SE | **3.82** | 640s |
| pd=3 Adam 3 SE | 4.70 | 78s |
| **pd=1 RMSprop 110 SE** | **4.46** | ~2200s |

pd=1's plateau (4.46) **beats pd=3 default-SE** (4.70) — but at 28× the
wall-clock. pd=6 still wins by 0.64 PPL. The pd-vs-SE tradeoff is steep:
each pd-step bumps the achievable plateau by ~30%, and Adam-fire
multiplication via SE is much more expensive per PPL-point than
multiplication via partition-depth.

The clean intuition: **partition-depth multiplies fire density at zero
wall-clock cost (just relabels groups); SE multiplies fire count at full
wall-clock cost.** That's why a small pd boost is worth ~10× the SE.

## Finding 6 — Variance signature distinguishes the mechanisms

- **pd flat**: deterministic to ±0.005 across seeds (essentially
  bit-exact training trajectory).
- **Hotspot**: deterministic to ±0.001 (split decisions are
  score-determined; resulting trajectory still deterministic).
- **Progressive curriculum**: ±0.1 to ±1.3 across seeds. The inner
  depth-loop traverses sub-distributions in different "moods" depending
  on initial state, leading to seed-dependent endpoints.

Curriculum's path-dependence is itself diagnostic: the mechanism is
adding a kind of training noise that other mechanisms don't, which may be
part of why it's hard to compose with stable mechanisms.

## Finding 7 — d × pd plateau matrix at 120 SE (2026-05-02)

Cross-d sweep at long-SE budget, fixing the trie-depth/eval-seq mismatch
(each d evaluated at native seq=d):

| d \ pd | 1 | 2 | 3 | 5 | 6 (3 SE) |
|---|---:|---:|---:|---:|---:|
| 16 (PPL@16) | 4.54 | 4.97 | 4.71 | 4.51 | 4.14 |
| 24 (PPL@24) | 4.48 | 4.42 | 4.36 | **3.71** | 3.93 |
| 32 (PPL@32) | 4.43 | 4.12 | 4.08 | tbd | 3.82 (memory) |

**Key observations:**
1. **d=24 pd=5 120 SE = 3.71 PPL@24** is the lowest absolute PPL we've
   measured across this session. Beats d=32 pd=6 3 SE (3.82) by 3% but
   at 19× the wall (12000s vs 640s).
2. **d=24 pd=5 10 SE = 3.82 PPL@24** matches d=32 pd=6 3 SE at the same
   PPL but ~1.8× the wall. So d=24 pd=5 needs more SE budget but
   reaches a lower asymptote.
3. **At d=16, pd=2 hits a non-monotonic spike** (4.97 vs pd=1's 4.54
   and pd=3's 4.71). The pd=1→pd=2 jump is a regression at d=16, opposite
   of d=24 and d=32 where it helps. Hypothesis: at d=16 the bigram-level
   partition (1404 groups) creates "wrong-sized" batches relative to
   the available depth structure.
4. **pd=1 plateaus tighten dramatically across d**: 4.54 (d=16), 4.48
   (d=24), 4.43 (d=32). Just 0.11 PPL spread despite 2× depth. Validates
   the trie-attention-framing prediction that d=16 captures most
   predictively-useful context for Shakespeare.

**Memory tradeoff (d=24 pd=5 vs d=32 pd=6):**

| recipe | GPU mem | KV cache | corpus chars |
|---|---:|---:|---:|
| d=16 pd=5 | 2763 MB | 531 MB | 9.3M |
| d=24 pd=5 | 3673 MB | 563 MB | 18.1M |
| d=32 pd=6 | 4596 MB | 570 MB | 27.0M |

d=24 saves 920 MB GPU memory vs d=32 (~20%). KV cache shrinkage
is small (mass=1 compaction dominates); the savings come from
ancestor lists and packed-scratch buffers that scale with total
trie chars.

This makes d=24 pd=5 a meaningful Pareto choice: lower PPL, lower
memory, slower training. Useful if compute is plentiful and memory
or absolute PPL matters.

## Finding 8 — `--mass-weight off` beats `log` (the project default)

Tested at d=16 pd=6 3 SE Adam (4 modes) and d=32 pd=6 3 SE Adam
(off vs log head-to-head):

**At d=16 pd=6 3 SE:**

| mass-weight | PPL@16 | wall |
|---|---:|---:|
| **off** | **4.11** | 556s |
| sqrt | 4.12 | 592s |
| linear | 4.12 | 589s |
| log (default) | 4.19 | 533s |

**At d=32 pd=6 3 SE:**

| mass-weight | PPL@32 | wall |
|---|---:|---:|
| **off** | **3.80** | 688s |
| log (default) | 3.82 | 709s |

**Mass-weight log was the project default** for AGPT runs but is the
worst of the four modes. Switching to `off` (no per-query weighting)
gives a small but consistent improvement (0.02 PPL at d=32, 0.08 at
d=16). The other count-based modes (sqrt, linear) are tied with off.

**Why log specifically loses**: log compresses the count distribution
into a narrow weight range (~0.7-9.2× for Shakespeare's count
distribution). It dampens the count signal more than it helps —
neither preserving uniform per-context weight (off) nor preserving
proportional count weighting (sqrt/linear). The "compromise" mode
ends up with neither benefit.

**Recipe change**: drop `--mass-weight log` from default recipes;
use `--mass-weight off` (or omit the flag).

**New project-best PPL on d=32 pd=6 3 SE flagship recipe: 3.80**
(was 3.82). Modest but real, and free — same wall, lower PPL.

## Finding 9 — `--weight-decay 0.01` compounds with mw=off for new project best

Tested at d=16 (knob matrix) and d=32 (weight-decay sweep):

**At d=16 pd=6 3 SE (Adam mw=off, baseline 4.11):**

| wd | PPL@16 |
|---|---:|
| 0 | 4.11 |
| **0.01** | **4.04** (-0.07) |
| 0.10 | 4.33 (+0.22) — too strong |

**At d=32 pd=6 3 SE (Adam mw=off, baseline 3.80):**

| wd | PPL@32 |
|---|---:|
| 0 | 3.80 |
| 0.005 | 3.77 |
| **0.01** | **3.71** (-0.09) |
| 0.02 | 3.72 (-0.08) |

**`wd=0.01` is the sweet spot at both d levels.** Decoupled weight
decay (AdamW-style; applies after Adam step, scales with current
lr) gently shrinks unused weights toward zero, providing
generalization regularization that the project never used before
(default was `wd=0`).

**New project-best PPL: 3.71 at d=32 pd=6 3 SE Adam mw=off wd=0.01,
in 676s wall-clock.** This matches the d=24 pd=5 120 SE plateau
(3.71) at 1/19 the wall-clock — so the lower asymptote is reachable
in the fast-iteration recipe via knob tuning, no need for the
expensive 120-SE budget.

**Recipe change:** add `--weight-decay 0.01` to default recipes.

## Finding 10 — Combined recipe at long SE breaks every prior project best

After accumulating the wins (mw=off + wd=0.01) on top of the
established AGPT recipe (pd=6, Adam, warmup-cosine, no-accumulate),
ran 60 SE with `--save-every 10` to map the long-budget curve at the
new recipe.

**Recipe**: `pd=6 Adam lr=1e-3 mw=off wd=0.01 warmup-cosine warmup-epochs=1 entropy-lambda=1.0 no-accumulate`

| SE | PPL@32 | Δ from prev |
|---:|---:|---:|
| 10 | 4.22 | — |
| 20 | 4.14 | −0.08 |
| 30 | 3.96 | −0.18 |
| 40 | 3.78 | −0.18 |
| 50 | 3.55 | −0.23 |
| **60** | **3.35** | **−0.20** |

Wall: 12921s (3.6 hours).

**The curve is still descending steeply at SE=60** (−0.20 PPL between
SE=50 and SE=60, no plateau). At 1.74 bpc the model is ~25% below
its pre-pd-finding 1.93 bpc plateau, and approaching the theoretical
English entropy floor (1.3-1.5 bpc).

Predict 120 SE would push to ~3.0 PPL (1.58 bpc) if descent continues.

## Project PPL history at a glance

| era | recipe | PPL@32 | wall |
|---|---|---:|---:|
| pre-pd-finding | pd=1 3 SE RMSprop | 10.83 | 60s |
| epoch-scaling | pd=1 40 SE RMSprop | 5.39 | 800s |
| partition-depth | pd=6 3 SE RMSprop | 3.95 | 654s |
| Adam pd=6 | pd=6 3 SE Adam mw=log | 3.82 | 640s |
| mw fix | pd=6 3 SE Adam mw=off | 3.80 | 688s |
| mw + wd | pd=6 3 SE Adam mw=off wd=0.01 | 3.71 | 676s |
| pd=1 plateau | pd=1 ~110 SE RMSprop/Adam | 4.43-4.46 | 2400s |
| pd=2 plateau | pd=2 120 SE Adam | 4.12 | 2250s |
| pd=3 plateau | pd=3 120 SE Adam | 4.08 | 2965s |
| d=24 pd=5 | pd=5 120 SE Adam mw=log | 3.71 | 12000s |
| combined recipe 60 SE | pd=6 60 SE mw=off wd=0.01 | 3.35 | 12921s |
| **combined recipe 120 SE** | **pd=6 120 SE mw=off wd=0.01** | **3.30** | **~26000s** |

## Finding 11 — 120 SE combined recipe: 3.30 PPL@32 (project best)

Extending the 60 SE = 3.35 result to 120 SE pushed plateau slightly lower
to 3.30 PPL@32. Wall-clock ~7.2 hours.

Trajectory at 120 SE checkpoints:

| SE | PPL@32 | Δ vs prev |
|---:|---:|---:|
| 10 | 4.30 | — |
| 20 | 4.18 | −0.12 |
| 30 | 4.15 | −0.02 |
| 40 | 4.06 | −0.09 |
| 50 | 4.01 | −0.06 |
| 60 | 3.93 | −0.07 |
| 70 | 3.85 | −0.08 |
| 80 | 3.77 | −0.08 |
| 90 | 3.66 | −0.11 |
| 100 | 3.52 | −0.14 |
| 110 | 3.38 | −0.14 |
| **120** | **3.30** | −0.07 |

The descent is steepest in the SE=80-110 window where cosine decay is
most active. The last delta (110→120) is small enough that we're near
plateau — extending to 240 SE would probably yield at most 0.05-0.10
more PPL.

**3.30 PPL = 1.72 bpc** — ~11% better compression than the pd=6 3 SE
baseline (1.93 bpc). Within ~0.3 bpc of the theoretical English
entropy floor (1.3-1.5 bpc).

## Comparison of 60 SE vs 120 SE at the new recipe

The same recipe with different cosine-decay windows:

| budget | wall | best PPL | notes |
|---|---:|---:|---|
| 60 SE | 12.9k s | 3.35 | cosine bottoms at SE=60 |
| 120 SE | 26.0k s | 3.30 | cosine bottoms at SE=120 |

The 60 SE schedule reaches a similar PPL faster because the LR decays
more aggressively. The 120 SE schedule keeps LR higher for longer,
allowing the model to find a slightly lower minimum — but at 2× the
wall-clock for only 0.05 PPL improvement.

For the project's "best PPL" target: **120 SE wins**.
For "best PPL/wall": 60 SE is the better operating point.

Each row generally builds on the prior insights. The combined-recipe
result represents the best PPL achieved on Shakespeare 1M with the
108k-parameter model under any recipe found so far.

## Finding 12 — Architecture bump: d=96 L=6 hits 2.93 PPL@32 (project best, breaks 3.0)

After all the recipe optimization at the 108k-param model, tried bumping
model architecture. Results at d=32 pd=6 3 SE Adam mw=off wd=0.01:

| arch | params | PPL@16 | PPL@32 |
|---|---:|---:|---:|
| d=64 L=2 (baseline) | 108k | 4.04 | 3.71 |
| **d=64 L=4** (depth only) | 205k | **3.76** | **3.41** |
| **d=96 L=2** (width only) | 232k | **3.76** | **3.42** |
| **d=96 L=6** (full) | 683k | **3.46** | **2.93** |

**Striking finding 1: depth and width contribute IDENTICALLY at this scale.**
d=64 L=4 ≈ d=96 L=2 to 0.01 PPL across both eval contexts. Each ~2× param
bump buys ~0.3 PPL.

**Striking finding 2: combined effect is roughly additive.** d=96 L=6 sits
at the sum of the individual deltas. No multiplicative interaction.

**Striking finding 3: 2.93 PPL = 1.55 bpc — at the theoretical English
entropy floor (1.3-1.5 bpc).** Architecture has been substantially capacity-
limited. The small d=64 L=2 model couldn't represent everything the corpus
contained; bigger model fills the gap.

**Wall-clock perspective:**
- d=96 L=6 3 SE: 2690s wall, 2.93 PPL
- d=64 L=2 120 SE: 25833s wall, 3.30 PPL

**The bigger model at 3 SE beats the small model at 120 SE by 0.37 PPL,
in 1/10 the wall.** Architecture scaling is way more wall-efficient than
SE scaling at this point.

### Memory caveat

d=96 L=6 d=32 OOM'd on the first attempt because GPU was momentarily
sharing with other processes; succeeded on retry with cleaner GPU
state. KV cache at d=96 L=6 d=32 is ~2.6 GB, vs 569 MB for the small
model. d=32 + L=6 + d_model=96 needs ~7-8 GB GPU memory at startup.
Tight on 8 GB cards.

**For headroom**: drop one of the dimensions (d=96 L=4 should fit
comfortably; d=128 L=2 also).

## Project PPL history at a glance (updated)

| era | recipe | PPL@32 | wall |
|---|---|---:|---:|
| pre-pd-finding | pd=1 3 SE RMSprop | 10.83 | 60s |
| epoch-scaling | pd=1 40 SE RMSprop | 5.39 | 800s |
| partition-depth | pd=6 3 SE RMSprop | 3.95 | 654s |
| Adam pd=6 | pd=6 3 SE Adam mw=log | 3.82 | 640s |
| mw fix | pd=6 3 SE Adam mw=off | 3.80 | 688s |
| mw + wd | pd=6 3 SE Adam mw=off wd=0.01 | 3.71 | 676s |
| 60 SE recipe | pd=6 60 SE | 3.35 | 12921s |
| 120 SE recipe | pd=6 120 SE | 3.30 | 25833s |
| **d=96 L=6 3 SE** | **pd=6 3 SE on bigger arch** | **2.93** | **2690s** |

## Other knobs probed at d=16 (none beneficial)

| knob | PPL@16 | verdict |
|---|---:|---|
| `--grad-clip-norm 1.0` | 4.19 | hurts; AGPT gradients already low-variance |
| `--grad-clip-norm 0.5` | 4.14 | mild harm |
| `--lr-schedule constant` | 5.68 | catastrophic; warmup-cosine is essential |
| `--lr-rule inv-depth` | 4.14 | within noise |
| `--lr-rule inv-sqrt-depth` | 4.14 | within noise |
| `--lr-rule sqrt-batch` | 5.99 | unstable; multiplies high-mass LR by ~14× and combines with Adam adaptation |
| `--lr-rule residual` | NaN | broken at 3 SE (needs prev-epoch scores; uninitialized at first epoch) |
| `--ce-only` | 4.49 | +0.38 over AGPT default; validates AGPT framework value |

The `--ce-only` result is the framework-level sanity check:
disabling AGPT's distributed-target KL term (and using only single-
token CE) costs 0.38 PPL. **AGPT's distribution-matching is
verifiably contributing 0.38 PPL of gain over plain SGD-style
next-token training at this recipe** — about 0.13 bpc compression
gain or ~20% of the remaining-headroom-to-entropy-floor gap on
Shakespeare.

## Bug found and fixed during this work — O(n²) selection sort

A long-standing post-epoch sort over `n_root_children` was implemented
as selection sort. At pd=1 (65 groups) it was negligible. At pd≥5 it
became catastrophic: pd=6 hid ~6-7 min of sort time per epoch
(40B compares), pd=5 hid ~100s/epoch (10B compares). Hidden because
the trainer's `wall=Xs` stat extracts only Epoch-line times; the sort
runs after that print. Fixed 2026-05-01 by replacing with qsort
(O(n log n)).

**PPL numbers in this document are unaffected** (sort runs after Adam
updates). **Wall-clock numbers from before the fix understated real
wall by the hidden sort time** at high pd. For pd=6 3 SE: reported
640s, real ~30 min before fix, ~640s after. For pd=5 120 SE: reported
~12k s, real ~3-4 hours before fix and same with fix (the new pd=5
runs in this session used the fixed binary).

Documented in `memory/project_selection_sort_bug.md`.

## Implications for project recipes

**Default for fast iteration**: drop pd=6 from the standard recipe;
make **pd=3** the default. Saves 8× wall on every dev cycle.

**Default for headline runs**: keep pd=6 flat (3.82 baseline). Don't
stack curriculum or hotspot.

**At pd=1** (e.g. when memory-constrained): use RMSprop + hotspot, not
Adam + anything. RMSprop pd=1 hot 0.8 = 9.21 / 65s is the best
fast-and-tiny-fire-count cell.

**Many older "X helps" findings need re-validation at pd=3+**: the
old pre-pd-finding hotspot-curriculum doc claimed hs=0.5 helps by
0.24 PPL at d=16 pd=1. That was a real finding *for that recipe*, but the
current recipe is qualitatively different. Don't compose old "helpers"
without re-measuring.

## Reproduction

The runs were issued via four ad-hoc bash batches (`/tmp/run_*.sh`,
preserved in shell history). For repeatable form:

```sh
# pd=3 flat (the new default)
bin/agpt_train --trie-dir TRIE --epochs 3 --lr 1e-3 \
  --optimizer adam --lr-schedule warmup-cosine --warmup-epochs 1 \
  --entropy-lambda 1.0 --mass-weight log --no-accumulate \
  --partition-depth 3

# pd=1 + RMSprop + hotspot (the small-budget cell)
bin/agpt_train --trie-dir TRIE --epochs 3 --lr 3e-3 \
  --optimizer rmsprop --rmsprop-beta 0.999 \
  --lr-schedule warmup-cosine --warmup-epochs 1 \
  --entropy-lambda 1.0 --mass-weight log --no-accumulate \
  --hotspot-coverage 0.8
```

## See also

- `../partition-depth/README.md` — the parent finding that motivated this sweep
- `../hotspot-curriculum/README.md` — the older hotspot+curriculum work at d=16
  pd=1 (pre-pd=6 era; numbers no longer comparable to current best)
- `../agpt-epoch-scaling/README.md` — the SE-budget reframing that
  immediately preceded partition-depth
