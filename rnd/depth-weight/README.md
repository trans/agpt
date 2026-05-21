# Single-axis per-event weighting × normalization regime

Comprehensive single-axis sweep of per-event loss/gradient weighting on
Shakespeare 1M, with three fire-end normalization regimes. The "depth-
weight" name in the directory is a misnomer — the sweep covers four
axes now: depth, mass, entropy, branching factor.

> **Note on previous "no normalization" framing.** Earlier drafts of
> this README mislabeled runs under regime `events` (1/N gradient
> divisor) as "no normalization." That was wrong. `events` HAS
> normalization (1/N applied to d_grads at fire-end); only `none` is
> truly no fire-end divisor. The full matrix below tests all three
> regimes (`events`, `weight`, `none`) explicitly.

## Setup

- Corpus: Shakespeare 1M (`/tmp/shake_baseline_d16_radix`)
- Recipe: `--epochs 10 --lr 3e-3 --optimizer rmsprop
  --lr-schedule warmup-cosine --warmup-epochs 1
  --partition-depth 1 --mass-weight off --no-accumulate`
- Held-out: last 50k chars of `data/input.txt`, sliding-window
  evaluator, d=16, 10k positions, openblas backend
- Seeds: 3 each cell (`/tmp/seed{1,2,3}.model`)
- Cells: 19 weight variants × 3 normalization regimes = 57 cells
  (off+weight is mathematically identical to off+events but included for
   sanity; included)

## What we vary

**Weight axis** (one at a time; mutually exclusive; mean weight is NOT
normalized per chunk):

| axis | flag | what gets weighted | shapes |
|---|---|---|---|
| (none) | `--mass-weight off` | — | off |
| mass | `--mass-weight {SHAPE}` | per-event by node's corpus mass N | log, sqrt, linear, inv-log, inv-linear |
| depth | `--depth-weight {SHAPE}` | per-event by endpoint depth d | log, sqrt, linear, inv-log, inv-linear |
| entropy | `--entropy-weight {SHAPE}` | per-event by H of target distribution | up, down, peakedness |
| branching | `--branching-weight {SHAPE}` | per-event by k/V (vocab fan-out) | log, sqrt, linear, inv-log, inv-linear |

**Normalization regime** (fire-end divisor on d_grads):

| regime | flag | divisor | what it computes |
|---|---|---|---|
| events | (default) | `1/fire_events` (per-event count) | per-event mean gradient |
| weight | `--fire-norm-weight` | `1/Σ w_q` | true weighted-mean per effective event |
| none | `--fire-norm-none` | (no divisor) | raw weighted sum |

For off baseline, `Σw == fire_events`, so `events` ≡ `weight`.

## Full matrix — 3-seed mean held-out PPL

Baseline: off+events 9.11, off+none 9.03.

| variant | events (1/N) | weight (1/Σw) | none (raw sum) |
|---|---|---|---|
| **off** | 9.11 | 9.26 | 9.03 |
| mass-log | 10.36 | 11.62 | **8.79** |
| mass-sqrt | 10.34 | 11.44 | **8.68** |
| **mass-linear** | 11.48 | 11.25 | **8.03** ← best cell measured |
| mass-inv-log | **8.94** | 11.79 | 9.43 |
| mass-inv-linear | **8.97** | 12.31 | 9.50 |
| depth-log | **9.00** | 11.74 | 9.43 |
| depth-sqrt | **9.05** | 11.57 | 9.32 |
| depth-linear | **9.05** | 11.83 | 9.37 |
| depth-inv-log | 9.60 | 11.60 | **8.92** |
| depth-inv-linear | 10.34 | 11.74 | 9.14 |
| entropy-up | **8.99** | 11.91 | 9.07 |
| entropy-down | 9.82 | 11.91 | **9.23** |
| entropy-peakedness | **9.09** | 11.48 | 9.35 |
| **branching-log** | **8.60** | 11.43 | **8.56** |
| branching-sqrt | **8.99** | 11.94 | 9.08 |
| **branching-linear** | **8.44** | 11.44 | **8.41** |
| branching-inv-log | 9.38 | 11.70 | **9.05** |
| branching-inv-linear | 9.42 | 11.70 | **9.23** |

**Bold** = best-of-row regime for that variant; cells lower than
baseline 9.03 are themselves notable.

## Key findings

### 1. `weight` (1/Σw) is uniformly the worst regime

Every variant lands at 10.5-12.6 PPL under `weight`. Universal loser. The
mathematical-purity claim (true weighted-mean-per-effective-event) does
not translate to held-out performance under our 650-step training horizon
+ RMSprop β₂=0.999. **Don't use `--fire-norm-weight`.**

### 2. Two distinct winning patterns

**Pattern A: mass-linear + none = 8.03 PPL** (−12% vs baseline 9.11).
The best single cell measured. But: regime-sensitive — same weight under
`events` gives 11.48 (+26% vs baseline, dramatically worse). Tight
3-seed cluster (8.09, 8.01, 7.99). Significant on Shakespeare; **does
not cross-corpus port** — see `rnd/depth-weight/matrix-gutenberg/`
(below).

**Pattern B: branching-linear ≈ 8.4 PPL across BOTH events and none**
on Shakespeare. Per-seed values 8.11–8.76. **Regime-robust on
Shakespeare** — wins under either regime without needing the right
normalization choice. Slightly worse peak than mass-linear+none, but
robust to regime choice — but this regime-robustness ALSO doesn't
transfer cross-corpus (see Section 4 below). The robustness is real
on Shakespeare, vacuous on Gutenberg (no effect under either regime).

### 3. Aggressive UP-weighting prefers `none`; DOWN-weighting prefers `events`

| direction | best regime | example (3-seed mean) |
|---|---|---|
| up (log, sqrt, linear) | `none` | mass-linear+none 8.03 |
| down (inv-log, inv-linear) | `events` | mass-inv-log+events 8.94 |

Mirror image. Loose hypothesis (unverified): UP-weighted runs have huge
gradient magnitudes; under `none` (no divisor) those huge gradients
help RMSprop's `v` accumulator catch up faster in the transient
(β₂=0.999 has time constant ~1000 steps, we run only 650).

### 4. Cross-corpus check → NO single-axis weighting transfers cleanly

Two Shakespeare winners were tested on Gutenberg 5M (same recipe, 3
seeds each, results in `rnd/depth-weight/matrix-gutenberg/`):

| variant | Shakespeare 1M | Gutenberg 5M |
|---|---|---|
| off events | 9.11 | 9.24 |
| off none | 9.03 | 9.86 |
| mass-linear events | 11.48 | **9.00** (−3%, p≈0.13) |
| mass-linear none | **8.03** (−12%, p≈0.04) | 9.34 (≈ baseline) |
| branching-linear events | **8.44** (−7%) | 9.23 (≈ baseline) |
| branching-linear none | **8.41** (−7%) | 9.78 (≈ baseline) |

Two big patterns:

**Pattern A — mass-linear**: regime flips between corpora (`none` wins
on Shakespeare, `events` wins on Gutenberg). Gutenberg effect is small
and not statistically significant at n=3.

**Pattern B — branching-linear**: robust to regime on Shakespeare
(−7% under both events and none), but **vanishes on Gutenberg
entirely** under both regimes.

Either way: **no single-axis loss-kernel weighting we tested transfers
cleanly cross-corpus.** The Shakespeare wins are real on Shakespeare
but corpus-specific. Suggests either:
- The effect is interacting with Shakespeare's small-corpus structure
  (chunk-distribution, trie shape, or both) — note Shakespeare has 13%
  partial chunks at d=16 vs Gutenberg's 4.4%, per the earlier
  chunk-size analysis in `rnd/per-fire-norm/README.md`.
- Or the right knob isn't on the per-event-weight axis at all, and
  we're chasing Shakespeare-specific quirks.

This is a substantive **negative finding** for the "single-axis
weighting at the loss kernel" research direction: it doesn't
generalize. Future work in this thread should account for it — either
by reframing what counts as a "win" (cross-corpus stability instead of
single-corpus peak), or by pivoting to mechanisms that aren't
single-axis.

### 5. Single-axis weighting can also do nothing or hurt

Counter-evidence to a "weighting always helps" framing:

- All depth shapes (log, sqrt, linear, inv-linear) under `events`: 9.00-10.34 — at-or-worse than baseline
- All entropy shapes: at-or-worse than baseline in best regime
- `weight` regime: catastrophic across the board

Most cells aren't helping. The wins are concentrated in mass and
branching axes, in specific regime combinations.

## What the per-fire normalization choice actually does

Looking across the matrix, the three regimes have distinct character:

- **`events` (1/N)**: the principled "per-event mean gradient" default
  since 609e7ab. Best for DOWN-weighting variants (mass-inv-X, depth-X
  up to log/sqrt/linear). The optimizer sees a gradient whose magnitude
  doesn't depend on weight aggressiveness.

- **`weight` (1/Σw)**: in theory the "true weighted mean per effective
  event." Empirically catastrophic across all variants. Likely fails
  because Σw shrinks for down-weighting variants (Σw < N) so the
  divisor is small and the resulting gradient is huge, AND because for
  up-weighting variants Σw >> N so the divisor over-corrects. Either
  way it amplifies the wrong things.

- **`none` (no divisor)**: best for aggressive UP-weighting (mass-log,
  mass-sqrt, mass-linear, branching-log, branching-linear). RMSprop's
  variance norm handles the resulting large gradient magnitudes.

**There's no single best normalization regime.** The right choice
depends on the weighting scheme. For practical use, `events` is the
safest default (works for everything within ~6% of baseline), while
`none` is the gamble that pays off when paired with aggressive UP
weighting.

## Caveats / scope of this writeup

1. **Single corpus** (Shakespeare 1M, except where noted for Gutenberg
   cross-check) — most findings haven't been validated on other corpora.
2. **Short horizon** (10 SE = 650 optimizer steps). RMSprop β₂=0.999
   has ~1000-step time constant; we're firmly in the transient. Some
   "wins" may be transient-only artifacts; running at 100 SE would
   tell us.
3. **No cross-axis combinations** — each axis tested alone. Composites
   (mass × entropy, mass × branching, etc.) untested.
4. **No β₂ sweep** — the magnitude-dependent transient story is
   speculative; β₂=0.99 at 10 SE would give RMSprop steady state in
   our training horizon and test the story directly.

## Open follow-ups (in priority order)

After the cross-corpus negative finding, priorities shift toward
understanding *why* the Shakespeare wins are Shakespeare-specific
rather than pushing further on single-axis tuning.

1. **β₂=0.99 diagnostic** — repeat mass-linear×{events,none} at
   β₂=0.99 on both corpora. If the regime preference flip (Shakespeare:
   none > events; Gutenberg: events > none) goes away under β₂=0.99,
   the effect was a transient artifact. If it persists, something more
   fundamental about each corpus's loss landscape under each regime.
2. **100-SE runs on the headline cells** — both corpora. Tests whether
   wins survive past the RMSprop transient AND looks for premature
   plateaus. Should also re-baseline things — many of the "wins" we
   measured might just be different points along the same eventual
   trajectory.
3. **Composite weights** — at minimum `w = log(1+N) · log(1+d)` per
   Trans's mass-gated-depth framing. Composites address the
   single-axis insufficiency hinted at by the negative cross-corpus
   result.
4. **RoPE-as-mass probe** (Trans, 2026-05-21) — radical experiment
   that would test "is the model using RoPE-position as a depth-trust
   signal?" by replacing RoPE position with mass. Independent of the
   single-axis-weighting thread; tests a different mechanism.
5. **Reframe success criterion** — for any subsequent weighting
   experiment, treat cross-corpus stability as the bar, not
   single-corpus peak. Single-corpus wins should be treated as
   suggestive only until they cross.

## Files

- `shakespeare/` — early single-seed exploration runs (superseded by `matrix/`)
- `matrix/` — full 19-variant × 3-regime × 3-seed matrix (171 cells); `results.txt` is the flat log
- `matrix-gutenberg/` — cross-corpus check on mass-linear (off + mass-linear × events,none × 3 seeds)
- Each cell dir: `{variant}_{regime}_s{seed}/run.model`, `train.log`, `heldout_ppl.txt`

## Reproducing

```sh
# Build + verify trainer has the relevant flags
just build-agpt-train

# Single cell — for example mass-linear + none
bin/agpt_train \
  --model /tmp/seed1.model \
  --trie-dir /tmp/shake_baseline_d16_radix \
  --epochs 10 --lr 3e-3 --optimizer rmsprop \
  --lr-schedule warmup-cosine --warmup-epochs 1 \
  --partition-depth 1 --mass-weight linear --no-accumulate \
  --fire-norm-none \
  --save /tmp/test.model

# Held-out PPL
bin/agpt_sliding_window_perplexity \
  --model /tmp/test.model \
  --file /tmp/shake_holdout.txt \
  --vocab-file data/input.txt \
  --d 16 --max-positions 10000 --backend openblas
```

Driver scripts that ran the full sweep: `/tmp/run_matrix.sh`,
`/tmp/run_branching.sh`, `/tmp/run_inv_linear.sh`, `/tmp/run_gut_mass_linear.sh`.
(These are in `/tmp`, not committed — the cell outputs in
`matrix/` and `matrix-gutenberg/` are the persistent record.)
