# 2026-05-30 — Canonical-methodology re-test

**Date:** 2026-05-30.
**Status:** CLOSURE. The two methodology caveats noted in the May 2026
investigation (training-loss only, no canonical held-out byte_PPL; legacy
`--mass-weight log` instead of canonical `linear` + `--fire-norm-mass`
default-on) are now closed. Null holds under canonical defaults across
the lr sweep we ran.

## Why this exists

The phase-2B investigation (May 27–28) concluded:

- All three injection forms (direct add, learnable additive, learnable K/V
  token) are flat-to-slightly-negative at d=16 Shakespeare under
  `--mass-weight log + --entropy-lambda 1.0`.
- Oracle test at lr=1e-2 rules out wiring bug — every null is honest.
- Three principled ceilings (KN ceiling, aggregation collapse,
  detached-gradient staleness) explain the null from first principles.

Two methodology caveats were flagged for follow-up:

1. **Training loss only.** All comparisons used the trainer's internal
   loss, not the canonical rolling byte_perplexity reported by
   `bin/agpt_experiment`. The tooling for that didn't exist when the
   investigation happened; it does now.

2. **Non-canonical mass weighting.** All runs used `--mass-weight log`
   (the old heuristic) instead of the now-canonical `--mass-weight linear`
   + `--fire-norm-mass` default-on (paper §2 objective, established by
   main commit `0252072`).

Both are now closed by this run dir.

## Protocol

Same architecture and setup as the phase-2B step-3 work:

- Corpus: `data/input.txt` (Shakespeare 1M chars), d=16.
- Trie: `/tmp/shake_d16_radix` (1,607,928 nodes).
- Predecessor table: `/tmp/shake_d16_predecessors.bin` (8.5M pairs).
- h_caps warm-start: `rnd/cap-recurrence/20260527-smoke/h_caps.bin`.
- Init model: `data/input.model` (d_model=64, n_layers=2, n_heads=4,
  d_ff=256; 108,481 weights).

What changed from phase-2B:

- `--mass-weight linear` (was `log`).
- `--fire-norm-mass` is now the trainer default (no flag needed); we
  do not opt out via `--fire-norm-events`.
- Everything else preserved: `--epochs 5` and `--epochs 25`,
  `--lr 3e-3`, `--optimizer rmsprop --rmsprop-beta 0.999`,
  `--partition-depth 1 --no-accumulate`, `--entropy-lambda 1.0`,
  `--lr-schedule warmup-cosine`.

3 pairs interleaved on `--shuffle-seed`:

- Pair 1: default seed (`0xa17b1ed`)
- Pair 2: `0xb17b1ed`
- Pair 3: `0xc17b1ed`

Each pair: baseline + the kv-mass condition(s) at matching seed. Eval is
HF-converted → `agpt_lm_eval.py` against the canonical Shakespeare
tail-95/5 heldout (`data/.splits/2b7ded401e96b610/heldout_corpus.txt`)
from the main worktree.

## Results

### 5-ep lr sweep — null at every lr

Mean Δ across 3 pairs (kv-mass minus baseline):

| lr | train mean Δ | byte_PPL mean Δ | per-pair byte Δ |
|---|---|---|---|
| 1e-5 | +0.003 | −0.04 | +0.46, +0.11, −0.68 |
| 1e-3 | +0.025 | +0.11 | +0.42, −0.07, −0.02 |
| 1e-2 | −0.024 | −0.12 | +0.29, −0.10, −0.55 |

All means well inside the per-pair noise band (±0.5 byte_PPL). Per-pair
sign of training-loss and byte_PPL Δ matches in every cell (no
directional divergence). Null holds at every lr.

### 25-ep at lr=1e-2 — suggestive amplification, control eliminates it

Extended to 25 epochs (matching the original oracle showcase) to let
`W` grow into the centroid-fitting regime. Two conditions tested at
this depth, paired against the same baselines:

| pair | baseline byte | kv-mass byte | kv-random byte | Δ mass | Δ random |
|---|---|---|---|---|---|
| 1 | 8.31 | 8.63 | 8.82 | +0.31 | +0.50 |
| 2 | 8.08 | 8.68 | 8.56 | +0.59 | +0.47 |
| 3 | 8.47 | 8.29 | 8.53 | −0.18 | +0.06 |
| mean | 8.29 | 8.53 | 8.64 | **+0.24** | **+0.35** |

| condition | mean Δ train | mean Δ byte | byte/train ratio |
|---|---|---|---|
| kv-mass lr=1e-2 | +0.084 | +0.241 | 2.9× |
| kv-random lr=1e-2 | +0.078 | +0.345 | 4.4× |

**The kv-random control eliminates the centroid-specific reading of
the amplification ratio.** Random produces a *higher* byte/train ratio
than mass — the opposite of what the false-continuation hypothesis
predicts crisply. (Hypothesis predicted: mass should have the larger
held-out hurt because it fits a misleading centroid; random should have
smaller held-out hurt because pure noise has no directional misprediction
component.)

Per-pair: 2 of 3 pairs favor "mass less harmful than random" (Δ ~0.10,
SE ~0.10 — ~1σ). Insufficient to conclude either:

- (a) kv-mass extracts weak generalization signal (since 1σ);
- (b) kv-mass = kv-random = generic noise injection.

What's clear: the byte_PPL amplification at this scale is **not
centroid-specific** — it's a generic perturbation-on-held-out effect
that kv-random reproduces.

## Interpretation

The closure language:

1. **Null holds under canonical methodology.** Both metrics (training
   loss, byte_PPL) confirm at lr ∈ {1e-5, 1e-3, 1e-2} for 5-ep, and
   at lr=1e-2 for 25-ep. Mean Δ is within per-pair noise for all
   conditions; per-pair signs of train and byte agree (no
   train-improves-while-byte-worsens divergence).

2. **The mass-weighted centroid is not distinguishable from random
   injection at this scale.** The kv-random control gives a similar
   (slightly larger) byte_PPL hurt as kv-mass. The "false continuation"
   refinement (mass-weighted centroid mispredicts specific contexts) is
   not crisply demonstrated; it survives as a theoretical reading
   consistent with the principled framework, but not as a measured
   effect.

3. **Any slight help-vs-harm direction is consistent with general
   syntax-continuation signal.** Even if the mass centroid carries a
   tiny amount of "what comes after this prefix in general"
   information, it doesn't tip the model into useful territory —
   AGPT-d16 byte_PPL is still ~8.3, while Kneser-Ney baselines on the
   same setup are around 4. There is no regime in this sweep where
   kv-injection lifts AGPT closer to KN; the principled ceiling
   reasoning (KN ceiling + aggregation collapse + detached-gradient
   staleness) remains the cleanest theoretical reading.

4. **`‖W‖` is uninstrumented.** Whether W actually reached the
   centroid-fitting magnitude at 25-ep / lr=1e-2 under canonical
   mass-weight is unknown. The original 25-ep oracle showcase had
   `‖W_v‖ → 100+` (under log mass-weight); we don't know our W is
   anywhere near that. If the canonical fire-norm-mass dampens W
   updates, the 25-ep run may not have reached the fitting regime —
   in which case "no signal here" doesn't yet rule out signal at
   larger ‖W‖. A definitive test would require ‖W‖ logging per
   epoch + possibly bigger model/longer training, both real research
   investments not closing-the-caveat ones.

## What this does NOT close

- **The Q2-D persona-clustering variant.** Preserves more cross-K
  structural signal than mass-weighted centroid; partial mitigation of
  aggregation collapse, no fix for detached gradient or distribution
  mismatch. Not tested here.

- **The low-d regime where `d << corpus dependency range`.** Gives
  recurrence more genuinely out-of-window context to carry; doesn't
  fix aggregation collapse but tilts the trade. Not tested here.

Both flagged in the prior memory as "would change picture only partially."

## Artifacts

The run dir holds:
- `p[1-3]-baseline.log` / `.model` / `.eval.json` — 5-ep baselines
- `p[1-3]-kvmass-lr{1e-5,1e-3,1e-2}.log` / `.model` / `.eval.json` —
  5-ep kv-mass at three learning rates
- `p[1-3]-{baseline,kvmass-lr1e-2,kvrandom-lr1e-2}-25ep.log` /
  `.model` / `.eval.json` — 25-ep extension
- `.hf/` per checkpoint — HF-converted weights used for byte_PPL eval
  via `agpt_lm_eval.py` (regenerable from `.model`)

Per branch convention, only this README is committed; logs and weights
are regenerable from the protocol above.
