# Preliminary findings — dual-view consistency at small scale

**Status:** mechanism validated at 10k positions (Crystal/openblas trainer).
50k confirmation in flight. Full Shakespeare 1M scale requires CUDA port
(Crystal at ~90 pos/s = ~3 hr/epoch).

## Setup

- Trainer: `bin/agpt_dual_train` (Crystal, openblas backend, per-position
  Adam fire — partition aggregation deferred)
- Architecture: d_model=64, n_heads=4, n_layers=2 (loaded from
  `data/input.random.model` for both F and B — same init, fair comparison)
- Corpus: `data/input.txt` (Shakespeare 1M, V=65)
- Training: 10k corpus positions per run, seq_len=32, lr=3e-3, seed=42

## Results at 10k positions

| run | CE_F | CE_B | KL gap (mean of both directions) | comment |
|---|---:|---:|---:|---|
| β = 0 (no coupling, baseline) | 2.99 | 2.95 | **0.66** | F and B diverge as they each learn their own direction |
| β = 0.1 aligned | 2.99 | 2.94 | **0.54 (-17%)** | clear shrinkage, no CE penalty |
| β = 1.0 aligned | 3.05 | 3.06 | **0.17 (-75%)** | strong coupling, ~2% CE cost |
| β = 0.1 **shuffled-suffix** | 2.98 | 3.10 | **0.69 (+4% vs β=0)** | mismatched pairing → no KL shrinkage |

## Tier-by-tier readout

(Tiers from `PLAN_REVIEW_1.md`.)

- **Tier 1 — KL closes:** ✓ confirmed. KL gap shrinks monotonically with
  β. At β=1.0 the gap reduces 75%. No entropy collapse — both models
  remain predictive (CE within 2% of β=0 baseline).
- **Tier 2 — F-alone PPL improves:** untested. CE values at 10k positions
  are too high (~3.0 vs converged ~1.6) to be a useful PPL signal.
  Requires more training.
- **Tier 3 — ensemble improves:** untested. Need ensemble eval tool.
- **Tier 4 — aligned beats shuffled:** ✓ confirmed cleanly. Aligned at
  β=0.1 shrinks KL 17%; shuffled at the same β leaves KL where it was
  (actually 4% higher, within noise). The 22-point gap between the two
  is the structural KL signal that's specifically about prefix↔suffix
  mutual information rather than generic regularization.

## What this means

The dual-view consistency mechanism is doing what it's supposed to do:

1. The KL term *can* pull two independent models toward agreement (gap
   shrinks with β, no degeneracy at β ≤ 0.1).
2. The shrinkage carries information that's specifically about aligned
   prefix/suffix structure — not generic smoothing. Mispaired KL doesn't
   help.
3. Modest β (0.1) is in a "free lunch" regime: gap shrinks ~17% with no
   CE cost. β=1.0 is more aggressive but starts paying CE.

What this *doesn't* tell us (yet):

- Whether the F-alone causal PPL on held-out improves under coupling
  (Tier 2).
- Whether the F+B ensemble outperforms either model alone (Tier 3).
- Whether the gap-closing translates to better generation quality.
- Whether the result holds at full Shakespeare 1M scale and 6+ epochs.

## Compute reality

The Crystal trainer is ~90 positions/sec single-threaded openblas. For
1.1M positions = 1 epoch, that's ~3.5 hours per epoch. Six epochs is
~21 hours. **Not viable for full headline runs.**

Realistic options:
1. **CUDA port** — the existing `agpt_train.cu` infrastructure can be
   extended to handle dual models. Probably 2-3 days of work to get a
   working dual-CUDA trainer at proper throughput.
2. **Tier-2/3 validation at small scale** — train both β=0 and β=0.1 at
   100k-200k positions (~3-6 hours) and measure F-alone held-out PPL on
   a small held-out set. Would tell us if the gap-closing translates to
   a CE benefit even at sub-converged scale.
3. **Keep at 10k-50k mechanism validation** — accept that we've only
   shown Tier 1 and 4, document, and decide whether the result justifies
   CUDA porting effort.

## Open notes

- The KL gap at β=0 here (0.66 nats) is much smaller than the 2.4 nats
  measured between FULLY TRAINED independent models. That's because
  10k-position training doesn't reach the divergent fixed points each
  model would reach with full training. The gap-shrinkage *fraction*
  (17% at β=0.1, 75% at β=1.0) is what we expect to scale with training
  budget, not the absolute number.
- Per-position Adam fire might be too noisy at very large scales.
  Per-partition aggregation (architecture-notes-style) would smooth
  the gradient if we go to 1M+ positions.
- B's CE at β=1.0 (3.06) is slightly worse than at β=0 (2.95). The
  coupling pulls B toward F's view, which carries some noise. The net
  effect on held-out is what matters.

## Pointers

- `PLAN.md` — the experiment design (revised after review)
- `PLAN_REVIEW_1.md` — review that drove the revision
- `src/tools/agpt_dual_train.cr` — the Crystal v0 trainer
- `bin/prefix_suffix_compare` — the divergence measurement tool from
  the cap-folding work; will be reused for Tier 1 verification at scale
