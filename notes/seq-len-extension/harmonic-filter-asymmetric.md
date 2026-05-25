# Harmonic Filter v2 — Asymmetric K-at-p_Q with DFT frequencies

**Status:** diagnostic-validated 2026-05-25 on Shakespeare 1M, d=16. Supersedes the
chord-vs-chord formulation in [harmonic-filter-brief.md](harmonic-filter-brief.md).

## TL;DR

The chord formulation works, but the right operator is **asymmetric** (one position
vs. one distribution), not the symmetric chord-vs-chord we'd drifted into. Pairing
that with **DFT frequencies** (not RoPE's geometric ladder) gives 1.4–1.6 IQR-units
of on/off-path separation at the all-pairs-summed level — usable as an additive
attention bias.

## Background

The original concept was: "Q at concrete position p_Q queries K's position distribution.
If K's distribution has mass at p_Q (mod W), attention should fire." Somewhere in the
brief we morphed this into chord(Q) · chord(K) — distribution-vs-distribution similarity
— which is a different operator with different semantics. Stratified diagnostics
([rnd/harmonic-filter-diagnostic/stratified/](../../rnd/harmonic-filter-diagnostic/stratified/))
showed the chord-vs-chord formulation works only at low-mass K and only on the first
3 RoPE dim-pairs.

Returning to the original asymmetric formulation, and switching to DFT frequencies,
substantially sharpens the signal.

## The asymmetric operator

For each substring K we precompute the **chord**
```
z_K[j] = Σ_p count_K(p) · e^(i · p · ω_j)         (j = 0 .. n_pairs-1)
```
where `count_K(p)` is the number of times K appears starting at corpus position
≡ p (mod W). This is what's already stored in the position table.

At attention time, the harmonic bias for Q at concrete position p_Q attending to
substring K is:
```
bias(K, p_Q) = (1/C_K) · Σ_j Re[ z_K[j] · e^(-i · p_Q · ω_j) ]
             = (1/C_K) · Σ_j [ z_K_re[j]·cos(p_Q·ω_j) + z_K_im[j]·sin(p_Q·ω_j) ]
```
This is literally an inverse-DFT projection of K's distribution evaluated at p_Q.
With `ω_j = 2π(j+1)/W`, j=0..n_pairs-1, we get the first `n_pairs` non-DC harmonics
of K's W-periodic distribution.

**Semantics:** for an on-path pair (Q descends from K starting at corpus position p),
one of K's counts lives at bin `p mod W`, so K's bin at p_Q has mass ≥ 1. The bias
elevates above the off-path baseline by ≈ 1/C_K. For random off-path pairs, the bias
is centered around 0 (DC was dropped).

## Diagnostic results (Shakespeare 1M, d=16)

Aggregate ASYM separation (on-median − off-median, in pooled-IQR units):

| Config             | mass=2-9 | 10-99 | 100-999 | ≥1000 |
|--------------------|----------|-------|---------|-------|
| RoPE HD=16 (orig)  | 0.85     | 0.25  | 0.02    | 0.02  |
| DFT HD=16 W=32     | 1.38     | 0.53  | 0.18    | 0.02  |
| **DFT HD=16 W=64** | **1.43** | 0.50  | 0.17    | 0.04  |
| DFT HD=16 W=128    | 1.15     | 0.34  | 0.11    | 0.03  |
| DFT HD=16 W=256    | 0.74     | 0.21  | 0.08    | 0.01  |
| DFT HD=24 W=64     | 1.55     | 0.62  | 0.20    | 0.03  |
| DFT HD=32 W=64     | 1.54     | 0.73  | 0.23    | 0.03  |
| **DFT HD=48 W=64** | **1.60** | **0.88** | **0.27** | 0.04 |

Per-dim-pair (DFT W=64, mass=2-9): every dim-pair contributes meaningfully
(0.50–0.54 each) and they sum coherently. With RoPE only pairs 0–2 carried signal;
pairs 3–7 were dead.

Raw run logs:
- `rnd/harmonic-filter-diagnostic/stratified/shake_*.txt`

## Why DFT beats RoPE here

RoPE's frequencies `ω_j = base^(-2j/HD)` are geometric, spanning many orders of
magnitude. They were designed for relative-position encoding across long sequences,
where each pair sees a different "speed". For Fourier-reconstructing a W=64-periodic
distribution, this is wasteful:

- Pairs 0–2 happen to land near W=64 frequencies, so they pick up the signal.
- Pairs 3–7 have such long periods (10K–700K positions) that K's distribution looks
  essentially flat in their basis — they contribute noise.

DFT frequencies `2π(j+1)/W` tile the W cycle uniformly. Every dim-pair sees a
distinct harmonic of the period-W signal. They reconstruct the distribution rather
than just sampling near-zero of slow rotations.

## Architectural shape

Don't replace RoPE — add the harmonic bias as a separate term:
```
attention(Q_i, K_j) = (Q_i · K_j_rope) / sqrt(d)        ← unchanged
                    + β · harmonic_bias(K_j, p_i)       ← new additive term
```
where `harmonic_bias` uses the asymmetric DFT operator above. β is a learnable
scalar (per head). This way RoPE keeps doing what it does (relative-position
encoding for content-based Q·K), and the harmonic bias adds a position-distribution
match score on top.

The chord `z_K` is per-substring (one vector per trie node), not per-token. So we
look it up once per K node, not per token. Storage: `n_pairs × 2` floats per node.
At HD=48 (24 pairs), that's 48 floats per K node.

Per attention pair compute: `n_pairs` cos + sin + 2 mul-adds. At HD=48, ~96 flops
on top of the normal Q·K compute. Cheap.

## Open design choices

1. **W = 64 or 32.** Both perform near-best; W=64 slightly edges W=32. W=64 matches
   the position-table window we're already using; default to W=64.

2. **HD = 32 vs 48.** HD=32 saturates at mass=2-9 (1.54 vs 1.60); HD=48 keeps
   improving mid-mass buckets (0.73 → 0.88 at mass=10-99). Lean toward HD=48 unless
   compute matters.

3. **β init and learning.** Suggest β=0 init (bias is off, model learns to use it),
   one β per head. Could also try β learnable but globally shared.

4. **What does the model do with this for high-mass K?** Signal is ~0.04 IQRs at
   mass≥1000. Effectively neutral — the model can either ignore it (β stays small
   for those K) or use it as a tiebreaker. Should not hurt.

## Open validation

- [ ] Repeat sweep on Gutenberg 5M to confirm pattern at larger corpus.
- [ ] Verify off-path baseline stays near zero at scale (it did on Shakespeare).
- [ ] Confirm 1/C_K elevation scaling holds.
- [ ] Off-baseline collapse from 4.67 → -0.33 in RoPE→DFT switch matches DFT theory:
      each non-DC bin has mean zero over random positions.

## What this changes about the brief

The current [harmonic-filter-brief.md](harmonic-filter-brief.md) committed to E4
(chord-chord with depth-shift) and E4-norm as the two candidates. Both are
symmetric distribution-vs-distribution operators that the all-pairs diagnostic
suggested were dead but which stratified diagnostic shows are actually
mass-dependent. They still work for low-mass K, but the asymmetric K-at-p_Q
formulation dominates them at every mass bucket once paired with DFT frequencies.

Recommend: when the implementation plan ([harmonic-filter-plan.md](harmonic-filter-plan.md))
moves to kernel work, build the asymmetric DFT operator first; treat chord-vs-chord
as a fallback if asymmetric underperforms in training.

## Related

- [harmonic-filter-brief.md](harmonic-filter-brief.md) — original chord-vs-chord brief
- [harmonic-filter-plan.md](harmonic-filter-plan.md) — implementation plan (needs update)
- [position-distributions-plan.md](position-distributions-plan.md) — broader project context
- `src/tools/harmonic_filter_stratified.py` — diagnostic with --freq-mode dft and --window
- `rnd/harmonic-filter-diagnostic/stratified/` — raw run outputs
