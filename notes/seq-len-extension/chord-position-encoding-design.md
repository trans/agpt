# Chord-based position encoding — design summary

**Date:** 2026-05-27.
**Status:** Mathematical design, not yet implemented. Open math questions
flagged for external review.

## Background and motivation

AGPT trains at small fixed depth `d` (e.g., 16). At inference we want
context lengths >> d. Standard RoPE breaks here: rotations R(p·ω) at
positions p > d are out-of-distribution, since training only saw
rotations at p ∈ [0, d).

Goal: a position encoding that stays in-distribution at long contexts
via wrap-around, while preserving the relative-offset semantics that
make RoPE work.

The harmonic-filter / chord direction is the candidate.

## The chord: definition

For each substring K in the corpus, with count(p) occurrences at
corpus position p, define:

  ẑ_K[j] := Σ_p count(p) · e^{i · p · ω_j}     for j = 0, 1, …, n_freq-1

where ω_j = 2π(j+1)/W (skip DC, j+1 starts at 1).

This is the DFT of K's mod-W position histogram. Equivalently, by the
convolution theorem, multiplication by `ẑ_K[j]` in the frequency
domain corresponds to circular convolution with K's position
histogram in the spatial domain.

Per-frequency interpretation:
- `|ẑ_K[j]|` ∈ [0, C_K]: coherence of K's distribution at this
  harmonic. C_K = Σ_p count(p) (the total mass).
- `arg(ẑ_K[j])` ∈ [0, 2π): circular mean of K's positions modulo
  the j-th period (W/(j+1)), weighted by count.
- α_K[j] := |ẑ_K[j]| / C_K ∈ [0, 1]: normalized coherence.

For typical substrings:
- Common high-mass substrings ('the'): histogram approximately
  uniform mod W ⇒ α_K[j] ≈ 0 for j > 0 (vectors cancel).
- Rare substrings concentrated at a few positions: α_K[j] ≈ 1 at
  every j (vectors align).
- Substrings tied to corpus structure (line breaks, scene markers):
  α_K[j] high at specific j matching the structural period.

## Operator forms considered

### Form A: additive bias to attention score

```
score(Q, K) += β · Σ_j Re[ẑ_K[j] · e^{−i p_Q ω_j}]
            = β · Σ_p count(p) · K(p − p_Q)
```

where K(Δ) := Σ_j cos(Δ · ω_j) is the implicit kernel.

Properties:
- Preserves K's full histogram (band-limited): multi-modal
  distributions show multiple peaks at p_Q values matching K's modes.
- Mass-weighted: bigger-mass modes give bigger peaks.
- Uniform K correctly gives zero bias (vectors cancel).
- Only learnable knob is β (scalar). No per-frequency learnability,
  no content-conditional shaping.

Variant A': per-frequency learnable γ_j:
```
score += Σ_j γ_j · α_K[j] · Re[ẑ_K[j] · e^{−i p_Q ω_j}]
```
Adds per-pair learnability and a coherence gate. Still additive,
still content-blind.

### Form B: phase-only rotation of K (RoPE-style)

```
K_pair_rotated[j] = K_pair_content · e^{i · arg(ẑ_K[j])} · α_K[j]
                   + K_pair_content · (1 − α_K[j])
score = standard Q · K_rotated bilinear (RoPE on Q's side)
```

Properties:
- Magnitude preserved per pair (pure rotation when α = 1).
- BILINEAR Q · K, so Wq/Wk can shape per-pair amplitudes
  (RoPE-equivalent).
- HISTOGRAM-COLLAPSE: bi-modal K's get represented as single-deltas
  at their circular mean per frequency. Two-mode {p=10, p=30}
  appears the same as single-delta at p=20 at the j=0 frequency.

Histogram collapse is a real and hard limitation if we want
multi-modal K's to remain distinguishable.

### Form C: chord-conditional key projection

```
K_pair_eff[j] = K_pair_content[j] + Σ_j W_chord_j · (ẑ_K[j] / C_K)
```

The chord enters K's effective projection via a learnable W_chord_j
per frequency per layer/head. Then standard Q · K bilinear with
RoPE on Q.

Properties:
- Preserves histogram (full ẑ_K, not normalized).
- Per-pair learnability (W_chord_j is learnable).
- Bilinear content × position interaction.
- Changes K's magnitude per pair (the chord is added, not rotated).
  Magnitude entanglement with content needs care (LayerNorm
  protection or similar).
- Adds parameters: W_chord_j is d_model × head_dim per (frequency,
  head, layer). For our small model (d_model=64, head_dim=16,
  4 heads, 2 layers, n_freq=16): ~130k extra params, ≈ doubles the
  base 108k model. Cost scales reasonably at larger models.

### Form D: stacked multi-resolution

Use multiple W values (e.g., W ∈ {32, 64, 128}) simultaneously,
each with its own chord channel and learnable weight. Different
lobe patterns per channel → redundancy. Combines with any of A/B/C.

Cost: ~3× chord storage per substring (still small), ~3× chord
bias compute per attention pair.

## The lobe issue

The implicit kernel K(Δ) = Σ_j cos(Δ · ω_j) from the bias form is
the partial Dirichlet kernel:

  K(0) = n_freq  (= peak)
  K(Δ)  oscillates: secondary lobes alternate sign, decay as 1/Δ
  K(W/2) ≈ 0 (anti-peak / global minimum within one period)
  K(W) = K(0) (periodic; wrap-around)
  Main lobe width ≈ 2·W/n_freq

For W=64, n_freq=16, sample values:
  Δ=0  → 16
  Δ=1  → 9.7    (main lobe)
  Δ=2  → −1
  Δ=3  → −3.8   (negative side lobe)
  Δ=4  → −0.2
  Δ=5  → 1.5    (small positive side lobe)
  Δ=8+ → noise around 0
  Δ=32 → ≈ −1   (deep anti-peak)
  Δ=64 → 16     (wrap)

The model gets a clear "K is at p_Q (or W away from p_Q)" signal at
the main lobes, and noisy values at intermediate offsets.

For form A this is uncorrectable (β just scales everything). For
form C, learned Wq/Wk amplitudes per pair can suppress or boost
specific frequencies, partially navigating the lobe structure (like
how RoPE's Q·K with per-pair learnable amplitudes manages its own
multi-frequency oscillations).

## Comparison to RoPE

RoPE applies rotation R(p·ω_j) per dim-pair. Properties:
- Magnitude preserved exactly (rotation is orthogonal).
- Bilinear Q · K shapes per-pair amplitudes through Wq/Wk learning.
- Position encoding is **single-positional**: each K has one position
  p_K. Multi-positional K (which is what makes the chord meaningful)
  doesn't fit RoPE's structure directly.
- Frequencies geometrically spaced (base=10000), covering periods
  from ~6 to ~20000. For small W (e.g., 64), the slow-period
  frequencies barely oscillate within the window — most pairs are
  "useless" at small W with default RoPE base. Setting RoPE base
  ≈ W concentrates frequencies inside the window.

DFT vs RoPE frequencies: DFT covers one octave (period W and its
sub-harmonics) uniformly within the band. RoPE covers many octaves
geometrically. For mod-W wraparound, DFT is the natural choice;
for fine-grained relative-offset at long ranges, RoPE.

## Open math questions

These deserve careful analysis before any implementation:

1. **Lobe artifacts in trained behavior.** Given the kernel K(Δ)
   structure described above, can a model with form C (chord-
   conditional projection) actually learn to suppress lobe noise?
   RoPE handles its own oscillations via Wq/Wk; the analogous
   mechanism in form C should work but hasn't been verified.
   Specifically: does the per-pair learnable W_chord_j give enough
   degrees of freedom to suppress messy lobe contributions?

2. **Magnitude-direction entanglement in form C.** Adding the
   chord to K's per-pair vector changes K's magnitude. Standard
   transformers do LayerNorm on K which would renormalize this.
   Does the chord information survive the LN? If LN normalizes
   the K vector, are we effectively losing the chord magnitude
   (= α) information?

3. **W choice and the bins-beyond-d issue.** If W > d (training
   depth), bins [d, W) of the chord are never probed during
   training (since p_Q only takes values [0, d) within the
   attention window). At inference at long context positions,
   positions ≡ values in [d, W) mod W give out-of-distribution
   chord-bias responses. Two solutions:
   - W = d (matches training, but reduces chord resolution to d bins)
   - Training augmentation: randomly offset p_Q during training to
     span all of [0, W)
   This is the same problem RoPE has with positions > d.

4. **Multi-modal vs unimodal distributions.** Form C with the
   full unnormalized chord should preserve multi-modal information,
   but the magnitude-direction entanglement might mask it. Form A
   (additive bias) preserves multi-modal cleanly but lacks
   learnable shaping. Open question: which form is closer to what
   the model can actually use?

5. **Content-conditioning depth.** In form C, K's effective
   projection depends on K's chord (positional fingerprint) plus
   K's content embedding. Both pass through Wq/Wk content-shaping.
   But the chord projection itself doesn't depend on Q's content.
   Is this analogous to RoPE (where rotation is content-blind but
   composes with content via Q·K) sufficient, or do we need a more
   integrated mechanism?

6. **The lobe trade-off with W and n_freq.** Main lobe width
   ≈ 2·W/n_freq. Side-lobe amplitude decays as 1/k. To get sharper
   main lobe: more frequencies. To get smoother side lobes:
   windowing/tapering (Hamming, Hann, Gaussian) at cost of wider
   main lobe. Is there an optimal choice?

## Practical concerns

Beyond the math:

a. Storage: ~2·n_freq floats per substring. For Shakespeare d=16
   (~10^6 substrings): 16 floats × n_freq=16 = 256 floats = 1KB
   per substring → ~1GB chord table. Manageable but non-trivial.

b. Inference plumbing: model needs Q's position p_Q to compute
   the chord-evaluated bias or to do the chord-conditional
   projection. Plus the per-substring chord lookup. Both feasible
   but require new attention machinery.

c. Training: every fire needs the chord data for the K substrings
   in its subtree. Per-substring lookup is fast (precomputed table).

d. The chord is a corpus-statistic. New corpora require fresh chord
   tables. Not a deal-breaker but worth noting.

## Suggested next experiment if pursued

If we proceed with form C:
1. Implement chord-conditional K projection in the PyTorch toy
   trainer first (small model, fast iteration).
2. Set W = d initially to sidestep the bins-beyond-d issue.
3. Train baseline (no chord) vs chord-augmented at multiple W's
   (e.g., 16 = d, 32, 64) to characterize the trade-off.
4. Probe per-pair learned amplitudes to verify the model is using
   the chord signal as expected.
5. Critical: test long-context inference (input lengths > d) to
   measure if chord-augmented degrades more gracefully than RoPE-only.

If those validate, port to the CUDA trainer for full-scale
evaluation.

## What this is NOT

- Not seq-len extension for free. The bins-beyond-d issue (point 3
  above) means the chord doesn't automatically generalize beyond
  training depth without additional handling.
- Not a selection mechanism. The chord doesn't pick which K's
  enter the K-set; that's a separate question (exposure).
- Not a replacement for RoPE. Likely composes alongside RoPE rather
  than replacing it.

## Related

- notes/seq-len-extension/exposure-vs-weighting.md — earlier reframe
  of the problem
- notes/seq-len-extension/harmonic-filter-asymmetric.md — original
  brief that motivated the wrong-shape asym-DFT operator
- rnd/harmonic-filter-diagnostic/ — corpus diagnostic showing chord
  separation by mass bucket
- rnd/harmonic-bias-prototype/ — null variance-check on additive-bias
  form (form A) at d=16 in-distribution; doesn't test the
  seq-len-extension use case
