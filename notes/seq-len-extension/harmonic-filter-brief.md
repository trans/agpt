# Harmonic-Filter / Chord-Correlation RoPE for AGPT — Brief

**Date:** 2026-05-25 (revised after first round of cross-AI review)
**Purpose:** Self-contained writeup of the multi-position-encoding idea for AGPT, plus three design variants we want help choosing between.

The document has two parts:

1. **The brief** — the whole idea, math, and motivation. Assumes some transformer/RoPE background but no AGPT-specific knowledge.
2. **The three design variants** — same goal, different mechanisms. Where reader input is wanted.

---

# Part 1: The Brief

## Background

### AGPT in one paragraph

AGPT (Aggregated-Gradient Pretraining) is a character-level transformer trainer that uses a **radix trie** over the corpus as its core data structure. Each unique substring of length 1..d (typically d=16) in the corpus becomes a **node** in the trie. During training, each node "fires" once per epoch: a forward+backward pass that processes the node's local context (an attention window over its d-deep trie path). Gradients aggregate across all corpus occurrences of the node automatically, because all occurrences route through the same node identity.

Concretely: in our Gutenberg 5M-character corpus at d=16, we have ~7M unique substrings (radix nodes). Each fire processes a small chunk of the trie. The transformer's attention operates within the d-depth window: each query at depth k attends to the trie ancestors at depths 0..k-1. So **seq_len = d = 16**. The model's effective context is locked to the trie depth.

### Standard RoPE in one paragraph

RoPE (Rotary Position Embedding) gives a transformer's attention a relative-position signal *without* explicitly storing positions. For each token at sequence position p, the query and key vectors are rotated by an angle `θ(p, i) = p / base^(2i/HD)` for each dim-pair `i` of the head_dim. After rotation, the attention dot product `Q·Kᵀ` depends only on `(p_q - p_k)` — the relative distance between Q and K. Different dim-pairs encode position at different frequencies (high-frequency for nearby positions, low-frequency for far positions). The model sees `(p_q - p_k)` at multiple scales simultaneously.

### What we want to do

**Extend the model's effective attention beyond d**, without growing the trie deeper (which has its own problems: most trie leaves past d=16 are singletons, training signal degrades). Each substring occurs at potentially many corpus positions, and we want the model to use that multi-position information in attention.

## The core mathematical object: the chord

For each substring (radix node) `i` and each RoPE dim-pair `j` with frequency `ω_j`, we can summarize the substring's distribution over corpus positions (modulo a chosen window size W) as a **complex chord**:

```
z[i, j] = Σ_p count(p) · e^{i p ω_j}
```

where the sum runs over all positions where substring `i` occurs in the corpus (modulo W). This is the position distribution's Fourier coefficient at frequency `ω_j`.

The chord has two parts that should be kept conceptually separate:

```
C[i]    = Σ_p count(p)         total mass / frequency of substring i
|z[i,j]| / C[i]    = r[i,j]    coherence at frequency ω_j  ∈ [0, 1]
z[i,j]  / |z[i,j]| = direction angle of average rotation at frequency ω_j
```

So we can decompose: `z[i,j] / C[i] = direction × r`. This is the **un-normalized, count-divided chord**:
- `direction` = unit vector pointing at the substring's average position-angle for this frequency
- `r` = 1 for sharp distributions (substring always at one position), → 0 for uniform (substring everywhere)

**Important:** the "chord" in our discussion always means `z/C` (= direction × r), not raw `z`. Raw `z` would let frequent substrings dominate purely by count, conflating *frequency* with *coherence*. The `z/C` form separates them: `direction` captures *where* the distribution lives, `r` captures *how concentrated* it is.

## The failed first attempt: dist-rope

We tried to encode each substring's position distribution into RoPE by replacing the per-position rotation with a per-substring "average rotation" using `z/C`:

```
eff_cos[i, j] = Re(z[i,j] / C[i])    (=  direction × r, real part)
eff_sin[i, j] = Im(z[i,j] / C[i])    (=  direction × r, imag part)
```

In the kernel, **both Q and K** used these eff_cos/eff_sin tables (indexed by substring_id) instead of the standard position-indexed cos/sin caches.

**Result: training loss regressed 18-30% on Shakespeare L=2 100SE.**

### Why dist-rope failed

Two compounding issues:

1. **r² magnitude collapse.** Applying z/C as the rotation on BOTH Q and K means both sides get scaled by `r`. The attention dot product gets scaled by `r²`. For high-mass substrings (which have `r ≈ 0`), this drives the dot product to noise — common substrings get effectively zero attention signal, exactly where AGPT's statistical evidence is strongest. The `r²` was the killer.

2. **Broken relative-position semantics.** Standard RoPE works because both Q's rotation θ(p_q, i) and K's rotation θ(p_k, i) live in the same position coordinate, so `Q·K` after rotation depends on `(p_q - p_k)`. dist-rope rotated both by per-substring summary angles that don't share any coordinate system, so the relative-position structure was destroyed.

The first issue (`r²` collapse) is the more fundamental one. The second is a symptom of the broader "use chord as rotation on both sides" mistake.

## What we want from a fix

A design that:

1. **Avoids `r²` collapse.** Either don't put `r` on both sides, or move position info out of the rotation entirely.
2. **Preserves standard RoPE's relative-position attention.** Don't break what already works.
3. **Encodes multi-position information** in a form the model can use.
4. **No multi-layer leak.** Positional encoding should be layer-independent (each layer computes its own Q/K normally; position info is applied on top).

The three variants in Part 2 take different paths to these goals.

---

# Part 2: The Design Variants

We have three candidate designs. They share the same underlying chord computation `z/C` per (substring, dim-pair), but differ in **where the chord lives** in the attention mechanism.

| variant | Q rotation | K rotation | position bias | how multi-position info enters attention |
|---|---|---|---|---|
| **H1: Chord-rotate K, standard Q** | standard RoPE at q | rotated by chord direction × r | — | inside the QK dot product, via K's rotation |
| **E3: Chord-chord scalar bias** | standard RoPE (chunk_local_depth) | standard RoPE (chunk_local_depth) | β · Σ_j Re(conj(z_Q/C_Q) · z_K/C_K) | additive logit bias, computed from both chords |
| **E4: Chord-chord with depth shift** | standard RoPE (chunk_local_depth) | standard RoPE (chunk_local_depth) | β · Σ_j Re(conj(z_Q/C_Q) · z_K/C_K · e^{iΔω_j}) where Δ = depth_q - depth_k | E3 + AGPT-native depth correction |

## Variant H1: Chord-rotate K, standard Q (the "harmonic filter")

### Description

The original idea: a matched filter / tuning-fork resonance design.

- **K rotation:** apply `direction × r` (the chord `z/C`) as a 2D rotation-and-scale on K's hour-hand dim-pairs.
- **Q rotation:** standard RoPE at Q's "current position" q (with `q` in the same coordinate system as K's chord).

Attention dot product (real part, per dim-pair):

```
Re(Q*·K) = Re(e^{-i q ω} · z_K/C_K · K_content)
         = Σ_p (count(p)/C_K) · cos((p - q) ω)
```

When `q = p_k` for some historical position p_k of the K-substring, that term contributes `cos(0) = 1` (positive spike). Other terms oscillate and average toward zero. **The dot product spikes when Q's current position matches one of K's historical positions.** Hence "matched filter" — K's hour-hand acts as a phase fingerprint of its position distribution; Q acts as a sharp probe at its current position.

### Why this avoids dist-rope's `r²` collapse

Only K is scaled by `r`. Q is unit-magnitude. So the dot product is scaled by `r` (not `r²`). For high-mass substrings with `r ≈ 0`, K's hour-hand contribution → 0 — they're effectively silent on hour-hand attention. This is the automatic α-gate.

The Q side keeps unit magnitude, so for sharp K's, full attention strength is preserved.

### The open question: what is Q's "current position"?

Standard RoPE for Q requires Q to have a single position. In AGPT, Q is a multi-position trie node (just like K is). What `q` do we use?

Options we've considered:
- **A. Chunk-local depth** (Q's depth in the trie path, 0..d-1). Cheapest implementation (zero changes — this is what AGPT uses today). But Q's coordinate `[0, d)` doesn't match K's chord coordinate `[0, W)`. The matched-filter math doesn't strictly apply; the model has to find emergent use of a coordinate-mismatched signal.
- **B. Sample Q's position per fire.** For each query in a chunk, sample one of the substring's corpus positions; use that for Q's RoPE. Math works as designed. Adds stochastic variance to training.
- **C. Per-fire chunk anchor.** Sample ONE corpus position per chunk; Q at depth k uses `(anchor + k) mod W`. Math works, preserves within-chunk relative-position structure, deterministic given the anchor sample.
- **D. Use trie-depth coordinate for both Q and K's chord.** Doesn't work — each substring has only one trie depth, so the chord becomes a delta, no multi-position info encoded.

None of A-C is obviously correct. The other AI reviewer suggested the framing: "Which occurrence of this node is this fire standing in for?" — making B or C the natural answer. But variance and anchor-selection complicate the design.

### Implementation cost

If A: trivial (kernel-only). If B or C: per-fire RNG + chunk metadata extension + sampler tables. ~half-day to a day.

### Verdict

Elegant matched-filter formulation. Open question on Q's coordinate is the load-bearing uncertainty. If we settle on A (cheapest), we're betting the model finds use of a math-mismatched signal. If B or C, we add stochastic variance or per-fire sampling logic.

## Variant E3: Chord-chord scalar bias

### Description

Don't put the chord inside the RoPE rotation at all. Keep standard RoPE on both Q and K (preserving relative-position attention exactly as today). Add a **separate scalar bias** to the attention logit, computed from both chords:

```
attn_logit(Q, K) = (Q · K) / sqrt(HD) + β · phase_correlation(Q_node, K_node)

phase_correlation(Q_node, K_node) = Σ_j Re(conj(z_Q,j / C_Q) · z_K,j / C_K)
                                  = Σ_j Re(conj(direction_Q,j) · direction_K,j) · r_Q,j · r_K,j
```

The phase correlation is high when:
- Q's chord direction at frequency `ω_j` aligns with K's chord direction at the same frequency (both have similar position-distribution structure)
- BOTH r values are high (both distributions are sharp / concentrated)

### Why this works (no `r²` collapse here, despite r_Q · r_K)

Wait — there IS an `r_Q · r_K` factor. Why isn't this the same problem as dist-rope?

Crucially, the `r_Q · r_K` factor scales only the BIAS term, not the QK dot product. The content-attention term `Q · K / sqrt(HD)` is untouched and uses standard RoPE. For high-mass substrings, the bias term goes to zero — but the content attention still works. The model attends by content, not by position-distribution, for those substrings. That's the desired behavior.

In dist-rope, the `r` factor scaled the ONLY attention signal. There was no content-attention fallback. That's why dist-rope failed and E3 doesn't.

### Architectural advantages

1. **Decouples content attention from positional attention.** Content rotation is unchanged; positional info is a separate channel the model can weight via β.
2. **No Q-position decision needed.** Both Q and K are multi-position; the cross-correlation between their two chords is a well-defined scalar. The "Which occurrence is the fire standing in for?" question goes away.
3. **Symmetric in Q and K.** Both contribute their chords; no asymmetry to design around.
4. **Falls back gracefully.** If β = 0, exactly equivalent to baseline. Easy to ablate.

### What E3 captures

Pairs of substrings whose position distributions are correlated (occur at the same corpus phases). Useful when there's macro-grid structure: indentation, paragraph rhythm, lists, code formatting.

What E3 does NOT capture (the subtlety E4 addresses):

In AGPT, when Q is at trie depth k_q and K (an ancestor) is at trie depth k_k, their relative corpus offset is **k_q - k_k** for any specific occurrence. Q and K shouldn't appear at the *same* corpus positions — they should appear at corpus positions separated by `Δ = k_q - k_k`. E3 doesn't account for this offset, so it actually computes a slightly wrong correlation (it asks "do they live at the same phases?" when it should ask "do they live at phases differing by Δ?").

### Implementation cost

- Precompute z_substring (one complex value per substring per hour-dim-pair). Same memory as the original chord precompute: ~225 MB on Gutenberg.
- Add one term to the attention-score kernel: compute `phase_correlation(Q, K)` per (Q, K) pair, add to attention logit with coefficient β.
- Compute cost: O(T_q × T_k × n_hour_dims) extra ops per chunk. Trivial.
- Total: ~half-day.

## Variant E4: Chord-chord with depth shift

### Description

E3 plus the AGPT-native fix for the expected relative offset.

```
phase_score(Q, K) = Σ_j Re(conj(z_Q,j / C_Q) · (z_K,j / C_K) · e^{i Δ ω_j})
                    where Δ = depth_q - depth_k
```

The shift `e^{i Δ ω_j}` is applying RoPE rotation `Δ` to K's chord before correlating with Q's. Equivalently: "shift K's distribution forward by Δ, then compare to Q's distribution."

If Q at depth k_q and K at depth k_k typically appear at corpus positions `(p, p - Δ)` (which is the case for ancestor-descendant pairs in any given path), then the shifted correlation peaks. If their distributions don't align at the expected offset, the correlation is low.

This is a **distributional relative-position kernel** — the AGPT-native analog of standard RoPE's `(p_q - p_k)` signal, but operating on distributions rather than individual positions.

### Why E4 should outperform E3 on AGPT specifically

Standard attention semantic: "tokens at the expected relative offset get more attention." E3 asks "do they live at the same phases?" which only fires when Q and K happen to occur at coincident phases. E4 asks "do they live at phases offset by Δ?" which fires when Q and K's distributions are *compatible with the ancestor-descendant structure of the trie path*.

For AGPT's typical attention pattern (query attends to its ancestors), E4 produces a meaningful signal where E3 produces near-zero (because Q's positions are typically Δ characters AFTER K's positions, not at the same positions).

### Implementation cost

E3 plus one rotation per (Q, K, hour-dim). Δ is already in chunk metadata (depth difference between Q and K). Negligible additional compute.

Total: ~half-day + a few extra hours over E3.

## Summary table

| concern | dist-rope (failed) | H1 (chord rotation) | E3 (chord-chord bias) | E4 (chord-chord + depth shift) |
|---|---|---|---|---|
| Avoids `r²` collapse | ✗ | ✓ (only K scaled by r) | ✓ (separate bias channel) | ✓ (same as E3) |
| Preserves relative-position attention | ✗ | partially (depends on Q's q) | ✓ (untouched) | ✓ (untouched) |
| Q-position question | n/a (broken) | **open** — main risk | **avoided** | **avoided** |
| Symmetric in Q/K | ✗ | ✗ | ✓ | ✓ |
| AGPT-native (uses trie structure) | ✗ | ✗ | partial | **yes** (depth shift uses trie depth) |
| Multi-layer leak | n/a | none (chord is positional) | none | none |
| Implementation cost | done (regressed) | ~day if Q option B/C | ~half-day | ~half-day + few hours |
| Ablation surface | n/a | enable / disable | β coefficient, easy ablation | β coefficient + Δ on/off |

## Our current lean

E4 looks like the strongest candidate:

- Avoids the Q-position problem entirely (both Q and K are multi-position; the cross-correlation is well-defined without picking Q's "current position")
- Adds positional info as a side-channel rather than mixing it into the QK rotation (no risk of corrupting content geometry)
- Uses information AGPT already has (trie depth difference Δ) to apply the right offset
- Cheapest to implement (~half-day)
- Easiest to ablate (β coefficient, set to 0 = baseline)

But we want input on:

1. **Is E4 actually the right framing for the problem?** Or does H1's elegance (matched-filter as a fundamental mechanism) carry more value than E4's pragmatism?
2. **Does E3-vs-E4's depth-shift matter as much as we think?** Empirically, would E3 alone capture enough signal that the depth correction is icing, or is the shift load-bearing?
3. **For H1 specifically, is the Q-position question (A/B/C) resolvable in a principled way, or is it inherently a design compromise?**
4. **Are there better variants we haven't considered?** The first-round review proposed E1 (asymmetric Q-K), E2 (chord as harmonic gating), and E3/E4. Is there an E5?
5. **What signal would tell us E4 isn't working as expected?** What's the diagnostic that would distinguish "model learned positional structure" from "model ignored the bias term"?

## Context for what we already have

- Per-substring position table (mod W counts): built, on disk for Shakespeare and Gutenberg
- Substring catalog with stable IDs
- dist-rope's eff_cos/eff_sin precompute (= z/C per substring per dim-pair): already implemented in `src/cuda/agpt_position_data_io.cuh`. This IS the chord we need for any of the three variants above.
- CUDA kernel that uses substring_id-keyed cos/sin lookup (currently does the broken dist-rope thing — needs to be rewired for whichever variant we pick).

The implementation is small once we pick the variant.
