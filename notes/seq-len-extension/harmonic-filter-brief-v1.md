# Harmonic-Filter RoPE for AGPT — Brief + Open Question

**Date:** 2026-05-25
**Purpose:** Self-contained writeup of the harmonic-filter RoPE idea for AGPT, plus a single open architectural question we want help thinking through.

The document has two parts:
1. **The brief** — the whole idea, math, and motivation. Assumes some transformer/RoPE background but no AGPT-specific knowledge.
2. **The open question** — the Q-position problem and resolution options. Where the reader's input is wanted.

---

# Part 1: The Brief

## Background

### AGPT in one paragraph

AGPT (Aggregated-Gradient Pretraining) is a character-level transformer trainer that uses a **radix trie** over the corpus as its core data structure. Each unique substring of length 1..d (typically d=16) in the corpus becomes a **node** in the trie. During training, each node "fires" once per epoch: a forward+backward pass that processes the node's local context (an attention window over its d-deep trie path). Gradients aggregate across all corpus occurrences of the node automatically, because all occurrences route through the same node identity.

Concretely: in our Gutenberg 5M-character corpus at d=16, we have ~7M unique substrings (radix nodes). Each fire processes a small chunk of the trie. The transformer's attention operates within the d-depth window: each query at depth k attends to the trie ancestors at depths 0..k-1. So **seq_len = d = 16**. The model's effective context is locked to the trie depth.

### Standard RoPE in one paragraph

RoPE (Rotary Position Embedding) gives a transformer's attention a relative-position signal *without* explicitly storing positions. For each token at sequence position p, the query and key vectors are rotated by an angle `θ(p, i) = p / base^(2i/HD)` for each dim-pair `i` of the head_dim. After rotation, the attention dot product `Q·Kᵀ` depends only on `(p_q - p_k)` — the relative distance between Q and K. Different dim-pairs encode position at different frequencies (high-frequency for nearby positions, low-frequency for far positions). The model sees `(p_q - p_k)` at multiple scales simultaneously.

### What we want to do

**Extend the model's effective attention beyond d**, without growing the trie deeper (which has its own problems: most trie leaves past d=16 are singletons, training signal degrades). Specifically: each substring occurs at potentially many corpus positions, and we want the model to use that multi-position information in attention.

## The failed first attempt: dist-rope

We tried to encode each substring's *position distribution* into RoPE by replacing the per-position rotation with a per-substring "average rotation":

```
eff_cos[substring_id, dim_pair_i] = Σ_p p(p) · cos(p · ω_i)
eff_sin[substring_id, dim_pair_i] = Σ_p p(p) · sin(p · ω_i)
```

where `p(p) = count(p) / total_count` is the normalized weight of position p in the substring's distribution, and `ω_i = 1/base^(2i/HD)` is the dim-pair's frequency.

In the kernel, **both Q and K** used these eff_cos/eff_sin tables (indexed by substring_id) instead of the standard position-indexed cos/sin caches.

**Result: training loss regressed 18-30% on Shakespeare L=2 100SE.**

### Why dist-rope failed

Two compounding issues:

1. **Magnitude collapse.** For a substring with broad position distribution (e.g., "the" appearing roughly uniformly across all 64 W-window positions), the weighted sum `Σ count(p) · cos(p · ω)` averages to near zero because the unit vectors point in many directions and destructively interfere. The resulting `(eff_cos, eff_sin)` 2D vector has magnitude near zero, and when applied as a "rotation" to Q and K, it actually *shrinks* them toward the origin. Attention dot products become tiny noise.

2. **Broken relative-position semantics.** Standard RoPE works because both Q's rotation θ(p_q, i) and K's rotation θ(p_k, i) live in the same position coordinate, so `Q·K` after rotation depends on `(p_q - p_k)`. dist-rope rotated both by per-substring summary angles that don't share any coordinate system, so the relative-position structure was destroyed.

## The harmonic-filter idea

Re-frame the encoding as a **matched filter** (think holographic memory / tuning-fork resonance):

- **K side**: store each substring's position distribution as a phase pattern that encodes "I have been at positions p_1, p_2, p_3, ...". Formally, the un-normalized weighted sum `Σ count(p) · e^{j p ω_i} = (eff_cos, eff_sin)` is the chord — a 2D vector per (substring, dim-pair).
- **Q side**: standard RoPE at the query's actual current position q. A sharp single-frequency probe: `Q_i = e^{j q ω_i}`.

The attention dot product becomes:

```
Re(Q*·K) = Σ_p (count(p) / total_count) · cos((p - q) · ω_i)
        = E_p[ cos((p - q) · ω_i) ]
```

When `q = p_k` for some historical position p_k of the substring, that term contributes `cos(0) = 1` (a positive spike). Other terms contribute oscillating values that average toward zero. **The attention dot product spikes when Q's current position matches one of K's historical positions.** The model attends to substrings whose chord includes the current position.

For substrings with sharp distributions (mass=1 or low-mass), the chord has high magnitude in one direction (essentially standard RoPE at that one position). The matched-filter signal is strong.

For substrings with broad distributions (high-mass like "the"), the chord has low magnitude due to destructive interference. The α-gate is **automatic via the un-normalized formula** — the K vector's hour-hand dims naturally become small for high-mass substrings, suppressing their hour-hand contribution to attention.

### Why this should work (in principle)

Three architectural properties:

1. **No magnitude collapse on the Q side.** Q uses standard RoPE → unit-magnitude rotation, preserving Q·K dot product magnitudes for sharp-distribution K's.
2. **Magnitude-aware suppression on the K side.** Broad-distribution K's have small hour-hand magnitude → don't inject phase noise into attention.
3. **No multi-layer leak.** The chord is a positional encoding, not content sharing. Each layer still computes its own K = X·W_K per occurrence; the chord is just a different rotation applied on top. Layer-independent.

This contrasts with several alternatives we considered (shared-key RoPE, multi-slot sampling) which all have multi-layer leak issues because they share content across occurrences. The harmonic filter shares only positional information.

### Implementation cost

If the design works, implementation is cheap:
- The chord precompute (~225 MB chord_cos/chord_sin table on Gutenberg) is mathematically identical to dist-rope's eff_cos/eff_sin without normalization — we already have the Crystal precompute from dist-rope.
- CUDA kernel change: K rotation uses chord lookup (already wired for dist-rope), Q rotation uses standard RoPE cos/sin at current position. **The only real change vs dist-rope is which side uses which.**
- Total: ~half-day to a day of work.

---

# Part 2: The Open Question — Q's "current position" in AGPT

## The assumption the harmonic filter makes

The matched-filter math above assumes **Q has a single, well-defined "current position" q**. The query is at a specific point in the sequence; its rotation `e^{j q ω}` is a sharp probe at that exact frequency.

In a standard transformer, this is trivially true. Token at sequence position p has q = p.

## Why AGPT breaks this assumption

In AGPT, **each query is at a radix-trie node**, not at a specific corpus position. A query at "chunk-local depth k" (= depth in the current fire's trie path) represents a substring (a node X) that occurs at MANY corpus positions. Q is just as multi-position as K. There is no single "q" for the tuning fork.

What does Q's RoPE actually use currently? **Chunk-local depth** (0..d-1 = 0..15). This is a position-like quantity, but:
- It's NOT a corpus position
- It lives in coordinate `[0, d) = [0, 16)`
- K's chord lives in coordinate `[0, W) = [0, 64)` (corpus positions mod W=64)
- **These don't share a coordinate system.**

So the matched-filter math doesn't strictly apply. Q's "tuning fork" at angle `chunk_local_depth · ω` doesn't have any geometric reason to spike against K's chord at angles `corpus_position_mod_W · ω`. The two coordinate systems are unrelated.

## What this means practically

We have a design that works mathematically in the transformer setting. To deploy it in AGPT, we have to **decide what Q's "current position" means** in our setting. There's no obviously correct answer.

## Resolution options

| option | Q's position | implementation | semantic argument |
|---|---|---|---|
| **A. Ignore the mismatch** | chunk_local_depth (current) | zero change, kernel-only | The math is suspect but the model might still find useful signal. Pure empirical gamble. Cheapest. |
| **B. Per-fire random sample** | sample one of Q-node's corpus positions at the start of each fire | per-fire RNG + chunk metadata extension | "This fire treats this query as if its specific occurrence was at this corpus position." Stochastic. Variance across fires averages the distribution. |
| **C. Per-fire chunk anchor** | sample a single anchor corpus position for the whole chunk; Q at depth k uses `(anchor + k) mod W` | per-fire RNG + chunk metadata extension; queries inherit relative offset from anchor | "This fire anchors the chunk's path at this corpus position; queries within the chunk preserve relative positions to the anchor." Cleaner than B because preserves within-chunk relative structure. |
| **D. Rebuild chord using trie-depth coordinate** | chunk_local_depth (current) | rebuild chord precompute with depth coordinate | Each substring has a single fixed trie depth → chord becomes a delta → equivalent to standard RoPE. No multi-position info encoded. **Doesn't actually work.** |

## Specific concerns

### Option A (ignore mismatch)
The matched-filter spike (`cos(0) = 1` when `q = p_k`) won't fire because Q's coordinate and K's coordinate are different scales. What WILL happen empirically is unclear — the model could find emergent use of the signal, or could just see noise.

Empirical test would be cheap (CUDA-only change), but interpretability would be murky.

### Option B (per-fire random sample)
Mathematically clean: Q gets a real corpus position from its substring's distribution. Matched-filter math applies as designed.

Cost: per-fire RNG; sampler tables (alias method) for efficient sampling; chunk metadata gains an extra int per query.

Concern: variance. Two consecutive fires of the same node will sample different positions for Q → different rotation → different attention pattern → different gradients. The gradient becomes a noisy estimate of the "true" expected attention behavior. May converge slower or to a worse minimum than a deterministic Q rotation would.

### Option C (per-fire chunk anchor)
Mathematically clean AND preserves within-chunk relative position. Sample ONE anchor per fire (a corpus position consistent with the chunk's whole path); use `(anchor + chunk_local_depth) mod W` for each query.

Cost: per-fire RNG; one extra int in chunk metadata (the anchor); query positions computed as `(anchor + depth) mod W`.

Concern: how do we sample the anchor in a way that's consistent with the chunk's path? The chunk's path is a length-d sequence of characters; it appears at certain corpus positions. We'd sample one of those — but how to sample efficiently from a chunk-level set of positions (vs a node-level set)?

Sub-option: use the FIRST query node's positions as the anchor source (since the chunk's path starts at the first query node). One node's positions are already in the side-table.

### Option D (rebuild chord with depth coordinate)
Doesn't carry multi-position information. Each substring has exactly one trie depth (= its endpoint_depth). So the chord becomes a delta at that depth → equivalent to standard RoPE at the substring's fixed depth. No new signal.

## Sub-question: is there an option E we're missing?

We've thought about:
- **Use no Q rotation at all on hour-hand dims** (only K). Math: `Q·K = Q·(α · chord_direction · K)`. Q is unrotated → K's chord direction encodes the substring's average-position direction; Q·K depends on cosine alignment between Q's content and K's chord direction. **No matched-filter spike property at all** — just a content-vs-chord-direction similarity. Doesn't seem useful.
- **Use Q's content vector to compute its position** (project Q's content into a position guess). Adds learnable parameters; complicates the math.
- **Defer the decision** by trying the cheapest option (A) first, then escalating only if A is ambiguous.

## Concrete question for the reader

**Given the harmonic-filter design and AGPT's multi-position-query reality, which resolution is the right one to start with?**

- If A: are there theoretical reasons to expect the model to find useful signal despite the coordinate mismatch, or is this just hoping the model is forgiving?
- If B or C: which is the more principled choice, and what's the right way to sample the anchor/position?
- Is there an option E we haven't considered?

Subsidiary questions:
- Does the variance from stochastic Q sampling (option B/C) damage gradient quality enough to compromise the experiment? Is there a deterministic alternative that preserves coordinate alignment?
- Is the "match on corpus position mod W" semantic actually what we want? Or should the chord encode some other signal (e.g., distance distribution, depth distribution) where the matched-filter property naturally aligns with Q's available coordinates?

## Context for what we already have built

- **Per-substring position table** with mod-W counts: built, on disk for Shakespeare and Gutenberg
- **CSR-format inverse map** (corpus_position → node_id): trivial to derive from the position tables
- **dist-rope precompute** (eff_cos, eff_sin un-normalized): already implemented in `src/cuda/agpt_position_data_io.cuh`
- **CUDA kernel for chord-based K rotation**: already wired in `agpt_train.cu` (currently applies to BOTH Q and K — needs to be split)

The implementation is small once we know what Q should look like.

## Why this matters

If the harmonic filter works in AGPT, it gives us:
- Multi-position encoding without sequence-length growth
- No multi-layer leak
- No K matrix expansion
- ~half-day to a day to ship

If it doesn't, we want to know *why* — and the Q-position resolution is the main load-bearing design choice. Picking the wrong option could either invalidate the experiment ("we wired something subtly broken") or get us stuck somewhere between empirical success and theoretical clarity ("the model improved but we don't know which signal it's using").

That's what we're asking for opinions on.

