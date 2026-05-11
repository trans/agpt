# Sliding-Window AGPT: Decoupling Tree Depth from Sequence Length

**Origin:** user design writeup, 2026-05-11. Supersedes the partial
treatment in `notes/agpt/shared_key_rope.md` by addressing the
which-trie-node-for-which-position reconciliation problem the earlier
write-up glossed over.

## Core Insight

Current AGPT implicitly couples:

    tree depth d == sequence length seq_len

because trie depth acts as the temporal coordinate.

Each node at depth `k` implicitly corresponds to token position `k`.

This works for standard AGPT, but prevents attention/context lengths
beyond the trie depth.

---

## The Problem

AGPT currently organizes computation around shared prefix nodes:

    h_n

where:
- `n` is a trie node
- not a sequence position

This is fundamentally *prefix-centric*, not *position-centric*.

The same node may represent many corpus occurrences.

As a result:

    node state == sequence state

and therefore:

    depth == time

This explains:
- why AGPT trains efficiently
- why seq_len is tied to depth
- why looping the tree caused modulo/phase artifacts

---

## Proposed Solution: Sliding Window AGPT

Instead of requiring the trie to span the entire sequence length:

    depth = seq_len

we treat the trie as a reusable *local distribution model*.

The trie only models local branching statistics over depth `d`.

We then slide this local model across a larger sequence window.

Example:

    window 0:  0..d-1
    window 1:  1..d
    window 2:  2..d+1
    ...

Thus:

    depth != global sequence position

The trie remains local while sequence length becomes externalized.

---

## Structural vs Positional State

This introduces a critical distinction.

### 1. Structural Node State

Persistent shared trie representation:

    h_n

Represents:
- prefix statistics
- branching structure
- learned local latent state

Shared across all corpus occurrences.

---

### 2. Positional Activation State

Dynamically constructed sequence representation:

    h_p

where:
- `p` is global sequence position

This is constructed from overlapping sliding-window activations.

Thus:

    node state GENERATES sequence state

rather than:

    node state IS sequence state

---

## Window Contributions

Each sliding window produces activations:

    h_(w,j)

where:
- `w` = window start
- `j` = local depth inside window

Global position is:

    p = w + j

Therefore each global position receives contributions from multiple
windows.

Example:

    window 0 depth 3 -> position 3
    window 1 depth 2 -> position 3
    window 2 depth 1 -> position 3
    window 3 depth 0 -> position 3

All of these contribute to the final sequence representation at
position 3.

---

## Pooling into seq_len K/V Rows

To recover exactly `seq_len` attention rows:

Pool all activations mapping to the same global position.

Example:

    h_p = Pool( h_(w,j) such that w+j=p )

Simplest implementation:

    h_p = average(all contributors)

Possible weighted pooling:
- uniform average
- entropy-weighted
- mass-weighted
- coverage-normalized
- learned gating

Then:

    K_p = RoPE(W_K h_p, p)
    V_p = W_V h_p

This yields exactly one K/V row per sequence position.

---

## Key Architectural Shift

Current AGPT:

    trie depth == usable context

Sliding-window AGPT:

    trie depth == local receptive field
    seq_len == span of overlapping positioned activations

Long-range attention no longer comes from deeper trees.

Instead:
- trie provides compressed local structure
- sliding overlap reconstructs sequence state
- attention operates over pooled positional activations

---

## Why Looping Failed

Looping reused trie topology temporally without explicit phase
coordinates.

Thus the model could not distinguish:

    depth 3 in window 0

from:

    depth 3 in window 128

These collapsed together, causing:
- modulo artifacts
- phase aliasing
- unstable identity regions

Sliding windows + global positional encoding solve this by separating:
- local structure
- global sequence phase

---

## Coverage Count Issue

Because windows overlap, interior positions receive more contributions.

Without normalization, these positions become overweighted.

Possible correction:

    contribution_weight = 1 / coverage_count(position)

or normalize by:
- total contributing mass
- total activations
- pooled attention energy

---

## Clean Mental Model

Trie:
    compressed reusable local structure engine

Sliding windows:
    sequence reconstruction mechanism

Attention:
    long-range interaction mechanism

The trie no longer directly represents time.

Instead:
- trie encodes local corpus distributions
- sequence state emerges from overlapping positioned activations
- attention operates over these reconstructed sequence representations

---

# Appendix — Implementation Outline (v1)

This appendix sketches the minimal v1 — uniform-average pooling,
coverage-normalized, two-stage attention that reuses existing AGPT
chunk machinery. Added by Claude 2026-05-11 as part of the same
discussion that produced the main text above.

## Stages

```
1. Per-window trie forward pass (existing AGPT chunk machinery, unchanged)
   For each corpus position w in [0, N-1]:
     - Walk the trie's d-window starting at w → ~9 radix nodes (radix-compressed)
     - Forward pass produces d position activations: h_(w, 0), ..., h_(w, d-1)
     - Each h_(w, j) is the residual stream value after the chunk's
       internal layers, at within-window depth j

2. Pooling to sequence positions
   For each global position p in [0, N-1]:
     - Gather contributors {(w, j) : w + j = p and 0 ≤ j < d}
     - h_p = mean({h_(w, j) for each contributor}) / 1
       (no extra normalization needed — mean already covers it)
     - For edge positions: contributor count is min(p+1, d, N-p)
     - Concretely: position p receives contributions from windows
       (p, 0), (p-1, 1), ..., (p-d+1, d-1)

3. Sequence-level attention
   With seq_len positions of pooled state h_p:
     - K_p = RoPE(W_K h_p, p)
     - V_p = W_V h_p
     - Q_q = RoPE(W_Q h_q, q)
     - Standard softmax attention over seq_len K/V rows
     - Apply for each transformer layer at this stage

4. Output / loss
   At each output position q:
     - predict next-token distribution
     - target = trie's distribution at q's terminal-attribution node
       (the deepest radix node whose path terminates at q,
       from the Phase 0 position→node map)
     - loss = KL(target, predicted) OR CE on the actual next char
```

## What stays from current AGPT

- Trie construction (same `bin/agpt_build_radix_corpus` output)
- Per-chunk forward pass through transformer layers
- AGPT-style trie-distribution targets at each position
- Mass-weighted loss scaling

## What's new

- The pooling step (between chunk-level forward and sequence attention)
- A second-stage attention pass over pooled sequence-position
  representations
- Position→contributors index (≈ inverse of the Phase 0 position→nodes
  map; can be derived from it in O(N) at startup)

## Pooling-step memory accounting

For Gutenberg 5M with d=16, sequence-train-window of S=32, d_model=64:
- Per chunk: d × d_model = 16 × 64 = 1024 floats of activation
- Per training step: walk S chunks → S × d × d_model = 32 × 16 × 64 =
  32768 floats of contributor activation, ~ 128 KB
- After pooling: S × d_model = 32 × 64 = 2048 floats, ~ 8 KB
- Negligible at any scale we care about

## Gradient flow

Under uniform-average pooling with c_p contributors at position p:
- `h_p = (1/c_p) Σ_(w,j) h_(w,j)`
- `dL/dh_(w,j) = (1/c_p) · dL/dh_p`

So the gradient from sequence-position p splits 1/c_p ways across its
contributors, then flows backward through each contributor's chunk
forward pass. Numerically: each chunk forward pass gets ~1/d of the
attention-level gradient (1/16 for d=16), spread across its d
positions. Probably fine; worth checking that magnitudes don't shrink
into numerical-noise range during early training.

## Compute scaling

Per training step (window of S sequence positions):
- S chunk forward passes (each d positions deep, d² attention) = S × d³
- One sequence-level attention over S positions = S²
- For d=16, S=32: chunk-side = 32 × 4096 = 131k; seq-side = 1024.
  Chunk-side dominates by 100×, which is fine — same regime as
  current AGPT.

The decoupling lets us push S up without rebuilding the trie, so
S=128 or S=512 at d=16 becomes feasible (the trie compute stays the
same; only the cheap sequence-attention grows).

## Recommended v1 starting point

1. Implement pooling step (CPU, then move to CUDA) on inference path
   first. Take an existing AGPT model, run chunks for a small eval
   window (say 200 corpus positions), pool, run sequence-level
   attention, score PPL.
2. Compare to PPL@d=16 baseline (8.01 on Gutenberg).
3. If sequence-level attention shows lift, train end-to-end with
   chunk-forward + pool + seq-attention + loss.

The inference-first prototype keeps risk low: it's testing whether the
sequence-level attention layer can extract useful signal from pooled
chunk activations *before* committing to training infrastructure.

## Open design questions deferred to v2

- Whether to learn the pooling weights (gating) vs. fixed uniform
- Whether trie chunks should run their *own* internal attention or be
  reduced to bag-of-position activations
- Multi-layer composition: does the sequence-level attention itself
  need multiple layers, or is one enough on top of the chunk-internal
  layers?
- Backward-tree (suffix) contributions — can the same pooling scheme
  combine forward-trie and backward-trie activations into one h_p?
  (Touches the unified-DAG line from
  `project_predictive_certainty_weighting.md`.)
