# Virtual-tree training — implementation plan

Branch: `agpt-virtual-tree`
Status: starting from current main (which has Lightning + BF16 +
mass=1 compact cache).

## Core idea

Extend training context past D* by attaching a new root-walk cycle at
each mass>1 leaf of the D*-trie. The trie itself stores only D* depth;
the virtual tree at attention time extends context to K·D* effective
depth. K/V storage doesn't grow — virtual positions reuse the real
D*-tree's cached K/V with a per-read RoPE rotation adjustment.

## Why the mass=1 skip composes cleanly

- Mass=1 leaves are single-corpus-suffix tails. No sibling exists, so
  extending a virtual cycle from there doesn't create a meaningful
  training target.
- Mass>1 leaves are depth-cap truncation points where multiple corpus
  positions converge and continue. Extending a virtual cycle at such a
  leaf is exactly where we'd want to — it represents branching choice.
- The compact cache already holds K/V for mass>1 positions and skips
  mass=1. Virtual-tree attachment points are *exactly* the mass>1 leaves
  — the cache naturally contains what we need.

## Architecture: delta-RoPE reads

The current scatter writes **post-RoPE** K into the cache (K is rotated
by its real character position's angle before scatter). For virtual
reuse, the same stored entry needs to serve multiple virtual positions
— each with its own RoPE angle.

Option chosen: **delta-RoPE at gather time.** Store post-real-RoPE K as
today. At gather, apply an additional rotation
`Δθ = θ(virtual_pos) − θ(real_pos)` per gathered slot. Net effect: K
ends up at the correct `θ(virtual_pos)` regardless of what real
position it was cached at.

- Scatter: unchanged.
- Gather: each gathered K slot carries its target read position; kernel
  computes the delta rotation and applies.
- V: no RoPE involved, no change.

## Phases

**Phase 0 — Delta-RoPE ancestor gather.**
Extend `kv_gather_anc_compact_bf16` to accept a per-slot target
`read_pos[T_anc]` and a per-slot source `real_pos[T_anc]`. Apply delta
rotation at gather. K=1 (no virtual) → pass real_pos as read_pos, delta
is zero — bit-identical to today. Verify parity on Lightning d=16.

**Phase 1 — Loop-point catalog.**
For each radix node, flag whether it's a loop point: mass>1 radix
node with no children in the trie (a D*-cap branching truncation).
Per-subtree lists of loop points precomputed once at trie load.

**Phase 2 — Virtual-tree sampling.**
Extend the Lightning sampler: with probability `p_loop` at a mass>1
leaf, chain a second L3 walk from root for cycle 2. Repeat for cycle
K. Output: a virtual training target at virtual depth
`d₀ + d₁ + … + d_{k-1} + d_k` with a concatenated ancestor chain.

**Phase 3 — Virtual forward.**
For each virtual sample, compute Q at every virtual position (new
fresh forward for Q only — token embeddings differ from cycle to
cycle in general). Gather ancestors: mix of cycle-1's cached K/V (via
delta-RoPE) + cycle-k's fresh d_k for own-edge. Attention runs as
today with expanded kv_length.

**Phase 4 — Virtual backward.**
Q grads flow back normally. K/V grads at cache positions need to
scatter-add (since multiple virtual reads may touch the same cache
entry). Option: accumulate dK/dV across virtual reads that hit the
same slot.

**Phase 5 — Benchmark.**
K ∈ {1, 2, 4} at d=16 and d=32 Shakespeare. Compare to baseline
Lightning at matched optimizer steps.

## Open questions

- **Which sampler for cycle k>1?** Old plan's Mj (Markov-cycle-relative
  depth) may not matter now — we sample a real L3 walk each cycle,
  which uses the actual D-trie statistics.
- **Which seed token for cycle k+1?** The terminating token at loop
  point's head. Cycle k+1 starts with that token at virtual position
  `d₀ + d₁ + … + d_{k-1} + 0`.
- **p_loop or always loop?** If we always loop at mass>1 leaves, cycle
  count is forced. Parameterize with `--virtual-cycles K` (like old
  branch) and `p_loop` (probability of extending a given leaf — default
  1.0 at cycles k < K).

## Not carried over from old branch

- Pre-computed Markov-walk priors (`compute_virtual_prior_tokens_per_rc`
  + neg-pos RoPE cache). Delta-RoPE makes those unnecessary.
- Prior-specific KV cache tail slots. Delta-RoPE makes those
  unnecessary.
- The Stage D mini-forward is replaced by the fact that compact-cache
  already holds mass>1 K/V, and own-edge comes from fresh d_k/d_v.
