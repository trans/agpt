# Descendant→ancestor scatter for Wk/Wv gradient flow

## Status

Open. Identified 2026-05-18 during deep-dive into gradient flow.
Originally noted in inline comment at `src/cuda/agpt_train.cu:6187-6189`
and in `train_epoch`'s comment at line 3604-3617.

## What's missing

In the per-subtree training path (`run_radix_training`, the path used
by all current recipes), the varlen attention backward computes
gradients at every K/V position the queries attended to, including
ancestor positions. But only the **own-edge slice** of these gradients
is used (extracted by `kv_uncopy_own_edge_kernel` at line 2167).

The **ancestor slice** is dropped:
- For each descendant query that attended to ancestor A's K (read from
  the cache via `kv_gather_anc_compact_bf16`), there's a gradient
  contribution at that ancestor K position
- That gradient SHOULD propagate back to A's Wk update via A's hidden
  state h_A (since K_A = Wk · h_A)
- Currently it doesn't — the gradient is computed and discarded

So Wk and Wv only learn from each node's OWN self-attention, not from
descendants attending to that node as an ancestor.

## Why it matters

The forward pass uses descendant→ancestor information richly (descendants
read ancestor K/V from the cache). The backward pass doesn't propagate
descendant gradient signal back to ancestor K/V parameters. This is a
real architectural approximation:
- Wq, Wo, FFN, embeddings, LN: all get full gradient — normal backprop
- Wk, Wv: thin own-edge-only gradient

Empirical evidence the model still trains well: held-out PPL 5.03 (vs
trie's 170). So the approximation isn't fatal. But it's a non-trivial
piece of the AGPT story that isn't being trained for, and finishing
this would test whether the model improves under proper backprop.

## Why it wasn't already done

Originally the assumption was that updating Wk would create K/V cache
staleness (cache contains values computed under OLD Wk; descendants
read them under NEW Wk → mismatch). The 2026-05-18 deep-dive
established that:
- At pd=1, the cache regions are disjoint per subtree → no real
  staleness even with Wk updates
- At pd>1, there's some cross-group cache reads → real staleness,
  but masked by `--shuffle-order` in current recipes

So the staleness concern that originally motivated the "skip Wk gradient"
shortcut was overstated. The shortcut became a permanent fixture
because nobody got around to revisiting it.

## What it would take to fix

In `run_radix_training`:

**KEY DESIGN POINT (corrected 2026-05-19):**
The K/V cache is indexed by COMPACT CHARACTER POSITION, not by radix
node ID. Specifically, mass=1 cap chars are skipped (per
`project_compact_kv_cache`); only mass>1 chars get cache slots, via
`compact_slot[char_pos] -> compact_idx` lookup. A radix node with
own_len characters has either ALL its chars in the cache (if mass>1)
or NONE (if mass=1). Descendant gradients flow back to specific
*character positions* in the cache, not to whole nodes. So all
per-subtree buffers and lookups must be scoped per-compact-char, not
per-radix-node.

1. Add per-layer K-grad and V-grad accumulators scoped to the current
   subtree's COMPACT CHARS:
   - `d_dkv_subtree_k[l]`: `[max_n_subtree_compact_chars, D]`
   - `d_dkv_subtree_v[l]`: `[max_n_subtree_compact_chars, D]`
   - `h_subtree[l]`: `[max_n_subtree_compact_chars, D]` — saved ln1_out
     per compact-char position for the chain-rule step
   - Plus a `[n_compact_chars]` int lookup `d_compact_to_subtree_idx`:
     global compact-cache slot → subtree-local index (-1 for slots not
     in current subtree)
   - All buffers sized at function entry to max-over-subtrees, reused
     across fires; zeroed for the active portion at each fire start

2. **At fire start** (per subtree, in the rc_idx loop):
   - Walk this subtree's radix nodes (skipping mass=1 caps)
   - For each mass>1 node, walk its own_len char positions
   - For each char_pos with `compact_slot[char_pos] >= 0`:
     - Assign next subtree-local index
     - Write to lookup: `h_lookup[compact_slot[char_pos]] = subtree_idx`
   - Final counter = `n_subtree_compact_chars` for this fire
   - cudaMemcpy lookup to device
   - cudaMemset buffers' active portion to 0

3. **In each chunk's forward**, save ln1_out per character position
   into `h_subtree[l][lookup[compact_slot[char_pos]]]`. The existing
   `sv_ln1_out[l]` chunk-scoped buffer has the values; we just need a
   small kernel that copies them per-char to the subtree buffer.

4. **In each chunk's backward**, after varlen attention backward
   produces `d_dk_pack` and `d_dv_pack`, scatter-add the ANCESTOR
   slice into `d_dkv_subtree_*[l]` indexed via the lookup.
   The ancestor list per query is already in `ancestor_ids` (currently
   used to gather K/V from cache). For backward, walk the same list,
   for each ancestor slot p: `compact_idx = compact_slot[ancestor_char_pos]`;
   `subtree_idx = d_compact_to_subtree_idx[compact_idx]`; if `subtree_idx >= 0`,
   atomic-add the gradient at d_dk_pack[query, p] to
   `d_dkv_subtree_k[l][subtree_idx]`. Same for V.

5. **At end of subtree fire** (before optimizer step), the chain-rule
   reduction is one batched matmul per layer:
   ```
   dW_kw[l] += grad_scale * d_dkv_subtree_k[l]^T · h_subtree[l]
              (effectively: D × D += D × N · N × D, batched across N
               compact chars; cuBLAS sgemm)
   ```
   Same for V. Apply RoPE-inverse on K-side first (mirrors the
   existing own-edge path).

6. The OWN-EDGE slice continues to feed Wk/Wv via the existing
   chunk-local path. Both contributions sum into dW_kw correctly
   (they're additive in the chain rule).

7. Optimizer fires with `d_grads` now containing the full gradient.

**Memory budget (corrected, per-compact-char scoping):**

For Shakespeare d=16 (mass=1 skip ≈ 94% of chars):
- Total compact chars: ~600K (out of 9.3M raw chars)
- Avg per subtree at pd=1: ~9K compact chars
- Max per subtree (space-char): probably ~150K
- Per layer: 150K × 64 × 4 × 3 buffers = ~115 MB
- 2 layers: ~230 MB
- Lookup [n_compact_chars]: 600K × 4 = 2.4 MB
- Total: ~230 MB

For Gutenberg d=16 (~98.5% skip):
- Total compact chars: similar order ~1M
- Max per subtree: maybe ~300K
- 2 layers × 3 buffers × 300K × 64 × 4 = ~460 MB

For WikiText d=32 BPE (hypothetical, high vocab so less mass=1):
- Skip rate unknown; assume ~50%
- Total compact chars: maybe 50M
- Max per subtree: maybe 1-2M
- 4 layers × 3 buffers × 1.5M × 256 × 4 = ~18 GB — would need BF16
  or per-layer freeing at this scale

**Restrict to pd=1 initially.** At pd>1 there's cross-group cache
staleness; adding descendant→ancestor flow on top would compound.

Estimated effort: ~2-3 days for first working implementation +
gradient parity test + 6-run experiment.

## Implementation status (2026-05-19)

In progress on `anc-grad` branch.

- ✅ CLI flag `--anc-grad` + Config field + mode validation
- ✅ Buffer plumbing (initially per-node-sized; **needs resizing to
  per-compact-char per the corrected design above**)
- ✅ Per-fire init (initially per-node-indexed; **needs rework to
  per-compact-char lookup**)
- ⬜ Resize buffers + rework lookup
- ⬜ Save ln1_out to h_subtree during forward (per-char kernel)
- ⬜ Ancestor-scatter call during backward
- ⬜ Chain-rule reduction at fire end (batched sgemm)
- ⬜ Gradient parity test
- ⬜ 6-run held-out comparison

The current scaffolding (commits `93ea429`, `00f6151` on
`anc-grad`) is functionally a no-op (buffers allocated, never used).
Resizing requires reverting the n_in_subtree-based sizing and
substituting compact-char counts.

## Watch-outs

1. **Cache staleness at pd>1**. Updating Wk between subtrees with
   descendant→ancestor flow will worsen the pd>1 cross-group cache
   staleness. May need cache-rebuild-between-fires (see separate
   intra-se-staleness.md todo if/when written).

2. **Numerical scaling**. The new gradient pathway will add to dW_k
   alongside the own-edge contribution. Relative scales may need
   tuning. The `grad_scale` used for own-edge may not be appropriate
   for ancestor contributions (which are aggregated across many
   descendants per ancestor).

3. **Memory**. `[total_nodes, D, n_layers]` x 2 (K and V) is a real
   buffer. For Shakespeare d=16 (~1M nodes, D=64, 2 layers): 1M × 64 ×
   4 × 2 × 2 = ~1GB. Manageable on the laptop's 4070 (8GB) but tight
   at d=32 Gutenberg. May need BF16 or per-layer freed-between-layers.

4. **Verification**. Gradient parity test against a numerical-gradient
   reference is essential. Easy to introduce subtle bugs in the scatter
   indexing.

## Falsifiable hypothesis

Implementing descendant→ancestor scatter:
- Should at minimum match baseline (since the extra gradient is
  mathematically correct)
- Should likely improve generalization PPL by a few percent
- May enable larger effective LR (since the descendant signal partially
  damps via the natural-gradient-like aggregation)
- May reduce sensitivity to seq_len since longer effective receptive
  fields get proper gradient signal

If the implementation matches baseline but doesn't improve, the
approximation was effectively benign and the model was finding ways
around it. If it improves, we've been training a stunted version of
AGPT.

If it hurts (and the gradient parity test passes), something subtle
about the AGPT setup is incompatible with full descendant→ancestor
flow. Worth understanding before moving on.
