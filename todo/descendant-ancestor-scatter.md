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

1. Add per-layer K-grad and V-grad accumulators at the global trie
   level (same shape as the K/V cache itself): `d_dkv_global_keys[l]`,
   `d_dkv_global_values[l]` of shape `[total_nodes, D]`.

2. After varlen attention backward produces `d_dk_pack` and
   `d_dv_pack`, scatter-add the ANCESTOR slice (positions 0 to anc_len
   per query) into `d_dkv_global_*` indexed by ancestor node ID.
   (Equivalent to `launch_kv_scatter_add` in train_epoch.)

3. The OWN-EDGE slice is still extracted for the current chunk's Wk
   update (existing code path).

4. After all chunks of a subtree are done (before optimizer step), the
   `d_dkv_global_*` accumulators contain accumulated grad-at-K-position
   from descendants attending to ancestors in THIS subtree's nodes.

5. Convert those per-position gradients to per-parameter gradients via
   the chain rule: for each ancestor a, `dW_k += d_dkv_global_keys[a]^T
   · h_a`. Requires knowing each ancestor's saved h, which means we
   need to save h per node during forward (already done as part of
   `saved_ln1_out`?).

6. Apply rotational inverse for RoPE on K-side before the Wk gradient
   step (mirrors what's already done for own-edge).

Estimated effort: ~1-2 days.

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
