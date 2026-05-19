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

1. Add per-layer K-grad and V-grad accumulators **scoped to the current
   subtree** (NOT global trie). Shape `[n_in_subtree, D]` per layer.
   - `d_dkv_subtree_k[l]`, `d_dkv_subtree_v[l]`, `h_subtree[l]` (saved
     ln1_out per node for the chain-rule step)
   - Plus a `[total_nodes]` int lookup `d_global_to_subtree_idx`
     mapping global trie node ID → subtree-local index (-1 for nodes
     not in current subtree)
   - All allocated/zeroed at subtree fire start, freed/reused per fire

2. In each chunk's forward, save ln1_out per query into `h_subtree[l]`
   at the query's subtree-local index. (K/V cache scatter stays as-is.)

3. After varlen attention backward produces `d_dk_pack` and
   `d_dv_pack`, scatter-add the ANCESTOR slice (positions 0 to anc_len
   per query) into `d_dkv_subtree_*[l]` indexed by
   `d_global_to_subtree_idx[ancestor_id]`. Uses atomic-add because
   multiple descendants can contribute to the same ancestor.

4. The OWN-EDGE slice continues to feed Wk via the existing chunk-local
   path (don't break what works). Both contributions sum into
   `dW_kw` correctly because they're additive in the chain rule.

5. At end of subtree fire (before optimizer step), for each ancestor a
   that has non-zero accumulated grad:
   ```
   dW_kw += grad_scale * d_dkv_subtree_k[l][a]^T · h_subtree[l][a]
   dW_kb += col-sum(d_dkv_subtree_k[l][a]) (scaled)
   ```
   Same for V. Apply RoPE-inverse on K-side before the matmul (mirrors
   the own-edge path).

6. Optimizer fires with `d_grads` now containing the full gradient.

**Memory budget (per-subtree-scoped):**
- Shakespeare d=16: ~15K nodes × 64 × 4 × 3 buffers (K grad, V grad,
  ln1_out) = ~12 MB per layer
- Gutenberg d=16: ~75K × 64 × 4 × 3 = ~58 MB per layer
- WikiText d=32 BPE (hypothetical): ~800K × 256 × 4 × 3 = ~2.5 GB per layer

Manageable at our scales. The `[total_nodes]` lookup table is one
int per node — 4 MB at 1M nodes, trivial.

**Restrict to pd=1 initially.** At pd>1 there's cross-group cache
staleness (the shuffle ablation confirmed it); adding descendant→ancestor
flow on top of that would compound. pd=1 is cache-disjoint per subtree
so the new gradient flow is on clean K/V values.

Estimated effort: ~2-3 days for first working implementation +
gradient parity test + 6-run experiment.

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
