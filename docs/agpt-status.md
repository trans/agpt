# AGPT Status: Where We Are and What's Blocking Us

*Updated: 2026-04-13*

## What AGPT Is

AGPT (Autoregressive GPT over Prefix Trie) replaces standard windowed training with a trie-structured approach. Instead of sampling random text windows and training on each independently, AGPT builds a prefix trie from the corpus where shared prefixes are represented once. The model trains by walking the trie depth-by-depth, computing next-token predictions at each node using the exact empirical distribution (not a single sample).

The core mathematical identity:

```
J_p · (Σ_s g_s) = Σ_s (J_p · g_s)
```

The left side (trie) aggregates gradients before applying the shared prefix Jacobian once. The right side (window) applies it separately for each path. Same result, fewer operations.

## What Works

- **Columnar trie** with binary-search child lookup, struct façade, 4-5× per-node storage reduction
- **NodeKVStore**: ~1 KB per node (K/V contributions only), reconstructs full KV caches from parent chain on demand
- **Batched depth-level forward/backward**: Q/K/V projections, FFN, layer norm, output projection all processed as single `[N, d_model] × W` matmuls across all nodes at a depth level
- **Depth-progressive subtrie training**: partitions nodes at each depth by root-level ancestor (~57 subtries per depth), each with its own forward+backward+update cycle
- **Local-depth backward**: truncated gradient horizon (like truncated BPTT) — only backwards the current depth, avoids O(D²) full-subtrie cost
- **Gradient normalization**: `dW /= subtrie_size` before each update
- **Held-out validation**: mode-agnostic CE evaluation for fair comparison
- **CUDA kernel**: batched variable-length attention kernel written and compiling (not yet the bottleneck path)

## Comparison Results (d32n1, seq_len=16, Shakespeare)

| Metric | Window Training | AGPT (subtrie) |
|--------|----------------|----------------|
| Best held-out CE | **2.35** | 2.62 |
| Wall-clock to best | 972s (50k steps) | 495s (20 epochs) |
| At equal wall-clock (~500s) | **~2.48** | 2.62 |
| Updates to reach best | 50,000 | ~18,000 |
| Per-update overhead | **1× (baseline)** | ~4.5× |
| Still improving at end? | No (oscillating) | Yes (smooth descent) |

AGPT converges smoothly but is 4.5× slower per weight update. Window reaches a better CE in the same wall-clock time, though it plateaus and oscillates while AGPT is still improving.

## The Bottleneck: Per-Node Attention

The trie's value is in batching: at each depth level, projections and FFN for all N nodes are one matmul each. This part is fast. But **attention** is per-node because each node has a different KV history (different prefix path through the trie). Currently:

```
For each of ~27,000 nodes at each depth level:
  1. Walk parent chain in KV store to reconstruct this node's KV cache  (~60% of time)
  2. Compute per-node attention: [1, depth] × [depth, head_dim]          (~25% of time)
  3. GC / temporary Mat allocation overhead                              (~15% of time)
```

This is 27,000 serial operations per depth level × 16 depth levels = 432,000 individual attention computations per epoch. The batched projections/FFN take <1 second; the per-node attention takes ~27 seconds. The attention loop is 95%+ of epoch time.

### Why This Matters

The trie was designed to reduce computation through shared prefixes. And it does — for projections/FFN. But attention is inherently prefix-dependent (each node attends to its own history), so it can't be trivially batched across nodes with different prefixes.

This means the trie's compute savings only apply to ~85% of FLOPs (projections/FFN) while the remaining ~15% (attention) dominates wall-clock time due to serial processing overhead.

## Known Fixes (Not Yet Implemented)

### 1. Sibling Attention Batching

Nodes with the same parent share the same KV cache through position d-1 and only differ at position d. For C siblings:
- Shared K/V (positions 0..d-1): one `[C, d-1] × [d-1, head_dim]` matmul
- Per-sibling K/V (position d): C dot products

This batches the majority of attention work since siblings are common (802 branching nodes with avg ~3 children at our test config).

### 2. Incremental KV Cache Propagation

Currently the forward pass reconstructs each node's KV cache from scratch by walking the parent chain in the KV store. With incremental propagation, a child extends its parent's cache by one entry — O(1) instead of O(depth). The mechanism exists (`parent_caches` parameter in `BatchedDepthForward`) but isn't fully working for all code paths.

### 3. GPU-Native KV Store

The CUDA batched attention kernel is written and compiles, but the data currently round-trips through CPU (pack → upload → kernel → download → unpack). If the KV store lived on GPU memory, the attention kernel could read K/V entries directly — no CPU involvement in the attention loop.

### 4. Construction Kit Kernel Compiler

The long-term fix: compile the entire depth-level computation (projections + attention + FFN) into a single fused CUDA kernel from the graph AST. No CPU round-trips, no kernel launch overhead. This is what the construction kit's executable graph engine is building toward.

## Scaling Argument

At our test scale (d_model=32, 1 layer, seq_len=16, 2000 starts), the per-node attention overhead dominates because the batched matmuls are tiny (`[2000, 32] × [32, 32]`). As models scale:

- **Larger d_model** (256+): projection/FFN matmuls become the dominant cost, and the trie's batching advantage grows proportionally (O(d_model²) savings)
- **More starts** (full corpus): deeper branching in the trie means more prefix sharing, more compute eliminated
- **GPU batching**: the attention bottleneck disappears with proper GPU implementation

The current 4.5× per-update overhead is a property of our small-scale CPU implementation, not the algorithm. At scale, the ratio should invert.

## Algorithmic Open Questions

### Update Frequency vs Gradient Correctness

The mathematically correct approach is one weight update per full trie traversal (global gradient aggregation with consistent weights). But this gives too few updates for good SGD convergence. The current solution — subtrie partitioning with local-depth backward — is a pragmatic compromise:

- **Correct within each subtrie**: weights are fixed during each subtrie's forward+backward
- **Approximate across subtries**: earlier subtries' cached K/V become stale after weight updates
- **Frequent updates**: ~912 updates per epoch (comparable to window training)

This is analogous to truncated BPTT in RNNs or pipeline parallelism in distributed training. It works in practice but isn't the exact gradient of any single loss function over the full trie.

### Cross-Position Gradient Flow

The local-depth backward drops the dK/dV gradient from deep nodes to shallow ancestors. This means shallow layers don't directly learn from deep predictions within an epoch — they only benefit indirectly through shared weights updated at later depth stages. Full subtrie backward preserves this flow but costs O(D²) per epoch. Whether the cross-position gradient matters in practice is an open empirical question.

### Entropy-Weighted Future Loss

Not yet explored: using entropy or other metrics at branching points to weight the loss contribution of different subtries. Nodes with high branching (uncertain futures) might benefit from stronger gradient signal. This is the "icing on the cake" mentioned in design discussions — not blocking the core comparison but potentially important for AGPT's advantage over window training.

## Files

| File | Purpose |
|------|---------|
| `src/agpt/trie_walk_trainer.cr` | Main training loop: depth-progressive subtrie training |
| `src/agpt/batched_depth_forward.cr` | Batched forward: projections as single matmuls, per-node attention |
| `src/agpt/batched_depth_backward.cr` | Batched backward: weight gradient accumulation, per-node attention backward |
| `src/agpt/node_kv_store.cr` | Compact K/V storage + parent-chain reconstruction |
| `src/agpt/trie_corpus.cr` | Columnar trie with binary-search child lookup |
| `src/agpt/trie_node.cr` | Thin struct façade over columnar storage |
| `src/agpt/kv_cache.cr` | Per-layer KV cache (pre-allocated, extend/truncate) |
| `src/agpt/node_state.cr` | NodeForwardState, BlockStepState, NodeGradAccum |
| `src/agpt/incremental_forward.cr` | Single-node forward (kept for gradient verification) |
| `src/agpt/incremental_backward.cr` | Single-node backward (kept for gradient verification) |
| `src/agpt/weighted_loss.cr` | Weighted next-token cross-entropy against empirical distribution |
| `src/cuda/kernels.cu` | CUDA kernels including batched varlen attention |
| `docs/agpt-prefix-memory.md` | Design spec for depth-layered storage + checkpointed vectors |
