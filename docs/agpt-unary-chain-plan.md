# Unary Chain Compression — Next Session Plan

*Target: implement window-style batched forward/backward over unary trie chains.*

## Quick Start

Branch: `agpt-sibling-attention`

To get back into context:
```
git log --oneline -8         # recent work
cat docs/agpt-bottleneck-analysis.md   # why we're doing this
crystal spec spec/agpt_chain_compression_spec.cr --link-flags="$(pwd)/build/kernels.o"
```

## Why This Matters (confirmed)

At 20k starts on Shakespeare seq_len=16:
- **97.3% of trie nodes** live in non-trivial unary chains
- Average chain length: **8.12**
- 22,860 chains of length > 1 covering 230,320 of 236,604 nodes

Each chain currently costs L per-node forward ops + L per-node attention ops.
Chain compression: 1 window forward + 1 causal attention matrix per chain.

## What's Already in Place

**`src/agpt/trie_corpus.cr`:**
- `TrieCorpus::Segment` record — `(id, node_ids, start_depth, parent_id)`
- `TrieCorpus#build_segments` — returns all segments in topological order
- `TrieCorpus#each_segment_group` — yields groups by start_depth

**`src/agpt/batched_depth_forward.cr`:**
- `forward_unary_chain` scaffolded with full signature + implementation plan in
  a comment block. Raises `NotImplementedError` — next session fills the body.

**`spec/agpt_chain_compression_spec.cr`:**
- Reference spec records per-node snapshots from the current path
- First test (self-consistency) passes
- Second test (`pending`) is ready to be activated once chain forward exists

## Implementation Steps (in order)

### Step 1: Fill in `forward_unary_chain`

Signature (already in place):
```crystal
def forward_unary_chain(
  chain_nodes : Array(TrieNode),
  parent_cache : Array(LayerKVCache),
  ancestor_ids_base : Array(Int32),
  start_depth : Int32,
  kv_store : NodeKVStore,
  model : MiniGPT,
  corpus : TrieCorpus
) : {Array(NodeResult), Array(LayerKVCache)}
```

**Body (~200 lines):**
1. `L = chain_nodes.size`, tokens = chain node token_ids, embed → `x : [L, d_model]`
2. For each block:
   - `x_norm = MicroGPT.backend.layer_norm_forward(x, ln1.gamma, ln1.beta)`
   - `q_all = x_norm * attn.wq.w + bias`, same for K, V
   - Split per head
   - For each head:
     - Apply RoPE at absolute positions `start_depth + i` for i in 0..L-1
     - Build combined K/V: parent_cache's prefix + new K/V per position
     - Attention: each position i attends to parent prefix (len = parent_cache.len) + chain positions 0..i
     - Causal mask within chain (position i cannot see j > i)
     - Save per-position attn_weights (prefix_len varies per position!)
   - Concat heads, WO projection, residual, LN2
   - FFN, residual
3. Final norm + output projection → `[L, vocab_size]`
4. Build `NodeResult` per position i with:
   - `logits = row i of output`
   - `block_states` = per-position intermediates (need to save per-position during block loop)
   - `final_x`, `final_normed`, `final_norm_out` = row i
   - `position = start_depth + i`
   - `ancestor_ids = ancestor_ids_base + chain_nodes[0..i].map(&.id)`
5. Store K/V per chain node into `kv_store.store_layer(node.id, li, k_row, v_row)`
6. Return `(results, extended_cache)` — the extended cache has parent_cache's rows + chain's L new rows per layer

### Step 2: Verify against reference

Activate the `pending` test in `spec/agpt_chain_compression_spec.cr`:
```crystal
it "chain-compressed forward reproduces reference snapshot" do
  MicroGPT.use_crystal!
  model = make_deterministic_model
  reference = reference_snapshot(model)
  compressed = chain_compressed_snapshot(model)
  # assertions as shown in file
end
```

`chain_compressed_snapshot` builds the same tiny trie, iterates via
`corpus.build_segments`, calls `forward_unary_chain` per chain, collects
per-node snapshots in the same format.

Run: `crystal spec spec/agpt_chain_compression_spec.cr --link-flags="$(pwd)/build/kernels.o"`

Target: each node's logits/final_x/K/V match per-node path to `1e-4` tolerance.

### Step 3: Wire into trainer

Once spec passes, modify `src/agpt/trie_walk_trainer.cr`:

Current loop:
```crystal
@corpus.each_depth_level do |depth, nodes|
  # ... batched forward over all nodes at this depth ...
end
```

New loop:
```crystal
@corpus.each_segment_group do |depth, segments|
  # Group A: unary chains (segment.node_ids.size > 1)
  segments.select { |s| s.node_ids.size > 1 }.each do |seg|
    parent_cache = # resolve from prev_caches or kv_store
    results, new_cache = BatchedDepthForward.forward_unary_chain(...)
    # accumulate into depth_results, loss_info, etc.
  end

  # Group B: length-1 segments at this depth
  short_nodes = segments.select { |s| s.node_ids.size == 1 }.map { ... }
  BatchedDepthForward.forward_depth(short_nodes, ...)
end
```

Complication: chain nodes span multiple depths. The current subtrie
partitioning is by root child at each depth — with chains, partitioning
happens at segment granularity instead. Rethink `node_root_child` mapping.

### Step 4: Backward

Two options:

**Option A (partial, faster):** Keep per-node backward. After chain forward,
the per-position BlockStepStates are available — backward runs through them
one at a time as it does today. Forward speedup only, no backward speedup.

**Option B (full):** Implement `backward_unary_chain` mirroring forward.
Batched backward through FFN, LN, attention in reverse. Backward attention
becomes one [L, parent_len + L] causal backward pass.

Start with Option A. Measure wall-clock impact. If not enough, do Option B.

## Correctness Watchpoints

1. **RoPE positions** — chain position i is at depth `start_depth + i`, which
   is its absolute position in the sequence from root. Not just `i`.

2. **Variable prefix_len per position** — within a chain, position 0 attends
   to parent_cache only, position 1 attends to parent_cache + chain[0]'s K/V,
   etc. The attention weights have different widths per position.

3. **Causal mask within chain** — position i must not see position j > i.
   Standard causal mask handles this if we compute attention as
   `[L, parent_len + L]` scores and mask positions > parent_len + i for row i.

4. **Storing attn_weights per position** — for backward, each chain position
   needs its own attn_weights mat of its own prefix_len. Don't store one
   uniform `[L, parent_len+L]` — store per-position slices.

5. **Ancestor chain identity** — for scatter_ancestor_grads to work in
   backward, each position's ancestor_ids must be correct. Position i's
   ancestors = parent's ancestors + chain[0..i-1] (NOT including i itself).

## Performance Expectation

- Forward: per-position attention ops drop from L to effectively 1 per chain
  (single batched op). Projections/FFN gain batching across chain positions.
- Backward (Option A): unchanged (per-node still).
- Backward (Option B): same collapse as forward.

Estimated speedup (at 20k starts, d64n2):
- Option A: ~1.5-2× overall epoch speedup
- Option B: ~3-4× overall epoch speedup

## Known Risks

- **GPU path**: `gpu_batched_attention` packs variable-length attention today
  but doesn't exploit chain structure. Chain forward on GPU requires a new
  kernel or clever use of the existing flat varlen kernel (one chain's
  positions as one "node" with explicit causal mask). Consider leaving GPU
  path per-node initially, compare CPU chain vs GPU flat for correctness.

- **Subtrie partitioning**: current scheme groups by root-level ancestor. With
  chain compression, a chain IS a "natural subtrie" from the branching parent.
  Update partitioning to match.

- **K/V cache invalidation**: if chain forward uses a fresh extended cache
  per chain (correct), make sure the parent cache isn't mutated in-place. The
  existing `cpu_per_node_attention` clones at branching; chain forward should
  be similar.

## After It Works

1. Run the d64n2 20k starts comparison again — that's the measurement we care about
2. Verify entropy weighting and rotation still work with segment-based iteration
3. Benchmark on CPU backend (where today's grouped attention already helps)
4. Commit
5. Write up results for paper Section 10
