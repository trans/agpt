# Partition KV scoping — implementation plan

**Branch**: `agpt-partition-kv-scoping`
**Goal**: make `--partition-depth N` genuinely reduce peak KV footprint so d=16 global (9.5 GB global KV) becomes trainable in one super-epoch.

## What's already true

- `--accumulate` (default) keeps weights stable across all partition groups within a training-unit call → K/V computed at any point in the super-epoch is consistent with current weights. **No staleness issue to solve.**
- KV cache layout is `[char_pos * D]` flat; indices computed CPU-side and uploaded as arrays (`h_char_pos`, `h_prefix_char_ids`). Kernels index by whatever we give them.
- Therefore: the whole optimization is a CPU-side remap + a smaller `cudaMallocManaged` region.

## Design

Opt-in flag: `--partition-kv-scoped` (for now — can become default once validated).

When enabled, inside `run_radix_training`:

1. **Pre-compute char-ranges per partition group.** For each group `g`, collect the union of char_pos values its nodes touch (each node's `edge_start..edge_start+edge_len` plus the node's `ancestor_char_ids` range from the trie).

2. **Allocate one reusable local-KV buffer** sized to `max_g(|chars[g]|) * D * 2 * L_layers`. Memory cap is largest partition, not whole file.

3. **Per partition group** (outer loop, already exists):
   a. Compute `global_to_local[char_pos] → slot` — a dense array `int[file_total_chars]` filled with -1, then populated with local slots for this group's chars. Rebuild per group (cheap — O(partition_chars)).
   b. Zero local KV.
   c. **Re-scatter ancestor K/V.** For the chars in `chars[g]` that are *ancestors* (not own-chars), run the embedding → Q/K/V pipeline on those positions to populate their K/V in local cache. Current weights, so this is just a cheap forward pass over the ancestor chain (~1–2 chars for partition_depth=2 bigram).
   d. **Process the group** through the existing chunk loop — but before each `launch_kv_scatter` / `launch_kv_gather`, rewrite `h_char_pos` and `h_prefix_char_ids` with `global_to_local[...]`. Kernels unchanged.

4. **End of group**: no optimizer fire (we're in `--accumulate` mode). Move to next group.

5. **End of all groups**: existing single-optimizer-step path fires.

## Correctness check

With `--partition-depth 2 --partition-kv-scoped`, output weights after 1 super-epoch should match `--partition-depth 2` (no scoping) within numerical noise. If PPL differs noticeably, the remapping has a bug.

## Memory goal

| config | file cache | peak at scoped partition=2 |
|---|---|---|
| d=16 biggest file (rc=2) | 1.3 GB | ~60 MB (23× smaller) |
| d=32 biggest file | 4 GB | ~300 MB (13× smaller) |

## Unlocks

Global d=16 radix + `--partition-depth 2 --partition-kv-scoped` — the "1 step over whole d=16 trie" experiment becomes runnable. Peak KV ~60 MB per partition; total ~9.3 GB of *sequential* partition processing rather than simultaneous allocation.

## Out of scope for this branch

- Making `--partition-kv-scoped` the default. First validate parity with the non-scoped path.
- Cross-file accumulation (for true "1 step per super-epoch across all files"). Separate change, easier.
- Measuring performance impact of the re-scatter overhead. Likely dominated by the partition's own compute, but worth profiling.

## Phase 1 (done, commit 48ee729)

- `--partition-kv-scoped` flag added.
- Per-partition char_pos ranges computed + peak-KV reduction reported.
- Measured on d=16 Shakespeare bigram: 8× reduction on the largest file.
- Projected global-d=16 single-step peak: ~162 MB (fits in 9.2 GB RAM).

## Phase 2 implementation steps (ordered)

**Step A: restructure (mechanical, ~40 lines moved)**
Move the KV allocation block (`agpt_train.cu:2807-2828`) from its current location to after the partition setup (after line ~3306). Partition setup has no dependency on KV allocation; KV allocation has no dependency on buffers allocated between the two. Verify with compile + existing tests.

**Step B: conditional KV sizing (~10 lines)**
In the moved KV allocation, when `partition_kv_scoped && partition_depth > 1`, size KV for `max_partition_chars * D * sizeof(float) * 2 * L_layers`. The `max_partition_chars` value is already computed in the Phase 1 stats block — hoist that to a variable available at allocation time.

**Step C: global_to_local mapping (~30 lines, CPU-side only)**
Allocate `int global_to_local[file_total_chars]` once. Before each partition group's processing, populate it: iterate the group's nodes, for each touched char_pos, assign the next local slot. Clear to -1 after the group.

**Step D: mini-forward for ancestor K/V (~200 lines, the real work)**
For each partition group, walk the path from root to the partition-ancestor radix node. This gives an ordered list of ancestor chars (usually 1-2 chars for partition_depth 2 or 3). For each layer l in 0..L-1:
- Embed each ancestor char.
- LayerNorm → Q/K/V projection.
- RoPE on Q, K using ancestor's semantic depth (char_depth in the trie).
- Scatter K, V to local cache at `global_to_local[char_pos]`.
- Self-attention among the ancestors (causal within the chain).
- FFN + residual.
- Hand off (layer l+1's input is the output of layer l attention+FFN).
- **Skip**: final-layer logits, loss, backward.

This is a cut-down version of the main chunk forward. Cleanest structure: factor existing per-chunk forward into a helper that takes `(token_ids[], char_pos_global[], rope_positions[], query_offsets, kv_offsets, ...)` and returns per-layer K/V, optionally loss+backward. Then call it twice per group: once for ancestors (K/V-only mode), once for group's own nodes (full loss+backward).

**Step E: remap indices in main chunk forward (~20 lines)**
In the chunk loop, when `partition_kv_scoped`, rewrite `h_char_pos[]` and `h_prefix_char_ids[]` through `global_to_local[]` before uploading to GPU. Kernels unchanged.

**Step F: smoke test correctness**
Run `--partition-depth 2 --partition-kv-scoped` vs `--partition-depth 2` (no scoping) vs `--single-subtree` (no partition) on d=16 per-subtree, 1 SE from random-init. All three should give PPL within noise (~0.3) of each other. If scoped diverges significantly, the remap is wrong.

**Step G: global d=16 single-step (the prize)**
Build global-radix d=16 manifest (if not already): `/tmp/agpt_input_d16_radix`. Run `bin/agpt_train --trie-dir /tmp/agpt_input_d16_radix --single-subtree --partition-depth 2 --partition-kv-scoped`. Memory should fit (~162 MB peak KV vs 9.5 GB previously). Measure PPL vs per-subtree baseline.

## Estimated effort

Step A+B+C+E: ~1h. Step D (the mini-forward): ~2-3h of careful work. Step F validation: ~30min. Step G experiment: ~30min. Total ~4-5h for a dedicated session.
