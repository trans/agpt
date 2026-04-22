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
