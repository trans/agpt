# Partition KV Scoping — Implementation Plan

**Branch**: `agpt-partition-kv-scoping`
**Flag**: `--partition-kv-scoped`
**Feature**: make `--partition-depth N` genuinely reduce peak KV cache footprint so the d=16 global trie becomes trainable in one super-epoch (currently OOMs at 9.5 GB RAM+swap requirement).

---

## Problem

The AGPT CUDA trainer allocates KV cache of size `total_edge_chars × d_model × 2 × n_layers × sizeof(float)` per call to `run_radix_training`.

- **d=16 per-subtree largest file (rc=2 "space" subtree)**: 1.3 M chars → **1.3 GB KV**.
- **d=16 global trie**: 9.3 M chars → **9.5 GB KV** — doesn't fit in 9.2 GB RAM+swap.
- **d=32 per-subtree largest**: ~4 M chars → **~4 GB KV**.

These are *file-level* allocations. When `--partition-depth N` breaks the file into groups (e.g., bigrams), we train each group sequentially but the cache is still sized for the whole file — meaning the ~23 bigram groups inside a root-child file all share one 1.3 GB allocation.

## Goal

Shrink peak KV footprint to the largest *partition group*'s char range, not the whole file. Measured upper bounds for d=16 Shakespeare bigram:

| file | unscoped KV | scoped KV | ratio |
|---|---|---|---|
| rc=2 (largest) | 1.3 GB | 162 MB | **8.0×** |
| rc=1 | 254 MB | 31 MB | 8.3× |
| rc=0 | 20 MB | 9.5 MB | 2.1× |

**Projected global-d=16**: ~162 MB peak KV (well within 9.2 GB available). **The feature unlocks single-super-epoch training of the full d=16 trie.**

---

## Approach: per-partition KV with CPU-side index remap

The KV cache layout is a flat `[char_pos × d_model]` array per layer. Every scatter/gather kernel takes a `char_pos` index array computed on the CPU before upload. That means we can shrink the allocation and transparently rewrite indices without touching any CUDA kernels.

### Architecture

1. **Shrink allocation**: size KV to `max_partition_chars × d_model × 2 × n_layers × sizeof(float)` when `--partition-kv-scoped` is on, instead of `total_edge_chars × ...`.

2. **`global_to_local[file_total_chars]` array**: rebuilt per partition group. For each char_pos touched by that group's nodes (ancestors + own), assign a local slot `[0, partition_chars)`. Dense int array, `-1` means "not used in this group."

3. **Pre-populate ancestor K/V per group**: at the start of each group, run a *mini-forward* over the ancestor chain (usually 1–2 chars for partition_depth 2–3) — embedding → Q/K/V projection → RoPE → scatter to local slots, repeated per layer. No loss, no backward. This ensures the group's nodes can gather their ancestors' K/V from the local cache.

4. **Remap in main chunk loop**: before each `launch_kv_scatter` / `launch_kv_gather`, rewrite `h_char_pos[]` and `h_prefix_char_ids[]` through `global_to_local[]` (CPU-side, O(T_q + T_kv) per chunk). Upload remapped arrays. Kernels unchanged.

5. **Between groups**: no clearing needed if remap is correct — each group writes its own slots via ancestor-preforward + chunk-scatter, reads only from slots it's written. BFS ordering within the group guarantees descendants find their ancestors' K/V present.

### Correctness argument

With `--accumulate` (default), weights are fixed across all partition groups in a super-epoch. Therefore the K/V produced for any given char is identical whether computed in its own group or in a downstream group's ancestor mini-forward. The mini-forward is a pure function of (weights, ancestor tokens, RoPE positions); all are stable.

---

## Status

### Phase 1 — flag + stats reporting (**DONE**, commit `48ee729`)

- `--partition-kv-scoped` CLI flag added.
- Per-partition char_pos ranges computed inside `run_radix_training` after partition grouping.
- Stats printed per file: max/min/mean partition chars, file KV vs scoped KV, reduction ratio.
- **No behavior change yet** — allocation still sized for whole file. The print output is what validates Phase 2's achievable savings.

Sample output at d=16 per-subtree bigram:
```
[kv-scoped stats] partition chars per group: max=157901  min=1  mean=22596
[kv-scoped stats] peak KV: file=1295.7 MB  scoped=161.7 MB  ratio=8.0x
```

### Phase 2 — actual scoped allocation + remap (**TODO**)

Ordered implementation steps below.

---

## Phase 2 implementation steps

### Step A: restructure — move KV allocation after partition setup

**File**: `src/cuda/agpt_train.cu`

Current order inside `run_radix_training`:
1. Line 2807: KV allocation (sized by `trie.total_edge_chars`).
2. Line 2830: RoPE cache, per-chunk working buffers.
3. Line 2948: partition grouping (root_child_of, subtree_nodes, optional single_subtree collapse, optional partition_depth regrouping).
4. Line 3234: Phase 1 stats computing max_partition_chars.
5. Line 3308: progressive-curriculum setup.
6. Line 3337: training loop.

Move lines 2807–2828 (KV alloc block) to after Phase 1 stats (after line 3306). Verification: nothing in lines 2830–3306 references `d_kv_keys`, `d_kv_values`, or `kv_bytes`. The RoPE cache and chunk buffers can stay where they are; they're functionally independent.

Effort: ~30 min (mechanical move + compile + parity test).

### Step B: conditional KV sizing

Modify the (now-moved) KV allocation:

```c
long long kv_char_count = trie.total_edge_chars;
if (partition_kv_scoped && partition_depth > 1) {
    kv_char_count = max_partition_chars;
}
long long kv_bytes = kv_char_count * (long long)D * sizeof(float);
long long total_kv_bytes = kv_bytes * 2 * L_layers;
```

`max_partition_chars` is already computed in Phase 1's stats block — hoist it out of the debug-print `if` so it's available for sizing.

Effort: ~15 min.

### Step C: `global_to_local` mapping

Add a persistent array `int* global_to_local = malloc(trie.total_edge_chars * sizeof(int))`, allocated once after partition setup, freed at function exit.

Before processing each partition group `g`:
1. Set `global_to_local[c] = -1` for each char `c` touched by the *previous* group (walk `subtree_nodes[g-1]` to clear).
2. For each char `c` touched by group `g`, assign `global_to_local[c] = next_slot++`.

Track `local_slot_count` for the group so we can catch overflow (`>= max_partition_chars`).

Effort: ~30 min.

### Step D: ancestor mini-forward (the bulk of Phase 2)

For each partition group, the ancestor chain = chars from root down to the partition-ancestor radix node's endpoint. Length = `partition_depth - 1` chars typically.

Factor the existing per-chunk forward logic in `run_radix_training` into a helper that does, for each layer `l`:
- Embedding gather.
- LayerNorm 1.
- Q / K / V linear projections.
- RoPE on Q and K (position = semantic depth).
- Scatter K, V to local cache at `global_to_local[char_pos]`.
- Attention (self-attention among the chain, causal mask).
- WO projection + residual.
- LayerNorm 2.
- FFN + residual.
- **Skip**: final LayerNorm → logits → loss → backward.

The helper takes `(token_ids[n], char_pos_global[n], rope_positions[n], layer_count)` and returns nothing (mutation: K/V in local cache, per-layer activations).

Called once per partition group, before the group's chunk loop begins.

Effort: ~2–3h. This is the most intricate step — careful to replicate weight-sharing and residual connections without introducing subtle mismatches vs the main forward.

### Step E: remap in main chunk loop

Inside the per-chunk setup (around line 3376 where `h_char_pos[j] = edge_start + j` is written, and line 3516 where `h_prefix_char_ids[fill++] = trie.ancestor_char_ids[anc_off + a]` is written):

After building `h_char_pos[]` and `h_prefix_char_ids[]`, if `partition_kv_scoped && partition_depth > 1`:
```c
for (int q = 0; q < T_q; q++) h_char_pos[q] = global_to_local[h_char_pos[q]];
for (int kv = 0; kv < total_kv_len; kv++)
    h_prefix_char_ids[kv] = global_to_local[h_prefix_char_ids[kv]];
```

Then upload to GPU as before.

Effort: ~20 min + debugging.

### Step F: smoke-test correctness

Three-way comparison, d=16 per-subtree, 1 super-epoch from `data/input.random.model`, lr=3e-3 constant, RMSProp:

1. `--single-subtree` (baseline). Expected PPL ~16.2.
2. `--single-subtree --partition-depth 2` (partition, no scoping). Expected ~16.3.
3. `--single-subtree --partition-depth 2 --partition-kv-scoped` (partition + scoping). **Should match 2 within 0.3 PPL.** If it diverges, the remap or mini-forward is wrong.

If step 3 matches, also verify `adam_t` = 65 (one step per file) in all three cases.

Effort: ~30 min.

### Step G: the prize — d=16 global single-step

1. Build global-radix d=16 index if not already: `/tmp/agpt_input_d16_radix` (the non-`_pst` variant).
2. Run `bin/agpt_train --trie-dir /tmp/agpt_input_d16_radix --model data/input.random.model --single-subtree --partition-depth 2 --partition-kv-scoped --epochs 3 --lr 3e-3 --optimizer rmsprop --mass-weight linear`.
3. Memory should stay below ~250 MB peak (vs the 9.5 GB requirement that previously refused the run).
4. Measure held-out PPL; compare to per-subtree baseline at 13.40 mean. Should match (within noise) since it's the same training data and weights, just differently organized.

**Deliverable**: reproducible d=16 single-super-epoch training with memory-bounded peak and matching PPL, closing the grant-pitch claim about "memory-scaling via n-gram partitioning."

Effort: ~30 min (build trie + one run + eval).

---

## Total estimate

| step | time |
|---|---|
| A. restructure | 30 min |
| B. conditional sizing | 15 min |
| C. global_to_local | 30 min |
| D. mini-forward | 2–3 h |
| E. remap in chunk loop | 20 min |
| F. parity test | 30 min |
| G. d=16 global experiment | 30 min |
| **total** | **~4.5 – 5.5 h** |

Best as one dedicated session. Budget ~6 hours.

---

## Risks / open questions

1. **Mini-forward parity with main forward.** Easy to miss a LayerNorm epsilon, a residual connection, or a RoPE position convention. Mitigation: factor the main forward into a reusable helper (Step D refactor) so the ancestor mini-forward uses the same code path.

2. **`global_to_local` dense array size.** At d=32 global, `total_edge_chars` ~ 27 M. The mapping array would be 108 MB — fine, but worth noting. Alternative: hash map (lookup overhead probably worse in practice than the 108 MB).

3. **Peak KV at d=32 per-subtree.** Largest file is ~4 M chars. Max bigram partition within that file is likely ~400 K chars → ~800 MB scoped KV. Good savings but not as dramatic as d=16.

4. **Mass-weight / entropy-lambda interaction.** Mass-weight is per-query; in scoped mode the query positions are the same as in unscoped mode (partition group's own nodes). Ancestor mini-forward produces no queries. Should "just work," but verify in Step F.

5. **Cross-file accumulation is a separate follow-up.** This branch does per-file scoping. For *true* "1 optimizer step over whole super-epoch across all 65 files," we additionally need `run_per_subtree_training` to share a gradient buffer across files and defer the optimizer. Out of scope for `agpt-partition-kv-scoping`; goes in `agpt-cross-file-accumulation` branch.

---

## Out of scope

- Making `--partition-kv-scoped` default. First validate parity; keep opt-in until broadly tested.
- Cross-file gradient accumulation (separate branch).
- Performance/throughput measurements (measure on the d=16 global run; expected to be dominated by compute, not KV overhead).
- Integration with `--subtree-splits` (deprecated; `--partition-depth` preferred).

---

## Checkpoint files

- `notes/agpt/partition-kv-scoping-plan.md` (this file).
- Phase 1 commit: `48ee729` on branch `agpt-partition-kv-scoping`.
- Starting point for Phase 2: branch is 2 commits ahead of `main`.
