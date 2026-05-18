# AGPT Level-Paged Trainer — Implementation Plan

*Goal: scale AGPT training beyond the ~50k-start RAM ceiling on a 16 GB
machine, by storing per-depth trie data and per-node K/V state on disk and
streaming only the working set into RAM.*

## Why

At 100k starts the trie is ~1M nodes. RAM use exceeds 6 GB Mat cap during
forward at the wider mid-depths. Below the actual blockers, ranked by size:

| Component | Size at 1M nodes | Lifetime | Streamable? |
|---|---|---|---|
| `NodeKVStore` entries (per-node K/V per layer per head) | ~1 GB | Whole epoch | Yes — by depth |
| Per-depth `BlockStepStates` (forward outputs) | ~200 MB per depth | One depth | Yes — per depth |
| Per-depth `prev_caches` (LayerKVCache copies) | ~1.6 GB at peak | Two depths | Yes — per depth |
| Per-node `next_token_counts` | ~50 MB | Whole epoch | Read-only — easy stream |
| Trie metadata (columnar arrays) | ~32 MB | Whole epoch | Yes — by depth |

The total static state at 1M nodes is well under 16 GB but exceeds the
3-6 GB practical cap. A level-paged design eliminates the unbounded growth
and lets us go to corpus-scale tries without RAM growth.

## Design

### File layout

```
trie.idx/
  meta.json           — vocab, max_depth, n_starts, hash, layer/head dims
  trie/
    depth_0.bin       — root metadata
    depth_1.bin       — NodeRecord[N_at_depth_1]
    ...
    depth_D.bin
  kv/
    depth_1.bin       — KVRecord[N_at_depth_1] for each (layer, head)
    ...
  counts/
    depth_1.bin       — next_token_counts varint streams per node
```

**NodeRecord (fixed size, mmap-friendly):**
```
struct NodeRecord {
  u32 node_id;          // global id within trie
  u32 parent_id;        // points to a node in depth_{d-1}.bin
  u32 first_child_id;   // points into depth_{d+1}.bin
  u16 child_count;
  u16 token_id;
}
```

**KVRecord per node:**
```
[layer 0][head 0][k_row hd*float32][v_row hd*float32]
[layer 0][head 1]...
[layer 1][head 0]...
```

For a typical d_model=64 / n_layers=2 / n_heads=2 setup: 2 × 2 × 64 × 4 =
1024 bytes per node — exactly the 1 KB we already pay in RAM, just on disk.

### Execution model

The trainer already runs forward+backward depth-by-depth. The paging design
maps cleanly:

```
for d in 1..max_depth:
  open mmap depth_d.bin (NodeRecords for this depth)
  open mmap parent depth_{d-1}.bin and ancestor mmaps as needed
  forward_depth: read parent K/V chains via mmap, compute, write depth_d KV
  backward_depth: read depth_d KV + ancestor chain from disk, scatter dQ/dK/dV
  release depth_d/depth_{d-1} mmap windows
```

**RAM working set = parent depth + current depth.** All other depths sit on
disk. Total RAM bounded by ~2 × (peak depth N × per-node size).

### Read pattern

- **Forward** at depth d:
  - Sequential read of depth_d's NodeRecords + parent depth_{d-1}'s K/V chains.
  - Sequential read = NVMe at 3-7 GB/s. 1 GB of ancestor K/V at 5 GB/s = 200 ms.
- **Backward** at depth d:
  - Random read of ancestor K/V (chain walk). NVMe random IOPS ~100-500k.
  - 20k nodes × 16-deep ancestor walks × 4-row reads each = ~1.3M reads.
  - At 200k IOPS = ~6 seconds per epoch worst case.
  - **Mitigation:** precompute and batch-read all ancestor K/V for the depth's
    nodes in one sequential sweep before the backward kernel runs. Buffer in RAM.

### Write pattern

- **Forward** writes depth_d's K/V to disk. Sequential write at 1-3 GB/s.
- For 100k starts with ~20k peak depth nodes: 20 MB per depth × 16 depths = 320
  MB written per epoch. 100 ms per epoch.

### Compatibility

- Existing `--agpt-save-index` already writes a binary trie file. Extend the
  format to support per-depth chunks rather than monolithic. Old format can
  remain readable for backward compatibility (small tries).
- Existing in-RAM `NodeKVStore` becomes the "small trie" path; new
  `LeveledNodeKVStore` is the disk-backed path.
- Switch via a new flag like `--agpt-disk-paged` or auto-detect from trie size.

## Phasing

### Phase A: Trie metadata to disk (smallest first cut)

Move `TrieCorpus` columnar arrays to disk-backed via mmap. NodeRecords by
depth in their own files. K/V stays in RAM for now.

**Effect:** Trie metadata's ~32 MB per million nodes paged to disk. Small
benefit alone but establishes the per-depth file format and load/release
machinery. ~2 sessions of work.

### Phase B: K/V store to disk

Replace `NodeKVStore.entries[id][layer][head]` Hash with a `LeveledKVStore`
that writes K/V rows by depth and reads on demand. The current
`reconstruct_layer_cache` becomes a sequential read of the ancestor chain
(potentially batched per-depth).

**Effect:** 1 GB of K/V at 1M nodes paged out. This is the biggest single
win — unblocks 100k+ start runs on this hardware. ~3 sessions.

### Phase C: Per-depth `BlockStepStates` and `prev_caches`

These are already per-depth ephemeral, but allocated per-node. With paging
in place, change to a flat per-depth buffer that gets written to disk between
forward (depth d) and backward (depth d) — useful only if we eventually
process depths out of strict BFS order. Skip for now; in-memory per-depth
state is fine after Phases A+B.

### Phase D: Streaming target counts

`next_token_counts_hash` for each node currently held in RAM. Stream these
from disk per depth during loss computation. ~1 session.

## Validation

- Existing `agpt_chain_compression_spec` and any backward correctness checks
  must pass with the paged trainer.
- Wall-clock at 20k starts should be within 10% of in-RAM baseline (no
  regression for sizes that fit RAM).
- Run 100k-start training and confirm it completes (currently OOMs).

## Risks

- **Random-read amplification on backward.** If the ancestor batch-read
  mitigation isn't enough, may need to keep a hot LRU cache of the most
  recently used ancestor K/V chains in RAM.
- **Writeable mmap corruption on crash.** Use atomic-rename for the final
  trie save; per-epoch K/V can be in tmpfs or scratch dir for transient runs.
- **NVMe wear.** Each epoch writes ~300 MB. 1000 epochs = 300 GB writes.
  Negligible for development; for long training runs use a dedicated scratch
  disk.

## After it works

The headline opportunity:
- AGPT at 200k-500k starts = corpus-scale coverage.
- Per-depth batches of ~50k-100k nodes saturate the GPU.
- The "sync overhead" and "per-rc cadence" debates become moot once epochs
  are an order of magnitude faster per-node and total epochs needed drop.
- Fair comparison vs window training becomes possible — both seeing the
  full corpus.

## Total estimate

3 phases × 2-3 sessions each = 6-9 focused sessions for Phases A+B+D
(skipping C). Validation + tuning adds 1-2 more.
