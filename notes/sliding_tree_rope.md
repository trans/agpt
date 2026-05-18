# Sliding-Tree AGPT: Corpus-Aware RoPE Positions

**Origin:** user design 2026-05-16. Companion to but distinct from
`notes/sliding_window_agpt.md` — that one addressed seq_len > d
via pooling overlapping windows; this one addresses the *position-OOD
problem* by training each trie path at its actual corpus positions
instead of within-chunk 0..d-1.

## Background

Current AGPT training applies RoPE at trie-internal positions:

    chunk position 0 → RoPE pos 0
    chunk position 1 → RoPE pos 1
    ...
    chunk position d-1 → RoPE pos d-1

Every chunk uses the same 0..d-1 rotation, regardless of where in the
corpus the chunk's path actually occurred. This is *why Phase 1A
failed*: the model never saw any RoPE position other than 0..d-1
during training. At inference seq_len > d, positions d..seq_len-1 are
out-of-distribution rotations the model has no idea how to handle.

## The Change

Replace within-chunk RoPE positions with the chunk's **actual corpus
positions**. For a chunk covering corpus span [p, p+d-1]:

    chunk position 0 → RoPE pos p
    chunk position 1 → RoPE pos p+1
    ...
    chunk position d-1 → RoPE pos p+d-1

Each chunk uses RoPE positions from its actual corpus location. Over a
full corpus pass, the same trie node gets trained at *many* different
RoPE positions — every corpus position where it occurs.

### Example: corpus "ABCDE..."

- Chunk anchored at p=0: A-root at pos 0, A→B path at pos 1, A→B→C
  at pos 2, A→B→C→D at pos 3
- Chunk anchored at p=1: B-root at pos 1, B→C at pos 2, B→C→D at pos
  3, B→C→D→E at pos 4
- Chunk anchored at p=2: C-root at pos 2, C→D at pos 3, ...

The B-unigram root gets trained at RoPE pos 1, 2, ... depending on
where 'B' occurs in the corpus. The same structural node sees diverse
positions across training.

## What This Solves vs. Doesn't

**Solves:** position-OOD at inference. Model has seen RoPE positions
across the full corpus range up to designated seq_len. No
catastrophic-position failures at extended seq_len.

**Doesn't solve:** the attention-budget problem. At any single
training step, attention still only spans d positions. The *capacity*
to integrate longer-range info isn't there. As the user noted, this
gives "indirect" extension only.

**What it composes with:**
- Streaming training (`notes/`: idea #1): same iteration shape.
  Stream through corpus, train at each position. The two ideas are
  the same loop described from two angles.
- ALiBi instead of RoPE: ALiBi handles distance extrapolation natively.
  Sliding-tree training + ALiBi may be the cheapest seq_len decoupling
  available — distance-bias + position-robust trie structure.
- Sliding-window pooling: stays available as v2 if sliding-tree alone
  doesn't extend far enough.

## Three Variants of "Which Corpus Position For This Fire?"

For mass-1 caps (~95% of nodes at d=32): unambiguous, one position.

For mass>1 internal nodes (the high-mass shallow nodes like 'space'
unigram with millions of occurrences): need a selection rule.

### V1a — Canonical position (first-observed)

Each node gets one fixed corpus position assigned at trie-build time
(e.g., the first position in corpus order where it was seen).

- **Pro:** deterministic, simplest. No new data structures beyond a
  single int32 per radix node.
- **Con:** biases position diversity toward early-corpus positions.
  The 'space' unigram only ever sees one RoPE position even though
  it occurs millions of times.

### V1b — Random position per training fire

Each time a node is trained, sample a random corpus position from its
occurrence list. Different fires of the same node see different
positions.

- **Pro:** position diversity for high-mass nodes; over many epochs
  the node accumulates training at many positions.
- **Con:** requires per-node position lists (memory = O(N) total).
  We already have the data from Phase 0's position-map work.

### V1c — All positions per node (mass-amplified expansion)

Each fire of a mass-K node expands into K separate training instances,
one per corpus position.

- **Pro:** complete position coverage; closest to SGD-equivalent
  position exposure.
- **Con:** ~Kx compute increase for high-mass nodes. Loses AGPT's
  aggregation advantage entirely — basically becomes per-position
  training.

**Recommendation:** start with **V1b**. Cheap (same compute as current
AGPT), uses our existing Phase 0 data, gives position diversity.

## Implementation in `agpt_train.cu`

### What changes

The current code applies RoPE via `launch_rope_batched(d_q,
d_rope_positions, ...)`. The `d_rope_positions` array currently
contains within-chunk positions (0..d-1).

The change: populate `d_rope_positions` with **actual corpus
positions** for each chunk's d query positions.

### Data structures needed

Add to the radix-trie format (or load alongside):

- **`canonical_pos[radix_id]`** (v1a) — one int32 per radix node, the
  canonical corpus position for that node.
- **`position_list_offset[radix_id], position_list[]`** (v1b/v1c) —
  CSR-style storage of all corpus positions per radix node. Total
  size O(N).

The Phase 0 position-map binary
(`rnd/seq-len-decouple/gutenberg_5m_d16_pos_to_node.bin`) is the
inverse map (position → nodes); we'd derive the node → positions map
from it in O(N) at startup.

### Per-chunk position computation

For a chunk processing a partition group of radix nodes, the
chunk's query positions need their corpus positions. With one
canonical or one-random position per node:

```
For each query position j in [0, T_q):
    n_idx = query_to_node[j]               // existing
    radix_id = radix_ids[n_idx]            // existing
    anchor_pos = canonical_pos[radix_id]   // V1a: load from trie file
                 or sample_position(radix_id)  // V1b: pick from position_list
    j_offset_in_path = query_to_path_offset[j]   // depth within the radix node's edge
    real_position = anchor_pos + (j_offset_in_path - node.endpoint_depth)
    d_rope_positions[j] = real_position
```

The `real_position` calculation handles radix edges: a radix node
spanning chars at corpus positions [anchor_pos - edge_len + 1,
anchor_pos] needs the right within-edge position assigned to each
query.

### Mass weighting interaction

Under V1b (random position per fire), a mass-K node still fires
**once** per epoch but at a random position each time. Over T epochs,
it accumulates T position samples. With mass-weight=log, the loss is
already scaled to reflect the node's prominence; the per-fire random
position just diversifies which RoPE rotation each gradient step sees.

Under V1c (mass-amplified), the node fires K times. Mass weighting
should be **off** here since each position is now its own training
instance.

### Estimated implementation effort

V1b implementation in `agpt_train.cu`:

- Load per-node position lists at startup (~1 hr)
- Modify RoPE position assignment in the chunk forward pass (~2 hr)
- Same change applied in the backward pass (~1 hr)
- Testing + numerical sanity checks (~2 hr)

Total: ~half-day to a day of CUDA work, mostly bookkeeping.

## ALiBi Variant

Worth considering as a separate or combined experiment.

### Replace RoPE with ALiBi

Drop the RoPE rotation entirely. Replace with attention bias:

    score(q, k) = (Q_q · K_k) / sqrt(d) − m_h · |q_pos − k_pos|

where `m_h` is a per-head fixed slope (typical: 2^(-8h/H) for head h,
H total heads), and `q_pos, k_pos` are the actual corpus positions.

### Why this helps

ALiBi's linear distance penalty *extrapolates*: at unseen distances,
the bias is still well-defined and continues to monotonically
downweight, instead of producing unseen RoPE rotations.

Published result (Press et al. 2022, "Train Short, Test Long"):
ALiBi-trained models extrapolate to 2-4x longer sequences with only
modest degradation; RoPE-trained models collapse at 1.1-1.5x.

### Implementation cost

- Replace `launch_rope_batched` calls with no-ops
- Add ALiBi bias computation in attention kernels
- ~Half-day modification

### Combination with sliding-tree

The ideas compose: sliding-tree provides position-aware *content*
(trie structure trained at varied positions); ALiBi provides
position-aware *distance handling*. Together they target the two
sub-problems of seq_len extrapolation independently.

## First Experiment Plan

### Setup

- Modify `agpt_train.cu` to V1b (random-position-per-fire RoPE)
- Train on Shakespeare 1M, d=16, 100 SE (the known-good pd=1 plateau
  budget). Compare against existing PPL@16 = 4.46 plateau on
  Shakespeare.
- Eval PPL at multiple seq_len: 8, 16, 24, 32, 48, 64.

### Pass / fail

| seq | current d=16 model | sliding-tree d=16 (expected) |
|---|---:|---:|
| 8  | 8.64 | similar |
| 16 | 8.01 | similar (sanity: PPL@d shouldn't regress) |
| 24 | 12.73 | **expect significant improvement** |
| 32 | 13.15 | **expect significant improvement** |
| 48 | 16.98 | improvement |
| 64 | 20.94 | improvement |

- **Pass:** PPL@24 and PPL@32 are substantially better than current
  (no catastrophic collapse). PPL@16 doesn't regress.
- **Fail:** PPL@24/32 same as before. Sliding-tree didn't actually
  fix the position-OOD issue, OR there's another bottleneck.

### If Pass

Move to ALiBi variant as separate experiment. Then combine.

### If Fail

Investigate: is the position-rotation aspect actually the OOD issue,
or is it the attention-window-size aspect? If the latter, sliding-tree
alone can't help and we're back to the pooling architecture.

## Files To Create / Modify

```
rnd/sliding-tree-rope/                              (experiment results)
├── README.md
├── findings.md
└── logs/

src/cuda/agpt_train.cu                              (modified)
src/agpt/radix_trie_reader.cr                       (extended to load position lists)
src/tools/agpt_build_position_lists.cr              (new — emits node→positions side-table)
notes/sliding_tree_rope.md                     (this file)
```

## Open Questions

1. Are corpus positions handled correctly in **virtual-tree** mode
   (when virtual_cycles > 1)? The current trainer's chunk-cycle-shift
   adds an offset on top of within-chunk position. Under sliding-tree
   the cycle-shift may need different semantics.

2. For partition groups, the chunk's query positions come from
   multiple radix nodes at potentially different anchor positions.
   Should each query use its node's anchor, or should the chunk
   choose one anchor and offset queries accordingly?

3. What about edge cases at corpus start (positions < d-1)? Walks
   would extend before position 0. Probably skip those nodes from
   training, or clamp position to 0.

4. Does the existing **mass=1 compaction** mechanism in agpt_train
   need changes? Compaction stores K positions per slot; the position
   per slot was the within-chunk depth before, would become corpus
   position now.

## Estimated Total Effort

- Implementation: 1 day of CUDA work + 0.5 day Crystal side-table
- First training run: 37 min wall (100 SE Gutenberg d=16)
- Eval sweep: 5 min
- Writeup: 0.5 day

**Total to first result: ~2 days.**

If the result is positive, ALiBi-variant adds another ~1 day. If
negative, the cost was 2 days for a decisive go/no-go on the cheapest
seq_len-decoupling path on the table.
