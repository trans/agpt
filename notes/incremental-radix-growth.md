# Incremental Radix Growth — Design Specification

## Goal

Replace `growth_build_radix_view_v2` (currently rebuilds the entire dense
radix structure from scratch on every growth step) with an incremental
maintenance scheme. Each growth step should do work proportional to
the trie mutations between steps, not proportional to total trie size.

Empirically: full rematerialization is ~25% of total growth-training
wall time. With the design below, it drops to roughly the cost of the
ingest phase itself.

## What we have today

Two structures coexist:

1. **`GrowthTrieStateV2`** (uncompressed): hash-map-keyed trie. Each node
   has `children: unordered_map<token, node_id>` and a count distribution.
   `growth_ingest_until_v2(frontier)` walks new corpus chars and updates
   this structure in-place. **This is already incremental.**
2. **`RadixTrieStructure`** (dense compressed): the GPU-facing format.
   Built fresh from the uncompressed trie at every growth step by
   `growth_build_radix_view_v2`. **This is the 25% waste.**

The incremental scheme keeps the uncompressed trie as-is and replaces
the rebuild with an *edit log + persistent dense arrays*.

## Core idea: stable monotonic IDs, append-only dense arrays

Three observations that make this clean:

1. **Growth training only adds.** No nodes are ever deleted, no edges
   ever lengthen. Edges can only stay the same length or *shrink*
   (when split). Counts only increment. This is a much friendlier
   problem than a general mutable radix trie.
2. **Radix node IDs can be made monotonic and stable.** Assign each
   new node a sequentially-increasing ID at creation. **Never
   renumber existing IDs.** All dense arrays grow append-only; no
   middle-insertions, no shifts.
3. **`edge_tokens_flat` tolerates orphans.** When an edge shrinks (a
   split), the tail bytes are simply leaked. A compaction pass can
   reclaim them every N growth steps if needed (probably never, given
   how little space splits leak in practice — usually 1-3 bytes per
   split).

## Three ops cover all radix mutations

```
ADD_EDGE(parent_id, edge_tokens, count_dist)
    Allocate a new radix node ID. Append edge_tokens to
    edge_tokens_flat. Append count_dist to counts_tok/counts_val.
    Update parents[new_id], edge_starts[new_id], edge_lens[new_id],
    counts_offset[new_id..], etc.

SPLIT_EDGE(radix_id, split_at_depth, suffix_edge_tokens, suffix_count_dist)
    Existing radix node `radix_id` has a path-from-parent of length L.
    A new ingest now diverges at depth split_at_depth ∈ [1, L).
    Action:
      1. Allocate a new radix node ID (call it `suffix_id`).
      2. edge_lens[radix_id] := split_at_depth     # shorten the original
         (the trailing edge_tokens are leaked in edge_tokens_flat;
          NOT freed, just unreferenced)
      3. Append suffix_edge_tokens (= original edge[split_at_depth..L])
         to edge_tokens_flat for `suffix_id`.
      4. Append suffix_count_dist (the original counts of `radix_id`)
         to the counts arrays for `suffix_id`.
      5. Clear counts of `radix_id` (it's now an internal node).
      6. children_of[radix_id] = {suffix_id, new_diverging_child_id}
         The new diverging child is added via a subsequent ADD_EDGE.

INC_COUNT(radix_id, token, delta)
    Existing endpoint sees more mass on `token`. Update counts_val
    for the (radix_id, token) pair. If `token` is a new continuation
    not previously in counts_tok[radix_id], append it.
```

These three ops cover every mutation a growing radix trie can undergo.
The log of `(op, args)` between two growth steps is exactly the delta.

## Algorithm: ingest → delta → apply

Per growth step:

1. **Ingest** (existing `growth_ingest_until_v2`): walk new corpus
   characters, updating the uncompressed trie. Mark each
   uncompressed-trie node touched by an ingest as "dirty" with a
   reason flag:
   - `NEW_NODE`: this node didn't exist last step
   - `COUNT_DELTA`: this endpoint saw new mass
2. **Compute delta**: walk the set of dirty uncompressed nodes, in
   any topological order (parent before child works). For each dirty
   node, decide:
   - If parent's radix-node has an edge that this node extends → emit
     `ADD_EDGE` (the most common case)
   - If parent's radix-node has an existing edge whose interior we now
     diverge from → emit `SPLIT_EDGE`, then `ADD_EDGE` for the new
     divergent branch
   - If this is just an existing endpoint with new mass → emit
     `INC_COUNT`
3. **Apply**: walk the op list, mutating the dense arrays. Because all
   ops are append-only or in-place (no shifts), this is O(|delta|),
   not O(|trie|).

## Data structure changes

### Uncompressed trie
Add one field per node:

```cpp
struct GrowthNodeV2 {
    ...existing...
    int radix_id = -1;        // -1 = not yet in radix structure;
                              // else, the radix ID this node belongs to
                              // (an edge in radix space spans multiple
                              // uncompressed nodes; all of them share
                              // the same radix_id once that edge is created)
    bool dirty = false;       // touched this growth step
    int  edge_offset = 0;     // depth within the radix edge (0 for head,
                              // edge_lens[radix_id]-1 for tail)
};
```

`radix_id` lets us look up "what radix node does this uncompressed-trie
node belong to" without searching. `edge_offset` tells us whether we're
at the head, middle, or tail of the radix edge — load-bearing for the
split decision.

### Dense radix arrays
Keep the same arrays, but mark them as **growable** (use
`std::vector<int>` not raw arrays, or `realloc()` periodically when
near capacity). The dense arrays now own their growth.

Add three extra fields:

```cpp
struct RadixTrieStructure {
    ...existing dense arrays...
    int next_radix_id = 1;             // monotone, increments on ADD_EDGE
    long long edge_tokens_capacity;     // total allocated; >= total_edge_chars
    long long counts_capacity;          // same for counts_tok/counts_val
    // Leaked space tracking (informational only):
    long long edge_tokens_leaked = 0;   // sum of (orig_edge_len - new_edge_len)
                                        // across all SPLIT_EDGE ops
};
```

## Worked example: single char, no split

Initial uncompressed trie: root → child[a] → child[b]
Initial radix: one node R₁ with edge "ab", count_dist = {a→1}
(suppose `a` is the next-char at the endpoint after the prefix "ab".)

New ingest sees one more occurrence of "ab" at corpus position 100.

Walk: root → a → b. Both nodes already exist. `b`'s count gets a token
update.

Delta:
- The uncompressed `b` node is marked dirty with `COUNT_DELTA`.

Op:
- `INC_COUNT(R₁, 'a', +1)`  (or whatever token follows "ab" at pos 100)

Apply: edit `counts_val` for (R₁, 'a'). O(1).

## Worked example: edge split

Initial radix: R₁ with edge "cat" (3 chars), count_dist = {... endpoint counts ...}
Uncompressed: root → c → a → t (all marked radix_id=R₁, edge_offset=0..2)

New ingest sees the prefix "carry". When we walk root → c → a → r,
the `r` is a NEW uncompressed node — child of `a`. But `a` is in the
middle of R₁'s edge (edge_offset=1, edge_lens[R₁]=3).

This is a mid-edge divergence. Emit:

1. `SPLIT_EDGE(R₁, split_at_depth=2, suffix_edge="t", suffix_count_dist=<old R₁ counts>)`
   - Allocate R₂.
   - edge_lens[R₁] := 2 (so R₁ now represents "ca")
   - edge_tokens_flat: R₁'s edge tail "t" is leaked; R₂ gets "t" appended.
   - counts of R₁ move to R₂; R₁'s counts become empty (internal node).
   - children_of[R₁] gains R₂ (the "ca→cat" continuation).

2. `ADD_EDGE(R₁, edge_tokens="r"..., count_dist={...})`
   - Allocate R₃.
   - Append "r..." to edge_tokens_flat for R₃.
   - children_of[R₁] gains R₃ (the "ca→car..." continuation).

3. Update uncompressed nodes:
   - root→c, root→c→a stay as radix_id=R₁
   - root→c→a→t becomes radix_id=R₂, edge_offset=0
   - root→c→a→r becomes radix_id=R₃, edge_offset=0

Apply cost: 2 new entries appended, 1 entry's edge_len updated, 1 entry's
counts cleared, 2 children pointers added. All O(1).

## Compaction (optional, deferred)

Edge_tokens_flat leaks space on each split (1 to L-1 bytes). After
many splits, this could matter for memory. A compaction pass:
- Walk all radix nodes
- Emit fresh edge_tokens_flat, counts_tok, counts_val arrays without leaks
- Update edge_starts[], counts_offset[]
- All other arrays unchanged (parents[], edge_lens[], children pointers)

Run compaction every N growth steps OR when leaked > X% of capacity.

For the current Shakespeare/Gutenberg sizes, the leak after a full
training run is typically < 1% of edge_tokens_flat. Compaction is
probably never needed in practice — but is available if it ever is.

## Correctness invariants

To check at the end of each growth step (cheap to verify):

1. `next_radix_id == radix_count + 1`
2. `edge_lens[r] >= 1` for all r
3. `counts_offset[r+1] - counts_offset[r]` matches the size of node r's
   count distribution
4. For every uncompressed-trie node N: if N.radix_id == R, then the
   tokens at depths [N.edge_offset .. N.edge_offset + remaining] of R's
   edge match the path from R's first ancestor uncompressed-node to N
5. For every radix node R with edge "x₁x₂...xL" and parent P: walking
   the uncompressed trie from P's tail by tokens x₁, x₂, ..., xL lands
   on a node with radix_id=R, edge_offset=L-1

These invariants are *checkable* after each apply step; useful for
debugging during initial implementation, can be #ifdef'd out in
release.

## What this is NOT

- **Not** a "rebuild affected subtrees" partial scheme. Those are still
  O(subtree_size) per split; this is O(1).
- **Not** a complete rewrite of the trie data structures. The dense
  array layout stays; mutation semantics change.
- **Not** a thread-safe / concurrent data structure. Growth-training
  ingest is single-threaded; ops apply sequentially.
- **Not** dependent on `--anc-grad` or any specific optimizer flag.
  It's a pure data-structure improvement.

## Test plan

1. **Unit-test the three ops** in isolation. Construct a radix
   structure by hand, apply ops, check invariants.
2. **Compare against rebuild for parity** on small corpora. Run
   ingest + incremental apply, then ingest + full rebuild. Assert
   the dense arrays are byte-identical (after compaction; before
   compaction, the leaked bytes will differ).
3. **End-to-end PPL parity** on Shakespeare 1M growth training: a
   trained model with incremental should produce PPL within seed
   noise of the rebuild version.
4. **Wall-time profile**: confirm the 25% disappears.

## Effort estimate

About a day of careful work. Most of it is the `compute_delta` step
(walking dirty uncompressed nodes and emitting the right ops); the
`apply_delta` step is mechanical given the three ops above. Add half a
day for invariant checks + parity tests against the rebuild version.

## What NOT to take shortcuts on

The temptation will be:

- "Just track which subtrees changed and rebuild those." This is the
  partial-rematerialization compromise. **Don't.** It's the same
  conceptual mess as full rebuild, just with smaller batches.
- "Skip the dirty-bit tracking and walk the whole uncompressed trie
  each step." This loses the incremental property; you're back to
  full rebuild.
- "Just renumber radix_ids on split — it's not that bad." It IS that
  bad. Renumbering means O(N) work on splits in the worst case, and
  the renumbering propagates to parent pointers, children lists, the
  uncompressed trie's radix_id field, and any GPU-side state holding
  radix IDs. Stable monotonic IDs are the right call.
