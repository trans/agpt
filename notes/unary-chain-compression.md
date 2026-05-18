# AGPT Optimization: Unary Chain Compression

## Core Observation

Yes — in a prefix trie, once a branch stops branching, its suffix becomes **unary**:

- each node has exactly one child
- no more prefix sharing exists beyond that point
- the structure is no longer really a tree there — it is a **linear chain**

So the trie can be decomposed into:

1. **Branching regions** — where sharing exists and batching matters
2. **Unary tails** — where no further sharing exists and the path can be compressed

---

## Why This Matters

Branching nodes are where AGPT gets its structural advantage.

Unary chains do **not** benefit from sibling batching because they have:

- one parent
- one child
- one path
- no alternative suffixes

So treating unary chains as ordinary trie nodes causes:

- unnecessary node objects
- unnecessary attention calls
- unnecessary parent-chain walks
- poor GPU utilization

---

## Main Idea

> **Collapse maximal unary chains into compressed path segments**

Instead of storing:

```text
A -> B -> C -> D -> E
```

store:

```text
A -> [B, C, D, E]
```

That compressed edge is a **path segment**.

---

## Structural Rule

A node starts a compressible unary chain if:

- it has exactly one child
- that child also has exactly one child
- and so on until:
  - a leaf is reached, or
  - a branching node is reached

---

## Definitions

### Branching node
A node with:
- 0 children (leaf), or
- 2+ children, or
- designated root/start boundary

### Unary chain
A maximal sequence of nodes where each internal node has exactly 1 child

### Compressed segment
A stored representation of the token sequence along a unary chain

---

## Example

Original trie fragment:

```text
root
 └── the
      └── cat
           ├── sat
           │    └── on
           │         └── mat
           └── slept
```

Compressed form:

```text
root
 └── [the, cat]
      ├── [sat, on, mat]
      └── [slept]
```

Or, if you only compress below branching points:

```text
root
 └── the
      └── cat
           ├── [sat, on, mat]
           └── [slept]
```

---

## Recommended Strategy

### Keep explicit nodes for:
- root
- branching points
- leaves
- subtrie partition boundaries

### Compress:
- maximal unary chains between those points

This preserves the high-value trie structure while eliminating low-value node overhead.

---

## Computational Consequences

### Before compression
Each unary node requires:
- node lookup
- KV reconstruction
- attention call
- backward bookkeeping

### After compression
A unary chain can be processed as:
- one contiguous token segment
- one sequential mini-window
- one fused forward/backward block

This reduces:
- number of nodes
- number of attention invocations
- parent-chain traversal cost
- memory traffic

---

## Attention Implication

This is the key distinction:

### Branching regions
Use:
- sibling attention batching
- shared prefix KV reuse

### Unary tails
Use:
- sequential segment processing
- no sibling batching needed
- treat as a normal contiguous token run

So AGPT becomes a **hybrid system**:

1. **Trie mode** for branching regions
2. **Sequence mode** for unary chains

---

## Training Interpretation

Unary chain compression does **not** change the training objective.

It only changes the execution structure.

A compressed segment still represents the same ordered token sequence and same next-token targets as the original chain of unary nodes.

So this is a structural optimization, not a mathematical change.

---

## Safe Compression Boundaries

Do **not** compress across:

- subtrie update boundaries
- explicit batching boundaries
- root partition boundaries
- nodes where you need separate loss accounting
- nodes where empirical next-token distributions differ

Compression is safest when the chain is strictly unary and each node's continuation is unique.

---

## Suggested Data Structure

### Branch node
```python
class BranchNode:
    token_id
    children            # branching children or compressed edges
    counts              # empirical next-token counts
    prefix_state_ref
```

### Compressed edge / segment
```python
class CompressedSegment:
    token_ids           # [t1, t2, t3, ...]
    next_node_ref       # next branching node or leaf
```

Or in a columnar layout:

- `segment_start_offsets`
- `segment_lengths`
- `segment_token_buffer`
- `segment_target_counts`
- `segment_next_node`

---

## Build-Time Compression Algorithm

### Input
Ordinary trie

### Output
Compressed trie with:
- explicit branch nodes
- compressed unary segments

### Pseudocode

```python
def compress_from(node):
    if node.is_leaf():
        return Leaf(node)

    if node.num_children() != 1:
        new_node = BranchNode(node)
        for child in node.children:
            new_node.add_edge(compress_edge(child))
        return new_node

    # unary entry points are usually handled by parent edge compression
    return compress_edge(node)


def compress_edge(node):
    tokens = []

    current = node
    while current.num_children() == 1 and not current.is_leaf() and not is_boundary(current):
        tokens.append(current.token_id)
        current = current.only_child()

    # include terminal unary node token as well if desired
    tokens.append(current.token_id)

    return CompressedSegment(
        token_ids=tokens,
        next_node_ref=compress_from_children_or_leaf(current)
    )
```

---

## Runtime Execution Model

For each training subtrie:

### If edge is branching:
- batch children normally
- do sibling attention batching

### If edge is a compressed unary segment:
- process the segment as a contiguous token run
- extend KV incrementally along the segment
- accumulate loss across positions
- optionally fuse into one mini forward/backward call

---

## Forward Pass over a Compressed Segment

Given:
- cached parent prefix state
- token segment `[t1, t2, ..., tk]`

Do:

```python
state = parent_state
for t in segment:
    state = step(state, t)
    compute_loss_if_needed(state)
```

This is equivalent to ordinary sequence processing, but without explicit trie-node overhead at each step.

---

## Backward Pass over a Compressed Segment

If using local-depth / truncated mode:
- backward only through the segment-local computations

If using exact subtrie mode:
- backward through the segment into its parent boundary state

Either way, segment compression reduces bookkeeping overhead.

---

## Best Combined Strategy

The best AGPT execution model is likely:

### 1. Branching frontier
- explicit trie nodes
- sibling-batched attention
- shared prefix reuse

### 2. Unary tails
- compressed path segments
- sequential or fused processing
- incremental KV extension

This gives the trie where it matters, and sequence execution where the trie gives no further benefit.

---

## Expected Benefits

Unary chain compression should improve:

- wall-clock time
- KV reconstruction overhead
- memory efficiency
- GPU utilization
- attention scheduling simplicity

Especially when:
- deep trie regions become mostly unary
- sibling counts shrink with depth
- serialized per-node attention dominates runtime

---

## Recommended Implementation Order

### Phase 1
Detect unary chains and mark them

### Phase 2
Replace unary-node iteration with compressed segments

### Phase 3
Run compressed segments through sequential segment kernels

### Phase 4
Combine with sibling attention batching for branching nodes

---

## Practical Rule of Thumb

> **Use trie execution only while there is actual sharing. Once sharing ends, switch to compressed sequence execution.**

---

## One-Line Summary

> **Most deep trie paths are unary tails; compress them into contiguous token segments and reserve full trie machinery for branching regions only.**
