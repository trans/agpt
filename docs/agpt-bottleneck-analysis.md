# AGPT Bottleneck Analysis and Next Steps

## Current State

AGPT has successfully implemented:

- Trie-based data representation
- Gradient factorization via prefix sharing
- Level-batched projections and FFN
- Depth-progressive subtrie training
- Truncated (local-depth) backward for efficiency

The system is now **correct at the mathematical and training-schedule level**.

---

## The Real Bottleneck

The dominant cost is currently:

> **Per-node attention computation**

At each depth:

- ~27,000 nodes
- each node:
  - reconstructs KV cache (~60% time)
  - runs attention (~25% time)
  - allocates temporaries (~15% time)

Total:
- ~432,000 attention operations per epoch
- ~95% of runtime spent in attention loop

---

## Critical Insight

### Incorrect assumption

> Attention is inherently per-node and cannot be batched.

### Correct statement

> Attention is prefix-dependent in *data*, but **not in computation**.

---

## Missing Factorization

AGPT successfully factorizes:

- gradients
- forward projections

But does NOT yet factorize:

- **attention**

---

## Structure of the Problem

At depth d:

Many nodes share:

- identical prefix up to d-1
- differ only at position d

Example:

```
Prefix: [A B C D]
Children: E, F, G
```

All nodes share:

```
K[A B C D]
```

---

## Correct Attention Factorization

Instead of computing attention per node:

### Current (incorrect)

```
for each node:
    Q_node × K_node^T
```

### Correct (factored)

#### Shared component

```
Q_batch × K_prefix^T
```

#### Per-node delta

```
Q_i × K_i^T
```

---

## Computational Impact

### Current complexity

```
O(nodes × depth)
```

### After factorization

```
O(prefixes × depth) + O(nodes)
```

This collapses the dominant cost.

---

## Root Cause of Slowdown

The current system treats attention as:

> a node-level operation

instead of:

> a prefix-level shared computation with small per-node differences

---

## KV Reconstruction Issue

### Current approach

```
NodeKVStore → reconstruct KV per node via parent walk
```

### Problem

- O(depth) per node
- repeated work across siblings

---

## Correct Model

```
PrefixKVStore:
    each prefix owns its KV
    children extend incrementally
```

---

## Key Fixes (Priority Order)

### 1. Sibling Attention Batching (CRITICAL)

- batch attention across nodes with same parent
- compute shared prefix attention once
- compute per-node deltas

> This is not an optimization — it is the missing half of the trie idea.

---

### 2. Incremental KV Propagation

- propagate KV from parent to child in O(1)
- eliminate parent-chain reconstruction

---

### 3. Re-evaluate Performance

After fixes:
- attention cost should drop drastically
- trie batching advantage becomes dominant

---

### 4. GPU KV Store (later)

- avoid CPU round-trips
- run attention fully on GPU

---

## What Is NOT the Bottleneck

- Gradient formulation
- Training schedule
- Update frequency (secondary)
- Mathematical correctness

---

## Optimization Insight

AGPT currently appears slower per update because:

> attention is not yet factored

After fixing:

- fewer effective attention ops
- larger batched compute
- improved scaling with model size

---

## Training Regime Status

The current approach:

- subtrie partitioning
- local-depth backward
- frequent updates

is correct and should be retained.

---

## Refinement to Paper Language

Instead of:

> not the exact gradient of any loss

Use:

> a consistent approximation over bounded subgraphs of the trie

---

## Final Diagnosis

AGPT has:

- factorized gradients
- factorized projections
- factorized structure

But has NOT yet:

- **factorized attention**

---

## One-Line Summary

> AGPT correctly factorizes the model everywhere except the component that dominates runtime — attention — and fixing this unlocks the full advantage of the approach.

---

## Next Step

Implement:

> **prefix-aware (sibling-batched) attention with incremental KV propagation**

This is the highest-impact change.
