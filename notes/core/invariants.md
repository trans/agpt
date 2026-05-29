# AGPT Invariants and Operational Spec

## Purpose

This note defines the **non-negotiable mathematical invariants** of AGPT so implementation work does not drift into a merely convenient but incorrect training regime.

The central point is:

> **AGPT is defined by subtree-local gradient factorization, not by level batching, chunking, or cache convenience.**

Levels, chunks, batching, and caching are implementation tools.  
The **bounded subtree** is the actual training object.

---

# 1. Core Training Object

AGPT does **not** train on:

- the full trie at once
- one branch at a time
- one depth layer at a time

AGPT trains on:

> **bounded subtries with fixed weights during the entire forward + backward pass**

Each update is defined over one such subtree.

---

# 2. Core Gradient Idea

Let a prefix node or shared prefix block be represented by \(p\).

Let the branches/leaves below that prefix inside the current bounded subtree contribute downstream gradient terms \(g_s\), where \(s\) ranges over descendant paths/suffixes in the subtree.

Let \(J_p\) denote the Jacobian of the shared prefix computation.

Then the AGPT factorization is:

\[
J_p \cdot \left(\sum_{s \in \mathrm{subtree}_d(p)} g_s\right)
=
\sum_{s \in \mathrm{subtree}_d(p)} \left(J_p \cdot g_s\right)
\]

## Meaning

This is the key AGPT identity.

It says:

- many branches share the same prefix
- each branch contributes a downstream gradient signal
- those branch signals may be **summed first**
- then the shared prefix Jacobian is applied **once**

Equivalent interpretation:

> **combine all branch complaints below a shared prefix, then backpropagate through the shared prefix once**

This is the mathematical reason prefix sharing helps not only on the forward pass, but also on the backward pass.

---

# 3. Main Invariants

## Invariant 1 — Subtree is the update unit

The **bounded subtree** is the unit over which:

- forward is defined
- backward is defined
- gradient aggregation is defined
- the optimizer update is applied

A level is **not** the update unit.  
A chunk is **not** the update unit.

---

## Invariant 2 — Weights are fixed within a subtree pass

During forward + backward over a subtree:

\[
W = \text{constant}
\]

No optimizer step may occur:

- between levels of the same subtree
- between chunks of the same subtree
- between prefix and suffix portions of the same subtree

This is required for gradient correctness.

---

## Invariant 3 — Shared prefix gradients are aggregated before update

Within a subtree, gradients from all descendant branches that share a prefix must be aggregated before that shared prefix is updated.

This is exactly what the factorization formula expresses.

Do **not** reduce the process to independent per-branch updates if that destroys prefix aggregation.

---

## Invariant 4 — Levels are an execution strategy, not a mathematical replacement

Level-by-level processing may be used to:

- batch matmuls
- improve GPU utilization
- control memory use

But levels may **not** replace the subtree as the training object.

In other words:

> levels may organize compute, but they do not define gradient boundaries

---

## Invariant 5 — Cache correctness is subordinate to subtree correctness

Caching is allowed only if it does not violate the subtree invariants.

In particular:

- cached values are valid **within a fixed weight snapshot**
- cached differentiable activations become stale after an optimizer update
- no implementation optimization may silently redefine the update unit from subtree to level/chunk

---

# 4. Progressive Depth Curriculum

AGPT uses a curriculum over increasing subtree depth.

Let \(d\) denote the current maximum curriculum depth.

Then training proceeds as:

- \(d = 1\): train all bounded subtries up to depth 1
- \(d = 2\): train all bounded subtries up to depth 2
- \(d = 3\): train all bounded subtries up to depth 3
- and so on

This gives:

- many updates
- progressively deeper context
- repeated reuse of shared prefixes
- a natural curriculum from shallow to deep structure

This curriculum is **not** equivalent to one update per epoch.

It is intentionally much more fine-grained.

---

# 5. What Went Wrong in the Drifted Version

A later implementation drift can happen when the priorities are reversed.

Correct priority order:

1. bounded subtree
2. gradient factorization
3. update once per subtree
4. progressive depth curriculum
5. batching / caching / chunking

Incorrect drifted order:

1. batching / caching / chunking
2. level execution
3. update wherever convenient
4. attempt to fit gradients afterward

When that happens, subtree AGPT gets replaced by something like level training or chunk training, and the mathematical heart is lost.

---

# 6. What Caching Is Actually Safe

This must be stated carefully.

## Safe to cache across updates
Only things that are **not** tied to the old differentiable weight state, such as:

- trie structure
- ancestry lists
- masks
- token ids
- partition metadata
- static curriculum assignments

## Not safe to reuse as live differentiable activations across updates
Examples include:

- hidden states computed under old weights
- K/V projections computed under old weights
- prefix activations that later backward passes expect to differentiate through under new weights

These are stale after an update unless recomputed.

---

# 7. Prefix Reuse Rule

Prefix reuse is allowed in two distinct modes:

## Mode A — Live reuse within one subtree pass
Valid.

- weights fixed
- forward/backward both occur within same subtree
- prefix activations remain part of the live computation graph

## Mode B — Reuse across subtree updates
Only valid if the reused prefix state is either:

- recomputed under the current weights, or
- treated as a detached boundary condition rather than a live differentiable activation

This distinction must be explicit in code.

---

# 8. Operational Rule for Updates

The optimizer step may occur only after the subtree has completed:

1. forward over the whole subtree
2. backward over the whole subtree
3. aggregation of shared prefix gradients inside that subtree

Then and only then:

\[
W \leftarrow W - \eta \nabla_W L_{\mathrm{subtree}}
\]

where \(L_{\mathrm{subtree}}\) is the aggregated loss for that subtree.

---

# 9. Minimal Operational Skeleton

```python
for depth in progressive_depths:   # d = 1, 2, 3, ...

    expand_trie_to_depth(depth)

    for subtree in bounded_subtries_up_to_depth(depth):

        # ---- begin subtree training unit ----
        freeze_weights()

        # forward over subtree
        # levels may be batched internally, but weights remain fixed
        activations = forward_subtree(subtree)

        # aggregate losses over leaves / suffixes in subtree
        loss = aggregate_subtree_loss(subtree, activations)

        # backward over same subtree
        loss.backward()

        # one update per subtree
        optimizer.step()
        optimizer.zero_grad()

        # invalidate or detach any stale differentiable caches
        refresh_detach_or_drop_stale_states()
        # ---- end subtree training unit ----
```

---

# 10. Internal Level Batching Rule

This is allowed:

```python
for level in subtree.levels:
    forward_level(level)

for level in reversed(subtree.levels):
    backward_level(level)
```

But only under the condition that:

- the entire level sequence belongs to one subtree pass
- weights are unchanged during that pass
- update happens only after the full subtree backward completes

So level batching is legal **inside** subtree training, not **instead of** subtree training.

---

# 11. Implementation Guardrails

Any implementation should enforce these checks:

## Guardrail A
If an optimizer step occurs before subtree backward completes, error.

## Guardrail B
If cached differentiable prefix activations from an old weight version are reused as though live under a new weight version, error.

## Guardrail C
If a level/chunk boundary is treated as an implicit update boundary for AGPT, error.

## Guardrail D
If branch gradients are being applied independently where prefix aggregation was intended, warning or error.

---

# 12. One-Sentence Definition of AGPT

> **AGPT trains on bounded, fixed-weight subtries; aggregates descendant branch gradients under shared prefixes using the factorized Jacobian identity; and applies one optimizer update per subtree within a progressive-depth curriculum.**

---

# 13. Short Reminder for Future Implementers

If performance work begins to obscure correctness, return to this question:

> **What object is the gradient defined over?**

For AGPT, the answer is:

> **the bounded subtree, not the level, not the chunk, and not the full trie at once**

Everything else must serve that fact.
