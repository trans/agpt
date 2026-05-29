# Full D-State Continuation Using the Real D-Trie Only

## Goal

Upgrade from **leaf-token-rooted virtual continuation** to **full D-state continuation** without building a giant explicit virtual tree.

The key idea is:

- keep the **real depth-D trie** as the only stored tree
- treat its observed depth-D nodes as **suffix states**
- implement continuation by **rolling the last-D-token window** and looking up the resulting suffix state in an index

So we do **not** materialize virtual copies of the tree.

---

## Core Distinction

### Current naive version
At a depth-D leaf ending in token `u`, continue from the root subtree of `u`.

This means:

- local structure inside a D-segment is order-D
- continuation across segment boundaries is only order-1

### Full D-state version
At any point, the active state is the **entire last-D-token suffix**, not just the final token.

This means:

- local structure is still order-D
- continuation across boundaries also remains order-D

---

## Important Clarification

We are **not** building a larger explicit tree.

Instead:

- the real D-trie contains all observed substrings of length `<= D`
- each observed depth-D node is treated as a reusable suffix state
- continuation is implemented as a **state transition**
- the “virtual depth > D” walk is represented by a **sequence of real D-state lookups**

So the deeper virtual walk is simulated by iterating over real states, not by copying the tree.

---

## What Is Reused

Even if Codex says “the virtual copies cannot literally reuse compute as copies of the same expanded tree,” there is still real reuse:

### 1. Prefix sharing inside the real D-trie
This remains exactly as before.

If several states share a parent prefix, compute for that parent can still be reused for child expansions.

### 2. State reuse across time
With full D-state continuation, the next state is derived from the current state by shifting the suffix window.

So we reuse:

- the current suffix state
- its parent chain
- cached node statistics
- cached subtree computations where applicable

### 3. No explicit virtual tree materialization
This is the biggest savings.

We do not build:

- repeated copies of D-subtrees
- a giant depth-`L` virtual tree
- explicit boundary-stitched expansions

We only keep:

- one real trie
- one suffix-state index
- one rolling state per active path

---

## Data Structures

### Real trie node

Each trie node should have at least:

```python
class TrieNode:
    token                 # token for this edge/node
    parent                # parent pointer
    children              # token -> TrieNode
    depth                 # depth from root
    count                 # visit count / frequency
    next_token_counts     # sparse distribution over next tokens
    suffix_tuple          # optional: exact tuple for depth-D nodes
    state_id              # optional stable identifier
```

### State index

A lookup from observed suffix tuples of length `D` to the corresponding trie node:

```python
state_index[(tok_1, tok_2, ..., tok_D)] -> TrieNode
```

This is the crucial upgrade.

Only **observed** suffix tuples need to be stored.

---

## Active State

At time `t`, maintain:

```python
buffer_t = last D tokens ending at t
state_t  = state_index.get(buffer_t)
```

If `len(buffer_t) < D`, use the corresponding shallower trie node.

---

## State Transition

Suppose current state is:

```text
(x_{t-D+1}, ..., x_t)
```

and the next token is:

```text
x_{t+1}
```

Then the next state is:

```text
(x_{t-D+2}, ..., x_t, x_{t+1})
```

So transition is:

1. drop oldest token
2. append new token
3. look up resulting tuple in `state_index`

### Pseudocode

```python
def shift_state(buffer, next_token, D, state_index):
    if len(buffer) < D:
        new_buffer = tuple((list(buffer) + [next_token])[-D:])
    else:
        new_buffer = tuple(list(buffer[1:]) + [next_token])

    return new_buffer, state_index.get(new_buffer)
```

---

## If Exact Depth-D State Is Missing

Some shifted tuples may not exist as observed depth-D states.

Then back off by suffix length:

```python
(x2, x3, ..., xD, y)   # try depth D
(x3, ..., xD, y)       # try depth D-1
...
(y)                    # try depth 1
root                    # fallback
```

### Pseudocode

```python
def lookup_with_backoff(buffer, state_index, trie_root):
    toks = list(buffer)
    for k in range(len(toks), 0, -1):
        key = tuple(toks[-k:])
        node = state_index.get(key)
        if node is not None:
            return node
    return trie_root
```

This backoff is for **state lookup**, not necessarily final probability blending.

---

## Why This Is Full D-State Continuation

Because what is carried forward is:

```text
the full last-D-token suffix
```

not merely the leaf token.

So:

- `(a, b)` and `(c, b)` are different states
- continuation after them can differ
- no boundary collapse to order-1

---

## Suffix Chain for Prediction or Blending

Once the current state node is found, its ancestry already gives all suffix levels:

```text
s^(D), s^(D-1), ..., s^(1), root
```

So if we later want:

- count-based backoff
- learned suffix-depth blending
- suffix-depth distillation targets

we already have the required chain via `parent` links.

No extra tree is needed.

---

## Training Interpretation

The sequence is no longer represented as one deep virtual branch.

Instead, it is represented as a sequence of rolling suffix states:

```text
state_1 -> state_2 -> state_3 -> ... -> state_T
```

Each state is a real node from the real D-trie (or a backed-off shallower node).

So “virtual depth > D” is simulated by time evolution over real states.

---

## Interaction With Prefix Reuse

This does **not** remove the parent-prefix reuse benefit.

Inside the trie, if your current trainer already does:

```text
compute(parent) once
reuse for each child
```

that remains valid.

What changes is only the continuation rule between training positions:

### Before
At boundary, continue from root subtree of final token.

### After
At boundary, continue from shifted full suffix state.

So:

- parent-prefix reuse still works locally
- state transitions become richer
- no explicit virtual-copy tree is required

---

## Why This Is Better Than Explicit Virtual Copies

Explicit virtual copies would create:

- repeated structure
- repeated storage
- repeated traversal overhead

Full D-state continuation instead gives:

- one canonical real trie
- one sparse state index
- one rolling suffix buffer
- exact order-D boundary continuation

So the “virtual tree” is now an **implicit process**, not a stored object.

---

## Minimal Implementation Plan

### Phase 1
Keep the current real D-trie as-is.

### Phase 2
Add `state_index` for observed depth-D suffix tuples.

### Phase 3
At training time, maintain rolling token buffer and current state.

### Phase 4
Replace leaf-token-rooted continuation with:

- shift rolling suffix buffer
- exact lookup in `state_index`
- suffix backoff if needed

### Phase 5
Preserve subtree-scoped training and parent-prefix reuse exactly as before.

---

## Minimal Pseudocode Sketch

```python
buffer = ()
state = trie_root

for next_token in training_stream:
    # update rolling suffix
    if len(buffer) < D:
        buffer = tuple((list(buffer) + [next_token])[-D:])
    else:
        buffer = tuple(list(buffer[1:]) + [next_token])

    # exact depth-D lookup, with backoff if needed
    state = lookup_with_backoff(buffer, state_index, trie_root)

    # local training logic continues from this real trie node
    train_from_state(state)
```

---

## Optional Later Optimization: Suffix Links

Once tuple lookup works, optimize state shift by storing suffix links.

For a depth-D node `(a, b, c, d)`:

- suffix link -> `(b, c, d)`
- from there, follow child `e` to get `(b, c, d, e)`

So transition becomes:

```text
(a,b,c,d) --suffix_link--> (b,c,d) --child e--> (b,c,d,e)
```

This can reduce repeated tuple hashing and make shifting more elegant.

But it is **not necessary** for the first implementation.

---

## Practical Summary

### Do not do:
- build giant explicit virtual trees
- copy D-subtrees recursively
- treat virtual depth as stored structure

### Do do:
- keep one real D-trie
- add suffix-state indexing
- roll the last-D-token state through time
- look up the resulting observed suffix state
- use parent links for ancestor chain / blending later

---

## One-Line Summary

**Full D-state continuation means replacing leaf-token boundary stitching with rolling exact last-D-token state lookup in the real D-trie, so virtual depth is simulated by state transitions rather than explicit tree copies.**


-----

# Preserving Prefix Reuse Under Full D-State Continuation

## Goal

Implement **full D-state continuation** (rolling suffix of length D) while **retaining prefix-recursive computation reuse**.

Key requirement:

> Do NOT degrade into “lookup-only.”  
> Maintain recursive representations so shared prefixes are computed once and reused.

---

## Core Idea

Augment the real D-trie with **suffix-parent links** so that a shift update can be computed by:

1. **drop left** via `suffix_parent`
2. **append right** via normal child expansion

This preserves the fundamental recursion:

    r(parent) → r(child) = F(r(parent), token)

even across sliding-window updates.

---

## Definitions

Let a node represent a token sequence:

    p = (t1, t2, ..., tk),  with k ≤ D

Store on each node:

- `parent(p) = (t1, ..., t{k-1})`
- `suffix_parent(p) = (t2, ..., tk)`  (if k ≥ 1; root otherwise)
- `children[p][u] = (t1, ..., tk, u)` if depth < D
- `repr(p)` = computed representation for prefix `p`

---

## State

At time `t`, maintain:

    state_t = node for suffix (x_{t-D+1}, ..., x_t)

If fewer than `D` tokens seen, use the corresponding shallower node.

---

## Transition (Full D-State Shift)

Given current state `p` (depth D) and next token `u`:

    q = suffix_parent(p)          # drops the oldest token
    next_state = children[q][u]   # append new token

Formally:

    next_state(p, u) = child(suffix_parent(p), u)

This is the exact rolling-window update.

---

## Representation Update (Prefix-Reuse Preserved)

Compute representation of the next state by **append-only recursion from the suffix parent**:

    r(next_state) = F(r(q), u)

Where:

- `q = suffix_parent(p)`
- `F` is your existing prefix extension function

### Important

- Do NOT recompute from root
- Do NOT use `r(p)` directly (it contains the dropped token)
- Always reuse `r(q)` (the correct length-(D-1) prefix)

---

## Why This Preserves Reuse

Original reuse (within subtree):

    r(a, b) computed once → reused for (a, b, c), (a, b, d), ...

New reuse (across time via suffix):

    r(a, b, c, d)
      ↓ suffix_parent
    r(b, c, d)  (already computed)
      ↓ append u
    r(b, c, d, u)

So reuse shifts from:

- **branch fan-out** (parent → many children)

to:

- **overlap reuse** (state_t → state_{t+1} share D−1 tokens)

Both use the same recursion `F`.

---

## Node Construction Requirements

When building the D-trie:

1. Create all nodes for substrings of length ≤ D (as usual).
2. For each node `p = (t1,...,tk)`, set:

       suffix_parent(p) = node for (t2,...,tk)
       (or root if k ≤ 1)

3. Ensure `children[q][u]` exists for observed transitions.

---

## Handling Missing Transitions

If `children[q][u]` does not exist:

1. Back off on the state:

       q = suffix_parent(q)

2. Retry `children[q][u]`
3. Repeat until found or reach root

Pseudocode:

```python
def next_state(p, u):
    q = suffix_parent(p)
    while q is not None:
        if u in children[q]:
            return children[q][u]
        q = suffix_parent(q)
    return root
```

Representation:

```python
r(next_state) = F(r(q_used), u)
```

---

## Training Loop (Minimal Sketch)

```python
state = root
repr_state = r(root)

for u in token_stream:
    # shift via suffix-parent + append
    q = suffix_parent(state)

    while q is not None and u not in children[q]:
        q = suffix_parent(q)

    if q is None:
        q = root

    next_state = children[q][u]

    # prefix-recursive update
    repr_next = F(r(q), u)

    # training step uses repr_next
    train_on(repr_next)

    state = next_state
    repr_state = repr_next
```

---

## Key Properties

- **Exact full D-state continuation**
- **No virtual tree materialization**
- **O(1)-ish incremental update per step**
- **Prefix recursion preserved**
- **Reuse via suffix overlap instead of subtree fan-out**

---

## What This Is NOT

- Not “lookup-only” (we still compute via `F`)
- Not root-reset stitching (no loss to order-1)
- Not explicit large virtual tree

---

## One-Line Summary

**Full D-state continuation preserves prefix reuse by updating each new state from the representation of its suffix-parent and appending one token, rather than recomputing from root or collapsing to the last token.**

