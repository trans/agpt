# Trie-Native Transformer Training — Formal Model

## Overview

This document formalizes a trie-native training system for language models.

The system decomposes into four layers:

1. **Symbolic Layer (Trie)** — defines prefix topology and empirical distributions  
2. **State Layer (Cache)** — holds active neural prefix states  
3. **Transition Layer (Transformer)** — defines the learned edge operator  
4. **Training Layer (Optimizer)** — defines loss and parameter updates  

This reframes language modeling as:

> **learning a transition operator over a weighted prefix graph**

---

# 1. Symbolic Layer — Trie

## Vocabulary

Let:
```
V = vocabulary
```

## Nodes

Each node `u` represents a prefix:

```
p(u) = (x₁, x₂, …, x_d),  where x_i ∈ V
depth(u) = d
```

Each node (except root) has:

```
parent: π(u)
token:  τ(u)
```

such that:

```
p(u) = p(π(u)) || τ(u)
```

## Counts

Each node stores:

```
c(u)         = number of times prefix occurs
n(u, t)      = count of token t following prefix
```

## Target Distribution

```
q_u(t) = n(u, t) / Σ_s n(u, s)
```

## Trie Structure

```
T = (U, E)

E = { (π(u), u) : u ≠ root }
```

---

## Responsibilities of Trie Layer

- prefix identity
- prefix relationships
- empirical next-token distributions
- counts / weights
- scheduling metadata

---

# 2. State Layer — Prefix Cache

Each active node `u` is assigned a neural state:

```
S(u)
```

## Transformer Parameters

Let:
```
L = number of layers
H = number of heads
D_h = head dimension
d_model = H * D_h
```

## KV Cache Representation

For prefix length `d`:

```
S(u) = { K_ℓ(u), V_ℓ(u) } for ℓ = 1..L
```

where:

```
K_ℓ(u), V_ℓ(u) ∈ ℝ^{H × d × D_h}
```

Optional:

```
h_L(u) = final hidden state
```

---

## Active State Set

Define:

```
A ⊆ U = active nodes
```

Only nodes in `A` have materialized states.

---

## Responsibilities of State Layer

- store active prefix states
- manage KV cache
- handle device placement
- handle eviction / reuse

---

# 3. Transition Layer — Edge Operator

Each edge:

```
e = (u → v), where τ(v) = token
```

defines a transition:

```
Φ_θ : (S(u), τ(v)) → (S(v), z(v))
```

where:

- `S(u)` = parent prefix state  
- `τ(v)` = appended token  
- `S(v)` = new prefix state  
- `z(v)` = logits  

---

## Per-Layer Computation

Let:

```
h₀(v) = embedding(τ(v)) + positional_encoding
```

For each layer ℓ:

### 1. Compute projections

```
q_ℓ = W_Q h_{ℓ-1}
k_ℓ = W_K h_{ℓ-1}
v_ℓ = W_V h_{ℓ-1}
```

### 2. Extend cache

```
K_ℓ(v) = append(K_ℓ(u), k_ℓ)
V_ℓ(v) = append(V_ℓ(u), v_ℓ)
```

### 3. Attention

```
scores = q_ℓ K_ℓ(v)^T / sqrt(D_h)
a_ℓ = softmax(scores)
c_ℓ = a_ℓ V_ℓ(v)
```

### 4. Residual + MLP

```
h_ℓ = Block_ℓ(h_{ℓ-1}, c_ℓ)
```

---

## Output

```
z(v) = W_O h_L(v)
```

---

## Responsibilities of Transition Layer

- define learned edge dynamics
- compute child states
- compute logits

---

# 4. Training Layer — Loss and Optimization

## Model Prediction

```
p_θ(t | p(v)) = softmax(z(v))
```

## Node Loss

```
L_v = w(v) * CE(q_v, p_θ)
```

### Weight Options

Basic:

```
w(v) = c(v)
```

Tempered:

```
w(v) = c(v)^α
```

Enhanced:

```
w(v) = c(v)^α * (1 + β * depth) * (1 + γ * entropy)
```

---

## Batch Loss

For batch `B ⊆ U`:

```
L = Σ_{v ∈ B} L_v
```

---

## Responsibilities of Training Layer

- sampling nodes/edges
- applying loss
- updating parameters θ
- scheduling traversal

---

# 5. Execution Model

## Breadth-First (Level-Synchronous)

Let:

```
U_d = nodes at depth d
```

Execution proceeds:

```
U₀ → U₁ → U₂ → … → U_d
```

At depth `d`:

- parents: `U_{d-1}`
- edges: all `(u → v)` where `u ∈ U_{d-1}`, `v ∈ U_d`

---

## Parallelism

- **within depth:** fully parallel
- **across depth:** sequential (dependency)

---

## Frontier-Based Execution

Define:

```
frontier_prev = U_{d-1}
frontier_next = U_d
```

Algorithm:

1. expand all edges from `frontier_prev`
2. compute `frontier_next`
3. compute losses
4. discard older frontier

---

# 6. Cache Lifetime

## Detached Prefix Mode

Only need:

```
frontier_prev + frontier_next
```

Older depths can be flushed.

---

## Truncated Gradient Mode

Keep:

```
last k frontiers
```

---

## Full Gradient Mode

Requires full chain or recomputation.

---

# 7. Delta-State Representation

Instead of full duplication:

```
S(v) = extend(S(u), Δ(v))
```

Where:

```
Δ(v) = (τ(v), {k_ℓ, v_ℓ}_{ℓ=1..L}, h_L(v))
```

So node state is:

- parent pointer
- token
- local KV delta

Dense tensors are only materialized for active computation.

---

# 8. System Decomposition

## Trie Layer

Owns:
- topology
- tokens
- counts
- distributions

## Cache Layer

Owns:
- active KV states
- memory management
- batching tensors

## Transformer Layer

Owns:
- transition operator Φ_θ
- attention + MLP computation

## Training Layer

Owns:
- sampling
- weighting
- optimization

---

# 9. Full System Definition

The system is defined by:

### 1. Graph
```
T = (U, E)
```

### 2. State Assignment
```
S : A → state space
```

### 3. Transition Operator
```
Φ_θ : state × token → state × logits
```

### 4. Loss
```
L = Σ w(v) * CE(q_v, softmax(z(v)))
```

---

# 10. Interpretation

This system reframes training as:

> **learning a shared transition operator over a weighted prefix graph**

Instead of:
- repeated linear sequences

We train over:
- equivalence classes of contexts
- connected by token transitions

---

# 11. Key Insight

A language corpus is not fundamentally a sequence.

It is:

> **a weighted branching structure of shared prefixes**

The trie makes this structure explicit.

The transformer learns how transitions act on that structure.

---

# 12. Conceptual Summary

- Trie = topology  
- Cache = state  
- Transformer = dynamics  
- Trainer = execution policy  

Together:

> **a distributed computation over prefix space**
