# AGPT Extension: Structure-Aware Loss Weighting (Entropy-Based)

## Status

- Core AGPT algorithm: ✔️ stable  
- Attention bottleneck: ✔️ being resolved  
- This feature: **optional extension ("icing on the cake")**

This does NOT change the training algorithm. It modifies only **loss weighting**.

---

## Goal

Leverage trie structure to improve learning by weighting nodes based on **uncertainty (entropy)**.

Window training cannot access this information. AGPT can.

---

## Core Idea

At each prefix node \(p\), instead of using:

\[
L_p = -\sum_x n(p,x)\log \pi(x|p)
\]

we use:

\[
L_p^{weighted} = w(p) \cdot L_p
\]

Where:

\[
w(p) = 1 + \lambda H(p)
\]

---

## Entropy Definition

Compute empirical entropy at node:

\[
H(p) = -\sum_x q(x|p)\log q(x|p)
\]

Where:

\[
q(x|p) = \frac{n(p,x)}{N_p}
\quad\text{and}\quad
N_p = \sum_x n(p,x)
\]

---

## Interpretation

- High entropy → many possible futures → **increase weight**
- Low entropy → deterministic path → **decrease relative importance**

---

## Why This Works

AGPT exposes structure:

- branching nodes = uncertain, information-rich
- unary chains = predictable, low-value for learning

This weighting:
- focuses updates on ambiguous regions
- avoids overtraining deterministic tails

---

## Implementation Requirements

### 1. Entropy computation

At trie build time or preprocessing:

For each node:

```python
H[p] = -Σ_x (n[p,x] / N_p) * log(n[p,x] / N_p)
```

Store:
```python
node.entropy
```

---

### 2. Loss modification

In `weighted_loss.cr` or equivalent:

Replace:

```python
loss += L_p
```

With:

```python
w = 1 + lambda * node.entropy
loss += w * L_p
```

---

### 3. Gradient scaling

Because loss is scaled:

- gradients are automatically scaled by `w`
- no additional changes needed in backward pass

---

## Hyperparameters

### λ (lambda)

Controls strength of effect.

Recommended starting values:

```text
λ = 0.0   # baseline (off)
λ = 0.1   # weak
λ = 0.25  # moderate
λ = 0.5   # strong
```

---

## Optional Normalization (recommended)

To avoid scale drift:

```python
H_norm = H / log(num_children)
```

or

```python
H_norm = H / H_max
```

Then:

```python
w = 1 + λ * H_norm
```

---

## Expected Effects

### Positive

- faster convergence in ambiguous regions  
- improved generalization  
- reduced overfitting on deterministic suffixes  

### Neutral/Low Risk

- minimal overhead (entropy precomputed)
- no change to compute graph

---

## Risks

- λ too large → over-focus on high-entropy nodes  
- may slightly slow convergence on simple patterns  

Mitigation:
- start small (λ ≤ 0.25)
- compare against baseline

---

## Evaluation Plan

Run side-by-side:

| Metric | Baseline | Entropy-weighted |
|--------|--------|----------------|
| Held-out CE | | |
| Convergence speed | | |
| Stability | | |

---

## Important Constraints

Do NOT:

- change trie structure  
- change batching strategy  
- modify backward logic  
- introduce new objectives  

This is strictly:

> **a scalar multiplier on existing node loss**

---

## Minimal Implementation Diff

```diff
- loss += L_p
+ w = 1 + lambda * node.entropy
+ loss += w * L_p
```

---

## Summary

> Use entropy of trie nodes to weight loss, emphasizing uncertain branching regions and de-emphasizing deterministic chains.

This leverages structure that only AGPT exposes.
