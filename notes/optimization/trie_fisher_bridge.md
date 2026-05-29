# From Counts to Curvature
## How a Prefix Trie Materializes Empirical Fisher Information

*A companion note to "Topological Optimization in Prefix-Trie Architectures (AGPT)."*

---

## Motivation

The original AGPT optimization proposal argued — by structural analogy — that a prefix trie's overlapping parent→child pair statistics serve as a proxy for the Hessian of the loss. The analogy is suggestive, but the claim deserves a rigorous bridge: a derivation that connects the count matrix $M$ sitting in the trie to the embeddings $e$, their gradients, and the second-order geometry of the loss landscape.

This note supplies that bridge. The headline: the trie does not *approximate* the Hessian — it *parameterizes* it, exactly, through the categorical-covariance identity for softmax cross-entropy.

---

## Setup

At node $v$ in the trie, with embedding $e_v \in \mathbb{R}^d$:

- **Empirical distribution from counts:** $q(c \mid v) = M[v, c] / \sum_{c'} M[v, c']$.
- **Model distribution:** $p(c \mid v) = \mathrm{softmax}(W e_v)_c$, where $W \in \mathbb{R}^{|V| \times d}$ is the output projection.
- **Local loss:** cross-entropy of $q$ under $p$:
$$\mathcal{L}_v = -\sum_c q(c \mid v) \log p(c \mid v).$$

Minimizing $\mathcal{L}_v$ is equivalent (up to an additive constant) to minimizing $\mathrm{KL}(q \,\|\, p)$.

---

## First Order: Gradient

Let $z = W e_v$ denote the logits. The gradient of cross-entropy w.r.t. $z$ is the canonical
$$\frac{\partial \mathcal{L}_v}{\partial z} = p - q.$$

Chain rule gives the embedding gradient:
$$\boxed{\frac{\partial \mathcal{L}_v}{\partial e_v} = W^\top (p - q).}$$

**Reading:** the trie row $M[v, \cdot]$ enters the gradient *only* through its difference from the model's predicted distribution. Counts pull the embedding toward producing $q$; the model resists with $p$.

---

## Jacobian: The Softmax Factor

The Jacobian of $p$ w.r.t. logits is
$$\frac{\partial p}{\partial z} = \mathrm{diag}(p) - p\,p^\top.$$

This matrix's structure governs how perturbations in $e_v$ propagate to the output distribution. Two regimes:

- **Concentrated $p$** (one mass near 1, rest near 0): the Jacobian is nearly rank-1, dominated by a single direction.
- **Diffuse $p$** (entropy near $\log |V|$): the Jacobian spreads its action across many directions and is closer to a scaled identity (minus a low-rank correction).

This is where the AGPT note's *branching entropy* intuition gets its first rigorous footing. Entropy is a scalar summary of the *spectrum* of this matrix.

---

## Second Order: The Exact Hessian

For softmax cross-entropy, the Hessian w.r.t. logits has a closed form that is not approximate:
$$H_z = \mathrm{diag}(p) - p\,p^\top.$$

This is exactly the **covariance matrix of the categorical distribution $p$**. By chain rule, the Hessian w.r.t. the embedding is
$$\boxed{H_{e_v} = W^\top \big( \mathrm{diag}(p) - p\,p^\top \big) W.}$$

**This is the bridge.** The distribution sitting at each trie node — its children's frequencies — controls the local Hessian *exactly* via its covariance, conjugated into embedding space by the shared output projection $W$.

The trie's count matrix $M$ does not *proxy* curvature. It parameterizes the distribution $p$ whose covariance *is* the curvature.

---

## The Fisher Connection

The empirical Fisher information at node $v$ is
$$F_v = \mathbb{E}_{c \sim q}\!\left[\nabla_{e_v} \log p(c \mid v) \, \nabla_{e_v} \log p(c \mid v)^\top\right] = W^\top \big( \mathrm{diag}(q) - q\,q^\top \big) W.$$

This is the same matrix as $H_{e_v}$, with $q$ in place of $p$. As training proceeds and $p \to q$, the empirical Fisher and the Hessian coincide.

So the precise statement of what AGPT's topology achieves is:

> **The prefix trie materializes per-node empirical Fisher information distributed across the tree. Node-local parameter updates using these quantities are natural-gradient descent with structural Bayesian smoothing on the Fisher estimate.**

That is the theorem-shaped claim that the structural-analogy framing was reaching for.

---

## Re-grounding the Three Mechanisms

With the bridge in place, the optimizer-in-node proposals from the original AGPT note acquire principled interpretations:

### A. Goodman Mass Anchoring
The empirical $q$ is a *noisy estimator* of the true conditional distribution. When counts at a node are small (especially singletons), $q$ is a poor estimate of the distribution whose covariance defines $H$. Goodman-style smoothing — interpolating $q$ with lower-order backoff — is Bayesian shrinkage of the distribution whose covariance defines the curvature. Anchoring the embedding update to the parent is the natural-gradient consequence: when Fisher is poorly estimated, trust the prior.

### B. Branching Entropy as Isotropic vs. Directional Scaling
Entropy of $q$ is a scalar summary of the spectrum of $\mathrm{diag}(q) - q q^\top$:

- Low entropy ⇒ near rank-1 covariance ⇒ Hessian concentrated along one direction ⇒ take large, directed steps.
- High entropy ⇒ broad spectrum ⇒ Hessian nearly isotropic ⇒ compress step magnitude to avoid representational collapse.

This is preconditioning by local curvature, computed for free from the trie row.

### C. Path Momentum — Open Question
The grandparent → parent → child path-alignment heuristic does not fall out of the categorical-covariance identity. It concerns *cross-node* consistency, not the local Hessian at a single node.

A principled reframing would treat it as an **agreement of empirical Fisher across a prefix path**: the sum or product of per-node score vectors along the trajectory. A high-alignment path indicates that the local curvature directions are mutually consistent and updates can be propagated with confidence; an abrupt change in direction indicates a regime change in the loss surface and warrants damping.

Whether this is the right formalization — or whether momentum should instead come from extending the chain one more link, as the categorical covariance suggests when looking at $M[v, c] \cdot M[c, c']$ — remains open.

---

## What $W$ Is Doing

A satisfying consequence of this derivation: the output projection $W$ is the *only* piece carrying global information about how embedding directions translate to logit-space curvature. Everything else — $p$, $q$, entropy, the covariance structure — is genuinely local to a node.

This is exactly the separation of concerns a tree-distributed optimizer needs. The trie holds the local geometry; $W$ provides the shared coordinate system in which that geometry is expressed. Updates can be node-local with respect to the curvature *structure*, while $W$ mediates the embedding-space *direction* of each step.

---

## Summary

| Quantity | Closed Form | What the Trie Holds |
| :--- | :--- | :--- |
| Gradient w.r.t. $e_v$ | $W^\top (p - q)$ | $q$ via $M[v, \cdot]$ |
| Logit Jacobian | $\mathrm{diag}(p) - p\,p^\top$ | spectrum summarized by entropy |
| Logit Hessian | $\mathrm{diag}(p) - p\,p^\top$ | exactly the categorical covariance |
| Embedding Hessian | $W^\top (\mathrm{diag}(p) - p\,p^\top) W$ | local + global via shared $W$ |
| Empirical Fisher at $v$ | $W^\top (\mathrm{diag}(q) - q\,q^\top) W$ | materialized per node |

The count matrix in a prefix trie is not a structural analogy for the Hessian. It parameterizes the exact local Hessian of cross-entropy loss, through the identity that the Hessian of softmax cross-entropy *is* the covariance of the predictive distribution. AGPT's optimizer-in-node program is, properly understood, natural-gradient descent with empirical Fisher distributed across a materialized tree.
