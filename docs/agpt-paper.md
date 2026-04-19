# Prefix-Trie Factorization of Autoregressive Gradient Descent

**Author:** Thomas Sawyer &nbsp; `transfire@gmail.com`
**Status:** v1.0 preprint &nbsp; · &nbsp; **Date:** 2026-04-19
**Code:** https://github.com/trans/microgpt

---

## Abstract

Autoregressive language-model training draws random windows from the
corpus and treats them as independent examples, requiring the optimizer
to reconstruct — through stochastic sampling — a count-weighted sum of
gradients that is already statically computable from the corpus. We
reformulate training as forward-backward on a **prefix trie**
representation of the corpus. Every unique prefix appears exactly
once; the gradient factorization (shown in §5) makes the
trie-based gradient mathematically equivalent to the expected gradient
of full-window training, up to minibatch noise.

This turns the optimizer loop into one gradient step per branching
subtree, with the trie's structure supplying the training schedule.
On a 1.1 MB Shakespeare corpus with a 108k-parameter transformer
(§11), trie-based training at depth 32 reaches held-out perplexity
**13.17** in **195 Adam steps**, versus **14.51** in **2000 SGD steps**
for a step-sweep-tuned window baseline at matched context (seq_len=32)
— **9% lower PPL with 10× fewer optimizer updates** at matched depth.

The paper has two halves. §1–9 develops the factorization theorem,
derives its computational-complexity implications, identifies valid
training regimes (full-trie, bounded-subtrie, truncated-local-depth),
and discusses systems considerations. §10–13 documents the
implementation that validates the framework on a consumer GPU:
radix compression of unary chains, a per-subtree KV-cache allocation
strategy that brings d=32 training within a 15 GB RAM budget, a
bigram subtree partition that brings peak memory 6× lower for deeper
d, and an auto-LR scaler that preserves the winning recipe across
subtree granularities. §12 states limitations honestly; §13 summarizes
contributions.

---

## 1. Introduction

Modern autoregressive language models are trained by optimizing next-token prediction over large corpora using sliding token windows. While effective, this training paradigm introduces significant redundancy: identical prefixes are recomputed repeatedly across overlapping windows, and gradient contributions from shared contexts are accumulated independently.

This redundancy is structural rather than incidental. Natural language exhibits strong prefix reuse, where many sequences share common initial segments before diverging. However, standard training methods fail to exploit this structure explicitly.

In this work, we show that autoregressive training admits a reformulation over a **prefix trie representation** of the corpus. In this formulation, shared prefixes are represented once, and gradient computation factorizes by commuting aggregation with the chain rule. This yields a mathematically equivalent training objective in which redundant prefix transformations are eliminated and gradient contributions are aggregated at shared nodes.

The resulting formulation exposes a different computational structure for training, in which updates are organized over a directed acyclic graph of prefixes rather than a multiset of independent token sequences.

---

## 2. From Sequences to Prefix Trie

Let a corpus be represented as a collection of token sequences. Instead of treating each sequence independently, we construct a prefix trie $\mathcal{T}$, where each node $p \in \mathcal{T}$ represents a unique prefix.

For each node $p$, define:

- $n(p,x)$: the number of times token $x$ follows prefix $p$
- $N_p = \sum_x n(p,x)$: the total number of continuations

Model recurrence:

$$
h_{p \cdot x} = f_\theta(h_p, x)
$$

Training objective:

$$
L = -\sum_{p \in \mathcal{T}} \sum_x n(p,x)\log \pi_\theta(x \mid h_p)
$$

This objective is equivalent to standard next-token training, but expressed over **unique prefixes** rather than token positions.

---

## 3. Gradient at a Node (Local Form)

Define the error at node $p$:

$$
e_{p,x} = \pi_{p,x} N_p - n_{p,x}
$$

Local gradient:

$$
g_p^{\text{local}} = \sum_x e_{p,x} W_x
$$

This represents the discrepancy between predicted and empirical next-token distributions at the node.

---

## 4. Recursive Backpropagation over the Trie

Gradients propagate through the trie as:

$$
G_p = g_p^{\text{local}} + \sum_x J_{p\to p\cdot x} \cdot G_{p\cdot x}
$$

where:

$$
J_{p\to p\cdot x} = \frac{\partial h_{p\cdot x}}{\partial h_p}
$$

This defines a backward pass over a **shared prefix DAG**, rather than independent sequences.

---

## 5. Core Identity: Gradient Factorization over the Prefix Trie

This section presents the central structural result underlying the prefix-trie formulation of autoregressive training. It formalizes how gradient computation over repeated token sequences can be factorized by exploiting shared prefix structure.

---

### 5.1 Setup

Let $p \in \mathcal{T}$ be a prefix node in the trie.

Let:
- $h_p$ denote the hidden state at node $p$
- $p \cdot x$ denote extension of prefix $p$ by token $x$
- $s$ index suffix paths descending from $p$
- $g_s$ denote the gradient contribution to $\frac{\partial L}{\partial h_{p'}}$ arising from a suffix path $s$

Define the **prefix Jacobian**:

$$
J_p := \prod_{k \in \text{prefix}(p)} \frac{\partial h_{k+1}}{\partial h_k}
$$

This represents the total derivative mapping from the root state $h_\epsilon$ to the state $h_p$, or more generally from any ancestor to $h_p$ depending on context.

---

### 5.2 Per-Path Gradient Formulation

In the standard (sequence-based) formulation, the total gradient at prefix $p$ is expressed as a sum over all suffix paths:

$$\frac{\partial L}{\partial h_p} = \sum_{s \in \text{subtree}(p)} \left( J_p \cdot g_s \right)$$

Here, each suffix path contributes independently, and the shared prefix Jacobian $J_p$ is applied repeatedly.

---

### 5.3 Aggregated Gradient Formulation

In the trie formulation, suffix contributions are first aggregated:

$$G_{\text{suffix}}(p) := \sum_{s \in \text{subtree}(p)} g_s$$

The total gradient is then:

$$\frac{\partial L}{\partial h_p} = J_p \cdot G_{\text{suffix}}(p)$$

---

### 5.4 Core Identity

Combining the above expressions yields the fundamental identity:

$$\boxed{J_p \cdot \left(\sum_{s} g_s\right) = \sum_{s} \left(J_p \cdot g_s\right)}$$

This follows from linearity of multiplication over summation, but its implications are structural in the context of gradient computation.

---

### 5.5 Interpretation

The identity shows that:

- Gradient accumulation over suffixes **commutes** with application of the shared prefix Jacobian
- Shared prefix transformations can be applied **once per node**, rather than once per path
- The trie formulation is mathematically equivalent to the standard formulation, but computationally factorized

Thus:

- Right-hand side corresponds to **standard training**, where each sequence contributes independently
- Left-hand side corresponds to **trie training**, where contributions are aggregated before transformation

---

### 5.6 Consequence: Factorization of Gradient Flow

The identity implies that gradient computation can be reorganized as:

- aggregate suffix gradients at each node
- apply shared prefix transformations once

This yields the recursive form:

$$\frac{\partial L}{\partial h_p} = J_p \cdot \left(\sum_{s \in \text{subtree}(p)} g_s\right)$$

which replaces:

$$\sum_{s} \left(J_p \cdot g_s\right)$$

---

### 5.7 Significance

Although algebraically simple, this identity enables:

- elimination of redundant Jacobian applications
- aggregation of gradient signal prior to transformation
- restructuring of training as computation over a shared prefix DAG

This constitutes the core mechanism by which the prefix-trie formulation achieves:

- reduced computational complexity
- improved gradient efficiency
- lower-variance updates

---

### 5.8 Summary

The central result of this section is:

> The chain rule is compatible with aggregation over shared prefix structure, allowing gradient computation to be factorized over the prefix trie.

This identity provides the mathematical foundation for the trie-based reformulation of autoregressive training.

---
## 6. Computational Implications and Complexity

This section translates the gradient factorization identity established in Section 5 into concrete computational and statistical consequences. The prefix-trie formulation reorganizes training from a path-based process into a node-based computation, eliminating redundancy and concentrating gradient signal.

---

### 6.1 Redundant Computation in Sequence-Based Training

In the standard sequence-based formulation, training proceeds over token positions. Each occurrence of a prefix is treated independently, even when identical prefixes appear multiple times across the corpus.

As a result:

- identical prefix states are recomputed repeatedly
- identical Jacobians are applied multiple times
- gradient contributions are accumulated independently per occurrence

This implicitly expands the prefix structure of the corpus into a multiset of paths, duplicating shared computation.

---

### 6.2 Trie Factorization as Common Subexpression Elimination

The prefix-trie formulation collapses identical prefixes into single nodes. Computation is performed once per unique prefix and reused across all occurrences.

This is analogous to:

- common subexpression elimination in compilers
- dynamic programming over shared subproblems
- evaluation of a directed acyclic graph (DAG)

Key observation:

> The corpus induces a computation graph with shared structure, and the trie formulation evaluates this graph directly.

---

### 6.3 Complexity Shift

Let:

- $T$ denote the total number of token occurrences in the corpus
- $P$ denote the number of unique prefixes

Then:

- Standard training complexity scales with $O(T)$
- Trie-based training scales with $O(P)$

In realistic corpora:

$$
P \ll T
$$

due to repeated prefixes across sequences.

Thus, the trie formulation reduces the number of distinct forward and backward computations required.

---

### 6.4 Gradient Efficiency

From Section 5, the gradient at a prefix node can be written as:

$$\frac{\partial L}{\partial h_p} = J_p \cdot \left(\sum_{s \in \text{subtree}(p)} g_s\right)$$

Rather than applying the prefix Jacobian $J_p$ separately for each suffix path, the trie formulation aggregates suffix gradients first.

Consequences:

- a single Jacobian application per prefix
- larger, aggregated gradient signals
- elimination of repeated transformation of identical contributions

This results in effectively larger gradient steps per update.

---

### 6.5 Variance Reduction

In sequence-based training:

- gradient estimates are computed per occurrence
- updates are stochastic and high variance

In trie-based training:

- gradient contributions are aggregated over all occurrences of a prefix
- updates reflect full empirical counts at that node

Thus:

- gradient variance is reduced
- updates are more stable
- convergence may require fewer iterations

Each prefix update incorporates all observed continuations simultaneously.

---

### 6.6 Memory–Compute Tradeoff

The trie formulation introduces additional memory requirements:

- storage of prefix nodes
- storage of hidden states per prefix

Memory scales with $P$, the number of unique prefixes.

However:

- compute is reduced due to shared evaluation
- repeated forward and backward passes are eliminated
- reuse of prefix states increases efficiency

This represents a tradeoff between memory usage and computational cost.

---

### 6.7 Summary

The prefix-trie formulation transforms training from a path-based computation to a node-based computation.

This yields:

- elimination of redundant forward and backward passes
- factorization of shared Jacobians
- aggregation of gradient signal prior to transformation
- reduced computational complexity
- improved gradient efficiency and stability

In combination with the identity established in Section 5, these properties form the basis for accelerated and structurally simplified training of autoregressive language models.

---

## 7. Training Regimes

The prefix-trie formulation admits multiple valid training procedures depending on how the computation graph is partitioned.

### 7.1 Full-Trie Training

A full traversal of the trie computes the exact gradient of the objective:

- forward pass over all nodes  
- backward pass over entire trie  
- single update  

This corresponds to full-batch gradient descent.

---

### 7.2 Bounded Subtrie Training

In practice, updates can be performed over **bounded subtries**:

- select a subset of roots  
- include all descendants up to depth $d$  
- compute forward and backward over this subgraph  
- update once per subtrie  

This preserves gradient consistency within each subgraph while allowing multiple updates per epoch.

---

### 7.3 Truncated (Local-Depth) Training

To reduce computational cost, gradient propagation can be truncated:

- reuse cached prefix states  
- compute forward only for current depth  
- backpropagate only through current-depth operations  

This is analogous to truncated BPTT in recurrent models and provides a practical tradeoff between accuracy and efficiency.

---

## 8. Systems Considerations

The efficiency of the prefix-trie formulation depends heavily on implementation.

### 8.1 Attention Bottleneck

While projections and feedforward layers can be batched across nodes, attention depends on prefix history and is often implemented per node. This can dominate runtime if not optimized.

### 8.2 Prefix Structure

The trie naturally decomposes into:

- **branching regions**, where multiple continuations exist  
- **unary chains**, where no further sharing occurs  

This structure suggests hybrid execution strategies.

---

### 8.3 KV Reconstruction

Naïve implementations reconstruct key-value histories per node, leading to repeated work. Incremental propagation and prefix-based storage reduce this cost.

---

### 8.4 Batching Strategies

Efficient implementations batch operations across nodes sharing prefixes, while treating unary chains as contiguous sequences.

---

### 8.5 Implementation Sensitivity

Performance depends strongly on:

- attention scheduling  
- memory layout  
- device synchronization  

Thus, computational advantages of the formulation are contingent on system-level design.

---

## 9. Structural Extensions

The trie representation exposes structural information unavailable in window-based training.

### 9.1 Entropy-Based Weighting

Each node has an empirical entropy:

$$
H(p) = -\sum_x q(x|p)\log q(x|p)
$$

Loss can be weighted:

$$
L_p^{weighted} = (1 + \lambda H(p)) \cdot L_p
$$

This emphasizes ambiguous contexts and reduces overtraining on deterministic chains.

---

### 9.2 Branching-Aware Learning

Nodes with higher branching factors represent more diverse futures and may warrant stronger gradient signal.

---

These extensions are optional and do not alter the core formulation.

---

## 10. Implementation and Memory Scaling

The factorization of §5 and the bounded-subtrie regime of §7.2 determine
what the optimizer must see. This section documents what it took to
make that regime run on commodity hardware.

### 10.1 Radix compression

Build a character-level prefix trie of the corpus to depth d. Every
branching endpoint (≥ 2 distinct next-tokens observed) is a training
example; every unary chain (exactly one next-token observed) contributes
no training signal and can be collapsed into a single multi-character
edge. The resulting **radix trie** stores:

```
radix_id, parent_radix_id, first_char_depth,
edge_len L, edge_tokens[L],
edge_mass (prefix count at head of edge),
entries[] (next-token counts at endpoint)
```

Radix compression is lossless: the expanded character sequence is
identical to the original trie. At d=32 on Shakespeare, the radix trie
has 1.67M nodes representing a 27M-node uncompressed leveled trie — a
**16.2× reduction in node count**. However, radix compression preserves
`total_edge_chars` (the sum over edges of their lengths equals the
original trie's node count), which is what bounds the KV-cache budget
at training time.

### 10.2 Per-subtree KV-cache allocation

A transformer's K/V cache must cover every character position attended
to by any query. The global cache size for the whole trie is:

```
KV_bytes = total_edge_chars × d_model × 2 (K+V) × n_layers × bytes
```

At Shakespeare d=32: 27M × 64 × 2 × 2 × 4 B = 27.6 GB. This exceeds the
15 GB RAM + 15 GB swap available on the commodity workstation used for
these experiments.

The per-subtree insight: attention at a query position only reaches
that position's *ancestors*. Different root-child subtrees share no
character positions — the 'A'-tree and the 'B'-tree never look at each
other's KV slots. So the KV cache can be **scoped to one subtree at a
time**: allocate KV sized to the current subtree's `total_edge_chars`,
run forward-backward over that subtree, take one optimizer step, free
the KV, move on.

At Shakespeare d=32, peak per-subtree KV drops from 27.6 GB (global) to
**4.0 GB** (largest subtree, rc=' ') — a 6.9× reduction that brings
training within the hardware budget.

### 10.3 Bigram subtree partition

The unigram per-subtree scheme bottlenecks on the largest root-child.
In Shakespeare that is the space character, responsible for ~30% of
corpus occurrences. At d=48+ the space subtree alone would exceed
available RAM.

**Bigram partitioning** splits each root-child subtree by the second
character of the path. For vocabulary size V, there are up to V²
partitions. At Shakespeare d=32 this produces 1,465 bigram subtrees
(vs. 65 unigram), with peak per-subtree KV 677 MB (vs. 4 GB unigram
— a further 6× reduction).

Each bigram subtree is self-contained (it carries its own ancestor
chain for every endpoint in it), so no cross-subtree coordination is
required at training time — bigram partitioning is trivially
parallelizable across a cluster. The extension to trigram and beyond
is the same pattern applied recursively.

### 10.4 Auto-LR scaling across granularities

More, smaller subtrees means more optimizer steps per pass over the
data. Naive `lr` at bigram granularity overshoots, because the product
`lr × steps_per_super_epoch` is approximately invariant at a fixed
depth (a well-calibrated per-step update magnitude does not change when
you split the batch finer). We therefore scale:

```
lr_effective = base_lr × (REFERENCE_STEPS / steps_per_super_epoch)
```

with REFERENCE_STEPS = 65 (calibrated at unigram d=16). This lets a
single `base_lr = 3e-3` hold across unigram, bigram, and deeper
partitions without per-config retuning. Empirically, bigram d=32 with
auto-LR reaches PPL 13.30 in one super-epoch, essentially matching
unigram's 13.17 (§11.2).

### 10.5 Frequency-based pruning

At large d, the majority of paths near the depth cap have prefix mass 1
(appear exactly once in the corpus) and contribute negligible gradient
signal under count-weighted loss. A `(min_mass, min_depth)` threshold
drops these paths at trie-build time:

```
if edge_mass < min_mass and first_char_depth >= min_depth: drop
```

Because corpus-path mass is monotonically non-increasing with depth,
this operation only prunes tails — never mid-trie branches that lead to
high-mass paths deeper down. At Shakespeare d=32 with `min_mass=2,
min_depth=4`, the trie shrinks from 27M to 1.17M edge chars (23×
reduction), peak per-subtree KV from 4 GB to 175 MB. The quality cost
is 3 PPL on this corpus; at billion-token scale the long tail of
mass=1 paths is closer to noise, and the tradeoff should improve.

### 10.6 Data-structure note

The radix trie in this paper is a reference implementation. For
billion-token BPE corpora, a suffix array + LCP gives O(N) storage
(8 bytes/token, depth-independent) and enables streaming enumeration
of branching events without ever materializing the trie. The radix
implementation's virtue is that it fits comfortably alongside existing
transformer training code; the SA-based implementation is the natural
step for scaling the framework to larger corpora.

---

## 11. Empirical Evaluation

### 11.1 Setup

- **Corpus**: Shakespeare `input.txt`, 1.12 MB, 65-character vocabulary.
- **Model**: 2-layer transformer, d_model=64, 4 heads × 16 head-dim,
  d_ff=256, seq_len=128. 108,481 parameters.
- **Hardware**: single 8 GB consumer GPU + 15 GB system RAM + 15 GB
  swap.
- **Evaluation**: held-out perplexity measured on the corpus tail,
  32,768 positions scored with a sliding seq_len=128 window.

### 11.2 Comparison vs. standard window training

All numbers from the same codebase, same model architecture, same
corpus. For each window-training seq_len we swept step counts
{1k, 2k, 3k, 5k, 10k} and report the saturated best. Trie-based
results use the recipe of §10.

| Model | Context | Optimizer steps | Held-out PPL |
|---|---:|---:|---:|
| Random init | — | 0 | 144.2 |
| Window seq=16 (saturated) | 16 | 2,000 | 16.92 |
| **Trie d=16**, RMSProp+warmup-cosine | 16 | 50 | **15.28** |
| Window seq=32 (saturated) | 32 | 2,000 | 14.51 |
| **Trie d=32**, per-subtree, 3 super-epochs | 32 | 195 | **13.17** |
| **Trie d=32 bigram**, auto-LR, 1 super-epoch | 32 | 1,465 | **13.30** |
| Window seq=128 (saturated) | 128 | 10,000 | 7.00 |
| Window seq=512 (saturated) | 512 | 10,000 | 7.13 |
| Window seq=1024 (saturated) | 1024 | 10,000 | 6.30 |

**At matched context, trie-based training wins by ~10%** in PPL with
40× (at d=16) or 10× (at d=32) fewer optimizer steps.

**At longer context, window training wins overall** — the 108k-parameter
window model absorbs substantially more context (seq=1024 reaches PPL
6.30, well below trie d=32). This is the natural scaling boundary: the
trie formulation's per-step efficiency is per-depth, and extending
depth costs memory in a way that extending window context does not.
Scaling the trie formulation to d ≥ 128 is the natural next experiment;
the per-subtree + bigram + pruning infrastructure of §10 was built for
exactly this.

### 11.3 Super-epoch sensitivity (d=32 unigram)

| Super-epochs | Adam steps | Held-out PPL |
|---:|---:|---:|
| 1 | 65 | 13.71 |
| 2 | 130 | 13.18 |
| **3** | **195** | **13.17** |
| 5 | 325 | 14.84 |
| 10 | 650 | 18.27 |
| 50 | 3,250 | 37.34 |

The trie-based objective converges faster than window training and is
correspondingly easier to over-train — 50 super-epochs catastrophically
overfits on this small corpus. An external best-checkpoint wrapper
(`agpt_train_best.sh` in the repository) writes a checkpoint per
super-epoch and selects the best by held-out PPL, removing the need
to hand-tune the super-epoch count.

### 11.4 Radix compression + pruning numbers

| Config | radix nodes | total edge chars | Peak KV (d=32) | PPL |
|---|---:|---:|---:|---:|
| Uncompressed leveled trie | 27.0 M | — | n/a | n/a |
| Radix compression | 1.67 M | 27.0 M | 4.0 GB | 13.17 |
| Radix + prune `m<2, d≥4` | 0.55 M | 1.17 M | 175 MB | 16.07 |

---

## 12. Limitations

Stated without hedging:

1. **Scale**. All results are on a 1.1 MB char-level corpus with a
   108k-parameter model. Modern LLM training is ~9 orders of magnitude
   larger. This paper's claims are that trie-based training beats window
   training *on a small corpus at matched context*; whether the advantage
   holds at BPE vocabularies and billion-token corpora is an open
   empirical question.

2. **KV-cache scaling to frontier-scale models**. At Opus-family
   hyperparameters (d_model ≈ 16k, layers ≈ 100) a 1B-token BPE corpus
   would require petabytes of aggregate KV memory across subtree files.
   Per-subtree training makes this distributable; aggressive pruning
   and trigram/n-gram partitions are likely needed to keep per-node
   budgets feasible on current hardware.

3. **Pruning at scale**. §11.4's pruning result cost 3 PPL on
   Shakespeare. On a billion-token corpus the long tail of mass=1
   paths is closer to noise and pruning should approach free — but
   this is unvalidated at scale.

4. **Generation quality**. This paper reports PPL only; qualitative
   comparison of generated text is not systematically done.

5. **Compute-matched comparison**. Our baseline is step-saturated at
   each context length; trie-based training uses ~10× fewer optimizer
   steps but each step covers a whole subtree, so wall-clock cost per
   step is higher. On the same 108k-param model + 8 GB GPU, trie d=32
   at best-PPL = 21 minutes, window seq=32 at best-PPL ≈ 45 seconds.
   Trie-based training spent more wall-clock compute per PPL-unit
   here, which tilts the honest framing from "faster" toward
   "different operating point."

6. **Peer comparison**. No head-to-head with tokenizer-free neighbors
   (ByT5, MambaByte, etc.) has been run. These are the natural peer
   group and a proper comparison requires running them on the same
   corpus at the same model scale.

7. **Representation scalability**. The radix trie representation is
   a reference implementation; it does not itself scale to
   billion-token BPE. A suffix-array-based builder is the clear
   engineering path to compact storage + streaming enumeration at
   scale.

The natural next validation is **WikiText-103 scale with a real BPE
tokenizer on cluster-grade hardware** — approximately 2 orders of
magnitude larger than the present experiment, achievable on a single
A100 for < $100 of cloud compute. This would establish whether the
trie-based advantage holds at BPE vocabularies and 100M-token corpora,
which would be the meaningful first step toward frontier scale.

---

## 13. Summary of Contributions

1. **Gradient-factorization theorem** (§5): a rigorous demonstration
   that autoregressive next-token training admits reformulation over a
   prefix trie, with the trie gradient mathematically equivalent to
   the count-weighted sum of standard-training gradients.

2. **Bounded-subtrie training regime** (§7.2): the AGPT training unit
   — one forward-backward-step per root-child subtree — preserves
   gradient consistency by holding parameters fixed through the
   subtree, avoiding the K/V-staleness problem that finer-grained
   updates introduce.

3. **Radix-compressed, per-subtree-file trie format** (§10.1–10.2):
   collapses unary chains and scopes KV-cache allocation to one
   subtree at a time, making d=32 training fit in a 15 GB consumer
   RAM budget.

4. **Bigram subtree partition** (§10.3): splits root-child subtrees
   by second-character of path, attacking the dominant-root-child
   memory ceiling (the space character's subtree in English). 6×
   further peak-memory reduction at Shakespeare d=32 and a clear path
   to trigram/n-gram partitioning for deeper d.

5. **Auto-LR scaling across subtree granularities** (§10.4): the
   empirical invariant `lr × steps_per_super_epoch ≈ constant` lets
   a single base LR hold across unigram, bigram, and deeper
   partitions without per-configuration retuning.

6. **Frequency-pruned trie construction** (§10.5): a (min_mass,
   min_depth) threshold gates out the mass=1 long tail at large d.
   Provably safe (monotonicity); modest quality cost on small
   corpus; expected to approach free at scale.

7. **Empirical validation at char-level Shakespeare** (§11): matched-
   context PPL improvement of 9% at d=32 and 10% at d=16 over
   step-saturated window baselines, in 10× and 40× fewer optimizer
   steps respectively.

Full code is published at the referenced repository under an MIT
license.

---

## 14. Conclusion

We have shown that autoregressive language-model training admits a
reformulation over a prefix trie, in which gradient computation
factorizes through shared prefixes by commuting aggregation with the
chain rule. The resulting objective eliminates redundant prefix
computation and aggregates gradient contributions at shared nodes,
exposing a different computational structure for training.

The empirical validation on a 1.1 MB corpus demonstrates that the
factorization translates to a real per-step efficiency advantage at
matched context (9-10% lower PPL in 10-40× fewer optimizer steps).
Per-subtree training and bigram partitioning make the approach
practical on commodity hardware at depth 32; the same mechanisms
extend naturally to deeper d.

The open question — whether the per-step efficiency advantage
persists at BPE vocabularies and billion-token corpora — is the
natural follow-on experiment. The infrastructure in this paper was
built with that scaling path in mind.

---

*Corrections, discussion, and collaboration inquiries welcome.*
