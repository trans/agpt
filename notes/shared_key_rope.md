# Shared-Key RoPE for Tree-Node Attention

## Core Idea

Do **not** collapse all positions of a node into a single spectral fingerprint.

Instead:

- Share the **content projection** across all occurrences
- Apply RoPE separately per occurrence position

This preserves standard RoPE semantics while still exploiting the fact that many positions share the same embedding.

---

# Standard Transformer

Normally, every token occurrence computes its own key:

\[
K_{i,p} = \operatorname{RoPE}(e_i W_K, p)
\]

Even if the same embedding appears thousands of times, the expensive projection:

\[
e_i W_K
\]

is redundantly recomputed for every occurrence.

---

# Shared-Key Formulation

For each unique tree node \(i\):

Compute the content key **once**:

\[
k_i = e_i W_K
\]

Then each occurrence only applies the cheap positional rotation:

\[
K_{i,p} = \operatorname{RoPE}(k_i, p)
\]

So:

- One learned key per node
- Many positional "views" over that key

---

# Interpretation

Factor attention into:

\[
\text{Key} = \text{content identity} \times \text{positional view}
\]

instead of treating every `(node, position)` pair as a completely separate key vector.

---

# Why This Is Better Than Spectral Fingerprints

The earlier Fourier/fingerprint idea collapsed all positions into one aggregate key:

\[
F_i[j] = \sum_{p \in P_i} e^{i\theta_j p}
\]

This changes attention semantics because attention softmax operates over separate occurrences.

By contrast, the shared-key approach preserves:

- exact RoPE behavior
- exact occurrence-level competition
- exact causal masking
- exact relative-position geometry

while still avoiding redundant projection work.

---

# Computational Structure

Instead of storing:

```text
(node_id, position) -> full key vector
```

store:

```text
node_id -> base key k_i
occurrence -> (node_id, position)
```

Then attention dynamically constructs the positioned key:

```python
k = rope(K_base[node_id], position)
score = q @ k
```

---

# Gradient Aggregation

All occurrences naturally accumulate gradients into the shared base key.

For node \(i\):

\[
\frac{\partial L}{\partial k_i}
=
\sum_{p \in P_i}
\operatorname{RoPE}^{-1}
\left(
\frac{\partial L}{\partial K_{i,p}}, p
\right)
\]

Then:

\[
\frac{\partial L}{\partial e_i}
=
\frac{\partial L}{\partial k_i} W_K^T
\]

So AGPT-style aggregation emerges naturally.

---

# Memory / Compute Advantages

## Standard Attention

Cost scales with:

\[
O(\text{occurrences})
\]

because every occurrence stores a full key.

## Shared-Key Attention

Projection cost scales with:

\[
O(\text{unique nodes})
\]

while positional rotation remains lightweight per occurrence.

For highly repetitive corpora this may significantly reduce:

- KV memory
- projection FLOPs
- optimizer state duplication
- gradient fragmentation

---

# Important Distinction

This does **not** remove occurrence-level attention.

There are still many positioned occurrences.

It only shares the expensive learned content representation across them.

---

# Potential Extensions

## Relative Position RoPE

Instead of absolute position:

\[
K_{i,p} = \operatorname{RoPE}(k_i, q-p)
\]

allowing direct relative-position construction.

## Cached Relative Rotations

Precompute:

\[
R(\Delta)
\]

for common relative offsets.

## Tree-Aware Relative Positions

Replace linear sequence position with:

- depth difference
- path divergence distance
- ancestor distance
- subtree locality metrics

while retaining the same rotational framework.

---

# Conceptual Summary

The correct optimization target is probably not:

```text
one key per node
```

but rather:

```text
one learned/content key per node
many cheap positional views over it
```

This preserves transformer attention semantics while leveraging AGPT's shared tree-node structure.

---

# Decoupled-Attention-Dimensions Framing

The shared-key + corpus-position tracking yields a clean factorization
that breaks the `d = seq_len` shackle. Three dimensions that are
currently locked together become independent:

| dimension | controls | current AGPT | proposed |
|---|---|---|---|
| `d` | trie depth → per-node identity quality | = seq_len | free |
| `seq_len` | max linear extent of usable history | = d | free |
| `A` | attention budget (#positions per query) | = seq_len | free |

In standard transformer there is no `d` and `A = seq_len`. In current
AGPT all three are pinned because each chunk's attention window *is*
the trie depth. With shared-key RoPE plus a `node_id → [corpus
positions]` map (or inverse `corpus_position → node_id`), they
decouple.

## Why Order Doesn't Matter

RoPE encodes position into the key vector itself: the score
`Q_q · RoPE(k_i, p)` depends only on `(q - p)`, not on where `K_p`
sits in the K matrix. Softmax is row-permutation-invariant. So the
attention input is **a bag of `(node_id, position)` pairs**, not an
ordered sequence.

Consequence: which `A` positions go into attention is a free choice.
Selection rule options:

1. **Most-recent A** — simplest; doesn't exploit `seq_len > d`
2. **Uniform-random A within seq_len** — cheap, gives long-range signal
   stochastically, works as a regularizer
3. **Retrieval-A** — pick positions whose `n_p` is salient for the
   query (e.g., longest matching suffix with `n_q`'s path)
4. **Mixed** — k_recent recent + (A - k_recent) sampled far

## Bookkeeping

What's needed beyond shared-key RoPE itself:

- `corpus_position → node_id` map: for every position p, the trie node
  whose d-suffix ends at p. Built once per corpus pass, size N entries
- `node_id → [corpus_position]` (inverse): mass per node already exists
  in the radix trie; explicit position lists cost O(N) total

Memory: ~N integers for the forward map. At Gutenberg 5M with
int32-ids that's 20 MB. Trivial.

## Training Loop Sketch

```
for each query position q in corpus:
    pick A positions {p_1, ..., p_A} from [q - seq_len, q] by chosen rule
    K = [RoPE(k_{n_{p_j}}, p_j) for j in 1..A]
    V = [v_{n_{p_j}}            for j in 1..A]
    out = attention(Q_q, K, V)        # normal multi-head softmax
    target = trie_distribution(n_q.children)   # standard AGPT target
    loss += KL(target, softmax(out · W_unembed))

backward:
    dL/dk_i accumulates across every query whose position-set
    included one of i's positions; RoPE^-1 each contribution then sum.
    (Exactly AGPT aggregation; nothing new.)
```

## Three Independent Choices

This means experiments can vary one knob while holding the other two:

- Hold `d=16, A=16`, vary `seq_len ∈ {16, 32, 64, 128, 256}` →
  measures pure long-range-context value at fixed identity quality
  and attention compute
- Hold `seq_len=128, A=16`, vary `d ∈ {8, 16, 32}` → measures
  identity-quality value at fixed attention shape
- Hold `d=16, seq_len=128`, vary `A ∈ {16, 32, 64}` → measures
  pure attention-budget effect at fixed everything else

Currently every AGPT experiment varies all three at once, conflated.

## The A=2 Extreme — Retrieval Attention

Once A is independently chosen, nothing forces it to be large. A could
be as small as 2 (or even 1) per layer, turning each attention step
into "self + one retrieved position":

```
For each query at position q (in each layer):
  K = [K_q (self),  K_p (one retrieved)]
  V = [V_q,         V_p]
  attention output = softmax-weighted mix of V_q and V_p
```

This makes attention **linear** in seq_len — O(seq_len × 2) per layer
instead of O(seq_len²). For seq_len = 10 000 with L = 12 layers:
- Full attention: 12 × 10 000² = 1.2 B ops
- A = 2 attention: 12 × 10 000 × 2 = 240 K ops  (~5000× cheaper)

### What you lose

Each layer can read only 2 positions per query. To route information
from an arbitrary past position to q, you need a chain of layers, each
hopping the information forward by one retrieval. The effective
receptive field of position q at layer L grows roughly as the union of
2^L hop-paths — depth substitutes for width.

### What you gain

Compute that scales linearly with seq_len opens the door to corpus-long
attention (seq_len in the tens of thousands or more) without
prohibitive cost. This is structurally what RWKV / SSMs achieve via
recurrence; this is the *attention* version of the same idea — with
the difference that the selection rule for the retrieved position has
direct trie-derived semantics.

### Why this fits shared-key RoPE

The retrieved position p has a node identity n_p in the trie. The
"useful retrieval" decision can use this identity directly:

- **Longest-matching-suffix retrieval**: pick p whose n_p shares the
  deepest backward-suffix with n_q's path. This is the canonical
  "find the most contextually similar prior point" rule.
- **Node-embedding similarity retrieval**: pick p whose base key
  k_{n_p} is closest (cosine, dot) to a learned query vector at q.
  Same shape as standard attention, but the search space is the
  trie's nodes rather than all corpus positions.
- **Random retrieval (training noise)**: pick uniformly at random for
  regularization.

The trie gives this retrieval a natural index structure that flat
attention lacks.

### Family across A

The choice of A puts us on a continuum:

| A | per-layer cost | what it looks like |
|---|---|---|
| 1 | O(seq_len) | pure recurrence-style |
| 2 | O(seq_len) | self + retrieval (this section) |
| log(seq_len) | O(seq_len · log seq_len) | hierarchical sparse |
| sqrt(seq_len) | O(seq_len^{1.5}) | BigBird-style |
| seq_len | O(seq_len²) | full attention |

The decoupling makes A a hyperparameter to tune, not a fixed
architectural decision. Different layers could even use different A —
shallow layers retrieve locally (A small), deep layers integrate
globally (A larger), or vice versa.

## Multi-Layer Caveat

Layer 1 input at every position p IS the node embedding at `n_p` —
shared-key RoPE applies cleanly. Layer 2+ inputs are the layer-1
attention outputs, which diverge per (occurrence, query history). At
deeper layers the "shared key per node" abstraction starts to leak —
two occurrences of the same node will have different layer-2 inputs
because layer 1 attended over different surrounding positions.

Options: (a) accept layer-1-only sharing and pay full per-position K
projection at deeper layers; (b) re-share at every layer using the
*current node identity* and forget per-occurrence divergence (lossy);
(c) hybrid — share at layers where divergence is small (early), full
per-position at deep layers.

(a) is the safe starting point and still saves substantial compute on
single-layer-heavy configs. The single-layer approximation is a
research question of its own.

## Status

Untested, but the formulation is clean enough to be worth a sweep.
Recommended next step: implement the `corpus_position → node_id` map
and a single-layer prototype with most-recent-A selection, just to
confirm gradient flow works as expected. Then move to the 3-knob
sweep above.
