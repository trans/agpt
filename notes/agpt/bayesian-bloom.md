# Bayesian Bloom: Unlocking Dense Suffix Signal from Sparse Prefix Tries for Unbounded Context

> **Status (2026-04-25):** Documented as a future direction; **not currently
> needed**. The wrap-around-via-synth-corpus approach (`bin/synth_wrap_corpus`,
> see `rnd/wrap-around/`) reached the SGD seq=128 ceiling (mean PPL 7.04) at
> d=32 without requiring Bayesian Bloom or any explicit suffix structure.
> Revisit when training-context demands push past what wrap-around can
> provide. The refinement in §8 is the current crystallized form of the
> idea; §1–§7 below predate that and frame the original formulation.

## Abstract

We present a novel extension to AGPT (Attention-Gated Prefix Trie) language models that addresses the fundamental context-length limitation imposed by trie sparsity. Rather than extending the trie depth, building a second suffix tree, or learning an alignment between spaces, we observe that a prefix trie implicitly encodes its own suffix view via Bayes' rule. At sparse tips — nodes where branching factor collapses to unity — we invert the path statistics analytically to recover a dense contextual distribution over preceding contexts. This enables inference and training over sequences arbitrarily longer than the trie's optimal depth *D*\*, with **constant memory beyond** *D*\* and **linear compute** in sequence length. The optimal depth *D*\* is shown to be a logarithmic function of corpus size, readable directly from the trie rather than tuned as a hyperparameter. The architecture makes the corpus itself — not arbitrary context window choices — the fundamental determinant of model behavior. No second tree is built, no additional parameters are learned, and no retraining is required: the dense signal is present in the original trie and is unlocked by a simple probabilistic inversion.

---

## 1. Introduction

Attention-based language models face a well-documented scaling wall: context memory grows linearly and attention compute grows quadratically with sequence length. State-of-the-art transformers at 128K–200K context windows require hundreds of gigabytes of GPU memory, making long-context inference the dominant hardware bottleneck in modern NLP infrastructure.

AGPT (prior work) replaces the flat attention mechanism with a learned transformation over a prefix trie, providing structural reuse across shared prefixes and bounding the K/V cache by tree depth rather than sequence length. However, AGPT inherits its own limitation: the trie has a corpus-dependent **optimal depth** *D*\*, beyond which nodes are nearly all unary and carry no predictive signal. Naïvely looping sequences back to root at *D*\* destroys positional continuity and confuses the model about sequence boundaries.

This paper presents a resolution to the sparsity problem that requires no additional architecture, no second tree, and no new parameters.

### Contributions

1. **The Bayesian Bloom**: A deterministic, analytically derived mechanism for converting sparse prefix tips into dense contextual distributions via Bayes' rule applied to the existing trie statistics.
2. **An alternating cascade model** in which sparse prefix paths bloom into dense suffix views, which bloom back into prefix views — covering sequences of arbitrary length with constant depth resources.
3. **A characterization of** *D*\* **as a logarithmic function of corpus size**, recoverable directly from the trie via sparsity analysis.
4. **Scaling results** showing that AGPT + Bayesian Bloom has memory `O(D*)` and compute `O(seq_len)`, compared to transformer `O(seq_len)` and `O(seq_len²)` respectively — a qualitative break from the standard scaling regime.

---

## 2. Background: The AGPT Prefix Trie

In AGPT, a training corpus is decomposed into a prefix trie of maximum depth *D*. Each node represents a token sequence `(t₁, …, tₖ)` for `k ≤ D`, and stores statistical information (path frequencies, child distributions) as well as a learned representation computed via attention over the root-to-node path. Global weights — token embeddings, Q/K/V projections, and output projection — are shared across all nodes; the trie itself is a computational structure, not a parameter store. RoPE provides positional encoding, with position defined by node depth.

Training proceeds depth-by-depth in a single batched matmul per depth, exploiting the independence of nodes at the same level given the previous level's computed representations. This yields substantial structural reuse: any two sequences sharing a prefix share their K/V cache up to that point.

### 2.1 The Sparsity Problem

Empirically, the branching factor of the trie collapses rapidly with depth. For a corpus with vocabulary size `V = 65` and typical natural-language sequence statistics, by depth `d ≈ 32` more than 99.99% of nodes are unary — a single observed continuation — meaning the trie has nothing left to say about sequential structure at that depth. Yet training sequences (`seq_len`) are routinely longer than this limit.

The existing leaf-to-root stitching approach loops back to root at maximum depth, creating an arbitrary reset point that destroys positional continuity. The model cannot distinguish "this is a genuine start" from "we just hit the depth cap." Prior attempts at resolving this — end-cap attention, depth-curriculum training, blended prefix suffixes — either added untenable computational cost, worsened convergence, or failed to yield measurable gains due to incorrect blending of incommensurable distributions.

---

## 3. The Key Insight: The Trie Already Contains the Suffix View

A prefix trie built over a corpus does not merely record prefix statistics. By Bayes' rule, it implicitly encodes **suffix statistics over the same corpus**. No second data structure is required.

### 3.1 Bayesian Inversion

At any node `n` representing prefix sequence `π = (t₁, …, tₖ)`, the following holds:

```
P(prev | π) ∝ P(π | prev) · P(prev)
```

Both factors are directly available in the trie:

- `P(prev)` is the unigram frequency over tokens — available at depth 1.
- `P(π | prev)` is the path frequency from `prev` through `π` — available as the normalized count of the path `prev → t₁ → t₂ → … → tₖ` relative to the count of `prev`.

The posterior `P(prev | π)` is a proper probability distribution over preceding tokens (or token sequences, if extended recursively) given the path `π`. It is computed in `O(V)` time per inversion, with no learned parameters.

### 3.2 Interpretation: Contextualization, Not Prediction

Crucially, the Bayesian inversion does not ask the model to predict backward. It answers a different question:

> *"Given that the sequence π appears in this corpus, what are the typical contexts in which it occurs?"*

This is a **usage profile** of the sequence — a statistical characterization of its role in the corpus. When the prefix path becomes sparse (i.e., the corpus has exhausted its forward-looking signal for this path), the Bayesian inversion provides genuine additional information: the contextual neighborhood within which `π` occurs.

This is analogous to bidirectional language models (e.g., BERT), which use both left and right context to enrich token representations. Here, however, the right-context signal is derived analytically from existing statistics rather than learned through a masked training objective.

---

## 4. The Bloom Handoff

We define the **Bayesian bloom** as the operational mechanism by which the suffix view is invoked at sparse prefix tips.

### 4.1 Sparsity Detection

A node is considered sparse when its branching ratio — the number of distinct observed continuations divided by the vocabulary size — falls below a threshold (typically ≤ 1/V, i.e., unary):

```
is_sparse(n) = (|children(n)| ≤ 1)
```

Alternatively, a softer threshold based on entropy of the continuation distribution may be used.

### 4.2 The Cascade

When a forward pass reaches a sparse tip at depth *D*\*, the sequence does not loop back to root. Instead:

1. The Bayesian inversion is applied at the tip, yielding a distribution `P(prev | π)` over preceding contexts.
2. This distribution is used as enriched input for the subsequent forward computation, either by:
   - Injecting the expected context as a token-distribution input to the next prediction head, or
   - Sampling a representative preceding context and re-entering the trie at the root with that context prepended.
3. If the resulting path also encounters sparsity, the inversion is applied again, now generating dense forward context from the sparse suffix tip.

This produces an **alternating cascade** of prefix and suffix views:

```
prefix_root → a → b → c → [sparse @ D*]
                            ↓ Bayesian inversion
                       dense suffix context
                            ↓
                     [sparse suffix tip] → Bayesian inversion → prefix context
                            ↓
                           ...
```

At each handoff, the system moves from sparse back to dense. This is possible because the corpus, when read forward and backward, has complementary sparsity structure: long specific prefixes and long specific suffixes are different sets of sequences, each dense where the other is sparse.

### 4.3 Gradient Handling

The Bayesian inversion is pure arithmetic over observed statistics and requires no gradient flow. We treat it as a **stop-gradient boundary**: backward propagation through the prefix trie proceeds normally up to the sparse tip, at which point the gradient is not propagated through the inversion step. Training therefore remains structurally identical to the original AGPT procedure, with the bloom operating only at inference-time and during forward passes at training-time.

---

## 5. Optimal Depth *D*\* and the Corpus as Input

### 5.1 *D*\* Is Not a Hyperparameter

A central observation is that *D*\* is not a tunable hyperparameter but a **measurable property of the corpus**. Branching factor collapses according to the expected relation:

```
branching_factor(d) ≈ V · (corpus_size / V^d)
```

Setting this to unity yields:

```
D* ≈ log(corpus_size) / log(V)
```

Thus *D*\* grows **logarithmically** with corpus size. Doubling the corpus yields a sub-unit increase in *D*\*; multiplying the corpus by V adds a single level of meaningful depth.

### 5.2 The Corpus Is the Real Input

This logarithmic relation is not an architectural artifact but a reflection of a deeper fact: the information content of sequential structure in a corpus is fundamentally bounded by the corpus itself. Beyond *D*\*, the trie has nothing genuine to say because the data does not support longer-range structural claims.

Under this view, **the corpus is the true input to the model**. The trie is the lossless sufficient statistic for all sequential structure up to *D*\*, and the Bayesian bloom is the honest acknowledgment of the corpus's boundary.

Transformers, with their fixed-context-window architecture, can be seen as over-provisioning along one axis (context length) while ignoring the corpus's own structural limits. AGPT with Bayesian Bloom provisions according to the corpus.

---

## 6. Scaling Properties

The cost structure of AGPT + Bayesian Bloom contrasts sharply with that of standard attention-based models.

| Property | Standard Transformer | AGPT + Bayesian Bloom |
|---|---|---|
| Memory (seq_len) | `O(seq_len)` — K/V cache grows linearly | `O(D*)` — constant beyond *D*\* |
| Attention compute | `O(seq_len²)` | `O(D*²)` per window + `O(seq_len)` blooms |
| Parameters | fixed | unchanged from base AGPT |
| K/V cache | unbounded in seq_len | bounded by *D*\* |
| Context limit | hardware-constrained | corpus-constrained |

For sequences longer than *D*\*, each additional token incurs only one Bayesian inversion — `O(V)` arithmetic over cached statistics. Memory does not grow. The K/V cache is permanently bounded.

**Practical implication**: Frontier transformer models at 128K context require hundreds of gigabytes of GPU memory and datacenter-scale infrastructure. AGPT + Bayesian Bloom's memory at 128K, 1M, or 1B context is identical — bounded by *D*\* ≈ 32, translating to K/V storage in the megabyte range for typical *d_model*. This is not a marginal improvement but a qualitative break from the transformer scaling regime.

---

## 7. Implementation

### 7.1 Position Encoding

**Critical:** RoPE positions must be defined **window-relative**, not depth-relative.

In the original AGPT, `position = depth`. This breaks immediately for sequences longer than *D*\*: position 3 in one window context is not semantically equivalent to position 3 in another. We redefine positions as relative to the rolling window:

```
position 0 = oldest token in current window
position D*-1 = newest token in current window
```

This must be consistent across the prefix path, the Bayesian inversion, and any subsequent prefix re-entry.

### 7.2 Bayesian Inversion Routine

```python
def bayesian_suffix_context(node, trie):
    # P(prev) = unigram frequencies
    p_prev = trie.unigram_frequencies  # [V]

    # P(π | prev) = path frequency from prev through node
    p_path_given_prev = trie.path_frequencies(node)  # [V]

    # Bayes inversion, then normalize
    posterior = p_path_given_prev * p_prev
    posterior /= posterior.sum()
    return posterior  # [V]
```

### 7.3 Forward Pass with Bloom

```python
def forward(sequence, trie, D_star):
    state = trie.root
    for i, token in enumerate(sequence):
        if depth(state) < D_star and not is_sparse(state):
            state = state.children[token]
        else:
            context = bayesian_suffix_context(state, trie)
            # Use `context` to enrich the prediction head and re-enter the trie
            state = trie.root_with_context(context)
    return state
```

The root re-entry with context is analogous to a continuation: the model begins a new window, but with the Bayesian posterior as prior belief about what preceded.

### 7.4 Training

Training is unchanged from base AGPT. Depth-parallel matmuls proceed as before. The bloom operates only when sequences exceed *D*\*, and its inversion step is a stop-gradient. No changes are required to the optimizer, loss, or weight-update cadence.

---

## 8. What This Is Not

To preclude misinterpretation, we list what the Bayesian Bloom is **not**:

- **Not a second suffix trie.** Only one trie is built.
- **Not a learned alignment.** No Procrustes rotation, no learned bridge network, no additional parameters.
- **Not a blended distribution.** Earlier attempts to blend multiple prefix distributions at a single point failed because the distributions were incommensurable. Bayesian inversion is a principled probabilistic operation, not an ad hoc blend.
- **Not a loop-to-root hack.** The bloom provides genuine dense signal; it does not reset positional context arbitrarily.
- **Not a change in training objective.** Same loss, same backprop, same optimizer.

---

## 9. Discussion

### 9.1 Why the Earlier Butterfly Intuition Was Incomplete

An initial conceptualization posited that prefix and suffix trees should be joined "front-to-front" at the vocabulary layer — a butterfly structure with dense centers and sparse wings. This is structurally elegant but yields no new information: walking up a prefix tree and back down a mirror prefix tree recovers only what is already in the prefix tree.

The correct structure is the **alternating cascade**: sparse prefix tips bloom into **new dense suffix subtrees**, whose sparse tips bloom into **new dense prefix subtrees**, and so on. Each handoff introduces genuinely new statistical content because it reorients the directional reading of the corpus.

### 9.2 Why One Trie Suffices

Because the corpus is a single object, read in either direction, both prefix and suffix statistics are derivable from the same counts. Bayes' rule is the exact transformation between the two views. Building a second trie would duplicate information already present in the first.

### 9.3 Relationship to Bidirectional Models

Bidirectional transformers (BERT and successors) use masked modeling to let a token's representation incorporate information from both sides of its position. The Bayesian Bloom achieves a similar goal without an auxiliary objective: the backward signal is extracted analytically from the corpus statistics already embedded in the trie.

### 9.4 Failure Modes and Empirical Caveats

The principal empirical question is whether the Bayesian enrichment at bloom boundaries preserves sufficient long-range coherence. In domains where genuine long-range dependencies exist in the data but fall outside *D*\*'s statistical support, the bloom can only provide the corpus's best marginal estimate — it cannot recover structure the corpus does not contain.

This is, however, a principled limitation: if the corpus does not support a dependency, no architecture can legitimately learn it.

---

## 10. Conclusion

The AGPT prefix trie, when viewed through Bayes' rule, already contains the dense suffix statistics needed to extend inference and training beyond its optimal depth. The sparsity problem dissolves once the right lens is applied: sparse prefix tips are not dead ends but doorways into dense, statistically grounded continuation via the corpus's own backward reading.

This yields a model with:

- **Constant memory** beyond *D*\*
- **Linear compute** in sequence length
- **No additional parameters**
- **No retraining**
- A **corpus-determined context limit** replacing arbitrary hardware-bound context windows

The architecture reframes what it means to "extend context." Instead of scaling hardware to accommodate longer windows, we scale the corpus: more data yields a logarithmically deeper trie, and the bloom does the rest.

In the end, the corpus is the real input. The architecture is the lens.

---

## 8. Refinement: Explicit Dual-Tree Loop

The original formulation (§1–§7) claimed *no second tree is required* —
the prefix trie alone, via Bayesian inversion at sparse tips, supplies
the suffix view. That framing was a useful simplification but slightly
misrepresented the cleaner construction now preferred. The refinement
makes the suffix tree explicit and uses Bayesian inversion as a
direction-flipping bridge, not as a sparsity workaround.

### 8.1 Two trees, one corpus

Build both:

- a **prefix radix trie** over the corpus (the existing AGPT structure)
- a **suffix radix trie** over the corpus — the same trie shape but
  with paths root→leaf representing suffixes of the corpus

Each radix node in either tree corresponds to a substring (a prefix of
length `k` in the prefix trie, or a suffix of length `k` in the suffix
trie). For any sequence π that appears as a substring of the corpus,
there is a node in the prefix trie representing it as a forward path
and (potentially) a corresponding node in the suffix trie representing
it as a reverse path. We refer informally to this pair as **dual nodes**.

> *Caveat: it is not yet established that every prefix-trie node has a
> meaningful dual in the suffix trie — the corpus's prefix structure and
> suffix structure are not symmetric in general. The dual mapping is
> well-defined for nodes whose substring is reachable from both sides;
> a precise statement of conditions is left for follow-up work.*

### 8.2 Connecting the trees

Two granularities of dual-tree connection:

- **Coarse (leaf-to-leaf):** connect prefix-trie leaves directly to
  suffix-trie leaves. Analogous to the wrap-around's leaf-to-root
  stitch, but instead of looping back to root, the walk continues
  *through* the suffix tree from a corresponding leaf.

- **Fine (node dual):** at every prefix-trie node, the dual node in
  the suffix trie is the natural continuation point. This makes the
  combined structure feel less like a stitched loop and more like a
  single object viewed from two directions.

The fine version is conceptually cleaner but requires the dual-mapping
caveat above to be resolved.

### 8.3 The training loop

A training pass now reads:

1. Walk forward through the prefix trie (root → leaf), accumulating
   forward-conditional state as in standard AGPT.
2. Cross over to the suffix trie at the appropriate dual point (leaf
   in the coarse version; arbitrary node in the fine version).
3. Continue walking *down* through the suffix trie. Suffix-trie
   probabilities are oriented "upside down" relative to forward
   training (the trie was built backwards), so apply Bayesian
   inversion to flip each step's conditional from `P(prev | suffix)`
   to `P(next | prefix)`. With the inversion, the suffix-trie walk
   yields forward-direction predictions exactly as the prefix-trie
   walk does.
4. The suffix-trie walk descends to the suffix-trie root.
5. **Loop closure:** suffix-trie root feeds back into prefix-trie root.
   The next training step starts a fresh prefix walk from there.

### 8.4 Relationship to wrap-around-via-synth-corpus

The wrap-around-via-synth-corpus approach (`rnd/wrap-around/`) achieves
a similar end without ever building a suffix trie or doing Bayesian
inversion: it walks the prefix trie root → leaf, picks a bridge token
from the leaf's endpoint counts, wraps to root seeded by that bridge,
and continues. The synthesized corpus is then trained on with standard
SGD, no architectural changes required.

It is plausible — though not proven — that wrap-around and the
explicit dual-tree loop are **mathematically equivalent**: both
implement "walk forward through context, transition to a new
context, continue forward." The dual-tree loop expresses the
transition with a Bayesian-inverted suffix-tree path; wrap-around
expresses it with a single endpoint-count bridge sample.

If the equivalence holds, wrap-around is the cheaper implementation
(no second trie, no inversion machinery, no architectural change).
The dual-tree loop has aesthetic appeal — bidirectional, dual-node,
elegant closure — but the engineering case for it depends on
identifying a concrete training scenario where wrap-around's coarser
single-bridge transition is insufficient.

### 8.5 Why this is "the big guns"

The dual-tree loop is what would be brought out if and when the
synth-corpus approach hits a ceiling we cannot move with seed
diversity, longer training, or larger models. At that point, the
question becomes whether the missing capacity is in:

- **the corpus** — no architectural fix can help
- **the model** — bigger model, more steps
- **the wrap mechanism** — *here* the dual-tree loop wins, because
  its per-token transition uses the suffix tree's full local
  distribution rather than a single bridge token sample

This third case is the one to watch for. Until it arrives, the
dual-tree loop stays in this document and out of the codebase.

---

## Acknowledgments

*This work extends the AGPT architecture developed in [prior work, reference to be added]. The Bayesian Bloom formulation emerged in collaborative discussion, benefiting from iterative refinement of an initially symmetric "butterfly" intuition into the asymmetric alternating-cascade formulation presented here. The §8 dual-tree refinement crystallized after the wrap-around-via-synth-corpus approach matched the SGD seq=128 ceiling, prompting a re-examination of what the suffix view actually wanted to be in this architecture.*

## References

*To be populated during review. Key references will include the base AGPT paper, bidirectional transformer literature (BERT family), n-gram language modeling with Bayesian smoothing, and the Semtrace work on embedding geometry.*
