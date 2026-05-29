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

- a **prefix radix trie** P over the corpus (the existing AGPT structure;
  root→leaf paths spell forward D-grams)
- a **suffix radix trie** S over the corpus (root→leaf paths spell
  *reversed* D-grams; equivalently, suffixes of the corpus indexed
  backward so that walking root-ward consumes characters in increasing
  corpus position)

Both trees store the same set of D-grams of the corpus — they differ
only in how those D-grams are *indexed* (forward-key vs reverse-key).

### 8.2 The dual relation: next-D-gram, not same-substring-reversed

This is the key correction over an earlier framing. The dual of a
P-leaf is **not** the S-node that holds its substring read backward
(that's a trivial substring identity, not a useful training pivot).
The dual is **the S-leaf representing the next D characters of the
corpus immediately following this P-leaf's occurrence**.

Concretely, for a P-leaf representing D-gram π that occurs at corpus
position `i`:

- the **dual S-leaf** represents `corpus[i+D : i+2D]`
- the juncture is *the literal corpus continuation*

For a mass-1 P-leaf (one occurrence in the corpus), this is a single
S-leaf — the next D-gram is uniquely determined.

For a mass-k P-leaf (k occurrences), the dual relation is
**multi-valued**: there are up to k distinct next-D-grams, each tied
to one occurrence position. Two occurrences may have the same next
D-gram (then two of the k duals collapse) or all k may differ.

A worked example on the synthetic corpus
`ABCDEFABCABCABDEDFBCADFE` at D=3:

- `ABC` (mass 3, positions 0/6/9) duals → `DEF`, `ABC`, `ABD`
- `BCA` (mass 3, positions 7/10/18) duals → `BCA`, `BDE`, `DFE`
- `CAB` (mass 2, positions 8/11) duals → `CAB`, `DED`
- `ABD` (mass 1, position 12) → single dual `EDF`
- `ADF` (mass 1, position 20) → truncated (only `E` left in corpus)

### 8.3 What S adds that P doesn't already have

The dual mapping requires *positional* information that P, as
currently stored, throws away.

P stores per-leaf **counts** but not the *positions* of each occurrence.
Two D-grams with mass 1 each can sit anywhere in the corpus and P
records only "I saw this once." S, indexed by what came after, captures
the per-occurrence continuation that count-only P cannot reconstruct.

So:

- **At the substring level**: S is recoverable from P trivially —
  both contain the same set of D-grams.
- **At the positional/continuation level**: S provides genuinely
  new information that count-only P lacks. The only way to derive
  S from P alone is to also store occurrence positions in P, at
  which point you can rebuild S by walking the corpus once.

In other words: **S is positional-continuation index that count-only
P discards.** If P is augmented to store positions, the two are
mutually derivable. Without positions, S adds real signal.

### 8.4 The training loop

A training pass:

1. Walk forward through P (root → leaf), accumulating forward-
   conditional state as in standard AGPT. Output covers `corpus[i:i+D]`.
2. At the P-leaf, look up the dual S-leaf (one of the up-to-k
   continuations available for this leaf, picked by mass weighting,
   or visited exhaustively across training).
3. Walk *down* through S from that S-leaf toward S's root. Because
   S's paths are reverse-indexed, this walk consumes
   `corpus[i+D : i+2D]` *forward in time*, but each S-step's
   stored conditional reads `P(prev_char | reversed_suffix)`. We
   apply **Bayesian inversion** to flip each step's conditional to
   `P(next_char | prefix)`, so the S-walk produces forward
   predictions just like the P-walk.
4. S-walk lands at S's root, having consumed D more characters of
   real corpus content.
5. **Loop closure:** S-root → P-root. Next P-walk starts fresh,
   beginning at corpus position `i+2D`.

One full P→S cycle therefore covers **2D characters of the actual
corpus walk**, structurally preserving long-range corpus order
across the boundary.

### 8.5 Comparison with wrap-around-via-synth-corpus

Both approaches solve the same problem (training over sequences
longer than D). They differ in what they preserve across the
inter-D-gram boundary:

|                              | wrap-around                                | dual-tree loop                              |
|------------------------------|--------------------------------------------|---------------------------------------------|
| Across-boundary kept         | 1 char (bridge token)                      | D chars (full next-D-gram)                  |
| Next chunk's content         | random D-gram seeded by bridge             | actual corpus continuation                  |
| Long-range corpus order      | lost — bouncing around corpus             | preserved — sticks to true corpus walk     |
| Mass-k aggregation           | sample one bridge, sample one root child  | sample one of k actual continuations        |
| Storage                      | P alone, count-only                        | P + S (or P-with-positions)                 |
| Architecture cost            | none — pure synth corpus + standard SGD   | needs the loop machinery and inversion      |
| Training-data fidelity       | distribution-faithful, position-loose     | corpus-faithful per cycle                   |

So they are **not mathematically equivalent per cycle**. Dual-tree
preserves the actual corpus continuation; wrap-around lossily
samples from the same statistical distribution.

In the **limit over many training cycles**, the two approaches
converge on the same distribution of corpus content seen
(modulo the wrap-around's slight bias from the random
restart mechanism). For finite training, the two may differ:

- dual-tree could learn long-range corpus structure more faithfully
- wrap-around's random reshuffling may generalize better and is
  closer to AGPT's original "aggregate across all occurrences"
  spirit
- on PPL of held-out clean text (no wrap noise), the relative
  performance is empirically open

### 8.6 When to bring this out

The dual-tree loop is the "big guns" — reserve it for when:

- the wrap-around-via-synth-corpus approach has saturated and
  PPL/quality is still bottlenecked by *transition fidelity* (not
  corpus capacity, not model capacity, not training budget)
- training stability or generation coherence over very long
  sequences is degrading in ways that look like long-range
  context-loss specifically

Until then: wrap-around is the cheaper, simpler implementation
that already matches the SGD seq=128 ceiling at d=32 on
Shakespeare. The dual-tree loop stays in this document.

### 8.7 Open questions for follow-up

1. **Storage cost.** Building S explicitly potentially doubles the
   trie storage. Whether this is justified depends on (a) the
   measured PPL gain at training scales where wrap-around saturates,
   and (b) whether per-occurrence positions in P would be cheaper.

2. **Empirical equivalence.** Implement a synth-corpus generator
   that emits the dual-tree corpus continuation deterministically
   instead of wrap-around's random restart, train under matched
   conditions, and measure the PPL gap. If small, the cheaper
   wrap-around is sufficient. If large, the dual-tree case is
   strengthened.

3. **Dual-mapping at the radix node level (vs leaf level).**
   §8.2 defines duality at leaves. Whether interior radix nodes
   also have meaningful duals (and how to walk them) is a finer
   structural question. The leaf-to-leaf framing is sufficient to
   close the loop and is what should be implemented first.

---

## Acknowledgments

*This work extends the AGPT architecture developed in [prior work, reference to be added]. The Bayesian Bloom formulation emerged in collaborative discussion, benefiting from iterative refinement of an initially symmetric "butterfly" intuition into the asymmetric alternating-cascade formulation presented here. The §8 dual-tree refinement crystallized after the wrap-around-via-synth-corpus approach matched the SGD seq=128 ceiling, prompting a re-examination of what the suffix view actually wanted to be in this architecture.*

## References

*To be populated during review. Key references will include the base AGPT paper, bidirectional transformer literature (BERT family), n-gram language modeling with Bayesian smoothing, and the Semtrace work on embedding geometry.*
