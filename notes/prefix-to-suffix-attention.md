# Prefix-to-Suffix Attention at the Leaf Juncture

> **Status (2026-04-26):** Design proposal, not implemented. Captured to
> retain the formulation for future work. Connects directly to the §8
> dual-tree loop in `bayesian-bloom.md`.

## Motivation

In a standard transformer, attention is **token-to-token**: each query
position attends to all earlier token positions in the context window.
Cost: O(L²) compute, O(L) K/V cache per inference step.

In AGPT today, attention is **prefix-to-prefix** (or token-to-prefix-via-K/V
cache): a node at depth d attends to its ancestor chain of K/V values built
from depths 0..d-1. Cost: O(D²) compute per training step, O(D) K/V cache —
which is what makes AGPT memory-bounded by *D* rather than seq\_len.

In both cases the prediction target is **a single token** at the next
position. The prediction problem is character-by-character, even when
the model has rich context.

This note proposes a different attention pattern: **prefix-leaf attending
to candidate suffix-leaves at the dual juncture**. Instead of asking
*"what token comes next?"*, ask *"what whole D-character continuation
follows this prefix?"* The next token falls out as the first character of
the chosen suffix.

## The proposal

Given the §8 setup — prefix radix trie P and suffix radix trie S over
the same corpus, with leaf-to-leaf duals carrying actual corpus
continuations — define an attention layer where:

- **Query**: a vector representation of a P-leaf π (its accumulated
  prefix-context state, exactly as AGPT computes today)
- **Keys/Values**: vectors representing each S-leaf σ in π's
  **dual set** (the next-D-grams that actually followed π in the
  corpus)
- **Output**: a distribution over candidate suffixes; the highest-mass
  suffix's first character is the predicted next token

Because the dual set is constrained by §8 — a P-leaf with mass k has
*at most* k distinct next-D-grams — the candidate set is **bounded by
the leaf's mass**, not by vocabulary size or corpus size.

## Tractability

The total attention work across all P-leaves in a training pass:

```
total (prefix → suffix) attention pairs
  = Σ_leaves mass(leaf)
  = corpus_size
```

Each individual attention call has K candidates where K = mass(leaf).
For mass-1 leaves (~99% of leaves at d=32), K=1 — attention is O(1).
For high-mass leaves (e.g. "the " in English), K can be in the hundreds,
but those are the rare informative cases; per-pass total stays linear in
corpus size.

Compare to standard transformer attention over a length-L context:

| Pattern                    | Per-query cost | Per-pass cost           |
|----------------------------|---------------:|------------------------:|
| Token-to-token (L=128)     |          L=128 |    corpus × 128 = O(NL) |
| Prefix-to-prefix (D=32)    |           D=32 |    corpus × 32  = O(ND) |
| **Prefix-to-suffix dual**  |        **k≈1** |  **corpus × ~1 = O(N)** |

The prefix-to-suffix pattern is the cheapest at scale, *because the
corpus's own statistical structure constrains the candidate set*. The
model exploits the trie rather than fighting it.

## Architecture

### Where do suffix K, V come from?

Two design choices:

**Option A — learned per-leaf K, V.** Each S-leaf carries its own
learnable parameters. Memory: `n_leaves × 2 × d_model`.

For 5M corpus (~7.5M S-leaves), d_model=64: ~960 MB just for the
suffix K/V table. For 1B corpus, ~200 GB — untenable without sharding
or paging.

Pro: O(1) lookup at inference, no recomputation.
Con: Memory grows with corpus.

**Option B — computed on-the-fly from the suffix's edge tokens.** Each
S-leaf has only its edge content (already on disk). On demand, pass
the edge through a small encoder to produce K, V for that leaf.

Memory: O(d_model) per *active* candidate. Trivial.
Cost: extra forward per attention step.
Pro: Memory bounded regardless of corpus size.
Con: Recomputation cost per query.

**Option B is the right choice for scale.** The recomputation overhead
is small because the active candidate set per query is small (k ≈ 1).
At a 1B-token corpus, this means we never materialize K, V for the
6.5M-leaves-we-don't-need-this-step. The S-tree storage on disk is
already paid; this just adds a per-step encoder pass.

### K/V cache implications

This addresses the user's bet that K/V is the bottleneck for any
extension of AGPT. Under Option B, **there is no per-leaf K/V cache** —
K, V are recomputed per attention call. The "cache" becomes the
S-tree's static edge-content storage, which we already need for the
§8 dual-tree loop and the prefix-to-suffix attention is a layer *on
top* of it, not a parallel allocation.

### Encoder for the suffix edge

The simplest design: reuse the same model. Pass the suffix edge tokens
through the existing AGPT forward, take the final hidden state as both
K and V (or split via small projections). No new architecture; just
two passes — one for the prefix, one for each candidate suffix.

A more sophisticated design uses a separate (smaller) encoder for the
suffix side, biased to capture "starts-with" structure differently
from "ends-with" structure. Worth ablating if the simple version
underperforms.

## Connection to §8 dual-tree loop

The §8 dual-tree loop trains the model by walking prefix → suffix →
prefix → ... using Bayesian-inverted suffix probabilities. That's a
*data-side* mechanism — the corpus walk uses both trees, the model
itself is unchanged.

This proposal is the *model-side* counterpart: an attention layer
that learns to choose suffixes the way the §8 loop emits them. If
the §8 loop is implemented and trained on, **this attention layer is
what the model would learn to compute internally**. Making it
explicit should:

- speed convergence (the model isn't learning the dual mapping from
  scratch — it's directly asked to attend over the dual set)
- improve generalization (the inductive bias is correct: language
  has chunked structure, not just per-token statistics)
- reduce inference cost (K/V cache compresses from per-token to
  per-leaf-candidate)

Mathematically, training the §8 loop with this attention layer should
converge to the same optimum as the current AGPT objective, but with
much faster per-step iteration on coarser units.

## Open questions

1. **Encoder design.** Reuse the AGPT forward, separate small encoder,
   or no encoder at all (treat suffix as another prefix into the same
   trie)?

2. **Multi-suffix mass.** When a P-leaf has k > 1 duals, how is the
   loss computed? Mass-weighted softmax over candidates is the
   obvious choice — matches AGPT's count-weighted loss.

3. **Suffix-tree storage.** §8 already requires the S-tree. The
   prefix-to-suffix attention is on top of that — no extra storage
   beyond what the dual-tree loop already pays for.

4. **Scaling validation.** For 1B-token corpora, does the candidate
   set stay bounded? In principle yes (mass ≤ k corresponds to
   k duals), but for very high-frequency short prefixes the mass can
   grow with corpus size. Need empirical validation.

5. **Position encoding.** The query position is the prefix's endpoint
   depth in P. The candidate position is "depth+1 to depth+D" in the
   continuation. RoPE works naturally here — both are corpus-local
   positions.

6. **Backward compatibility.** Can this attention layer slot into the
   existing AGPT trainer alongside today's prefix-to-prefix attention,
   either as a replacement or as an additional head? The latter
   gives a clean ablation path.

## Why "this might be it"

The reason this proposal is worth returning to: **it makes the
corpus's own structure into the attention pattern**. Today's
attention either (a) ignores corpus structure and attends over
arbitrary fixed-length context windows, or (b) attends over the
prefix path only.

Prefix-to-suffix attention attends over **what the corpus says
follows this prefix**. The candidate set is not 65 tokens (vocab),
not 128 positions (context), not even D ancestor states (AGPT
prefix path) — it's *the actual k continuations the corpus
recorded*. That's the smallest, most informative set possible.

When the corpus is genuinely informative (k > 1, multiple
continuations to weigh), the model gets a meaningful attention
problem. When the corpus is deterministic (k = 1, mass-1 leaf),
attention is O(1) and the model just emits the unique continuation.

The architecture and the data align.

## Practical next step

Independent of implementation: the §8 dual-tree loop is a
prerequisite. Once §8 is built (corpus walks emit prefix/suffix
pairs at junctures), this attention layer can be added as one of
several training heads to compare against today's prefix-to-prefix
attention. If the prefix-to-suffix head produces equivalent or
better PPL with lower compute per step, it becomes the default.

Estimated work: §8 implementation (~1-2 days), then this attention
layer as a CUDA kernel + gradient (~3-5 days). Plus ablation runs.

Until §8 is implemented, this stays a design proposal.
