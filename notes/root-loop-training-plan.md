# Root-Loop AGPT Training — Design Plan

Branch: `agpt-root-loop`

## Objective

Extend current AGPT training so the transformer is exposed to
longer context than the D-trie depth, while preserving AGPT's
aggregation property at the trie level.

## Where we are today

Current AGPT (`bin/agpt_train`, CUDA trainer):

- Trie built to depth D (say D=32).
- Training unit: per-root-child subtree.
- For each radix node in a subtree, the transformer's forward input
  is the node's ancestor chain (≤ D characters).
- Loss: weighted CE on the node's endpoint counts.
- One Adam step per subtree (AGPT invariant).
- Compute: O(unique radix nodes × D²) per pass.

The transformer is only ever trained at context length ≤ D. At
inference it can process seq_len tokens, but that's out of
distribution.

## What the virtual tree framing gives us

Conceptually extend the D-trie into a seq_len-deep virtual tree by
attaching a copy of the D-trie at every D-leaf (recursive, fractal).
The virtual tree's node distributions are Markov-of-order-D — each
D-segment's distribution depends only on the last D tokens.

At shallow depths (≤ D), the virtual tree = the real D-trie.
Aggregation is strong (many corpus occurrences per node).

At deeper virtual depths, most paths become mass-1 (unique corpus
sequences). Aggregation degrades smoothly to 1-per-node past
depth ~16 on Shakespeare.

The virtual tree is never materialized. It's just a way to think
about "what does training the transformer over longer windows with
a D-trie-based signal look like."

---

## Two concrete designs, differ on how aggregation is preserved

### Option A: Window-style training with D-trie teacher per position

**Training loop:**
- Training examples are corpus windows of length seq_len = K · D.
- Transformer forward: full seq_len window.
- At each position p in the window, look up the trie using the
  last D tokens (`[x_{p-D+1}, ..., x_p]`). Get `P_trie`.
- Loss at position p: cross-entropy of transformer's logits
  against `P_trie` (soft target) plus hard CE on true next token.
- Standard causal attention across the whole window.

**Aggregation:** None at trie-node level. The transformer's output
at each position depends on the full window, so different corpus
positions with the same last-D-tokens produce different gradients.
Per-window forward/backward — no AGPT compute saving.

**Pros:**
- Simplest to implement.
- Transformer learns at true seq_len context.
- Matches what we already have in `bin/microgpt` window training,
  just with richer loss.

**Cons:**
- Loses AGPT's compute saving entirely. Per-window compute is
  what standard window training already costs.
- Not really "AGPT" anymore; it's window training with trie teacher.

### Option B: AGPT-style with extended forward (canonical prior)

**Training loop:**
- Training unit: per-root-child subtree (current AGPT).
- For each radix node N at ancestor depth d_N:
  - Define a **canonical prior** of (K-1)·D tokens preceding the
    node's D-gram. Options for the canonical prior:
      (i)  First occurrence in corpus: `prior = corpus[p_first-(K-1)D : p_first]`
           for some designated first-occurrence position.
      (ii) Most frequent prior: the specific prior D-gram that
           precedes this radix node most often in the corpus.
      (iii) Null prior: zero-padding (equivalent to current AGPT
           but padded). This is the degenerate case.
  - Transformer forward input: `[canonical_prior, ancestor_chain(N)]`,
    total length (K-1)·D + d_N.
  - Standard causal attention across the full input.
- Loss: weighted CE on node's endpoint counts.
- Aggregation at trie level still holds because canonical prior
  is deterministic per node. One forward/backward per radix node
  represents all corpus occurrences of that specific D-gram
  rooted at that canonical prior.

**Aggregation:** Preserved at the trie level. Each radix node still
gets one forward/backward; one Adam step per subtree.

**Pros:**
- Keeps AGPT's compute saving structure.
- Transformer sees longer context (K·D total) during training.

**Cons:**
- The "canonical prior" is a design choice that doesn't correspond
  to how the corpus actually looks at inference. At inference, the
  prior is whatever the actual preceding corpus tokens are — not
  necessarily the canonical one.
- Training/inference mismatch: trained with canonical prior,
  inferring with actual prior. Model may underperform on
  non-canonical priors.
- How to pick the canonical prior is nontrivial. First-occurrence
  is cheap but arbitrary. Most-frequent is principled but expensive
  to compute.

### Option C: Hybrid (not the simplest first step)

Run current AGPT (Option shallow) at D=16 or D=32 for shallow
aggregation, AND use Option A separately for longer-context training
on a smaller sample. Combine in a two-phase training or as two loss
terms. More complex; park for later if A or B doesn't work.

---

## Recommendation for the simplest first experiment

**Option B with canonical prior = most-frequent** is the only one
that preserves AGPT's identity as a training method. Option A is
really "window training with trie teacher" which is a different
algorithm dressed up as AGPT.

But Option A is much easier to implement and test. It's a clean
data point: "does the trie signal help window training?" That
answer is independently interesting.

Proposed sequencing:

1. Implement Option A first. It's 1-2 days of work on the window
   trainer (`bin/microgpt`), gets us a clean measurement of
   "window + trie teacher vs. plain window."
2. If A shows a win over plain window training, move to Option B.
   B is a more substantial CUDA trainer modification (2-4 days) —
   the ancestor-chain data structure has to be extended to include
   the canonical prior.
3. Compare all three (plain window, A, B, current AGPT) on same
   corpus and model. That's the real answer to "what's the best
   way to give the transformer long context while keeping AGPT
   benefits."

## Open questions before implementation

1. For Option B, which canonical prior? First-occurrence is cheap
   (one pass over corpus). Most-frequent requires counting priors
   per radix node (extra trie-like structure). Start with first-
   occurrence for simplicity.
2. For Option A, what α (soft-target weight) to use? Sweep over a
   few values; 0.5 is a natural starting point.
3. Both options: do we retrain from scratch or fine-tune from
   current AGPT checkpoint? Fine-tuning is faster but mixes effects.
   Cleaner to retrain from init for a fair comparison.
4. Evaluation: held-out PPL at matched wall-clock compute (since
   Option A and current AGPT have very different per-step costs,
   step-matched comparisons are misleading).

## What I'm NOT proposing

- Building an actual seq_len-deep trie (we agreed that's wasteful —
  most paths are mass-1 past depth ~16).
- Changing the trie format.
- Changing the transformer architecture (yet — chunked attention,
  recurrent state, hierarchical setups are follow-ups).
- Touching inference (inference is standard transformer; no trie
  consultation at inference in either option).

---

## Next step

Pick A or B (or both sequentially) and start coding. I'd advocate
starting with A because it's quick to run — result in a day or two,
either positive or negative, and either way it informs whether B
is worth the larger engineering cost.
