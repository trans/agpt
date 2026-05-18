# Subset-Attention AGPT: Sparse Position Training Within AGPT

## The Problem

AGPT's d=seq_len shackle: position embeddings (or RoPE) are only
trained for offsets 1..d because each firing's K/V cache has at most
d slots (the trie ancestor chain). Past d, attention reads garbage
because positions there were never trained.

We want to extend the model's effective attention range without
abandoning AGPT-the-trainer (which gives us the per-trie-node loss
and structured-batch optimization that delivered our headline 60%
PPL win on Gutenberg 5M).

## The Idea

User's question: *"Does the attention system need to see all the
positions at once to learn? Can it not see them in selected portions
and still build the entire model over time?"*

Answer: yes, sparse/subset position training works. Sparse-attention
papers (BigBird, Longformer, Reformer) demonstrate that a transformer
trained with restricted attention patterns generalizes to broader
attention at inference. The principle: a transformer's job is to
learn pairwise relationships at every relative offset. It doesn't need
all offsets visible simultaneously per forward pass — just enough
coverage of each offset across the training distribution.

For AGPT this maps cleanly to:

- Each AGPT firing currently has K/V budget of ~d slots (the trie
  ancestor chain).
- Replace those d slots with **K positions sampled from a
  seq_len-wide window of the corpus around the firing's
  position**.
- Position encoding via RoPE on corpus positions, so each K/V slot
  has a known relative offset to Q.
- The model trains over millions of firings, each with a different
  random subset of the seq_len window. Cumulatively this covers all
  relative offsets in [1, seq_len].

At inference: K/V cache can hold the full preceding context. The
model attends over all of it. Training has implicitly taught the
model "for any subset of preceding positions in seq_len, attend
appropriately" — which generalizes to "for all of them."

## What Stays the Same

- **AGPT loss**: per-trie-node KL on count distribution. Unchanged.
- **Structured-batch optimization**: pd=6 partition groups, shuffle,
  warmup-cosine LR. All preserved.
- **Trie itself**: same offline structural extractor at depth d.
- **Per-firing compute**: K/V budget is K (~d), so attention cost
  per firing is unchanged regardless of seq_len.

## What Changes

- **Each firing samples K positions from a seq_len window** instead
  of using the d-ancestor chain.
- **Position encoding switches from depth-relative to corpus-relative**
  (use RoPE on actual corpus positions, or relative offsets within
  the window).
- **Trainer needs corpus-occurrence info per trie node**: for each
  firing, pick a corpus position where this trie node's d-context
  appears, then sample preceding positions within seq_len.
- **The model's seq_len becomes a hyperparameter independent of d**.

## Sampling Strategies

For each firing at corpus position p with seq_len-wide window
[p-seq_len+1, p-1], pick K positions:

1. **Uniform random**: simple but rare offsets get under-sampled.
2. **Distance-stratified**: K/2 from recent (p-d..p), K/2 from far
   (p-seq_len..p-d). Forces long-range coverage every firing.
3. **Distance-decaying**: sample density ~ 1/distance, so close
   positions appear often, far positions occasionally. Biases toward
   local, gives uniform coverage of *log-distance* offsets (which is
   probably the right inductive bias for natural language —
   long-range matters less but in well-understood patterns).
4. **Mass-weighted**: positions sampled proportional to their
   trie-node mass. Common contexts get more training signal.
5. **Combined / stratified mass**: hybrid of distance and mass.

Distance-decaying (option 3) is probably the right default — it
matches how language naturally has stronger short-range correlations
and weaker long-range ones.

## Counting things

Gutenberg 5M, d=16, target seq_len=128:

- Per firing K/V slots: d=16 (unchanged from current AGPT)
- Per firing positions sampled from: seq_len=128 window
- Position pairs covered per firing: 16 × 16 = 256 (Q × K)
- Total firings per SE: ~5M (one per trie node firing event)
- Position pairs covered per SE: ~5M × 256 = 1.28B coverage events
- Distinct (Q-position, K-position) pairs in seq_len: 128 × 128 = 16k

Each position-pair covered ~80,000× per SE on average — vastly
more than necessary for the model to learn that relationship.

For seq_len=1024:
- 1024 × 1024 = 1M distinct pairs
- 5M × 256 / 1M = 1280× coverage per pair per SE. Still plenty.

For seq_len=4096:
- 16M pairs, ~80× coverage per pair per SE. Marginal but workable
  with multiple SE.

So **K=16 K/V slots and seq_len=128 to 4096 are all comfortably
within training budget**. The question is whether the model learns
the mapping cleanly.

## Inference

K/V cache holds full preceding seq_len positions during generation
or eval. Each new query position attends over all of them. Standard
transformer inference.

The asymmetry: training sees K/V subsets, inference sees full K/V.
But this asymmetry is exactly what makes sparse-attention training
work — the model is forced to be robust to which positions are
present, so it generalizes to "all positions present."

Initial trie walk per generation step: ~d operations (walk trailing
d chars from root) to compute the new K/V slot's trie-node embedding.

## Position Encoding Options

1. **RoPE on corpus position**: each K/V slot's RoPE rotation is its
   actual corpus position p. Q's RoPE is the firing's corpus position.
   The dot product gives relative offsets natively. Standard.

2. **RoPE on relative offset**: just encode `Q_pos - K_pos`. Cleaner
   for the model. Both options should converge, RoPE-relative is
   slightly tighter.

3. **Multi-scale Fourier RoPE**: standard RoPE has one frequency
   per dim; multi-scale uses several frequencies covering different
   distance scales. Lets a single embedding carry "this is offset X
   at fine scale, Y at medium scale, Z at coarse scale." Provides
   richer position info per dimension; useful if seq_len gets big.

Default for POC: RoPE on relative offset. Multi-scale is a tunable
upgrade.

## Implementation Sketch

**Offline (one-time):**

- For each trie node n with mass M, store a sample of K corpus
  positions where n's d-context appears (or all of them if M small).
  This is "n's occurrence index."
- Memory: ~5M corpus positions × 4 bytes = 20 MB plus the trie.

**Per training firing:**

```
For trie node n at firing event in partition group:
  occurrence_p = sample(n.occurrences)        # pick a corpus position
  window = [occurrence_p - seq_len + 1, occurrence_p - 1]
  sampled_positions = sample_K_from(window, strategy="dist-decay")
  for each pos in sampled_positions:
    trie_node_at_pos = node_at_position[pos]   # offline-precomputed index
    K[i] = node_embedding(trie_node_at_pos)
    V[i] = same or learned-V-projection
    rope_offset[i] = pos - occurrence_p        # relative
  Q = current node's embedding
  Q_rope = 0  # at the firing position
  attention(Q, K, V, rope_offsets)  # standard
  loss = KL(softmax(output), node.counts_dist) # standard AGPT
```

**Trainer changes:**

- Replace ancestor-chain K/V loader with subset sampler
- Add corpus-position bookkeeping per trie node
- RoPE position encoding (microgpt has it; agpt_train CUDA needs to
  add or adapt)

## Hard Bits

1. **CUDA kernel changes**: AGPT trainer's K/V cache currently uses
   compact bf16 storage indexed by trie-node depth. Subset attention
   needs per-firing dynamic K/V from sampled positions. Probably
   easier to gather K/V at firing time (extra memory ops) than to
   maintain a static cache.

2. **Per-trie-node corpus-position index**: scaling. At Gutenberg
   5M with 7M radix nodes, average mass per node is ~0.7 (most are
   caps with mass=1). Storing one corpus position per cap: 7M × 4
   bytes = 28 MB. Trivial. Storing ALL occurrences for high-mass
   nodes: bounded by corpus size (5M positions × 4 bytes = 20 MB).
   Total disk + RAM impact: ~50 MB.

3. **Embedding parameterization**: same options as
   `notes/trie-node-attention.md` — per-cluster, factorized,
   or hybrid.

4. **Training stability**: introducing position variance per firing
   could destabilize early training. Warmup might need to be longer.
   Or: start with K/V = ancestor chain (current AGPT), gradually
   blend in distant samples over warmup epochs.

## What This Buys

- **seq_len decoupled from d**. Train d=16 with seq_len=128 or 1024
  or 4096. The trie still encodes 16 chars per slot; the attention
  spans whatever you train at.
- **Generation past d works naturally**: the K/V cache at inference
  holds the full preceding context, which the model has learned to
  attend over.
- **AGPT mechanism preserved**: per-trie-node loss, structured-batch
  pd=6 firing, shuffle, cosine. The 60% PPL win on Gutenberg should
  carry forward.
- **Compute per firing unchanged**: K=d=16 K/V slots, attention is
  K×K = 256 per-firing operations, regardless of seq_len.

## Comparison to Trie-Node Attention POC

| Aspect | Trie-Node Attention (SGD-flavored) | Subset-Attention AGPT |
|---|---|---|
| Loss | Per-corpus-position next-token CE | Per-trie-node KL on counts (AGPT) |
| Training schedule | Standard SGD batches | Structured pd=6 batches with shuffle/cosine |
| K/V at each step | seq_len trie-node embeddings | K (~d) sampled positions' trie-node embeddings |
| Compute per step | O(seq_len²) | O(K²) |
| Loses AGPT mechanism? | YES (becomes SGD with trie features) | NO (preserves per-firing AGPT loss) |
| Headline PPL bet | Comparable to standard SGD on chars + better via richer input | Same 60%-better-than-SGD ceiling, extended to long seq_len |

Subset-Attention preserves what we know works.

## POC Plan

- Implement on top of existing agpt_train CUDA. Modifications:
  - Add per-trie-node corpus-occurrence index (offline pass)
  - Add subset sampler (CUDA kernel or CPU-pre-sampled list per firing)
  - Add RoPE-relative position encoding
  - Adapt K/V gather to sampled positions
- **Testbed**: Shakespeare 1M d=16 (we have headline numbers there).
  Try seq_len=64, 128, 256 with K=16.
- **Pass criterion**: at seq_len=128, eval-PPL beats standard SGD at
  seq_len=128 by similar 60% margin AGPT showed at seq_len=d.
- **Stretch**: at seq_len=1024 or 4096, generation past 128 chars is
  coherent (qualitative test, not just PPL).

## Effort

Substantial CUDA modifications to agpt_train (~1-2 weeks for clean
implementation), but every piece is contained:

| Task | Time |
|---|---|
| Per-trie-node corpus-occurrence index | 1 day |
| Subset sampler + K/V gather | 3 days |
| RoPE-relative position encoding | 2 days |
| Integration + sanity tests | 2 days |
| Shakespeare 1M POC at seq_len=128 | 1 day |
| Eval + comparison to AGPT-d=16 baseline | 1 day |
| **Total** | **~2 weeks** |

If POC succeeds, scaling to Gutenberg 5M and seq_len=1024+ is mostly
training time + minor optimizations.

## Status

Idea documented. No implementation yet. The cleanest entry point
(if we proceed) is implementing the offline corpus-occurrence index
first as a small Crystal tool, then layering the trainer modifications.
