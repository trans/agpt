# Trie-Node Attention: Breaking the d=seq_len Shackle

## The Bind

AGPT trains a transformer where each query position attends within a
trained context of length d (the trie depth). Past d, position
embeddings (or RoPE rotations) are untrained and attention reads
garbage. So d=seq_len is a hard ceiling: a d=32 model cannot generate
or evaluate coherently past 32 chars of context.

This ceiling is fundamental to AGPT's training data — the trie itself
only encodes d chars of context per node — but it doesn't have to be
the model's ceiling. The trie's depth limits *what each input encodes*,
not *how many inputs the model can attend over*.

## The Idea

For a corpus span of length **seq_len** (independent of d):

1. For each corpus position p in the span, find the trie node that
   represents the d-char path ending at p:
   `node[p] = trie_walk(corpus[p-d+1 .. p])`

2. The attention input at sequence-position p is `node_embedding(node[p])`.
   Position info comes from the RoPE/positional encoding indexed by p
   (the actual corpus position — or relative position within the span).

3. Train the model at any seq_len you want. The trie depth d only
   bounds *what each token's input encodes*, not the attention range.

The d=seq_len shackle breaks because:

- Each input slot is a content-derived embedding (the trie node), not
  a token+position pair where position must lie in a trained range.
- RoPE/position encoding sees positions 1..seq_len at training. d is
  a separate hyperparameter that just controls the encoding width per
  slot.
- At inference, seq_len can be whatever you trained at (or extended
  via standard RoPE-scaling tricks — but the trie content provides
  rich structural signal that may make extension easier than for
  raw-token transformers).

The user's framing: *"positions in the corpus determine the positions
in the trie, so the position numbers are assigned. The key is that
tree nodes can have more than one position."* When a d-char substring
repeats in the window, the same trie node embedding shows up at
multiple positions — the model learns to use this repetition.

## Counting things

Gutenberg 5M, d=16:

- Corpus positions: ~5M
- Total radix nodes: 7M (most caps)
- Unique distribution-clusters at d=12: ~109k (per `agpt_dist_sim`)
- Attention input at any position: 1 trie node embedding
- Attention window of seq_len=128: 128 embeddings (same dim as
  current attention) but each encodes 16 chars of context
- Attention window of seq_len=1024: 1024 embeddings; effective
  context = 1024 chars; standard transformer compute

Per training/eval window of length N:
- N corpus positions
- N trie nodes (often with repetitions when contexts recur)
- O(N²) attention as usual
- Each attention computation now relates two d-char contexts, not two
  single chars

## Embedding parameterization

7M unique trie nodes is too many for direct lookup, but:

**Option A — Per-cluster embedding (~100-200k embeddings):**
Use the distribution-cluster id from `agpt_dist_sim` as the embedding
key. Nodes that "predict the same thing" share an embedding. Lookup
is O(1) via a precomputed `node_id → cluster_id` table.

**Option B — Factorized via path content (~tiny):**
`node_emb(n) = MLP(concat([char_emb(c) for c in path(n)]))`
or similar. The model learns d position-conditioned char projections;
total params: V × d_emb × d ≈ 65 × 64 × 16 = 67k. Very compact.

**Option C — Hybrid:**
Per-cluster for high-frequency clusters (top 10k), factorized for
the long tail. Gives sharp signal where it matters, generalization
elsewhere.

Option B is the simplest POC starting point.

## Training data layout

**Offline preprocessing:**

For each corpus position p:
- Walk corpus[p-d+1 .. p] from trie root → trie node id (i32)
- Store as a flat array: `node_at_pos[p] = i32` for p ∈ [0, corpus_len)
- Memory: 4 bytes × corpus_len. For Gutenberg 5M: 20 MB.

**Per training step:**

- Pick random window start s ∈ [0, corpus_len - seq_len)
- Look up `node_ids = node_at_pos[s : s+seq_len]` (one cache-friendly
  array slice)
- Convert each node_id to embedding via factorized lookup or cluster table
- Forward pass: standard transformer over seq_len embeddings + position
  encoding
- Loss: per-position next-token CE against `corpus[s+1 : s+seq_len+1]`

**Per inference step:**

- Maintain a running buffer of last seq_len char's trie node ids
- Each new position: walk trailing-d corpus slice → trie node id
- Append to buffer (O(d) per step)
- Forward pass using the buffered embeddings

## Comparison to existing AGPT

| Aspect | Current AGPT | Trie-node attention |
|---|---|---|
| Training data | Trie nodes (loss at every node firing) | Corpus positions, but inputs are trie nodes |
| seq_len | Bounded by d | Independent of d |
| Position info | Standard pos emb (limited to d) | Standard RoPE/pos emb (any length) |
| Per-position input | One char + position | One trie-node embedding |
| Effective context | d | d × seq_len-influence-range |
| Training-position generalization | None past d | Standard RoPE-extension techniques apply |
| Total trainable params | ~108k (current model) | Similar + node embedding atoms (67k for option B) |

## What this changes architecturally

The current AGPT trainer fires loss per radix-trie node, processing
the trie's structure directly. **This proposal trains over corpus
positions** — closer to standard SGD — but uses trie-node embeddings
as inputs.

It's a hybrid:
- Use the trie to pre-compute per-position content embeddings (offline)
- Train a standard transformer over those embeddings (online)
- The trie's job is feature engineering at fixed depth d
- The transformer's job is long-range attention over those features

Crucially, this **is no longer AGPT-the-trainer** — it's "SGD with
trie-derived input embeddings." The AGPT mechanism (per-trie-node
loss, structured-batch optimization) is replaced by per-corpus-position
loss.

What we keep from AGPT:
- The trie itself (the offline structural extractor)
- The trie's depth-d local-context summarization
- All the parrot-style structural insights

What we lose:
- The per-trie-node loss (which gave AGPT its distinct gradient signal)
- The structured-batch optimization that pd=6 + shuffle gave us

What we gain:
- Position-free attention range (seq_len decoupled from d)
- Standard transformer + standard SGD = compatible with all existing
  training infrastructure
- Generation past d works naturally
- Long-range coherence becomes possible

## Open architectural questions

1. **Single node per position vs full ancestor chain?**

   Per position p, we have ancestor chain: trie nodes for
   corpus[p-1..p], corpus[p-2..p], ..., corpus[p-d..p]. d nodes per
   position. Two architectures:

   - **Deepest only** (1 node per position): standard seq_len = N
     attention. Simpler. Each input encodes the full d-context.
   - **Full ancestor chain** (d nodes per position): N × d "tokens"
     in the attention window. Richer; each position contributes d
     graded context summaries from depth 1 to d. Harder to compose
     (need to decide how to attend across positions vs across depths).

   Start with deepest-only. Ancestor chains can be added later if
   needed (likely as a hierarchical attention layer).

2. **Position encoding scheme?**

   - Standard learned positional embeddings (1..seq_len): simple,
     same shackle but at seq_len granularity not d.
   - RoPE: extends naturally to longer seq_len at inference time
     with known caveats.
   - Relative attention: anchor-node-friendly; positions become
     "how many anchor units behind."

   Default: RoPE with seq_len matching what we train at. Extension
   strategies (linear/NTK/YaRN scaling) are well-studied and orthogonal
   to this proposal.

3. **How do we train the embedding atoms?**

   In options B and C the per-char-per-depth atoms are just embedding
   parameters that backprop through normally. No special handling.

4. **What's the right seq_len to train at?**

   Pick the largest seq_len memory allows. seq_len=512 or 1024 is
   easy at d_model=64 with 2-layer transformer. seq_len=4k+ on big
   corpus is the test that proves long-range coherence.

5. **Is this still AGPT?**

   No. The AGPT distinct-mechanism contribution becomes moot — we're
   doing SGD with trie-derived embeddings. The trie becomes feature
   engineering, not a training framework.

   But: AGPT might still be useful for *training the trie-node
   embeddings* via its per-node loss before plugging them into the
   transformer. That's a possible hybrid: pre-train per-node embeddings
   via AGPT, then freeze them and train transformer over corpus
   positions with those embeddings as inputs.

## Minimal POC

- **Corpus**: Shakespeare 1M (small + we know its structure)
- **d**: 16 (dense enough)
- **seq_len**: 128 (4× current d=32 ceiling)
- **Embedding**: option B (factorized)
- **Architecture**: 2-layer transformer, d_model=64, RoPE, trained
  with standard SGD on corpus position windows
- **Comparison**: vs current AGPT d=32 pd=6 cosine 12SE (PPL 3.55 at
  seq=32) and SGD seq=128 100k (PPL ~7 on Shakespeare).

**Pass criterion**: PPL at seq_len=128 should be substantially better
than SGD's PPL at seq_len=128 *because the model has trie-derived
local-context features available at every position.* Match or beat
the AGPT seq=32 number while extending the eval range to 128.

## Effort estimate

| Task | Time |
|---|---|
| Build per-position trie-node-id index (offline) | 1 hr |
| Add factorized embedding layer to microgpt | 2-3 hrs |
| Modify dataset loader to use node_id index | 2 hrs |
| Training loop unchanged (it's standard SGD) | 0 |
| First Shakespeare run | 30 min wall |
| Evaluation + comparison | 1 hr |
| **Total minimal POC** | **~1 day** |

If the POC works, scaling to Gutenberg 5M is mostly a matter of memory
for the node-id index (20 MB) and longer training time.

## Risks

- **Trie collisions cap distinguishability.** ~109k unique
  distributions at d=12 — multiple corpus positions share the same
  trie-node embedding. The model can't tell them apart from
  embedding alone. Position encoding must carry the disambiguation.
  RoPE handles this in principle.

- **Embedding table memory at large d.** Per-cluster option (option A)
  needs a `node_id → cluster_id` map (~7M entries × 4 bytes = 28 MB).
  Plus cluster embeddings (~100k × d_emb × 4 bytes = 26 MB). Fine.

- **Generation needs trie walks per token.** Each new token requires
  walking the trie with the trailing-d context to get the new node id.
  ~d operations per token. Fast on CPU; would need optimization for
  GPU inference.

- **Mismatch between offline-trie and online-corpus**: at inference,
  the model generates chars not in any training-position d-context.
  The trailing-d walk might find no matching trie node (an OOV
  d-context). Need a fallback: walk shallowest matching prefix and
  pad embedding, or always include a "no-match" embedding atom.

## Status

Idea documented; no implementation yet. Branch reservation:
`trie-node-attention`. The minimal POC requires modifying microgpt
to accept content-derived position embeddings, which is a moderate
but contained change.
