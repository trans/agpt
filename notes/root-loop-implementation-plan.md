# Root-Loop Implementation Plan (Phase 2 spec)

Branch: `agpt-root-loop`
Goal: extend current AGPT training to context K·D while preserving
AGPT aggregation and avoiding classical-backoff sparse regions.

**Phase 1 status**: CLI flags `--virtual-cycles K` and `--corpus
<path>` added to `bin/agpt_train`. K=1 is unchanged current AGPT;
K>1 is validated and rejected with a Phase 2 not-implemented message
until this spec is built.

**Phase 2 status**: complete at K=2.
- Stage A (bbb090d): `virtual_cycles` threaded through `run_radix_training`
  and `run_per_subtree_training`. CLI → training entry points.
- Stages B+C (384fed6, 1439130): `compute_virtual_prior_tokens_per_rc`
  (per-root-child prior via D-trie bigram-valid Markov-1-greedy walks)
  + `build_rope_cache_prior` (neg-pos RoPE cache).
- Stage D (28d6756): per-layer prior K,V mini-forward at each subtree
  entry. Stop-gradient.
- Stages E+F+G (58534ee): KV cache extended with prior tail slots;
  prior scattered into those slots instead of separate buffer;
  cycle-k>1 forward+backward runs on the same chunk with prior-prepended
  prefix_char_ids; grads accumulate into one RMSProp step per subtree.
  sv_attn_weights[l] stride extended to cover prior prepend.
- Behavior: K=1 bit-identical to pre-Phase-2. K=2 d=8 runs at ~2× K=1
  runtime, losses within run-to-run noise. K=2 d=16 per-subtree
  completes in ~18s/super-epoch on the test machine.
- K≥3 is supported architecturally but `compute_virtual_prior_tokens_per_rc`
  only handles single-segment priors today — needs recursive D-segment
  walks for K≥3.

## Construction (to confirm before coding)

Within each D-segment of the virtual tree, we walk the D-trie from
some seed token through D steps. At the D-boundary, the leaf token
becomes the seed for the next segment.

**Open detail:** what determines the distribution *within* a segment
at segment-relative depth j (0 < j ≤ D)?

- **Option M1 (Markov-1 everywhere):** at every step, the next
  token's distribution depends only on the immediately previous
  token, via the D-trie's root-child→grandchild transitions. Simple.
  Uses only the shallowest layer of the D-trie throughout.

- **Option Mj (Markov-cycle-relative-depth, default for root-loop):**
  at segment depth j, distribution depends on the last j tokens
  *within the current segment*. Uses progressively deeper D-trie
  layers within a segment, resetting to Markov-1 at each D-boundary.
  This uses the D-trie's structure fully within a segment while
  still avoiding crossing into sparse deep regions.

Your example was consistent with both, since D=2 didn't exercise the
distinction. I believe **Mj** is what you want — it exploits the
full D-trie within each segment and resets at the boundary, which
matches "root-looping" more literally. M1 throws away the intra-
segment D-trie structure.

Confirm: **use Mj.** Let me know if M1 was what you meant instead.

## Training structure

For each root-child subtree (as in current AGPT), train on the
virtual tree subtree rooted at that root-child. The virtual tree
subtree extends beyond the D-trie subtree via recursive D-segment
attachment.

At each virtual-tree node V in the subtree:

- Forward input = V's virtual-ancestor chain (≤ K·D tokens).
- Loss target at V = D-trie lookup at (V's current segment-relative
  depth) within V's current segment.
- Gradient aggregates at V level.
- Subtree-scoped Adam step: one step per root-child subtree, as today.

## Virtual-ancestor construction

For a radix node N at D-trie depth d_N, its virtual-ancestor chain
at virtual depth (k-1)·D + d_N (= its k-th virtual occurrence) is:

```
[ segment 1 tokens (D of them),
  segment 2 tokens (D of them, seeded by segment 1's leaf),
  ...
  segment (k-1) tokens (D of them),
  segment k's first d_N tokens = N's D-trie ancestor chain ]
```

Each segment's D tokens are a D-trie path from the segment's seed
token. Segment 1's seed is the D-trie root; segments 2..k−1 have
seeds determined by the previous segment's leaf.

## What's the canonical virtual position for N?

A radix node N has multiple potential virtual positions (one per
cycle k ∈ {1, 2, ..., K}). For training, we need to pick one — or
train on all.

**Option A: Train only at cycle 1 (no extension).** This is current
AGPT. No gain.

**Option B: Train at cycle K only.** The deepest virtual position
for N. Gives the transformer the longest context. Requires
constructing K−1 prior segments. Need to pick the canonical prior
segments.

**Option C: Train at all cycles.** K forward/backward passes per
radix node (at cycles 1, 2, ..., K). K× compute.

**Option D: Pick cycle k at random per pass.** Like Option C but
stochastically sampled. Same expected gradient as Option C, 1/K
the compute.

**Option D is probably the right starting choice** — random cycle
per training pass gives the transformer exposure to contexts of all
different positional offsets without K× compute.

## What's the canonical prior for a chosen cycle > 1?

When we train N at virtual cycle k > 1, we need to fill in the
k−1 prior segments. Options:

**Option i: First-occurrence in corpus.** For each N, find the first
corpus position where N's D-gram appears with at least (k−1)·D
corpus tokens before it. Use the actual tokens at those prior
positions as the prior segments. **Cost:** one pass over corpus to
index. **Pro:** prior is natural (real corpus text). **Con:**
arbitrary choice among many occurrences; training/inference may
still mismatch for other occurrences.

**Option ii: Markov-1-greedy from root.** Starting at D-trie root,
at each step pick the highest-probability transition; at D-boundary
the leaf seeds the next segment. Deterministic; always the same
prior for all cycle-k training of N *with the same final D-gram*.
**Pro:** fully canonical, always defined. **Con:** prior is a
specific synthetic path, not a real corpus sequence.

**Option iii: Zero-padding.** Prior = (k−1)·D pad tokens. Gives the
transformer "nothing" for the prior. Test whether just having the
positional encoding extend helps.

**My recommendation**: **Option ii (Markov-1-greedy) for the first
experiment.** Deterministic, cheap, fully automatic. If it works,
we have a result. If it doesn't, Option i is the more realistic
but costlier fallback.

## Settled design (from the root-loop design discussion)

These points are settled — Phase 2 should implement them, not
re-debate them:

1. **Within-segment Markov order: Mj** (not M1). Within each
   D-segment, the distribution at segment-relative depth j uses the
   last j tokens of the segment. At the D-boundary, the leaf token
   seeds the next segment at depth 1 (reset).
2. **Cycle iteration: K× forward/backward passes per radix node**.
   Each of N's K virtual positions (one per cycle) gets its own
   training pass with cycle-specific ancestry. No random sampling —
   we iterate all cycle positions deterministically. Compute scales
   roughly linearly with K (not K²), because prefix-sharing
   compounds: cycle 1 compute = current AGPT; cycle 2 ≈ 2× cycle 1;
   etc. Net for K=2: ~3× current AGPT.
3. **Canonical prior: D-trie-derived, per root-child subtree** (NOT
   per-radix-node, and NOT from the corpus file). All radix nodes
   in a root-child subtree share that subtree's one canonical
   prior. This keeps KV cache sizing tractable and keeps the
   trainer self-contained (no external corpus dependency).
   The prior for root-child R is any D-trie depth-D path whose
   leaf-token has R's first-token as a valid D-trie bigram
   continuation. Simplest realization: Markov-1-greedy walk from
   D-trie root, constrained so the resulting leaf can transition
   into R. Deterministic per R, reproducible from just the trie.
4. **Optimizer: RMSProp**. The winning recipe on the d=16 and d=32
   AGPT runs used RMSProp+warmup-cosine. Same primitive here; just
   swap `cuda_adam_bulk` for `cuda_rmsprop_bulk`.
5. **K=2 first**. "Start small" — get K=2 at D=16 working first,
   measure, iterate. K=4+ comes later if the pattern holds.
6. **Aggregation semantics**: subtree-scoped optimizer step stays
   at "one RMSProp step per root-child subtree." Within the step,
   gradients accumulate over all K cycle-positions of all radix
   nodes in the subtree.

## Phase 2 implementation outline

### Step 1: Prior computation (CPU side, from the D-trie alone)

For K>1, we need per-root-child-subtree priors. One per root-child,
(K-1)·D tokens each, derived purely from the D-trie (no corpus
file needed). Algorithm for each root-child R:

1. Identify R's first-token t_R.
2. Find a D-trie depth-D leaf whose leaf-token has t_R as a
   D-trie-depth-1 child (i.e., the bigram `leaf_token → t_R` is
   present in the trie's depth-1-to-2 edges).
3. If multiple such leaves exist, pick one deterministically — e.g.,
   highest `edge_mass` along the path to the leaf. Walk that path
   to collect (K-1)·D tokens.
4. If no valid leaf exists (rare edge case for small tries), fall
   back to zero-padding.
5. Store priors array: `int prior[n_root_children][(K-1)·D]`.

Cost: O(n_root_children × D-trie-search). For Shakespeare K=2 D=16:
65 × quick walk = negligible. Storage: 65 × 16 = 1040 ints = 4 KB.

Implementation note: this can sit directly in agpt_train.cu CPU-side
after `load_radix_trie`; no builder-side change needed.

### Step 2: KV-cache allocation extension

Current KV cache is sized to `total_edge_chars × d_model × 2 × n_layers`.
Add prior KV storage: `n_root_children × (K-1)·D × d_model × 2 × n_layers`.

For Shakespeare K=2 D=16: 65 × 16 × 64 × 2 × 2 × 4 bytes = 1 MB.
Negligible.

Allocate as a separate `d_prior_kv[layer][root_child_id]` buffer
(so it can be indexed by subtree at training time).

### Step 3: Prior K,V computation at subtree entry

In the per-subtree training loop, before iterating the subtree's
radix nodes:

1. Load this subtree's root-child ID.
2. Look up this subtree's prior (from CPU → GPU upload of tokens).
3. Run embedding + per-layer K,V projection + RoPE on the prior's
   (K-1)·D tokens.
4. Store the resulting K,V in `d_prior_kv[layer][root_child_id]`.

These K,V are then reused across ALL the subtree's radix nodes and
both cycle positions.

### Step 4: Forward-pass attention range extension

Currently each radix-node's query attends over its ancestor_char_ids
positions (in the global KV cache). For K>1, the query's attention
range is:

- At cycle 1 (k=1): positions [ancestor_char_ids of N]. Same as today.
- At cycle 2 (k=2): positions [prior positions] ∪ [ancestor_char_ids
  of N]. The prior positions are (K-1)·D contiguous slots in
  d_prior_kv.

Concatenate these two ranges for the query's attention op. Attention
mask is causal over the combined length.

### Step 5: RoPE position indexing

RoPE must label positions correctly. In the virtual-tree picture:

- Cycle 1 positions: RoPE pos 0..D-1.
- Cycle 2 positions: RoPE pos D..2D-1.
- ...
- Cycle K positions: RoPE pos (K-1)·D .. K·D-1.

For cycle 2 training, the prior occupies RoPE pos 0..D-1, and the
radix node's ancestor chain occupies RoPE pos D..D+d_N-1.

Current RoPE cache: sized to `cfg.seq_len` (128). If K·D ≤ 128, no
change needed to the cache itself — just reference the right indices.
For K=4 D=32 (= 128), we're right at the boundary. K=2 D=16 = 32, well
within.

### Step 6: Training loop modification

In the per-subtree pass:

```
for each radix node N in the subtree, in BFS order:
    for k in 1..K:
        # k=1: standard AGPT forward (prior ignored)
        # k>1: prepend prior to attention
        forward_pass(N, cycle=k)
        backward_pass(N, cycle=k)
        accumulate gradients
# one RMSProp step at subtree end, as today
optimizer_step(subtree)
```

### Step 7: Aggregation semantics

Subtree-scoped RMSProp step unchanged. Within the step, we now
accumulate K × (subtree-radix-count) gradient contributions instead
of 1 × (subtree-radix-count). Aggregation still happens at each
virtual-tree node (one gradient per cycle-position).

### Estimated Phase 2 LOC

- Prior computation + corpus read: ~60 LOC
- KV-cache allocation extension: ~30 LOC
- Prior K,V computation at subtree entry: ~50 LOC
- Forward-pass attention range extension: ~80 LOC
- Training loop K-iteration: ~40 LOC
- Testing + verification: ~40 LOC

Total: ~300 LOC, concentrated in `agpt_train.cu`.

## Testing plan

1. **Unit-level**: with K=1, output must match pre-Phase-2 behavior
   bit-for-bit. Regress the d=16 PPL 15.28 recipe first.
2. **Smoke**: K=2 at D=16 for 1 super-epoch. Check for NaNs,
   sensible loss decrease.
3. **Full**: K=2 at D=16 for 50 super-epochs. Measure PPL vs
   current-AGPT d=16 baseline (15.28) and current-AGPT d=32
   baseline (13.17). Hypothesis: K=2 D=16 ≈ D=32 in PPL, with
   lower per-subtree memory footprint.

## Where the branch is

- Branch: `agpt-root-loop`
- Last commit: Phase 1 CLI scaffolding
- Main is untouched: d=16 PPL 15.28 and d=32 PPL 13.17 baselines
  are reproducible from main for comparison.

## Comparison plan

Against the following baselines on same model / same corpus / same
compute budget:

1. Plain window training (bin/microgpt, seq_len=K·D).
2. Current AGPT at D=K·D (if memory allows).
3. Current AGPT at D (what we have).

The clean story we're testing: does root-loop AGPT at (D, K·D)
match or exceed AGPT at D=K·D (which blows up memory for large K·D),
while being cheaper than window training?

---

## Waiting on your confirmation of the four open questions above
before starting the code changes.
