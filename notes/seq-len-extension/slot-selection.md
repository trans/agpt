# Slot Selection

## The reframing

AGPT's "context window = trie depth" constraint is self-imposed. The transformer attention layer knows nothing about the trie per se — it sees a query, a stack of K/V slots, computes softmax-weighted attention, returns. The rule that "K/V slots = path ancestors only, exactly `d` of them" is how we *populate* the slots, not a constraint the architecture imposes.

Once we accept that the trie organizes *training* (how we factorize the loss, how we batch radix nodes) but not what attention is allowed to see, the wall dissolves. The K/V pool can include any trie nodes that are likely to carry useful signal for the current query — path ancestors are one obvious source, but not the only one.

This reframing is what the closed cap-recurrence investigation was groping at without naming. Cap-recurrence added one extra slot per query, populated with a mass-weighted centroid over all corpus predecessors, with detached gradient. That specific configuration was null (see `project-cap-recurrence-null`), but the broader idea — "expose more context to attention than the path" — survives intact. The failure was the centroid + detached choice, not the extra-slot principle.

## Why this lifts the KN ceiling

Kneser-Ney smoothing interpolates the d-gram with shorter n-gram backoffs at fixed per-depth discount coefficients. It is mechanistically a backoff mixture. The most natural new K/V slots to expose are **the trie nodes representing those same backoffs** — drop the front token, look up the cousin trajectory in another root-child subtree. Attention with backoff K/V slots is the same mixture KN performs, with the interpolation coefficients learned per-context per-head instead of hardcoded per-depth.

Result: the architecture *contains* KN. The model is no longer ceiling-bounded below KN — KN is the floor it learns to start from, and the route to "beat KN" is just letting attention pick context-sensitive mixing weights that KN's fixed discounts can't.

## Phase 1: Heuristic baseline (hard selection)

Before learning anything about routing, give the model a deterministic slot mixture. Candidate slot sources:

- **Path ancestors.** The `d` nodes from root to the current query (current AGPT behavior).
- **Backoff cousins.** Lower-order n-gram trie nodes formed by dropping front tokens from the path (`c₂..c_d`, `c₃..c_d`, … — KN's backoffs).
- **Global landmarks.** A small fixed set: the root node, possibly the highest-mass shallow nodes ("hubs").

If a specific high-order node lacks sufficient gradient or frequency data, the attention mechanism naturally shifts weight to the lower-order backoff nodes in the pool. Backoff is essentially a deterministic, structural dropout designed to handle data sparsity — KN handles it via fixed discount; here attention handles it learnably.

Phase 1 is the proof-of-concept: with these structural slots present, can attention extract the KN-equivalent gain without any learned routing? Step 0 below pins this down concretely.

## Phase 2: Differentiable Top-K routing (soft selection)

To make selection learnable, score candidate slots against the current query state via a router (Mixture-of-Experts style):

                [ Current Query State: h_t ]
                             |
                     [ Router Linear Layer ]
                             |
                  [ Softmax / Gumbel-Softmax ]
                             |
               -----------------------------
              |                             |
     [Selected Node 1]             [Selected Node 42]
     (High Attention)               (Low Attention)

Routing score: `Score(t, i) = h_t · W_r · e_iᵀ` for each candidate node `i` with summary embedding `e_i`. Apply Top-K (with Gumbel-Softmax or sparsemax for differentiability) to pick the slots that go into attention.

Phase 2 design risks borrow from the MoE / Sparse-Routing literature (router collapse, load balancing, training instability) and the known countermeasures apply: load-balancing auxiliary losses, expert capacity caps, sparsemax with temperature annealing, Switch-Transformer's hard top-1.

**Adaptive K via entropy** (open exploration). One lever for Phase 2: scale K with router uncertainty. Low entropy → confident pick → small K. High entropy → multiple candidates look equally good → larger K. Caveat: routing entropy ≠ prediction entropy — a confident router with an uncertain prediction needs help (more slots) but routing entropy wouldn't trigger it. The cleaner signal may be **next-char-prediction entropy** as the budget signal, or a learned controller combining both. Out of scope for Step 0; flagged for Phase 2.

## Phase 3: Two-stage hierarchical attention

If the candidate pool grows to thousands of nodes (or the entire trie), scoring every node per query is infeasible. Hierarchical retrieval mirrors retrieval-augmented LMs:

- **Coarse selection (router).** A lightweight, low-dimensional linear layer projects `h_t` to flag a subset of `N` candidate nodes or sub-trees.
- **Fine selection (attention).** Standard multi-head attention does the precise weighting over those `N` candidates.

The coarse pre-filter is borrowed engineering — FAISS-style ANN over a learned embedding of `h_p[node]`, learned hash buckets (LSH / product quantization), or structural pre-filters (e.g., "all nodes within Hamming-K of the current suffix hash"). The fine attention is the same machinery Phase 2 introduces.

## The ultimate realization

By making selection learnable, the prefix trie ceases to be a rigid data structure and becomes a **differentiable memory graph**. Tree structure initializes with strong inductive bias (KN backoff is the architectural prior, preventing the cold-start problem that dooms many graph-neural-network approaches), and routing layers learn to teleport across branches, pulling in contextually relevant aggregated gradients from entirely different parts of the corpus.

## Cold-start guarantee

Worth saying explicitly: because Phase 1 slots are always available in the candidate pool, **the system never does worse than KN-with-attention**. The learnable router (Phase 2+) can only add value, not subtract it. The structural slots are the safety net. This is the property cap-recurrence lacked — its centroid injection had to earn its own benefit from scratch and didn't, so the path was floor-less. Here the floor is set by KN, and everything above is upside.

---

# Step 0: Heuristic path + backoff — implementation spec

The concrete first experiment. Goal: prove that the slot-expansion mechanism delivers at all, before any router complexity.

## What it is

Per query at depth `d` in the trie:

- The existing `d` path-ancestor slots (current AGPT behavior, unchanged).
- `B` new backoff slots, one per backoff level `i ∈ {1..B}`. Each carries `h_p[K_back_i]`, where `K_back_i` is the trie node found by descending from root using chars `c_{i+1}..c_d`.

For `d=16, B=4`: 20 K/V slots per query, vs the current 16. ~25% slot-count increase.

## Design decisions

1. **Trainer**: `src/cuda/agpt_train.cu` (v1) on a fresh branch off main. v1 over v2 for minimum moving parts; v2's growth and incremental-radix paths add concerns we don't need yet. Cap-recurrence branch's `kv-inject` infrastructure is *not* reused — it carried the wrong assumptions (single slot, centroid, env-var-gated). Add the backoff path fresh.

2. **`K_back` identification**: deterministic via trie descent. For backoff level `i`, walk from root using chars `c_{i+1}..c_d` of the current query's path. If the radix trie has a node at that suffix, that's `K_back_i`. If not (rare — suffix never occurred in the corpus), skip that slot. KN backs off the same way.

3. **`h_p[K_back]` provenance**: in-flight forward pass during the current fire, computed *alongside* K's path by widening the per-depth batches (see the implementation sketch below for what "alongside" means concretely). Same parameters as the path-ancestor forward; no new weights for the backoff path itself.

4. **Gradient flow**: end-to-end backprop through the extra forward. `K_back`'s parameters thereby receive two gradient signals per epoch:
   - Its own fire's "predict next char from `K_back` as endpoint" gradient (existing behavior).
   - Every fire that backs off to it: "be a useful upstream representation for queries that attend to me" gradient (new).
   
   The two signals stack — every node gets pressure to be useful upstream, not just useful as an endpoint. This is the core mechanism by which the architecture is supposed to beat KN: cached/detached `h_p` would only carry the endpoint signal and would collapse to "soft KN as a fixed prior."

5. **Shared Q/K/V parameters**: the same `W_q`, `W_k`, `W_v` matrices project both path-ancestor states and backoff states into the attention space. No new parameters introduced. Preserves apples-to-apples ablation vs current AGPT (baseline = same architecture with `B=0`).

6. **Position encoding for backoff slots — A/B test in Step 0**: two candidate schemes, both run as paired experimental conditions:

   - **(a) Sentinel `d+i`**: simplest scheme; distinguishes backoff slots from path slots by giving them a position the path never visits. Argument: cheap, no position duplication, model learns "slot at d+i = backoff slot of level i."
   - **(b) Same-position-as-K (RoPE position `d`)**: K_back_i is a *present-moment alternative prediction* for the same target slot as K's query, not a past-token or future-token. Its hidden state was computed as the final layer output of a length-(d-i) walk, which is itself a query-ready prediction state. From K's attention frame, K_back_i sits at K's current temporal moment with relative rotation 0. This is the semantically aligned choice — K and every K_back_i are alternative shadow predictions for the same next-char slot ("cat on the mat" vs "at on the mat" predict the same thing at the same position).

   (An earlier draft had depth-relative `d−i` as the second variant. That was rejected: K_back_i isn't a context token *i* steps in the past; its final hidden state is itself a query-ready prediction for the same slot K is predicting. Same-position-as-K replaces depth-relative as the principled choice; sentinel is the "tag as special" fallback.)

   Both are zero-parameter changes. Step 0 runs both. Other position-encoding variants — learnable per-level embeddings, zero RoPE, per-head specialization — remain Phase 1.5 follow-ups once we know which Step-0 candidate wins.

7. **No within-fire dedup of `K_back` paths**: queries in the same chunk that share a `K_back` independently include that `K_back`'s path as a parallel mini-path inside the depth-batched forward (see below). Cost is bounded (~4–5× compute per query); within-fire dedup is a Step-0.5 optimization.

8. **Start at `partition_depth: 0`** (single fire per epoch over the whole trie). At pd=1 the trie is sharded into 65 root-children that fire separately, and K_back_i lives in a *different* shard from K — bringing its forward into K's fire means importing chunk data across shards. At pd=0 there is only one fire, K and every K_back_i are already part of the same training unit, and the cross-subtree concern dissolves into a much smaller within-fire chunk-membership concern (do K and K_back_i fall in the same `chunk_queries`-sized chunk?). Step 0 runs at pd=0. Once we know the architecture works, the pd=1 generalization is a separate (harder) implementation step that crosses subtree fires.

## Implementation sketch

### How AGPT v1's fire works today

A "fire" in v1 = one per-subtree forward + backward + optimizer step. At `--partition-depth 1` the trie is partitioned by root char (65 root-children for Shakespeare ASCII), and one fire processes one root-child's subtree. **At pd=0 there is only one fire per epoch over the whole trie** — the trainer chunks for memory (via `--chunk-queries`, default 50000) but otherwise treats the entire trie as a single training unit. Step 0 runs at pd=0 (decision 8) — every K and every K_back_i are members of the same fire by construction, no cross-subtree imports.

Within a fire, the kernel **does not walk paths serially**. Chunks of positions are sorted by depth, and at each depth `j` the kernel processes *all chunk positions whose endpoint depth is ≥ `j`* in parallel as one batched layer step. Positions whose endpoint depth equals `j` consume their loss target at that depth step and **drop out of the batch**; deeper steps process only the still-growing paths. The active batch naturally shrinks from depth 0 to `max_depth`.

So the kernel already knows how to handle "positions with different endpoint depths sharing one chunk." Adding backoff slots reuses exactly this machinery — we don't add a new masking concept.

### The right model: backoff slots ride on anc-grad

The earlier drafts of this section over-engineered the gradient flow problem by treating chunked backoff as fundamentally new. Two drafts back: "no satellites needed at pd=0." Last draft: "cross-chunk forces a two-pass + deep-to-shallow scheme + stash_grad." Both missed that **the same problem is already solved for path-ancestors** by the existing `--anc-grad` infrastructure (`todo/descendant-ancestor-scatter.md`, on main since 2026-05-20).

**The anc-grad insight.** When a descendant query attends to an ancestor A's K/V slot, K_A = W_k^l · LN1^l(h_A^{l-1}). The gradient on W_k^l from that descendant is a **closed-form matmul**:

```
dW_k^l += dK_A^T · LN1^l(h_A^{l-1})
```

No backprop through A's prior layers needed. The descendant→ancestor gradient flow becomes:

1. **Forward (per chunk)**: save `ln1_out` at each compact-char position into a fire-global buffer `h_subtree[l]` indexed by `(subtree-local idx, layer)`. Already in the code at line 2641-2672.
2. **Backward (per chunk)**: when attention backward produces gradient on an ancestor K/V slot, atomic-scatter it into `d_dkv_subtree_k[l]` / `d_dkv_subtree_v[l]` at that ancestor's compact-char slot. Already in the code at line 2573-2640.
3. **Fire end**: one batched sgemm per layer: `dW_k^l += d_dkv_subtree_k[l]^T · h_subtree[l]`. Already in the code.

The approximation: only W_k^l / W_v^l receive the descendant→ancestor signal. M's pre-layer-l parameters (Wq, Wo, FFN, LN of layers 0..l-1) don't get the cross-chunk contribution. The codebase has empirical evidence this is fine — anc-grad shipped with measurable improvement on held-out PPL.

**Backoff slots are exactly the same shape.** K's backoff slot at level i is `K_back = W_k^l · LN1^l(h_M^{l-1})` where M = K_back_i. The same closed-form matmul applies. So we ride directly on the existing anc-grad infrastructure:

- **Forward (gather extension)**: at K's primary endpoint, after appending path-ancestors + own-edge to `d_kv_pack_k/v`, also append B backoff K/V slots. For each level i: read `h_subtree[l][M's compact-char slot]` (which IS `LN1^l(h_M^{l-1})`), project through W_k^l/W_v^l, apply RoPE at the chosen position (decision 6 — `d` for same-as-K, `d + i` for sentinel), append.

- **Backward (scatter extension)**: extend the existing anc-grad scatter to also handle backoff slots. The attention backward already produces gradient on every K/V position the queries attended to, including the new backoff positions. The new scatter walks each query's backoff slots and atomic-adds the gradient at the backoff slot to `d_dkv_subtree_k[l]` / `d_dkv_subtree_v[l]` at M's compact-char slot — the same accumulator the ancestor scatter writes into.

- **Fire end**: the existing anc-grad reduction (one matmul per layer) now sums both the ancestor and backoff contributions. No code change at the reduction step.

### Why no two-pass, no deep-to-shallow, no stash_grad

**No two-pass.** Chunks are already iterated in BFS order — shallow nodes first, deep nodes last. M (= K_back_i, shallower than K) is processed in an earlier chunk than K. By the time K's chunk's forward gather runs, `h_subtree[l][M's slot]` has already been populated by M's chunk's forward (it's `ln1_out` at M's endpoint, written via `launch_save_ln1_to_subtree` at line 2668). No need for a forward-only stash-population pass.

**No deep-to-shallow.** Backward doesn't need M to "consume" stash_grad before K writes to it. K's backward writes to `d_dkv_subtree_k[l]` (just like K's existing ancestor-backward writes); M's params get the contribution via the fire-end matmul, regardless of chunk order. The backward chunk order is just the reverse of forward (deep-first), which is already what v1 does.

**No stash_grad buffer.** `d_dkv_subtree_k[l]` IS the gradient accumulator we need. Already allocated, already zeroed at fire start, already reduced at fire end. The backoff backward just adds entries into the same buffer.

**No d_stash buffer.** `h_subtree[l]` IS the forward stash we need. Already allocated, already populated per-chunk, already accessible by compact-char slot.

### Compute and memory

**Compute cost: small.** The new forward gather appends B extra K/V positions per primary endpoint query. The new backward scatter adds B entries per primary endpoint per layer. Both are bandwidth-bound and small relative to the existing attention. The fire-end reduction matmul is already there; backoff entries just contribute more rows. Order-of-magnitude: ~5-10% overhead, not 2×.

**Memory cost: zero new buffers.** Uses `h_subtree[l]` (already allocated) and `d_dkv_subtree_k/v[l]` (already allocated). The only new per-fire host-side data is a small `M.id → compact-char slot` lookup, built once at fire start.

For Shakespeare d=16 the existing 25-ep run is ~10 min; this becomes ~10-11 min. The cost of the backoff mechanism is dominated by the extra attention bandwidth and the matmul width at fire-end reduction.

### What's actually new vs the existing kernel

1. **Sidecar load + rev_lookup** (host-side, once per fire). Already on the branch (`55cbe30`). Read `<trie-dir>/backoff_B<N>.bin`, build the inverse `M.id → list of (K.id, i)`.
2. **K_back-to-compact-slot lookup** (host-side, once per fire). For each sidecar entry where K_back_id != SENTINEL, compute M's endpoint compact-char position via existing trie tables; map K_id → array of B compact slots (or -1 for sentinels). Upload to device.
3. **Forward gather extension** (small CUDA kernel). At K's primary endpoint, read `h_subtree[l][K_back_compact_slot]`, project through `W_k^l`/`W_v^l`, apply RoPE at chosen position, append to `d_kv_pack_k/v`. Update per-query `kv_lengths_per_q`. ~40 lines of CUDA.
4. **Attention kernel extension**. `cuda_batched_varlen_attention_L_queries` takes an optional `kv_lengths_per_q`. When non-null, the per-query loop length uses that instead of per-node `kv_lengths`. Null preserves bit-exact B=0 parity.
5. **Backward scatter extension** (small CUDA kernel). After attention backward produces gradient on backoff K/V slots, scatter into `d_dkv_subtree_k/v[l]` at M's compact-char slot, with the inverse RoPE rotation applied to the K-side (mirroring the existing ancestor inverse-RoPE in line 2641-area). ~30 lines of CUDA.
6. **Sentinel handling**: K_back_id == SENTINEL or "no compact slot" → that backoff slot is just skipped at both gather (no append) and scatter (no contribution).
7. **Constraint**: backoff slots requires `--anc-grad` enabled (it's the default since 2026-05-20; rejected with a clear error if disabled, since the closed-form path is what we ride on).

### Case 2 (mid-edge backoff targets): dropped for Step 0

Some K_back_i targets land mid-edge of a compressed radix node N rather than at a node endpoint. The K/V cache in v1 only stores entries for branching (mass>1) positions — mass=1 positions (compressed-edge interiors) are NOT in the cache, by design (see `agpt_train.cu:2326-2337` and the `compact_slot` mechanism). For Step 0 these "case 2" backoff slots are marked SENTINEL in the sidecar and dropped at gather time. We measure the case-2 rate during sidecar construction.

If the empirical case-2 rate is low, we accept the lost slots and move on. If high, a Step 0.5 design choice opens up: tap intra-edge hidden states by either (b) adding selective mass=1 KV cache writes when those positions are someone's backoff target, or (c) rebuilding the trie without compressing across positions any K backs off to. Both are kernel-touching but doable. None of this is in scope for the initial Step 0 implementation.

### Other observations

- **RoPE position-of-record within K_back's path forward.** Within the depth-batched forward, K_back_i's position-`j` uses RoPE at position `j` — same as a normal path position uses its own depth. Only at the *endpoint-time gather* into K's K/V stack does the sentinel-position swap happen, because that's where K's attention sees the backoff slot. Within K_back's own walk, RoPE-at-own-depth keeps the forward semantics standard.

- **Within-fire dedup (already free in the anc-grad scheme).** Multiple primary queries can share a K_back. K_back's forward runs **once** (as its own primary query) and its `ln1_out` is written once into `h_subtree[l][K_back.compact_slot]` by the existing `launch_save_ln1_to_subtree` kernel. In K's gather, each consumer reads from the same compact-char slot. No duplicated forward work; no explicit dedup needed.

- **Schema gate.** `experimental.backoff_slots: B` (with `B=0` disabled, recovering current AGPT exactly). Trainer-side wired through v1's `apply_yaml_config_v1` as a recognized experimental key.

### Precomputed sidecar: `agpt_build_backoff_table`

New tool (`src/tools/agpt_build_backoff_table.cr`, mirrors the existing radix-build tooling) that emits a per-node sidecar of `B` backoff-target radix IDs:

- **Input**: built radix trie dir + `B`.
- **Output**: `<trie-dir>/backoff_B<N>.bin` — header (magic, version, n_nodes, B, trie_corpus_hash, case2_count) followed by a flat `uint32` array of shape `(num_radix_nodes, B)`. Entry `[k, i]` is `K_back_i.id` for node `k` if it exists as a radix endpoint at depth `d-i`, or `UINT32_MAX` (case-2 sentinel) if not.
- **Caching**: content-hash against `(trie-dir, B)`; stored as `<trie-dir>/backoff_B<N>.bin`.
- **Loading**: trainer reads at startup, pins in host memory, builds `rev_lookup[M.id] → list of (K.id, i)` once per fire.

#### Algorithm choice: single-tree (Aho-Corasick suffix links) for Step 0

Two implementations produce the same sidecar:

- **Single-tree**: build suffix links in the prefix trie. Classical Aho-Corasick setup phase. One BFS over the prefix trie computing `suffix_link[K] = the radix node whose path is K's path with first char dropped`. Then materialize: for each K, follow suffix links B times. ~80 lines of Crystal. Self-contained; no extra prerequisite tooling.

- **Dual-tree**: build a suffix radix trie via `bin/agpt_build_radix_corpus --reverse`, build `SubstringCatalog` + `RadixToSubstring` maps via `bin/agpt_build_position_table`. Use the substring catalog as the 1-to-1 pairing between forward and reversed node IDs; K's backoffs are then σ's ancestors in the suffix tree, mapped back through the catalog. Conceptually elegant; reuses infrastructure that exists for other reasons. Adds prerequisite build steps for callers who haven't built those artifacts already.

**Chosen for Step 0: single-tree**. Smaller blast radius, no extra prerequisite tooling, doesn't depend on artifacts that may or may not be present for a given experiment. Can be swapped to dual-tree later if `SubstringCatalog` becomes the canonical infrastructure for similar tools.

The sidecar's binary format is identical either way, so the kernel doesn't care which built it.

### Order of implementation

(Reflects the anc-grad-piggyback design. Earlier drafts had a multi-step pass 1 / pass 2 / stash / stash_grad / deep-to-shallow scheme; that was retracted on 2026-05-31 once we recognized backoff slots are mechanistically identical to ancestor K/V slots and ride on `--anc-grad`'s closed-form path.)

1. ✓ **`bin/agpt_build_backoff_table`** — Crystal sidecar tool (single-tree Aho-Corasick). Already on the branch (`b2e042e`).
2. ✓ **YAML gate + sidecar load + rev_lookup** — host-side. Already on the branch (`55cbe30`).
3. **K_back → compact-slot resolution** (host-side, once per fire). For each sidecar entry where `K_back_id != SENTINEL`, walk to M's endpoint character position via `trie.edge_starts/edge_lens`, look up `compact_slot[char_pos]`, and store into a per-K array. Upload to device. Marked -1 for either sentinel sidecar entries or mass=1 endpoint positions (those don't have compact-cache slots and can't be reached by the anc-grad path).
4. **Forward gather extension** (small CUDA kernel). At K's primary endpoint, after the existing path-ancestors + own-edge gather populates `d_kv_pack_k/v`, append B backoff K/V slots: read `h_subtree[l][K_back_compact_slot]`, project through `W_k^l`/`W_v^l` + biases, apply RoPE at the chosen position (`d` for same-as-k, `d + i + 1` for sentinel; V has no RoPE). Update per-query `kv_lengths_per_q`. ~40 lines of CUDA.
5. **Attention kernel extension**. `cuda_batched_varlen_attention_L_queries` (declaration in agpt_train.cu, definition in kernels.cu) gains an optional `const int* kv_lengths_per_q`. When non-null, per-query loop length comes from that array instead of per-node `kv_lengths`. Null preserves bit-exact B=0 parity.
6. **Backward scatter extension** (small CUDA kernel). After attention backward produces `d_dk_pack` and `d_dv_pack`, the existing anc-grad ancestor-scatter (line 2573-ish) gets a sibling kernel that walks each query's backoff slots and atomic-adds the gradient at the backoff slot to `d_dkv_subtree_k[l]` / `d_dkv_subtree_v[l]` at M's compact-cache slot. K-side first applies inverse RoPE at the position used during the forward gather (mirror of the existing ancestor inverse-RoPE). ~30 lines of CUDA.
7. **Fire-end reduction**: no change. The existing matmul `dW_k^l += d_dkv_subtree_k[l]^T · h_subtree[l]` already sums anc + backoff contributions because they share the accumulator.
8. **Backward parity check** — with `experimental.backoff_slots: 0`, forward AND backward must be bit-exact vs the baseline build. Non-negotiable before any `B>0` runs.
9. **YAML gate + smoke** — already in place via step 2 (sidecar load); add the runtime gate that requires `--anc-grad` to be enabled (the default since 2026-05-20); `B=0` produces identical results to baseline; `B=4` runs end-to-end and produces a checkpoint.

## Experimental setup

- Canonical Shakespeare d=16 baseline. Carved at `data/.splits/2b7ded401e96b610/`.
- Init: `data/input.model` (d_model=64, n_layers=2, n_heads=4, d_ff=256).
- Canonical training: `--mass-weight linear`, `--fire-norm-mass` default-on, **`--partition-depth 0`** (per decision 8; single fire per epoch, K and K_back always in the same fire). Note this means one optimizer step per epoch, which is a slow learning regime — Step 0 is a feasibility check, not a high-throughput training run.
- 25 epochs to start (enough to see clear separation from baseline; matches cap-recurrence comparison anchors).
- Multiple shuffle seeds for noise control. 3 pairs minimum, 5+ if results are noisy.
- Eval: canonical `byte_perplexity` via `bin/agpt_experiment` + canonical heldout.
- Conditions (3 per shuffle seed):
  - **`B=0`** — baseline, current AGPT (no backoff).
  - **`B=4, position=sentinel`** — backoff slots at RoPE position `d+i`.
  - **`B=4, position=same-as-K`** — backoff slots at RoPE position `d` (relative rotation 0 against K's query; the present-moment-alternative-prediction reading).
- Schema gate: `experimental.backoff_slots: B` and `experimental.backoff_position: sentinel|same-as-k` (default `same-as-k`, the principled choice).

## Success criteria

- **Strong success**: either `B=4` condition's `byte_PPL` ≤ KN's ~4 on Shakespeare. The architecture beats KN.
- **Soft success**: at least one `B=4` condition's `byte_PPL` better than baseline but worse than KN. Mechanism works; tuning/scale needed to close the KN gap. The better-performing position-encoding becomes the canonical choice going forward.
- **Null**: both `B=4` conditions ≈ baseline. Mechanism didn't take. Diagnostics: instrument `‖attention-weight-mass-on-backoff-slots‖` to see whether the model uses the new slots at all; check gradient magnitudes on K_back's params to confirm the new gradient signal is flowing.
- **Hurt**: both `B=4` conditions > baseline. Implementation bug or fundamental architectural problem; debug before drawing conclusions. If sentinel hurts but same-as-K helps (or vice versa), the position encoding was the issue — informative either way.
- **Split**: sentinel helps and same-as-K hurts (or vice versa). The position-encoding A/B has done its job — go with the winner, document the failure mode of the loser.

## Risks and mitigations

- **Compute**: backoff slots ride on the existing `--anc-grad` infrastructure (closed-form `dW_k += dK^T · h_subtree`, one matmul per layer at fire end — same path anc-grad uses for ancestor K/V gradients). Forward gather appends B extra K/V positions per primary endpoint query (~5-10% extra attention bandwidth). Backward scatter adds B entries per primary endpoint into the same `d_dkv_subtree_*` accumulators anc-grad already uses. No two-pass, no extra forward work. Total overhead ~5-10%, not 2× or 4×. (Earlier drafts had a 2× two-pass design and a 4× satellite design; both were unnecessary once we recognized the equivalence to anc-grad's closed-form path.)
- **Memory**: zero new buffers. Reuses anc-grad's existing `h_subtree[l]` (forward stash for h_M^{l-1}) and `d_dkv_subtree_k/v[l]` (backward gradient accumulator). Only new per-fire host-side data is a small `M.id → compact-char slot` lookup, built once at fire start.
- **Reverse lookup table**: `M.id → list of (K.id, i)`. Built once per fire from the sidecar. At most `num_radix × B` entries; ~50 MB for Shakespeare d=16/B=4.
- **Case-2 rate**: we drop backoff slots whose target lands mid-edge of a compressed node (mass=1 position not in the K/V cache). **Measured on Shakespeare d=16/B=4 (2026-05-31): 65% of slots are case-2** — consistent with the trie's 88.9% mass=1 char rate; most case-2 misses are end-cap unary chains and almost certainly don't carry generalizable backoff signal. **Step 0 proceeds with these slots dropped** (per design decision logged 2026-05-31). If Step 0 shows null with just the 34% case-1 slots filled, case-2 expansion (Step 0.5) becomes worthwhile; if Step 0 succeeds at this density, case-2 is upside left on the table.
- **Training instability**: backoff `K_back` parameters now receive richer gradient signal (endpoint + upstream). May destabilize early training. Mitigations: gradient clipping (`--grad-clip-norm 1.0`), warmup-cosine LR (already canonical).
- **Trie sparsity / case-1 vs case-2**: every backoff *substring* exists in the corpus by sliding-window construction, but the radix node corresponding to `c_{i+1}..c_d` may not exist as a stored entity (it lands mid-edge of a compressed multi-token node). The sidecar marks those entries `UINT32_MAX` (the case-2 sentinel); the gather code drops the corresponding K/V slot. Log the case-2 rate.
- **RoPE position semantics**: sentinel `d+i` vs same-as-K `d` is the A/B in Step 0 (decision 6). Further variants (learnable per-level, zero RoPE, per-head specialization) remain Phase 1.5 follow-ups once Step 0 picks a winner.
- **pd=1 generalization**: Step 0 runs at pd=0 (decision 8) where K and K_back_i are in the same subtree fire, so `h_subtree[l][M's slot]` is populated when K's chunk reads it. At pd=1 the trie is partitioned into 65 root-child subtrees and K_back_i lives in a *different* subtree fire — `h_subtree[l]` is scoped per-subtree-fire and doesn't carry M's hidden state across the boundary. The pd=1 case would need either a fire-spanning stash buffer (regressing toward the two-pass design we rejected) or satellite forward of K_back_i in K's fire. Deliberate Step-N+ concern, not Step 0.
- **Backward-pass parity check**: when `experimental.backoff_slots: 0`, the kernel must produce identical forward AND backward results to the baseline (no-backoff) build. This is a non-negotiable regression check before any `B>0` runs.

## After Step 0

- **If strong/soft success**: Phase 1.5 — add landmark slots (root, high-mass shallow hubs); try further position-encoding variants (learnable per-level, zero RoPE, per-head); within-fire K_back path dedup; pd=1 generalization. Then Phase 2 (learnable routing).
- **If null on both position encodings**: diagnose the gradient flow and attention-mass instrumentation before declaring the mechanism doesn't work. Check that `K_back`'s parameters are actually receiving the new gradient signal. Consider B=2 to rule out training-instability-from-richer-signal as a confound.
- **If hurt on both**: either an implementation bug (the B=0 backward-parity check should catch most of these) or a fundamental issue with in-flight gradient flow at this scale. Try smaller models / shorter epochs to localize.
- **If split** (one encoding helps, the other hurts): adopt the winner. The split itself is information about what RoPE positions are doing for the model.

## Connection to prior work

- **Cap-recurrence** (closed, `project-cap-recurrence-null`): the null was about centroid aggregation + detached gradient + single slot. Step 0 is none of those (per-instance slots, gradient flow via anc-grad's closed-form path, multiple slots) — the closure doesn't apply.
- **Existing path-ancestor attention**: Step 0 is a strict superset; `B=0` recovers current AGPT exactly.
- **`--anc-grad` (descendant→ancestor scatter)**: backoff slots are **mechanistically identical** to ancestor K/V slots — both are `W_k^l · LN1^l(h_M^{l-1})` where M is some radix node and the descendant attends to it. Step 0 rides directly on anc-grad's infrastructure: same forward stash (`h_subtree[l]`), same gradient accumulator (`d_dkv_subtree_k/v[l]`), same fire-end matmul reduction. Same approximation too: M's pre-layer-l params don't receive the new gradient. See `todo/descendant-ancestor-scatter.md`.
- **KN baseline**: known floor at ~4 byte_PPL on Shakespeare. Step 0 target.

## Rejected design alternatives

- **Two-pass + deep-to-shallow chunk ordering + stash_grad buffer** (proposed by Claude on 2026-05-30 → 31). The idea: pass 1 forward-only populates a fire-global `d_stash[M_count, n_layers, D]` with `h_M^{l-1}`; pass 2 forward+backward in deep-first chunk order; backward on backoff slots accumulates into `d_stash_grad[M, l]`; M's chunk backward adds `d_stash_grad` to upstream at M's endpoint. Cost ~2× normal training, ~300 MB scratch, ~9 kernel-side steps including an attention-kernel extension for per-query `kv_lengths`. **Reject** — this re-invents anc-grad's closed-form path with additional bookkeeping. Backoff slots' gradient on W_k/W_v is `dK_back^T · LN1(h_M^{l-1})`; M's pre-layer-l params don't need to be touched (same approximation anc-grad already accepts). Riding on anc-grad's `h_subtree` + `d_dkv_subtree_*` infrastructure produces the same gradient contribution to W_k/W_v with no new buffers, no two-pass, no chunk reordering, ~5-10% overhead instead of 2×.

- **Satellite forward of K_back_i inside K's chunk** (the design before the two-pass). The idea: when K's chunk loads, also load K_back_i's path as additional chunk positions (skip-loss flag); K_back_i's forward runs in K's chunk's autograd graph; backward through K's attention flows into K_back_i's params during K's chunk's backward. Cost ~4-5× per-query compute. **Reject** — also unnecessary once anc-grad's closed-form applies. Satellites duplicate forward work the existing forward already does for K_back_i (as its own primary query in its own chunk).

- **Static KV cache + top-1 in-flight hybrid** (proposed in external review). The idea: B−1 backoff slots use globally cached `h_p` from previous super-epochs (detached gradient); 1 slot runs in-flight (gradient-connected). **Reject** — this recreates cap-recurrence's failure regime for B−1 of the B slots. Detached `h_p` cannot reshape its upstream representation; the cap-recurrence investigation (`project-cap-recurrence-null`) tested this directly across mass / inverse / random / rand-weights / constant / none aggregation modes at 5-ep and 25-ep, with both training loss and canonical byte_PPL. Null at every condition.

- **Stream compaction at every depth-step boundary** (proposed in external review as a warp-divergence fix). Reasonable kernel-level optimization for Step 0.5 if profiling shows warp divergence. Not needed for Step 0 correctness: the existing depth-sorted chunk structure clusters positions by their current active depth, so warps at a given depth step process homogeneous workloads. Stream compaction would replace the implicit clustering with explicit per-step compaction; cleaner conceptually but not free in implementation effort.
