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

### The cross-chunk problem (not solved by pd=0 alone)

A previous draft of this section framed the implementation as a single-pass "stash + gather + reverse lookup" at pd=0, claiming the satellite scheme from earlier was unnecessary because "at pd=0 every K_back_i is already being processed as a primary query somewhere in the fire."

The cross-chunk gap (not addressed in that draft): **at pd=0 the fire spans many chunks** (Shakespeare d=16 with default `chunk_queries=50000` and ~8.9M edge chars → ~180 chunks per fire). K and K_back_i generally land in *different* chunks. The "stash within chunk" framing only works for same-chunk K/M pairs; cross-chunk pairs would need either detached gradient (cap-recurrence regime — rejected) or a more elaborate cross-chunk mechanism.

Step 0 resolves this with a **two-pass per-fire scheme + deep-to-shallow chunk ordering**, which makes the new gradient signal correct regardless of which chunks K and K_back_i land in.

### The implementation: two-pass + deep-to-shallow ordering

The full design (see also the implementation plan in `.claude/plans/`):

**Pass 1 (forward-only, any chunk order)** — populate a fire-global stash buffer with h_M^l for every M that's a backoff target.

- Iterate every chunk once.
- Per chunk, run the full per-layer forward (LN1 → Q/K/V matmul → RoPE → cache scatter → gather → attention → WO/LN2/FFN).
- At each layer `l`, scatter `d_ln_out[endpoint_q, :]` (the post-LN1 hidden state of M's endpoint position) into `d_stash[M.stash_idx, l, :]`.
- Skip loss, backward, and the per-chunk save buffers that backward would need.
- Cost ≈ one full forward over the fire's queries.

After pass 1: `d_stash` is populated for every backoff target M, regardless of which chunk it was processed in.

**Pass 2 (full forward + backward, DEEP-TO-SHALLOW chunk order)** — normal training with two additions:

- **Gather extension**: at K's primary endpoint, append B extra K/V slots from `d_stash[K_back_i.stash_idx, l, :]`, projected through layer-l's `W_k_l`/`W_v_l` + RoPE at the chosen position (decision 6).
- **stash_grad accumulation**: K's attention backward writes the gradient on each backoff K/V slot back into a parallel `d_stash_grad[M.stash_idx, l, :]` buffer (after backprop through RoPE + W_k_l/W_v_l).

The KEY observation that makes this work without a pass 3: chunks of pass 2 are ordered **deep-to-shallow** so that all K's (the "consumers" of backoff slots) are processed before their K_back_i targets (the "producers"). At chunk-load time, `radix_list` is sorted by `endpoint_depth` descending (ties broken by `radix_id`). Since K_back_i.depth = K.depth − i < K.depth, M is guaranteed to land in a later chunk after this sort.

When M's chunk's backward runs in pass 2, every K that backed off to M has already written to `d_stash_grad[M, l]`. M's chunk's backward at M's endpoint then incorporates `d_stash_grad[M, l]` as **additional upstream gradient** at M's `d_ln_out[endpoint_q]` — in addition to M's own endpoint-loss gradient. The combined gradient propagates back through M's normal backward into d_grads.

Result: M's params receive `grad(M's own endpoint loss) + grad(K's loss through K's attention through stash[K, i])`. Two-signal stacking, in-flight gradient flow, chunk-membership independent.

Same-chunk K and M is handled identically — K and M live in disjoint stash and stash_grad slots, so within-chunk write/read ordering doesn't matter.

### Buffer sizing (Shakespeare d=16/B=4 / d_model=64 / L=2)

- `d_stash[M_count, n_layers, D]` bf16. `M_count` = nodes that appear in rev_lookup (376k for Shakespeare). 376k × 2 × 64 × 2 = **~96 MB**.
- `d_stash_grad[M_count, n_layers, D]` fp32. 376k × 2 × 64 × 4 = **~192 MB**.
- `d_M_to_stash[radix_count]` int32. ~6 MB.

Total new per-fire memory ≈ **300 MB**. Fits comfortably on 8 GB GPU alongside existing per-fire scratch.

### Compute cost

**Pass 1 ≈ 1× forward (no backward). Pass 2 ≈ normal forward + backward. Total ≈ 2× normal training.**

(An earlier draft said "few percent overhead." That was wrong — it ignored the cross-chunk gap. Correcting: ~2× is the price for in-flight gradient flow across chunks.)

For Shakespeare d=16 the existing 25-ep run is ~10 min; this becomes ~20 min. Acceptable for the Step 0 feasibility check.

### Backward — gradient flow

h_M (read from stash during K's gather) has two downstream consumers in the gradient graph:

- M's own endpoint loss (existing): "predict the next char after M from h_M" — flows during pass 2's processing of M's chunk.
- K's attention via the stash slot (new): h_M used as a K/V slot in K's depth-(d−1) step — flows during pass 2's processing of K's chunk, accumulating into `d_stash_grad[M, l]`.

The deep-to-shallow ordering guarantees stash_grad is fully populated for M before M's chunk's backward runs. The "add stash_grad to LN1's upstream at M's endpoint" hook combines the two gradient signals, and the rest of the backward proceeds as normal.

This is the in-flight gradient flow that distinguishes Step 0 from the cap-recurrence regime (where the detached/cached h_M produced no signal). Cap-recurrence's null is direct evidence the detached version collapses to a soft-KN information ceiling.

### What's actually new vs the existing kernel

In rough order (covered in more detail by the implementation plan):

1. **Sidecar load + rev_lookup + M_to_stash** (host-side, once per fire). Read `<trie-dir>/backoff_B<N>.bin`, build the inverse `M.id → (stash_idx, list of (K.id, i))`. Upload to device.
2. **Stash + stash_grad allocation** (fire-global). Shape `(M_count, n_layers, D)` each. ~300 MB for Shakespeare d=16/B=4/L=2.
3. **Chunk reordering** (pd=0 only when backoff_slots>0). Sort `radix_list` by `endpoint_depth` descending, breaking ties by `radix_id`, before chunking. Pass 2 will iterate chunks in this order.
4. **Pass-1 stash-write hook** in the per-layer forward (after LN1's `d_ln_out` is produced, before Q/K/V matmuls): scatter `d_ln_out[endpoint_q]` into `d_stash[M.stash_idx, l]`.
5. **Pass-2 gather extension** at the existing gather (line ~5969): in addition to ancestors + own-edge, append B backoff K/V slots per primary endpoint query (read from `d_stash`, project through `W_k_l`/`W_v_l`, apply RoPE at chosen position, append). Update per-query `kv_lengths_per_q`.
6. **Pass-2 backward stash_grad write**: after the attention backward, route the gradient on backoff K/V slots back through RoPE + W_k_l/W_v_l projection into `d_stash_grad[M, l]`.
7. **Pass-2 backward stash_grad incorporation at M's endpoint**: before LN1 backward consumes the `d_ln_out` gradient at M's endpoint position, add `d_stash_grad[M, l]` to it.
8. **Attention kernel extension**: `cuda_batched_varlen_attention_L_queries` takes an optional `kv_lengths_per_q`. Null preserves bit-exact B=0 parity.
9. **Sentinel handling**: sidecar entries equal to `UINT32_MAX` mean "no K_back" — gather code skips them. K's attention runs over `d + B'` slots where `B'` is the count of non-sentinel backoffs.

### Case 2 (mid-edge backoff targets): dropped for Step 0

Some K_back_i targets land mid-edge of a compressed radix node N rather than at a node endpoint. The K/V cache in v1 only stores entries for branching (mass>1) positions — mass=1 positions (compressed-edge interiors) are NOT in the cache, by design (see `agpt_train.cu:2326-2337` and the `compact_slot` mechanism). For Step 0 these "case 2" backoff slots are marked SENTINEL in the sidecar and dropped at gather time. We measure the case-2 rate during sidecar construction.

If the empirical case-2 rate is low, we accept the lost slots and move on. If high, a Step 0.5 design choice opens up: tap intra-edge hidden states by either (b) adding selective mass=1 KV cache writes when those positions are someone's backoff target, or (c) rebuilding the trie without compressing across positions any K backs off to. Both are kernel-touching but doable. None of this is in scope for the initial Step 0 implementation.

### Other observations

- **RoPE position-of-record within K_back's path forward.** Within the depth-batched forward, K_back_i's position-`j` uses RoPE at position `j` — same as a normal path position uses its own depth. Only at the *endpoint-time gather* into K's K/V stack does the sentinel-position swap happen, because that's where K's attention sees the backoff slot. Within K_back's own walk, RoPE-at-own-depth keeps the forward semantics standard.

- **Within-fire dedup (already free in the two-pass scheme).** Multiple primary queries can share a K_back. In pass 1, K_back's forward runs **once** (as its own primary query) and `d_ln_out` is scattered once into `d_stash[K_back.stash_idx, l]`. In pass 2's gather, each consumer K reads the same stashed value. No duplicated forward work; no explicit dedup needed.

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

1. **`agpt_build_backoff_table`** — Crystal tool, mirrors `agpt_build_radix_corpus`. Standalone; verifiable independently. Output: a `uint32` table you can dump and spot-check against the radix trie. Implementation approach: **single-tree Aho-Corasick suffix-link construction** on the prefix trie (see "Sidecar tool" below for rationale on this choice vs the dual-tree alternative).
2. **rev_lookup construction at startup** — invert the sidecar to produce `M.id → list of (K.id, i)`. One pass over the sidecar. Stays resident for the duration of the trainer.
3. **Stash buffer allocation** — per-chunk scratch of shape `(num_primary_queries_in_chunk, B, n_layers, d_model)`.
4. **Stash-write hook** at the endpoint-depth-step kernel: after M's h_M is computed, scatter into `stash[K, i]` for each `(K, i) ∈ rev_lookup[M.id]`. Standard scatter pattern.
5. **Gather + projection + RoPE hook** at K's endpoint depth-step: read `stash[K, 0..B-1]`, project through `W_k`/`W_v`, apply RoPE at the chosen position (`d+i` or `d` per the A/B), append to K's K/V stack. Skip slots with sentinel entries.
6. **Backward parity check** — with `experimental.backoff_slots: 0`, forward AND backward must be bit-exact vs the baseline build. Non-negotiable before any `B>0` runs.
7. **YAML gate + smoke** — `experimental.backoff_slots: B` and `experimental.backoff_position: sentinel|same-as-k` recognized in v1's `apply_yaml_config_v1`; `B=0` produces identical results to baseline; `B=4` runs end-to-end and produces a checkpoint.

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

- **Compute**: ~2× normal training (pass 1 ≈ forward, pass 2 ≈ forward + backward). The earlier "few percent overhead" framing missed the cross-chunk gap — see the "two-pass + deep-to-shallow ordering" section above. For Shakespeare d=16, 25-ep runs go from ~10 min to ~20 min.
- **Stash buffer memory**: per-chunk scratch of shape `(num_primary_queries_in_chunk, B, n_layers, d_model)`. For Shakespeare d=16 / d_model=64 / L=2 at the default `chunk_queries=50000`: 50000 × 4 × 2 × 64 × 4 bytes ≈ 100 MB per chunk. Real but well under the existing KV-cache footprint. Larger configs scale linearly; lower `chunk_queries` directly reduces this.
- **Reverse lookup table**: `M.id → list of (K.id, i)`. Built once per fire from the sidecar. At most `num_radix × B` entries; ~50 MB for Shakespeare d=16/B=4.
- **Case-2 rate**: we drop backoff slots whose target lands mid-edge of a compressed node (mass=1 position not in the K/V cache). **Measured on Shakespeare d=16/B=4 (2026-05-31): 65% of slots are case-2** — consistent with the trie's 88.9% mass=1 char rate; most case-2 misses are end-cap unary chains and almost certainly don't carry generalizable backoff signal. **Step 0 proceeds with these slots dropped** (per design decision logged 2026-05-31). If Step 0 shows null with just the 34% case-1 slots filled, case-2 expansion (Step 0.5) becomes worthwhile; if Step 0 succeeds at this density, case-2 is upside left on the table.
- **Training instability**: backoff `K_back` parameters now receive richer gradient signal (endpoint + upstream). May destabilize early training. Mitigations: gradient clipping (`--grad-clip-norm 1.0`), warmup-cosine LR (already canonical).
- **Trie sparsity / case-1 vs case-2**: every backoff *substring* exists in the corpus by sliding-window construction, but the radix node corresponding to `c_{i+1}..c_d` may not exist as a stored entity (it lands mid-edge of a compressed multi-token node). The sidecar marks those entries `UINT32_MAX` (the case-2 sentinel); the gather code drops the corresponding K/V slot. Log the case-2 rate.
- **RoPE position semantics**: sentinel `d+i` vs same-as-K `d` is the A/B in Step 0 (decision 6). Further variants (learnable per-level, zero RoPE, per-head specialization) remain Phase 1.5 follow-ups once Step 0 picks a winner.
- **pd=1 generalization**: Step 0 runs at pd=0 (decision 8) where K and K_back_i are always in the same fire. The pd=1 case is genuinely harder — it requires either importing K_back chunk data across subtree fires, or accepting detached gradient on the cross-subtree K_back (which collapses to the cap-recurrence regime). That's a deliberate Step-N+ concern, not a Step 0 risk.
- **Backward-pass parity check**: when `experimental.backoff_slots: 0`, the kernel must produce identical forward AND backward results to the baseline (no-backoff) build. This is a non-negotiable regression check before any `B>0` runs.

## After Step 0

- **If strong/soft success**: Phase 1.5 — add landmark slots (root, high-mass shallow hubs); try further position-encoding variants (learnable per-level, zero RoPE, per-head); within-fire K_back path dedup; pd=1 generalization. Then Phase 2 (learnable routing).
- **If null on both position encodings**: diagnose the gradient flow and attention-mass instrumentation before declaring the mechanism doesn't work. Check that `K_back`'s parameters are actually receiving the new gradient signal. Consider B=2 to rule out training-instability-from-richer-signal as a confound.
- **If hurt on both**: either an implementation bug (the B=0 backward-parity check should catch most of these) or a fundamental issue with in-flight gradient flow at this scale. Try smaller models / shorter epochs to localize.
- **If split** (one encoding helps, the other hurts): adopt the winner. The split itself is information about what RoPE positions are doing for the model.

## Connection to prior work

- **Cap-recurrence** (closed, `project-cap-recurrence-null`): the null was about centroid aggregation + detached gradient + single slot. Step 0 is none of those (per-instance slots, in-flight gradient, multiple slots) — the closure doesn't apply.
- **Existing path-ancestor attention**: Step 0 is a strict superset; `B=0` recovers current AGPT exactly.
- **KN baseline**: known floor at ~4 byte_PPL on Shakespeare. Step 0 target.

## Rejected design alternatives

- **Static KV cache + top-1 in-flight hybrid** (proposed in external review). The idea: B−1 backoff slots use globally cached `h_p` from previous super-epochs (detached gradient); 1 slot runs in-flight (gradient-connected). Compute drops to ~1.5× baseline. **Reject** — this recreates cap-recurrence's failure regime for B−1 of the B slots. Detached `h_p` cannot reshape its upstream representation; the cap-recurrence investigation (`project-cap-recurrence-null`) tested this directly across mass / inverse / random / rand-weights / constant / none aggregation modes at 5-ep and 25-ep, with both training loss and canonical byte_PPL. Null at every condition. Additionally, cached KV from prior super-epochs is *stale* (KN's coefficients are stationary; cached KV is not). The right ways to save compute without touching the mechanism are: lower B (e.g., B=2, still ~2.5× compute, full gradient flow on all B slots), within-fire dedup of shared K_back paths (Step 0.5 optimization), or smaller `chunk_queries` (reduces stash buffer linearly).

- **Stream compaction at every depth-step boundary** (proposed in external review as a warp-divergence fix). Reasonable kernel-level optimization for Step 0.5 if profiling shows warp divergence. Not needed for Step 0 correctness: the existing depth-sorted chunk structure clusters positions by their current active depth, so warps at a given depth step process homogeneous workloads. Stream compaction would replace the implicit clustering with explicit per-step compaction; cleaner conceptually but not free in implementation effort.
