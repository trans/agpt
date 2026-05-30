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
   - **(b) Depth-relative `d−i`** (K_back_i's own endpoint depth): preserves RoPE's shift-invariance property. K_back_i is a *suffix-shifted view* of the same temporal context, and `d−i` is its true semantic offset relative to K's query. Two slots end up at the same RoPE rotation (K's path-ancestor at depth `d−i` and the backoff cousin); they are disambiguated by content. This is effectively a soft mixture of "what your path knows at this depth" and "what the suffix-shifted cousin would say at the equivalent depth" — mechanistically the KN interpolation with attention learning the mixing weights.

   Both are zero-parameter changes. Sentinel `d+i` is the simpler implementation, depth-relative `d−i` is the theoretically cleaner mapping. Empirically settling which wins is cheap (one extra pair of runs) and is the kind of question that's much faster to answer here than as a "Phase 1.5 follow-up." Step 0 runs both.

   Other position-encoding variants — learnable per-level embeddings, zero RoPE, per-head specialization — remain Phase 1.5 follow-ups once we know whether sentinel or depth-relative is the better starting point.

7. **No within-fire dedup of `K_back` paths**: queries in the same chunk that share a `K_back` independently include that `K_back`'s path as a parallel mini-path inside the depth-batched forward (see below). Cost is bounded (~4–5× compute per query); within-fire dedup is a Step-0.5 optimization.

8. **Start at `partition_depth: 0`** (single fire per epoch over the whole trie). At pd=1 the trie is sharded into 65 root-children that fire separately, and K_back_i lives in a *different* shard from K — bringing its forward into K's fire means importing chunk data across shards. At pd=0 there is only one fire, K and every K_back_i are already part of the same training unit, and the cross-subtree concern dissolves into a much smaller within-fire chunk-membership concern (do K and K_back_i fall in the same `chunk_queries`-sized chunk?). Step 0 runs at pd=0. Once we know the architecture works, the pd=1 generalization is a separate (harder) implementation step that crosses subtree fires.

## Implementation sketch

### How AGPT v1's fire works today

A "fire" in v1 = one per-subtree forward + backward + optimizer step. At `--partition-depth 1` the trie is partitioned by root char (65 root-children for Shakespeare ASCII), and one fire processes one root-child's subtree. **At pd=0 there is only one fire per epoch over the whole trie** — the trainer chunks for memory (via `--chunk-queries`, default 50000) but otherwise treats the entire trie as a single training unit. Step 0 runs at pd=0 (decision 8) — every K and every K_back_i are members of the same fire by construction, no cross-subtree imports.

Within a fire, the kernel **does not walk paths serially**. Chunks of positions are sorted by depth, and at each depth `j` the kernel processes *all chunk positions whose endpoint depth is ≥ `j`* in parallel as one batched layer step. Positions whose endpoint depth equals `j` consume their loss target at that depth step and **drop out of the batch**; deeper steps process only the still-growing paths. The active batch naturally shrinks from depth 0 to `max_depth`.

So the kernel already knows how to handle "positions with different endpoint depths sharing one chunk." Adding backoff slots reuses exactly this machinery — we don't add a new masking concept.

### The implementation: satellite positions with null loss + stash buffer

Each "primary query" K in the chunk (a real radix endpoint at depth d) gets `B` satellite positions added to the chunk:

- `K_back_1`'s path (endpoint at depth `d−1`)
- `K_back_2`'s path (endpoint at depth `d−2`)
- ...
- `K_back_B`'s path (endpoint at depth `d−B`)

Two ways these satellites differ from a normal chunk position:

1. **No loss target.** A normal position contributes a cross-entropy loss at its endpoint depth. Satellites do not — K_back_i's own loss is computed during a different processing of K_back_i (when its own row in the chunk reaches its endpoint as a primary query). For the satellite, the loss kernel is skipped.

2. **Hidden state stashed into a backoff slot buffer.** When the satellite hits its endpoint depth, its final hidden state is written to a per-fire scratch buffer indexed by `(primary_query_id, backoff_level i)`. Later, when the primary query K hits its endpoint depth `d−1`, it reads back those `B` stashed values to assemble its K/V stack.

Everything else — the depth-by-depth batch shrinking, the layer ops, the existing drop-out-at-endpoint logic — is unchanged. The satellite is just "a normal chunk position with two extra bookkeeping flags: skip-loss, stash-output." The "masking" framing in the previous draft was overcomplicated; the existing endpoint-drop-out handles "this position only runs to depth d−i" for free.

### At K's endpoint: gather + sentinel-position projection

When the primary query K finishes at depth `d−1`, K's attention K/V stack is assembled from:

- The `d` path-ancestor K/V values (existing — produced by K's own path forward at depths 0..d−1).
- The `B` stashed `h_p[K_back_i]` values from the scratch buffer.

For each stashed backoff value:
- Project it through the shared `W_k` / `W_v` (no new parameters).
- Apply RoPE at the sentinel position `d + i` (NOT at K_back_i's own depth).
- Slot it into K's K/V stack after the `d` path slots.

K's attention runs as normal over `d + B = 20` slots.

### Per-fire compute and memory

Per primary query, the fire adds `Σ_{i=1..B} (d − i) = Bd − B(B+1)/2` satellite-position layer steps. For `d=16, B=4`: 16 path positions + 54 satellite positions = 70 total vs baseline 16. Roughly **4.4× per-query compute**.

Stash buffer memory: `(num_queries_in_chunk, B, n_layers, d_model)` float scratch. For a Shakespeare chunk of 50000 queries × 4 × 2 × 64 × 4 bytes ≈ 100 MB per fire. Real but well under the existing KV-cache footprint.

The depth-batched shape *doesn't* multiply naive memory by 4.4×: satellites drop out at their own endpoints, so the chunk's *peak* batch size happens at depth 0 (where everything is alive) and shrinks monotonically afterwards. Peak per-chunk position count ≈ `(1+B) × num_queries` = 5× at depth 0, then dropping. Average is ~3×. Not architecture-breaking.

### Backward

Free via autodiff. Every satellite position was part of the forward graph by construction; its gradient flows back to K_back_i's parameters along the same path-forward edges that the path-ancestor positions use. Two gradient signals per epoch stack on `K_back`'s params:

- The existing endpoint-predictor signal from K_back_i's own loss target (when K_back_i appears in the chunk as a primary query).
- The new "be a useful upstream representation" signal from every K whose chunk includes K_back_i as a satellite.

This is the gradient flow that distinguishes Step 0 from the cached/detached approach. Cap-recurrence's null is direct evidence the detached version collapses to a soft-KN information ceiling. The in-flight forward is the price we pay (4.4× compute) for the new gradient signal.

### What's actually new vs the existing kernel

In the order needed by the kernel:

1. **Chunk extension.** At chunk-load time, after gathering primary queries, expand the chunk by appending B satellite positions per primary query. Satellite chunk entries carry: path chars, endpoint depth (= primary depth − backoff level), a `skip_loss` flag, and a `stash_slot` index `(primary_query_id, backoff_level)`.

2. **Skip-loss handling.** At depth-step `j`, the loss kernel currently fires for any chunk position whose endpoint depth equals `j`. Add a check: if the position has `skip_loss=true`, write its final hidden state to the stash buffer at `stash_slot` and skip the cross-entropy.

3. **Endpoint-time K/V gather.** At a primary query's endpoint depth, before its existing attention math runs, gather the `B` stashed values from the scratch buffer, project through `W_k`/`W_v`, apply RoPE at sentinel positions `d+i`, and append to the K/V stack.

4. **Stash buffer allocation.** Allocate one per-fire scratch buffer of shape `(num_queries_in_chunk, B, n_layers, d_model)` at fire setup.

5. **Sidecar lookup.** At chunk-load time, identify K_back_i for each primary query via the `agpt_build_backoff_table` sidecar (below). When a sidecar entry is the sentinel value (suffix not in trie), mark that satellite as inactive — no chunk position added, the K/V gather produces no contribution for that slot, and we log the skip rate as a health metric.

### Other observations

- **RoPE position-of-record within K_back's path forward.** Within the depth-batched forward, K_back_i's position-`j` uses RoPE at position `j` — same as a normal path position uses its own depth. Only at the *endpoint-time gather* into K's K/V stack does the sentinel-position swap happen, because that's where K's attention sees the backoff slot. Within K_back's own walk, RoPE-at-own-depth keeps the forward semantics standard.

- **Within-fire dedup.** Multiple primary queries can share a K_back. Step 0 does not dedup: each shared K_back's path forward runs once per query that names it. Step 0.5 optimization: dedup the satellite paths at chunk-load time and broadcast the result. This is straightforward but adds bookkeeping; not worth doing until Step 0 results are in.

- **Schema gate.** `experimental.backoff_slots: B` (with `B=0` disabled, recovering current AGPT exactly). Trainer-side wired through v1's `apply_yaml_config_v1` as a recognized experimental key.

### Precomputed sidecar: `agpt_build_backoff_table`

New tool (`src/tools/agpt_build_backoff_table.cr`, mirrors the existing radix-build tooling) that iterates every radix node in a built trie and emits a per-node sidecar of `B` backoff-target IDs:

- Input: built radix trie dir + `B`.
- Output: `<trie-dir>/backoff_B<N>.bin` — a flat `uint32` array of shape `(num_radix_nodes, B)`. Entry `[k, i]` is the radix-node ID of the trie node found by descending from root using `K[k].path_chars[i+1..d]`, or a sentinel value (e.g., `UINT32_MAX`) if that suffix is not a node in the trie.
- One-time precomputation: same caching idiom as the trie itself, can be content-hashed against `(trie-dir, B)` and stored at `data/.tries/<hash>/backoff_B<N>.bin`.
- Loaded at trainer startup, pinned in GPU memory, indexed once per query at chunk-loading time to populate `K_back_i.path_chars[*]` for each query in the chunk.

Implementation order: this sidecar comes first, because the fire-kernel widening depends on knowing K_back_i identities per query at chunk-load time. With the sidecar in hand, the kernel changes have a clean API: "for query at radix-node `k`, K_back IDs are `sidecar[k, 0..B-1]`."

### Order of implementation

1. **`agpt_build_backoff_table`** — Crystal tool, mirrors `agpt_build_radix_corpus`. Standalone; verifiable independently. Output is a `uint32` table you can dump and spot-check against the radix trie's structure.
2. **Chunk extension at load time** — sidecar-driven. For each primary query, append `B` satellite chunk entries marked `skip_loss=true` with a `stash_slot` index. Skip satellites where the sidecar entry is the sentinel value (suffix not in trie).
3. **Stash buffer allocation** — per-fire scratch of shape `(num_queries_in_chunk, B, n_layers, d_model)`.
4. **Skip-loss + stash-output in the loss kernel** — at endpoint-depth step, route satellite positions to "write hidden state to stash buffer" instead of "compute cross-entropy."
5. **Endpoint-time K/V gather** — at the primary query's endpoint depth, gather B stashed values, project through `W_k`/`W_v`, apply RoPE at sentinel positions `d+i`, append to K's K/V stack.
6. **Backward parity check** — with `experimental.backoff_slots: 0`, forward AND backward must be bit-exact vs the baseline build. Non-negotiable before any `B>0` runs.
7. **YAML gate + smoke** — `experimental.backoff_slots: B` recognized in v1's `apply_yaml_config_v1`; `B=0` produces identical results to baseline; `B=4` runs end-to-end and produces a checkpoint.

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
  - **`B=4, position=depth-relative`** — backoff slots at RoPE position `d−i` (K_back_i's own endpoint depth).
- Schema gate: `experimental.backoff_slots: B` and `experimental.backoff_position: sentinel|depth-relative` (default `sentinel`).

## Success criteria

- **Strong success**: either `B=4` condition's `byte_PPL` ≤ KN's ~4 on Shakespeare. The architecture beats KN.
- **Soft success**: at least one `B=4` condition's `byte_PPL` better than baseline but worse than KN. Mechanism works; tuning/scale needed to close the KN gap. The better-performing position-encoding becomes the canonical choice going forward.
- **Null**: both `B=4` conditions ≈ baseline. Mechanism didn't take. Diagnostics: instrument `‖attention-weight-mass-on-backoff-slots‖` to see whether the model uses the new slots at all; check gradient magnitudes on K_back's params to confirm the new gradient signal is flowing.
- **Hurt**: both `B=4` conditions > baseline. Implementation bug or fundamental architectural problem; debug before drawing conclusions. If sentinel hurts but depth-relative helps (or vice versa), the position encoding was the issue — informative either way.
- **Split**: sentinel helps and depth-relative hurts (or vice versa). The position-encoding A/B has done its job — go with the winner, document the failure mode of the loser.

## Risks and mitigations

- **Compute**: ~4.4× per-query forward+backward compute (`d` path positions + `Σ(d−i) = 54` satellite positions for d=16, B=4). Real but per-fire wall is still ~30s on the existing setup; 25-ep runs go from ~2.5 min to ~10 min. Acceptable for PoC. The satellite design reuses the kernel's existing depth-by-depth + endpoint-drop-out machinery — no new control-flow concept, just more positions per chunk.
- **Stash buffer memory**: per-fire scratch of shape `(num_queries_in_chunk, B, n_layers, d_model)`. For Shakespeare d=16 / d_model=64 / L=2 at the default `chunk_queries=50000`: 50000 × 4 × 2 × 64 × 4 bytes ≈ 100 MB per fire. Real but well under the existing KV-cache footprint. Larger configs scale linearly; lower `chunk_queries` directly reduces this.
- **Peak chunk batch size**: satellites added to the chunk multiply the *peak* batch by `(1+B)` at depth 0. The active set shrinks as satellites drop out at their endpoints, so average chunk batch is ~3× baseline, but the peak-at-depth-0 is ~5×. If this exceeds GPU memory headroom, lower `chunk_queries` before lowering `B`.
- **Training instability**: backoff `K_back` parameters now receive richer gradient signal (endpoint + upstream). May destabilize early training. Mitigations: gradient clipping (`--grad-clip-norm 1.0`), warmup-cosine LR (already canonical).
- **Trie sparsity**: deep `K_back` may not exist for some prefixes. The sidecar marks missing entries with a sentinel value; the chunk-load step skips those satellites — no chunk position added, the K/V gather produces no contribution for that slot. Log the skip rate as a health metric. If frequent, this signals a corpus-coverage issue more than an architecture issue.
- **RoPE position semantics**: sentinel vs depth-relative is now an A/B in Step 0 (decision 6). Further variants (learnable per-level, zero RoPE, per-head specialization) remain Phase 1.5 follow-ups once Step 0 picks a winner.
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
