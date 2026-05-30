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

6. **Position encoding for backoff slots**: **RoPE at sentinel position `d+i`** for the `i`-th backoff slot. This is the simplest scheme that distinguishes backoff slots from path slots positionally without adding parameters.

   Position encoding is expected to be a rich vein of variants — flagged for follow-up exploration once Step 0 results are in:
   - RoPE at `K_back`'s actual depth `d-i` (gives the model the "this is a position-(d-i) token" signal but introduces position-duplication with the path-ancestor at depth `d-i`).
   - Learnable backoff embeddings (one per backoff level; most expressive, adds parameters).
   - Zero RoPE (no rotation; rely on content + attention head selection alone).
   - Per-head specialization (some heads RoPE at sentinel, others at actual depth).

   For the first run we pick the obvious cheap option (sentinel) and see what gives.

7. **No within-fire dedup of `K_back` paths**: queries in the same chunk that share a `K_back` independently include that `K_back`'s path as a parallel mini-path inside the depth-batched forward (see below). Cost is bounded (~4–5× compute per query); within-fire dedup is a Step-0.5 optimization.

## Implementation sketch

### The correct mental model: depth-batched, not path-by-path

AGPT v1's per-fire kernel does **not** walk each query's path serially. Chunks are sorted by depth (lowest first), and at each depth `d` the kernel processes *all positions in the chunk at depth `d`* in parallel as one batched layer step. The inputs at depth `d` are the `h_{d-1}` values computed during the previous depth step plus the new char at depth `d`. The fire walks depth-by-depth, layer-by-layer, with batch dimension = "all positions at this depth."

So at the moment K reaches its final depth `d` in this fire, only K's own path ancestors have been forward-passed *within this fire's autograd graph*. K_back's hidden states are not present: K_back lives in a different root-child subtree (its own subtree fires separately during its own fire), and its training-time hidden state from those other fires is not part of K's autograd graph (even if cached, the gradient would be detached — the failure mode we ruled out in cap-recurrence).

### The implementation that falls out: widen the per-depth batches

To make `h_p[K_back_i]` part of K's autograd graph in-flight, we add `K_back_1..B`'s paths to K's fire as **parallel mini-paths** processed alongside K's path at the same depth tempo:

- At depth 0, the per-depth batch consists of K's path position-0 (1 entry per query) AND each `K_back_i`'s position-0 (B entries per query). All processed as one wider batch by the same layer ops.
- At depth 1, similarly: K's path position-1 + K_back_1..B's position-1. Inputs at this step are the corresponding `h_0` from the previous step.
- Continues until depth `d−i−1`: at this depth, `K_back_i`'s path finishes (it has only `d−i` chars). Its final hidden state is stashed into a "backoff slot output buffer" for K. K_back_i drops out of the active batch for subsequent depth steps.
- At depth `d−1`: K's own path finishes. Now the K/V stack for K's attention is assembled from:
  - The `d` path-ancestor K/V values (existing behavior).
  - The `B` stashed `h_p[K_back_i]` values from when each backoff finished.
- The stashed `h_p[K_back_i]` is projected to K/V using the shared `W_k`/`W_v` and applied to RoPE at the sentinel position `d+i` before being slotted into K's attention.
- Run final-depth attention as normal over `d + B` slots.

Per-query compute: `d` path positions + `Σ_{i=1..B} (d−i) = Bd − B(B+1)/2` backoff positions. For `d=16, B=4`: 16 + 54 = 70 vs baseline 16 ≈ 4.4× compute. Same ratio as the cap-recurrence-era estimate, but achieved through wider batches at each depth — not through `B` separate side-fires per query.

Backward: free via autodiff. Every K_back mini-path was part of the forward graph; gradient flows back to K_back's parameters along the same edges the path-ancestor forward uses. Two gradient signals per epoch stack on `K_back`'s params (its own fire's endpoint-predictor gradient, plus the new "be a useful upstream representation" gradient from every fire that backed off to it).

### What's actually hard

The above is conceptually clean but has three real engineering concerns at the kernel level:

1. **Per-position active-path masking.** Each mini-path has a different "max active depth": K's path is active 0..d−1; K_back_i is active 0..(d−i−1). Once a path finishes, its entries in the batch tensor must not be re-processed at deeper steps — masked-out positions still occupy slots in the batch but contribute zero to the layer ops. The simplest implementation maintains a per-batch-position `active[d]` mask updated each step. The kernel either skips inactive entries (control divergence on warp) or zero-pads and trusts the mask (memory waste). Plain control-flow masking is the cleanest start; vectorization can come later if it bottlenecks.

2. **Hidden-state stash for finished mini-paths.** When K_back_i finishes at depth `d−i−1`, its final hidden state needs to be saved until K's own attention step at depth `d−1`. Memory cost: per-query × B × n_layers × d_model. For typical fires with thousands of queries this is real but bounded — well under the existing KV-cache size. Allocate one per-fire scratch buffer of shape `(num_queries, B, n_layers, d_model)`; index into it by `(query, backoff_level)` at stash and gather time.

3. **K/V stack construction at K's final depth + sentinel-position RoPE projection.** K's existing attention currently gathers `d` slots from the path forward. New: also gather `B` slots from the stash buffer, project them through `W_k`/`W_v`, and apply RoPE at position `d+i` (not at K_back's own depth). Implementation is a separate gather kernel that runs once at K's depth-`d−1` step before the existing attention math. The shared-parameter choice means we reuse the existing `W_k`/`W_v` projection — only the position passed to the RoPE rotation differs.

### Other observations

- **Chunk metadata for K_back paths.** Identifying K_back_i's character sequence per query happens upstream, at chunk-loading time, via a precomputed sidecar table (see `agpt_build_backoff_table` below). The fire kernel just reads `(K_back_i.path_chars, K_back_i.depth)` for each query; no runtime trie walk needed inside the kernel.

- **Position-of-record for K_back's own forward steps.** Within the depth-batched forward, K_back_i's position-`j` should use RoPE at position `j` (its own depth), exactly the same as path positions use their own depths. Only at the *final gather* into K's K/V stack does the sentinel-position swap matter — that's where K's attention layer sees the backoff slot, and that's where the model needs to know "this is a backoff slot, not a depth-(d-i) path token." Within K_back's own depth-by-depth walk, RoPE-at-own-depth keeps the forward semantics standard.

- **Trie-sparsity handling.** When K_back_i does not exist in the trie (suffix never occurred), the sidecar table marks that slot as inactive for all positions of K_back_i. The mask code from concern (1) handles it for free; the K/V gather produces no contribution for that slot. We log the skip rate as a health metric.

- **Schema gate.** `experimental.backoff_slots: B` (with `B=0` = disabled, recovering current AGPT exactly). Trainer-side wired through v1's `apply_yaml_config_v1` as a recognized experimental key.

### Precomputed sidecar: `agpt_build_backoff_table`

New tool (`src/tools/agpt_build_backoff_table.cr`, mirrors the existing radix-build tooling) that iterates every radix node in a built trie and emits a per-node sidecar of `B` backoff-target IDs:

- Input: built radix trie dir + `B`.
- Output: `<trie-dir>/backoff_B<N>.bin` — a flat `uint32` array of shape `(num_radix_nodes, B)`. Entry `[k, i]` is the radix-node ID of the trie node found by descending from root using `K[k].path_chars[i+1..d]`, or a sentinel value (e.g., `UINT32_MAX`) if that suffix is not a node in the trie.
- One-time precomputation: same caching idiom as the trie itself, can be content-hashed against `(trie-dir, B)` and stored at `data/.tries/<hash>/backoff_B<N>.bin`.
- Loaded at trainer startup, pinned in GPU memory, indexed once per query at chunk-loading time to populate `K_back_i.path_chars[*]` for each query in the chunk.

Implementation order: this sidecar comes first, because the fire-kernel widening depends on knowing K_back_i identities per query at chunk-load time. With the sidecar in hand, the kernel changes have a clean API: "for query at radix-node `k`, K_back IDs are `sidecar[k, 0..B-1]`."

### Order of implementation

1. **`agpt_build_backoff_table`** — Crystal tool, mirrors `agpt_build_radix_corpus`. Standalone; verifiable independently. Output is a `uint32` table you can dump and spot-check against the radix trie's structure.
2. **Chunk-loader extension** — when loading per-query chunk metadata, also load the `B` K_back paths' chunk metadata via the sidecar. Quiet preparation step before the kernel changes.
3. **Per-depth batch widening + active-path mask** — the meaty kernel change. Forward only first; verify per-position outputs against a CPU reference at small d.
4. **Hidden-state stash buffer** — allocate, write at finish-depth per backoff level, read at K's depth-`d−1` gather.
5. **K/V gather + sentinel-position RoPE projection** — final-depth gather kernel that fills K's attention K/V stack with `d + B` entries.
6. **Wire backward** — autodiff propagates through everything we built; verify gradient parity (e.g., `B=0` recovers baseline gradients bit-exact).
7. **YAML gate + smoke** — `experimental.backoff_slots: B` recognized; `B=0` produces identical results to baseline; `B=4` runs end-to-end and produces a checkpoint.

## Experimental setup

- Canonical Shakespeare d=16 baseline. Carved at `data/.splits/2b7ded401e96b610/`.
- Init: `data/input.model` (d_model=64, n_layers=2, n_heads=4, d_ff=256).
- Canonical training: `--mass-weight linear`, `--fire-norm-mass` default-on, `--partition-depth 1`, `--no-accumulate` (forced under YAML — see schema).
- 25 epochs to start (enough to see clear separation from baseline; matches cap-recurrence comparison anchors).
- Multiple shuffle seeds for noise control. 3 pairs minimum, 5+ if results are noisy.
- Eval: canonical `byte_perplexity` via `bin/agpt_experiment` + canonical heldout.
- Conditions: `B=0` (baseline) vs `B=4` (Step 0).

## Success criteria

- **Strong success**: `B=4` `byte_PPL` ≤ KN's ~4 on Shakespeare. The architecture beats KN.
- **Soft success**: `B=4` `byte_PPL` better than baseline but worse than KN. Mechanism works; tuning/scale needed to close the KN gap.
- **Null**: `B=4` `byte_PPL` ≈ baseline. Mechanism didn't take. Diagnostic: instrument `‖attention-weight-mass-on-backoff-slots‖` to see whether the model uses the new slots at all.
- **Hurt**: `B=4` `byte_PPL` > baseline. Position encoding or gradient flow has a bug; debug before drawing architectural conclusions.

## Risks and mitigations

- **Compute**: ~4.4× per-query forward+backward compute. Real but per-fire wall is still ~30s on the existing setup; 25-ep runs go from ~2.5 min to ~10 min. Acceptable for PoC. The widened-batch design preserves the kernel's existing depth-by-depth execution pattern, so peak occupancy should be similar — wider batches at each depth, not separate side-fires.
- **Per-position active-path masking**: the natural depth-batched layout puts entries with different max-depths into the same batch tensor. Control divergence on the warp from skipping inactive entries is a perf concern but not a correctness one. Start with plain control-flow masking; profile and vectorize if it bottlenecks.
- **Hidden-state stash buffer**: per-fire scratch of shape `(num_queries, B, n_layers, d_model)`. For Shakespeare d=16 / d_model=64 / L=2 / typical chunk ~5000 queries: 5000 × 4 × 2 × 64 × 4 bytes ≈ 10 MB per fire — well under the existing KV-cache footprint. Larger configs scale linearly.
- **Training instability**: backoff `K_back` parameters now receive richer gradient signal (endpoint + upstream). May destabilize early training. Mitigations: gradient clipping (`--grad-clip-norm 1.0`), warmup-cosine LR (already canonical).
- **Trie sparsity**: deep `K_back` may not exist for some prefixes. The sidecar marks missing entries inactive; the mask handles it for free. Log the skip rate as a health metric. If frequent, this signals a corpus-coverage issue more than an architecture issue.
- **RoPE position semantics**: the sentinel-position choice (`d+i`) is the simplest but not necessarily the best. Open question for Phase 1.5 / follow-up — see decision (6) above for variants.
- **Backward-pass parity check**: when `experimental.backoff_slots: 0`, the kernel must produce identical forward AND backward results to the baseline (no-backoff) build. This is a non-negotiable regression check before any `B>0` runs.

## After Step 0

- **If strong/soft success**: Phase 1.5 — add landmark slots (root, high-mass shallow hubs). Then position-encoding alternatives. Then Phase 2 (learnable routing).
- **If null**: diagnose the gradient flow and attention-mass instrumentation before declaring the mechanism doesn't work. Check that `K_back`'s parameters are actually receiving the new gradient signal.
- **If hurt**: position encoding is the prime suspect. Try RoPE-at-actual-depth; try learnable backoff embeddings; try zero RoPE.

## Connection to prior work

- **Cap-recurrence** (closed, `project-cap-recurrence-null`): the null was about centroid aggregation + detached gradient + single slot. Step 0 is none of those (per-instance slots, in-flight gradient, multiple slots) — the closure doesn't apply.
- **Existing path-ancestor attention**: Step 0 is a strict superset; `B=0` recovers current AGPT exactly.
- **KN baseline**: known floor at ~4 byte_PPL on Shakespeare. Step 0 target.
