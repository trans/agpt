# Per-Substring Position Information — Implementation Plan

**Status:** REDIRECTED 2026-05-25. The original "dist-rope" direction was explored and ruled out. New direction: **shared-key RoPE** per `notes/seq-len-extension/shared_key_rope.md`.

**Design docs:**
- `notes/seq-len-extension/per-node-position-distributions.md` (substrate: per-substring position tables, storage layout, consumer interfaces)
- `notes/seq-len-extension/shared_key_rope.md` (the *correct* architectural approach — share content projection per node, apply RoPE per occurrence; decouples d/seq_len/A)

## Summary of where this stands

What we built (and what survives):
- ✅ PR 1 — shared `CorpusTrieWalker` (commits c8f600a, c5cb4df).
- ✅ PR 2 — substring catalog + prefix+suffix position tables (commit c8f600a). 8.1M substrings on Gutenberg, 91s build, all 5 files written. Substrate is general-purpose and feeds shared-key RoPE too.
- ✅ PR 3 — `--position-data` + `--pos-encoder {default,expected,dist-rope}` wired into agpt_train (commit c5cb4df). Baseline preserved under `default`.

What we ruled out (today's experiments):
- ❌ **dist-rope** (per-substring Fourier-of-distribution rotation) — regressed Shakespeare L=2 100SE training loss by **+18%** (1.753 vs baseline 1.489).
- ❌ **expected_pos** (per-substring scalar mean position) — regressed by **+30%** (1.929 vs 1.489).
- ❌ The per-substring substitution paradigm in general — both variants break standard RoPE's relative-position attention semantics.

What we learned:
- The collapse-into-one-rotation framing was wrong. Replacing chunk-local position with ANY per-substring summary breaks attention's relative-position machinery. Magnitude collapse (for high-mass substrings with broad distributions) was a secondary issue; the bigger one was simply that `Q · K^T` no longer encoded `(p_q - p_k)`.
- The substrate (substring catalog, position tables, radix→substring lookups) is correct and reusable. It just feeds a different consumer.

## The new direction: shared-key RoPE

Per `notes/seq-len-extension/shared_key_rope.md`. Core idea, in one sentence:

> Don't collapse positions into a fingerprint. Share the **content key** per node (compute one base `k_i = e_i · W_K` per node), and apply standard RoPE *per occurrence*: `K_{i,p} = RoPE(k_i, p)`.

Each occurrence stays a SEPARATE entry in the attention K matrix. The model sees all positions naturally because they're all present as real keys — they just share the underlying projection. Attention math is unchanged per occurrence. Relative-position semantics fully preserved. No magnitude collapse, no broken `Q·K^T`.

**Why this is structurally different from what we tried:**

The dist-rope / expected_pos approaches asked: *how do we summarize a distribution into one rotation per substring?* The answer is "you can't without losing the multi-position information." Shared-key RoPE refuses the summarization — every occurrence is its own entry. The "distribution" is implicit in *which* `A` positions get selected for attention; the model never sees a summary statistic.

The bigger payoff from the shared_key_rope design: it **decouples `d` (trie depth), `seq_len` (linear context), and `A` (attention budget)** — three knobs currently locked together. With shared-key RoPE plus a `node_id → [corpus positions]` map, you can run e.g. `d=16, seq_len=128, A=16` and the model attends over 16 positions selected from a 128-position context, with trie identity quality coming from d=16. The doc also flags the `A=2` extreme: linear-in-seq_len attention, enabling corpus-long context with manageable compute.

## Hypothesis (shared-key RoPE)

If the multi-position information genuinely helps (which dist-rope failed to test cleanly because of its broken semantics), shared-key RoPE will surface that — because attention now sees ALL positions per substring, not a collapsed summary. The expected PPL improvement is unknown; the design just removes the bottleneck that made dist-rope incoherent.

The bigger expected impact is from `seq_len > d`: attention can reach further back than the trie's d-depth lets the identity layer go. That's where the architectural unlock lives.

## Architecture decisions (settled — shared-key RoPE direction)

- **Single learned content key per trie node** (`k_i = e_i · W_K`). Stored once, reused across all occurrences.
- **Per-occurrence RoPE** at the occurrence's actual corpus position (or position-mod-W, or whatever the chosen positional coordinate is). Standard math, no fingerprinting.
- **`corpus_position → node_id` map** required. Built once per corpus pass. ~20 MB at Gutenberg 5M (int32 per position). We already have the inverse (`node_id → [positions]`) from PR 2's position tables; we just need to invert it.
- **Selection rule for A positions per query**: start with most-recent-A (simplest, doesn't exploit `seq_len > d` but proves wiring works). Subsequent variants: uniform-random within seq_len, longest-matching-suffix retrieval, learned node-embedding similarity.
- **First experiment: single-layer prototype with most-recent-A**. The doc explicitly recommends this as the gradient-flow sanity check before any of the harder extensions.
- **Defer**: multi-layer sharing semantics (`shared_key_rope.md` flags this as an open question — layer 2+ inputs diverge per occurrence's attention history), smart retrieval rules, the `A=2` linear-attention regime.

## Implementation scope (~3-4 days for the single-layer prototype)

### PR-A: Build the corpus-position ↔ node_id maps (~half-day)

We already have `prefix_position_table.bin` (per-substring → list of positions). What we need additionally:
- **`corpus_position_to_node.bin`** — int32 array of length N (corpus size). Entry at index p = the radix_id whose `d-suffix` ends at corpus position p (or equivalently, the node whose path matches `corpus[p-d+1 .. p]`).
- Or: derive on-the-fly from the inverse already in position tables. The position_table.bin has, for each substring, all the positions where it occurs. Building the inverse is one pass over the table.

A small Crystal tool: `bin/agpt_build_position_inverse` (or just inline into `agpt_build_position_table`). ~half-day including tests.

### PR-B: Shared-key K computation in attention forward (~CUDA day)

The current K matrix is `K = X · W_K` then `K = RoPE(K, chunk_positions)`. For shared-key RoPE:

- Each "occurrence j" in attention's K matrix corresponds to a (node_id_j, position_j) pair.
- Build `K_j = RoPE(k_{node_id_j}, position_j)` where `k_{node_id_j}` is the shared base key for that node.
- Implementation: gather `k_{node_id_j}` from a per-node base-key table on GPU, then apply standard RoPE with `position_j`.
- The base-key table is `[num_unique_nodes, head_dim]` — ~bytes(d_model) per node. For Gutenberg's 7M nodes × 128 d_model × 4 = 3.5 GB. Tractable on A100 (80 GB VRAM).

Selection rule (most-recent-A) is just "take the last A positions in the corpus_position list for the query's context window."

CLI: `--shared-key-rope` to enable. New flag, distinct from the (now-deprecated) `--pos-encoder`.

### PR-C: Shared-key K computation in attention backward (~CUDA day)

The gradient of L w.r.t. the shared base key `k_i` is the sum across all occurrences `p` where `i` was used:

```
∂L/∂k_i = Σ_p RoPE⁻¹(∂L/∂K_{i,p}, p)
```

RoPE is invertible (it's just a rotation), so the inverse-rotation backward is clean — we already have `launch_rope_batched_inverse` from the current code. Add a scatter-sum over node_id to accumulate per-node gradients into the base-key table.

`∂L/∂e_i = ∂L/∂k_i · W_K^T` — standard projection backward.

### PR-D: Validation + experiment (~half-day + wall)

- Smoke test: with `--shared-key-rope` disabled, baseline is preserved bit-for-bit.
- Single-layer (`L=1`) most-recent-A prototype: confirm gradient flow, loss decreases. ~Shakespeare 10 SE, smoke check.
- Shakespeare L=2 100 SE: same recipe as the dist-rope smoke test. Goal: at least match baseline (1.489), ideally beat it because the model now sees true multi-position info.
- Gutenberg L=4 d=128 100 SE 3 seeds: headline test. Compare to L=4 d=128 baseline (3.7450 ± 0.012).
- If headline wins, then run the **A-sweep**: `A ∈ {4, 8, 16, 32, 64}` at fixed seq_len and d, to find the right attention budget.

## Cost

- ~3-4 days implementation (single-layer prototype with most-recent-A).
- ~3.5 GB GPU memory for the base-key table (Gutenberg, d_model=128).
- ~20 MB for the corpus-position-to-node map.
- Compute per chunk: identical to standard attention if A == seq_len; cheaper if A < seq_len (linear in A).

## Go/no-go criteria for the shared-key RoPE experiment

- **PR-A:** built inverse map matches the forward map under a round-trip check. If off, the position-table machinery has a bug; fix before proceeding.
- **PR-B+C single-layer:** loss decreases on a 10 SE Shakespeare run. If not, gradient flow is broken — debug.
- **Shakespeare L=2 100 SE:** training loss within 5% of baseline (1.489), no NaNs. Doesn't need to win at this stage; just confirms the wiring is correct.
- **Gutenberg L=4 d=128 100 SE headline:** ≥ 1% PPL improvement over 3.7450 → ship. ≥ 2% → strong signal, proceed to A-sweep. < 1% → park, document what was learned.

## What survives from the dist-rope direction

The data structures from PR 2 are reusable infrastructure:
- `src/agpt/substring_catalog.cr` — still the canonical substring identity layer.
- `src/agpt/position_table.cr` — per-substring position lists feed the corpus-position-to-node inverse map for PR-A.
- `src/agpt/radix_to_substring.cr` — still useful for any cross-trie substring joins.
- `src/agpt/corpus_trie_walker.cr` — substrate for any future trie-walking tool.
- `bin/agpt_build_position_table` — Gutenberg artifact already on disk in `/tmp/gut_position_data/`.

The `--pos-encoder {expected, dist-rope}` modes wired into agpt_train (PR 3) are now ruled-out variants. They can stay in the code as "explored, doesn't help" — useful for future re-evaluation if someone wants to confirm — but should not be the default.

## Files (after shared-key RoPE landing)

Crystal source (existing, reused):
- `src/agpt/corpus_trie_walker.cr`, `substring_catalog.cr`, `radix_to_substring.cr`, `position_table.cr` — all from PR 2.

Crystal source (new for shared-key RoPE):
- `src/tools/build_position_inverse.cr` → `bin/agpt_build_position_inverse` (PR-A)

CUDA source (modifications):
- `src/cuda/agpt_train.cu` — `--shared-key-rope` flag, per-occurrence K construction, gradient scatter to base-key table.
- `src/cuda/agpt_position_data_io.cuh` — extend with base-key table + corpus-position map readers.

Data artifacts (per corpus):
- Existing: `<corpus>_position_data/{substrings.bin, prefix_radix_to_substring.bin, suffix_radix_to_substring.bin, prefix_position_table.bin, suffix_position_table.bin}`
- New: `<corpus>_position_data/corpus_position_to_node.bin`

Docs:
- `notes/seq-len-extension/shared_key_rope.md` (the design)
- `notes/seq-len-extension/per-node-position-distributions.md` (substrate layout, still current)
- `notes/seq-len-extension/position-distributions-plan.md` (this file)
- `rnd/dist-rope-smoke/` (experiment results that ruled out the original direction)
- `rnd/shared-key-rope/` (experiment results when run)

## Appendix: dist-rope post-mortem

Why dist-rope failed:

1. **Per-substring substitution breaks relative position.** Standard RoPE's `Q·K^T` depends only on `(p_q - p_k)` because both Q and K are rotated by angles that share a chunk-position coordinate. dist-rope rotates them by per-substring angles that don't share any coordinate system — `Q·K^T` no longer encodes a meaningful distance.
2. **Magnitude collapse for broad distributions.** Fourier of a uniform distribution averages to ~0, so the rotation matrix shrinks Q/K vectors to near zero. Hits high-mass substrings hardest (which are exactly the ones that dominate training events).
3. **Same problem at the math level.** dist-rope's `eff_cos = Σ p(p) cos(θ(p,i))` is mathematically `Σ_i Σ_j w_qi · w_kj · f(p_qi - p_kj)` after attention — i.e., the matrix of pairwise position differences gets weighted-summed into a scalar. Information collapse is inherent.

The "matrix of differences" framing makes clear that any single per-substring rotation is a summary — even with a perfect summary the model loses access to individual position pairs. Shared-key RoPE solves this by keeping every occurrence as a separate attention entry.

Empirical results that ruled out the direction:
- Shakespeare L=2 d=64 100 SE, identical recipe except for `--pos-encoder`:
  - `default`: training loss 1.489
  - `dist-rope`: 1.753 (+18%)
  - `expected_pos`: 1.929 (+30%)

Logs in `rnd/dist-rope-smoke/`. Magnitude inspection (`src/tools/inspect_eff_rope.py`) showed magnitudes were mostly preserved for low-mass substrings (66% of nodes); the regression was driven by the substitution paradigm, not magnitude collapse alone.

---

# Addendum: Codex's Multi-Slot Sampling Formulation (2026-05-25)

After the original plan was written, codex-agpt independently proposed a related but distinct formulation of the same decoupling problem. Documented here as an addendum; the original shared-key-RoPE plan above is preserved unchanged.

## Codex's framing

> d = trie/path depth used to build/query AGPT structure
> seq_len/window = number of position-coded K/V/query slots exposed to attention
>
> Right now those are effectively tied. Multi-positional encoding gives a path to decouple them: a node/path event can contribute multiple RoPE-positioned views, drawn from the corpus positions where that node occurs, so attention can see more than one positional realization without increasing trie depth.

## Implementation shape

```
radix node -> occurrence positions side-table
training fire -> choose K positions for that node/event
emit K RoPE-positioned K/V slots
attention window bound = independent seq_len cap
loss still tied to original AGPT target/event
```

CLI knob: `--pos-samples-per-node K`. Start with K=1 (one sampled or round-robin position per fire), then K=2/4/8, then `--seq-len 32/64 while d remains 16`.

## Within-edge position handling

For radix nodes with multi-char edges, sampled corpus position `p_end` (the terminal position of the sampled occurrence) extends to within-edge characters via:

```
real_pos(query_depth) = p_end - (edge_end_depth - query_depth)
```

So a char in the middle of the edge inherits a real corpus position relative to the sampled occurrence's terminal. Different sampled occurrences yield different position assignments for the same intra-edge character — naturally averaged across training fires.

## Deterministic sampling rule

```
position_index = hash(node_id, epoch, fire_count) % occurrence_count
```

Reproducible per (node, epoch, fire), avoids RNG state coordination. Over enough epochs, every position is visited; per-epoch variance is bounded.

## Prerequisite: CSR-format `node_id → [corpus terminal positions]`

```
node_pos_offsets[node_id]
node_positions[offsets[node_id] : offsets[node_id+1]]
```

Compact (~20 MB on Gutenberg for 7M nodes' position lists), one pass over the existing position tables to build. This is the same prerequisite as the shared-key-RoPE plan; the two approaches share this substrate.

## How Codex's plan differs from shared-key RoPE

| aspect | shared-key RoPE (original plan above) | Codex's multi-slot sampling |
|---|---|---|
| **Persistent storage** | 3.5 GB base-key table on GPU (k_i = e_i·W_K per node, kept across epochs) | ~20 MB CSR map only |
| **K projection cost** | Once per node, persistent (compute saving across epochs if reused) | Once per node per fire (same as current AGPT) |
| **Multi-position mechanism** | Each (node, position) is a separate attention entry; selection rule picks which | Each fire emits K positional views by sampling K positions and applying RoPE K times |
| **Cross-fire sharing** | Explicit — shared base keys persist across fires/epochs | Implicit — each fire re-projects (matches current AGPT pattern) |
| **Selection rule needed?** | Yes (most-recent-A / random / similarity / etc.) | Only when K × nodes_in_chunk > seq_len or when seq_len > d |

## Why Codex's approach may be a better fit for AGPT

AGPT already projects K per node per fire (the K/V cache is per-node). The "shared projection across occurrences" benefit that shared-key RoPE offers in a standard transformer is largely redundant in AGPT — the trie structure already groups occurrences by node.

The cross-epoch base-key sharing in shared-key RoPE adds 3.5 GB of GPU memory for benefits that are mostly already captured by AGPT's per-node firing pattern. Codex's plan skips this cost by re-projecting per fire and applying RoPE to the freshly-projected K vector K times per node per fire.

**For our setting specifically, Codex's plan is more memory-efficient at no compute cost.** The multi-positional capability is preserved (K positional views per node per fire); the persistent base-key table is unnecessary.

## What Codex's plan is missing

**Selection rule for which positions a query attends to.** For K=1 with seq_len = d (Codex's first experiment), the question doesn't arise — same number of K/V slots as current AGPT. For K>1 with seq_len = d, attention sees K × nodes_in_chunk slots; need a rule when this exceeds seq_len. For seq_len > d (Codex's later experiment), the central new design choice is which past positions enter each query's attention.

The selection-rule design from shared-key RoPE plugs in here. The two approaches are complementary: Codex's plan handles the *emission* side (how to generate K positional views per node), shared-key RoPE handles the *selection* side (how to pick which K positions to actually attend to).

## Shared concerns

The **multi-layer leak** applies to both approaches identically. Once layer 2 has context-dependent inputs per occurrence, the "share something per node" assumption no longer holds. Three options (accept layer-1-only sharing, re-share lossily, hybrid) remain the same.

## Progressive experiment plan (Codex's order)

```
Phase A (~half-day): build CSR node→positions map
Phase B (~CUDA day): K=1 mode — one sampled real corpus position per fire,
                     deterministic hash sampling, d=16, seq_len=16
Phase C: K=2/4/8 with seq_len = d — emit multiple positional views per fire
Phase D: seq_len > d with selection rule — true decoupling
```

Each phase is one knob change, isolating effects.

## Status

Codex's plan and the shared-key RoPE plan are alternative paths to the same architectural endpoint (decoupled d/seq_len/A, multi-positional attention per node). They share the CSR prerequisite. The choice between them is largely about memory vs implementation complexity tradeoffs.

**Update 2026-05-25:** Neither was chosen. A third design — the **harmonic filter** — emerged that dominates both. See `notes/seq-len-extension/harmonic-filter-plan.md` for the chosen direction. This file is preserved as the historical record of dist-rope's failure and the shared-key / multi-slot alternatives we considered along the way.
