# Harmonic Filter for Multi-Position Encoding — Plan

**Status:** Chosen direction 2026-05-25.
**Origin:** user design 2026-05-25 (after dist-rope was empirically ruled out and shared-key-RoPE / multi-slot-sampling alternatives were scoped).
**Companion docs:** `position-distributions-plan.md` (dist-rope post-mortem + alternatives we considered), `shared_key_rope.md` (a different multi-position approach we did not choose).

## What

Encode a node's distribution over historical corpus positions as a **normalized phase chord** on a chosen subset of RoPE dim-pairs. The query side stays standard RoPE (a sharp single-frequency probe). Attention's dot product naturally becomes a matched filter: the score spikes when the query's current position matches any of the historical positions encoded in the key's chord.

No content-key sharing across occurrences. No K matrix expansion. No selection rule. Per-fire compute and per-token K/V cache shape unchanged from current AGPT.

## Why this design

The dist-rope failure (training-loss regression 18-30% on Shakespeare L=2 100 SE) traced to two compounding problems:

1. **Magnitude collapse** — averaging unit vectors at different angles destructively interferes for broad distributions. The eff_cos/eff_sin vector shrinks toward origin, and the attention "rotation" becomes a scale-then-rotate that shrinks Q and K vectors.
2. **Broken relative-position semantics** — replacing Q's chunk-local position with a per-substring summary destroys `Q·K^T ∝ f(p_q - p_k)`.

The harmonic filter addresses both:

- **Normalization** (`K_chord = Normalize(...)`) fixes magnitude collapse — the key always has unit length, only PHASE matters.
- **Query stays on its actual position** (no per-substring substitution on the Q side). The phase-difference math `e^{j(q - p)·ω}` is what attention reads — and `e^0 = 1` whenever q matches one of K's historical p's. Constructive interference at matching positions, destructive cancellation elsewhere.

Plus a clean property neither shared-key nor multi-slot has:
- **No multi-layer leak.** The chord is a *positional encoding*, not content sharing. Each layer still computes its own K = X·W_K per occurrence; the chord rotation is applied on top, based on node identity. Node identity is layer-independent, so the chord is layer-independent. No leak.

## The math (user's formulation)

For each radix node i with position distribution `P_i = {p : count(p)}` over W-window positions, and for each "hour-hand" dim-pair index j with frequency ω_j:

**Key encoding (broadcast / chord):**

```
K_hour,j[i] = Normalize( Σ_p count(p) · e^{j p ω_j} )
            = ( eff_cos_normalized[i, j], eff_sin_normalized[i, j] )
```

Normalization divides the raw weighted sum by its magnitude, yielding a unit-length 2D vector per (node, dim-pair). Stored as flat `chord_cos[radix_count][hour_dims]` and `chord_sin[radix_count][hour_dims]` tables.

**Query encoding (tuning fork):**

```
Q_hour,j[q] = e^{j q ω_j} = ( cos(q · ω_j), sin(q · ω_j) )
```

Where `q` is the query's actual corpus position (not chunk-local). Standard RoPE rotation at the query's true position.

**Attention dot product (matched filter):**

```
Q*_hour,j · K_hour,j ∝ (1/|S|) · Σ_p count(p) · e^{j(p - q)·ω_j}
                     = (1/|S|) · Σ_p count(p) · cos((p - q)·ω_j)     [real part]
```

When `q == p_k` for some k, that term contributes `cos(0) = 1`. Other terms with `(p - q)·ω_j ≠ 0` contribute rotating values that scatter across multiple frequencies and don't constructively reinforce. The model attends most to positions whose K's chord includes q.

## Architectural decisions

- **Dim-pair split.** Reserve a subset of head_dim's dim-pairs for the chord ("hour hand"), leave the rest for standard chunk-position RoPE ("minute hand"). Start with the dim-pairs whose periods are closest to W=64 — these resolve positions within W with reasonable precision. For head_dim=16 with base=10000, pairs 1-3 (periods ~20, ~63, ~199) are the natural chord candidates; pairs 0 and 4-7 stay standard.
- **Normalization is enforced at precompute.** Magnitude information is discarded; only phase information enters the model. This is the deliberate fix for dist-rope's magnitude collapse.
- **Q uses corpus position, not chunk position.** Required for the matched-filter math. Each query's corpus position is already known at chunk-assembly time (it's the corpus offset of the chunk's anchor + the query's chunk-local position).
- **K uses chord_cos/chord_sin lookup** for hour-hand dims, standard RoPE for minute-hand dims. Hybrid per-pair within the same K vector.
- **No multi-layer leak.** Chord is positional encoding, layer-independent. Layers 2+ compute their K's per occurrence normally; the chord lookup applies the same way at every layer.
- **No K matrix expansion.** One K entry per occurrence. Same attention shape as today.

## Implementation scope (~1-2 days total)

### PR 1: Chord precompute tool (~half-day, Crystal)

New tool: `bin/agpt_build_chord_table` (or extend `bin/agpt_build_position_table`).

For each radix node:
- Read its position list from the existing `prefix_position_table.bin`
- For each hour-hand dim-pair j with frequency ω_j:
  - Compute `raw_cos = Σ_p count(p) · cos(p · ω_j)`, `raw_sin = Σ_p count(p) · sin(p · ω_j)`
  - Magnitude `m = sqrt(raw_cos² + raw_sin²)`
  - If m > epsilon: `chord_cos[i, j] = raw_cos / m`, `chord_sin[i, j] = raw_sin / m`
  - Else: `chord_cos[i, j] = 1.0`, `chord_sin[i, j] = 0.0` (identity rotation fallback for degenerate nodes)

Output: `chord_table.bin` with magic "ACRD", radix_count, hour_dim_count, then float32[radix_count × hour_dims × 2] (cos+sin interleaved per dim).

Storage estimate (Gutenberg 7M nodes, 4 hour dims): 7M × 4 × 2 × 4 bytes = 224 MB. Modest.

Build target: `just build-agpt-build-chord-table`.

### PR 2: Trainer integration (~CUDA day)

Extend the existing `--rope-mode split` / `--rope-split-secondary` machinery:

- Add `--rope-split-secondary chord` as a new mode.
- Add `--chord-table <path>` to load the precomputed table at startup.
- Hour-hand dims (the ones the existing `--rope-split-depth-heads` reserves for secondary) lookup `chord_cos / chord_sin` per (substring_id, dim-pair) instead of using a position-derived rotation.
- Minute-hand dims continue to use standard chunk-position RoPE on the K side AND the Q side.
- Q side for hour-hand dims: use the query's corpus position (not chunk position) to compute standard RoPE cos/sin. Q's hour-hand encoding is the "tuning fork" matching against K's "chord."

The CUDA kernel changes are minimal: the per-dim-pair branch already exists in the `split` mode dispatch. Just need:
- Pass `chord_cos`, `chord_sin` device pointers and `substring_id` per query to the kernel.
- For K's hour-hand pair lookup, use chord_cos[sid, j] / chord_sin[sid, j] instead of cos_cache[K_pos, dim].
- For Q's hour-hand pair lookup, use cos_cache[Q_corpus_pos, dim] / sin_cache[Q_corpus_pos, dim].

Backward: the chord is fixed input data (no gradient flows back to the chord table). Standard backward through the rotation matrix applies — the chord just acts like a constant rotation per node. Crystal kernel changes negligible.

### PR 3: Validation + experiment (~half-day + wall)

- Smoke test: with `--rope-split-secondary chord` disabled, baseline is preserved bit-for-bit.
- Smoke test: with hour-dims-count = 0 (all minute-hand), chord has no effect — equivalent to baseline.
- Sanity test: for a single-position node, chord_cos / chord_sin should equal standard RoPE cos/sin at that position. Equivalent to attending only when Q is at that exact position.
- Shakespeare L=2 d=64 100 SE, hour-hand on pairs 2-3 (periods ~63, ~199). Compare to baseline (1.489 final training loss).
- Gutenberg L=4 d=128 100 SE 3 seeds, headline test. Compare to L=4 d=128 baseline (3.7450 ± 0.012).

## Cost

- ~1.5-2 days implementation
- ~225 MB chord_table on disk for Gutenberg (per direction, if dual-trained; prefix-only training is just 1 table)
- ~225 MB GPU memory for the chord table at training time
- Per-position lookup overhead in attention kernel: no algorithmic change, just a different cache pointer for hour-hand dim-pairs
- No K matrix expansion, no sequence-length changes

## Go/no-go for the experiment

- **PR 1 sanity:** chord_table for a single-position node equals standard RoPE cos/sin at that position. If not, precompute is wrong.
- **PR 2 smoke:** baseline preserved when chord disabled. Within run-variance (TF32 nondeterminism gives ~0.02 loss spread).
- **Shakespeare L=2 100 SE with chord:** within 5% of baseline training loss (1.489). Doesn't need to win at this stage; just confirms the kernel works and gradients flow.
- **Gutenberg L=4 d=128 100 SE headline:** ≥ 1% PPL improvement over 3.7450 → ship. ≥ 2% → strong signal, sweep hour-hand-dims and frequency-selection. < 1% → park with full writeup of what was tested.

## Open design questions

1. **Which dim-pairs are "hour hand"?** Sweet spot is periods comparable to W=64. For head_dim=16 base=10000, that's pairs 1-3. Could also test pairs 0-3 (more chord channels, includes the highest-frequency / sub-W periods) or pairs 2-3 (just the two pairs whose periods straddle W).
2. **Different RoPE base for hour-hand dims?** Current standard base is 10000. For W=64, octave-spaced periods (16, 32, 64, 128) would use a smaller base. Could rescale just the hour-hand frequencies to match W more cleanly. Tradeoff: easier interpretation vs further from standard RoPE convention.
3. **What's the Q's position coordinate?** Most natural: corpus position. Could also try `corpus_position mod W` for periodicity matching, or chunk position for backward compatibility. Probably "corpus position" since K's chord is in absolute corpus-position coordinates.
4. **Does the chord need to be unified across prefix/suffix tries?** For prefix-only training, the prefix chord_table is enough. For dual-model training (`agpt_dual_train`), each side uses its own chord table. Same substring_id catalog, different chord tables per direction.
5. **Should we keep magnitude info anywhere?** The user's design explicitly discards magnitude (normalization). For interpretability/diagnostics, could log per-node entropy or magnitude separately, but it doesn't enter training.

## Files (when implemented)

Crystal source (new):
- `src/tools/build_chord_table.cr` → `bin/agpt_build_chord_table` (PR 1)
- `src/agpt/chord_table.cr` — Crystal-side reader (PR 1)

CUDA source (modifications):
- `src/cuda/agpt_train.cu` — `--rope-split-secondary chord` mode + `--chord-table` flag (PR 2)
- `src/cuda/agpt_position_data_io.cuh` — extend with chord_table reader (PR 2)

Data artifacts (per corpus):
- `<corpus>_position_data/chord_table.bin` (prefix-side) — or `prefix_chord_table.bin` / `suffix_chord_table.bin` if dual

Docs:
- `notes/seq-len-extension/harmonic-filter-plan.md` (this file)
- `notes/seq-len-extension/position-distributions-plan.md` (related — dist-rope post-mortem + shared-key/multi-slot alternatives we passed on)

## Why this beats the alternatives we considered

| approach | beats baseline? | breaks attention semantics? | multi-layer leak? | K matrix grows? | memory | implementation |
|---|---|---|---|---|---|---|
| dist-rope | NO (regressed 18%) | YES (broken relative-pos) | N/A (single layer test showed it doesn't work) | NO | 520 MB Gut | 2 days (done) |
| expected_pos | NO (regressed 30%) | YES | N/A | NO | small | done |
| shared-key RoPE | untested | NO | YES | NO | 3.5 GB Gut | 3-4 days |
| Codex multi-slot | untested | NO | YES | YES (K × N entries) | 20 MB Gut | 2-3 days |
| **harmonic filter** | untested | **NO** | **NO** | **NO** | **225 MB Gut** | **1-2 days** |

The harmonic filter is the only design that's both architecturally clean (no semantic breaks, no multi-layer leak, no K matrix expansion) and cheap to implement (~1-2 days). The other approaches each have at least one significant downside the harmonic filter avoids.
