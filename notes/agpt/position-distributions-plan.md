# Per-Node Position Distributions — Implementation Plan

**Status:** Sketch, pending go/no-go decision.
**Date:** 2026-05-24
**Design doc:** `notes/agpt/per-node-position-distributions.md` (architecture, storage, consumer interfaces).

## What we're testing

Extend AGPT's effective position window beyond the trie's natural depth (d=16) by giving each radix node a distribution over *long-window positions* (W=64) where its prefix occurs. The model conditions on this distribution via a position-encoding consumer — first cut: **distribution-aware RoPE** (`eff_cos[node][i] = Σ_p (count[p]/total) · cos(p / base^(2i/HD))`).

Build the substrate for **both prefix and suffix tries** in one go (per `notes/prefix-suffix-fold-architecture.md` — each substring has both an "appears as prefix here" and "appears as suffix here" distribution; the dual-model training framework `agpt_dual_train.cr` is the eventual consumer). All per-substring data structures (position tables, future KN distributions, anything else) are keyed by a **canonical `substring_id`** assigned by a unified substring catalog. Each trie has a small lookup array (`prefix_radix_to_substring[]`, `suffix_radix_to_substring[]`) that translates trie-local radix_ids to the canonical substring_id. Suffix trie is already buildable via `build_radix_corpus --reverse` (existing tool).

The trie stays at d=16 (so trie quality stays high — past d=16, mass=1 leaves blow up). The position signal is what extends.

## Hypothesis

The trie taps out at d=10-20 because most leaves go mass=1 past that point. The model can't learn from one-hot targets at unreachable mass-1 leaves. But the *position information* (where in the long window does this prefix tend to occur?) is recoverable from the corpus even when the trie can't grow deeper.

Distribution-aware RoPE encodes "the distribution of positions where this prefix lives in the corpus" as a vector inside the unit circle: sharp distribution → rotation near the unit circle (strong positional identity); broad distribution → rotation near origin (positionally ambiguous). The model gets to learn how to use these distinctions.

**Expected PPL gain: uncertain. 3-10% on Gutenberg if the hypothesis holds.**

## Risks

1. **The position info might not be there.** If the corpus's long-window structure is uninformative (every prefix appears at every position roughly uniformly), the distribution carries no useful signal. Mitigated by inspecting the empirical entropy of position distributions before training — if most nodes have near-uniform distributions, the experiment is dead before we wire anything to the loss.
2. **The model can't use it.** Replacing per-instance position with a per-node distribution loses the *instance-level* position signal. The model used to know "this specific occurrence is at position 47 in the chunk." It would now know "this prefix tends to occur at these positions across the corpus." The latter may be less actionable. The expected_pos baseline (PR 2) helps distinguish this from "the kernel is buggy."
3. **Implementation bugs in dist-rope.** The eff_cos/sin precompute is novel for our codebase. Geometric reasoning is correct, but a sign or normalization error could silently corrupt training. Mitigated by an analytical unit test: for a delta-distribution (count concentrated at one position), eff_cos/sin must equal the standard RoPE cos/sin at that position.
4. **Wrong W.** W=64 might be too small to see benefit beyond d=16. Or too big and over-smoothing. Cheap to sweep once the infrastructure exists.

## Architecture decisions (settled — modify if first results say otherwise)

- **W = 64** for first cut. Big enough to be meaningfully larger than d=16 (4x), small enough to keep storage modest.
- **Sliding regime** (sub-paths at every offset). Matches current trie build. Bigger tables than aligned but more info per node.
- **Sparse storage** (`PosBin { u16 pos; u32 count }`, indexed by `pos_offsets[substring_count + 1]`). Per design doc, ~100 MB on Shakespeare, ~400 MB on Gutenberg. Two tables (prefix-position + suffix-position) → ~200 MB Shakespeare, ~800 MB Gutenberg total.
- **Canonical substring_id indexing.** Build a substring catalog once (every unique substring of length 1..d gets a dense sequential `substring_id`). All per-substring data tables are keyed by `substring_id`. Each trie has a small lookup array (`prefix_radix_to_substring[]`, `suffix_radix_to_substring[]`) to translate trie-local radix_ids. Forward passes do: `radix_id → substring_id` (one indirect array load) → index into the position table. Catalog itself ~100 MB on Gutenberg; lookup arrays ~30 MB each. Future per-substring data (KN, anc-grad stats, anything) joins naturally on substring_id without per-data-structure bridging.
- **Reuse Codex's walk logic** from `src/tools/agpt_position_map.cr` — the per-position corpus-walk-through-trie inner loop is the same code we need. Factor into a shared module rather than reimplement.
- **First consumer: distribution-aware RoPE.** It's the most architecturally interesting, gives a single per-node lookup at training time (cheap), and the substitution is a one-kernel change. Wire into the dual-model trainer `agpt_dual_train.cr` so both prefix and suffix forward passes benefit.
- **Second consumer: expected_pos** (deterministic, single float per node). Simplest possible baseline; useful as a sanity check — if dist-rope wins, does it win over expected_pos? If not, the distribution per se isn't what's helping.
- **Defer:** sampled, Fourier, wavelet, CRT, ALiBi. All sit on the same data structure; add later if dist-rope / expected_pos show signal.

## Implementation scope (~3 days / 5 PRs)

### PR 1: Extract shared walk module (~2-3 hours)

Codex's `agpt_position_map.cr` already implements the corpus-walk-through-trie inner loop, including:
- Loading the trie via `RadixTrieReader`
- Building the `(parent_id, first_token) → child_record` index
- Walking from each corpus start position, matching edge tokens, recording terminal node IDs

Refactor that inner loop into `src/agpt/corpus_trie_walker.cr` with a callback API:

```crystal
walker = CorpusTrieWalker.new(reader, dataset)
walker.walk { |radix_id, start_corpus_pos, terminal_corpus_pos|
  # called once per (node, contribution) pair as the walk progresses
}
```

Update `agpt_position_map.cr` to use the shared module. Verify its output is unchanged (regression test).

### PR 2: Build substring catalog + prefix + suffix position tables (~6-8 hours)

New Crystal binary: `bin/agpt_build_position_table`. Takes both tries and both corpus directions; produces four artifacts in one run.

**Inputs:**
- `--prefix-trie <dir>` — prefix radix trie (forward corpus order)
- `--suffix-trie <dir>` — suffix radix trie (reverse corpus order, already built via `build_radix_corpus --reverse`)
- `--corpus <file>` — the canonical (forward) corpus text

**Build flow:**
1. Walk the forward corpus through the prefix trie via shared `CorpusTrieWalker`. For each `(radix_id, start_pos, terminal_pos)` contribution:
   - Look up the substring chars at this position (length = node's endpoint depth).
   - Assign or retrieve the canonical `substring_id` from the catalog (hash-keyed on substring chars). New substrings get the next sequential ID.
   - Record `prefix_radix_to_substring[radix_id] = substring_id`.
   - Accumulate `prefix_pos_counts[substring_id][bin] += 1` where `bin = start_pos % W`.
2. Walk the reversed corpus through the suffix trie. Same loop, looking up the reversed substring (i.e., the original substring read right-to-left). For unification: the substring catalog stores substrings in their **original (forward) form**, so when the suffix walk lands on a node representing the reversed string `s_rev`, we look up the original `s = reverse(s_rev)` in the catalog. Same substring_id falls out.
   - Record `suffix_radix_to_substring[suffix_radix_id] = substring_id`.
   - Accumulate `suffix_pos_counts[substring_id][bin] += 1`.

**Outputs (four binary files):**

```
substrings.bin    — magic "ASUB", substring_count, then for each id:
                    (length:u8, chars:length bytes)
                    Sorted by substring_id (== insertion order); load into memory as
                    a hash table on chars → substring_id.

prefix_radix_to_substring.bin — magic "PRTS", prefix_radix_count,
                                int32[prefix_radix_count] of substring_ids

suffix_radix_to_substring.bin — magic "SRTS", suffix_radix_count,
                                int32[suffix_radix_count] of substring_ids

prefix_position_table.bin     — magic "APOS", "prefix", window_size W, regime,
                                substring_count, total_bins,
                                pos_offsets[substring_count+1],
                                pos_bins[total_bins]  (PosBin = u16 pos + u32 count)

suffix_position_table.bin     — magic "APOS", "suffix", ... (same layout)
```

**Validation:**
- `total_bins ≤ substring_count × W`.
- For prefix table: `sum(prefix_pos_counts[id]) = prefix-trie's edge_mass for the corresponding radix node`. Sanity-check via spot-comparison.
- Cross-check unification: spot-check several short substrings (e.g., "the", "and"); both `prefix_radix_to_substring` and `suffix_radix_to_substring` should map to the same substring_id when looking up the corresponding radix nodes.

**Build:** `just build-build-position-table`.

If the suffix trie isn't already on disk for our test corpora, build it first via `build_radix_corpus --reverse` (a few minutes).

### PR 3: Loaders + expected_pos consumer (~4 hours)

Three loaders, all in `src/agpt/`:

- `substring_catalog.cr` — loads `substrings.bin`. Exposes `id_for(chars : Bytes) : Int32?` and reverse lookup.
- `radix_to_substring.cr` — loads either trie's lookup array; method `substring_id_for(radix_id : Int32) : Int32`.
- `position_table.cr` — loads a `*_position_table.bin` (keyed by substring_id). Methods: `expected_pos(substring_id) : Float32` (precomputed once at load), `bins(substring_id) : Slice(PosBin)` (raw for richer consumers).

Wire into `agpt_train` as one bundle CLI flag: `--position-data <dir>` (loads all four files from one directory). When set, **expected_pos** consumer activates: forward pass does `radix_id → substring_id → expected_pos`, uses the resulting scalar instead of the chunk-local position when computing RoPE.

For prefix-only training: only prefix-side lookup needed. For dual: both. Loader detects what's present in the directory.

Smoke test: should match baseline within noise (since expected_pos is just a scalar — basically a per-node fixed position bias).

### PR 4: Distribution-aware RoPE consumer (~4-6 hours)

- At load time, precompute per (substring_id, head_dim_pair):
  - `eff_cos[substring_count][HD/2]`, `eff_sin[substring_count][HD/2]`, float32
  - Storage: ~100 MB on Shakespeare at HD=16, ~400 MB on Gutenberg (per direction)
- New CUDA kernel that replaces the standard `cos_cache[pos][i]` / `sin_cache[pos][i]` lookup with `eff_cos[substring_id][i]` / `eff_sin[substring_id][i]`. Forward pass does `radix_id → substring_id` lookup once per attention position; one extra indirect load. Same attention math otherwise.
- CLI: `--pos-encoder dist-rope` (vs `expected` vs `default`).
- For dual training: both prefix and suffix forward passes use their respective eff_cos/sin table (built from their respective position table).

### PR 5: Validation + experiment (~half-day + wall)

- Smoke test: `--pos-encoder default` (or no flag) = baseline reproducible bit-for-bit.
- Sanity test: for a delta-distribution (synthetically-constructed node with count concentrated at one position), eff_cos/sin must equal standard RoPE cos/sin at that position.
- Shakespeare L=2 small run, 3 seeds, 3 modes (default / expected / dist-rope). ~15 min each.
- Gutenberg headline: L=4 d=128 100 SE 3 seeds, all three modes. ~3h each on the pod.
- Compare to L=4 d=128 100 SE baseline (3.7450 ± 0.012).
- Dual-model test if time permits: `agpt_dual_train.cr` with prefix+suffix tables.

## Cost

- ~3-4 days implementation (added a day for prefix+suffix scope and Codex-walk extraction; +2 hours for substring catalog)
- Gutenberg disk: ~800 MB position tables + ~100 MB substring catalog + ~60 MB lookup arrays = **~960 MB total**
- Gutenberg RAM: ~800 MB precomputed eff_cos/sin (per direction) + catalog + lookup arrays held in CPU memory
- Per-position lookup overhead in attention kernel: one extra indirect array load (`radix_id → substring_id`), no algorithmic change

## Why this is the right next step

The trie taps out at d=10-20. Every smoothing experiment so far has worked around that ceiling — anc-grad routes information across nodes, weighting reshapes per-event gradients, distillation reshapes targets. None changes what context the model can condition on. This one does: it separates *position info* from *depth info*, so the model can see structure beyond the trie's natural reach without growing the trie deeper.

The substrate is also load-bearing for follow-up work: the same position table feeds sampled, expected, dist-rope, Fourier, wavelet, CRT, and ALiBi consumers. If dist-rope misses, the next variant is half a day away on the same data structure. Building the substrate is most of the investment; trying additional consumers is cheap.

## Go/no-go criteria for the experiment itself (if we proceed)

- **PR 1 smoke test:** total bin counts match an analytical formula. If off, debug before continuing.
- **PR 2 (expected_pos):** small Shakespeare ≥ baseline - 0.05 PPL. Confirms the loader and per-position substitution path are sound. If massively worse, the substitution is bug'd before we add complexity.
- **PR 3 (dist-rope) Shakespeare check:** within ±5% of baseline, no NaNs, training converges. Doesn't need to win at this stage; just confirms the kernel works.
- **Gutenberg headline:** dist-rope ≥ 2% PPL improvement (3 seeds, p < 0.05) over baseline → ship. < 2% → park, document, possibly try Fourier or one of the other consumers since the substrate is already built.

## Open design questions (defer until data)

1. **W sweep**: W=32, 64, 128, 256? Cost is linear-ish in W for tables; pick after first results.
2. **Aligned vs sliding**: sliding for first cut; compare later.
3. **dist-rope vs expected vs sampled**: deferred to data. The substrate enables all of them.
4. **Per-fire sampled position (consumer 1)**: the noise-as-feature angle is interesting but needs a small-scale ablation before committing to it as the default. Skip for now.
5. **Should the eff_cos/sin be a strict replacement for RoPE, or added as a residual?** Per design doc it's a replacement. Could test as residual (additive) variant if pure replacement underperforms.

## Files (when implemented)

Crystal source:
- `src/agpt/corpus_trie_walker.cr` — PR 1, shared walk module (extracted from Codex's `agpt_position_map.cr`)
- `src/agpt/substring_catalog.cr` — PR 3, loads `substrings.bin`
- `src/agpt/radix_to_substring.cr` — PR 3, loads either trie's lookup array
- `src/agpt/position_table.cr` — PR 3, loads `*_position_table.bin`
- `src/tools/build_position_table.cr` → `bin/agpt_build_position_table` (PR 2)
- `src/cuda/agpt_train.cu` — CLI (`--position-data <dir>`, `--pos-encoder <mode>`) + RoPE kernel modification (PR 4)

Data artifacts (per corpus, when built):
- `<corpus>_position_data/substrings.bin`
- `<corpus>_position_data/prefix_radix_to_substring.bin`
- `<corpus>_position_data/suffix_radix_to_substring.bin`
- `<corpus>_position_data/prefix_position_table.bin`
- `<corpus>_position_data/suffix_position_table.bin`

Docs:
- `notes/agpt/per-node-position-distributions.md` (existing design doc)
- `notes/agpt/position-distributions-plan.md` (this file)
- `rnd/position-distributions/` (experiment results when run)
