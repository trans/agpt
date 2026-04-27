# Prefix-to-Suffix Attention via Structural Max-Overlap Matching

> **Status (2026-04-26): planning + design.** No code yet. Pivoted here from
> `rnd/lightning-cap-warmup/` after confirming the current AGPT architecture
> is capacity-bound at PPL ~29 on Gutenberg 5M (d_model=64, n_layers=2). Need
> a different training signal, not just better recipes.

## Hypothesis

A prefix-leaf π should attend over a small **menu of plausible suffixes** the
corpus structurally supports — not just predict the next token from an
isolated prefix. The model's job becomes "pick which suffix best follows this
prefix" rather than "guess the next character". This gives:

1. **Richer training signal**: per-prefix supervision over D-character
   continuations, not just one token
2. **Generalization-friendly target**: structural matching gives multiple
   compatible suffixes per prefix (not just the literal corpus continuation),
   reducing memorization pressure
3. **Bounded compute**: matching policy gates the candidate set per prefix to
   ~1-10, avoiding the combinatorial blowup of naive every-prefix-attends-
   every-suffix

Connects to `notes/agpt/prefix-to-suffix-attention.md` (design proposal) and
`notes/agpt/bayesian-bloom.md` §8 (dual-tree loop) — but **does not require
the corpus-walk** of §8. Matching is purely structural over the two radix
trees.

## Three matching policies (decided: max-overlap)

For a P-leaf π with edge content ending in `...XYZABC`, find S-leaves whose
edge content begins with the longest possible suffix of π. The "overlap"
between π and a candidate S-leaf σ is the max k such that π's last k chars =
σ's first k chars.

| Policy | Per-prefix candidate count | Notes |
|---|---:|---|
| **Tip-to-tip** (k=1: just last/first char must match) | ~N/V | ~N/65 candidates per prefix → millions for 5M corpus → infeasible |
| **Sliding** (all k≥1, weighted) | even bigger | combinatorial blow-up, low-quality matches dominate |
| **Max-overlap** (only the single max k per π, all S-leaves at that k) | ~1-10 typical | tractable AND informative — picked |

For natural language at d=32, max overlap is typically 5-15 chars (word/phrase
boundaries), so candidate sets are small. Verified empirically before
committing to kernel work.

## Critical: structural ≠ positional

§8's dual is **positional**: at corpus position p, the dual is (prefix at p,
suffix at p) — gives mass-1 P-leaves a single literal continuation
(memorization risk).

This proposal's dual is **structural**: any S-leaf whose first chars match
π's last chars by k+ characters is a candidate, regardless of whether the
corpus actually placed σ after π at any position. The model attends over the
plausibility menu, not the literal corpus continuation.

Trade-off: structural matching may dilute the actual corpus signal. Whether
that's a feature (regularization → better held-out) or bug (loses information)
is the empirical question this experiment will answer.

## Build plan (data side first, no model changes)

### Phase 1: Suffix-radix tree builder

Mirror of the existing `bin/agpt_build_radix_corpus` for the suffix tree.
Suffix tree built on the same Gutenberg corpus, depth-32, struct-of-arrays
compact representation. Same on-disk format as prefix tree.

Out: `bin/agpt_build_suffix_radix_corpus`, output to
`/home/trans/agpt-tries/gutenberg_5m_d32_suffix_radix_corpus/`.

Estimated effort: ~1 day (reuse `corpus_radix_builder.cr` pattern).

### Phase 2: Match-index construction

For each P-leaf π, compute its max-overlap S-leaf set. Implementation:

- For each P-leaf π (in mass-order, descending — biggest first for cache
  warmth):
  - Take π's edge content's last D chars (the "tail")
  - Walk the S-tree from root, matching characters of the tail against S-tree
    edges; the deepest match gives an S-tree node n_match
  - All S-leaves descending from n_match are π's max-overlap matches; record
    the depth k of n_match as the overlap length

Out: a static index file `match_index.bin` mapping P-leaf-id →
(overlap_k, [S-leaf-id list]).

Estimated effort: ~1 day.

### Phase 3: Distribution stats

Before any model work, measure:
- Distribution of overlap lengths k across P-leaves (mean, median, max, hist)
- Distribution of candidate-set sizes |M(π)| (mean, median, max, hist)
- Coverage: what fraction of P-leaves get ≥1 match?
- Cache feasibility: total bytes if we precompute K/V per S-leaf at d_model=64

Out: `analysis/match_stats.md` and plots.

Estimated effort: 0.5 day.

### Phase 4 (gated by Phase 3 results): attention layer + gradient

Per the design note's Option B (compute K/V on-the-fly): no per-leaf K/V
table. Per attention call, encoder pass produces K/V for the active candidate
set.

Two encoder choices:
1. **Reuse AGPT forward** for the suffix's edge tokens (simplest, one forward
   per S-leaf candidate per query)
2. **Separate small encoder** biased for "starts-with" structure (more
   complex, possibly faster to converge)

Start with (1).

Loss: mass-weighted softmax CE over candidates. The first char of the chosen
suffix is the predicted next token — standard CE on top.

Run alongside current attention as an **additional head**, not a replacement,
so we get clean A/B ablation against the deterministic AGPT baseline.

Estimated effort: ~3-5 days (CUDA kernel + gradient + integration).

## Success criteria

The pivot is justified if Phase 4 produces, on Gutenberg 5M d=32 d_model=64:

- **PPL < 25** (beats current ceiling ~29 by ≥4 PPL): structural prefix-to-
  suffix attention is providing real signal
- **PPL < 20** (beats current by ≥9 PPL): clear win, worth deeper investment
- **PPL ≥ 27**: marginal — investigate whether other variants (bigger encoder,
  positional duals, etc.) help

If Phase 3 reveals candidate sets routinely exceed ~50 per prefix, abort the
"max-overlap" policy and reconsider — either tighten to single-max or
investigate whether the radix structure naturally bounds it.

## Out of scope

- Positional duals (§8 corpus-walk) — defer until structural is measured
- §8 dual-tree training loop (the iterative refinement) — defer
- Replacement of current attention layer (vs additional head) — defer to
  ablation
- Inference-side cost analysis — relevant for production, not for this
  experiment's go/no-go decision
