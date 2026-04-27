# p2s-attention Phase 2/3 Findings (Match Index Distribution)

> **Status (2026-04-26): Phase 2/3 complete.** Structural max-overlap matching
> on Gutenberg 5M d=32 produces a strikingly tight match-set distribution.
> Architectural validation confirmed; greenlight for Phase 4 (attention
> kernel + gradient).

## Headline

| Metric | Value |
|---|---:|
| Prefix leaves total | 4,988,254 |
| With any match (k≥1) | **100.0%** |
| Mean candidate-set size | **1.49** |
| Mode overlap k | **21** out of max 32 |
| Wall-clock for full match-index build | 3:20 |
| Peak memory | ~6 GB (Crystal Array 2× capacity overhead; nominal 1 GB) |

## Match-set size distribution (over all prefix leaves)

| Candidates | % of prefix leaves |
|---|---:|
| 1 | **95.44%** |
| 2-4 | 2.89% |
| 5-19 | 0.94% |
| 20-99 | 0.73% |
| 100+ | <0.01% |

**95.4% of prefix leaves have exactly one structurally compatible suffix
candidate.** This is essentially the corpus-positional dual; the model gets
a deterministic continuation target for the vast majority of prefixes.
The 4.6% with multiple candidates are the genuinely-ambiguous prefixes
where attention has to choose — exactly the cases where the model is
expected to do useful work.

## Overlap k histogram (k = max chars shared between prefix-edge and suffix-edge)

```
k= 1:       1477 ( 0.03%)
k= 5:       4118 ( 0.08%)
k=10:      26853 ( 0.54%)
k=15:     162892 ( 3.27%)
k=17:     307459 ( 6.16%)
k=18:     404356 ( 8.11%)
k=19:     508285 (10.19%)
k=20:     597623 (11.98%)
k=21:     639179 (12.81%)  ← peak
k=22:     609886 (12.23%)
k=23:     505907 (10.14%)
k=24:     352572 ( 7.07%)
k=25:     194177 ( 3.89%)
k=30:         74 (  0.0%)
k=31:          1 (  0.0%)
```

The bell curve centered at k≈21 says: typical leaf-edge content is ~21
characters long, and the corpus has structural continuations matching
those 21 characters. This represents real linguistic structure (word
fragments, phrase patterns), not noise.

## What this validates

1. **Tractability claim from the design note** (`notes/agpt/prefix-to-suffix-attention.md`):
   "the candidate set is bounded by the leaf's mass" — confirmed.
   For Gutenberg, mean=1.49 is even tighter than expected.
2. **Structural matching is sufficient** — we don't need §8's corpus-walk
   dual-tree loop. Pure tree-on-tree string overlap gives matches that
   are essentially the corpus-positional dual for ~95% of prefixes.
3. **Per-step compute cost is O(N)** at training time — total attention
   work over an epoch is `Σ_leaves mean_match_size ≈ corpus_size × 1.49`.
4. **Build cost is one-time and modest** — 3:20 wall-clock for 5M
   corpus, ~6GB peak memory.

## What this doesn't yet measure

- **Whether the model trained on this signal beats the current PPL=29
  ceiling** — Phase 4 goal.
- **Whether the few-candidate cases are the "informative" ones** —
  expect yes, since these are exactly where the corpus has multiple
  continuations for the same prefix.
- **Inference-side cost** — our 1.49 mean is for training; at inference
  we'd query the match index on each token. Hash/trie lookup cost is
  negligible vs the model forward.

## Reproduce

```sh
just build-p2s-match
bin/agpt_p2s_match \
  --prefix /home/trans/agpt-tries/gutenberg_5m_d32_radix_corpus \
  --suffix /home/trans/agpt-tries/gutenberg_5m_d32_suffix_radix \
  --min-overlap 1 \
  --max-candidates 64 \
  --out /tmp/g5m_p2s_match.bin
```

The `--out` writes the binary match index with header `'P2SC'` (4-byte magic),
prefix_count + suffix_count + suffix_leaf_count (3 × int32), then per-prefix
records: `(prefix_id, max_k, num_sigmas, sigma_id × num_sigmas)` all int32.

## Implementation notes

- The head trie is a struct-of-arrays (4 int arrays per node + 2 for terminal
  list). Total nodes: 63.3M for Gutenberg, 1003 MB nominal in those arrays.
  Actual RSS hit ~6 GB due to Crystal `Array<Int32>` doubling on push;
  pre-sized `Slice`s would cut this to ~2 GB if needed.
- Matching loop is O(L²) per prefix leaf (try k from L down to 1, restart
  walk for each k). With L ≤ 32, total ~7.5B ops, ran in ~3 minutes.
  Aho-Corasick fail-link upgrade would make it O(L) per leaf if matching
  wall-clock matters; not the bottleneck today.
- 100% match coverage isn't surprising: every prefix leaf's last char
  appears somewhere as a "first char of a reversed suffix-tree leaf edge"
  in the head trie (i.e., as a char somewhere in the corpus). At minimum,
  k=1 always works.

## Phase 4 (next)

Build the actual attention layer. Per the design note's Option B:
- No per-leaf K/V table; compute on-the-fly from each candidate suffix's
  edge tokens via the AGPT forward pass
- Add as a separate head alongside current attention for clean A/B
- Target: PPL < 25 (vs current ceiling 29)

Estimated effort: 3-5 days for the CUDA kernel + gradient + integration.
