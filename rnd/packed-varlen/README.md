---
title: Packed variable-length forward
kind: experiment
status: concluded
outcome: negative
question: >-
  Does packing deep unary-chain segments into one variable-length forward per group (a two-regime
  trainer: sibling-grouped shallow depths, packed deep chains) speed up the Crystal trie-walk
  trainer without changing the update cadence?
answer: >-
  No speedup. Packing per root child kept the update cadence (training loss 3.0370 vs 3.0858
  baseline at seq_len=32, 20k starts) but ran 8% slower (3505 s vs 3252 s). Batching across
  root children at each depth is worth more than batching along chains. The fused cross-root-child
  packed attention kernel this would need was deferred.
opened: 2026-04-15
updated: 2026-04-15
code: {branch: agpt-packed-varlen, tag: exp/packed-varlen}
eval: none
headline:
- {label: 'baseline trainer, seq_len=32, 20k starts', metric: wall-clock seconds (commit 9019fe8),
  value: 3252}
- {label: 'two-regime per-rc packed deep, D_branch=21', metric: wall-clock seconds (commit
    9019fe8), value: 3505}
tags: [trainer, trie-structure, cadence]
---

# Packed variable-length forward

This branch tried to speed up the Crystal trie-walk trainer by packing variable-length
segments. `forward_segments` concatenates many unary-chain segments into one
`[positions, d_model]` batch, so projections, LayerNorm and FFN run as single matmuls while
attention stays per-segment. A two-regime trainer (opt-in `AGPT_TWO_REGIME=1`;
`AGPT_D_BRANCH` sets the split depth) keeps the sibling-grouped forward for shallow depths
and packs deeper chains per root child, so each (depth, root child) still gets one update.
At seq_len=32 with 20k starts this matched the baseline loss (3.0370 vs 3.0858) but ran 8%
slower (3505 s vs 3252 s), so there was no speedup. The fused cross-root-child packed
attention kernel it would need was deferred.

Code: branch `agpt-packed-varlen`, tag `exp/packed-varlen`. Key files are
`src/agpt/batched_depth_forward.cr` (`forward_segments`, D_branch helpers),
`src/agpt/trie_walk_trainer.cr`, `src/agpt/trie_corpus.cr` and
`spec/agpt_chain_compression_spec.cr` (a parity spec against the per-node path).
