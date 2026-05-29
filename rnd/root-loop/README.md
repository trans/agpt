# Root-Loop / Virtual-Tree Training (K > 1)

**Status**: complete — K=2 did not improve over K=1 baseline at d=16.

**Trainer note**: pre-fix, needs reassessment.

The negative result here was obtained on the pre-fix AGPT trainer. Keep the
design and failure analysis, but treat the quantitative comparison as needing
reassessment before using it as a current conclusion.

**Code**: branch `agpt-root-loop` (never merged). Key commits: bbb090d..ce4cc3e.

## Hypothesis

Attach a copy of the D-trie at every depth-D leaf of the D-trie, creating a
virtual tree of depth K·D. Train the transformer over longer sequences
(seq_len = K·D) while preserving the D-trie as the only stored structure.
Segment 2 attaches at segment 1's leaf — "leaf-to-root stitching."

If the model benefits from exposure to longer training sequences, K=2 should
improve PPL.

## What we built

- CLI flag `--virtual-cycles K` on `bin/agpt_train`.
- Per-root-child prior (16 tokens for K=2 D=16) derived via Markov-1-greedy
  walk from the D-trie.
- Negative-position RoPE cache so prior K,V sit at RoPE positions
  -(K-1)·D..-1 without re-projecting ancestors. Attention-relative-position
  identity means the query↔prior relative positions come out correct.
- Per-layer prior K,V mini-forward at each subtree entry; stop-gradient.
- KV cache extended with prior slots at the tail.
- Cycle-k>1 forward/backward duplicated on each chunk; gradients accumulate
  into one RMSProp step per subtree.

## What we tried as "fixes" when K=2 didn't help

1. Rolling-D target instead of Mj (segment-relative). State_index lookup,
   prior can enter target via boundary-spanning D-windows.
2. Mj fallback when rolling-D lookup missed.
3. Intra-edge extension to state_index (later reverted — was corrupting
   lookups by returning wrong-depth distributions).
4. Corpus-sampled priors (per-subtree). Didn't substantially help — per-subtree
   prior matches only one radix node in the subtree, not most.
5. K-averaging via 1/K scaling of d_d_logits so K=2 matches K=1 per-step
   update magnitude.

## Results (d=16 per-subtree, 3 super-epochs, rmsprop+warmup-cosine+entropy)

| Config | Mean PPL (n=3) |
|---|---|
| K=1 baseline | 13.66 |
| K=2 Mj target | 13.51 (tied within noise) |
| K=2 rolling-D + Mj fallback | 14.61 (worse) |

Conclusion: leaf-to-root with segment-relative targets doesn't give the prior
any gradient signal (target is blind to the prior by construction). Rolling-D
"fixes" all introduced more issues than they solved because the synthetic
prior rarely forms real corpus D-grams at the segment boundary.

## What's in the codebase (branch `agpt-root-loop`)

- `--virtual-cycles K` flag (plumbed)
- `--blend-alpha F` flag (blending — moved to its own experiment, see [../blending](../blending/))
- `AGPT_NOPRIOR=1` env var (diagnostic: prior_extend=0 at cycle k>1)
- `--corpus PATH` flag (corpus-sampled priors, partial)

## Related docs

- [notes/trainer/root-loop-implementation-plan.md](../../notes/trainer/root-loop-implementation-plan.md) — Phase 2 spec
- [notes/trie-structure/full-d-state-continuation.md](../../notes/trie-structure/full-d-state-continuation.md) — alternative design (not yet built)
