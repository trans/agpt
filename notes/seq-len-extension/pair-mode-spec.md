# Pair-mode AGPT training: spec

Break the d=seq_len ceiling by training over joined parent-child windows
of a small-d trie. The trie stays dense (real AG events at depths 1..h);
RoPE positions span 1..2d to give the transformer effective seq_len=2d
of context. One Adam step per pair, gradients accumulated over both
halves.

## Setup

- **Trie**: build at small d (d=12 or d=16). Dense: branching to depth
  ~9 (d=12) or ~11 (d=16); cap edges short (~3 chars).
- **Pair**: 2d-char corpus window. `corpus[i .. i+d-1]` is the parent
  walk; `corpus[i+d .. i+2d-1]` is the child walk. Parent's last cap
  bridges into child's root walk via the corpus's actual continuation.
- **RoPE**: positions 1..2d. Parent half occupies 1..d; child half
  d+1..2d. Transformer attention is causal across the full window;
  child positions can attend back into the parent half.

## Per-position target

At every position `k` in the joined window, the loss target comes from
the d=12 trie's walk of the trailing-d chars at that position:

- Position `k = 0..d-1` (parent half): trie walk of `corpus[i..i+k-1]`
  (depth `k`). For `k=0`, target is the prior context (assume corpus
  start; in practice the first window of training is a special case).
- Position `k = d..2d-1` (child half): trie walk of
  `corpus[i+d..i+k-1]` — the **child's own** trailing d-char context,
  which is shorter than d for early child positions but reaches full
  d-char depth by the end.

In all cases the target is whatever the trie node says: KL on counts
if branching, one-hot CE on next char if mid-cap.

The cross-attention from child positions back into parent positions
is what the model has to *learn* — that's the new behavior pair-mode
introduces.

## Why this can break the ceiling

Single-mode AGPT at d=12 gives the model 12 chars of effective
context (capped by RoPE allocation). Pair-mode with RoPE 1..2d gives
24 chars of effective context, while every per-position target still
comes from the dense d=12 trie. So the model gets:
1. Richer per-position distributions than SGD (~5-10% KL events vs SGD's 0%)
2. Twice the context window of d=12-alone (24 chars vs 12)
3. But still trained against trie-derived signal (no PPL ceiling from
   data scarcity at long context)

## One Adam step per pair

User's call: keep parent loss live, accumulate gradients over both
halves, ONE Adam step at end. Same compute pattern as a regular AGPT
chunk, just with chunk = 2d positions instead of d.

## Mass-bias for the corpus-true path

At cap-edge positions (where the trie target is one-hot), the corpus
path is deterministic by construction. Up-weight the mass at these
positions to bias training toward the actual chain. The cap's count
(=1 for unique caps) becomes an explicit weight knob — caps that
appear N times in corpus get N× mass at their tunnel positions.

This isn't enrichment-of-one-hots-into-distributions (that failed in
the virtual-tree experiment). It's just letting the model spend more
gradient on positions where the corpus tells us the truth precisely.

## Implementation paths

Two viable architectures, increasing in scope:

### Path A: Modify microgpt with trie-derived per-position targets

Smallest diff. The existing microgpt trainer does SGD with one-hot CE.
Extend it to accept a per-corpus-position target side-table (tokens +
probs sparse), KL-against-trie-distribution at positions with non-empty
slot, fall back to one-hot CE elsewhere. Trie targets pre-computed
offline.

- **New tool**: `agpt_build_pair_targets` — for each corpus position
  i ≥ d, compute trie node from `corpus[i-d..i-1]` walk, store top-K
  of its counts as the target for position i. ~1M corpus positions × 16
  top-K × 6 bytes ≈ 100 MB side-table.
- **microgpt patch**: load side-table, swap one-hot CE for trie KL at
  positions that have a non-empty slot.
- **No trainer surgery on agpt_train.cu** — pair-mode is just SGD with
  a richer target side-table. RoPE positions 1..seq_len are automatic
  (microgpt does seq_len=24 directly).

This is actually the cleanest version. The "pair" concept reduces to
"train microgpt at seq_len=24 with d=12-trie-derived targets."

### Path B: Extend agpt_train.cu pair-chunks

Larger diff. Build pair-mode chunks (parent-cap-prepended + child-walk)
inside the AGPT trainer. K/V cache spans both halves. Native AGPT
chunking + ancestor handling extended to pair structure.

Path A wins for POC speed. Path B might be needed if Path A's
SGD-with-richer-targets framing turns out to be too SGD-like and
loses the AG benefits of structured-batch training.

## Falsifiable success criteria

1. Build pair-mode (Path A) at d=12, train at seq_len=24.
2. Compare to:
   - SGD at seq_len=24 (one-hot CE only)
   - AGPT d=12 single-mode at seq_len=12
   - AGPT d=24 single-mode at seq_len=24 (denser tree comparison)
3. Pass criteria:
   - Pair-mode at seq=24 beats SGD at seq=24 (richer targets help)
   - Pair-mode at seq=24 ≥ AGPT d=12 single-mode at seq=12 (longer
     context is at least no worse)
   - Pair-mode at seq=24 vs AGPT d=24 single-mode at seq=24:
     informative either way

## Risks

- **Most positions land at cap edges** (one-hot, same as SGD). KL
  events at branching positions are few. Pair-mode might end up
  numerically equivalent to SGD-with-occasional-distillation — small
  improvement at best.
- **Cross-attention from child to parent might not learn meaningful
  patterns**. If the model just uses the trailing d chars as effective
  context (ignoring deeper parent attention), pair-mode = SGD
  at seq_len=2d.
- **Trie targets at long context aren't fundamentally richer than
  immediate trie walks**. Each position's target depends only on its
  own trailing d chars, not on the joined 2d window. So the model
  gets 2d-context attention but per-position targets are 1-d-context
  marginal predictions.

## Open question

Whether the "live parent loss" mass-weighting + cap-bias is enough to
make pair-mode genuinely AG-distinct from SGD-at-seq=2d, or whether
breaking the d=seq_len ceiling requires a more fundamental
mechanism (the topological-navigation attention bridge from
notes/prefix-suffix/topological-navigation.md).
