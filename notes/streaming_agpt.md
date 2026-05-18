# Streaming / Progressive AGPT Training

**Origin:** user idea 2026-05-16. The cleanest articulation of why
AGPT is fundamentally different from SGD on streaming data, and the
practical consequence.

## The Core Insight

SGD on streaming data suffers catastrophic forgetting: the model's
gradients are dominated by recent batches; older data, having been
seen and dropped, fades from the parameters. Continual-learning
research has documented this extensively.

**AGPT does not have this pathology.** Old corpus data does not "leave"
because it is permanently encoded in the trie's accumulated counts.
Every training event references the trie's current state, and the
trie's state at any node is the cumulative count over *all* corpus
positions that ever traversed it. As the trie grows, the target
distributions at high-mass shallow nodes stabilize; the model's
gradient signal at those nodes pulls toward the long-run empirical
distribution, not a recent window.

This means: we can train AGPT *while the trie is still being built*,
adding optimizer steps without inflating compute or losing
generalization to early-corpus content.

## What This Buys Us

**More optimizer steps per corpus pass.** Currently, AGPT trains by
walking the trie once per super-epoch (SE). Streaming inserts
training passes during trie construction, multiplying the effective
optimizer step count by the number of checkpoints used.

**A genuine AGPT-vs-SGD differentiator.** This is one of the cleanest
experiments to show that AGPT's mechanism (trie-target training) has
properties that SGD-style training fundamentally cannot match.

**Foundation for other extensions.** More optimizer steps means more
chances for things like sliding-tree (per-position RoPE diversity) to
hit good states. Streaming is *additive* with future work, not
exclusive.

## What This Doesn't Buy Us

- Doesn't address the seq_len decoupling problem directly. That work
  remains.
- Doesn't reduce wall time of a single training run; it just shifts
  *when* training happens.

## Mechanism

Define K checkpoint fractions of the corpus (initial v1: 5 evenly
spaced — 20%, 40%, 60%, 80%, 100%).

For each checkpoint c_i:

1. **Build trie** from `corpus[0 : c_i × N]` where N is total corpus
   length. Reuse the existing `bin/agpt_build_radix_corpus`.
2. **Train** the model on this trie for some SE budget, *continuing
   from the previous checkpoint's saved model state* (including
   optimizer state — AGPT's optimizer state is per-model-parameter,
   not per-trie-node, so it transfers cleanly).
3. **Save** the resulting model as the start for the next stage.

The final model has been trained K times, each round on a
progressively richer trie. Total optimizer fires = K × per-stage SE
× chunks-per-SE.

## Critical Implementation Concerns

### Vocabulary consistency

Each checkpoint's trie is built from a different corpus subset and
hence a potentially different `CharDataset` vocab. If vocab differs
across stages, model parameters become invalid.

**Mitigation:** always use the full corpus as `--vocab-file` (or
pre-build the dataset from full corpus and use that for vocab
encoding). The trie's structure is built from the corpus subset, but
the token-to-id mapping comes from the full corpus.

For Shakespeare 1M (65 chars distributed throughout): even the first
10% likely contains all 65 chars, but verify before relying on it.
Gutenberg 5M: 5 books concatenated, vocab consistency across
checkpoints is plausible but worth verifying.

### Model state carry-over

`bin/agpt_train --model PATH` loads weights from PATH. Whether
optimizer state (Adam/RMSprop moments) also loads depends on the
checkpoint format. **Verify:** does the current `MiniGPT.save/load`
save optimizer moments?

If not, optimizer state resets at each stage, which weakens the
"continuous" property of streaming. Worth fixing if needed.

### Trie format compatibility

Each stage's trie has a different `meta.bin` (different
corpus_token_count, total_edge_chars, etc.). Does `bin/agpt_train`
care? It should treat the trie as the source of truth for training
data; differing metadata across stages is normal.

### Stage compute imbalance

Earlier stages have smaller tries (less compute per SE). Later stages
have larger tries (more compute per SE). If we match SE budget per
stage, wall time is unbalanced. Either:

- Match SE per stage (simpler, total SE = K × per-stage budget)
- Match wall time per stage (more even compute distribution, harder to
  configure)

For v1, match SE per stage.

## Cadence Variants (deferred)

- **v1 — 5-checkpoint linear:** 20/40/60/80/100% checkpoints.
- **v2 — periodic cadence:** train every K corpus positions, e.g.,
  every 100k chars. For Gutenberg 5M with K=100k, that's 50 stages.
- **v3 — log-spaced cadence:** more checkpoints early when the trie is
  growing fast, fewer later. Captures early-stage curriculum dynamics.
- **v4 — adaptive cadence:** train when some "trie stability" metric
  indicates the targets have shifted enough. Avoids wasted compute on
  small incremental changes.

v1's simplicity is the right starting point. If streaming shows lift
at 5 checkpoints, finer cadence will be a sweep.

## First-Experiment Plan

### Setup

- **Corpus:** Shakespeare 1M (faster iteration; tries build in
  seconds). Move to Gutenberg 5M after first signal.
- **Trie depth:** d=16 (smaller, faster). Move to d=32 in follow-up.
- **Checkpoints:** 5 at 20/40/60/80/100% of corpus.
- **Per-stage SE budget:** 20 SE. Total streaming = 100 SE.
- **Baseline:** standard AGPT trained on full-corpus trie for 100 SE.
- **Recipe (both):** rmsprop, lr=3e-3, warmup-cosine, mass-weight=log,
  entropy-lambda=1.0, no-accumulate.

### Pass / fail

| condition | Shakespeare d=16 PPL@16 |
|---|---:|
| Baseline (100 SE full corpus) | known (~4.5-5 range; need to confirm exact) |
| Streaming (5 × 20 SE) | TBD |

**Pass:** streaming ≤ baseline, indicating the additional optimizer
steps + curriculum-like growth help.

**Strong pass:** streaming meaningfully under baseline (5%+ PPL
improvement). Confirms the "more optimizer steps without forgetting"
hypothesis.

**Fail:** streaming above baseline. Either the carry-over is broken,
or extra steps on growing tries don't help in the way we expect.

### Risks

- Optimizer state lost between stages → training restarts each stage,
  hurts convergence.
- Vocab mismatch between stages → undefined behavior, possibly silent.
- Trie at 20% is much smaller (fewer high-mass nodes) → first
  stage's gradient signal may be noise.
- Compute mismatch — streaming's total compute may exceed baseline's
  due to per-stage overhead. Need to measure.

### Tooling

- Orchestration: bash script `rnd/streaming-agpt-v1/run.sh` that
  loops through checkpoints, runs build + train, captures logs.
- No new C++/CUDA work required. Reuses
  `bin/agpt_build_radix_corpus`, `bin/agpt_train`, `bin/perplexity`.

## Files

```
notes/streaming_agpt.md             — this file
rnd/streaming-agpt-v1/
├── README.md                             — experiment plan
├── findings.md                           — results once we have them
├── run.sh                                — orchestration script
└── logs/
    ├── baseline_100SE.log
    ├── streaming_stage1_20pct.log
    ├── streaming_stage2_40pct.log
    ├── ...
    └── streaming_stage5_100pct.log
```

## Estimated Effort

- Orchestration script: ~half a day
- First experiment run: ~30 min (Shakespeare d=16, 100 SE total)
- Eval + writeup: ~1 hour

**Total to first result: 1 day.**

If positive, scale up: Gutenberg 5M, d=32, finer cadence. If
negative, root-cause investigation (vocab, optimizer state, etc.).

## Followup: Combine with sliding-tree

Streaming and sliding-tree (`notes/sliding_tree_rope.md`) are
orthogonal:

- Streaming: more optimizer steps via growing trie
- Sliding-tree: position-OOD fix via corpus-aware RoPE

Combined, the streaming loop iterates over corpus positions, training
the model at each position's actual corpus-RoPE. This is the natural
unified framework, but implementing them separately first lets us
attribute lift to the right mechanism.
