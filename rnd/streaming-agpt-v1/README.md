# Streaming AGPT v1 — 5-checkpoint linear cadence

**Status:** planning (started 2026-05-16)
**Design doc:** `notes/streaming_agpt.md`

## Goal

Test whether streaming training — building the trie incrementally and
training between checkpoints — gives a PPL improvement over standard
AGPT trained once on the full-corpus trie at matched SE budget.

The mechanism this tests: AGPT's lack of catastrophic forgetting (the
trie permanently encodes corpus history) should let us train with
many more effective optimizer steps without the "recency bias"
problem SGD has.

## Variants

### Streaming
- 5 checkpoints: 20%, 40%, 60%, 80%, 100% of corpus
- At each, build trie from corpus prefix, train for 20 SE continuing
  from previous model state
- Total = 100 SE across stages

### Baseline
- Build full-corpus trie
- Train for 100 SE in one run
- Same SE budget; same recipe

## Setup

- **Corpus:** Shakespeare 1M (`data/input.txt`)
- **Vocab:** full-corpus charset (verify all 65 chars present from
  20% onward)
- **Trie depth:** d=16 (smaller, faster iteration)
- **Model:** d_model=64, n_layers=2, n_heads=4, d_ff=256 (current standard)
- **Recipe:** rmsprop lr=3e-3, **constant LR** (no warmup/cosine —
  see confounds section), mass-weight=log, entropy-lambda=1.0,
  --no-accumulate, --partition-depth 1
- **Eval:** PPL@16 against `data/input.txt`, max-positions 4096

## Pass / fail

| | PPL@16 expectation |
|---|---|
| Baseline 100 SE | establish first (TBD; per memory pd=1 d=32 SE=110 plateau is 4.46 PPL@32 on Shakespeare; d=16 ought to be in similar range) |
| Streaming 5×20 SE | should be ≤ baseline if hypothesis holds |

Pass: streaming improves on baseline.
Strong pass: 5%+ PPL improvement.
Fail: streaming regresses, indicating broken carry-over or that more steps don't help.

## Confounds verified before running (2026-05-16)

1. **Vocab consistency: confirmed problem.** First 20% of Shakespeare
   has only 62 of 65 chars (missing some lowercase letters). Mitigation:
   pass `--vocab-file data/input.txt` (full corpus) to the trie builder
   at every stage. The trie's CharDataset will be built from the full
   corpus, so token IDs match across stages.

2. **Optimizer state carry-over: NOT supported.**
   `save_model_weights` in `agpt_train.cu` only writes model weights,
   not Adam/RMSprop moment buffers. Each stage starts with cold
   optimizer state.

   Implications:
   - Streaming has 5 cold-restart optimizer warmups vs baseline's 1
   - Warmup-cosine schedule would restart 5× under streaming,
     unfairly hurting it
   - **Mitigation:** use **constant LR** for both streaming and
     baseline so LR schedule isn't a confound. Lower expected absolute
     PPL but matched conditions.
   - Adam β₂=0.999 needs ~1000 steps to warm. AGPT at pd=1 fires 65
     steps/SE → 20 SE = 1300 steps, so β₂ warms within a stage.
     Acceptable.
   - The "fix optimizer state save/load" task is parked as a
     followup.

3. **Per-stage runtime imbalance.** Stage 1 (20% trie) runs faster
   than stage 5 (100%). Total wall time of streaming may exceed
   baseline. Capture and report wall time alongside PPL.

## Orchestration

`run.sh` — bash script that:
1. Truncates corpus at each checkpoint % (already-tokenized full
   corpus is hashed for the trie; need to verify the trie builder
   can accept a truncated text file).
2. Builds trie for each checkpoint via `bin/agpt_build_radix_corpus`.
3. Runs `bin/agpt_train` for each stage, chaining models.
4. Runs `bin/perplexity` to score each stage's model.
5. Runs the baseline (full-corpus 100 SE) for comparison.
6. Logs everything to `rnd/streaming-agpt-v1/logs/`.

## Followups if pass

- Test on Gutenberg 5M
- Test at d=32
- Test finer cadence (10, 20, 50 checkpoints)
- Combine with sliding-tree RoPE for unified streaming + position-aware
  training

## Followups if fail

- Investigate optimizer state carry-over
- Test with truly orthogonal seeds at each stage (no carry-over) to
  isolate the "growing trie" effect from the "more steps" effect
- Compare against per-stage cold-start training to bound the lift
