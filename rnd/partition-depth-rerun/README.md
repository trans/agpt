---
title: Partition-depth rerun
kind: experiment
status: concluded
outcome: positive
question: >-
  What does the partition-depth record's best recipe (Adam pd=6, 3 passes, depth 32 on
  Shakespeare, legacy PPL@32 3.82, 2026-04-30) score on held-out text under the canonical
  evaluator adopted on 2026-05-25?
answer: >-
  4.89 and 4.95 canonical rolling byte PPL on the 10 held-out chunks (two repeats), with the
  code of that day. The rebuilt toolchain reproduces the record (3.836 on its own trie and
  scorer, against 3.82). Held-out text barely moves the old score: the same checkpoints give
  3.80-3.82 on the start of data/input.txt and 3.97 on the held-out chunks at a fixed 32-character
  window. The step from 3.97 to 4.9 is the scorer: lm-eval's rolling windows give each
  prediction 0-31 characters of context instead of a full 32.
opened: 2026-11-18
updated: 2026-11-18
code: main
eval: canonical
family: attention
headline:
- label: 'Adam pd=6, 3 passes, depth 32, the 2026-04-30 code, canonical Shakespeare split, repeat 1'
  metric: rolling byte PPL (canonical, the 10 Shakespeare held-out chunks)
  value: 4.8891
  run: 20261118T172913-pd6-adam-r1
- label: 'Same, repeat 2'
  metric: rolling byte PPL (canonical, the 10 Shakespeare held-out chunks)
  value: 4.9465
  run: 20261118T174540-pd6-adam-r2
- label: 'Same checkpoints, fixed 32-character window on the held-out chunks (repeat 1)'
  metric: agpt_fixed_token_perplexity (the 10 Shakespeare held-out chunks)
  value: 3.9754
  run: 20261118T172913-pd6-adam-r1
tags: [partitioning, evaluation, reproduction, shakespeare]
related: [partition-depth, refined-trie-counts, one-pass-gap]
---

# Partition-depth rerun

The user, 2026-11-18, after I set the refined-counts runs' 8.98 (PG-19,
depth 8, 2 passes) beside `partition-depth`'s 3.82: "I don't recall it
ever being that bad", and that the old work used held-out text. The record's
3.82 is legacy PPL@32 (`bin/perplexity`), from before the switch to
lm-evaluation-harness on 2026-05-25, and its sweep script scores the start
of `data/input.txt`, the text the trie was built from (its logs were not
kept). This puts the recipe on today's scale. "Yes, queue the Shakespeare
pd=6 rerun."

## Setup

- **The code of that day.** `build_old.sh` rebuilds `agpt_train` and the
  trie builder at `a8ddb15` (2026-04-29, the last commit before the sweep,
  with every flag the sweep used) against the microgpt kernels its
  `shard.lock` pins (`bdd8e82`), with that commit's own build lines (nvcc
  13.4 here). v1 has changed a great deal since (the ancestor gradient,
  2026-05-19/20; new normalizers and defaults, 2026-05-26; the barrier fix,
  2026-09-25), so today's v1 would not be the recipe.
- **The recipe**, `rnd/partition-depth/README.md`'s Adam result: init
  `data/input.random.model` (d64/L2, h4, d_ff 256; the record's "d=32" is
  the depth), trie depth 32, `--epochs 3 --lr 1e-3 --optimizer adam
  --lr-schedule warmup-cosine --warmup-epochs 1 --entropy-lambda 1.0
  --mass-weight log --no-accumulate --partition-depth 6`.
- **`old_v1_trainer.py`** lets `bin/agpt_experiment` drive the old trainer:
  it maps the config onto those flags, builds the old-format trie from
  `corpus.path`, and sets the checkpoint header's seq_len to 32. The old
  trainer keeps the init's 128 though it trains contexts of at most 32; the
  model uses RoPE, so seq_len only sets the evaluation window (the sweep
  scored at `--seq-len 32`). The old trainer has no seed: the two repeats
  differ by GPU nondeterminism only, as the sweep's did.
- **Three steps** (`run.sh`, slot 91, 2026-11-18, 17:12–18:01 UTC, log
  `grid.out`):
  1. Toolchain check (`configs/repro.yml`, run directly): the sweep's own
     trie (all of `data/input.txt`, built 2026-04-28 and unchanged), scored
     as the sweep scored it.
  2. The recipe on the canonical Shakespeare split
     (`data/.splits/4fa9aec1db6b3aea`, 10 held-out chunks of 5,576
     characters), twice, through `bin/agpt_experiment`.
  3. Each checkpoint also scored the old way: 8,192 positions at a fixed
     32-character window, on the start of `data/input.txt` and on the
     held-out text, by `bin/perplexity` and by `src/tools/agpt_ppl.py`
     (the PyTorch model the HF wrapper is built on).

## Results

| run | trained on | old scorer, start of `data/input.txt` | fixed window, held-out | canonical rolling, held-out |
|---|---|---:|---:|---:|
| record (2026-04-30) | all of `data/input.txt` | 3.82 | – | – |
| repro (toolchain check) | all of `data/input.txt` | 3.836 | (in training) | – |
| pd6-adam-r1 | canonical split | 3.801 | 3.975 (3.701) | **4.889** |
| pd6-adam-r2 | canonical split | 3.821 | 3.970 (3.681) | **4.947** |

"Fixed window, held-out" is `result.json`'s `agpt_fixed_token_perplexity`
(every position of each chunk, scored within its chunk); in brackets
`agpt_ppl.py`'s 8,192 positions of the concatenated held-out file from
position 32. `bin/perplexity` can't score the held-out file: it builds its
vocabulary from the file it scores (62 characters there against the
model's 65) and stops with a config mismatch. On `data/input.txt` the two
legacy scorers agree to 1e-4 on every checkpoint (and on the random init,
180.2588 against 180.2589), so the canonical eval reads the old
checkpoints as the old scorer did.

The toolchain check ran 283,309 groups a pass, the record's count for pd=6,
in 185 s a pass (the record: 643 s for 3 passes; here 942 and 938 s with
the trie build and loading).

## Reading

- **The record reproduces.** The 2026-04-30 code, trie and recipe give
  3.836 against the record's 3.82 (its repeats spread about 0.05).
- **Held-out text barely moves the old score.** Trained without the
  held-out chunks, the same checkpoints score 3.80–3.82 on the start of
  `data/input.txt` (in their training text) and 3.97 on the held-out chunks
  at the same fixed window, 4% higher. The record's 3.82 was not inflated by
  scoring training text to any large degree.
- **The scorer is the difference.** On the same held-out chunks the
  canonical rolling score is 4.89–4.95 against 3.97 at a fixed window.
  lm-eval's `loglikelihood_rolling` cuts the text into windows of the
  model's 32 positions, so a prediction gets 0–31 characters of context, 16
  on average; the fixed window gives every prediction all 32. Same model,
  two protocols: the pre-2026-05-25 rule that legacy numbers are not
  compared with canonical ones is about exactly this.
- **Where the recipe stands.** 4.89–4.95 in 3 passes, canonical. For
  reference, at depth 16 on the same split (`rnd/one-pass-gap`, same model):
  AGPT pd=3 scores 7.81 after one pass, and minibatch Adam (128 windows a
  step) 4.83 after one pass of every window start and 4.53 after eight.
  The rolling score depends on the window, so depth 32 and depth 16 are not
  strictly comparable.
