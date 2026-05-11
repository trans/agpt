# AGPT Restart / Triage Plan

*Snapshot: 2026-04-28*

This note is for getting back to productive work without pretending the repo
is cleaner than it is.

## Current reality

### 1. The core codebase is alive

- Crystal-side specs pass with:

  ```sh
  CRYSTAL_CACHE_DIR=/tmp/crystal-cache just test-crystal
  ```

- Result on 2026-04-28: **6 examples, 0 failures, 1 pending**.
- `src/`, `notes/`, and the AGPT-specific tools are still coherent enough to
  build and reason about.

### 2. The AGPT / µGPT split is still incomplete

- AGPT depends on the `microgpt` shard in `shard.yml`.
- The `Justfile` still builds AGPT against `lib/microgpt/src/cuda/kernels.cu`
  and builds `bin/microgpt` + `bin/perplexity` from the shard for comparison.
- The foundational shell test still carries assumptions from the pre-split
  era, which is a good proxy for the broader state: the split happened
  structurally, but not yet fully operationally.

### 3. The main disorder is research accumulation, not source-code collapse

- `rnd/` is **1.2 GB**.
- Under `rnd/` there are **544 log files** but only **14 top-level README /
  findings docs**.
- Several experiment areas are narrative and reproducible:
  `p2s-attention`, `wrap-around`, `gutenberg-5m`, `lightning-training`,
  `post-fix-baseline`, `root-loop`, `blending`, `sparsity-profile`.
- Several others are mostly raw output with little or no front-door summary:
  `agpt-optimizers`, `hotspot-curriculum`, `sgd-ceiling`.
- There is also path drift: `rnd/rnd/sgd-sanity-check` exists while
  `rnd/README.md` still lists `sgd-sanity-check` as if it were a direct child.

### 4. The latest research direction just closed

- `rnd/p2s-attention/` was closed on **2026-04-27**.
- Its headline says:
  - direct transformer baseline on Gutenberg 5M: **PPL 6.50**
  - inference-time structural mask: **PPL 6.35**
  - the original cross-attention architectural ambition did **not** survive
    investigation
- Translation: there is a fresh result, but not yet a clearly chosen follow-on
  program for AGPT itself.

### 5. There is active uncommitted trainer work

- `src/cuda/agpt_train.cu` is modified on `main`.
- The diff introduces depth-routed K/V gradient handling and
  decision-only-loss controls (`AGPT_DEPTH_ROUTE_*`, `AGPT_DECISION_*`).
- `rnd/gutenberg-5m/logs/` also has new untracked outputs.

This is important because cleanup should not erase the fact that there is a
live hypothesis in flight.

## Recommendation

Do **not** start with a large repo-wide cleanup. First reduce ambiguity about
what counts as current work.

### Phase 1: Declare one active thread

Pick exactly one of these as the current frontier:

1. **AGPT trainer hypothesis**
   Work the new `agpt_train.cu` changes:
   depth-routed K/V gradients, decision-only loss, radix-cap / d* ideas.

2. **P2S follow-up hypothesis**
   Treat `p2s-attention` as the new lead:
   direct training + structural inference prior, without reviving the failed
   cross-attention design.

3. **Repo split / infrastructure hypothesis**
   Finish the AGPT/µGPT separation so experiments stop paying an ambient
   complexity tax.

If more than one is "active", none of them is active enough.

### Phase 2: Bring the research index back in sync with reality

Minimal target, not a perfection project:

1. Update `rnd/README.md` so every first-level experiment directory is either:
   `active`, `closed`, `archived`, or `log-only`.
2. Add a 10-line `README.md` to each log-only directory:
   `agpt-optimizers`, `hotspot-curriculum`, `sgd-ceiling`.
3. Decide whether `rnd/rnd/sgd-sanity-check` should be promoted, renamed, or
   explicitly archived as a stray historical path.

This will do more for clarity than moving files around.

### Phase 3: Make the active thread reproducible

Once Phase 1 chooses the thread:

- If the active thread is **AGPT trainer work**:
  write one short experiment note that states:
  hypothesis, flags/env vars, corpus, expected metric, and stop condition.

- If the active thread is **P2S follow-up**:
  write one note that states what is still open after the 2026-04-27 closure,
  because right now the findings are strong but the next experiment is not.

- If the active thread is **repo split / infrastructure**:
  close the known operational leaks first:
  test assumptions, CLI ownership, and result/log placement.

### Phase 4: Only then prune artifacts

After the above:

- Move non-essential checkpoints and bulky generated outputs out of the
  narrative experiment directories, or explicitly mark them as canonical.
- Keep the writeups in-tree; move only what is expensive and reproducible.
- Avoid deleting logs until each experiment has a readable summary.

## What I would do next

If continuing from today's state, I would choose:

1. **AGPT trainer hypothesis as the active thread**.
2. Create one focused experiment note for the current `agpt_train.cu` work.
3. Update `rnd/README.md` plus add the three missing log-only READMEs.
4. Only after that, decide whether `p2s-attention` is a side branch or the
   new main direction.

Reason: there is already live code in `src/cuda/agpt_train.cu`, and that is
the most likely place where "bit rot" turns into silent confusion if it is not
named and boxed in.

## Anti-goals

- Do not rewrite all historical notes into one grand document.
- Do not merge experiment conclusions just because they share vocabulary.
- Do not start deleting logs before the experiment index is readable.
- Do not treat the split from µGPT as solved just because the shard dependency
  compiles.
