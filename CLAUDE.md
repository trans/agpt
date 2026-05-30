# AGPT — agent guidance

## Running experiments

**Reportable runs go through `bin/agpt_experiment`.** The orchestrator
binds config, checkpoint, logs, and `result.json` into one immutable
per-run directory with full provenance (git SHA, corpus SHA-256, init
model header, environment fingerprint, etc).

Direct invocation of `bin/agpt_train_v2`, `agpt_ppl.py`, or
`agpt_sliding_window_perplexity` is still fine for smoke tests, kernel
debugging, and one-off diagnostics — just don't treat the resulting
numbers as reportable. Anything you'd cite in a memory, a doc, or a
chat update needs a `result.json` behind it.

Workflow:

```sh
just build-agpt-experiment            # one-time
bin/agpt_experiment --config FOO.yml  # creates rnd/<exp>/<run-id>/
```

A run directory contains (committed files in **bold**):
- **`config.yml`** — the input (verbatim)
- **`resolved_config.json`** — input + defaults applied
- **`meta.json`** — provenance (git sha/branch/dirty, corpus sha,
  init model header, environment, full command line)
- **`result.json`** — the canonical numbers (with `eval.split` recorded
  so a number is never ambiguous about what it measured)
- **`eval_raw.json`** — full lm-eval output (small)
- `train.log` — trainer output (gitignored raw debug trace; force-add only
  for a specific postmortem)
- `eval.log` — verbose tqdm/eval output (gitignored, large)
- `checkpoint.model` — native trainer checkpoint (gitignored, regenerable)
- `hf_checkpoint/` — HF-format model (gitignored, regenerable from
  `checkpoint.model` + vocab via `agpt_hf.py convert`)
- If `corpus.train_frac < 1`: `train_corpus.txt` + `heldout_corpus.txt`
  (typically gitignored — the offset is recorded in `result.json` and
  the slices are reproducible from `corpus.path`).

**A run is canonical only after `result.json` exists with non-empty
metrics.** A run dir without `result.json` is incomplete (crashed,
killed, or in progress); incomplete dirs can be `rm -rf`'d freely
without ceremony — nothing references them.

**Changing a setting = new run directory.** Finished run dirs are
immutable. Editing `config.yml` or re-running with the same hash is
rejected (exit code 5).

Sample YAML: `configs/_test/smoke.yml`. Spec:
`notes/operations/experiment-runner-design.md`.

### Eval splits

Every config must declare `eval.split` explicitly. Values:

- `train` — evaluator gets the training slice (or full corpus if
  `corpus.train_frac == 1`). Useful for sanity checks; not a
  generalization metric.
- `tail-heldout` — evaluator gets the last `(1 - train_frac)` of
  `corpus.path`. Requires `corpus.train_frac < 1`. This is the default
  "held-out PPL" for AGPT runs.
- `external-heldout` — evaluator uses `eval.external_file`, a corpus
  the model never saw during training (e.g. a different book or
  different author).
- `benchmark` — evaluator runs a built-in lm-eval task (e.g. `wikitext`)
  via `eval.benchmark`. Comparable to published numbers, less aligned
  with our training distribution.

Numbers from different splits are NOT comparable. `result.json` records
the split + source SHA so this can't be confused later.

## Canonical PPL

`byte_perplexity` reported by `lm-evaluation-harness` against the
HF-wrapped AGPT model. The orchestrator writes this to `result.json`.

Numbers from `bin/perplexity`, `bin/agpt_sliding_window_perplexity`, and
`src/tools/agpt_ppl.py` are **legacy**. They use different rolling
protocols and are NOT directly comparable to the new canonical. Treat
PPL numbers from before 2026-05-25 as legacy-canonical; re-evaluate
under `bin/agpt_experiment` before comparing.

Standalone eval (when not training): convert .model to HF then run
the driver:

```sh
python3 src/tools/agpt_hf.py convert \
    --model CHECKPOINT.model --vocab-file CORPUS.txt --out HF_DIR
python3 src/tools/agpt_lm_eval.py \
    --hf-dir HF_DIR --text-file CORPUS.txt
```

## Builds

Crystal/CUDA builds go through `just` (never `crystal build` directly,
never `nvcc` directly). See `Justfile`.

## Memory + research history

**Claude-specific:** per-project memory lives at
`~/.claude/projects/-home-trans-Projects-microgpt/memory/`. Index in
`MEMORY.md`. Other agents have their own persistence mechanisms; this
section is what Claude reads at session start.

Notable entries for orientation:
- `feedback_evaluator_consistency.md` — full canonical-PPL story
- `project_experiment_runner.md` — discipline commitment + design doc ref
- `feedback_persist_results.md` — runs must persist to disk before
  describing them

In-repo documentation (read by all agents):
- `notes/` — design docs. Codex has been reorganizing into subdirs
  (e.g. `notes/operations/`, `notes/seq-len-extension/`); follow
  references rather than assuming a flat layout.
- `todo/` — concrete next steps and parked work
- `rnd/<exp>/` — per-experiment dirs (orchestrator-generated going forward)
