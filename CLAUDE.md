# AGPT — agent guidance

## Running experiments

All training + evaluation goes through `bin/agpt_experiment`. Do not run
`bin/agpt_train_v2` or any PPL tool directly for reportable results — the
orchestrator binds config, checkpoint, logs, and result.json into one
immutable per-run directory, with full provenance (git SHA, corpus
SHA-256, init model header, etc).

Workflow:

```sh
just build-agpt-experiment            # one-time
bin/agpt_experiment --config FOO.yml  # creates rnd/<exp>/<run-id>/
```

A run directory contains:
- `config.yml`, `resolved_config.json` — the input
- `meta.json` — provenance (git, corpus sha, init model header)
- `train.log`, `eval.log`, `eval_raw.json` — process output
- `checkpoint.model`, `hf_checkpoint/` — the trained model in both formats
- `result.json` — the canonical numbers
- aggregated into `rnd/<exp>/runs.json` and the table in
  `rnd/<exp>/README.md` (auto-regenerated)

**Changing a setting = new run directory.** Finished run dirs are
immutable. Editing config.yml or rerunning with the same hash is rejected
(exit code 5).

Sample YAML: `configs/_test/smoke.yml`. Spec:
`notes/experiment-runner-design.md`.

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

Per-project memory lives at
`~/.claude/projects/-home-trans-Projects-microgpt/memory/`. Index in
`MEMORY.md`. Notable entries:

- `feedback_evaluator_consistency.md` — full canonical-PPL story
- `project_experiment_runner.md` — discipline commitment + design doc ref
- `feedback_persist_results.md` — runs must persist to disk before
  describing them

Documentation:
- `notes/` — design docs (see `notes/seq-len-extension/` for active research)
- `todo/` — concrete next steps and parked work
- `rnd/<exp>/` — per-experiment dirs (orchestrator-generated going forward)
