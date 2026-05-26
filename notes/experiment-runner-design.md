# Experiment Runner Design

**Status:** spec, awaiting sign-off and CUDAX-work resolution before implementation.

## Why

Every "did X improve PPL?" investigation has cost hours that should have cost
minutes. The failure mode is consistent: config, eval command, checkpoint,
corpus split, logs, and result are not bound together as a single artifact.
A run produces a number with no machine-checkable provenance, and the next
session — Claude, Codex, or human — has to reconstruct what was actually done.

This spec defines a small infrastructure that makes runs **immutable, fully
provenanced, and tabulated automatically**. Configs are YAML. Each run is a
directory. PPL numbers come from `result.json`, never from a hand-written
table.

## Directory layout

```
rnd/<experiment>/                          # one experiment = one hypothesis
  README.md                                # hypothesis, scope, table (generated)
  runs.json                                # cached aggregate of all run results
  <run-id>/                                # one config = one run
    config.yml                             # the input — human-edited
    resolved_config.json                   # config.yml + defaults applied
    meta.json                              # git_sha, command, corpus_sha, ...
    train.log                              # tee'd stdout/stderr of trainer
    eval.log                               # tee'd stdout/stderr of evaluator
    checkpoint.model                       # final trained model
    result.json                            # canonical numbers
```

Rules:
- A `<run-id>` directory is **immutable** after the run completes. To change a
  setting, create a new run-id.
- `README.md` is hand-edited prose (hypothesis, scope, conclusion) but the
  results table inside it is regenerated from `runs.json`.
- `runs.json` is a single-file aggregate of all `<run-id>/result.json` under
  the experiment, kept in sync by the orchestrator.

### Run-id format

`<UTC-stamp>-<slug>`, e.g. `20260525T1430-asym-dft-W64-HD48`. Stamp gives
uniqueness + lex sort; slug aids reading.

If a config has been run before (same SHA-256 of `resolved_config.json` plus
same corpus SHA-256), the orchestrator should refuse and point at the existing
run, rather than silently re-running.

## YAML schema (v1)

Top-level keys mirror the two stages:

```yaml
# config.yml — example for a training+eval experiment
meta:
  description: "Asymmetric harmonic filter, DFT W=64, HD=48, Shakespeare prefix-95"
  experiment: harmonic-filter
  hypothesis_ref: notes/seq-len-extension/harmonic-filter-asymmetric.md

corpus:
  path: data/input.txt
  split:
    train_frac: 0.95
    heldout_frac: 0.05           # tail
  vocab_source: data/input.txt   # explicit, no auto-derivation

model:
  init_from: checkpoints/shake_d128_init.model
  # OR specify scratch architecture here

train:
  tool: agpt_train_v2
  mode: train-growth
  growth:
    frontiers: linear-128        # named recipe, or explicit list
    min_epochs: 3
    epoch_ramp: linear           # ramps min..max over the schedule
    max_epochs: 10
  optimizer:
    name: rmsprop
    lr: 0.003
    beta: 0.999
  lr_schedule:
    name: warmup-cosine
    warmup_epochs: 0
  flags:
    anc_grad: true
    chunk_queries: 50000
    growth_max_depth: 16

eval:
  tool: agpt_ppl.py              # canonical until HF wrapper lands
  mode: fixed                    # standard PPL
  also_record: [uniform]         # diagnostic-only, never compared
  slice:
    name: heldout_tail_first_10k
    source: heldout              # uses corpus.split.heldout_frac
    max_positions: 10000
  d_window: 16
  device: cpu
  batch_size: 256
```

Rules:
- Unknown keys fail validation. No silent ignores.
- Defaults are applied at resolution time and recorded in `resolved_config.json`.
- CLI overrides exist but emit a warning so we know provenance is being broken.

## meta.json fields

Auto-captured by the orchestrator before the run starts:

```json
{
  "git_sha": "deadbeef...",
  "git_branch": "main",
  "git_dirty": false,
  "command": "bin/agpt_experiment --config foo.yml --experiment harmonic-filter",
  "host": "trans-cachyos",
  "cuda_device": "RTX 4070 Laptop",
  "started_utc": "2026-05-25T14:30:00Z",
  "ended_utc":   "2026-05-25T17:01:08Z",
  "wall_seconds": 9068,
  "corpus": {
    "path": "data/input.txt",
    "sha256": "abc123...",
    "byte_count": 1115394
  },
  "model_init": {
    "path": "checkpoints/shake_d128_init.model",
    "sha256": "def456...",
    "header": {"d_model": 128, "n_heads": 8, "n_layers": 8, "d_ff": 512, "vocab": 65}
  }
}
```

## result.json fields

Written by the orchestrator after the evaluator returns:

```json
{
  "evaluator": "agpt_ppl.py",
  "evaluator_sha": "deadbeef...",
  "fixed_ppl": 5.4312,
  "diagnostics": {
    "uniform_ppl": 7.3321
  },
  "slice": {
    "name": "heldout_tail_first_10k",
    "source": "heldout",
    "first_position": 1059624,
    "max_positions": 10000,
    "positions_scored": 10000
  },
  "wall_seconds": 9068,
  "optimizer_steps": 49021
}
```

## Orchestrator: `bin/agpt_experiment`

Single entry point. Behavior:

```
bin/agpt_experiment --config <path> --experiment <name> [--run-id <slug>]
```

1. Resolve config (defaults applied, types checked, unknown keys rejected).
2. Compute config-hash. Refuse if `<experiment>/*/resolved_config.json` already
   has this hash with same corpus_sha.
3. Create `rnd/<experiment>/<run-id>/`.
4. Write `config.yml` (verbatim copy), `resolved_config.json`, `meta.json`
   (start-time fields).
5. Run trainer with flags derived from `train:` block. Tee to `train.log`.
6. Move trainer's saved model to `checkpoint.model`.
7. Run evaluator with flags derived from `eval:` block. Tee to `eval.log`.
8. Parse PPL from `eval.log`. Write `result.json`. Update `meta.json` with
   end-time fields.
9. Append result to `<experiment>/runs.json` and regenerate the table in
   `<experiment>/README.md`.

Exit codes:
- 0: success
- 2: config validation failed
- 3: trainer failed (tail of train.log printed)
- 4: evaluator failed (tail of eval.log printed)
- 5: duplicate config (points at existing run-id)

## Sequencing

**Stage 1: HF wrapper (Python).** Build `src/tools/agpt_hf.py` with
`AGPTConfig(PretrainedConfig)`, `AGPTModel(PreTrainedModel)`, and char
tokenizer. Verify it reproduces canonical PPL on a known baseline (L=8 d=128
100SE seed 1 → 3.6945) via `lm-evaluation-harness`. Once this lands, the
canonical evaluator is `lm-eval-harness` against the HF wrapper — no more
`agpt_ppl.py` for new work.

**Stage 2: Orchestrator (Crystal).** Build `bin/agpt_experiment` knowing
`lm-eval-harness` is the evaluator from day 1. No throwaway wiring. Crystal
because that matches existing tool conventions (`agpt_train_v2`, etc.) and
because the `microgpt` shard already provides config + model primitives we
can reuse.

(Codex's `src/cudax/` work was committed 2026-05-25 as 7cd3d08 before either
stage started.)

## Stage 1 implementation order

1. **`AGPTConfig` + `AGPTModel`** in `src/tools/agpt_hf.py`. Reuses
   `agpt_ppl.py`'s `load_model()` and forward code.
2. **Char tokenizer** in same file. 65-char vocab loaded from a vocab file.
3. **`from_pretrained` round-trip test**: load .model → HF state_dict →
   single forward pass matches `agpt_ppl.py`'s forward bit-for-bit.
4. **`lm-eval-harness` parity test**: run with `--task wikitext` or a
   custom Shakespeare task; verify PPL on the baseline matches 3.6945.
5. **Retire `agpt_ppl.py` and `agpt_sliding_window_perplexity`** to legacy
   (move under `legacy/` or note in CLAUDE.md). Update
   `feedback_evaluator_consistency` memory with the new canonical command.

## Stage 2 implementation order

1. **Crystal config structs** under `src/experiment/`: YAML::Serializable
   types for each top-level YAML block.
2. **Run-dir lifecycle** module: creates dir, writes config.yml +
   resolved_config.json + meta.json (with git_sha, corpus_sha, etc).
3. **Subprocess wrapping**: trainer (existing CLI) and evaluator
   (`lm-eval-harness` with HF wrapper). Tee logs.
4. **Result parsing**: read `lm-eval-harness` output → result.json.
5. **`runs.json` aggregation + `README.md` table generation**.
6. **`shard.yml` target + `Justfile` rule** for `bin/agpt_experiment`.
7. **End-to-end verification**: run the same known baseline through the
   orchestrator; numbers in `result.json` match the Stage 1 verification.
8. **Backfill 1–2 recent experiments** by hand-creating run dirs from
   existing logs (no retraining).
9. **CLAUDE.md update** documenting the workflow.

## Related

- `rnd/EXPERIMENT_TEMPLATE.md` — current hand-written template, will be
  superseded by orchestrator-generated `README.md` scaffold.
- `todo/proper-ppl-heldout-methodology.md` — multi-heldout proposal; the
  YAML `corpus.split` block is designed to support this when we wire it.
- `todo/hf-model-wrapper.md` — Stage 1 plan.
- `[[project_yaml_config]]` memory — flagged this as overdue 2026-05-22.

## Open questions for sign-off

1. **Run-id format**: `<UTC-stamp>-<slug>` vs `<config-hash-prefix>-<slug>`?
   Stamp is human-friendly and sorts chronologically; hash is reproducible.
2. **Backfill scope**: just the very recent active experiments (harmonic
   filter, dmodel-scaling), or any experiment with a clear single-run result?

Resolved:
- Orchestrator language: Crystal (`bin/agpt_experiment`), matches existing
  tools and reuses the `microgpt` shard.
- Evaluator: `lm-evaluation-harness` against HF wrapper from day 1; no
  interim `agpt_ppl.py` wiring.
