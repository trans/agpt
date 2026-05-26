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

## Sequencing with other in-flight work

Codex has uncommitted changes under `src/cudax/` (sampled-bin multi-position
probe). These touch the trainer and must be either committed or parked before
infrastructure work begins, so that the trainer's CLI surface is stable when
we wire YAML through it. **This is the gate** — no implementation starts until
the CUDAX state is settled.

## Implementation order (after CUDAX is settled)

1. **Schema + validator** (Python lib, ~150 lines). Used by orchestrator and
   by `agpt_ppl.py`'s `--config` support.
2. **`agpt_ppl.py --config X.yml`** (eval is the easier integration; Python
   already does YAML).
3. **`bin/agpt_experiment` v0**: orchestrator that wraps a YAML-aware
   `agpt_ppl.py` + an existing-CLI `agpt_train_v2`. The trainer call is built
   from the `train:` block; trainer doesn't need to know about YAML yet.
4. **`agpt_train_v2 --config X.yml`** (Crystal-side, harder). When this lands,
   the orchestrator stops constructing flag strings and just passes
   `--config resolved_config.json` through.
5. **Run a known baseline through the system end-to-end** to verify a
   well-established number reproduces (e.g. L=8 d=128 100SE → Fixed PPL 3.6945
   on Shakespeare).
6. **Backfill one or two recent experiments** as runs under the new layout
   (does not require re-training; just creates dirs + meta.json + result.json
   from existing logs).
7. **CLAUDE.md update** documenting the workflow.

## Out of scope (deferred to HF migration)

- Industry-standard evaluator integration (`lm-evaluation-harness`).
- `AgptForCausalLM(PreTrainedModel)` subclass + char tokenizer wrapper.
- `agpt_ppl.py` and `agpt_sliding_window_perplexity` retirement.

Until HF lands, `agpt_ppl.py --mode fixed` is the canonical evaluator. The
orchestrator's `eval.tool` field is parameterized so swapping to
`lm-evaluation-harness` later is a config change, not a code change.

## Related

- `rnd/EXPERIMENT_TEMPLATE.md` — current hand-written template, will be
  superseded by orchestrator-generated `README.md` scaffold.
- `todo/proper-ppl-heldout-methodology.md` — multi-heldout proposal; the
  YAML `corpus.split` block is designed to support this when we wire it.
- `todo/hf-model-wrapper.md` — Step 2 (HF compatibility).
- `[[project_yaml_config]]` memory — flagged this as overdue 2026-05-22.

## Open questions for sign-off

1. **Run-id format**: `<UTC-stamp>-<slug>` vs `<config-hash-prefix>-<slug>`?
   Stamp is human-friendly and sorts chronologically; hash is reproducible.
2. **Where does `agpt_experiment` live**: `bin/` (compiled, matches other
   tools) or `scripts/` (Python, easier to iterate)? Lean Python initially,
   move to Crystal later if needed.
3. **Backfill scope**: just the very recent active experiments (harmonic
   filter, dmodel-scaling), or any experiment with a clear single-run result?
4. **YAML library**: `pyyaml` (standard) or `ruamel.yaml` (round-trips
   comments)? Lean pyyaml.
