# Canonical YAML Schema for AGPT Runs

**Status:** draft v0.4 · **Date:** 2026-05-28
**Scope:** the single configuration surface for `bin/agpt_train` (v1),
`bin/agpt_train_v2` (CUDAX, v2), `bin/microgpt`, `bin/agpt_carve`,
and `bin/agpt_experiment`.
Replaces all CLI flags and `AGPT_*` env vars.

## Philosophy

1. **One config artifact per run.** The YAML file is the complete,
   reproducible description of a run. Anything that affects the run
   lives in the YAML. Anything not in the YAML is not part of the run.
2. **The YAML describes an *experiment*, not a *trainer invocation*.**
   Which trainer runs it is decided at invocation (CLI), not in the
   YAML. The same YAML can be handed to any trainer; mismatches
   surface as errors at trainer startup.
3. **Structured by information, not by tool.** Each section describes
   one aspect of the experiment (identity, corpus, model, training,
   evaluation). What tool consumes a field falls out from that.
4. **Shared concepts share one canonical name.** If two trainers do
   the same thing, the field has the same name.
5. **Files, not actions.** All data references in the YAML are concrete
   file paths. Operations that *produce* files (e.g., carving a corpus
   into train + held-out chunks) happen *before* the trainer runs,
   either via dedicated tools (`bin/agpt_carve`) or by the orchestrator
   invoking those tools. The YAML never asks a trainer to perform an
   action it cannot perform; if a referenced file doesn't exist, the
   trainer errors cleanly (file not found), with no silent fallback.
6. **Validation is strict but scope-defined.**
   - **Schema validation (universal):** every field must exist in the
     canonical registry below. Unknown field names are errors.
   - **Trainer validation (within trainer's domain):** trainers
     strict-reject fields they can't honor within their domain
     (`corpus.{path,vocab_source}`, `model`, `train`, `trie` for AGPT).
   - **Outside trainer's domain:** trainers ignore sections that are
     not their concern (`description`, `experiment`, `run_slug`, `eval`,
     `corpus.heldout`, `corpus.carve`). These describe other facets of
     the experiment — provenance, eval setup, carve recipe — that the
     trainer doesn't need.
7. **CLI surface is intentionally tiny.**
   - Trainers: `--config <path>` (required) + `--seed <int>` (optional override).
   - Carve tool: `bin/agpt_carve --config <path>` (uses `corpus.carve` block).
   - Orchestrator: `--config <path>` + `--trainer <name|tool>` + `--seed <int>`.
   - No other flags. No `AGPT_*` env vars.

## Top-level structure

```yaml
description: ...                # optional, highly recommended
experiment: ...                 # required (orchestrator)
run_slug: ...                   # required (orchestrator)

corpus: {...}                   # data + heldout + carve provenance
trie: {...}                     # AGPT only — trie/index spec
model: {...}                    # architecture
train: {...}                    # training protocol
eval: {...}                     # evaluation protocol
```

Field tables list:
- **field** — canonical name
- **type** — JSON-Schema-like type
- **default** — value when omitted (or "required")
- **consumers** — load-bearing per-consumer declaration:
  - `v1` = `bin/agpt_train`
  - `v2` = `bin/agpt_train_v2`
  - `mg` = `bin/microgpt`
  - `orch` = `bin/agpt_experiment`
  - `carve` = `bin/agpt_carve`
  - `(ignored)` = consumer that knows the field exists but does nothing with it (no error)
- **notes** — semantics

A trainer NOT listed in **consumers** for a given field — and not explicitly marked `(ignored)` — will reject any YAML containing that field.

---

## Experiment identity (top-level)

| field | type | default | consumers | notes |
|---|---|---|---|---|
| `description` | str | — | orch (recorded); v1, v2, mg (ignored) | Free-form human description. Highly recommended for orchestrated runs. |
| `experiment` | str | required | orch (used for rundir); v1, v2, mg (ignored) | Experiment family name. Rundir = `rnd/<experiment>/<UTC>-<run_slug>/`. |
| `run_slug` | str | required | orch (used for rundir); v1, v2, mg (ignored) | Run identifier. |

---

## `corpus`

| field | type | default | consumers | notes |
|---|---|---|---|---|
| `corpus.path` | path | required | v1, v2, mg, orch | The training corpus file. The trainer trains on this file directly. For carved experiments, this points at the carved train file (e.g., `data/.splits/<hash>/train_corpus.txt`). |
| `corpus.heldout` | path | absent | orch (used for eval); v1, v2, mg (ignored) | The held-out corpus file. Eval scores against this. For carved experiments, points at the carved heldout file. Absent → no held-out eval available unless `eval.external_file` is set. |
| `corpus.vocab_source` | path | = `corpus.path` | v1, v2, mg | Vocabulary derivation source. |
| `corpus.carve` | block | absent | orch, carve; v1, v2, mg (ignored) | Optional. Records *how* the carved files at `corpus.path` and `corpus.heldout` were produced — `agpt_carve` reads it to (re)generate the files if missing; orchestrator uses it for automation and recorded provenance. |
| `corpus.carve.source` | path | required if carve block present | orch, carve | The original (uncarved) source corpus. |
| `corpus.carve.mode` | enum | required if carve block present | orch, carve | `sample` (multi-chunk random) \| `tail` (single tail slice). |
| `corpus.carve.ratio` | float ∈ (0, 1) | required if carve block present | orch, carve | Held-out fraction. (Training fraction = 1 − ratio.) |
| `corpus.carve.chunks` | int ≥ 2 | required when `mode: sample` | orch, carve | Number of disjoint chunks. |
| `corpus.carve.seed` | int | required when `mode: sample` | orch, carve | RNG seed for chunk placement. |

### How carving works

Carving is a **pre-processing step**, not a runtime configuration. The YAML
always references concrete file paths via `corpus.path` and `corpus.heldout`.
The optional `corpus.carve` block is *provenance + automation hint*: it
records how the carved files were produced (or how to produce them).

Three workflows:

1. **Manual carve:**
   ```sh
   bin/agpt_carve --source data/input.txt --mode sample \
                  --ratio 0.05 --chunks 10 --seed 42 \
                  --out-dir data/.splits/<auto-hash>/
   ```
   The tool writes `train_corpus.txt`, `heldout_corpus.txt`,
   `heldout_chunks/`, `manifest.json` to the cache dir. It prints the
   YAML snippet for you to paste into your experiment config.
2. **Carve via YAML:** if you've written the YAML with a `corpus.carve`
   block, run `bin/agpt_carve --config <yaml>` and it produces the files
   referenced by `corpus.path` and `corpus.heldout`.
3. **Orchestrator-automated:** `bin/agpt_experiment --config <yaml> ...`
   detects missing `corpus.path` / `corpus.heldout` and (if `corpus.carve`
   is set) invokes `agpt_carve` to populate them before training. If
   `corpus.carve` is absent and files are missing, the orchestrator
   errors with "carve spec required to auto-generate files."

In every case, trainers see only the `corpus.path` and `corpus.heldout`
file paths and the resulting data; they never see or perform the carving
itself.

### Content-hashed split cache

The standard cache location for carved splits is
`data/.splits/<hash>/`, where `<hash>` is the SHA-256 of
`(source_sha256, mode, ratio, chunks, seed)`. Contents:

```
data/.splits/<hash>/
  train_corpus.txt
  heldout_corpus.txt
  heldout_chunks/chunk_NN.txt    (mode: sample only)
  manifest.json                  (source + carve params + sha256s)
```

`agpt_carve` writes here by default. Same carve params → same hash →
files are reused across runs. Cache eviction is manual for now.

### Carve modes

- `tail` — `train_corpus.txt` = first `1 − ratio` of source,
  `heldout_corpus.txt` = remaining tail. Single contiguous split.
- `sample` — `train_corpus.txt` = source with `chunks` disjoint
  randomly-placed chunks removed, concatenated. `heldout_corpus.txt` =
  those chunks concatenated. Seeded rejection sampling.

---

## `trie` (AGPT-only)

The radix trie is a derived artifact built from `corpus.path`. Like
split caches, tries are content-hashed and cached at
`data/.tries/<hash>/`.

| field | type | default | consumers | notes |
|---|---|---|---|---|
| `trie.max_depth` | int | = `train.max_depth` | v1, v2, orch; mg (ignored) | Trie depth. Defaults to match `train.max_depth`. |
| `trie.prune_min_mass` | int ≥ 1 | 1 | v1, v2, orch; mg (ignored) | Minimum prefix count for nodes to survive pruning. |
| `trie.prune_min_depth` | int ≥ 0 | 0 | v1, v2, orch; mg (ignored) | Pruning depth threshold. |
| `trie.path` | path | (orch: auto-build + cache; direct trainer: required) | v1, v2; mg (ignored) | Pre-built trie location. **For orchestrator-driven runs:** if omitted, orchestrator builds and caches. **For direct AGPT trainer invocation:** required — direct trainers do not build tries themselves. |
| `trie.virtual_tree` | bool | false | v1, v2; mg (ignored) | Virtual-tree augmentation (advanced). |

---

## `model`

| field | type | default | consumers | notes |
|---|---|---|---|---|
| `model.d_model` | int | required if `init_file` absent; optional if present | v1, v2, mg | Embedding dim. When `init_file` is set: optional; if provided must match checkpoint header. |
| `model.n_layers` | int | required if `init_file` absent; optional if present | v1, v2, mg | Layer count. Same rule. |
| `model.n_heads` | int | required if `init_file` absent; optional if present | v1, v2, mg | Head count. Same rule. |
| `model.d_ff` | int | `4 × d_model` | v1, v2, mg | Feed-forward inner dim. |
| `model.head_dim` | int | `d_model / n_heads` | v1, v2, mg | Per-head dim. Must divide `d_model`. |
| `model.init_file` | path | (fresh init) | v1, v2, mg | Pre-trained checkpoint to load. |
| `model.init_seed` | int | = `train.seed` | v1, v2, mg | RNG seed for fresh init (when `init_file` absent). |
| `model.save_file` | path | orch: `<rundir>/checkpoint.model`; trainer-direct: omit ⇒ warn + no save | v1, v2, mg, orch | Where the trained model is written. When absent under direct invocation, the trainer logs `WARN: model.save_file not set; trained model will not be persisted` and exits without writing. Orchestrated runs always have it filled in. |

---

## `train`

### Universal

| field | type | default | consumers | notes |
|---|---|---|---|---|
| `train.budget.unit` | enum | required | v1, v2, mg | `epochs` \| `steps` \| `wall_seconds`. **`wall_seconds` is a stopping policy, not a reproducible optimization budget** — actual completed epochs/steps are recorded in `result.json` as the true execution result. |
| `train.budget.value` | int / float | required | v1, v2, mg | Budget amount in the specified unit. |
| `train.seed` | int | 42 | v1, v2, mg | Run-level RNG seed. The single CLI-overridable field. |
| `train.quiet` | bool | false | v1, v2, mg | Suppress per-iteration print. |

### `train.optimizer`

| field | type | default | consumers | notes |
|---|---|---|---|---|
| `train.optimizer.name` | enum | required | v1, v2 (full enum); mg (adam only) | `adam` \| `rmsprop` \| `sgd` \| `momentum` \| `lbfgs` (v1/v2 only). Microgpt: anything but `adam` → error. |
| `train.optimizer.lr` | float | required (except `lbfgs`) | v1, v2, mg | Peak learning rate. |
| `train.optimizer.beta` | float | 0.999 | v1, v2 | RMSProp β / Adam β₂. |
| `train.optimizer.momentum_beta` | float | 0.9 | v1, v2 | Adam β₁ / Momentum β. |
| `train.optimizer.weight_decay` | float | 0.0 | v1, v2 | Decoupled weight decay. |
| `train.optimizer.grad_clip_norm` | float | 0.0 | v1, v2 | 0 disables. |

### `train.lr_schedule`

| field | type | default | consumers | notes |
|---|---|---|---|---|
| `train.lr_schedule.name` | enum | `constant` | v1, v2, mg | `constant` \| `cosine` \| `warmup-cosine`. |
| `train.lr_schedule.warmup_epochs` | int | 0 | v1, v2, mg | Warmup duration in epochs. |

### Context window — sibling fields with cross-check rule

| field | type | default | consumers | notes |
|---|---|---|---|---|
| `train.seq_len` | int | required if mg | mg | Microgpt sliding-window length. |
| `train.max_depth` | int | required if v1/v2 | v1, v2 | AGPT trie max depth. |

**Cross-check rule.** Either may appear alone. If both appear, they must match (AGPT cannot currently support `seq_len ≠ max_depth`).

### AGPT (v1 + v2)

| field | type | default | consumers | notes |
|---|---|---|---|---|
| `train.partition_depth` | int ≥ 0 | 1 | v1, v2 | Subtree partitioning depth. `pd=0` = single fire over whole trie. **Migration:** `pd=0` semantics must be implemented in v1/v2 (currently they clamp/reject) before `--accumulate` can be removed. |
| `train.chunk_queries` | int | 50000 | v1, v2 | Per-chunk T_q budget. |
| `train.anc_grad` | bool | true | v1, v2 | Descendant→ancestor gradient flow. |
| `train.mass_weight` | enum | `linear` | v1, v2 | `off` \| `linear` \| `sqrt` \| `log` \| `inv-log` \| `inv-linear`. Per-event loss weighting. |
| `train.fire_norm` | enum | `mass` | v1, v2 | `events` \| `mass` \| `weight` \| `none`. Per-fire gradient divisor. |
| `train.entropy_lambda` | float | 0.0 | v1, v2 | Entropy regularizer weight. |
| `train.ce_only` | bool | false | v1, v2 | Force endpoint queries to single-target CE. |

### `train.growth` — v2 today, v1 planned

| field | type | default | consumers | notes |
|---|---|---|---|---|
| `train.growth` | block | absent | v2 (today), v1 (planned); mg rejects | Optional. Block absent = static training. |
| `train.growth.divisions` | int ≥ 1 | required if block present | v2, v1 (planned) | Number of stages. |
| `train.growth.min_epochs` | int ≥ 1 | required if block present | v2, v1 (planned) | Per-stage minimum epoch budget. |
| `train.growth.epoch_ramp` | enum | required if block present | v2, v1 (planned) | `fixed` \| `linear`. |

### Microgpt-only

| field | type | default | consumers | notes |
|---|---|---|---|---|
| `train.backend` | enum | `crystal` | mg | `crystal` \| `cuda`. **Status:** `cuda` backend is not actively used and may be broken. |
| `train.heads` | enum | `uniform` | mg | Head initialization scheme. |
| `train.lookahead` | int | 0 | mg | Lookahead heads (0 = none, -1 = future model). |

### Removed / not in the schema

- `--accumulate` / `--no-accumulate`: replaced by `partition_depth=0`.
- `--curriculum: flat | progressive`: removed.
- `--mode` (legacy v1/v2 CLI): trainer infers from rest of schema.
- `extra_args` (legacy orchestrator escape hatch): removed.
- `corpus.holdout` block: replaced by `corpus.heldout` (file path) +
  `corpus.carve` (provenance + automation). Carving is now an
  explicit pre-processing step, not a runtime instruction to trainers.
- All `AGPT_*` env vars: closed-branch research toggles. If
  reintroduced, add explicit YAML fields and document.

---

## `eval`

| field | type | default | consumers | notes |
|---|---|---|---|---|
| `eval.external_file` | path | absent | orch; v1, v2, mg (ignored) | Evaluate against this external corpus instead of `corpus.heldout`. The model has never seen it. |
| `eval.benchmark` | str | absent | orch; v1, v2, mg (ignored) | Invoke the named lm-evaluation-harness registered task (e.g., `wikitext`). |
| `eval.train_sanity` | bool | false | orch; v1, v2, mg (ignored) | If true, also eval on `corpus.path` as a sanity check. |
| `eval.batch_size` | int | 1 | orch; v1, v2, mg (ignored) | lm-eval batch size. |
| `eval.device` | enum | cpu | orch; v1, v2, mg (ignored) | `cpu` \| `cuda`. **Run-defining** (timing and sometimes numerics) — recorded in resolved YAML. |
| `eval.limit` | int | (no limit) | orch; v1, v2, mg (ignored) | Optional cap on positions/samples. |

### Source-selection rules

- Neither `external_file` nor `benchmark` set → evaluate on `corpus.heldout`. If `corpus.heldout` is also absent → orchestrator errors at startup.
- `external_file` set → evaluate on that file.
- `benchmark` set → invoke lm-eval's named task. `corpus.path` is irrelevant.
- `external_file` and `benchmark` are mutually exclusive.

### Multi-chunk eval mechanics

When `corpus.heldout` corresponds to a `sample`-mode carve (the carved
directory has `heldout_chunks/`), the evaluator scores each chunk
independently and aggregates per-token NLL across chunks for the
overall PPL, with per-chunk breakdown in `result.json`.

### Reported metrics

Both `fixed_token_ppl` and `rolling_byte_ppl` are computed and stored
in `result.json` when applicable.

### lm-eval task naming

The orchestrator derives the internal task name from `run_slug`
(`<run_slug>_holdout`, `<run_slug>_external`, etc.). Not in the YAML.

---

## `experimental`

Free-form scratch space for in-development knobs that haven't earned a
canonical schema slot yet. **The strict-rejection rule that applies to
every other section is relaxed inside `experimental`:** trainers emit
`WARN: unknown experimental flag <name>; ignoring` rather than erroring.

| field | type | default | consumers | notes |
|---|---|---|---|---|
| `experimental.<name>` | any | absent | per-trainer | Pass-through to whichever trainer reads it. Flat key/value (not namespaced by trainer in the input). |

### Discipline

- `experimental` is **opt-in chaos**. Anything inside it loses the
  schema's typo-detection. Use it for one-off prototyping; graduate a
  flag into the canonical schema (with a real entry in this doc) before
  citing its numbers in a report.
- `result.json` records `experimental_used: {<trainer>: [keys]}`
  whenever the input had a non-empty `experimental:` block. Filters
  canonical-vs-experimental runs when aggregating.
- The relaxed rule applies only to **unknown** experimental names.
  Trainers may still hard-error on a known experimental flag they
  reject (e.g., conflicting values, unsupported combinations).
- Comparisons across runs with different `experimental:` content are
  not apples-to-apples; the orchestrator does not block them but the
  marking in `result.json` makes the divergence findable.

### Example

```yaml
experimental:
  attention_v3_alpha: 0.5
  kernel_chunked_softmax: true
```

---

## Conflict rule

When a trainer reads two fields that represent overlapping concepts
and their values are incompatible, the trainer raises an error at
startup citing the conflicting fields. Currently the only such pair
is `train.seq_len` / `train.max_depth` (cross-check rule).

---

## CLI surface

```
# Direct trainer invocation
<trainer> --config <yaml-path> [--seed <int>]

# Carve tool
bin/agpt_carve --config <yaml-path>
bin/agpt_carve --source <path> --mode <mode> --ratio <r> [--chunks <c> --seed <s>] --out-dir <dir>

# Orchestrator
bin/agpt_experiment --config <yaml-path> --trainer <name|tool> [--seed <int>]
```

No other flags. No `AGPT_*` env vars. The OS-level env vars
`CUDA_VISIBLE_DEVICES`, `OMP_NUM_THREADS`, and `CUBLAS_WORKSPACE_CONFIG`
remain permitted (recorded in `meta.json` for provenance).

---

## Orchestrator → trainer flow

1. **Read** the input YAML.
2. **Verify carved files exist.** If `corpus.path` or `corpus.heldout`
   refers to a file that doesn't exist:
   - If `corpus.carve` is set: invoke `bin/agpt_carve` to populate.
   - Otherwise: error.
3. **Build trie** (AGPT only) if `trie.path` is unset; populate it in
   the resolved YAML.
4. **Apply defaults** (e.g., `model.save_file` → `<rundir>/checkpoint.model`).
5. **Write** the resolved YAML to `<rundir>/resolved_config.yml`. This
   contains the full input + applied defaults; nothing is stripped.
   Trainers ignore metadata sections per the validation rule.
6. **Spawn** trainer with `--config <rundir>/resolved_config.yml`.

A trainer invoked directly with `--config foo.yml` reads only that
file; if any referenced files don't exist, it errors with a clear
file-not-found message.

---

## Provenance

The orchestrator writes the following into the run directory:

- **`config.yml`** — input YAML verbatim.
- **`resolved_config.yml`** — input + applied defaults; what the
  trainer actually saw.
- **`meta.json`** — provenance (git SHA, branch, dirty status, corpus
  SHA-256, init model header, env fingerprint, full invocation
  command line, `--trainer` value, content-hashes of cached
  splits/tries).

---

## Example config

```yaml
description: "CUDAX d16 baseline at d_model=64, L=2, 100 epochs, multi-chunk holdout"
experiment: cudax-d16-multi-chunk-baseline
run_slug: d16-d64L2-static100

corpus:
  path: data/.splits/abc123/train_corpus.txt
  heldout: data/.splits/abc123/heldout_corpus.txt
  vocab_source: data/input.txt
  carve:                       # provenance + automation
    source: data/input.txt
    mode: sample
    ratio: 0.05
    chunks: 10
    seed: 42

trie:
  prune_min_mass: 1
  prune_min_depth: 0

model:
  init_file: data/input.model
  # d_model/n_layers/n_heads omitted — taken from checkpoint header.

train:
  budget:
    unit: epochs
    value: 100
  seed: 42
  quiet: true
  optimizer:
    name: rmsprop
    lr: 0.003
    beta: 0.999
  lr_schedule:
    name: warmup-cosine
    warmup_epochs: 0
  max_depth: 16
  partition_depth: 1
  anc_grad: true
  mass_weight: linear
  fire_norm: mass

eval:
  batch_size: 1
  device: cpu
```

If the carved files at `data/.splits/abc123/` don't exist yet,
`bin/agpt_experiment` invokes `bin/agpt_carve` with the `corpus.carve`
block to produce them, then runs training. If you'd rather carve
manually first, run `bin/agpt_carve --config foo.yml` once; subsequent
runs of the experiment reuse the cached split.

---

## Migration plan

**Sequencing matters.**

1. **Schema doc** (this file) — review iterations.
2. **`bin/agpt_carve` tool.** Crystal CLI that takes either a config
   YAML (uses `corpus.carve`) or explicit CLI flags, produces carved
   files in `data/.splits/<hash>/` with `manifest.json`. Smallest
   prerequisite for the whole pipeline.
3. **Pilot: microgpt migrate to YAML config.** Smallest trainer
   surface, native YAML support.
4. **Multi-chunk in `agpt_lm_eval.py`.** Per-chunk score + aggregate.
   Needed for `eval` to score multi-chunk holdouts.
5. **Install `yam` (Arch package)** and wire into v1/v2 builds. No vendoring — just `pacman -S yam` (or equivalent) and link `-lyam` from the trainer build recipes.
6. **Implement `partition_depth=0` semantics in v1 and v2.** Required
   before `--accumulate` can be removed.
7. **Migrate CUDAX (v2) to YAML.** Blocks on #5, #6.
8. **Migrate CUDA (v1) to YAML.** Blocks on #5, #6.
9. **Orchestrator simplification.** Drop `train.kind` +
   `build_*_args`. Add `--trainer` CLI flag. Write `resolved_config.yml`.
   Auto-invoke `agpt_carve` when files missing and `corpus.carve`
   present. Auto-build trie when `trie.path` missing. Multi-chunk
   eval flow.
10. **Migrate existing rnd configs** to new schema. Validation pass.
11. **Add growth mode to v1** (planned future).

---

## Resolved decisions log

- **Trainer selection** is a CLI argument to the orchestrator
  (`--trainer`), not a YAML field.
- **Carving is a pre-processing step**, not a runtime configuration.
  The YAML references concrete file paths; `corpus.carve` records
  *how* those files were produced (or how to regenerate them).
  Trainers see file paths only; missing files → clear file-not-found
  error.
- **`bin/agpt_carve`** is the dedicated tool that produces carved
  files. `bin/agpt_experiment` can invoke it automatically as a setup
  step before training.
- **Validation is scope-defined.** Trainers strict-reject within
  their domain (corpus.{path,vocab_source}, model, train, trie for
  AGPT); ignore outside domain (description, experiment, run_slug,
  eval, corpus.heldout, corpus.carve).
- **No stripping.** Resolved YAML preserves all input fields plus
  applied defaults.
- **`description`** — optional in general; treat as required for
  orchestrated runs.
- **`backend: cuda`** for microgpt — kept; status flagged.
- **Cache eviction for `data/.splits/` and `data/.tries/`** — manual
  only for now.
- **Budget unit** supports `epochs`, `steps`, `wall_seconds` (the
  latter being a stopping policy, not a reproducible budget).
- **Architecture fields under `model`** are optional when `init_file`
  is set; if specified, must match checkpoint header.
- **lm-eval task name** is derived from `run_slug` by the orchestrator.
- **`eval.device`** is run-defining and recorded in `resolved_config.yml`.
