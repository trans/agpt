# `agpt_experiment.cr` rewrite plan (task #29)

**Status:** rewrite landed (2026-05-28). Schema-parsing, validation,
carve/trie auto-invocation, resolved-YAML writing, trainer dispatch via
`--trainer`, microgpt + v2 direct-`--config` spawn, and the legacy v1
bridge are all in place. Verified via `--validate` smoke tests on both
migrated configs; full end-to-end run pending against v2.
**Reference:** `docs/yaml-schema.md`, `notes/operations/migrate-config-schema.md`.

## What changes

The orchestrator currently:
- Knows about two trainers via `train.kind` + `train.tool` (hardcoded to `cudax` / `microgpt-sgd`).
- Has per-kind CLI-flag builders (`build_trainer_args`, `build_microgpt_sgd_args`).
- Performs tail-only inline carving (`carve_corpus`).
- Validates an old-schema `eval.split` enum and special-cases each.

After the rewrite:
- Trainer is chosen via `--trainer <v1|v2|microgpt|<path>>` CLI flag. No YAML field for it.
- Each trainer's CLI is just `<tool> --config <resolved-yaml>` (plus optional `--seed`). The orchestrator writes the resolved YAML to the run dir and spawns.
- Carving is delegated to `bin/agpt_carve`. The orchestrator invokes it when the YAML's referenced files don't exist and `corpus.carve` is set.
- Trie build/cache is automated similarly (Codex's `bin/agpt_build_radix` invocation).
- Eval uses `--chunks-dir` when the heldout is a sample-mode carve; `--text-file` otherwise.

## CLI surface (final)

```
bin/agpt_experiment --config <yaml> --trainer <v1|v2|microgpt|<path>> [--seed <int>]
bin/agpt_experiment --validate <yaml>     # parse-only, no run
```

Trainer name resolution:
- `v1` → `bin/agpt_train`
- `v2` → `bin/agpt_train_v2`
- `microgpt` → `bin/microgpt_yaml` (the YAML adapter)
- Anything else with `/` in it → treat as explicit binary path

## Config types

Replace the existing `MetaBlock` / `CorpusBlock` / `ModelBlock` /
`TrainBlock` / `EvalBlock` / `Config` with new-schema versions
mirroring `docs/yaml-schema.md`. Key changes:

- `MetaBlock` is gone; identity fields (`description`, `experiment`,
  `run_slug`) are top-level properties of `Config`.
- `CorpusBlock`: drop `train_frac`; add `heldout: String?` and
  `carve: CarveBlock?`.
- `CarveBlock`: `source: String`, `mode: String` (sample|tail),
  `ratio: Float64`, `chunks: Int32?`, `seed: Int32?`.
- `TrieBlock` (new): `max_depth: Int32?`, `prune_min_mass: Int32 = 1`,
  `prune_min_depth: Int32 = 0`, `path: String?`, `virtual_tree: Bool = false`.
- `ModelBlock`: `init_file: String?`, `init_seed: Int32?`, `save_file: String?`,
  `d_model: Int32?`, `n_layers: Int32?`, `n_heads: Int32?`, `d_ff: Int32?`,
  `head_dim: Int32?` (all optional; required only when init_file absent).
- `TrainBlock`: completely restructured — `budget: {unit, value}`,
  `optimizer: {name, lr, beta, momentum_beta, weight_decay, grad_clip_norm}`,
  `lr_schedule: {name, warmup_epochs}`, plus AGPT fields (`max_depth`,
  `partition_depth`, `chunk_queries`, `anc_grad`, `mass_weight`,
  `fire_norm`, `entropy_lambda`, `ce_only`, `growth: {...}?`) and
  microgpt fields (`seq_len`, `backend`, `heads`, `lookahead`).
- `EvalBlock`: drop `split` + `tool` + `task_name`; add
  `external_file: String?`, `benchmark: String?`, `train_sanity: Bool = false`;
  keep `batch_size`, `device`, `limit`.

## Flow (final)

1. Parse `--config` against new types (strict; YAML::Serializable does typo
   detection naturally with the `discriminator` / `extra_properties: raise` options).
2. Apply defaults (model/train budget defaults).
3. Compute config SHA-256 over the resolved YAML for duplicate-run detection.
4. Create `rnd/<experiment>/<UTC>-<run_slug>/`.
5. **Carve:** if `corpus.path` doesn't exist and `corpus.carve` is set,
   invoke `bin/agpt_carve --config <yaml>` (or with explicit flags).
   On success, files appear at the schema-declared paths.
6. **Trie:** for AGPT trainers, if `trie.path` is unset, invoke
   `bin/agpt_build_radix` (or equivalent) to populate the cache at
   `data/.tries/<hash>/`; set `trie.path` in the resolved YAML.
7. **Resolved YAML:** apply defaults like `model.save_file → <rundir>/checkpoint.model`;
   write to `<rundir>/resolved_config.yml`.
8. **Spawn trainer:** `<trainer-binary> --config <rundir>/resolved_config.yml`
   (plus `--seed` override if given).
9. **HF convert** (unchanged).
10. **Eval:** if `corpus.heldout` exists AND there's a sibling
    `heldout_chunks/` dir, use `--chunks-dir`; else use `--text-file <heldout>`.
    If `eval.external_file` set, use that. If `eval.benchmark` set, use
    `--builtin-task`.
11. **`result.json`** (mostly unchanged shape; field names refreshed).

## Transition status (post-rewrite)

- **v2 (`--trainer v2`)** — direct `--config` spawn. v2's YAML support
  landed; the orchestrator hands it `<rundir>/resolved_config.yml` and
  passes `--seed` only when overridden.
- **microgpt (`--trainer microgpt`)** — direct `--config` spawn to
  `bin/microgpt_yaml`, which adapts to microgpt's CLI internally.
- **v1 (`--trainer v1`)** — bridge function in
  `build_v1_bridge_args` translates the new-schema Config to v1's
  legacy CLI flags. Bridge covers the common knobs only; raises a clear
  error if `train.growth` is set (v1 lacks growth — task #31) or if
  budget unit is not `epochs` (v1 doesn't take `steps`/`wall_seconds`).
  Bridge is deleted once #28 lands.
- **Custom path (`--trainer /path/to/binary`)** — treats it as a v2-style
  spawn (passes `--config <resolved>` + optional `--seed`). Trie
  auto-build is skipped for custom paths — the user is responsible for
  setting `trie.path` if their tool needs it.

## Validation rules

The orchestrator validates the full schema (typo detection at field
level). Trainer-specific validation happens at the trainer (it errors
on fields it can't consume). The orchestrator just ensures:

- `description`, `experiment`, `run_slug` present at top level.
- `corpus.path` present and file exists (after carve resolution).
- `model.init_file` exists if specified.
- One of `train.budget.unit` values: `epochs | steps | wall_seconds`.
- `eval.external_file` / `eval.benchmark` are mutually exclusive.
- `seq_len`/`max_depth` cross-check rule (if both set, must match).

## Files that change

- `src/tools/agpt_experiment.cr` — substantial rewrite of Config types,
  flow, and removal of per-kind builders. Add bridge function.
- `notes/operations/experiment-runner-design.md` — update spec to match
  new schema.
- Existing tests/fixtures (if any) — update.

## Tests

- Parse-only smoke (`--validate`) on a migrated new-schema config
  (start with `rnd/cudax-d16-linear-mass-rerun/d16-d64L2-static100.new-schema.yml`).
- Full end-to-end run on the same config once v2 `--config` is in.
- Multi-chunk carve + eval end-to-end on Shakespeare.

## Out of scope

- Schema bumps to add new fields (handled per-PR as needed).
- v1 growth implementation (#31, separate work).
