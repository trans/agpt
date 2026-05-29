# Migrating rnd/ configs to the new YAML schema

**Status:** active migration, applied incrementally as orchestrator support lands.
**Reference:** `docs/yaml-schema.md`.

## Migration table

Apply these renames/restructures to convert an old-schema config to the new:

| Old schema (legacy) | New schema |
|---|---|
| `meta.description` | `description` (top-level) |
| `meta.experiment` | `experiment` (top-level) |
| `meta.run_slug` | `run_slug` (top-level) |
| `meta.hypothesis_ref` | dropped (use description for the link, or omit) |
| `corpus.train_frac: 0.95` | Pre-carve via `bin/agpt_carve --source <path> --mode tail --ratio 0.05`; set `corpus.path` to the resulting `train_corpus.txt`, `corpus.heldout` to `heldout_corpus.txt`, and add `corpus.carve` block as provenance. |
| `model.init_from` | `model.init_file` |
| `train.kind: cudax` | dropped — trainer chosen via `--trainer v2` CLI |
| `train.kind: microgpt-sgd` | dropped — `--trainer microgpt` |
| `train.tool: bin/agpt_train_v2` | dropped — implied by `--trainer` |
| `train.mode: train-growth` | dropped — implied (presence of `train.growth` block = growth mode; absence = static) |
| `train.growth_divisions: N` + all other `growth_*` | → `train.growth: {divisions, min_epochs, epoch_ramp}` block. **Important:** `divisions: 1` is NOT a no-op in legacy v2 — it selects train-growth mode + the **incremental-radix materializer** (in-memory build, distinct code path from train-epoch's prebuilt static trie). Always preserve the growth block for numeric parity with legacy runs, even at divisions=1. |
| `train.epochs: 100` | `train.budget: {unit: epochs, value: 100}` |
| `train.steps: 25000` | `train.budget: {unit: steps, value: 25000}` |
| `train.optimizer: rmsprop` + `lr: 0.003` + `rmsprop_beta: 0.999` | `train.optimizer: {name: rmsprop, lr: 0.003, beta: 0.999}` |
| `train.momentum_beta: ...` | `train.optimizer.momentum_beta: ...` |
| `train.lr_schedule: warmup-cosine` + `warmup_epochs: 0` | `train.lr_schedule: {name: warmup-cosine, warmup_epochs: 0}` |
| `train.warmup_steps` | dropped — `train.lr_schedule.warmup_epochs` only; trainers convert as needed |
| `train.partition_depth` | unchanged |
| `train.chunk_queries` | unchanged |
| `train.anc_grad` | unchanged |
| `train.ablate_anc_grad` | dropped — use `anc_grad: false` directly |
| `train.accumulate: true` / `--accumulate` | dropped — use `partition_depth: 0` (one fire over whole trie, one optimizer step per epoch) |
| `train.extra_args` | dropped — every option must be in the schema |
| (implicit AGPT context depth — was `growth_max_depth`) | `train.max_depth` (now explicit; required for AGPT) |
| `train.seq_len` | `train.seq_len` (microgpt) — must match `max_depth` if both set |
| `eval.tool` | dropped — orchestrator always uses `src/tools/agpt_lm_eval.py` |
| `eval.split: train` | `eval.train_sanity: true` (and source defaults to `corpus.path`) |
| `eval.split: tail-heldout` | dropped — default eval source is `corpus.heldout` |
| `eval.split: external-heldout` + `eval.external_file` | just `eval.external_file: <path>` |
| `eval.split: benchmark` + `eval.benchmark` | just `eval.benchmark: <task>` |
| `eval.task_name` | dropped — orchestrator derives from `run_slug` |
| `eval.batch_size`, `device`, `limit` | unchanged |

## Pattern: `corpus.train_frac` → carved files + carve block

Most existing configs use `corpus.train_frac: 0.95` (tail-95/5). The new
schema requires concrete file paths, so:

1. Pre-carve once:
   ```sh
   bin/agpt_carve --source data/input.txt --mode tail --ratio 0.05
   # Prints: "Carved split written to data/.splits/<hash>/"
   ```
2. Replace the corpus block:
   ```yaml
   corpus:
     path: data/.splits/<hash>/train_corpus.txt
     heldout: data/.splits/<hash>/heldout_corpus.txt
     vocab_source: data/input.txt
     carve:
       source: data/input.txt
       mode: tail
       ratio: 0.05
   ```

For multi-chunk carves (`mode: sample`), include `chunks` and `seed`.

## Pattern: legacy `growth_divisions: N` (including N=1)

Every recently-active legacy CUDAX (v2) config has a `growth_*` block,
including configs labeled "static" that set `growth_divisions: 1`:
```yaml
growth_divisions: 1
growth_max_depth: 16
growth_min_epochs: 100
growth_epoch_ramp: fixed
```

**Do not drop the growth block at divisions=1.** Legacy v2 treats
`divisions: 1` as "one growth stage" via the train-growth mode, which
invokes the **incremental-radix materializer** (in-memory radix build).
That is a different code path from train-epoch + prebuilt static trie,
and it produces measurably different numbers (confirmed via the d16
parity smoke run, 2026-05-28).

Migration: always emit `train.growth` whenever the legacy config has any
`growth_*` field. `max_depth` moves out of the growth block to a
sibling under `train`:

```yaml
train:
  max_depth: 16            # was growth_max_depth
  growth:
    divisions: 1           # preserved — selects train-growth+incremental-radix
    min_epochs: 100        # was growth_min_epochs
    epoch_ramp: fixed      # was growth_epoch_ramp
```

For divisions > 1, same shape; just record the legacy value:
```yaml
train:
  max_depth: 16
  growth:
    divisions: 3
    min_epochs: 50
    epoch_ramp: linear
```

A legacy CUDAX config with **no** `growth_*` fields at all (rare; some
early configs only) migrates to the new schema with **no** `train.growth`
block — those used train-epoch + the orchestrator-built static trie.

## Example migration

A canonical example is preserved in
`rnd/cudax-d16-linear-mass-rerun/d16-d64L2-static100.new-schema.yml`
alongside the original `d16-d64L2-static100.yml` (legacy). Compare them
side by side for the full pattern.

## When to apply

- **Now:** new YAMLs going forward should use the new schema.
- **As migration support lands:** convert legacy configs in-place
  (replacing the `.yml`) once the orchestrator can consume new-schema
  configs and the trainers (v1/v2/microgpt-via-adapter) honor them.
- **For preserving Codex's in-flight runs:** keep both `.yml` and
  `.new-schema.yml` until the legacy orchestrator path is retired, so
  ongoing experiments aren't disrupted.

## Bulk migration (2026-05-28)

All 48 source configs under `rnd/*/` and `configs/*/` now have a
`.new-schema.yml` sibling produced by `tools/migrate_config.py` and
verified via `bin/agpt_experiment --validate` (48/48 pass).

Breakdown by trainer hint:
- **v2 (CUDAX):** 34 configs — straightforward; growth block preserved
  per the "divisions=1 is not a no-op" rule above.
- **microgpt:** 3 configs — `configs/baselines/`,
  `rnd/window-d124-baseline/`, `rnd/window-baseline-wallmatch/`.
- **trainer hint `?` (custom tools):** 11 configs — research prototypes
  that legacy-pointed `train.tool` at a Python script:
  - `rnd/harmonic-bias-prototype/configs/*` (6 files) — uses
    `rnd/harmonic-bias-prototype/tools/train.py` with `--harmonic-bias`
    and related flags carried via legacy `extra_args`. **Migrated with
    `extra_args` dropped** (warning emitted). If those runs need to be
    reproducible under the new orchestrator, the flags must be either
    added to the canonical schema or expressed under `experimental:`
    once trainer-side experimental support lands.
  - `rnd/v1-vs-v2-comparison/configs/*` (5 files) — uses
    `rnd/v1-vs-v2-comparison/tools/v1_train.py`. Has `accumulate: false`
    in the legacy YAML; migration drops it with a note that
    `partition_depth=0` is the new way to express "one fire over the
    whole trie" (the closest equivalent, not necessarily semantically
    identical to legacy `--no-accumulate`).

Legacy `.yml` files are preserved alongside their `.new-schema.yml`
siblings. The decision on when to swap (overwrite the old `.yml` with
the new content) is held until trainer-side experimental support lands
and the custom-tool configs above are decided on.
