# Rename d_adam_m / d_adam_v to optimizer-agnostic names

The buffers `d_adam_m`, `d_adam_v`, `h_adam_m`, `h_adam_v` (and the
matching opt-state file format using `adam_t`) are misnamed. They're
shared between Adam and RMSprop:

- Adam uses both: `m` (first moment), `v` (second moment).
- RMSprop uses only `v` (squared-grad EMA, aka `s` in standard
  RMSprop notation). `m` is allocated, zero-filled, and serialized
  but never read or written under RMSprop.
- SGD / momentum: `m` repurposed as velocity for momentum; `v` unused.

Result: anyone reading the code under `--optimizer rmsprop` sees
"adam" all over and assumes a bug. Surfaced 2026-05-22 by user.

## Proposed rename

| Current             | Proposed              | Meaning                            |
|---------------------|-----------------------|-------------------------------------|
| `d_adam_m`/`h_adam_m` | `d_opt_m1`/`h_opt_m1` | First-moment buffer (Adam, momentum velocity) |
| `d_adam_v`/`h_adam_v` | `d_opt_m2`/`h_opt_m2` | Second-moment buffer (Adam v, RMSprop s) |
| `adam_t`              | `opt_step`            | Step counter (Adam only uses it; others harmless) |
| `loaded_adam_t`       | `loaded_opt_step`     | Same, on load path |
| `state.adam_t`        | `state.opt_step`      | Same, in TrainState |

Pick *one* naming convention (suggested `m1`/`m2` since they map onto
"first moment" / "second moment" which is the actual mathematical
content). Avoid `s` (RMSprop) or `v` (Adam) standalone — those alias
to the existing names.

## Scope of changes

- `src/cuda/agpt_train.cu`: `cudaMalloc(&d_adam_m, ...)` and
  `cudaMalloc(&d_adam_v, ...)` (around line 3887-3888); every read/write
  site; `append_optimizer_state` / `load_optimizer_state` signatures
  (around lines 1902, 1949).
- `src/cudax/` (v2 trainer, codex's tree): mirror the same rename for
  symmetry. Coordinate with codex-agpt.
- `src/agpt/*.cr`: grep for `adam_m`, `adam_v` in any Crystal tool
  that reads/writes the model file footer.
- **Wire format**: the on-disk format does NOT change. `OPT_MAGIC` +
  `total_floats` + `adam_t` + `m1_buf` + `m2_buf` stays byte-identical;
  only the C++ symbol names change. Existing saved checkpoints still
  load.

## Bonus cleanup while you're in there

- RMSprop currently still allocates and serializes the unused `m1`
  buffer (zero-fill, ~433 KB on disk for d=64 L=2). Acceptable as-is
  but a `--optimizer rmsprop` switch could skip the `m1` allocation
  + footer block. Minor savings; only worth doing if rename touches
  the same lines anyway.

## When

After current init-bug investigation and any in-flight parity work.
Pure refactor, no behavior change; defer until there's a quiet patch
window. Touches ~30-50 call sites across two trainers; estimate
30 minutes including build verification.
