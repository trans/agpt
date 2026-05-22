# AGPT CUDA Trainer V2

This directory is the clean-room replacement for the current standalone CUDA
trainer in `src/cuda/agpt_train.cu`.

## Scope

V2 is a trainer-core rewrite, not a model rewrite.

It is meant to preserve:

- the AGPT objective
- the radix/trie file formats
- the current model weight layout where practical
- reusable CUDA kernels from the existing engine

It is meant to replace:

- the monolithic radix training loop
- implicit cache/coordinate conventions
- the current accumulation of feature flags in one execution path

The baseline contract today is the `pd=1` path:

- unigram / root-child subtrees
- parent-before-child depth order
- chunks as memory slices, not optimizer boundaries
- one optimizer step per training unit
- persistent compact K/V cache across chunks in a unit
- explicit `pre-RoPE` / `post-RoPE` coordinate spaces

## Initial target

The first working V2 milestone is intentionally narrow:

- radix trainer only
- `partition_depth = 1`
- no Lightning
- no virtual cycles
- baseline AGPT train/eval parity only
- compare against the legacy trainer on Shakespeare `d16`

## Core contracts

V2 should make these explicit:

1. Training unit boundary
   Weights are fixed for a training unit; optimizer fires at unit end.

2. Cache coordinate space
   K cache space must be explicit (`pre-RoPE` vs `post-RoPE`).
   V cache space must be explicit.

3. Backward coordinate space
   Weight-gradient reductions must state whether they consume pre- or
   post-RoPE gradients.

4. Forward-state ownership
   Saved activations, cache state, and packed attention scratch should have
   separate owners.

5. Persistence boundary
   K/V cache entries written by earlier chunks in the same training unit must
   remain readable by later chunks. Chunk boundaries are for memory slicing,
   not cache resets.

6. Runtime ownership
   Reusable resources such as cuBLAS handles and RoPE tables should live in
   the runtime object, not inside the hot path of the passes.

## Current milestone

The current scaffold now owns a validated baseline execution path:

- host chunk metadata build
- device chunk upload
- embedding and final `LN + logits + baseline AGPT chunk loss`
- per-layer `LN1 -> Q/K/V -> RoPE -> compact-cache scatter`
- per-layer packed `[ancestors | own-edge]` K/V gather
- per-layer L-query variable-length attention forward
- per-layer `WO + residual`
- per-layer `LN2 + FFN + residual`
- full reverse pass through output head, transformer layers, and embeddings
- optional descendant→ancestor scatter + fire-end `Wk/Wv` reduction via `--anc-grad`
- stateful SGD/RMSProp one-step checks
- save/reload checks for weights and optimizer state
- `train-small` and `train-epoch` accumulation modes

The current baseline Shakespeare `d16` result is aligned with the legacy
trainer within the observed run-to-run spread.

## CLI shape

The v2 binary is separate from the legacy trainer:

- `bin/agpt_train_v2`

Its execution surface should stay explicit and narrow. The primary entrypoint is:

- `--mode plan`
- `--mode instantiate-runtime`
- `--mode upload`
- `--mode forward`
- `--mode backward-head`
- `--mode one-step-sgd`
- `--mode one-step-rmsprop`
- `--mode multi-step-sgd`
- `--mode multi-step-rmsprop`
- `--mode save-reload-sgd`
- `--mode save-reload-rmsprop`
- `--mode train-epoch`
- `--mode train-small`
- `--anc-grad`
- `--lr-schedule constant|warmup-cosine`
- `--warmup-epochs N`

Older flags like `--instantiate-runtime`, `--instantiate-chunk-upload`,
`--run-forward-prefix`, and `--run-backward-head` are only compatibility aliases.

`--mode one-step-sgd` is the first trainer-core update sanity check:
- run forward on the first planned chunk
- run backward on that same chunk
- apply one plain SGD update
- rerun forward on the same chunk
- report `loss_before`, `loss_after`, and `delta`

`--mode one-step-rmsprop` runs the same test with a single RMSProp update,
using the v2 optimizer buffer as adaptive state.

`--mode multi-step-sgd` repeats the same-chunk SGD sanity cycle multiple
times and prints a short loss trace. Use `--steps N` to control the repeat
count.

`--mode multi-step-rmsprop` does the same for RMSProp and keeps the adaptive
state live across steps so you can compare it to SGD directly.

`--mode save-reload-sgd` applies one SGD step, writes the updated weights to a
checkpoint, reloads them, compares the bytes, and reruns forward from the
reloaded weights.

`--mode save-reload-rmsprop` does the same while also round-tripping the
stateful RMSProp accumulator in a sidecar file.

`--mode train-epoch` runs the baseline `pd=1` trainer across all training
units, accumulating gradients across chunks inside each unit and applying one
stateful RMSProp update per unit. Use `--units N` to cap the number of units
for a short smoke test. If `--save PATH` is provided, the final weights are
written there so you can run the legacy `bin/perplexity` tool on the result.
`--lr-schedule warmup-cosine` enables a unit-level linear warmup followed by a
cosine decay across the full `train-epoch` run. `--warmup-epochs N` controls
how many epochs of unit updates are used for warmup.

`--mode train-small` runs a tiny real training-unit loop over the first few
chunks of the largest `pd=1` unit, accumulates gradients across those chunks,
then fires one stateful RMSProp update. The chunks are just memory slices; the
optimizer step happens once per unit.

## Extracted modules

- `types.cuh`
- `model_layout.cuh`
- `cache_layout.cuh`
- `training_unit.cuh`
- `chunk_plan.cuh`
- `chunk_metadata_v2.cuh`
- `chunk_upload_v2.cuh`
- `runtime_contracts.cuh`
- `runtime_objects.cuh`
- `buffer_layout_v2.cuh`
- `forward_stages_v2.cuh`
- `backward_stages_v2.cuh`
- `kernels_v2.cuh`
- `checkpoint_io_v2.cuh`
- `forward_pass.cuh`
- `backward_pass.cuh`
- `optimizer_step.cuh`
- `agpt_train_v2.cu`

## Feature re-entry strategy

1. Keep the baseline `pd=1` path frozen as the reference execution model.
2. Add new features as planner/executor modules, not inline branches in the core passes.
3. Keep the top-level binary thin: parse args, build a plan, choose a mode, dispatch to a coordinator.
4. Add parity checks before reintroducing each feature family.
5. Treat this README as the contract for ownership, coordinate spaces, and execution boundaries.
