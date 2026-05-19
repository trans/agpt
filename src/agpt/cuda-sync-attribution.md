# AGPT CUDA Sync Attribution

*Updated: 2026-04-13*

This note records the current AGPT CUDA optimization state, what actually helped, what did not, and where the remaining CPU<->GPU communication cost is coming from.

## Scope

This is about the AGPT path only:

- `src/agpt/trie_walk_trainer.cr`
- `src/agpt/batched_depth_forward.cr`
- `src/agpt/batched_depth_backward.cr`
- `src/agpt/weighted_loss.cr`
- `src/microgpt/micro_gpt.cr`

It does **not** cover construction-kit or UI work.

## Benchmark Shape

Primary benchmark command:

```sh
./bin/microgpt data/input.txt --steps 1 --d-model 32 --n-layers 1 --seq-len 16 \
  --agpt --agpt-max-starts 500 --seed 42 --no-save --backend cublas
```

Heavier comparison run:

```sh
./bin/microgpt data/input.txt --steps 1 --d-model 32 --n-layers 1 --seq-len 16 \
  --agpt --agpt-max-starts 2000 --seed 42 --no-save --backend cublas
```

Tracing was enabled with:

```sh
MICROGPT_PERF_TRACE=1
```

## What Worked

### 1. Keep more of AGPT CUDA forward on GPU

Committed as:

```text
7ec0d72 Keep AGPT CUDA forward path on GPU
```

Main changes:

- AGPT embedding gather moved to GPU
- varlen attention output stayed GPU-backed into `WO`
- AGPT forward bias add / fused bias ReLU used backend GPU ops instead of CPU loops

Files:

- `src/agpt/batched_depth_forward.cr`
- `src/cuda/kernels.cu`
- `src/microgpt/backend.cr`

Effect on the primary benchmark:

- before: about `45.1s`
- after: about `11.9s`
- loss unchanged: `3.7650`

This was the first large CUDA-path win. The main issue was not the attention kernel itself. It was accidental GPU -> CPU -> GPU fallback around it.

### 2. Rewrite AGPT backward attention to avoid tiny matrix churn

Current branch work, validated but not yet part of `7ec0d72`.

Main changes in `src/agpt/batched_depth_backward.cr`:

- no per-node/head `k_slice` / `v_slice`
- no `dk_full` / `dv_full` temporary matrices
- direct read-only use of cached `k_parts` / `v_parts`
- direct ancestor gradient accumulation
- preserved `Float32` accumulation behavior to match the old path exactly

Validation:

- `spec/agpt_backward_attention_spec.cr`
- spec compares outputs, loss contribution, `dQ`, `dK`, `dV`, ancestor accumulation, and derived parameter-gradient contributions
- tested across multiple prefix lengths, including branching cases

Effect on the primary benchmark:

- before: about `12.6s`
- after: about `2.7s`
- loss unchanged: `3.7650`

Effect on the heavier benchmark:

- before: about `44.1s`
- after: about `6.5s`
- loss unchanged: `3.5696`

The original rewrite briefly changed some inner accumulations to `Float64`, which shifted the one-step loss slightly. That was fixed by restoring `Float32` accumulation behavior.

## What Did Not Help

### CPU sibling batching

This was the wrong target.

- it optimized the CPU fallback path rather than the active CUDA path
- the trie shape at this benchmark is heavily unary
- the extra grouping/allocation overhead made it slower

This was reverted.

### More aggressive CUDA forward refactor

An attempt to bypass more of `split_cols` and CPU RoPE in one shot regressed badly and was reverted.

Conclusion:

- the direction may still be valid
- the attempted implementation changed too much at once

### `split_cols` contiguous-copy tweak

This was measured and reverted.

It did not materially reduce hidden sync count. A code comment remains in `src/microgpt/micro_gpt.cr` so the same detour is less likely to be repeated.

### Forward row-extraction tweak

A small attempt to improve saved-state row extraction was flat to slightly worse and was reverted.

Conclusion:

- row extraction itself is not the main remaining bottleneck
- the real remaining sync sources were elsewhere

## What The Trace Showed

After the forward and backward attention wins, the primary traced CUDA run reported:

- epoch time about `2.9s`
- loss `3.7650`
- `sync_to_cpu.calls=22591`
- `sync_to_cpu=481.0ms`

The full sync count is now accounted for exactly:

- `agpt.loss.auto_sync=7145`
- `agpt.subtrie_loss_grads.auto_sync=7145`
- `agpt.backward.auto_sync=7997`
- `agpt.forward.auto_sync=304`

These sum to `22591`.

### Interpretation

The largest remaining communication source is not forward `split_cols`, not KV storage, and not saved-state extraction.

It is the weighted loss path being evaluated twice per observed node:

1. once in the main loss accumulation pass
2. once again when building `subtrie_grads`

That points directly at `src/agpt/weighted_loss.cr` and its use from `src/agpt/trie_walk_trainer.cr`.

## Backward Sync Breakdown

Within the `agpt.backward.auto_sync=7997` bucket, the largest identified source is:

- `agpt.backward.batched_ln_backward.sync=4797`

Repeated once-per-subtrie sync buckets also appear in:

- `agpt.backward.layer0.relu_backward.sync=800`
- `agpt.backward.layer0.residual1_copy.sync=800`
- `agpt.backward.layer0.split_heads.sync=800`
- `agpt.backward.embedding.sync=800`

Confirmed non-sources in the current trace:

- `agpt.backward.layer0.gather_state.sync=0`
- `agpt.backward.layer0.qkv_bias.sync=0`
- `agpt.backward.output_stack.sync=0`
- `agpt.backward.output_norm_out.sync=0`
- `agpt.backward.final_norm_prep.sync=0`

So the backward side is now much narrower:

- biggest remaining target: `batched_ln_backward`
- secondary cleanup: the repeated `800`-sync buckets

## Forward Sync Breakdown

Forward is now a small part of the remaining sync count:

- `agpt.forward.auto_sync=304`

Observed forward buckets:

- `agpt.forward.layer0.split_qkv.sync=48`
- `agpt.forward.layer0.state_extract.sync=80`
- `agpt.forward.layer0.kv_store.sync=0`
- `agpt.forward.layer0.qkv_rope.sync=0`

So forward still has some cleanup potential, but it is no longer the dominant communication problem.

## Memory Check

A single monitored heavier run used:

```sh
MICROGPT_PERF_TRACE=1 timeout 25s ./bin/microgpt ... --agpt-max-starts 2000 --backend cublas
timeout 25s nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits -l 1
```

Tracked host-side `Mat` usage remained modest:

- `mat.allocated_bytes=57.27 MiB`
- `agpt.forward_stage_bytes=42.76 MiB`
- `agpt.backward_stage_bytes=56.38 MiB`
- `agpt.update_stage_bytes=56.38 MiB`

Observed GPU memory:

- baseline about `93 MiB`
- during run about `245-267 MiB`
- card total `8188 MiB`
- temperature about `40-41 C`

This does **not** prove the earlier reboot was unrelated to memory, but this monitored one-step run did not operate near obvious host or GPU memory limits.

## Current Best Next Steps

Priority order from the measured data:

1. Stop doing weighted next-token loss twice per node.
   Reuse the loss/gradient result instead of recomputing it for `subtrie_grads`.

2. Move `batched_ln_backward` off the CPU fallback path.
   It is now the largest named backward-side sync source.

3. Clean up the repeated `800`-sync backward buckets.
   `relu_backward`, `residual1_copy`, `split_heads`, and embedding update are now clearly visible fixed-cost per-subtrie sync sources.

4. Revisit forward cleanup only after the above.
   Forward syncs are now comparatively small.

5. Keep unary suffix compression in view.
   It remains a meaningful structural optimization opportunity, especially because the trie is still heavily unary at these benchmark settings.

## Bottom Line

The original hypothesis that "forward attention / CPU packing is still the main remaining bottleneck" is no longer true after the last two wins.

The current picture is:

- the major forward CUDA round-trip problem was fixed
- backward attention temporary-matrix churn was fixed
- the largest remaining CPU<->GPU communication source is now duplicated weighted-loss work
- the largest named backward-side sync source is `batched_ln_backward`

That is the current optimization frontier.
