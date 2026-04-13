# AGPT Float Precision Discrepancy — Needs Verification

*Created: 2026-04-13*

## Observation

The same AGPT configuration (d32n1, seq16, 500 starts, seed 42, 1 epoch) produces different loss values on CPU vs GPU:

```
CPU (crystal backend): loss = 3.7650
GPU (cublas backend):  loss = 3.7677
```

Difference: **0.0027** (0.07%)

## Assumed Explanation

The working assumption is that this is a float32 accumulation order difference in the softmax kernel:

- CPU `softmax_rows`: iterates rows sequentially, may use higher intermediate precision
- GPU `softmax_rows_kernel`: uses shared-memory parallel reduction in float32

The batched loss path was verified to match the per-node path exactly on CPU (both give 3.7650), so the batched loss code itself is not the source.

## Why This Needs Verification

Codex encountered a similar situation where a "float ordering" explanation masked an actual bug (float32 vs float64 accumulation in backward). The 0.07% discrepancy here is larger than typical float32 reordering noise for a simple softmax, which suggests it might be worth investigating further.

## Things to Check

1. **Does the CPU softmax use float64 intermediates?**
   Check `CrystalBackend#softmax_rows` in `src/microgpt/backend.cr` — does it accumulate `exp` sum in Float64?

2. **Does the GPU softmax kernel use float32 throughout?**
   Check `softmax_rows_kernel` in `src/cuda/kernels.cu` — shared memory is `float`, reductions are `float`.

3. **Is the logits copy in the batched loss path losing precision?**
   Line ~106 in `trie_walk_trainer.cr` copies `results[i].logits[0, j]` (Float32) into `logits_batched[i, j]` (Float32). If the source logits are GPU-backed and the Mat `[]` accessor does a float32 round-trip through CPU, this should be lossless. But verify.

4. **Is the gradient affected too?**
   The gradient (`grad[0, j] = all_probs[prob_offset + j]`) is built from the downloaded probs. If the probs differ by 0.07%, the gradient differs too. Over many training steps this could compound.

5. **Does the old per-node GPU path give the same 3.7677?**
   If so, it's the GPU softmax kernel, not the batching. If the old path gives 3.7650 on GPU, the batching introduced the discrepancy.

## Resolution

Either:
- Confirm it's a benign GPU softmax precision difference (document and accept)
- Find and fix the actual precision bug if one exists
- Consider using float64 accumulation in the GPU softmax kernel (`double` for sum/max reduction in shared memory)

## Files Involved

- `src/agpt/trie_walk_trainer.cr` — batched loss computation (lines ~98-140)
- `src/agpt/weighted_loss.cr` — per-node loss with bulk probs download
- `src/cuda/kernels.cu` — `softmax_rows_kernel`
- `src/microgpt/backend.cr` — `CrystalBackend#softmax_rows`, `CuBLASBackend#softmax_rows`
