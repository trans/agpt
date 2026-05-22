# Retire bin/agpt_sliding_window_perplexity (Crystal) in favor of Python

## Status

Both `bin/agpt_sliding_window_perplexity` (Crystal, `src/tools/agpt_sliding_window_perplexity.cr`) and `python src/tools/agpt_ppl.py` exist and compute identical PPL. Cross-validated 2026-05-22 to 4 decimals on the same checkpoint + held-out:

  fixed     Python 9.8997   Crystal --pool deep_only 9.8997  ✓
  uniform   Python 9.8666   Crystal --pool uniform   9.8666  ✓

## Why we kept the Crystal tool for now

1. Speed — Crystal + OpenBLAS multithreaded, runs faster than PyTorch CPU
   for big sweeps. 3-seed parity tests today were ~60s/seed.
2. Cross-check value — having two independent implementations producing
   the same number is a stronger correctness signal than either alone.
   Lose one, lose that.
3. Non-trivial removal — `test_agpt_fundamentals.sh`, `Justfile quick-test`,
   and any scripts using it need to switch over. Not free.

## Why we should retire it eventually

1. Crystal tool shares CUDA kernels with the trainer (via build/kernels.o
   when built with --backend cublas). The whole point of having an
   independent judge is to NOT share code with the trainer.
2. Crystal tool depends on lib/microgpt (Mat, MiniGPT, Config). The
   microgpt severance arc says we want agpt-side tools to not pull from
   microgpt where avoidable.
3. PyTorch is the industry-standard reference; numbers from it are
   easier to communicate and cross-check externally.

## Plan when picked up

1. Pick a couple of upcoming experiments. Run BOTH evaluators on the
   results. Confirm Python and Crystal agree to ≥ 4 decimals on
   real workloads (not just the smoke test we already did).
2. Switch `test_agpt_fundamentals.sh` Test 4 to call Python instead.
3. Switch `quick-test` (Justfile) to Python.
4. Grep for any other call sites (`bin/agpt_sliding_window_perplexity`
   in rnd/, scripts/, sweep .sh, etc.). Replace.
5. Delete `src/tools/agpt_sliding_window_perplexity.cr` and the Justfile
   target `build-agpt-sliding-window-perplexity`.
6. Drop the prereq check from `test_agpt_fundamentals.sh`.

## Effort

~1-2 hours. Pure cleanup; no architecture changes. Pick this up after
the active research arc (cap-fold / RoPE-as-mass / etc.) hits a quiet
patch.

## Watch-out

If Python and Crystal ever DISAGREE on a real-world result, that's
information — don't paper over it. Investigate first (kernel diff,
load-order bug, edge case). Only then decide which to keep.
