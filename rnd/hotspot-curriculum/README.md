# Hotspot Curriculum

**Status**: incomplete

**Trainer note**: likely post-fix, but needs writeup to confirm scope.

**Code**: commits `0511735`, `c94cf73`, `797e14b`, plus extra logs in `2af0a80`
(all 2026-04-24 to 2026-04-25).

## Hypothesis

Per-subtree residual loss is highly concentrated: a small number of root-child
subtrees carry most of the remaining error mass late in training. If AGPT
adapts its subtree partitioning to focus more on those hotspots, it might
learn faster or reach a better final PPL than a uniform per-root-child sweep.

## What was built

Two main mechanisms landed in `agpt_train` during this thread:

1. **Residual measurement** (`0511735`)
   Compute per-subtree excess-loss score during normal training:

   `score[rc] = mass[rc] × max(avg_loss[rc] − global_avg_loss, 0)`

2. **Adaptive hotspot splitting** (`c94cf73`)
   New flag `--hotspot-coverage F`:
   between epochs, split the highest-scoring subtrees until they cover `F` of
   the total excess-loss.

A later sweep (`797e14b`) added per-subtree LR rules:

- `none`
- `inv-depth`
- `inv-sqrt-depth`
- `sqrt-batch`
- `residual`

## Results

Headline numbers recorded directly in commit `c94cf73`:

| config | mean PPL | min |
|---|---:|---:|
| d=16 base 3SE | 14.44 | 13.96 |
| d=16 hs0.5 3SE | **14.20** | **13.77** |
| d=16 hs0.8 3SE | 14.76 | 14.29 |
| d=16 hs0.8 4SE | 15.26 | 14.22 |
| d=32 base 3SE | 13.83 | 13.22 |
| d=32 hs0.5 3SE | **13.42** | **13.21** |
| d=32 hs0.8 3SE | 13.63 | 13.12 |
| d=32 hs0.8 4SE | 14.41 | 13.63 |

Reading from that commit:

- `hs=0.5` gave a modest win at both `d=16` and `d=32`
- `hs=0.8` was too aggressive and regressed
- adding more epochs at high coverage overfit the rapidly-growing hotspot list

The later LR-rule sweep (`797e14b`) found that differentiated LR multipliers
did not help under RMSProp:

| rule | d=16 mean | d=32 mean |
|---|---:|---:|
| none | **15.90** | **13.78** |
| inv-depth | 15.59 | 14.50 |
| inv-sqrt-depth | 16.02 | 13.90 |
| sqrt-batch | 15.88 | 16.97 |
| residual | diverged | diverged |

So the best version of the hotspot idea was the structural splitting itself,
not the later per-subtree LR scaling.

## Additional logs

Commit `2af0a80` added missed logs for:

- `hs0.5` at 4 and 5 super-epochs
- extra SGD reference logs used elsewhere

The commit message states that the added `hs0.5` SE sweep supported the idea
that simply running more super-epochs did not solve the curriculum problem.

## Conclusion

This thread produced a real positive signal:

- adaptive hotspot splitting looked mildly helpful at moderate coverage
- aggressive coverage and LR tricks hurt

But it did **not** finish cleanly enough to count as a fully closed experiment.
The commit itself pointed to the next step:

- staged curriculum
- broad warmup first
- begin splitting only after the model is less random

That follow-up does not appear to have been written up here, so the thread is
best treated as **incomplete but promising**.

## Reproduction note

No run script was preserved in this directory. The logs and the commit
summaries are the main record. If revived, rebuild the recipe from commits
`c94cf73` and `797e14b`.
