# Window Baseline vs CUDAX Section 2

Status: recorded comparison from the 2026-05-27/28 Shakespeare tail-heldout runs.

## Scope

This note records the corrected comparison between the standard window baseline
and the current CUDAX Section 2 trainer results.

Important correction: run names such as `d128L6` mean `d_model=128`,
`n_layers=6`. They do not mean tree depth 128. The CUDAX runs below used
`growth_max_depth: 16`, and the window baseline used `seq_len=16`.

## Baseline

Window baseline:

- trainer: `bin/microgpt`
- model: `d_model=128`, `n_layers=6`, `d_ff=512`
- context: `seq_len=16`
- steps: `180000`
- optimizer: likely Adam-style microgpt training; do not label this as SGD
- checkpoint: `rnd/window-baseline-wallmatch/20260528T014059-window-adam-d128l6-s16-180k/checkpoint.model`
- recovered eval: `rnd/window-baseline-wallmatch/20260528T014059-window-adam-d128l6-s16-180k/eval_recovered.json`

The original post-run eval failed because `datasets/lm_eval` reported an
insufficient disk-space condition. The checkpoint was valid; rerunning eval with
the cache redirected into the run directory recovered the metrics.

## Results

| Run | Model | Depth/window | Fixed PPL | Rolling byte PPL |
|---|---:|---:|---:|---:|
| Window baseline | `d_model=128`, `L=6` | `seq_len=16` | `6.3818` | `6.9604` |
| CUDAX progressive `16x25` | `d_model=128`, `L=6` | tree depth `16` | `6.1003` | `6.4484` |
| CUDAX static `100` | `d_model=128`, `L=6` | tree depth `16` | `5.8981` | `6.4163` |
| CUDAX static `200` | `d_model=128`, `L=6` | tree depth `16` | `5.6529` | `6.1636` |

Current best CUDAX result in this set is `d_model=128`, `L=6`, static 200
epochs:

- fixed improvement vs window baseline: `6.3818 -> 5.6529`
- rolling improvement vs window baseline: `6.9604 -> 6.1636`

## External Reference

KenLM Kneser-Ney remains stronger on the same Shakespeare split:

- order 6: rolling PPL `5.2490`
- order 8: rolling PPL `5.2374`

CUDAX does not yet beat the KN 8-gram reference, but the gap from the best
current CUDAX run to KN 8-gram is roughly `0.93` rolling PPL.

## Interpretation

This comparison is useful because the model size and effective context/tree
depth are aligned:

- window baseline: `d_model=128`, `L=6`, `seq_len=16`
- CUDAX: `d_model=128`, `L=6`, tree depth `16`

The window baseline should be treated as a standard window-transformer baseline,
not as a clean SGD baseline, until the microgpt optimizer semantics are pinned
down explicitly.
