# Progressive Growth vs SGD Baseline

Status: initial run set complete

## Question

Compare CUDAX progressive growth at several division/epoch schedules against a
standard Crystal µGPT sliding-window baseline at the same model size and context
length.

## Protocol

Each run is a subdirectory under this directory. The copied `config.yml`,
`resolved_config.json`, `meta.json`, `eval_raw.json`, and `result.json` in each
run directory are the committed canonical record for that run. Raw logs and
checkpoints remain local debugging artifacts.

## Runs

| Run | Purpose |
|---|---|
| `cudax-16x1` | 16 progressive divisions, 1 epoch per stage |
| `cudax-16x3` | 16 progressive divisions, 3 epochs per stage |
| `cudax-16x6` | 16 progressive divisions, 6 epochs per stage |
| `cudax-64x1` | 64 progressive divisions, 1 epoch per stage |
| `cudax-64x3` | 64 progressive divisions, 3 epochs per stage |
| `cudax-64x6` | 64 progressive divisions, 6 epochs per stage |
| `cudax-256x6` | 256 progressive divisions, 6 epochs per stage |
| `sgd-s16-10k` | Crystal µGPT SGD baseline, seq_len=16, 10k steps |

## Caveats

- `train (s)` is the primary timing column. For CUDAX runs it is the sum of
  trainer-reported `growth-stage-timing` totals. For the µGPT run it is
  approximated from the train log close time because the old µGPT binary did
  not report elapsed training time.
- `total (s)` is full harness time: split, training, HF conversion, rolling
  evaluation, fixed-token evaluation, and aggregation.
- Some runs overlapped with other GPU/CPU work, so timing is noisy. Treat PPL
  as the primary result from this batch.
- The SGD baseline is preliminary. A fair baseline needs explicit agreement on
  matching criterion: wall time, optimizer steps, target-token exposures, or
  some combination of these.

## Results

<!-- agpt-experiment-table:start -->
| Run ID | fixed_token_ppl | rolling_byte_ppl | bits/byte | train (s) | total (s) |
|--------|----------------:|-----------------:|----------:|----------:|----------:|
| `20260526T055319-cudax-64x1` | 7.2606 | 8.6681 | 3.1157 | 216 | 249 |
| `20260526T055745-cudax-64x3` | 6.8696 | 8.9280 | 3.1583 | 769 | 802 |
| `20260526T064328-cudax-64x6` | 6.7244 | 9.0170 | 3.1727 | 1681 | 1711 |
| `20260526T073057-cudax-16x1` | 9.2729 | 9.9573 | 3.3158 | 57 | 86 |
| `20260526T083945-cudax-16x3` | 7.4665 | 8.6196 | 3.1076 | 158 | 184 |
| `20260526T133412-cudax-16x6` | 6.8595 | 8.2664 | 3.0473 | 306 | 334 |
| `20260526T135924-cudax-256x6` | 6.7331 | 9.2635 | 3.2116 | 4604 | 4634 |
| `20260526T153756-sgd-s16-10k` | 10.0631 | 10.2344 | 3.3553 | 66 | 88 |
<!-- agpt-experiment-table:end -->
