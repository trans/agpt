# cudax-section2-progressive

**Status:** active

## Hypothesis

Re-run progressive-growth CUDAX benchmarks after restoring the Section 2
event-count weighting term. The previous progressive table is not comparable
because compressed radix queries were effectively weighted as unique rows
instead of corpus events.

## Results

<!-- agpt-experiment-table:start -->
| Run ID | fixed_token_ppl | rolling_byte_ppl | bits/byte | train (s) | total (s) |
|--------|----------------:|-----------------:|----------:|----------:|----------:|
| `20260526T224149-section2-16x6` | 7.567 | 7.9789 | 2.9962 | 313.0 | 345.0 |
| `20260526T224750-section2-64x6` | 7.0247 | 7.5008 | 2.9071 | 1179.0 | 1208.0 |
| `20260526T231511-section2-256x6` | 6.9358 | 7.3926 | 2.8861 | 4750.0 | 4777.0 |
<!-- agpt-experiment-table:end -->

## Notes

Initial rerun set: `16x6`, `64x6`, `256x6`.
