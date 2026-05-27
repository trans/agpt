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
| `20260527T025333-section2-16x10` | 7.0592 | 7.5558 | 2.9176 | 523.0 | 554.0 |
| `20260527T030528-section2-64x10` | 6.9576 | 7.4577 | 2.8987 | 1983.0 | 2014.0 |
| `20260527T034618-section2-16x3to10` | 7.293 | 7.7103 | 2.9468 | 378.0 | 408.0 |
| `20260527T035317-section2-64x3to10` | 6.8859 | 7.3954 | 2.8866 | 1423.0 | 1453.0 |
| `20260527T043319-section2-16x1to6` | 7.8375 | 8.2522 | 3.0448 | 213.0 | 243.0 |
| `20260527T044738-section2-16x1to10` | 7.6878 | 8.1091 | 3.0195 | 347.0 | 378.0 |
| `20260527T050713-section2-128x6` | 6.8494 | 7.3506 | 2.8779 | 2345.0 | 2374.0 |
| `20260527T055358-section2-16x25` | 6.5925 | 7.0509 | 2.8178 | 1265.0 | 1294.0 |
| `20260527T072703-section2-d128l6-16x25` | 6.1003 | 6.4484 | 2.6889 | 7621.0 | 7669.0 |
| `20260527T142250-section2-d128l6-static100` | 5.8981 | 6.4163 | 2.6818 | 4029.0 | 4080.0 |
<!-- agpt-experiment-table:end -->

## Notes

Initial rerun set: `16x6`, `64x6`, `256x6`.
