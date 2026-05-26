# cudax-exposure-fix

**Status:** active

## Hypothesis

(fill in)

## Scope

(fill in)

## Results

Note: `20260526T183923-static-25ep` was produced by the intermediate
per-chunk-normalized exposure patch. It is invalid for Section 2 parity
and should not be used as evidence.

<!-- agpt-experiment-table:start -->
| Run ID | fixed_token_ppl | rolling_byte_ppl | bits/byte | train (s) | total (s) |
|--------|----------------:|-----------------:|----------:|----------:|----------:|
| `20260526T183923-static-25ep` | 8.4962 | 8.9954 | 3.1692 | 157.0 | 186.0 |
| `20260526T191259-static-25ep-section2` | 8.5035 | 8.7956 | 3.1368 | 154.0 | 184.0 |
| `20260526T191714-static-10ep-section2` | 10.0784 | 10.2799 | 3.3618 | 66.0 | 93.0 |
| `20260526T192559-static-100ep-section2` | 6.6623 | 7.0634 | 2.8204 | 600.0 | 627.0 |
<!-- agpt-experiment-table:end -->

## Conclusion

(fill in once enough runs have landed)
