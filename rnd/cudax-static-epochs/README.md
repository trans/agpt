# cudax-static-epochs

**Status:** initial batch complete

## Hypothesis

Straight full-tree training should provide a baseline for the progressive
growth runs. If the rolling PPL irregularity is caused by progressive staging,
then non-progressive runs should behave more monotonically as epoch count
increases.

## Scope

- Trainer: `bin/agpt_train_v2`
- Mode: `train-growth` with `growth_divisions: 1`
- Interpretation: static full-prefix tree, not progressive growth
- Corpus: `data/input.txt`
- Training split: prefix 95%
- Evaluation split: held-out tail 5%
- Model init: `data/input.model`
- Depth/window: `growth_max_depth: 16`
- Optimizer: RMSProp, `lr=0.003`, `rmsprop_beta=0.999`
- Schedule: `warmup-cosine`, `warmup_epochs=0`
- Ancestor gradients: enabled

## Results

<!-- agpt-experiment-table:start -->
| Run ID | fixed_token_ppl | rolling_byte_ppl | bits/byte | train (s) | total (s) |
|--------|----------------:|-----------------:|----------:|----------:|----------:|
| `20260526T172856-static-1ep` | 15.8772 | 16.1633 | 4.0146 | 10.0 | 40.0 |
| `20260526T172936-static-3ep` | 11.7741 | 12.3392 | 3.6252 | 22.0 | 52.0 |
| `20260526T173028-static-6ep` | 10.2582 | 11.0735 | 3.469 | 40.0 | 68.0 |
| `20260526T173136-static-10ep` | 9.169 | 10.1812 | 3.3478 | 63.0 | 93.0 |
| `20260526T173309-static-25ep` | 7.5187 | 8.5709 | 3.0994 | 156.0 | 185.0 |
| `20260526T180318-static-100ep` | 6.0879 | 7.8023 | 2.9639 | 601.0 | 629.0 |
<!-- agpt-experiment-table:end -->

## Conclusion

The static epoch sweep is monotonic in both fixed-token and rolling-byte PPL:
more epochs improve both metrics through 100 epochs. This differs sharply from
the progressive-growth table, where rolling PPL regressed for some larger
division counts. That points suspicion back at progressive staging/order or
its interaction with the optimizer, not at CUDAX training in general.
