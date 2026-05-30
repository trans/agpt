# cudax-growth-heldout-rerun

**Status:** active

## Hypothesis

The earlier CUDAX progressive-growth numbers should be reproducible under the
YAML experiment harness with an explicit held-out tail split. If the progressive
schedule is useful, increasing the number of epochs per growth frontier should
improve held-out byte PPL, but gains may saturate quickly.

## Scope

- Trainer: `bin/agpt_train_v2`
- Mode: `train-growth`
- Growth divisions: 16
- Epochs per frontier: 1, 3, 6
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
| Run ID | byte_ppl | bits/byte | word_ppl | wall (s) |
|--------|---------:|----------:|---------:|---------:|
| `20260526T041157-progressive-16x1` | 9.5708 | 3.2586 | 305118.35 | 85.0 |
| `20260526T041333-progressive-16x3` | 8.5556 | 3.0969 | 162996.4 | 183.0 |
| `20260526T041703-progressive-16x6` | 8.5672 | 3.0988 | 164240.94 | 394.0 |
<!-- agpt-experiment-table:end -->

## Conclusion

The 3-epoch and 6-epoch progressive runs land close together on held-out byte
PPL: 8.5556 vs 8.5672. The 1-epoch run is worse at 9.5708. In this batch,
extra work beyond 3 epochs per frontier did not produce a clear held-out
improvement.
