# CUDAX Progressive Growth Experiments

## Protocol

- Trainer: `bin/agpt_train_v2 --mode train-growth`
- Corpus: `data/input.txt`
- Model init: `data/input.model`
- Depth: `--growth-max-depth 16`
- Chunk queries: `--chunk-queries 50000`
- Optimizer: `--optimizer rmsprop`
- Learning rate: `--lr 0.003`
- Schedule: `--lr-schedule warmup-cosine --warmup-epochs 0`
- Ancestor gradients: `--anc-grad`
- Held-out split: train on prefix-95, evaluate on tail 5%
- Final prefix frontier: `1059623`
- Growth frontiers can now be generated directly with:
  `--growth-divisions N --growth-train-frac 0.95`
- Eval range: `[1059624, 1069624)` using 10k targets
- Eval tool:
  `python src/tools/agpt_ppl.py --file data/input.txt --vocab-file data/input.txt --d 16 --eval-tail-frac 0.05 --max-positions 10000 --mode both --device cpu --batch-size 256`

## Results

| run | wall | optimizer steps | fixed PPL | uniform PPL |
| --- | ---: | ---: | ---: | ---: |
| static prefix-only 10 SE | 62.93s | 650 | 8.5857 | 9.2964 |
| progressive 10x1 | 41.22s | 636 | 9.3219 | 9.8117 |
| progressive 16x1 | 1m03s | 1015 | 8.5456 | 9.2195 |
| progressive 16x3 | 2m56.966s | 3045 | 7.2063 | 8.6266 |
| progressive 16x6 | 5m41.642s | 6090 | 6.8535 | 8.1846 |
| progressive 16 ramp 3..10 | 7m5.058s | 6204 | 6.4405 | 7.9689 |
| progressive 16x10 | 9m34.979s | 10150 | 6.2447 | 8.0943 |
| progressive 64x1 | 3m41.842s | 4047 | 7.1758 | 8.3680 |
| progressive 64x3 | 10m49.957s | 12141 | 6.4774 | 8.4878 |
| progressive 64x6 | 21m31.694s | 24282 | 5.8857 | 8.0540 |
| progressive 64 ramp 3..10 | 26m54.992s | 24746 | 5.4312 | 7.3321 |
| progressive 64x10 | 35m52.084s | 40470 | 5.8108 | 8.1543 |
| progressive 256x1 | 13m56.055s | 16167 | 6.7170 | 8.6781 |
| progressive 256x3 | 39m34.213s | 48501 | 5.8903 | 8.4176 |
| progressive 256x6 | 78m32.639s | 97002 | 5.7378 | 7.2617 |
| progressive 256 ramp 1..6 | 52m11.891s | 49300 | 5.2731 | 7.8485 |
| progressive 256 ramp 3..10 | 89m43.653s | 97945 | 5.0974 | 7.3630 |
| progressive 256 ramp 3..14 | 126m31.978s | 130806 | 5.2955 | 7.4388 |
| progressive 256x10 | 135m30.084s | 161670 | 5.4945 | 7.8872 |
| progressive 512 ramp 3..10 | 185m16.648s | 195992 | 4.9878 | 7.6500 |
| progressive 1024x1 | 56m03.639s | 64646 | 5.4516 | 7.9755 |
| progressive 1024x3 | 152m49.961s | 193938 | 5.4391 | 7.4354 |
| progressive 4096x1 | 210m33.814s | 258547 | 5.2960 | 7.7038 |

## Notes

- `256x6` has the best uniform PPL in this set.
- Fixed PPL improves faster than uniform PPL as training volume increases.
- More divisions help, but the cost scales strongly with optimizer steps.
- At 10 epochs per stage, fixed PPL keeps improving while uniform PPL regresses
  for both `64x10` and `256x10`, suggesting the added training is specializing
  toward the fixed/high-mass criterion rather than improving broad coverage.
- The linear 1..6 epoch ramp is compute-efficient and gives the best fixed PPL
  so far, but it does not recover the `256x6` uniform result.
- The linear 3..10 epoch ramp uses nearly the same update budget as `256x6`
  and comes close on uniform PPL while substantially improving fixed PPL.
- `4096x1` improves over `1024x1`, but the gain is too small for its runtime
  and it still trails the best 256-division regimes.
- `16 ramp 3..10` improves uniform PPL over `16x10` with a much smaller update
  budget, supporting the idea that ramped per-stage epochs are useful.
- `64 ramp 3..10` is a strong same-budget improvement over `64x6`, nearly
  matching `256 ramp 3..10` uniform PPL at much lower wall time.
- `1024x3` improves over `1024x1`, but does not beat the best 64/256 ramped
  regimes despite substantially higher runtime.
- `256 ramp 3..14` regresses relative to `256 ramp 3..10`; adding more
  late-stage training did not push fixed PPL below 5.
- `512 ramp 3..10` is the first sub-5 fixed PPL result, but its uniform PPL
  trails the best 64/256 regimes.
