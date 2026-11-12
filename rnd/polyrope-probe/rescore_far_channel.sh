#!/bin/sh
# Rescore the far-channel probe's w16 and exact (reach 128) checkpoints on this
# probe's targets (>= 1040 preceding characters). CPU; the checkpoints are local.
set -e
for seed in 42 43 44 45; do
  for arm in w16 exact; do
    src=rnd/far-channel-probe/20261111T002822-30k-$arm-s$seed/model.pt
    out=rnd/polyrope-probe/rescore/far-channel-$arm-r128-s$seed
    [ -f "$out/eval.json" ] && continue
    python3 src/tools/agpt_polyrope_probe.py --arm $arm --reach 128 --eval-only "$src" \
      --device cpu --threads 2 --eval-batch 4 --out "$out" | tail -1
  done
done
