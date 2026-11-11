#!/bin/sh
# Far-channel probe: the four arms at 30k steps, one run at a time (the GPU is shared).
# Usage: sh rnd/far-channel-probe/run_arms.sh RUN_TIMESTAMP SEED [SEED ...]
# Runs of 2026-11-11 used RUN_TIMESTAMP 20261111T002822 with seeds 42 43, then 44 45.
set -e
TS=$1; shift
for seed in "$@"; do
  for arm in w16 exact decay cq; do
    out=rnd/far-channel-probe/$TS-30k-$arm-s$seed
    [ -f "$out/metrics.json" ] && continue
    python3 src/tools/agpt_far_channel_probe.py --arm $arm --steps 30000 --seed $seed --out "$out" > /dev/null 2>&1
    echo "$arm s$seed exit=$? $(tail -1 "$out/train.log")"
  done
done
