#!/bin/sh
# PolyRoPE probe: arms at reach 1024, 30k steps, one run at a time (the GPU is shared).
# Waits until no other process holds the GPU before each run.
# Usage: sh rnd/polyrope-probe/run_arms.sh RUN_TIMESTAMP "ARM ARM ..." SEED [SEED ...]
set -e
TS=$1; ARMS=$2; shift 2
for seed in "$@"; do
  for arm in $ARMS; do
    out=rnd/polyrope-probe/$TS-r1024-$arm-s$seed
    [ -f "$out/metrics.json" ] && continue
    while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 120; done
    python3 src/tools/agpt_polyrope_probe.py --arm $arm --reach 1024 --steps 30000 --seed $seed \
      --threads 2 --out "$out" > /dev/null 2>&1 || true
    echo "$arm s$seed $(tail -1 "$out/train.log")"
  done
done
