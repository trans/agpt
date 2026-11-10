#!/usr/bin/env bash
# Superposed-tail RoPE probe sweep: 4 arms x 2 seeds, 20k steps, run one at a time.
set -euo pipefail
cd "$(dirname "$0")/../.."
stamp=$(date -u +%Y%m%dT%H%M%S)
for seed in 42 43; do
  for arm in none fourier wavelet decay; do
    out="rnd/superposed-tail-rope/${stamp}-sweep20k/${arm}-s${seed}"
    python3 src/tools/agpt_tail_rope_probe.py --arm "$arm" --seed "$seed" \
      --steps 20000 --eval-every 5000 --out "$out" > /dev/null
    echo "$(date -u +%H:%M:%S) $arm s$seed $(python3 -c "import json;m=json.load(open('$out/metrics.json'));print(round(m['heldout_nll'],4), m['by_position'])")"
  done
done
