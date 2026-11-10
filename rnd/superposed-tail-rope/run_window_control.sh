#!/usr/bin/env bash
# Positive control: the baseline (no tail) with longer softmax windows, 32 and 64 chars.
# If a wider window helps and the superposed tail does not, the useful older
# information needs softmax selection or content keys, not character-offset sums.
set -euo pipefail
cd "$(dirname "$0")/../.."
stamp=$(date -u +%Y%m%dT%H%M%S)
for seed in 42 43; do
  for w in 32 64; do
    out="rnd/superposed-tail-rope/${stamp}-window-control20k/none-w${w}-s${seed}"
    python3 src/tools/agpt_tail_rope_probe.py --arm none --window "$w" --seed "$seed" \
      --steps 20000 --eval-every 5000 --out "$out" > /dev/null
    echo "$(date -u +%H:%M:%S) w$w s$seed $(python3 -c "import json;m=json.load(open('$out/metrics.json'));print(round(m['heldout_nll'],4), m['by_position'])")"
  done
done
