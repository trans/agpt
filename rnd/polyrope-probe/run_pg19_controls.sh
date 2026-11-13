#!/bin/sh
# PolyRoPE probe, PG-19 headroom check: how much does exact attention gain from
# offsets 129-1,023 on books (imm's 20M split), against Tiny Shakespeare's
# 0.006-0.009 nats/char? Three GPU runs back to back (exact at reach 1,024 and 128,
# then w16), then CPU rescores of the reach-1,024 model with its attention cut.
# Run it only inside a slot from @air's tools/slots.
# Usage: sh rnd/polyrope-probe/run_pg19_controls.sh RUN_TIMESTAMP SEED
set -e
renice -n 19 -p $$ > /dev/null
ionice -c3 -p $$
export OMP_NUM_THREADS=2
TS=$1; SEED=$2
PG=/home/trans/Projects/air/imm/data/pg19/imm-split-20m
DATA="--data $PG --vocab-file $PG/vocab.txt --piece-chars 25000"
for run in "exact 1024" "exact 128" "w16 1024"; do
  set -- $run
  out=rnd/polyrope-probe/$TS-pg19-r$2-$1-s$SEED
  [ -f "$out/metrics.json" ] && continue
  python3 src/tools/agpt_polyrope_probe.py --arm $1 --reach $2 --steps 30000 --seed $SEED \
    --threads 2 $DATA --out "$out" > /dev/null 2>&1 || true
  echo "$1 r$2 s$SEED $(tail -1 "$out/train.log")"
done
src=rnd/polyrope-probe/$TS-pg19-r1024-exact-s$SEED/model.pt
for r in 512 256 128; do
  python3 src/tools/agpt_polyrope_probe.py --arm exact --reach $r --eval-only "$src" --device cpu \
    --threads 2 --eval-batch 4 $DATA --out rnd/polyrope-probe/rescore/pg19-exact-r1024-s$SEED-at$r | tail -1
done
