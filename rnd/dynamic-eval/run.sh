#!/bin/sh
# Dynamic-evaluation grid (CPU only, no slot: d64/L2, well under 2 GB).
# Two streams of: checkpoint x {own past, other chunk's text (control)}.
# Usage: sh rnd/dynamic-eval/run.sh > rnd/dynamic-eval/run.out 2>&1
renice -n 19 -p $$ > /dev/null
ionice -c3 -p $$
export OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
D=rnd/dynamic-eval
R=rnd/refined-trie-counts
mkdir -p $D/results
SET=static,full-adam-3e-4,full-adam-1e-3,full-adam-3e-3,lora4-adam-1e-3,lora4-adam-3e-3,lora4-adam-1e-2,lora1-adam-3e-3,lora16-adam-3e-3,full-sgd-0.01,full-sgd-0.03
one() {  # one SLUG RUNDIR PAST
  echo "$(date -u +%H:%M) $1 $3"
  python3 $D/dyneval.py --model $R/$2/checkpoint.model --vocab data/pg19/imm-split/vocab.txt \
    --chunks data/pg19/imm-split/heldout_chunks --out $D/results/$1-$3.json --past $3 --settings $SET \
    > $D/results/$1-$3.log 2>&1 || echo "  FAILED $1 $3"
}
{ one mb-a1m 20261118T055241-mb-a1m own; one mb-a1m-ref100m 20261118T053638-mb-a1m-ref100m own
  one mb-100m-st100 20261119T045935-mb-100m-st100 own; one agpt-pd1-ref100m 20261118T124830-agpt-a1m-pd1-ref100m own; } &
{ one mb-a1m 20261118T055241-mb-a1m other; one mb-a1m-ref100m 20261118T053638-mb-a1m-ref100m other
  one mb-100m-st100 20261119T045935-mb-100m-st100 other; one agpt-pd1-ref100m 20261118T124830-agpt-a1m-pd1-ref100m other; } &
wait
echo "$(date -u +%H:%M) all done"
