#!/bin/sh
# Whole-book dynamic evaluation (CPU only, no slot: about 0.7 GB a stream).
# 20 PG-19 test books, characters 10,000-310,000 of each (make_books.py); books
# 0-9 tune, 10-19 report. Two streams, one per checkpoint, own past then the
# other-book control.
# Usage: sh rnd/dynamic-eval/run_books.sh > rnd/dynamic-eval/run_books.out 2>&1
renice -n 19 -p $$ > /dev/null
ionice -c3 -p $$
export OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
D=rnd/dynamic-eval
R=rnd/refined-trie-counts
B=data/pg19/test-books-300k
mkdir -p $D/results
[ -f $B/manifest.json ] || python3 $D/make_books.py --src ../mobius/data/pg19/test --out $B --n 20
BINS=0,1000,2000,5000,10000,20000,50000,100000,200000,300000
OWN=static,full-adam-1e-4,full-adam-3e-4,full-adam-1e-3,lora4-adam-1e-2,lora4b-adam-3e-3,lora4b-adam-1e-2
OTHER=full-adam-1e-4,full-adam-3e-4,full-adam-1e-3
one() {  # one SLUG RUNDIR PAST SETTINGS
  echo "$(date -u +%H:%M) $1 $3"
  python3 $D/dyneval.py --model $R/$2/checkpoint.model --vocab data/pg19/imm-split/vocab.txt \
    --chunks $B --out $D/results/books-$1-$3.json --past $3 --settings $4 --bins $BINS \
    > $D/results/books-$1-$3.log 2>&1 || echo "  FAILED $1 $3"
}
{ one mb-a1m 20261118T055241-mb-a1m own $OWN; one mb-a1m 20261118T055241-mb-a1m other $OTHER; } &
{ one mb-100m-st100 20261119T045935-mb-100m-st100 own $OWN
  one mb-100m-st100 20261119T045935-mb-100m-st100 other $OTHER; } &
wait
echo "$(date -u +%H:%M) all done"
