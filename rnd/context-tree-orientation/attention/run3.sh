#!/bin/sh
# Attention grouping control, cosine check (cfirst, cwindow; make_configs.py):
# 4 runs in two streams (first, window), one pass of warmup then cosine to 0 over 16
# passes, peak 0.006 then 0.003. Slot 68 started them and gave way to another job two
# minutes in; they ran in slot 70.
# Run only inside a slot (sh /home/trans/Projects/air/tools/slots add @agpt ...).
# A failed run is reported and skipped.
# Usage: sh rnd/context-tree-orientation/attention/run3.sh
renice -n 19 -p $$ > /dev/null
ionice -c3 -p $$
export OMP_NUM_THREADS=2
D=rnd/context-tree-orientation/attention
one() {
  slug=$1; log=$D/last_run_$2.log
  bin/agpt_experiment --config $D/configs/$slug.yml --trainer src/tools/agpt_minibatch_train.py > $log 2>&1
  s=$?
  case $s in
    0) echo "$slug: $(grep -m1 'lm_eval_rolling_byte_perplexity' $log | tr -s ' ')" ;;
    5) echo "$slug: already run, skipped" ;;
    *) echo "$slug: FAILED (exit $s)"; cp $log $D/failed-$slug.log ;;
  esac
}
{ one att-cfirst-lr0.006 cf; one att-cfirst-lr0.003 cf; } &
sleep 60
{ one att-cwindow-lr0.006 cw; one att-cwindow-lr0.003 cw; } &
wait
echo "cosine done"
