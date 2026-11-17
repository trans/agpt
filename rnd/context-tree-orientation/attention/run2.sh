#!/bin/sh
# Attention grouping control, second slot: the 8 runs left after slot 66 (which ran
# first, window and last at 0.003 and last at 0.006), in two streams. The GPU is the
# bottleneck: in slot 66 it read 100% busy with three runs, and a `first` pass took 36 s
# against 12 s alone, so streams share it rather than overlap. Two keep one stream's CPU
# evals under the other's GPU work. A `last` pass costs about 7 times a `first` or
# `window` pass; the streams are balanced by that.
# Run only inside a slot (sh /home/trans/Projects/air/tools/slots add @agpt ...).
# A failed run is reported and skipped.
# Usage: sh rnd/context-tree-orientation/attention/run2.sh
renice -n 19 -p $$ > /dev/null
ionice -c3 -p $$
export OMP_NUM_THREADS=2
D=rnd/context-tree-orientation/attention
C=$D/configs
just build-agpt-experiment > $D/build.log 2>&1 || { echo "build failed, see $D/build.log"; exit 1; }

stream() {
  name=$1; shift
  for slug in "$@"; do
    c=$C/$slug.yml
    bin/agpt_experiment --config $c --trainer src/tools/agpt_minibatch_train.py > $D/last_run_$name.log 2>&1
    s=$?
    case $s in
      0) echo "$slug: $(grep -m1 'lm_eval_rolling_byte_perplexity' $D/last_run_$name.log | tr -s ' ')" ;;
      5) echo "$slug: already run, skipped" ;;
      *) echo "$slug: FAILED (exit $s)"; cp $D/last_run_$name.log $D/failed-$slug.log ;;
    esac
  done
  echo "stream $name done"
}

stream A att-window-lr0.0015 att-first-lr0.0015 att-window-lr0.006 att-first-lr0.006 att-window-lr0.003-s2 att-first-lr0.003-s2 &
sleep 120
stream B att-last-lr0.0015 att-last-lr0.003-s2 &
wait
echo "all done"
