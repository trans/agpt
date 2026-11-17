#!/bin/sh
# Attention grouping control (proviso check 2): the 12 configs of make_configs.py
# in three concurrent streams on the GPU, started 2 minutes apart so their plan
# builds (~1.6 GB of host RAM each) do not coincide. (The streams were meant to
# overlap; they don't: the GPU is the bottleneck, see run2.sh. Slot 66 stopped this
# script's launch loops after 4 runs, and run2.sh runs the other 8.) A `last` pass
# costs about 7 times a `first` or `window` pass; the streams are balanced by that.
# Run only inside a slot (sh /home/trans/Projects/air/tools/slots add @agpt ...).
# A failed run is reported and skipped.
# Usage: sh rnd/context-tree-orientation/attention/run.sh
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

stream A att-last-lr0.003 att-last-lr0.003-s2 att-first-lr0.003-s2 &
sleep 120
stream B att-first-lr0.003 att-window-lr0.003 att-last-lr0.0015 att-window-lr0.003-s2 &
sleep 120
stream C att-last-lr0.006 att-first-lr0.0015 att-first-lr0.006 att-window-lr0.0015 att-window-lr0.006 &
wait
echo "all done"
