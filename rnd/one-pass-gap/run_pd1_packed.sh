#!/bin/sh
# Follow-up 6: equal-mass steps at pd=1's cadence. Packed plans with 65 equal
# steps (pk260833 ... pk84) and random windows in batches of pd=1's unit sizes
# (mbu1), 4 rates each. Run only inside a slot
# (sh /home/trans/Projects/air/tools/slots add @agpt ...). A failed run is
# reported and skipped.
# Usage: sh rnd/one-pass-gap/run_pd1_packed.sh
renice -n 19 -p $$ > /dev/null
ionice -c3 -p $$
export OMP_NUM_THREADS=2
D=rnd/one-pass-gap
just build-agpt-experiment > $D/build.log 2>&1 || { echo "build failed, see $D/build.log"; exit 1; }
for c in $D/configs/pk260833-e1-*.yml $D/configs/mbu1-e1-*.yml $D/configs/pk16302-e1-*.yml \
         $D/configs/pk1338-e1-*.yml $D/configs/pk84-e1-*.yml $D/configs/pk65208-e1-*.yml; do
  bin/agpt_experiment --config $c --trainer src/tools/agpt_minibatch_train.py > $D/last_run.log 2>&1
  s=$?
  case $s in
    0) echo "$(basename $c .yml): $(grep -m1 'lm_eval_rolling_byte_perplexity' $D/last_run.log | tr -s ' ')" ;;
    5) echo "$(basename $c .yml): already run, skipped" ;;
    *) echo "$(basename $c .yml): FAILED (exit $s)"; cp $D/last_run.log $D/failed-$(basename $c .yml).log ;;
  esac
done
echo "all done"
