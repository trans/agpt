#!/bin/sh
# doped-counts-depth16 grid (GPU slot; each run through bin/agpt_experiment).
# Usage: sh rnd/doped-counts-depth16/run.sh [NAME ...]   (default: every name in order.txt)
# Exit code 5 from the runner means the run already exists: skipped.
D=rnd/doped-counts-depth16
names="$*"
[ -z "$names" ] && names=$(cat $D/order.txt)
for n in $names; do
  echo "$(date -u +%H:%M:%S) $n"
  bin/agpt_experiment --config $D/configs/$n.yml --trainer src/tools/agpt_minibatch_train.py > $D/last_run.log 2>&1
  s=$?
  case $s in
    0) ;;
    5) echo "  already run, skipped" ;;
    *) echo "  FAILED (exit $s)"; cp $D/last_run.log $D/failed-$n.log ;;
  esac
done
echo "$(date -u +%H:%M:%S) done"
