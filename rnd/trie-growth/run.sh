#!/bin/sh
# Trie-growth grid (GPU slot; each run through bin/agpt_experiment).
# Usage: sh rnd/trie-growth/run.sh [NAME ...]   (default: every name in order.txt)
# Exit code 5 from the runner means the run already exists: skipped.
D=rnd/trie-growth
names="$*"
[ -z "$names" ] && names=$(cat $D/order.txt)
for n in $names; do
  echo "$(date -u +%H:%M:%S) $n"
  bin/agpt_experiment --config $D/configs/$n.yml > $D/last_run.log 2>&1
  s=$?
  case $s in
    0) ;;
    5) echo "  already run, skipped" ;;
    *) echo "  FAILED (exit $s)"; cp $D/last_run.log $D/failed-$n.log ;;
  esac
done
echo "$(date -u +%H:%M:%S) done"
