#!/bin/sh
# Run the one-pass-gap grid through the orchestrator. Run only inside a GPU slot
# (sh /home/trans/Projects/air/tools/slots add @agpt ...). Rebuilds the two
# binaries first so they match the checked-out commit. A config whose run already
# exists is skipped (the orchestrator exits 5 on a duplicate).
# Usage: sh rnd/one-pass-gap/run_grid.sh [config ...]   (default: all of configs/,
# one-pass runs first, then 2, 4 and 8 passes)
renice -n 19 -p $$ > /dev/null
ionice -c3 -p $$
export OMP_NUM_THREADS=2
D=rnd/one-pass-gap
just build-agpt-train-v2 build-agpt-experiment > $D/build.log 2>&1 || { echo "build failed, see $D/build.log"; exit 1; }
[ $# -gt 0 ] || set -- $D/configs/*-e1-*.yml $D/configs/*-e2-*.yml $D/configs/*-e4-*.yml $D/configs/*-e8-*.yml
for c in "$@"; do
  case $(basename $c) in
    agpt-*|pd*) T=v2 ;;
    mb*)    T=src/tools/agpt_minibatch_train.py ;;
    *)      echo "skip $c"; continue ;;
  esac
  bin/agpt_experiment --config $c --trainer $T > $D/last_run.log 2>&1
  s=$?
  case $s in
    0) echo "$(basename $c .yml): $(grep -m1 'lm_eval_rolling_byte_perplexity' $D/last_run.log | tr -s ' ')" ;;
    5) echo "$(basename $c .yml): already run, skipped" ;;
    *) echo "$(basename $c .yml): FAILED (exit $s), see $D/last_run.log"; exit $s ;;
  esac
done
