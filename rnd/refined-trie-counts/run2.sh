#!/bin/sh
# Refined trie counts at other plan depths (the user, 2026-11-18: pd=1 and pd=0;
# pd=6 is the deepest plan on record): own counts against the 100M text's at
# pd=0 (one full-batch step a pass, 512 passes), pd=1 (89 steps a pass, 128
# passes, two order seeds) and pd=6 (379,312 steps a pass, 2 passes, Adam
# 0.001). Three streams started 2 minutes apart; pd=6's tiny steps make it the
# slow arm (about 30-45 minutes a pass), so each pd=6 run gets its own stream.
# Run only inside a slot (sh /home/trans/Projects/air/tools/slots add @agpt ...).
# A failed run is reported and skipped.
# Usage: sh rnd/refined-trie-counts/run2.sh
renice -n 19 -p $$ > /dev/null
ionice -c3 -p $$
export OMP_NUM_THREADS=2
D=rnd/refined-trie-counts
C=$D/configs
just build-agpt-experiment > $D/build.log 2>&1 || { echo "build failed, see $D/build.log"; exit 1; }

stream() {
  name=$1; shift
  for slug in "$@"; do
    bin/agpt_experiment --config $C/$slug.yml --trainer src/tools/agpt_minibatch_train.py > $D/last_run_$name.log 2>&1
    s=$?
    case $s in
      0) echo "$(date -u +%H:%M) $slug: $(grep -m1 'lm_eval_rolling_byte_perplexity' $D/last_run_$name.log | tr -s ' ')" ;;
      5) echo "$(date -u +%H:%M) $slug: already run, skipped" ;;
      *) echo "$(date -u +%H:%M) $slug: FAILED (exit $s)"; cp $D/last_run_$name.log $D/failed-$slug.log ;;
    esac
  done
  echo "$(date -u +%H:%M) stream $name done"
}

echo "$(date -u +%H:%M) start"
stream A agpt-a1m-pd6 agpt-a1m-pd1 agpt-a1m-pd1-s2 &
sleep 120
stream B agpt-a1m-pd6-ref100m &
sleep 120
stream C agpt-a1m-pd0 agpt-a1m-pd0-ref100m agpt-a1m-pd1-ref100m agpt-a1m-pd1-ref100m-s2 &
wait
echo "$(date -u +%H:%M) all done"
