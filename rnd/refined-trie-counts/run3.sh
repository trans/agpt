#!/bin/sh
# Refined trie counts, weighted (the user, 2026-11-18: loading the small trie with
# a much larger text's statistics should raise the sharing; go-ahead 2026-11-19).
# Each event of the 1M slice weighted by its context's count in the 100M text over
# its count in the slice (experimental.trie_weights_corpus), at pd=3, pd=1 and in
# minibatch, against ordinary minibatch on the 100M text itself. Two streams
# started 2 minutes apart, about 35 minutes. Run only inside a slot
# (sh /home/trans/Projects/air/tools/slots add @agpt ...). A failed run is
# reported and skipped.
# Usage: sh rnd/refined-trie-counts/run3.sh
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
stream A agpt-a1m-w100m mb-a1m-w100m &
sleep 120
stream B agpt-a1m-pd1-w100m mb-100m-st100 &
wait
echo "$(date -u +%H:%M) all done"
