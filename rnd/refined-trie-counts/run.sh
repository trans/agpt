#!/bin/sh
# Refined trie counts: the 10 configs of make_configs.py in three streams on the
# GPU, started 2 minutes apart so their plan builds and the 100M reference count
# (about 2 GB of host RAM at its peak) do not coincide. The AGPT replay takes
# about 20,000 small steps a pass, so the streams should overlap on the GPU; the
# CPU evals (about a minute a checkpoint) overlap the other streams' training.
# Run only inside a slot (sh /home/trans/Projects/air/tools/slots add @agpt ...).
# A failed run is reported and skipped.
# Usage: sh rnd/refined-trie-counts/run.sh
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
stream A agpt-a1m agpt-a1m-ref100m agpt-a1m-ref20m mb-a1m &
sleep 120
stream B agpt-a1m-s2 agpt-a1m-ref100m-s2 mb-5m &
sleep 120
stream C agpt-5m agpt-a1m-ref5m mb-a1m-ref100m &
wait
echo "$(date -u +%H:%M) all done"
