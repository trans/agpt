#!/bin/sh
# Forward-backward coupling grid, on the CPU (2 threads a run, under 1 GB each,
# no slot). Stream A: the backward model, its teacher file, then the forward runs
# coupled to it. Stream B (2 minutes later): the plain baseline, then the
# forward-teacher control. About 2 hours.
# Usage: sh rnd/forward-backward-coupling/run.sh > rnd/forward-backward-coupling/run.out 2>&1
renice -n 19 -p $$ > /dev/null
ionice -c3 -p $$
export OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
D=rnd/forward-backward-coupling
C=$D/configs
T=/home/trans/agpt-tries/coupling
just build-agpt-experiment > $D/build.log 2>&1 || { echo "build failed, see $D/build.log"; exit 1; }

run() {  # run NAME SLUG
  bin/agpt_experiment --config $C/$2.yml --trainer src/tools/agpt_minibatch_train.py > $D/last_run_$1.log 2>&1
  s=$?
  case $s in
    0) echo "$(date -u +%H:%M) $2: $(grep -m1 'lm_eval_rolling_byte_perplexity' $D/last_run_$1.log | tr -s ' ')" ;;
    5) echo "$(date -u +%H:%M) $2: already run, skipped" ;;
    *) echo "$(date -u +%H:%M) $2: FAILED (exit $s)"; cp $D/last_run_$1.log $D/failed-$2.log; return 1 ;;
  esac
}

echo "$(date -u +%H:%M) start"
{ if run A mb-a1m-rev; then
    B=$(ls -d $D/2026*-mb-a1m-rev | tail -1)
    python3 $D/teacher_preds.py --model $B/checkpoint.model --vocab data/pg19/imm-split/vocab.txt \
      --corpus data/pg19/a1m/train_corpus.txt --direction backward --out $T/backward_mb-a1m-rev.npy
    python3 $D/make_configs.py > /dev/null
    run A fb-b0.1; run A fb-b1
  fi; echo "$(date -u +%H:%M) stream A done"; } &
sleep 120
{ run B mb-a1m-base; run B ff-b0.1; run B ff-b1; echo "$(date -u +%H:%M) stream B done"; } &
wait
echo "$(date -u +%H:%M) all done"
