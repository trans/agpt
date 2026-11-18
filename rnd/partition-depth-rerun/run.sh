#!/bin/sh
# Partition-depth rerun (the user, 2026-11-18: "yes, queue the Shakespeare pd=6
# rerun"): the 2026-04-30 Adam pd=6 recipe, with the code of that day, put on
# today's scale. One stream on the GPU, about 11 minutes a training.
#  1. Toolchain check: the sweep's own trie, trainer and scorer (configs/repro.yml).
#     It should land near the record's 3.82.
#  2. The same recipe trained on the canonical Shakespeare split, through
#     bin/agpt_experiment (canonical eval on its 10 held-out chunks), twice.
#  3. Each checkpoint scored the old way too (bin/perplexity, 8,192 positions at
#     seq-len 32) on the start of data/input.txt, as the sweep scored, and on the
#     held-out text; and by agpt_ppl.py (the PyTorch model the HF wrapper uses)
#     on the same positions, to check it reads the old checkpoints as
#     bin/perplexity does.
# Run only inside a slot (sh /home/trans/Projects/air/tools/slots add @agpt ...).
# Usage: sh rnd/partition-depth-rerun/run.sh
renice -n 19 -p $$ > /dev/null
ionice -c3 -p $$
export OMP_NUM_THREADS=2
D=rnd/partition-depth-rerun
C=$D/configs
H=data/.splits/4fa9aec1db6b3aea/heldout_corpus.txt
just build-agpt-experiment > $D/build.log 2>&1 || { echo "build failed, see $D/build.log"; exit 1; }

score() {  # score CHECKPOINT LABEL
  for f in data/input.txt $H; do
    p=$(bin/perplexity --model "$1" --file $f --max-positions 8192 --seq-len 32 --backend cublas 2>&1 \
        | awk '/^Perplexity:/ {print $2}')
    q=$(CUDA_VISIBLE_DEVICES= python3 src/tools/agpt_ppl.py --model "$1" --file $f --vocab-file data/input.txt \
        --d 32 --max-positions 8192 --mode fixed 2>&1 | awk '/^Perplexity:/ {print $2}')
    echo "$(date -u +%H:%M) $2 on $f: bin/perplexity $p | agpt_ppl.py $q"
  done
}

echo "$(date -u +%H:%M) start"
mkdir -p $D/repro
if python3 $D/old_v1_trainer.py --config $C/repro.yml > $D/repro/train.log 2>&1; then
  score $D/repro/checkpoint.model repro
else
  echo "$(date -u +%H:%M) repro: FAILED, see $D/repro/train.log"
fi
for r in r1 r2; do
  bin/agpt_experiment --config $C/pd6-adam-$r.yml --trainer $D/old_v1_trainer.py > $D/last_run.log 2>&1
  s=$?
  if [ $s -eq 0 ]; then
    echo "$(date -u +%H:%M) pd6-adam-$r: $(grep -m1 'lm_eval_rolling_byte_perplexity' $D/last_run.log | tr -s ' ')"
    score "$(ls -d $D/2026*-pd6-adam-$r | tail -1)/checkpoint.model" pd6-adam-$r
  else
    echo "$(date -u +%H:%M) pd6-adam-$r: FAILED (exit $s)"; cp $D/last_run.log $D/failed-$r.log
  fi
done
echo "$(date -u +%H:%M) all done"
