#!/bin/sh
# @air's proposed test of the user's framing that the prior is a data resource
# (air .ai/memory/todo.md, 2026-11-12): the plain 10.8M char transformer, with no
# prior, trained on the books prior's 200M book chars plus the 20M split, for the
# same 50k steps as air's runs. Compare with air's s10.8m_none (20M only, 3.0386)
# and pd_s10.8m_prior_in (20M + the books prior, prior dropout 0.5, 2.9326).
# air's experiments/prior_scaling/prior_transformer.py, unchanged; only the
# config's train_input differs (config.json). Run only inside a slot.
# Usage: sh rnd/prior-scaling-plain-220m/run.sh
set -e
renice -n 19 -p $$ > /dev/null
ionice -c3 -p $$
export OMP_NUM_THREADS=2
D=rnd/prior-scaling-plain-220m
T=$(python3 -c "import json; print(json.load(open('$D/config.json'))['train_input'])")
echo "f5815b65bdb8354126b87ac21dd16aa2faeb832be383c050fcebda9332897280  $T" | sha256sum -c --quiet
python3 /home/trans/Projects/air/experiments/prior_scaling/prior_transformer.py \
  --config $D/config.json --output $D/s10.8m_none_books220m.json --prior none \
  --layers 6 --heads 6 --width 384 --block 256 --dropout 0.2 --batch 64 --iters 50000 \
  --lr 1e-3 --min-lr 1e-4 --warmup 100 --weight-decay 0.1 --eval-every 2500 --seed 1337 \
  --save $D/s10.8m_none_books220m.pt > $D/s10.8m_none_books220m.log 2>&1
tail -1 $D/s10.8m_none_books220m.log
