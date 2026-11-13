#!/bin/sh
# The control for a count prior counted from more text than the network can read:
# the plain 10.8M char transformer, no prior, trained on the 1 GB of books that
# mobius's pg19split20m_gated_d16_pg1g prior counted plus the 20M split, for the
# same 50k steps (819M characters seen, about 0.8 of a pass). Compare with the
# 10.8M + that prior (input, prior dropout 0.5; mobius rnd/unit-lattice, 1.5139
# bits/char) and agpt's rnd/prior-scaling-plain-220m (1.5579).
# prior_transformer_bytes.py is air's prior_transformer.py with the training text
# held as bytes (identical output, checked on CPU); only the config's train_input
# differs (config.json). Run only inside a slot.
# Usage: sh rnd/prior-scaling-plain-1g/run.sh
set -e
renice -n 19 -p $$ > /dev/null
ionice -c3 -p $$
export OMP_NUM_THREADS=2
D=rnd/prior-scaling-plain-1g
T=$(python3 -c "import json; print(json.load(open('$D/config.json'))['train_input'])")
echo "21eaf56afc24206a12648b8a72c38f39a4912ca2b2e4a1debd06c1c3f2e0fe72  $T" | sha256sum -c --quiet
python3 $D/prior_transformer_bytes.py \
  --config $D/config.json --output $D/s10.8m_none_books1g.json --prior none \
  --layers 6 --heads 6 --width 384 --block 256 --dropout 0.2 --batch 64 --iters 50000 \
  --lr 1e-3 --min-lr 1e-4 --warmup 100 --weight-decay 0.1 --eval-every 2500 --seed 1337 \
  --save $D/s10.8m_none_books1g.pt > $D/s10.8m_none_books1g.log 2>&1
tail -1 $D/s10.8m_none_books1g.log
