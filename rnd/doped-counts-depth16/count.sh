#!/bin/sh
# The doped tables (GPU slot): a GPU check of the counter against trie_plan on a 100k slice,
# then A's depth-16 contexts counted over the 10G text with tables at 100M, 1G and 10G.
D=rnd/doped-counts-depth16
mkdir -p $D/tables
python3 src/tools/doped_counts.py --corpus data/pg19/a1m/train_corpus.txt --vocab data/pg19/imm-split/vocab.txt \
  --seq-len 16 --ref data/pg19/imm-split-10g-dedup/train_corpus.txt --marks 3000000,8000000 --out $D/tables/check-100k \
  --max-corpus-chars 100000 --max-dense-gib 0.15 --device cuda --check || exit 1
python3 src/tools/doped_counts.py --corpus data/pg19/a1m/train_corpus.txt --vocab data/pg19/imm-split/vocab.txt \
  --seq-len 16 --ref data/pg19/imm-split-10g-dedup/train_corpus.txt --marks 100000000,1000000000,10000000000 \
  --out $D/tables/a1m-s16 --device cuda
