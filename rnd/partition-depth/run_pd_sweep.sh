#!/usr/bin/env bash
# Reproduce the partition-depth sweep on Shakespeare 1M d=32. Each pd N
# trains for the SE count where it stops improving meaningfully. 3 reps
# each. Total wall-clock ~2-3 hours.
set -eu -o pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
cd "$PROJECT_ROOT" || { echo "Cannot cd to PROJECT_ROOT=$PROJECT_ROOT" >&2; exit 1; }
[ -f Justfile ] || { echo "Run from agpt project root" >&2; exit 1; }

OUT=rnd/partition-depth/logs
mkdir -p "$OUT"

INIT=data/input.random.model
TRIE=/home/trans/agpt-tries/shakespeare_d32_radix_corpus
[ -d "$TRIE" ] || bin/agpt_build_radix_corpus --corpus data/input.txt --max-depth 32 --out "$TRIE"

eval_ppl () {
    bin/perplexity --model "$1" --file data/input.txt --max-positions 8192 \
        --seq-len 32 --backend cublas 2>&1 | awk '/^Perplexity:/ {print $2}'
}

run_one () {
    local PD=$1 SE=$2 R=$3
    local M=/tmp/dr_pd_${PD}_${SE}SE_r${R}.model
    cp "$INIT" "$M"
    bin/agpt_train --model "$M" --trie-dir "$TRIE" --save "$M" \
        --epochs $SE --lr 3e-3 \
        --optimizer rmsprop --rmsprop-beta 0.999 \
        --lr-schedule warmup-cosine --warmup-epochs 1 \
        --entropy-lambda 1.0 --mass-weight log --no-accumulate \
        --partition-depth $PD \
        > $OUT/pd${PD}_${SE}SE_r${R}.log 2>&1
    printf "pd=%s SE=%-2s r%s  PPL@32=%s\n" "$PD" "$SE" "$R" "$(eval_ppl $M)"
}

echo "=== Partition-depth sweep on Shakespeare 1M d=32 ==="
echo "  Expected progression of project-best:"
echo "    pd=2 20 SE → 4.61   pd=3 10 SE → 4.35   pd=4 5 SE → 4.21"
echo "    pd=5 3 SE → 4.02    pd=6 3 SE → 3.95    pd=7 3 SE → 4.00 (degrades)"
echo

for R in 1 2 3; do run_one 2 20 $R; done
for R in 1 2 3; do run_one 3 10 $R; done
for R in 1 2 3; do run_one 4 5 $R; done
for R in 1 2 3; do run_one 5 3 $R; done
for R in 1 2 3; do run_one 6 3 $R; done
run_one 7 3 1
