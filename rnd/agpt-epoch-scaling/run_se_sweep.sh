#!/usr/bin/env bash
# Reproduce the SE sweep: 3, 5, 7, 10, 15, 20, 25, 30, 40 super-epochs of
# pure AGPT (no dropout) on Shakespeare 1M d=32. Each cell 3 reps; eval
# at matched seq=32. Total wall-clock ~30-45 min.
set -eu -o pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
cd "$PROJECT_ROOT" || { echo "Cannot cd to PROJECT_ROOT=$PROJECT_ROOT" >&2; exit 1; }
[ -f Justfile ] || { echo "Run from agpt project root" >&2; exit 1; }

OUT=rnd/agpt-epoch-scaling/logs
mkdir -p "$OUT"

INIT=data/input.random.model
TRIE=/home/trans/agpt-tries/shakespeare_d32_radix_corpus
[ -d "$TRIE" ] || bin/agpt_build_radix_corpus --corpus data/input.txt --max-depth 32 --out "$TRIE"

eval_ppl () {
    bin/perplexity --model "$1" --file data/input.txt --max-positions 8192 --seq-len 32 --backend cublas 2>&1 \
      | awk '/^Perplexity:/ {print $2}'
}

run_one () {
    local SE=$1 R=$2
    local M=/tmp/dr_se_sweep_${SE}SE_r${R}.model
    cp "$INIT" "$M"
    bin/agpt_train --model "$M" --trie-dir "$TRIE" --save "$M" \
        --epochs $SE --lr 3e-3 \
        --optimizer rmsprop --rmsprop-beta 0.999 \
        --lr-schedule warmup-cosine --warmup-epochs 1 \
        --entropy-lambda 1.0 --mass-weight log --no-accumulate \
        > $OUT/${SE}SE_r${R}.log 2>&1
    local P=$(eval_ppl "$M")
    printf "%2dSE r%s  PPL@32=%s\n" "$SE" "$R" "$P"
}

echo "=== AGPT SE sweep on Shakespeare 1M d=32 ==="
echo "  Expected: 3SE~10.82, 10SE~8.24, 20SE~6.59, 40SE~5.39"
echo

for SE in 3 5 7 10 15 20 25 30 40; do
    for R in 1 2 3; do
        run_one $SE $R
    done
done
