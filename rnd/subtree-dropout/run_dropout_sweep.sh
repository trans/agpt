#!/usr/bin/env bash
# Reproduce the subtree-dropout sweep on Shakespeare 1M d=32.
# Configs at 5 SE matched-epoch: p=0, 0.2, 0.3, 0.4, 0.5
# Plus higher SE counts at p=0.3 to show dropout doesn't compound at high SE.
set -eu -o pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
cd "$PROJECT_ROOT" || { echo "Cannot cd to PROJECT_ROOT=$PROJECT_ROOT" >&2; exit 1; }
[ -f Justfile ] || { echo "Run from agpt project root" >&2; exit 1; }

OUT=rnd/subtree-dropout/logs
mkdir -p "$OUT"

INIT=data/input.random.model
TRIE=/home/trans/agpt-tries/shakespeare_d32_radix_corpus
[ -d "$TRIE" ] || bin/agpt_build_radix_corpus --corpus data/input.txt --max-depth 32 --out "$TRIE"

eval_ppl () {
    bin/perplexity --model "$1" --file data/input.txt --max-positions 8192 --seq-len 32 --backend cublas 2>&1 \
      | awk '/^Perplexity:/ {print $2}'
}

run_one () {
    local P=$1 SE=$2 R=$3
    local M=/tmp/dr_dropout_p${P}_${SE}SE_r${R}.model
    cp "$INIT" "$M"
    local ENV=""
    if [ "$P" != "0" ]; then ENV="AGPT_SUBTREE_DROPOUT=$P"; fi

    env $ENV bin/agpt_train --model "$M" --trie-dir "$TRIE" --save "$M" \
        --epochs $SE --lr 3e-3 \
        --optimizer rmsprop --rmsprop-beta 0.999 \
        --lr-schedule warmup-cosine --warmup-epochs 1 \
        --entropy-lambda 1.0 --mass-weight log --no-accumulate \
        > $OUT/p${P}_${SE}SE_r${R}.log 2>&1
    printf "p=%-4s SE=%-2s r%s  PPL@32=%s\n" "$P" "$SE" "$R" "$(eval_ppl $M)"
}

echo "=== Subtree-dropout matched-epoch sweep on Shakespeare 1M d=32 ==="
echo "  Expected ranking at 5 SE: p=0 (9.64) << p=0.3 (10.13) < p=0.5 (10.77) < p=0.4 (11.27)"
echo "  Expected high-SE: dropout NOT compounding (p=0 always wins at matched SE)"
echo

# Matched-epoch comparison at 5 SE
for P in 0 0.2 0.3 0.4 0.5; do
    for R in 1 2 3; do
        run_one $P 5 $R
    done
done

# High-SE comparison: dropout vs no-dropout
for SE in 7 10 15; do
    for P in 0 0.3; do
        for R in 1 2 3; do
            run_one $P $SE $R
        done
    done
done
