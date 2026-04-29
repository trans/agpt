#!/usr/bin/env bash
# Reproduce the d=16/32/48 AGPT training sweep on Shakespeare 1M.
# Confirms the framing's second descriptive prediction: d=32 is the sweet
# spot for English at 1-5M corpus size, with d=16 deficient (insufficient
# identity zone) and d=48 saturated (extra trie depth doesn't help).
#
# Recipe: post-fix best (rmsprop + warmup-cosine + entropy-icing,
# linear mass weighting, no-accumulate, 3 super-epochs).
set -eu -o pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
cd "$PROJECT_ROOT" || { echo "Cannot cd to PROJECT_ROOT=$PROJECT_ROOT" >&2; exit 1; }
[ -f Justfile ] || { echo "Run from agpt project root" >&2; exit 1; }

OUT=rnd/trie-attention-framing/logs
mkdir -p "$OUT"

INIT_CKPT=data/input.random.model
EVAL_POS=16384

eval_ppl () {
    bin/perplexity --model "$1" --file data/input.txt \
        --max-positions $EVAL_POS --backend cublas 2>&1 \
      | awk '/^Perplexity:/ {print $2}'
}

build_trie () {
    local depth="$1" out="$2"
    if [ -d "$out" ]; then return 0; fi
    bin/agpt_build_radix_corpus --corpus data/input.txt --max-depth "$depth" --out "$out"
}

run_d () {
    local d="$1"
    local trie="/tmp/agpt_input_d${d}_radix"
    build_trie "$d" "$trie"

    local model="/tmp/dr_dsweep_d${d}.model"
    local log="$OUT/d_sweep_d${d}.log"
    cp "$INIT_CKPT" "$model"

    bin/agpt_train --model "$model" --trie-dir "$trie" --save "$model" \
        --epochs 3 --lr 3e-3 \
        --optimizer rmsprop --rmsprop-beta 0.999 \
        --lr-schedule warmup-cosine --warmup-epochs 1 \
        --entropy-lambda 1.0 --mass-weight linear --no-accumulate \
        > "$log" 2>&1

    local ppl=$(eval_ppl "$model")
    echo "d=$d  PPL=$ppl"
}

echo "=== AGPT d-sweep on Shakespeare 1M ==="
echo "    Expected: d=16 ≈ 15.4, d=32 ≈ 13.0, d=48 ≈ 12.9"
echo

run_d 16
run_d 32
run_d 48
