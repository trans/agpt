#!/usr/bin/env bash
# Reproduce the static depth-routing sweep (AGPT_DEPTH_ROUTE_K).
# Hard binary mask at integer threshold k: queries at depth ≤ k feed
# Wk-grad only; queries at depth > k feed Wv-grad only.
#
# Result: k=11 within noise, k=7 lucky outlier didn't replicate, k=20
# clearly worse. No setting beat baseline on the mean.
set -eu -o pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
cd "$PROJECT_ROOT" || { echo "Cannot cd to PROJECT_ROOT=$PROJECT_ROOT" >&2; exit 1; }
[ -f Justfile ] || { echo "Run from agpt project root" >&2; exit 1; }

OUT=rnd/trie-attention-framing/logs
mkdir -p "$OUT"

INIT_CKPT=data/input.random.model
TRIE=/tmp/agpt_input_d32_radix
EVAL_POS=16384

[ -d "$TRIE" ] || bin/agpt_build_radix_corpus --corpus data/input.txt --max-depth 32 --out "$TRIE"

eval_ppl () {
    bin/perplexity --model "$1" --file data/input.txt \
        --max-positions $EVAL_POS --backend cublas 2>&1 \
      | awk '/^Perplexity:/ {print $2}'
}

run_one () {
    local tag="$1" k="$2"
    local model="/tmp/dr_static_${tag}.model"
    local log="$OUT/static_${tag}.log"
    cp "$INIT_CKPT" "$model"

    local env_prefix=""
    if [ "$k" -gt 0 ]; then env_prefix="AGPT_DEPTH_ROUTE_K=$k"; fi

    env $env_prefix bin/agpt_train --model "$model" --trie-dir "$TRIE" --save "$model" \
        --epochs 3 --lr 3e-3 \
        --optimizer rmsprop --rmsprop-beta 0.999 \
        --lr-schedule warmup-cosine --warmup-epochs 1 \
        --entropy-lambda 1.0 --mass-weight linear --no-accumulate \
        > "$log" 2>&1

    local ppl=$(eval_ppl "$model")
    printf "%-20s k=%-2s PPL=%s\n" "$tag" "$k" "$ppl"
}

echo "=== Static depth-routing k-sweep on Shakespeare 1M d=32 ==="
echo

# 3 baseline runs, 3 k=11, single probes at other k values
run_one baseline_r1 0
run_one baseline_r2 0
run_one baseline_r3 0
run_one k11_r1 11
run_one k11_r2 11
run_one k11_r3 11
run_one k5     5
run_one k7     7
run_one k9     9
run_one k20    20
