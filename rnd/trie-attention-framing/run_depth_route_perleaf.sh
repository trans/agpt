#!/usr/bin/env bash
# Reproduce the per-leaf d* routing experiment (AGPT_DEPTH_ROUTE_PERLEAF=1).
# Each query's threshold is its node's d_split (depth at which the path
# first becomes mass=1) instead of a flat global k. Multi-mass intermediate
# nodes get d_split=INT_MAX (all queries route to Wk).
#
# Result: across-session mean gap of 0.29 PPL (2.1%) in favor of per-leaf;
# borderline t≈2.0, p≈0.07. Within-session-only: gap drops to 0.16 PPL,
# p≈0.37. Initial 3-run "strict ordering" of per-leaf < baseline was a
# small-sample artifact.
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
    local tag="$1" perleaf="$2"
    local model="/tmp/dr_perleaf_${tag}.model"
    local log="$OUT/perleaf_${tag}.log"
    cp "$INIT_CKPT" "$model"

    local env_prefix=""
    if [ "$perleaf" -gt 0 ]; then env_prefix="AGPT_DEPTH_ROUTE_PERLEAF=1"; fi

    env $env_prefix bin/agpt_train --model "$model" --trie-dir "$TRIE" --save "$model" \
        --epochs 3 --lr 3e-3 \
        --optimizer rmsprop --rmsprop-beta 0.999 \
        --lr-schedule warmup-cosine --warmup-epochs 1 \
        --entropy-lambda 1.0 --mass-weight linear --no-accumulate \
        > "$log" 2>&1

    local ppl=$(eval_ppl "$model")
    printf "%-20s perleaf=%s PPL=%s\n" "$tag" "$perleaf" "$ppl"
}

echo "=== Per-leaf d* routing head-to-head on Shakespeare 1M d=32 ==="
echo

# 3 baseline runs, 3 per-leaf runs (within-session for clean variance estimate)
run_one baseline_r1 0
run_one baseline_r2 0
run_one baseline_r3 0
run_one perleaf_r1 1
run_one perleaf_r2 1
run_one perleaf_r3 1
