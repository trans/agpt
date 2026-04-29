#!/usr/bin/env bash
# Reproduce the decision-only loss buffer sweep (AGPT_DECISION_ONLY=1
# with AGPT_DECISION_BUFFER=N).
#
# Skips the loss + gradient at queries past d_split + buffer:
#   buffer=0  → strict (drops ALL deterministic-tail events; 8.2% kept)
#   buffer=N  → keeps events for N chars past d_split (less aggressive)
#
# Result (the framing's strongest empirical support, despite no PPL win):
# decision-only at 8.2% events captures 96.8% of baseline's total learning
# (in nats, relative to random). PPL appears slightly worse because PPL
# is exp(CE) and small CE differences are tiny PPL differences.
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
    local tag="$1" buf="$2"
    local model="/tmp/dr_decision_${tag}.model"
    local log="$OUT/decision_${tag}.log"
    cp "$INIT_CKPT" "$model"

    AGPT_DECISION_ONLY=1 AGPT_DECISION_BUFFER=$buf \
        bin/agpt_train --model "$model" --trie-dir "$TRIE" --save "$model" \
            --epochs 3 --lr 3e-3 \
            --optimizer rmsprop --rmsprop-beta 0.999 \
            --lr-schedule warmup-cosine --warmup-epochs 1 \
            --entropy-lambda 1.0 --mass-weight linear --no-accumulate \
            > "$log" 2>&1

    local ppl=$(eval_ppl "$model")
    local events=$(grep "nodes)" "$log" | tail -1 | sed 's/.*chunks, \([0-9]*\) nodes.*/\1/')
    printf "%-15s buf=%-2s events=%-12s PPL=%s\n" "$tag" "$buf" "$events" "$ppl"
}

run_baseline () {
    local tag="$1"
    local model="/tmp/dr_decision_${tag}.model"
    local log="$OUT/decision_${tag}.log"
    cp "$INIT_CKPT" "$model"

    bin/agpt_train --model "$model" --trie-dir "$TRIE" --save "$model" \
        --epochs 3 --lr 3e-3 \
        --optimizer rmsprop --rmsprop-beta 0.999 \
        --lr-schedule warmup-cosine --warmup-epochs 1 \
        --entropy-lambda 1.0 --mass-weight linear --no-accumulate \
        > "$log" 2>&1

    local ppl=$(eval_ppl "$model")
    local events=$(grep "nodes)" "$log" | tail -1 | sed 's/.*chunks, \([0-9]*\) nodes.*/\1/')
    printf "%-15s buf=-- events=%-12s PPL=%s\n" "$tag" "$events" "$ppl"
}

echo "=== Decision-only buffer sweep on Shakespeare 1M d=32 ==="
echo "    Expected: baseline ~13.25 PPL, buf=0 ~13.93, buf=5 ~13.54, buf=15 ~13.33"
echo

# 3 baseline runs to estimate variance
run_baseline baseline_r1
run_baseline baseline_r2
run_baseline baseline_r3
# Buffer sweep — single probes at extremes, 3 reps at buf=5 for variance
run_one buf0   0
run_one buf2   2
run_one buf5_r1 5
run_one buf5_r2 5
run_one buf5_r3 5
run_one buf10  10
run_one buf15  15
