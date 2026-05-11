#!/usr/bin/env bash
# SGD vs. AGPT sanity-check. Apples-to-apples:
#
#   - Same init: both tools start from data/input.agpt.model (the random-init
#     checkpoint all AGPT experiments in this project use).
#   - Matched training context: SGD at seq_len=16 vs AGPT at d=16;
#                               SGD at seq_len=32 vs AGPT at d=32.
#   - Same architecture (d_model=64, n_heads=4, n_layers=2, d_ff=256).
#   - Same eval: bin/perplexity --max-positions 16384.
#   - Wall-clock matched: SGD at step budgets chosen to roughly match AGPT's
#     ~1 min / 3 super-epochs (estimate: ~5000 SGD steps at seq_len=16).
#     Plus long-budget SGD runs to probe convergence.

set -eu -o pipefail

cd "$(dirname "$0")/.."  # project root
OUT="rnd/sgd-sanity-check/logs"
RES="rnd/sgd-sanity-check/results"
mkdir -p "$OUT" "$RES"
N_RUNS=3
EVAL_POS=16384
INIT_CKPT="data/input.agpt.model"

eval_ppl () {
    local checkpoint="$1"
    bin/perplexity --model "$checkpoint" --file data/input.txt \
        --max-positions $EVAL_POS --backend cublas 2>&1 \
      | awk '/^Perplexity:/ {print $2}'
}

run_agpt () {
    local label="$1"; local trie="$2"; shift 2
    for i in $(seq 1 $N_RUNS); do
        local work="/tmp/sgd_sanity_${label}_r${i}.model"
        cp "$INIT_CKPT" "$work"
        local log="$OUT/${label}_r${i}.log"
        /usr/bin/time -f '%e sec' bin/agpt_train --model "$work" --trie-dir "$trie" \
            --epochs 3 --lr 3e-3 --optimizer rmsprop --rmsprop-beta 0.999 \
            --lr-schedule warmup-cosine --warmup-epochs 1 \
            --single-subtree --entropy-lambda 1.0 \
            --save "$work" \
            "$@" > "$log" 2>&1
        local ppl=$(eval_ppl "$work")
        local elapsed=$(awk '/sec$/{print $1}' "$log" | tail -1)
        echo "$label  run $i  PPL = $ppl  time = ${elapsed}s"
    done
}

run_sgd () {
    local label="$1"; local seq_len="$2"; local steps="$3"
    for i in $(seq 1 $N_RUNS); do
        local work="/tmp/sgd_sanity_${label}_r${i}.model"
        cp "$INIT_CKPT" "$work"
        local log="$OUT/${label}_r${i}.log"
        # Plain window training (no --agpt). Loads weights from --model
        # file (we just copied init into it) and saves back to the same path.
        # Architecture (d_model/n_heads/n_layers/d_ff) is loaded from the
        # init checkpoint — don't re-specify on the command line or we
        # risk mismatching (bin/microgpt doesn't expose --d-ff at all,
        # and it uses --heads for a different meaning than n_heads).
        /usr/bin/time -f '%e sec' bin/microgpt --file data/input.txt --model "$work" \
            --seq-len "$seq_len" --steps "$steps" \
            --lr 3e-3 --backend cublas \
            > "$log" 2>&1
        local ppl=$(eval_ppl "$work")
        local elapsed=$(awk '/sec$/{print $1}' "$log" | tail -1)
        echo "$label  run $i  PPL = $ppl  time = ${elapsed}s"
    done
}

echo "=== d=16 comparison (SGD seq_len=16 vs AGPT d=16) ==="
echo ""

# SGD per step = 1 window of seq_len tokens. For comparison, AGPT per
# optimizer step aggregates gradients across ~125k query positions. So
# matched wall-clock (~60s/run) ≈ 5-10k SGD steps; matched gradient-
# events would need ~1.5M steps. We span budgets to see the PPL curve.
echo "-- SGD at seq_len=16 --"
run_sgd sgd-s16-2000    16  2000
run_sgd sgd-s16-10000   16  10000
run_sgd sgd-s16-50000   16  50000
run_sgd sgd-s16-200000  16  200000

echo ""
echo "-- AGPT at d=16 (4 mass-weight modes) --"
run_agpt agpt-d16-off       /tmp/agpt_input_d16_radix_pst  # default mass-weight off
run_agpt agpt-d16-log       /tmp/agpt_input_d16_radix_pst  --mass-weight log
run_agpt agpt-d16-sqrt      /tmp/agpt_input_d16_radix_pst  --mass-weight sqrt
run_agpt agpt-d16-linear    /tmp/agpt_input_d16_radix_pst  --mass-weight linear

echo ""
echo "=== d=32 comparison (SGD seq_len=32 vs AGPT d=32) ==="
echo ""
echo "-- SGD at seq_len=32 --"
run_sgd sgd-s32-2000    32  2000
run_sgd sgd-s32-10000   32  10000
run_sgd sgd-s32-50000   32  50000
run_sgd sgd-s32-200000  32  200000

echo ""
echo "-- AGPT at d=32 (4 mass-weight modes) --"
run_agpt agpt-d32-off       /tmp/agpt_input_d32_per  # default mass-weight off
run_agpt agpt-d32-log       /tmp/agpt_input_d32_per  --mass-weight log
run_agpt agpt-d32-sqrt      /tmp/agpt_input_d32_per  --mass-weight sqrt
run_agpt agpt-d32-linear    /tmp/agpt_input_d32_per  --mass-weight linear

echo ""
echo "Done. Logs in $OUT/."
echo "Summary: grep -H 'PPL = \\|sec' $OUT/*.log"
