#!/usr/bin/env bash
# β₂ vs SE diagnostic.
#
# Two hypotheses about RMSprop's slow transient (~1/(1-β₂) steps):
#   H1: β₂=0.999 is fine, we just need to train long enough (100 SE+)
#       for the moving-average to converge.
#   H2: The transient is the bottleneck. Lowering β₂=0.99 gives the
#       same steady-state at 10x fewer steps.
#
# Cells: β₂ ∈ {0.999, 0.99} × SE ∈ {SE_SHORT, SE_LONG}, 3 seeds.
# Pass1 (default): SE_SHORT=10, SE_LONG=100.
# Pass2 (escalation, if Pass1 is inconclusive): SE_SHORT=100, SE_LONG=1000.
#
# Interpretation:
#  - If all four cells land within seed-noise, β₂ doesn't matter at this
#    scale; transient isn't the bottleneck.
#  - If β₂=0.99 short matches β₂=0.999 long, H2 wins: lower β₂ buys
#    convergence cheaply.
#  - If β₂=0.999 long is meaningfully better than everything else, H1
#    wins: must train long, β₂=0.999 is the right slow knob.
#  - If β₂=0.99 long beats β₂=0.999 long: both knobs help, stack them.
#
# Usage on pod:
#   bash rnd/beta2-diagnostic/run.sh                         # 10/100 SE (~10 min)
#   SE_SHORT=100 SE_LONG=1000 bash rnd/beta2-diagnostic/run.sh   # escalation (~35 min)
#
# Outputs:
#   rnd/beta2-diagnostic/results.txt   — per-cell train loss + heldout PPL
#   rnd/beta2-diagnostic/summary.txt   — mean ± std per cell
#   rnd/beta2-diagnostic/<cell>/train.log + run.model + heldout_ppl.txt
#
# Cells run sequentially; cached if heldout_ppl.txt exists (idempotent).

set -eu

SE_SHORT="${SE_SHORT:-10}"
SE_LONG="${SE_LONG:-100}"

OUT=rnd/beta2-diagnostic
mkdir -p $OUT
RESULTS=$OUT/results.txt
SUMMARY=$OUT/summary.txt

echo "# β₂ diagnostic: SE_SHORT=$SE_SHORT SE_LONG=$SE_LONG  ($(date -u))" >> $RESULTS

run_cell() {
    local beta2="$1"
    local epochs="$2"
    local seed="$3"
    local label="b${beta2}_e${epochs}_s${seed}"
    local D=$OUT/$label
    if [ -f $D/heldout_ppl.txt ]; then
        local PPL=$(cat $D/heldout_ppl.txt)
        echo "[cached] $label heldout_ppl=$PPL" | tee -a $RESULTS
        return
    fi
    rm -rf $D && mkdir -p $D
    echo "[run] $label..."
    local START=$(date +%s)
    agpt_train --init --init-seed ${seed} \
        --trie-dir /tmp/shake_baseline_d16_radix \
        --epochs $epochs --lr 3e-3 --optimizer rmsprop \
        --rmsprop-beta $beta2 \
        --lr-schedule warmup-cosine --warmup-epochs 1 \
        --partition-depth 1 --mass-weight off --no-accumulate \
        --save $D/run.model > $D/train.log 2>&1
    local TRAIN_WALL=$(($(date +%s) - START))
    local LOSS=$(grep "^Epoch $epochs:" $D/train.log | sed 's/.*loss=\([0-9.]*\).*/\1/')
    local PPL=$(agpt_sliding_window_perplexity --model $D/run.model \
        --file /tmp/shake_holdout.txt \
        --vocab-file data/input.txt \
        --d 16 --max-positions 10000 --backend openblas 2>&1 \
        | grep -iE "^perplexity" | awk '{print $2}')
    echo "$PPL" > $D/heldout_ppl.txt
    echo "$label train_wall=${TRAIN_WALL}s train_loss=$LOSS heldout_ppl=$PPL" | tee -a $RESULTS
}

# 12 cells: 2 β₂ × 2 SE × 3 seeds.
for seed in 1 2 3; do
    for beta2 in 0.999 0.99; do
        for epochs in $SE_SHORT $SE_LONG; do
            run_cell $beta2 $epochs $seed
        done
    done
done

# Summary: mean ± std per cell.
echo "" | tee -a $SUMMARY
echo "=== β₂ diagnostic summary (SE_SHORT=$SE_SHORT SE_LONG=$SE_LONG) ===" | tee -a $SUMMARY
for beta2 in 0.999 0.99; do
    for epochs in $SE_SHORT $SE_LONG; do
        PPLS=$(grep "^b${beta2}_e${epochs}_" $RESULTS | grep -oE 'heldout_ppl=[0-9.]+' | sed 's/heldout_ppl=//')
        if [ -n "$PPLS" ]; then
            STATS=$(echo "$PPLS" | awk '{s+=$1; ss+=$1*$1; n+=1} END {m=s/n; v=ss/n-m*m; sd=sqrt(v>0?v:0); printf "%.3f ± %.3f (n=%d)", m, sd, n}')
            echo "β₂=$beta2, $epochs SE: $STATS" | tee -a $SUMMARY
        fi
    done
done

echo "" | tee -a $SUMMARY
echo "Full per-run results: $RESULTS" | tee -a $SUMMARY
