#!/usr/bin/env bash
# Composite mass × entropy on Gutenberg 5M d=16.
#
# Hypothesis: Single-axis weighting (mass-linear, branching-linear) hit
# strong Shakespeare wins (-7% to -12%) but vanished on Gutenberg. Does
# composing mass × entropy generalize where single-axis didn't?
#
# Cells: mass ∈ {linear, log} × entropy ∈ {up, down, peakedness} ×
#        regime ∈ {events, none} × 3 seeds = 36 cells.
# Plus baseline reference cells already in rnd/depth-weight/matrix-gutenberg/.
#
# Requires the agpt_train binary built from the mass×entropy-composition
# patch (src/cuda/agpt_train.cu modified to multiply entropy into the
# mass weights instead of overwriting). Default behavior unchanged when
# only one of mass/entropy is set.
#
# Outputs:
#   rnd/composite-weights/gutenberg/<mass>_<entropy>_<regime>_s<seed>/
#       {run.model, train.log, heldout_ppl.txt}
#   rnd/composite-weights/gutenberg/results.txt   per-cell log
#   rnd/composite-weights/gutenberg/summary.txt   mean ± std per cell

set -eu

OUT=rnd/composite-weights/gutenberg
mkdir -p $OUT
RESULTS=$OUT/results.txt
SUMMARY=$OUT/summary.txt

echo "# composite mass×entropy on Gutenberg, $(date -u)" >> $RESULTS

run_cell() {
    local mass="$1"
    local entropy="$2"
    local regime="$3"   # events | none
    local seed="$4"
    local label="m${mass}_e${entropy}_${regime}_s${seed}"
    local D=$OUT/$label
    if [ -f $D/heldout_ppl.txt ] && [ -s $D/heldout_ppl.txt ]; then
        echo "[cached] $label heldout=$(cat $D/heldout_ppl.txt)" | tee -a $RESULTS
        return
    fi
    rm -rf $D && mkdir -p $D
    local regime_flag=""
    [ "$regime" = "none" ] && regime_flag="--fire-norm-none"
    local START=$(date +%s)
    agpt_train --model /tmp/seed${seed}.model \
        --trie-dir /tmp/gutenberg_5m_baseline_d16_radix \
        --epochs 10 --lr 3e-3 --optimizer rmsprop \
        --lr-schedule warmup-cosine --warmup-epochs 1 \
        --partition-depth 1 --no-accumulate \
        --mass-weight $mass --entropy-weight $entropy $regime_flag \
        --save $D/run.model > $D/train.log 2>&1
    local TRAIN_WALL=$(($(date +%s) - START))
    local LOSS=$(grep "^Epoch 10:" $D/train.log | sed 's/.*loss=\([0-9.]*\).*/\1/')
    local PPL=$(agpt_sliding_window_perplexity --model $D/run.model \
        --file /tmp/gut_holdout.txt \
        --vocab-file data/gutenberg_5m.txt \
        --d 16 --max-positions 10000 --backend openblas --workers 8 2>&1 \
        | grep "^Perplexity" | awk '{print $2}')
    echo "$PPL" > $D/heldout_ppl.txt
    echo "$label train_wall=${TRAIN_WALL}s train_loss=$LOSS heldout=$PPL" | tee -a $RESULTS
}

# 2 mass × 3 entropy × 2 regime × 3 seeds = 36 cells.
for seed in 1 2 3; do
    for mass in linear log; do
        for entropy in up down peakedness; do
            for regime in events none; do
                run_cell $mass $entropy $regime $seed
            done
        done
    done
done

# Aggregate: mean ± std per (mass, entropy, regime) cell.
echo "" | tee -a $SUMMARY
echo "=== composite mass × entropy on Gutenberg ===" | tee -a $SUMMARY
echo "Baseline reference: off+events 9.24, off+none 9.86 (from rnd/depth-weight/matrix-gutenberg/)" | tee -a $SUMMARY
echo "" | tee -a $SUMMARY
echo "mass × entropy × regime: PPL mean ± std" | tee -a $SUMMARY
for mass in linear log; do
    for entropy in up down peakedness; do
        for regime in events none; do
            PPLS=$(grep "^m${mass}_e${entropy}_${regime}_" $RESULTS | grep -oE 'heldout=[0-9.]+' | sed 's/heldout=//')
            if [ -n "$PPLS" ]; then
                STATS=$(echo "$PPLS" | awk '{s+=$1; ss+=$1*$1; n+=1} END {m=s/n; v=ss/n-m*m; sd=sqrt(v>0?v:0); printf "%.3f ± %.3f (n=%d)", m, sd, n}')
                echo "mass=$mass entropy=$entropy regime=$regime: $STATS" | tee -a $SUMMARY
            fi
        done
    done
done

echo "" | tee -a $SUMMARY
echo "Full per-run results: $RESULTS" | tee -a $SUMMARY
