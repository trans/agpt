#!/usr/bin/env bash
# Falsification test for the step-count hypothesis on mass-linear+events.
#
# Matrix observation (Shakespeare 1M d=16, 10 SE):
#   mass-linear+events:  11.48 PPL (much worse than baseline 9.11)
#   mass-linear+none:     8.03 PPL (best single cell measured)
#
# Cross-corpus check (Gutenberg 5M d=16, 10 SE):
#   mass-linear+events:   9.00 PPL (below baseline 9.24)  ← REGIME FLIPPED
#   mass-linear+none:     9.34 PPL
#
# Gutenberg is 5× larger so 10 SE there = ~3250 fire steps vs
# Shakespeare's ~650. RMSprop β₂=0.999 has time constant ~1000 steps;
# Shakespeare 10 SE is solidly inside the transient, Gutenberg 10 SE is
# past it.
#
# Hypothesis: the regime flip isn't about corpus content — it's about
# whether RMSprop's v accumulator has equilibrated. mass-linear weights
# span ~10⁶×; under `events` (1/N divisor), the optimizer is bouncing
# off the few top-mass events. Once v equilibrates, that bouncing
# damps out.
#
# Prediction: mass-linear+events on Shakespeare should improve
# dramatically going from 10 → 30 → 100 SE. At ~3000 fire steps (~50
# SE on Shakespeare) it should approach Gutenberg's 10-SE behavior.
#
# Cells: mass-linear+events × SE ∈ {10, 30, 100} × 3 seeds = 9.
# Plus a single 30-SE reference at off+events for sanity (off should
# get monotonically better with more SE too — see β₂ diagnostic).
#
# Outputs:
#   rnd/composite-weights/se-sweep/<config>_se<N>_s<seed>/
#       {run.model, train.log, heldout_ppl.txt}
#   rnd/composite-weights/se-sweep/results.txt
#   rnd/composite-weights/se-sweep/summary.txt

set -eu

OUT=rnd/composite-weights/se-sweep
mkdir -p $OUT
RESULTS=$OUT/results.txt
SUMMARY=$OUT/summary.txt

echo "# mass-linear+events SE sweep on Shakespeare, $(date -u)" >> $RESULTS

run_cell() {
    local config="$1"       # mass-linear-events | off-events
    local epochs="$2"
    local seed="$3"
    local label="${config}_se${epochs}_s${seed}"
    local D=$OUT/$label
    if [ -f $D/heldout_ppl.txt ] && [ -s $D/heldout_ppl.txt ]; then
        echo "[cached] $label heldout=$(cat $D/heldout_ppl.txt)" | tee -a $RESULTS
        return
    fi
    rm -rf $D && mkdir -p $D
    local mass_flag="--mass-weight off"
    [ "$config" = "mass-linear-events" ] && mass_flag="--mass-weight linear"
    local START=$(date +%s)
    agpt_train --model /tmp/seed${seed}.model \
        --trie-dir /tmp/shake_baseline_d16_radix \
        --epochs $epochs --lr 3e-3 --optimizer rmsprop \
        --lr-schedule warmup-cosine --warmup-epochs 1 \
        --partition-depth 1 --no-accumulate $mass_flag \
        --save $D/run.model > $D/train.log 2>&1
    local TRAIN_WALL=$(($(date +%s) - START))
    local LOSS=$(grep "^Epoch $epochs:" $D/train.log | sed 's/.*loss=\([0-9.]*\).*/\1/')
    local PPL=$(agpt_sliding_window_perplexity --model $D/run.model \
        --file /tmp/shake_holdout.txt \
        --vocab-file data/input.txt \
        --d 16 --max-positions 10000 --backend openblas --workers 8 2>&1 \
        | grep "^Perplexity" | awk '{print $2}')
    echo "$PPL" > $D/heldout_ppl.txt
    echo "$label train_wall=${TRAIN_WALL}s train_loss=$LOSS heldout=$PPL" | tee -a $RESULTS
}

# mass-linear+events at three SE points.
for seed in 1 2 3; do
    for epochs in 10 30 100; do
        run_cell mass-linear-events $epochs $seed
    done
done

# off+events at 30 SE only (the in-between datapoint — 10 SE and 100 SE
# off+events already came from the β2 diagnostic, summarized in
# rnd/beta2-diagnostic/summary.txt: 10 SE 9.245, 100 SE 6.250).
for seed in 1 2 3; do
    run_cell off-events 30 $seed
done

# Summary.
echo "" | tee -a $SUMMARY
echo "=== mass-linear+events vs off+events on Shakespeare, by SE ===" | tee -a $SUMMARY
echo "(off+events 10 SE 9.245, 100 SE 6.250 from rnd/beta2-diagnostic/)" | tee -a $SUMMARY
echo "" | tee -a $SUMMARY
for config in mass-linear-events off-events; do
    for epochs in 10 30 100; do
        PPLS=$(grep "^${config}_se${epochs}_" $RESULTS | grep -oE 'heldout=[0-9.]+' | sed 's/heldout=//')
        if [ -n "$PPLS" ]; then
            STATS=$(echo "$PPLS" | awk '{s+=$1; ss+=$1*$1; n+=1} END {m=s/n; v=ss/n-m*m; sd=sqrt(v>0?v:0); printf "%.3f ± %.3f (n=%d)", m, sd, n}')
            echo "$config ${epochs} SE: $STATS" | tee -a $SUMMARY
        fi
    done
done

echo "" | tee -a $SUMMARY
echo "Full per-run results: $RESULTS" | tee -a $SUMMARY
