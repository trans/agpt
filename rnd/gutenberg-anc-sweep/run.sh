#!/usr/bin/env bash
# Gutenberg 5M sweep at d=16 10 SE with --anc-grad on (new canonical baseline).
# Tests log/sqrt/linear/inv-log/inv-linear modes for each weighting signal
# {mass, entropy, branching, depth-weight}. Cross-corpus replication of the
# Shakespeare findings from 2026-05-22, where --branching-weight log won by
# −0.58 PPL on top of --anc-grad.
#
# Recipe per cell: rmsprop, lr=3e-3, warmup-cosine 1 epoch, --partition-depth 1
# --no-accumulate, --anc-grad. Default --mass-weight off (overridden when the
# mass signal is being swept).
#
# Knobs (env-var-driven so we can target a focused sub-sweep):
#   SIGNALS  default: "mass entropy-w branching depth-weight"
#   MODES    default: "log sqrt linear inv-log inv-linear"
#   SEEDS    default: "1 2 3"
#   OUT      default: rnd/gutenberg-anc-sweep
#
# Usage:
#   # Full sweep (4 signals × 5 modes × 3 seeds = 60 cells, ~5h on Gutenberg):
#   bash rnd/gutenberg-anc-sweep/run.sh
#
#   # Focused (just branching across all modes):
#   SIGNALS=branching bash rnd/gutenberg-anc-sweep/run.sh
#
#   # Focused (top-2 signals only):
#   SIGNALS="branching entropy-w" bash rnd/gutenberg-anc-sweep/run.sh
#
# Outputs: $OUT/logs/<label>/{train.log, run.model, heldout_ppl.txt}
#          $OUT/results.txt  (flat label PPL=X log)
#
# Cells with completed heldout_ppl.txt are skipped on re-invocation (caching),
# so partial runs can resume cleanly.

set -eu

OUT=${OUT:-rnd/gutenberg-anc-sweep}
mkdir -p $OUT/logs
RESULTS=$OUT/results.txt
echo "# Gutenberg --anc-grad sweep, started $(date -u)" >> $RESULTS

SIGNALS="${SIGNALS:-mass entropy-w branching depth-weight}"
MODES="${MODES:-log sqrt linear inv-log inv-linear}"
SEEDS="${SEEDS:-1 2 3}"

TRIE=/tmp/gutenberg_5m_baseline_d16_radix
HOLDOUT=/tmp/gut_holdout.txt
VOCAB=data/input.txt   # Vocab built from Shakespeare; Gutenberg uses same 65-char alphabet

# Regenerate Kaiming init checkpoints if not present. The --init code in
# src/cuda/agpt_train.cu is deterministic from --init-seed; bytes match
# whatever's already on disk on the laptop.
for s in $SEEDS; do
    if [ ! -f /tmp/agpt_init_kaiming_s${s}.model ]; then
        echo "Building Kaiming init for seed $s..."
        bin/agpt_train --init --init-seed $s --epochs 0 \
            --trie-dir $TRIE \
            --save /tmp/agpt_init_kaiming_s${s}.model > /dev/null 2>&1
    fi
done

# Auto-detect available evaluator. Prefer Python reference (independent
# of the trainer's CUDA kernels); fall back to the Crystal sliding-window
# tool with --pool deep_only when torch isn't available (runpod images
# lack Python). The Crystal deep_only mode produces byte-identical PPL
# to Python --mode fixed (cross-validated 2026-05-22, 4-decimal match).
EVAL_CMD=""
if python3 -c "import torch" 2>/dev/null; then
    EVAL_CMD="python3 src/tools/agpt_ppl.py --d 16 --max-positions 10000 --mode fixed"
    echo "Evaluator: PyTorch reference (src/tools/agpt_ppl.py)"
elif [ -x bin/agpt_sliding_window_perplexity ]; then
    EVAL_CMD="bin/agpt_sliding_window_perplexity --d 16 --max-positions 10000 --backend openblas --workers 8 --pool deep_only"
    echo "Evaluator: Crystal sliding-window (--pool deep_only)"
else
    echo "ERROR: no evaluator available — need python3+torch OR bin/agpt_sliding_window_perplexity"
    exit 1
fi

run_cell() {
    local label="$1"
    local flag="$2"
    local seed="$3"
    local D=$OUT/logs/$label
    if [ -f $D/heldout_ppl.txt ] && [ -s $D/heldout_ppl.txt ]; then
        echo "[cached] $label heldout=$(cat $D/heldout_ppl.txt)" | tee -a $RESULTS
        return
    fi
    rm -rf $D && mkdir -p $D
    local START=$(date +%s)
    bin/agpt_train --model /tmp/agpt_init_kaiming_s${seed}.model \
        --trie-dir $TRIE \
        --epochs 10 --lr 3e-3 --optimizer rmsprop \
        --lr-schedule warmup-cosine --warmup-epochs 1 \
        --partition-depth 1 --no-accumulate \
        --mass-weight off \
        --anc-grad \
        $flag \
        --save $D/run.model > $D/train.log 2>&1
    local TRAIN_WALL=$(($(date +%s) - START))
    local PPL=$($EVAL_CMD --model $D/run.model --file $HOLDOUT --vocab-file $VOCAB 2>/dev/null \
        | awk '/^Perplexity/ {print $2}')
    echo "$PPL" > $D/heldout_ppl.txt
    echo "$label train_wall=${TRAIN_WALL}s PPL=$PPL" | tee -a $RESULTS
}

# Baseline (no extra weighting)
for s in $SEEDS; do
    run_cell "baseline_s${s}" "" "$s"
done

# Sweep: for each signal × mode × seed
for sig in $SIGNALS; do
    case $sig in
        mass)         flag_prefix="--mass-weight" ;;
        entropy-w)    flag_prefix="--entropy-weight" ;;
        branching)    flag_prefix="--branching-weight" ;;
        depth-weight) flag_prefix="--depth-weight" ;;
        *) echo "unknown signal: $sig" >&2; exit 1 ;;
    esac
    for mode in $MODES; do
        for s in $SEEDS; do
            run_cell "${sig}-${mode}_s${s}" "$flag_prefix $mode" "$s"
        done
    done
done

# Summary
echo "" | tee -a $OUT/summary.txt
echo "=== Gutenberg anc-grad sweep, $(date -u) ===" | tee -a $OUT/summary.txt
echo "Signals: $SIGNALS  Modes: $MODES  Seeds: $SEEDS" | tee -a $OUT/summary.txt
echo "" | tee -a $OUT/summary.txt

for label_pattern in "baseline" $SIGNALS; do
    for mode in "" $MODES; do
        # Construct grep pattern
        if [ "$label_pattern" = "baseline" ] && [ -z "$mode" ]; then
            pat="^baseline_s"
            label="baseline (depth+anc, mw=off)"
        elif [ "$label_pattern" != "baseline" ] && [ -n "$mode" ]; then
            pat="^${label_pattern}-${mode}_s"
            label="${label_pattern} ${mode}"
        else
            continue
        fi
        PPLS=$(grep "$pat" $RESULTS | grep -oE 'PPL=[0-9.]+' | sed 's/PPL=//' | tr '\n' ' ')
        if [ -n "$PPLS" ]; then
            STATS=$(echo "$PPLS" | awk '{s=0; ss=0; n=0; for (i=1; i<=NF; i++) {s+=$i; ss+=$i*$i; n+=1}; m=s/n; v=ss/n-m*m; sd=sqrt(v>0?v:0); printf "%-40s mean=%.3f ± %.3f (n=%d)", "'"$label"'", m, sd, n}')
            echo "$STATS" | tee -a $OUT/summary.txt
        fi
    done
done

echo "" | tee -a $OUT/summary.txt
echo "Full per-cell results: $RESULTS" | tee -a $OUT/summary.txt
