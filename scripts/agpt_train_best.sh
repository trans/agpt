#!/usr/bin/env bash
# Train for up to N super-epochs, evaluate each intermediate checkpoint against
# a held-out tail of the corpus, and print the best PPL + checkpoint path.
#
# This is the external-process version of "train-to-convergence". It doesn't
# stop the trainer early when loss plateaus — it runs the full requested
# budget, then picks the best checkpoint from the ones written every
# super-epoch. Spending the full budget is usually cheap ($< 30 min at d=32)
# and far simpler than threading an eval loop into the CUDA trainer.
#
# Usage:
#   scripts/agpt_train_best.sh \
#       --model-init /path/to/init.model \
#       --trie-dir /tmp/agpt_input_d32_per \
#       --epochs 10 \
#       --save-best /tmp/best.model \
#       --eval-positions 16384 \
#       -- \
#       [remaining args passed verbatim to bin/agpt_train]
set -eu -o pipefail

MODEL_INIT=""
TRIE_DIR=""
EPOCHS=""
SAVE_BEST=""
EVAL_POSITIONS=16384
EVAL_FILE="data/input.txt"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-init)      MODEL_INIT="$2"; shift 2 ;;
        --trie-dir)        TRIE_DIR="$2"; shift 2 ;;
        --epochs)          EPOCHS="$2"; shift 2 ;;
        --save-best)       SAVE_BEST="$2"; shift 2 ;;
        --eval-positions)  EVAL_POSITIONS="$2"; shift 2 ;;
        --eval-file)       EVAL_FILE="$2"; shift 2 ;;
        --) shift; break ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

: "${MODEL_INIT:?--model-init is required}"
: "${TRIE_DIR:?--trie-dir is required}"
: "${EPOCHS:?--epochs is required}"
: "${SAVE_BEST:?--save-best is required}"

WORK_MODEL="${SAVE_BEST}.work"
cp "$MODEL_INIT" "$WORK_MODEL"

echo "=== Training $EPOCHS super-epochs with --save-every 1 ==="
bin/agpt_train \
    --model "$WORK_MODEL" \
    --trie-dir "$TRIE_DIR" \
    --epochs "$EPOCHS" \
    --save-every 1 \
    --save "$WORK_MODEL" \
    "$@"

echo
echo "=== Evaluating each checkpoint on held-out PPL ==="
BEST_PPL=""
BEST_EP=""
for ep in $(seq 1 "$EPOCHS"); do
    CK="${WORK_MODEL}.ep${ep}"
    [[ -f "$CK" ]] || continue
    PPL=$(bin/perplexity \
            --model "$CK" \
            --file "$EVAL_FILE" \
            --max-positions "$EVAL_POSITIONS" \
            --backend openblas 2>&1 \
          | awk '/^Perplexity:/ {print $2}')
    echo "  ep${ep}: PPL = $PPL"
    if [[ -z "$BEST_PPL" ]] || awk "BEGIN{exit !($PPL < $BEST_PPL)}"; then
        BEST_PPL="$PPL"
        BEST_EP="$ep"
    fi
done

if [[ -z "$BEST_EP" ]]; then
    echo "No checkpoints produced." >&2
    exit 1
fi

cp "${WORK_MODEL}.ep${BEST_EP}" "$SAVE_BEST"
echo
echo "=== Best: ep${BEST_EP} with PPL = $BEST_PPL ==="
echo "Saved to: $SAVE_BEST"
