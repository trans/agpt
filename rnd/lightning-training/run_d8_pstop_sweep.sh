#!/usr/bin/env bash
# Lightning L3 p_stop sweep at d=8. Matched total optimizer steps against
# per-root-child baseline (65 steps/SE × 3 SE = 195 steps at lr=3e-3 → 17.99).
# Sweep p_stop ∈ {0.2, 0.3, 0.5}. 3 runs per cell.

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
cd "$PROJECT_ROOT" || { echo "Cannot cd to PROJECT_ROOT=$PROJECT_ROOT"; exit 1; }
if [ ! -f Justfile ]; then
    echo "PROJECT_ROOT=$PROJECT_ROOT doesn't look like the microgpt project root" >&2
    exit 1
fi

OUT="rnd/lightning-training/logs"
mkdir -p "$OUT"
N_RUNS=3
EVAL_POS=16384
INIT_CKPT="data/input.random.model"
TRIE=/tmp/agpt_input_d8_radix
BIN="bin/agpt_train"

GIT_HASH=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)

eval_ppl () {
    local checkpoint="$1"
    bin/perplexity --model "$checkpoint" --file data/input.txt \
        --max-positions $EVAL_POS --backend cublas 2>&1 \
      | awk '/^Perplexity:/ {print $2}'
}

run_lightning () {
    local label="$1"; local pstop="$2"; local steps="$3"; local epochs="$4"; local lr="$5"
    local cfg="$OUT/${label}.json"
    cat > "$cfg" <<EOF
{
  "label": "$label",
  "tool": "$BIN",
  "trie_dir": "$TRIE",
  "trie_format": "global-radix",
  "init_checkpoint": "$INIT_CKPT",
  "epochs": $epochs,
  "lr": $lr,
  "optimizer": "rmsprop",
  "rmsprop_beta": 0.999,
  "lr_schedule": "constant",
  "entropy_lambda": 1.0,
  "mass_weight": "linear",
  "eval_positions": $EVAL_POS,
  "eval_backend": "cublas",
  "n_runs": $N_RUNS,
  "lightning_steps": $steps,
  "lightning_sampler": "l3",
  "lightning_p_stop": $pstop,
  "git_hash": "$GIT_HASH",
  "timestamp": "$(date -Iseconds)"
}
EOF
    for i in $(seq 1 $N_RUNS); do
        local work="/tmp/lightning_${label}_r${i}.model"
        cp "$INIT_CKPT" "$work"
        local log="$OUT/${label}_r${i}.log"
        local t0=$SECONDS
        $BIN \
            --model "$work" \
            --trie-dir "$TRIE" \
            --save "$work" \
            --epochs "$epochs" \
            --lr "$lr" \
            --optimizer rmsprop --rmsprop-beta 0.999 \
            --lr-schedule constant \
            --entropy-lambda 1.0 \
            --mass-weight linear \
            --lightning-steps "$steps" \
            --lightning-sampler l3 \
            --lightning-p-stop "$pstop" \
            --lightning-seed "$((42 + i))" \
            > "$log" 2>&1
        local rc=$?
        local elapsed=$((SECONDS - t0))
        local ppl=$(eval_ppl "$work")
        echo "$label  run $i  PPL = $ppl  time = ${elapsed}s  (rc=$rc)"
    done
}

echo "=============================================================="
echo "Lightning L3 p_stop sweep at d=8 (global radix)"
echo "  init: $INIT_CKPT  trie: $TRIE  git: $GIT_HASH"
echo "  baseline ref (d=8, 3 SE, 65 steps/SE, per-root-child, lr=3e-3): 17.99 mean"
echo "=============================================================="

# Matched step budget: 65 samples/SE × 3 SE = 195 total optimizer steps.
run_lightning l3-pstop-0.2-s65-ep3  0.2  65  3  3e-3
run_lightning l3-pstop-0.3-s65-ep3  0.3  65  3  3e-3
run_lightning l3-pstop-0.5-s65-ep3  0.5  65  3  3e-3

echo ""
echo "Done."
