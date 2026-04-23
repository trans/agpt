#!/usr/bin/env bash
# Lightning L3 step-budget × LR sweep at d=8.
# Prior s65 best: L3 p_stop=0.3 mass-off → 18.71 (baseline 17.99).
# Prior s260 at fixed lr=3e-3 was 26.13 — LR not retuned.
# This sweep tests whether lower LRs at 4× and 10× step budgets close the gap.

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
  "lightning_mass_lr": "off",
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
echo "Lightning L3 step-budget × LR sweep at d=8"
echo "  init: $INIT_CKPT  trie: $TRIE  git: $GIT_HASH"
echo "  baseline det 65×3 lr=3e-3: 17.99  |  L3 p_stop=0.3 mass-off s65: 18.71"
echo "=============================================================="

# 4× budget (260/SE × 3 SE = 780 total steps). LR sweep.
run_lightning l3-s260-lr1e-3  0.3  260  3  1e-3
run_lightning l3-s260-lr3e-4  0.3  260  3  3e-4
run_lightning l3-s260-lr1e-4  0.3  260  3  1e-4

# 10× budget (650/SE × 3 SE = 1950 total steps). LR sweep.
run_lightning l3-s650-lr3e-4  0.3  650  3  3e-4
run_lightning l3-s650-lr1e-4  0.3  650  3  1e-4

echo ""
echo "Done."
