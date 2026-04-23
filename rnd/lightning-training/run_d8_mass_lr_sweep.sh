#!/usr/bin/env bash
# Per-sample mass-weighted LR scaling sweep at d=8.
# Tests whether --lightning-mass-lr {log, sqrt, linear} helps L3.
#
# Control: L3 p_stop=0.3, 65 steps × 3 SE, matched to deterministic baseline.
# Baseline: 17.99 PPL.
# Prior L3 at same config (no mass-lr): 20.12 PPL mean.

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
    local label="$1"; local pstop="$2"; local mass_lr="$3"; local steps="$4"; local epochs="$5"; local lr="$6"
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
  "lightning_mass_lr": "$mass_lr",
  "git_hash": "$GIT_HASH",
  "timestamp": "$(date -Iseconds)"
}
EOF
    for i in $(seq 1 $N_RUNS); do
        local work="/tmp/lightning_${label}_r${i}.model"
        cp "$INIT_CKPT" "$work"
        local log="$OUT/${label}_r${i}.log"
        local t0=$SECONDS
        local extra_flags=""
        if [ "$mass_lr" != "off" ]; then
            extra_flags="--lightning-mass-lr $mass_lr"
        fi
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
            $extra_flags \
            > "$log" 2>&1
        local rc=$?
        local elapsed=$((SECONDS - t0))
        local ppl=$(eval_ppl "$work")
        echo "$label  run $i  PPL = $ppl  time = ${elapsed}s  (rc=$rc)"
    done
}

echo "=============================================================="
echo "Lightning L3 mass-lr sweep at d=8"
echo "  init: $INIT_CKPT  trie: $TRIE  git: $GIT_HASH"
echo "  baseline det 65×3: 17.99  |  L3 p_stop=0.3 no-mass-lr: 20.12"
echo "=============================================================="

# L3 p_stop=0.3 matched-budget (65 × 3) across mass-lr modes.
run_lightning l3-0.3-mass-off    0.3  off    65  3  3e-3
run_lightning l3-0.3-mass-log    0.3  log    65  3  3e-3
run_lightning l3-0.3-mass-sqrt   0.3  sqrt   65  3  3e-3
run_lightning l3-0.3-mass-linear 0.3  linear 65  3  3e-3

# Also check L3 p_stop=0.5 with log — the mildly-shallow config.
run_lightning l3-0.5-mass-log    0.5  log    65  3  3e-3

echo ""
echo "Done."
