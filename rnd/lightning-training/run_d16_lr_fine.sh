#!/usr/bin/env bash
# Finer LR sweep at d=16, s=260 p_stop=0.3 (the tight-variance sweet spot).
# Prior d=16 s=260 lr=3e-4: 16.06 mean, spread 0.64.
# Bracket with 5e-4, 2e-4, 1e-4, 5e-5.

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
TRIE=/tmp/agpt_input_d16_radix_pst
BIN="bin/agpt_train"

GIT_HASH=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)

eval_ppl () {
    bin/perplexity --model "$1" --file data/input.txt \
        --max-positions $EVAL_POS --backend cublas 2>&1 \
      | awk '/^Perplexity:/ {print $2}'
}

run_lightning () {
    local label="$1"; local pstop="$2"; local steps="$3"; local epochs="$4"; local lr="$5"
    local cfg="$OUT/${label}.json"
    cat > "$cfg" <<EOF
{"label":"$label","trie_dir":"$TRIE","init_checkpoint":"$INIT_CKPT",
 "epochs":$epochs,"lr":$lr,"optimizer":"rmsprop","rmsprop_beta":0.999,
 "lr_schedule":"constant","entropy_lambda":1.0,"mass_weight":"linear",
 "n_runs":$N_RUNS,"lightning_steps":$steps,"lightning_sampler":"l3",
 "lightning_p_stop":$pstop,"lightning_mass_lr":"off",
 "git_hash":"$GIT_HASH","timestamp":"$(date -Iseconds)"}
EOF
    for i in $(seq 1 $N_RUNS); do
        local work="/tmp/lightning_d16_${label}_r${i}.model"
        cp "$INIT_CKPT" "$work"
        local log="$OUT/${label}_r${i}.log"
        local t0=$SECONDS
        $BIN --model "$work" --trie-dir "$TRIE" --save "$work" \
            --epochs "$epochs" --lr "$lr" --optimizer rmsprop --rmsprop-beta 0.999 \
            --lr-schedule constant --entropy-lambda 1.0 --mass-weight linear \
            --lightning-steps "$steps" --lightning-sampler l3 --lightning-p-stop "$pstop" \
            --lightning-seed "$((42 + i))" > "$log" 2>&1
        local rc=$?
        local elapsed=$((SECONDS - t0))
        local ppl=$(eval_ppl "$work")
        echo "$label  run $i  PPL = $ppl  time = ${elapsed}s  (rc=$rc)"
    done
}

echo "=============================================================="
echo "d=16 Lightning fine LR sweep at s=260 (per-subtree format)"
echo "  baseline det 3 SE × 65 lr=3e-3: 14.59"
echo "  prior s=260 lr=3e-4: 16.06 mean, spread 0.64"
echo "=============================================================="

run_lightning d16-l3-s260-lr5e-4  0.3  260  3  5e-4
run_lightning d16-l3-s260-lr2e-4  0.3  260  3  2e-4
run_lightning d16-l3-s260-lr1e-4  0.3  260  3  1e-4
run_lightning d16-l3-s260-lr5e-5  0.3  260  3  5e-5

echo ""
echo "Done."
