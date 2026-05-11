#!/usr/bin/env bash
# AGPT mass-weight mode sweep. Simpler, non-functional script — no set -e,
# no functions. Easier to debug if something goes wrong.

# The caller's CWD should be the project root. Capture it via PROJECT_ROOT
# env var (defaults to $PWD at invocation time). Don't rely on $0 or
# BASH_SOURCE — under some launch mechanisms (Bash-tool background,
# nohup with an unusual parent state, cron, etc.) CWD may have been
# reset to / before the script starts, so $0 can resolve wrong.
PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
cd "$PROJECT_ROOT" || { echo "Cannot cd to PROJECT_ROOT=$PROJECT_ROOT"; exit 1; }
# Sanity check: project root should have the Justfile.
if [ ! -f Justfile ]; then
    echo "PROJECT_ROOT=$PROJECT_ROOT doesn't look like the microgpt project root"
    echo "(no Justfile present). Invoke from the project root, or set"
    echo "PROJECT_ROOT env var explicitly." >&2
    exit 1
fi
OUT="rnd/sgd-sanity-check/logs"
RES="rnd/sgd-sanity-check/results"
mkdir -p "$OUT" "$RES"
EVAL_POS=16384

GIT_HASH=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)

do_run () {
    local label="$1"; shift
    local trie="$1"; shift
    local mass_mode="$1"; shift  # informational only, for config record
    # remaining args are passed through to agpt_train
    local log="$OUT/mw_${label}.log"
    local cfg="$OUT/mw_${label}.json"
    # Record the config as a JSON sidecar for reproducibility.
    cat > "$cfg" <<EOF
{
  "label": "$label",
  "tool": "bin/agpt_train (via scripts/agpt_train_best.sh)",
  "trie_dir": "$trie",
  "init_checkpoint": "data/input.agpt.model",
  "epochs": 3,
  "lr": 3e-3,
  "optimizer": "rmsprop",
  "rmsprop_beta": 0.999,
  "lr_schedule": "warmup-cosine",
  "warmup_epochs": 1,
  "single_subtree": true,
  "entropy_lambda": 1.0,
  "mass_weight": "$mass_mode",
  "eval_positions": $EVAL_POS,
  "eval_backend": "cublas",
  "git_hash": "$GIT_HASH",
  "timestamp": "$(date -Iseconds)"
}
EOF
    echo "  -> $label ..."
    bash scripts/agpt_train_best.sh \
        --model-init data/input.agpt.model \
        --trie-dir "$trie" \
        --epochs 3 \
        --save-best "/tmp/mw_${label}.model" \
        --eval-positions $EVAL_POS \
        -- \
        --lr 3e-3 --optimizer rmsprop --rmsprop-beta 0.999 \
        --lr-schedule warmup-cosine --warmup-epochs 1 \
        --single-subtree --entropy-lambda 1.0 \
        "$@" > "$log" 2>&1
    local rc=$?
    local best_ppl
    best_ppl=$(grep -E "^=== Best:" "$log" | sed -E 's/.*PPL = ([0-9.]+).*/\1/')
    if [[ -z "$best_ppl" ]]; then
        echo "     FAILED (rc=$rc) — see $log"
    else
        echo "     $label  best PPL = $best_ppl  (rc=$rc)"
    fi
}

echo "=============================================================="
echo "AGPT mass-weight mode sweep"
echo "=============================================================="

for depth in d16 d32; do
    case "$depth" in
        d16) trie=/tmp/agpt_input_d16_radix_pst ;;
        d32) trie=/tmp/agpt_input_d32_per ;;
    esac
    for mode in off log sqrt linear; do
        flag=""
        [[ "$mode" != "off" ]] && flag="--mass-weight $mode"
        for i in 1 2 3; do
            do_run "agpt-${depth}-${mode}_r${i}" "$trie" "$mode" $flag
        done
    done
done

echo ""
echo "All done. Summarize:"
echo "  grep -H '=== Best:' $OUT/mw_agpt-*.log"
