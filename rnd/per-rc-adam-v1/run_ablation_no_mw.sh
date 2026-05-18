#!/usr/bin/env bash
# Stage 1 ablation: same comparison as run_compare.sh but with NO mass weighting.
# Tests whether the per-rc regression was amplified by mass-weighting cold-start
# interaction, or whether per-rc fails regardless.

set -euo pipefail

PROJ="${PROJ:-$(pwd)}"
CORPUS="${CORPUS:-${PROJ}/data/input.txt}"
RND="${PROJ}/rnd/per-rc-adam-v1"
LOGS="${RND}/logs_no_mw"
MODELS="${RND}/models_no_mw"
TRIE="/tmp/shake_baseline_d16_radix"

SE_BUDGET=50
SEEDS=(100 200 300)
D=16

mkdir -p "${LOGS}" "${MODELS}"

run_one() {
    local variant=$1
    local seed=$2
    local init="/tmp/init_seed${seed}.model"
    [ -f "${init}" ] || { echo "Missing: ${init}"; exit 1; }
    local out_model="${MODELS}/${variant}_seed${seed}.model"
    local log="${LOGS}/${variant}_seed${seed}_train.log"
    local ppl_log="${LOGS}/${variant}_seed${seed}_ppl.log"

    cp "${init}" "${out_model}"

    local per_rc_flag=""
    [ "${variant}" = "per_rc" ] && per_rc_flag="--per-rc-adam"

    echo "===== ${variant} seed=${seed} (no mass-weight) ====="
    local start=$(date +%s)
    "${PROJ}/bin/agpt_train" \
        --model "${out_model}" --trie-dir "${TRIE}" \
        --save "${out_model}" --epochs "${SE_BUDGET}" \
        --partition-depth 1 --no-accumulate \
        --lr 3e-3 --lr-schedule warmup-cosine --warmup-epochs 1 \
        --optimizer rmsprop --rmsprop-beta 0.999 \
        --entropy-lambda 1.0 \
        ${per_rc_flag} \
        > "${log}" 2>&1
    local end=$(date +%s)
    local wall=$((end - start))

    "${PROJ}/bin/perplexity" \
        --model "${out_model}" --file "${CORPUS}" \
        --seq-len "${D}" --backend openblas --max-positions 4096 \
        > "${ppl_log}" 2>&1
    local ppl=$(grep "^Perplexity:" "${ppl_log}" | awk '{print $2}')
    echo "  ${variant} seed=${seed}: PPL=${ppl}, wall=${wall}s"
    echo "${variant},${seed},${ppl},${wall}" >> "${RND}/no_mw_results.csv"
}

echo "variant,seed,ppl,wall_sec" > "${RND}/no_mw_results.csv"

for SEED in "${SEEDS[@]}"; do
    run_one baseline "${SEED}"
    run_one per_rc   "${SEED}"
done

echo ""
echo "===== Stage 1 ablation (no mass-weight) ${SE_BUDGET} SE Shakespeare d=${D} summary ====="
python3 - "${RND}/no_mw_results.csv" <<'PYEOF'
import csv, sys, statistics
rows = list(csv.DictReader(open(sys.argv[1])))
by_v = {}
for r in rows:
    by_v.setdefault(r['variant'], []).append((float(r['ppl']), int(r['wall_sec'])))
for v in ('baseline','per_rc'):
    ppls = [p for p,_ in by_v.get(v,[])]
    walls = [w for _,w in by_v.get(v,[])]
    if not ppls: continue
    mean = statistics.mean(ppls); sd = statistics.stdev(ppls) if len(ppls)>1 else 0
    mw = statistics.mean(walls)
    print(f"  {v:10s}: PPL {mean:.4f} +/- {sd:.4f}  (wall mean {mw:.0f}s, n={len(ppls)})")
PYEOF
