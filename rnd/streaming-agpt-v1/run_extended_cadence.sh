#!/usr/bin/env bash
# Extended cadence runs at higher total SE budget.
# Each spec is "N_STAGES PER_STAGE_SE" — total budget = product.
# Sets --total-epochs-budget to the product so LR schedule is global.

set -euo pipefail

PROJ="${PROJ:-$(pwd)}"
CORPUS="${PROJ}/data/input.txt"
RND="${PROJ}/rnd/streaming-agpt-v1"
LOGS="${RND}/logs"
MODELS_DIR="${RND}/models"

LR=3e-3
PD=1
D=16
INIT_MODEL="${PROJ}/data/input.random.model"

# Specs: "n_stages per_stage_se" pairs. Defaults to both user requests.
# If only one arg given, runs just that one.
if [ $# -ge 2 ]; then
  SPECS=("$@")
elif [ $# -eq 1 ]; then
  SPECS=("$1")
else
  SPECS=("20 10" "50 10")
fi

mkdir -p "${LOGS}" "${MODELS_DIR}"
CORPUS_BYTES=$(wc -c < "${CORPUS}")
echo "Corpus: ${CORPUS} (${CORPUS_BYTES} bytes)"
echo ""

declare -A RESULTS_PPL RESULTS_WALL

for spec in "${SPECS[@]}"; do
  read N_STAGES PER_STAGE_SE <<< "$spec"
  TOTAL_BUDGET_SE=$((N_STAGES * PER_STAGE_SE))
  TAG="n${N_STAGES}_se${PER_STAGE_SE}"

  echo "========== Cadence: ${N_STAGES} × ${PER_STAGE_SE} SE = ${TOTAL_BUDGET_SE} SE total =========="

  prev_model="${INIT_MODEL}"
  cadence_start=$(date +%s)

  for ((i = 1; i <= N_STAGES; i++)); do
    pct=$((100 * i / N_STAGES))
    bytes=$((CORPUS_BYTES * pct / 100))
    trunc_corpus="/tmp/shake_${pct}pct_${TAG}.txt"
    head -c "${bytes}" "${CORPUS}" > "${trunc_corpus}"

    trie_dir="/tmp/shake_${pct}pct_${TAG}_d${D}_radix"
    rm -rf "${trie_dir}"

    "${PROJ}/bin/agpt_build_radix_corpus" \
      --corpus "${trunc_corpus}" \
      --vocab-file "${CORPUS}" \
      --max-depth "${D}" \
      --out "${trie_dir}" \
      > "${LOGS}/cadence_${TAG}_stage${i}_build.log" 2>&1

    out_model="${MODELS_DIR}/cadence_${TAG}_stage${i}.model"
    cp "${prev_model}" "${out_model}"

    "${PROJ}/bin/agpt_train" \
      --model "${out_model}" \
      --trie-dir "${trie_dir}" \
      --save "${out_model}" \
      --epochs "${PER_STAGE_SE}" \
      --total-epochs-budget "${TOTAL_BUDGET_SE}" \
      --partition-depth "${PD}" --no-accumulate \
      --lr "${LR}" --lr-schedule warmup-cosine --warmup-epochs 1 \
      --optimizer rmsprop --rmsprop-beta 0.999 \
      --mass-weight log --entropy-lambda 1.0 \
      > "${LOGS}/cadence_${TAG}_stage${i}_train.log" 2>&1

    prev_model="${out_model}"
  done

  cadence_end=$(date +%s)
  wall=$((cadence_end - cadence_start))

  ppl_log="${LOGS}/cadence_${TAG}_final_ppl.log"
  "${PROJ}/bin/perplexity" \
    --model "${prev_model}" \
    --file "${CORPUS}" \
    --seq-len "${D}" \
    --backend openblas \
    --max-positions 4096 \
    > "${ppl_log}" 2>&1
  ppl=$(grep "^Perplexity:" "${ppl_log}" | awk '{print $2}')

  RESULTS_PPL[$TAG]="$ppl"
  RESULTS_WALL[$TAG]="$wall"
  echo "  → ${N_STAGES} × ${PER_STAGE_SE} SE (${TOTAL_BUDGET_SE} total): PPL=${ppl}, wall=${wall}s"
  echo ""
done

echo "========== Extended cadence summary =========="
printf "%-20s %-10s %-10s\n" "n × se (total)" "PPL@${D}" "wall (s)"
printf "%-20s %-10s %-10s\n" "--------------" "------" "----"
for spec in "${SPECS[@]}"; do
  read N PSE <<< "$spec"
  TAG="n${N}_se${PSE}"
  TOT=$((N * PSE))
  if [ -n "${RESULTS_PPL[$TAG]:-}" ]; then
    printf "%-20s %-10s %-10s\n" "${N} × ${PSE} (${TOT})" "${RESULTS_PPL[$TAG]}" "${RESULTS_WALL[$TAG]}"
  fi
done
echo ""
echo "For reference (100 SE total budget):"
echo "  Baseline 100 SE:      PPL=4.74, wall=596s"
echo "  Streaming 20 × 5 SE:  PPL=4.33, wall=361s ← best at 100 SE"
echo "  Streaming  5 × 20 SE: PPL=4.47, wall=366s"
