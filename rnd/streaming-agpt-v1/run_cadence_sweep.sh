#!/usr/bin/env bash
# Cadence sweep for streaming AGPT. Same total SE budget (100) but varying
# number of stages. Compares against the existing baseline from run.sh.
#
# Usage: bash run_cadence_sweep.sh [N_STAGES_1 N_STAGES_2 ...]
#   Default: tests 5, 10, 20, 50 stages.

set -euo pipefail

PROJ="${PROJ:-$(pwd)}"
CORPUS="${PROJ}/data/input.txt"
RND="${PROJ}/rnd/streaming-agpt-v1"
LOGS="${RND}/logs"
MODELS_DIR="${RND}/models"

LR=3e-3
PD=1
TOTAL_BUDGET_SE=100
D=16
INIT_MODEL="${PROJ}/data/input.random.model"

if [ $# -gt 0 ]; then
  CADENCES=("$@")
else
  CADENCES=(5 10 20 50)
fi

mkdir -p "${LOGS}" "${MODELS_DIR}"
CORPUS_BYTES=$(wc -c < "${CORPUS}")
echo "Corpus: ${CORPUS} (${CORPUS_BYTES} bytes), total budget ${TOTAL_BUDGET_SE} SE"
echo ""

declare -A RESULTS_PPL RESULTS_WALL

for N_STAGES in "${CADENCES[@]}"; do
  PER_STAGE_SE=$((TOTAL_BUDGET_SE / N_STAGES))
  if [ $((PER_STAGE_SE * N_STAGES)) -ne $TOTAL_BUDGET_SE ]; then
    echo "  Skipping N_STAGES=$N_STAGES (doesn't divide budget cleanly)" >&2
    continue
  fi

  echo "========== Cadence: ${N_STAGES} stages × ${PER_STAGE_SE} SE =========="

  prev_model="${INIT_MODEL}"
  cadence_start=$(date +%s)

  for ((i = 1; i <= N_STAGES; i++)); do
    pct=$((100 * i / N_STAGES))
    bytes=$((CORPUS_BYTES * pct / 100))
    trunc_corpus="/tmp/shake_${pct}pct_n${N_STAGES}.txt"
    head -c "${bytes}" "${CORPUS}" > "${trunc_corpus}"

    trie_dir="/tmp/shake_${pct}pct_n${N_STAGES}_d${D}_radix"
    rm -rf "${trie_dir}"

    "${PROJ}/bin/agpt_build_radix_corpus" \
      --corpus "${trunc_corpus}" \
      --vocab-file "${CORPUS}" \
      --max-depth "${D}" \
      --out "${trie_dir}" \
      > "${LOGS}/cadence_n${N_STAGES}_stage${i}_build.log" 2>&1

    out_model="${MODELS_DIR}/cadence_n${N_STAGES}_stage${i}.model"
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
      > "${LOGS}/cadence_n${N_STAGES}_stage${i}_train.log" 2>&1

    prev_model="${out_model}"
  done

  cadence_end=$(date +%s)
  wall=$((cadence_end - cadence_start))

  # Eval final stage
  ppl_log="${LOGS}/cadence_n${N_STAGES}_final_ppl.log"
  "${PROJ}/bin/perplexity" \
    --model "${prev_model}" \
    --file "${CORPUS}" \
    --seq-len "${D}" \
    --backend openblas \
    --max-positions 4096 \
    > "${ppl_log}" 2>&1
  ppl=$(grep "^Perplexity:" "${ppl_log}" | awk '{print $2}')

  RESULTS_PPL[$N_STAGES]="$ppl"
  RESULTS_WALL[$N_STAGES]="$wall"
  echo "  → ${N_STAGES} × ${PER_STAGE_SE} SE: PPL=${ppl}, wall=${wall}s"
  echo ""
done

echo "========== Cadence sweep summary =========="
printf "%-15s %-10s %-10s\n" "n_stages × se" "PPL@${D}" "wall (s)"
printf "%-15s %-10s %-10s\n" "--------------" "------" "----"
for n in "${CADENCES[@]}"; do
  if [ -n "${RESULTS_PPL[$n]:-}" ]; then
    per_stage=$((TOTAL_BUDGET_SE / n))
    printf "%-15s %-10s %-10s\n" "${n} × ${per_stage}" "${RESULTS_PPL[$n]}" "${RESULTS_WALL[$n]}"
  fi
done
echo ""
echo "For reference (from prior run):"
echo "  5 × 20  (v3 streaming): PPL=4.48, wall=388s"
echo "  baseline 100 SE:        PPL=4.74, wall=596s"
