#!/usr/bin/env bash
# Generic multi-seed streaming runner.
# Args: N_STAGES PER_STAGE_SE [SEEDS...]
# Defaults: SEEDS=(100 200 300)
# Total SE budget = N_STAGES * PER_STAGE_SE

set -euo pipefail

PROJ="${PROJ:-$(pwd)}"
CORPUS="${CORPUS:-${PROJ}/data/input.txt}"
RND="${PROJ}/rnd/streaming-agpt-v1"
LOGS="${RND}/logs"
MODELS_DIR="${RND}/models"

if [ $# -lt 2 ]; then
  echo "Usage: $0 N_STAGES PER_STAGE_SE [SEEDS...]"
  echo "  Example: $0 100 1            # 100 × 1 SE = 100 SE total, default seeds"
  echo "  Example: $0 250 2 100 200 300 # 250 × 2 SE = 500 SE total"
  exit 1
fi

N_STAGES=$1
PER_STAGE_SE=$2
shift 2
if [ $# -ge 1 ]; then
  SEEDS=("$@")
else
  SEEDS=(100 200 300)
fi

SE_BUDGET=$((N_STAGES * PER_STAGE_SE))
TAG="ms_n${N_STAGES}_se${PER_STAGE_SE}"

LR=3e-3
PD=1
D=16

mkdir -p "${LOGS}" "${MODELS_DIR}"
CORPUS_BYTES=$(wc -c < "${CORPUS}")
echo "Generic multiseed: N_STAGES=${N_STAGES} PER_STAGE_SE=${PER_STAGE_SE} (total ${SE_BUDGET} SE)"
echo "Seeds: ${SEEDS[*]}"
echo "Corpus: ${CORPUS} (${CORPUS_BYTES} bytes)"
echo ""

declare -A PPL WALL
for SEED in "${SEEDS[@]}"; do
  INIT="/tmp/init_seed${SEED}.model"
  [ -f "${INIT}" ] || { echo "Missing init: ${INIT}"; exit 1; }

  echo "===== SEED ${SEED} — Streaming ${N_STAGES} × ${PER_STAGE_SE} SE ====="
  prev_model="${INIT}"
  prev_trie=""
  s_start=$(date +%s)
  for ((i = 1; i <= N_STAGES; i++)); do
    pct=$((100 * i / N_STAGES))
    bytes=$((CORPUS_BYTES * pct / 100))
    trunc_corpus="/tmp/shake_${TAG}_${pct}pct.txt"
    head -c "${bytes}" "${CORPUS}" > "${trunc_corpus}"
    trie_dir="/tmp/shake_${TAG}_${pct}pct_d${D}_radix"
    if [ ! -d "${trie_dir}" ]; then
      "${PROJ}/bin/agpt_build_radix_corpus" \
        --corpus "${trunc_corpus}" \
        --vocab-file "${CORPUS}" \
        --max-depth "${D}" \
        --out "${trie_dir}" \
        > "${LOGS}/seed${SEED}_${TAG}_build_${pct}.log" 2>&1
    fi

    out_model="${MODELS_DIR}/seed${SEED}_${TAG}_stage${i}.model"
    cp "${prev_model}" "${out_model}"
    "${PROJ}/bin/agpt_train" \
      --model "${out_model}" \
      --trie-dir "${trie_dir}" \
      --save "${out_model}" \
      --epochs "${PER_STAGE_SE}" \
      --total-epochs-budget "${SE_BUDGET}" \
      --partition-depth "${PD}" --no-accumulate \
      --lr "${LR}" --lr-schedule warmup-cosine --warmup-epochs 1 \
      --optimizer rmsprop --rmsprop-beta 0.999 \
      --mass-weight log --entropy-lambda 1.0 \
      > "${LOGS}/seed${SEED}_${TAG}_stage${i}_train.log" 2>&1
    # cleanup previous stage's trie + truncated corpus
    if [ -n "${prev_trie}" ] && [ "${prev_trie}" != "${trie_dir}" ]; then
      rm -rf "${prev_trie}"
    fi
    prev_model="${out_model}"
    prev_trie="${trie_dir}"
  done
  s_end=$(date +%s)
  wall=$((s_end - s_start))
  ppl_log="${LOGS}/seed${SEED}_${TAG}_ppl.log"
  "${PROJ}/bin/perplexity" \
    --model "${prev_model}" \
    --file "${CORPUS}" \
    --seq-len "${D}" \
    --backend openblas \
    --max-positions 4096 \
    > "${ppl_log}" 2>&1
  ppl=$(grep "^Perplexity:" "${ppl_log}" | awk '{print $2}')
  PPL[$SEED]="$ppl"
  WALL[$SEED]="$wall"
  # Clean last trie (no later stage will use it)
  rm -rf "${prev_trie}"
  echo "  seed=${SEED}: PPL=${ppl}, wall=${wall}s"
  echo ""
done

mean_std() {
  printf "%s\n" "$@" | awk '
    { n++; s += $1; ss += $1*$1 }
    END {
      m = s / n; v = (ss / n) - (m * m); sd = sqrt(v < 0 ? 0 : v)
      printf "%.4f ± %.4f", m, sd
    }
  '
}

vals=()
walls=()
for SEED in "${SEEDS[@]}"; do
  vals+=("${PPL[$SEED]}")
  walls+=("${WALL[$SEED]}")
done

echo "===== ${N_STAGES} × ${PER_STAGE_SE} SE multi-seed summary ====="
echo "  per-seed PPL: ${vals[*]}"
echo "  PPL:  $(mean_std "${vals[@]}")"
echo "  wall: $(mean_std "${walls[@]}") s"
