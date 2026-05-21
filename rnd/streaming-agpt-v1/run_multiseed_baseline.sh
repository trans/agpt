#!/usr/bin/env bash
# Generic multi-seed BASELINE runner (single-call training on full-corpus trie).
# Args: SE_BUDGET [SEEDS...]

set -euo pipefail

PROJ="${PROJ:-$(pwd)}"
CORPUS="${CORPUS:-${PROJ}/data/input.txt}"
RND="${PROJ}/rnd/streaming-agpt-v1"
LOGS="${RND}/logs"
MODELS_DIR="${RND}/models"

if [ $# -lt 1 ]; then
  echo "Usage: $0 SE_BUDGET [SEEDS...]"
  exit 1
fi

SE_BUDGET=$1
shift
if [ $# -ge 1 ]; then
  SEEDS=("$@")
else
  SEEDS=(100 200 300)
fi

LR=3e-3
PD=1
D=16
CORPUS_BASE=$(basename "${CORPUS}" .txt)
TAG="ms_baseline_${CORPUS_BASE}_se${SE_BUDGET}"

mkdir -p "${LOGS}" "${MODELS_DIR}"

# Build full-corpus trie (once, reused across seeds)
baseline_trie="/tmp/${CORPUS_BASE}_baseline_d${D}_radix"
if [ ! -d "${baseline_trie}" ]; then
  echo "Building baseline trie: ${baseline_trie}"
  "${PROJ}/bin/agpt_build_radix_corpus" \
    --corpus "${CORPUS}" \
    --vocab-file "${CORPUS}" \
    --max-depth "${D}" \
    --out "${baseline_trie}" \
    > "${LOGS}/${TAG}_build.log" 2>&1
fi

echo "Baseline ${SE_BUDGET} SE on ${CORPUS}"
echo "Seeds: ${SEEDS[*]}"
echo ""

declare -A PPL WALL
for SEED in "${SEEDS[@]}"; do
  INIT="/tmp/init_seed${SEED}.model"
  [ -f "${INIT}" ] || { echo "Missing: ${INIT}"; exit 1; }

  echo "===== SEED ${SEED} — Baseline ${SE_BUDGET} SE ====="
  out_model="${MODELS_DIR}/${TAG}_seed${SEED}.model"
  cp "${INIT}" "${out_model}"
  start=$(date +%s)
  "${PROJ}/bin/agpt_train" \
    --model "${out_model}" \
    --trie-dir "${baseline_trie}" \
    --save "${out_model}" \
    --epochs "${SE_BUDGET}" \
    --partition-depth "${PD}" --no-accumulate \
    --lr "${LR}" --lr-schedule warmup-cosine --warmup-epochs 1 \
    --optimizer rmsprop --rmsprop-beta 0.999 \
    --mass-weight log --entropy-lambda 1.0 \
    > "${LOGS}/${TAG}_seed${SEED}_train.log" 2>&1
  end=$(date +%s)
  wall=$((end - start))
  "${PROJ}/bin/perplexity" \
    --model "${out_model}" \
    --file "${CORPUS}" \
    --seq-len "${D}" \
    --backend openblas \
    --max-positions 4096 \
    > "${LOGS}/${TAG}_seed${SEED}_ppl.log" 2>&1
  ppl=$(grep "^Perplexity:" "${LOGS}/${TAG}_seed${SEED}_ppl.log" | awk '{print $2}')
  PPL[$SEED]="$ppl"
  WALL[$SEED]="$wall"
  echo "  seed=${SEED}: PPL=${ppl}, wall=${wall}s"
  echo ""
done

mean_std() {
  printf "%s\n" "$@" | awk '
    { n++; s += $1; ss += $1*$1 }
    END { m = s / n; v = (ss / n) - (m * m); sd = sqrt(v < 0 ? 0 : v); printf "%.4f ± %.4f", m, sd }
  '
}

vals=()
walls=()
for SEED in "${SEEDS[@]}"; do
  vals+=("${PPL[$SEED]}")
  walls+=("${WALL[$SEED]}")
done

echo "===== Baseline ${SE_BUDGET} SE summary on ${CORPUS} ====="
echo "  per-seed PPL: ${vals[*]}"
echo "  PPL:  $(mean_std "${vals[@]}")"
echo "  wall: $(mean_std "${walls[@]}") s"
