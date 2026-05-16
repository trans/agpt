#!/usr/bin/env bash
# Matched 500 SE baseline for comparison against streaming 50 × 10 SE.
# Same trie (full Shakespeare 1M d=16), same recipe, same total SE budget.

set -euo pipefail

PROJ="${PROJ:-$(pwd)}"
CORPUS="${PROJ}/data/input.txt"
RND="${PROJ}/rnd/streaming-agpt-v1"
LOGS="${RND}/logs"
MODELS_DIR="${RND}/models"

LR=3e-3
PD=1
SE=500
D=16
INIT_MODEL="${PROJ}/data/input.random.model"

mkdir -p "${LOGS}" "${MODELS_DIR}"

# Build full-corpus trie (rebuilt for cleanliness)
baseline_trie="/tmp/shake_baseline_d${D}_radix"
rm -rf "${baseline_trie}"
"${PROJ}/bin/agpt_build_radix_corpus" \
  --corpus "${CORPUS}" \
  --vocab-file "${CORPUS}" \
  --max-depth "${D}" \
  --out "${baseline_trie}" \
  > "${LOGS}/baseline_500_build.log" 2>&1

# Train
baseline_model="${MODELS_DIR}/baseline_500.model"
cp "${INIT_MODEL}" "${baseline_model}"

start=$(date +%s)
"${PROJ}/bin/agpt_train" \
  --model "${baseline_model}" \
  --trie-dir "${baseline_trie}" \
  --save "${baseline_model}" \
  --epochs "${SE}" \
  --partition-depth "${PD}" --no-accumulate \
  --lr "${LR}" --lr-schedule warmup-cosine --warmup-epochs 1 \
  --optimizer rmsprop --rmsprop-beta 0.999 \
  --mass-weight log --entropy-lambda 1.0 \
  > "${LOGS}/baseline_500_train.log" 2>&1
end=$(date +%s)
wall=$((end - start))

# Eval
"${PROJ}/bin/perplexity" \
  --model "${baseline_model}" \
  --file "${CORPUS}" \
  --seq-len "${D}" \
  --backend openblas \
  --max-positions 4096 \
  > "${LOGS}/baseline_500_ppl.log" 2>&1
ppl=$(grep "^Perplexity:" "${LOGS}/baseline_500_ppl.log" | awk '{print $2}')

echo "Baseline ${SE} SE: PPL@${D}=${ppl}, wall=${wall}s"
echo ""
echo "For comparison:"
echo "  Streaming 50 × 10 SE (500 SE): PPL=3.996, wall=1607s"
