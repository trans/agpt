#!/usr/bin/env bash
# Streaming AGPT v1: 5-checkpoint linear cadence on Shakespeare d=16.
#
# Compares:
#   STREAMING: build trie at 20/40/60/80/100% corpus, train 20 SE each,
#              chaining model state across stages
#   BASELINE:  build trie at 100%, train 100 SE in one run
#
# See README.md for design details and confounds. Run from project root.

set -euo pipefail

PROJ="${PROJ:-$(pwd)}"
CORPUS="${PROJ}/data/input.txt"
RND="${PROJ}/rnd/streaming-agpt-v1"
LOGS="${RND}/logs"
MODELS_DIR="${RND}/models"

# Recipe. With optimizer-state persistence (commit 6a655e7) and the global
# LR-schedule horizon override (this commit), we can now run warmup-cosine
# across the streaming sequence as a single coherent schedule.
LR=3e-3
PD=1
PER_STAGE_SE=20
BASELINE_SE=100
TOTAL_BUDGET_SE=100   # = N_STAGES × PER_STAGE_SE; used by --total-epochs-budget
D=16
INIT_MODEL="${PROJ}/data/input.random.model"  # random-init checkpoint

CHECKPOINTS=(20 40 60 80 100)

mkdir -p "${LOGS}" "${MODELS_DIR}"

# Sanity: make sure binaries exist
for bin in agpt_build_radix_corpus agpt_train perplexity; do
  [ -x "${PROJ}/bin/${bin}" ] || { echo "ERROR: ${PROJ}/bin/${bin} missing. Run 'just build-all' and 'just build-microgpt-tools' first."; exit 1; }
done

# Sanity: input model
[ -f "${INIT_MODEL}" ] || { echo "ERROR: ${INIT_MODEL} missing (random-init checkpoint)"; exit 1; }

# Corpus byte count for truncation
CORPUS_BYTES=$(wc -c < "${CORPUS}")
echo "Corpus: ${CORPUS} (${CORPUS_BYTES} bytes)"

# ===== STREAMING =====
echo ""
echo "===== STREAMING (5 checkpoints × ${PER_STAGE_SE} SE each = $((${#CHECKPOINTS[@]} * PER_STAGE_SE)) SE total) ====="

prev_model="${INIT_MODEL}"
streaming_start=$(date +%s)

for pct in "${CHECKPOINTS[@]}"; do
  echo ""
  echo "--- Stage ${pct}% ---"

  # Truncate corpus
  bytes=$((CORPUS_BYTES * pct / 100))
  trunc_corpus="/tmp/shake_${pct}pct.txt"
  head -c "${bytes}" "${CORPUS}" > "${trunc_corpus}"

  # Build trie. --vocab-file points to FULL corpus so vocab is consistent
  # across stages (truncated corpus may lack some chars).
  trie_dir="/tmp/shake_${pct}pct_d${D}_radix"
  rm -rf "${trie_dir}"
  build_log="${LOGS}/streaming_stage${pct}_build.log"

  "${PROJ}/bin/agpt_build_radix_corpus" \
    --corpus "${trunc_corpus}" \
    --vocab-file "${CORPUS}" \
    --max-depth "${D}" \
    --out "${trie_dir}" \
    > "${build_log}" 2>&1

  # Train this stage
  out_model="${MODELS_DIR}/streaming_stage${pct}.model"
  cp "${prev_model}" "${out_model}"  # agpt_train writes back to --save path

  train_log="${LOGS}/streaming_stage${pct}_train.log"

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
    > "${train_log}" 2>&1

  # Eval this intermediate
  ppl_log="${LOGS}/streaming_stage${pct}_ppl.log"
  "${PROJ}/bin/perplexity" \
    --model "${out_model}" \
    --file "${CORPUS}" \
    --seq-len "${D}" \
    --backend openblas \
    --max-positions 4096 \
    > "${ppl_log}" 2>&1

  final_loss=$(grep -E "^Epoch [0-9]+: loss" "${train_log}" | tail -1 | grep -oE 'loss=[0-9.]+' | cut -d= -f2)
  ppl=$(grep "^Perplexity:" "${ppl_log}" | awk '{print $2}')
  echo "  Stage ${pct}%: final loss=${final_loss} PPL@${D}=${ppl}"

  prev_model="${out_model}"
done

streaming_end=$(date +%s)
streaming_wall=$((streaming_end - streaming_start))
echo ""
echo "STREAMING total wall: ${streaming_wall}s"
final_streaming_ppl=$(grep "^Perplexity:" "${LOGS}/streaming_stage100_ppl.log" | awk '{print $2}')

# ===== BASELINE =====
echo ""
echo "===== BASELINE (full corpus, ${BASELINE_SE} SE single run) ====="

baseline_start=$(date +%s)

# Build full-corpus trie (same dimension as stage 100% but rebuilt for cleanliness)
baseline_trie="/tmp/shake_baseline_d${D}_radix"
rm -rf "${baseline_trie}"
"${PROJ}/bin/agpt_build_radix_corpus" \
  --corpus "${CORPUS}" \
  --max-depth "${D}" \
  --out "${baseline_trie}" \
  > "${LOGS}/baseline_build.log" 2>&1

baseline_model="${MODELS_DIR}/baseline.model"
cp "${INIT_MODEL}" "${baseline_model}"

"${PROJ}/bin/agpt_train" \
  --model "${baseline_model}" \
  --trie-dir "${baseline_trie}" \
  --save "${baseline_model}" \
  --epochs "${BASELINE_SE}" \
  --partition-depth "${PD}" --no-accumulate \
  --lr "${LR}" --lr-schedule warmup-cosine --warmup-epochs 1 \
  --optimizer rmsprop --rmsprop-beta 0.999 \
  --mass-weight log --entropy-lambda 1.0 \
  > "${LOGS}/baseline_train.log" 2>&1

"${PROJ}/bin/perplexity" \
  --model "${baseline_model}" \
  --file "${CORPUS}" \
  --seq-len "${D}" \
  --backend openblas \
  --max-positions 4096 \
  > "${LOGS}/baseline_ppl.log" 2>&1

baseline_end=$(date +%s)
baseline_wall=$((baseline_end - baseline_start))
baseline_ppl=$(grep "^Perplexity:" "${LOGS}/baseline_ppl.log" | awk '{print $2}')

# ===== SUMMARY =====
echo ""
echo "===== SUMMARY ====="
echo ""
printf "%-30s %-10s %-10s\n" "variant" "PPL@${D}" "wall (s)"
printf "%-30s %-10s %-10s\n" "------" "-----" "----"
printf "%-30s %-10s %-10s\n" "streaming 5 × ${PER_STAGE_SE} SE" "${final_streaming_ppl}" "${streaming_wall}"
printf "%-30s %-10s %-10s\n" "baseline ${BASELINE_SE} SE" "${baseline_ppl}" "${baseline_wall}"
echo ""
echo "Per-stage streaming trajectory:"
for pct in "${CHECKPOINTS[@]}"; do
  ppl=$(grep "^Perplexity:" "${LOGS}/streaming_stage${pct}_ppl.log" | awk '{print $2}')
  printf "  stage %3d%%  PPL=%s\n" "${pct}" "${ppl}"
done
