#!/usr/bin/env bash
# Multi-seed runs of the headline streaming vs baseline comparison at 500 SE.
# 3 seeds each → mean ± std for each variant.
#
# Uses pre-generated /tmp/init_seed{100,200,300}.model as init checkpoints.

set -euo pipefail

PROJ="${PROJ:-$(pwd)}"
CORPUS="${PROJ}/data/input.txt"
RND="${PROJ}/rnd/streaming-agpt-v1"
LOGS="${RND}/logs"
MODELS_DIR="${RND}/models"

LR=3e-3
PD=1
D=16
SE_BUDGET=500
N_STAGES=50
PER_STAGE_SE=$((SE_BUDGET / N_STAGES))   # 10

SEEDS=(100 200 300)

mkdir -p "${LOGS}" "${MODELS_DIR}"
CORPUS_BYTES=$(wc -c < "${CORPUS}")

# Build full-corpus trie once (reused across all baselines and streaming final stages)
baseline_trie="/tmp/shake_multiseed_d${D}_radix"
if [ ! -d "${baseline_trie}" ]; then
  "${PROJ}/bin/agpt_build_radix_corpus" \
    --corpus "${CORPUS}" \
    --vocab-file "${CORPUS}" \
    --max-depth "${D}" \
    --out "${baseline_trie}" \
    > "${LOGS}/multiseed_baseline_build.log" 2>&1
fi

# Streaming intermediate tries (reused across seeds since they only depend on % of corpus)
echo "Building streaming trie cache (one-time)..."
for ((i = 1; i <= N_STAGES; i++)); do
  pct=$((100 * i / N_STAGES))
  bytes=$((CORPUS_BYTES * pct / 100))
  trunc_corpus="/tmp/shake_ms_${pct}pct.txt"
  head -c "${bytes}" "${CORPUS}" > "${trunc_corpus}"
  trie_dir="/tmp/shake_ms_${pct}pct_d${D}_radix"
  if [ ! -d "${trie_dir}" ]; then
    "${PROJ}/bin/agpt_build_radix_corpus" \
      --corpus "${trunc_corpus}" \
      --vocab-file "${CORPUS}" \
      --max-depth "${D}" \
      --out "${trie_dir}" \
      > "${LOGS}/multiseed_stream_build_${pct}.log" 2>&1
  fi
done
echo "  Done building trie cache."
echo ""

declare -A STREAM_PPL STREAM_WALL BASE_PPL BASE_WALL

for SEED in "${SEEDS[@]}"; do
  INIT="/tmp/init_seed${SEED}.model"
  [ -f "${INIT}" ] || { echo "Missing init model ${INIT}"; exit 1; }

  echo "===== SEED ${SEED} — Streaming 50 × 10 SE ====="
  prev_model="${INIT}"
  s_start=$(date +%s)
  for ((i = 1; i <= N_STAGES; i++)); do
    pct=$((100 * i / N_STAGES))
    trie_dir="/tmp/shake_ms_${pct}pct_d${D}_radix"
    out_model="${MODELS_DIR}/seed${SEED}_stream_stage${i}.model"
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
      > "${LOGS}/seed${SEED}_stream_stage${i}_train.log" 2>&1
    prev_model="${out_model}"
  done
  s_end=$(date +%s)
  s_wall=$((s_end - s_start))
  ppl_log="${LOGS}/seed${SEED}_stream_ppl.log"
  "${PROJ}/bin/perplexity" \
    --model "${prev_model}" \
    --file "${CORPUS}" \
    --seq-len "${D}" \
    --backend openblas \
    --max-positions 4096 \
    > "${ppl_log}" 2>&1
  ppl=$(grep "^Perplexity:" "${ppl_log}" | awk '{print $2}')
  STREAM_PPL[$SEED]="$ppl"
  STREAM_WALL[$SEED]="$s_wall"
  echo "  seed=${SEED}: PPL=${ppl}, wall=${s_wall}s"
  echo ""

  echo "===== SEED ${SEED} — Baseline 500 SE ====="
  base_model="${MODELS_DIR}/seed${SEED}_baseline.model"
  cp "${INIT}" "${base_model}"
  b_start=$(date +%s)
  "${PROJ}/bin/agpt_train" \
    --model "${base_model}" \
    --trie-dir "${baseline_trie}" \
    --save "${base_model}" \
    --epochs "${SE_BUDGET}" \
    --partition-depth "${PD}" --no-accumulate \
    --lr "${LR}" --lr-schedule warmup-cosine --warmup-epochs 1 \
    --optimizer rmsprop --rmsprop-beta 0.999 \
    --mass-weight log --entropy-lambda 1.0 \
    > "${LOGS}/seed${SEED}_baseline_train.log" 2>&1
  b_end=$(date +%s)
  b_wall=$((b_end - b_start))
  ppl_log="${LOGS}/seed${SEED}_baseline_ppl.log"
  "${PROJ}/bin/perplexity" \
    --model "${base_model}" \
    --file "${CORPUS}" \
    --seq-len "${D}" \
    --backend openblas \
    --max-positions 4096 \
    > "${ppl_log}" 2>&1
  ppl=$(grep "^Perplexity:" "${ppl_log}" | awk '{print $2}')
  BASE_PPL[$SEED]="$ppl"
  BASE_WALL[$SEED]="$b_wall"
  echo "  seed=${SEED}: PPL=${ppl}, wall=${b_wall}s"
  echo ""
done

# Summary with mean ± std (via awk)
mean_std() {
  printf "%s\n" "$@" | awk '
    { n++; s += $1; ss += $1*$1 }
    END {
      m = s / n
      v = (ss / n) - (m * m)
      sd = sqrt(v < 0 ? 0 : v)
      printf "%.4f ± %.4f", m, sd
    }
  '
}

echo "===== Multi-seed summary (3 seeds each) ====="
stream_vals=("${STREAM_PPL[100]}" "${STREAM_PPL[200]}" "${STREAM_PPL[300]}")
base_vals=("${BASE_PPL[100]}" "${BASE_PPL[200]}" "${BASE_PPL[300]}")
stream_walls=("${STREAM_WALL[100]}" "${STREAM_WALL[200]}" "${STREAM_WALL[300]}")
base_walls=("${BASE_WALL[100]}" "${BASE_WALL[200]}" "${BASE_WALL[300]}")

echo "Streaming 50 × 10 SE:"
echo "  per-seed: ${stream_vals[*]}"
echo "  PPL:  $(mean_std "${stream_vals[@]}")"
echo "  wall: $(mean_std "${stream_walls[@]}") s"
echo ""
echo "Baseline 500 SE:"
echo "  per-seed: ${base_vals[*]}"
echo "  PPL:  $(mean_std "${base_vals[@]}")"
echo "  wall: $(mean_std "${base_walls[@]}") s"
