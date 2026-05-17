#!/usr/bin/env bash
# Multi-seed for 100 × 5 SE cadence comparison.
# Same total 500 SE budget as 50 × 10 and baseline 500.
# Reuses pre-built tries from run_multiseed.sh cache.

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
N_STAGES=100
PER_STAGE_SE=$((SE_BUDGET / N_STAGES))   # 5

SEEDS=(100 200 300)

mkdir -p "${LOGS}" "${MODELS_DIR}"
CORPUS_BYTES=$(wc -c < "${CORPUS}")

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
    trunc_corpus="/tmp/shake_ms100_${pct}pct.txt"
    head -c "${bytes}" "${CORPUS}" > "${trunc_corpus}"
    trie_dir="/tmp/shake_ms100_${pct}pct_d${D}_radix"
    # Build trie just-in-time (avoid pre-allocating all 100 tries → /tmp full)
    if [ ! -d "${trie_dir}" ]; then
      "${PROJ}/bin/agpt_build_radix_corpus" \
        --corpus "${trunc_corpus}" \
        --vocab-file "${CORPUS}" \
        --max-depth "${D}" \
        --out "${trie_dir}" \
        > "${LOGS}/seed${SEED}_100x5_build_${pct}.log" 2>&1
    fi

    out_model="${MODELS_DIR}/seed${SEED}_stream100x5_stage${i}.model"
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
      > "${LOGS}/seed${SEED}_stream100x5_stage${i}_train.log" 2>&1
    # Clean up previous stage's trie and truncated corpus to bound /tmp use
    if [ -n "${prev_trie}" ] && [ "${prev_trie}" != "${trie_dir}" ]; then
      rm -rf "${prev_trie}"
    fi
    prev_model="${out_model}"
    prev_trie="${trie_dir}"
  done
  s_end=$(date +%s)
  wall=$((s_end - s_start))
  ppl_log="${LOGS}/seed${SEED}_stream100x5_ppl.log"
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
  echo "  seed=${SEED}: PPL=${ppl}, wall=${wall}s"
  echo ""
done

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

echo "===== 100 × 5 SE multi-seed summary ====="
vals=("${PPL[100]}" "${PPL[200]}" "${PPL[300]}")
walls=("${WALL[100]}" "${WALL[200]}" "${WALL[300]}")
echo "  per-seed PPL: ${vals[*]}"
echo "  PPL:  $(mean_std "${vals[@]}")"
echo "  wall: $(mean_std "${walls[@]}") s"
echo ""
echo "For comparison:"
echo "  Streaming 50 × 10 SE: 4.2283 ± 0.0951 PPL, 1571 ± 7 s"
echo "  Baseline 500 SE:      4.2651 ± 0.0181 PPL, 2899 ± 17 s"
