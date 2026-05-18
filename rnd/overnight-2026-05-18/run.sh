#!/usr/bin/env bash
# Overnight 2026-05-17/18: validate streaming-AGPT edge on Gutenberg.
#
# Phase 1 (~5hr): streaming-AGPT × Gutenberg × 3 seeds at 100×5 SE
# Phase 2 (~3hr): one more baseline seed (seed 100) to round out the
#                  Gutenberg baseline to 3 seeds.
#
# Existing baseline seeds 200 and 300 from RunPod 2026-05-17 give us
# the rest. Result: 3-vs-3 Welch's t-test on streaming-vs-baseline @ Gutenberg.

set -e

cd "$(dirname "$0")/../.."

LOG="rnd/overnight-2026-05-18/run.log"
exec > >(tee -a "${LOG}") 2>&1

echo "=========================================="
echo "Overnight $(date -Iseconds): streaming vs baseline @ Gutenberg"
echo "=========================================="

echo ""
echo "--- Phase 1/2: streaming-AGPT × Gutenberg × 3 seeds @ 100×5 SE ---"
echo "Started: $(date -Iseconds)"
CORPUS="$PWD/data/gutenberg_5m.txt" \
    bash rnd/streaming-agpt-v1/run_multiseed_generic.sh 100 5 100 200 300

echo ""
echo "--- Phase 2/2: extra Gutenberg baseline seed (100) @ 500 SE ---"
echo "Started: $(date -Iseconds)"
CORPUS="$PWD/data/gutenberg_5m.txt" \
    bash rnd/streaming-agpt-v1/run_multiseed_baseline.sh 500 100

echo ""
echo "=========================================="
echo "Done: $(date -Iseconds)"
echo "=========================================="
echo ""
echo "Results to compare:"
echo "  Streaming:  rnd/streaming-agpt-v1/logs/ms_n100_se5_*"
echo "  Baseline:   rnd/streaming-agpt-v1/logs/ms_baseline_gutenberg_5m_se500_*"
