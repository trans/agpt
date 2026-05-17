#!/usr/bin/env bash
# Setup script to run ON the RunPod instance after first ssh.
# Installs Crystal + just + builds AGPT binaries. ~10-15 min first run.
#
# Assumes:
#  - You're in /workspace/agpt (or wherever the code was rsync'd to)
#  - The pod has a CUDA 12+ image (pre-installed CUDA + cuBLAS)
#  - data/input.txt + data/gutenberg_5m.txt already rsync'd in
#  - /tmp/init_seed*.model already rsync'd in
#
# Usage (on pod):
#   cd /workspace/agpt && bash rnd/runpod/setup_pod.sh

set -euo pipefail

PROJ="${PROJ:-$(pwd)}"

echo "=========================================="
echo "AGPT pod setup"
echo "=========================================="
echo "PROJ=${PROJ}"
echo ""

# 1. System deps
echo "--- 1/5 Installing system deps ---"
apt-get update -qq
apt-get install -y -qq build-essential cmake git curl pkg-config \
    libssl-dev libbsd-dev libpcre2-dev libevent-dev libgmp-dev \
    libz-dev libxml2-dev libyaml-dev libreadline-dev libffi-dev || true

# 2. Crystal compiler
echo "--- 2/5 Installing Crystal compiler ---"
if ! command -v crystal &>/dev/null; then
    curl -fsSL https://crystal-lang.org/install.sh | bash
fi
crystal --version

# 3. just (build runner)
echo "--- 3/5 Installing just ---"
if ! command -v just &>/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin
fi
just --version

# 4. Shards (Crystal package manager) — fetches microgpt
echo "--- 4/5 Fetching microgpt shard ---"
cd "${PROJ}"
shards install

# Apply local microgpt TF32 patch (the lib/ vendored copy doesn't have our
# local TF32 enable; reapply it here so cuBLAS path uses tensor cores).
if [ -f rnd/runpod/microgpt_tf32.patch ]; then
    echo "  Applying microgpt TF32 patch..."
    (cd lib/microgpt && patch -p1 < "${PROJ}/rnd/runpod/microgpt_tf32.patch") || \
        echo "  (patch may have already been applied; continuing)"
fi

# 5. Build binaries
echo "--- 5/5 Building binaries ---"
just build-agpt-train
just build-agpt-build-radix-corpus
just build-microgpt-tools

# Quick smoke test
echo ""
echo "=========================================="
echo "Smoke test: train 1 SE Shakespeare d=8"
echo "=========================================="
if [ ! -d /tmp/shake_d8_test_radix ]; then
    bin/agpt_build_radix_corpus --corpus data/input.txt --max-depth 8 --out /tmp/shake_d8_test_radix
fi
cp data/input.random.model /tmp/smoke.model || cp /tmp/init_seed100.model /tmp/smoke.model
bin/agpt_train --model /tmp/smoke.model --trie-dir /tmp/shake_d8_test_radix \
    --save /tmp/smoke.model --epochs 1 --partition-depth 1 --no-accumulate \
    --lr 3e-3 --optimizer rmsprop --mass-weight log

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo "Run experiments from this directory, e.g.:"
echo "  CORPUS=\$PWD/data/gutenberg_5m.txt bash rnd/streaming-agpt-v1/run_multiseed_baseline.sh 500 200 300"
