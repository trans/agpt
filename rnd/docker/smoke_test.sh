#!/usr/bin/env bash
# Verify the built agpt:latest image. Checks binaries exist and print
# their usage messages. Does NOT exercise the GPU — that requires
# nvidia-container-toolkit + CDI setup which is environment-specific.

set -euo pipefail

TAG="${TAG:-agpt:latest}"

if ! podman image exists "${TAG}"; then
    echo "ERROR: ${TAG} not found. Run rnd/docker/build.sh first."
    exit 1
fi

echo "=========================================="
echo "Smoke testing ${TAG}"
echo "=========================================="

run() { echo ""; echo "+ $*"; podman run --rm "${TAG}" "$@"; }

run /opt/cuda/bin/nvcc --version
run crystal --version
run just --version
run bash -c 'ls -la bin/'
run bin/agpt_train --help 2>&1 | head -10 || true
run bash -c 'ldd bin/agpt_train | grep -E "cuda|cublas" || true'

echo ""
echo "=========================================="
echo "Image size: $(podman images "${TAG}" --format '{{.Size}}')"
echo "=========================================="
