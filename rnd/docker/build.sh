#!/usr/bin/env bash
# Build the AGPT Docker image locally with podman.
# Run from the project root.

set -euo pipefail

cd "$(dirname "$0")/../.."

TAG="${TAG:-agpt:latest}"

echo "Building ${TAG} from $(pwd)..."

podman build \
    -t "${TAG}" \
    -f rnd/docker/Dockerfile \
    --ignorefile rnd/docker/.dockerignore \
    .

echo ""
echo "Built ${TAG}."
echo "Image size:"
podman images "${TAG}" --format "{{.Size}}"
echo ""
echo "Quick smoke test:"
echo "  podman run --rm ${TAG} /opt/cuda/bin/nvcc --version"
echo "  podman run --rm ${TAG} bin/agpt_train  # should print usage"
echo ""
echo "Interactive shell:"
echo "  podman run --rm -it ${TAG}"
