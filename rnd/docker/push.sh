#!/usr/bin/env bash
# Push the locally-built agpt:latest image to Docker Hub.
#
# Prerequisites:
#   1. Docker Hub account at hub.docker.com
#   2. podman login docker.io (will prompt for username and password
#      or access token)
#
# Usage:
#   bash rnd/docker/push.sh [USERNAME]
#
# Default username: 7rans.

set -euo pipefail

USERNAME="${1:-7rans}"
LOCAL_TAG="${LOCAL_TAG:-agpt:latest}"
REMOTE_TAG="docker.io/${USERNAME}/agpt:latest"

if ! podman image exists "${LOCAL_TAG}"; then
    echo "ERROR: ${LOCAL_TAG} not found. Run rnd/docker/build.sh first."
    exit 1
fi

echo "Tagging ${LOCAL_TAG} -> ${REMOTE_TAG}"
podman tag "${LOCAL_TAG}" "${REMOTE_TAG}"

echo "Pushing ${REMOTE_TAG} (this is ~9.6 GB; on a typical home WiFi"
echo "upload, expect ~15-25 minutes)..."
echo ""
podman push "${REMOTE_TAG}"

echo ""
echo "Pushed ${REMOTE_TAG}."
echo "RunPod can now reference this image as:"
echo "  ${REMOTE_TAG}"
