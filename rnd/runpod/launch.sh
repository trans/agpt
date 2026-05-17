#!/usr/bin/env bash
# Laptop-side launcher for RunPod sessions. Handles:
#   - Push code (rsync excluding /tmp, models, logs)
#   - Push corpora + init models (the gitignored runtime data)
#   - Trigger setup + run on pod via ssh
#   - Pull results back when done
#
# Usage:
#   bash rnd/runpod/launch.sh setup POD_USER@POD_IP[:PORT]
#       # First-time setup on a fresh pod: rsync code + data, run setup_pod.sh
#
#   bash rnd/runpod/launch.sh run POD_USER@POD_IP[:PORT] 'COMMAND'
#       # Run a command on the pod (ssh wrapper that ensures cwd is /workspace/agpt)
#
#   bash rnd/runpod/launch.sh pull POD_USER@POD_IP[:PORT]
#       # Pull logs and findings back to laptop. Run when experiments are done
#       # but BEFORE stopping the pod.
#
#   bash rnd/runpod/launch.sh full POD_USER@POD_IP[:PORT] 'COMMAND'
#       # All-in-one: setup, run, pull.
#
# Notes:
#   - Pod target format examples:
#       root@69.30.85.92                  (default port 22)
#       root@69.30.85.92:18234            (RunPod typically uses custom port)
#   - RunPod normally provides:
#       ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519
#     Use the colon form above and the script handles it.

set -euo pipefail

PROJ="${PROJ:-$(pwd)}"
REMOTE_BASE="${REMOTE_BASE:-/workspace/agpt}"

usage() { sed -n '4,30p' "$0"; exit 1; }
[ $# -lt 2 ] && usage

CMD=$1
POD=$2
shift 2

# Parse user@host[:port] → SSH_HOST + SSH_PORT_ARG
if [[ "${POD}" == *:* ]]; then
    SSH_HOST="${POD%:*}"
    SSH_PORT_ARG="-p ${POD##*:}"
    RSYNC_RSH="ssh -p ${POD##*:}"
else
    SSH_HOST="${POD}"
    SSH_PORT_ARG=""
    RSYNC_RSH="ssh"
fi

# --- helper: rsync code (no /tmp, no models, no logs, no build artifacts) ---
push_code() {
    echo "→ rsync code to pod..."
    rsync -avzP --delete \
        --exclude='/tmp/' \
        --exclude='bin/' \
        --exclude='build/' \
        --exclude='lib/' \
        --exclude='.shards/' \
        --exclude='rnd/streaming-agpt-v1/models/' \
        --exclude='rnd/streaming-agpt-v1/logs/' \
        --exclude='rnd/seq-len-decouple/*.bin' \
        --exclude='data/gutenberg_5m.txt' \
        --exclude='data/*.model' \
        --exclude='data/wormhole_*.txt' \
        --exclude='*.dwarf' \
        --exclude='.claude/' \
        --exclude='notes/agpt/paper.html' \
        --rsh="${RSYNC_RSH}" \
        "${PROJ}/" "${SSH_HOST}:${REMOTE_BASE}/"
}

# --- helper: push runtime data (corpora + init models, gitignored) ---
push_data() {
    echo "→ rsync corpora + init models..."
    # Corpora
    rsync -avzP --rsh="${RSYNC_RSH}" \
        "${PROJ}/data/input.txt" \
        "${PROJ}/data/gutenberg_5m.txt" \
        "${SSH_HOST}:${REMOTE_BASE}/data/"
    # Init models (random + seeded)
    rsync -avzP --rsh="${RSYNC_RSH}" \
        "${PROJ}/data/input.random.model" \
        "${SSH_HOST}:${REMOTE_BASE}/data/" 2>/dev/null || true
    # Seeded init models (in /tmp on laptop)
    if ls /tmp/init_seed*.model &>/dev/null; then
        rsync -avzP --rsh="${RSYNC_RSH}" \
            /tmp/init_seed*.model "${SSH_HOST}:/tmp/"
    fi
}

# --- helper: run command on pod ---
pod_run() {
    local cmd=$1
    echo "→ ssh: cd ${REMOTE_BASE} && ${cmd}"
    ssh ${SSH_PORT_ARG} "${SSH_HOST}" "cd ${REMOTE_BASE} && ${cmd}"
}

# --- helper: pull results ---
pull_results() {
    echo "→ rsync logs back to laptop..."
    rsync -avzP --rsh="${RSYNC_RSH}" \
        "${SSH_HOST}:${REMOTE_BASE}/rnd/streaming-agpt-v1/logs/" \
        "${PROJ}/rnd/streaming-agpt-v1/logs/"
    # Also pull any findings.md updates
    rsync -avzP --rsh="${RSYNC_RSH}" \
        "${SSH_HOST}:${REMOTE_BASE}/rnd/streaming-agpt-v1/findings.md" \
        "${PROJ}/rnd/streaming-agpt-v1/findings.md" 2>/dev/null || true
}

case "${CMD}" in
    setup)
        push_code
        push_data
        pod_run "bash rnd/runpod/setup_pod.sh"
        echo ""
        echo "✓ Pod setup complete. Next: bash rnd/runpod/launch.sh run ${POD} 'YOUR EXPERIMENT'"
        ;;
    run)
        [ $# -lt 1 ] && { echo "Usage: launch.sh run POD 'COMMAND'"; exit 1; }
        pod_run "$*"
        ;;
    pull)
        pull_results
        ;;
    full)
        [ $# -lt 1 ] && { echo "Usage: launch.sh full POD 'COMMAND'"; exit 1; }
        push_code
        push_data
        pod_run "bash rnd/runpod/setup_pod.sh && $*"
        pull_results
        ;;
    *)
        usage
        ;;
esac
