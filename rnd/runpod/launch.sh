#!/usr/bin/env bash
# Laptop-side launcher for RunPod sessions. Handles:
#   - Push code (rsync excluding /tmp, models, logs)
#   - Push corpora + init models (the gitignored runtime data)
#   - Trigger setup + run on pod via ssh
#   - Pull results back when done
#
# Usage:
#   bash rnd/runpod/launch.sh setup POD_USER@POD_IP[:PORT]
#       # First-time setup on a fresh pod (manual install path): rsync
#       # code + data, run setup_pod.sh. Use when pod is provisioned from
#       # a generic image (e.g. runpod/pytorch).
#
#   bash rnd/runpod/launch.sh setup-image POD_USER@POD_IP[:PORT]
#       # First-time setup when pod was provisioned from
#       # docker.io/7rans/agpt:latest (binaries already pre-built). Only
#       # pushes runtime data; skips Crystal install / binary compile.
#
#   bash rnd/runpod/launch.sh push-code POD_USER@POD_IP[:PORT]
#       # Rsync code (src/, Justfile, scripts) without data. Useful with
#       # setup-image when your laptop has uncommitted code changes you
#       # want to test on the pod — follow with `just build-agpt-train`
#       # inside the pod.
#
#   bash rnd/runpod/launch.sh run POD_USER@POD_IP[:PORT] 'COMMAND'
#       # Run a command on the pod (ssh wrapper that ensures cwd is /workspace/agpt)
#
#   bash rnd/runpod/launch.sh pull POD_USER@POD_IP[:PORT]
#       # Pull logs and findings back to laptop. Run when experiments are done
#       # but BEFORE stopping the pod.
#
#   bash rnd/runpod/launch.sh full POD_USER@POD_IP[:PORT] 'COMMAND'
#       # All-in-one (manual install path): setup + run + pull.
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

usage() { sed -n '4,40p' "$0"; exit 1; }
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
    # Whitelist approach for rnd/: only ship the experiment scripts we
    # actually run on the pod. The convergence archives and old research
    # subdirs are local-only and would otherwise add ~4GB to the push.
    #
    # NO --delete. push_code is additive only. Pod-side state — including
    # in-progress experiment outputs under rnd/<exp>/ — must not be
    # destroyed by a code sync. Burned by this 2026-05-21: ran push-code
    # while a Gutenberg sweep was running, --delete wiped the in-progress
    # output dir, killed the experiment.
    rsync -avzP --no-owner --no-group --no-perms \
        --include='rnd/' \
        --include='rnd/streaming-agpt-v1/' \
        --exclude='rnd/streaming-agpt-v1/models/' \
        --exclude='rnd/streaming-agpt-v1/logs/' \
        --include='rnd/streaming-agpt-v1/**' \
        --include='rnd/runpod/' \
        --include='rnd/runpod/**' \
        --include='rnd/beta2-diagnostic/' \
        --include='rnd/beta2-diagnostic/**' \
        --include='rnd/composite-weights/' \
        --include='rnd/composite-weights/**' \
        --exclude='rnd/*' \
        --exclude='/tmp/' \
        --exclude='.git/' \
        --exclude='bin/' \
        --exclude='build/' \
        --exclude='lib/' \
        --exclude='.shards/' \
        --exclude='data/' \
        --exclude='*.dwarf' \
        --exclude='.claude/' \
        --exclude='notes/paper.html' \
        --rsh="${RSYNC_RSH}" \
        "${PROJ}/" "${SSH_HOST}:${REMOTE_BASE}/"
}

# --- helper: push runtime data (corpora + init models + tries, gitignored) ---
push_data() {
    echo "→ rsync corpora + init models..."
    # Corpora
    rsync -avzP --no-owner --no-group --no-perms --rsh="${RSYNC_RSH}" \
        "${PROJ}/data/input.txt" \
        "${PROJ}/data/gutenberg_5m.txt" \
        "${SSH_HOST}:${REMOTE_BASE}/data/"
    # Init models (random + seeded)
    rsync -avzP --no-owner --no-group --no-perms --rsh="${RSYNC_RSH}" \
        "${PROJ}/data/input.random.model" \
        "${SSH_HOST}:${REMOTE_BASE}/data/" 2>/dev/null || true
    # Seeded init models (in /tmp on laptop)
    if ls /tmp/init_seed*.model &>/dev/null; then
        rsync -avzP --no-owner --no-group --no-perms --rsh="${RSYNC_RSH}" \
            /tmp/init_seed*.model "${SSH_HOST}:/tmp/"
    fi
    # Per-seed init models (used by recent weighting experiments)
    if ls /tmp/seed*.model &>/dev/null; then
        rsync -avzP --no-owner --no-group --no-perms --rsh="${RSYNC_RSH}" \
            /tmp/seed*.model "${SSH_HOST}:/tmp/"
    fi
    # Radix tries (gitignored, expensive to rebuild — push if present).
    # Both Shakespeare 1M and Gutenberg 5M tries from the current
    # weighting / per-fire-norm experiments. ~100-500 MB each.
    for trie_dir in /tmp/shake_baseline_d16_radix /tmp/gutenberg_5m_baseline_d16_radix; do
        if [ -d "$trie_dir" ]; then
            echo "→ rsync $(basename $trie_dir)..."
            rsync -avzP --no-owner --no-group --no-perms --rsh="${RSYNC_RSH}" \
                "$trie_dir/" "${SSH_HOST}:${trie_dir}/"
        fi
    done
    # Held-out files used by the heldout PPL evaluator
    for ho in /tmp/shake_holdout.txt /tmp/gut_holdout.txt; do
        if [ -f "$ho" ]; then
            rsync -avzP --no-owner --no-group --no-perms --rsh="${RSYNC_RSH}" \
                "$ho" "${SSH_HOST}:${ho}"
        fi
    done
}

# --- helper: run command on pod ---
pod_run() {
    local cmd=$1
    echo "→ ssh: cd ${REMOTE_BASE} && ${cmd}"
    # CRYSTAL_WORKERS prevents Crystal 1.20+ thread-init overflow on
    # high-vCPU container hosts (RunPod sees host's full core count).
    ssh ${SSH_PORT_ARG} "${SSH_HOST}" "export CRYSTAL_WORKERS=8 && cd ${REMOTE_BASE} && ${cmd}"
}

# --- helper: pull results ---
pull_results() {
    echo "→ rsync logs back to laptop..."
    rsync -avzP --no-owner --no-group --no-perms --rsh="${RSYNC_RSH}" \
        "${SSH_HOST}:${REMOTE_BASE}/rnd/streaming-agpt-v1/logs/" \
        "${PROJ}/rnd/streaming-agpt-v1/logs/"
    # Also pull any findings.md updates
    rsync -avzP --no-owner --no-group --no-perms --rsh="${RSYNC_RSH}" \
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
    # New mode: pod is provisioned from docker.io/7rans/agpt:latest, so the
    # toolchain + pre-built binaries (agpt_train, agpt_build_radix_corpus,
    # microgpt, perplexity) already exist at /workspace/agpt/bin/. Skips
    # setup_pod.sh (Crystal install, shards install, binary compile). Just
    # pushes the gitignored runtime data (corpora + init models). If you
    # have local code changes, run launch.sh push-code afterwards.
    setup-image)
        push_data
        echo ""
        echo "✓ Image-based pod ready. Binaries already at /workspace/agpt/bin/."
        echo "  Next: bash rnd/runpod/launch.sh run ${POD} 'YOUR EXPERIMENT'"
        echo "  Or:   bash rnd/runpod/launch.sh push-code ${POD}  (if you have local code changes,"
        echo "        then 'just build-agpt-train' inside the pod to rebuild)"
        ;;
    push-code)
        # Useful with --setup-image when laptop's code has diverged from
        # the image's snapshot. Pushes code without touching data.
        push_code
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
