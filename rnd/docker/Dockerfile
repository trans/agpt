# AGPT build environment + pre-compiled binaries.
#
# Arch base because the project's Justfile is calibrated to Arch paths
# (/opt/cuda/bin/nvcc, libopenblas_64.so naming). Matches the laptop dev
# environment exactly, sidestepping the platform-mismatch issues we hit
# with Ubuntu-based RunPod images (rnd/runpod/setup_pod.sh).
#
# Image size: ~5-8GB (cuda package alone is ~3GB).
# Cold-start on a fresh pod after `docker run`: instant — binaries are
# at /opt/agpt/bin/.
#
# Filesystem layout — important for RunPod:
#   /opt/agpt/bin/  — pre-built binaries (NOT inside RunPod's /workspace
#                      volume mount, so they survive the FUSE shadow).
#   /workspace/agpt — RunPod mounts persistent storage here. Empty by
#                      default; user rsyncs source code + data here per
#                      session. Anything the image COPYs into /workspace/
#                      at build time gets shadowed at container start.
#                      Confirmed via cyclops investigation 2026-05-21:
#                      RunPod's mfs#us-wa-1.runpod.net:9421 FUSE mount
#                      hides image content at /workspace/*.

FROM archlinux:latest

# Refresh pacman keyring (sometimes the base image's keys are stale) and
# install everything in one layer to keep image size down.
RUN pacman-key --init && \
    pacman-key --populate archlinux && \
    pacman -Syu --noconfirm && \
    pacman -S --noconfirm --needed \
        base-devel cuda crystal shards just openblas64 \
        pkg-config rsync git wget openssh && \
    pacman -Scc --noconfirm

# Configure sshd for RunPod-style key-only access (no passwords).
# The entrypoint will inject $PUBLIC_KEY into authorized_keys at start.
RUN mkdir -p /run/sshd /root/.ssh && \
    chmod 700 /root/.ssh && \
    sed -i 's/^#\?PermitRootLogin .*/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/^#\?PasswordAuthentication .*/PasswordAuthentication no/' /etc/ssh/sshd_config && \
    sed -i 's/^#\?PubkeyAuthentication .*/PubkeyAuthentication yes/' /etc/ssh/sshd_config

# Crystal 1.20+ overflows default_workers_count on high-vCPU container
# hosts (RunPod sees host's 100+ cores). Cap to 8.
#
# ENV alone isn't enough: it applies to the entrypoint + children but
# fresh SSH sessions don't inherit it (sshd spawns a new login shell
# that only reads PAM env files). Also write to /etc/profile.d/ so
# `ssh root@pod 'crystal --version'` works without the user setting it.
# Verified by cyclops 2026-05-21: without profile.d, ssh-spawned shells
# crash with `Unhandled exception: Arithmetic overflow (OverflowError)`.
ENV CRYSTAL_WORKERS=8
RUN echo 'export CRYSTAL_WORKERS=8' > /etc/profile.d/crystal.sh && \
    chmod 644 /etc/profile.d/crystal.sh

WORKDIR /workspace/agpt

# Copy only what's needed for build. Data + experiment scripts come via
# rsync to the running container; image stays focused on toolchain + bins.
COPY shard.yml shard.lock Justfile ./
COPY src/ ./src/
COPY rnd/runpod/microgpt_tf32.patch ./rnd/runpod/
COPY rnd/docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Fetch microgpt shard, apply TF32 patch.
RUN shards install && \
    (cd lib/microgpt && patch -p1 < /workspace/agpt/rnd/runpod/microgpt_tf32.patch || \
     echo "WARNING: TF32 patch did not apply cleanly; check microgpt version drift")

# Build all required binaries. Same flags as the laptop Justfile —
# -O3 --use_fast_math -gencode=sm_80,sm_89,sm_90.
RUN just build-agpt-train && \
    just build-agpt-build-radix-corpus && \
    just build-microgpt-tools

# Stash the built binaries at /opt/agpt/bin/ — OUTSIDE /workspace —
# so RunPod's volume mount can't shadow them. Verified by cyclops
# 2026-05-21: /workspace gets overlaid by mfs#us-wa-1.runpod.net:9421
# FUSE on container start, hiding anything the image COPYs there.
RUN mkdir -p /opt/agpt && \
    cp -r bin /opt/agpt/ && \
    echo 'export PATH="/opt/agpt/bin:$PATH"' > /etc/profile.d/agpt.sh && \
    chmod 644 /etc/profile.d/agpt.sh

# Expose built binaries on PATH for convenience. /opt/agpt/bin is where
# the binaries actually live in the running container; /workspace/agpt/bin
# kept as a fallback for users who push their own builds via launch.sh
# push-code + just build (those land in the live volume).
ENV PATH="/opt/agpt/bin:/workspace/agpt/bin:${PATH}"

# Run the entrypoint, which generates SSH host keys, injects $PUBLIC_KEY
# into authorized_keys, then runs sshd in foreground. sshd keeps the
# container alive AND lets `ssh root@<IP> -p <PORT>` work for the
# launch.sh automation (rsync + non-interactive commands), which the
# RunPod ssh.runpod.io proxy doesn't support.
EXPOSE 22
CMD ["/entrypoint.sh"]
