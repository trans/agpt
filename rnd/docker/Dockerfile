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
#
# CUDA is pinned to 12.4.1-4 from Arch's archive, NOT the rolling current
# version (which would be CUDA 13.2+ as of 2026-05). Rationale: RunPod's
# cheap-tier hosts ship NVIDIA driver 550.x, which only speaks the CUDA
# 12.4 protocol. CUDA 13's libcudart.so issues ioctls the 550 kernel
# module doesn't understand → "driver version is insufficient" at any
# runtime call. Forward-compat libs have a hard floor at driver 560 for
# CUDA 13, so they don't bridge this either. The Justfile already passes
# nvcc --allow-unsupported-compiler so a host gcc newer than CUDA 12.4
# supports (gcc≤13) doesn't break the build.
RUN pacman-key --init && \
    pacman-key --populate archlinux && \
    pacman -Syu --noconfirm && \
    pacman -S --noconfirm --needed \
        base-devel crystal shards just openblas64 \
        pkg-config rsync git wget openssh && \
    pacman -U --noconfirm \
        https://archive.archlinux.org/packages/c/cuda/cuda-12.4.1-4-x86_64.pkg.tar.zst && \
    pacman -Scc --noconfirm

# Configure sshd for RunPod-style key-only access (no passwords).
# The entrypoint will inject $PUBLIC_KEY into authorized_keys at start.
#
# SetEnv is critical: `ssh host 'cmd'` runs a non-login bash shell which
# sources NEITHER /etc/profile.d nor ~/.bash_profile. Without sshd setting
# the env directly, PATH and CRYSTAL_WORKERS are missing from
# non-interactive sessions — breaking launch.sh automation. Verified via
# direct probe: bash -lc works, plain `ssh host 'cmd'` fails. /etc/environment
# (set below) covers PAM-mediated paths; SetEnv covers everything sshd spawns.
RUN mkdir -p /run/sshd /root/.ssh && \
    chmod 700 /root/.ssh && \
    sed -i 's/^#\?PermitRootLogin .*/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/^#\?PasswordAuthentication .*/PasswordAuthentication no/' /etc/ssh/sshd_config && \
    sed -i 's/^#\?PubkeyAuthentication .*/PubkeyAuthentication yes/' /etc/ssh/sshd_config && \
    printf '\n# AGPT environment for non-login SSH sessions (added 2026-05-21).\nSetEnv CRYSTAL_WORKERS=8\nSetEnv PATH=/opt/agpt/bin:/workspace/agpt/bin:/usr/local/sbin:/usr/local/bin:/usr/bin:/opt/cuda/bin\n' >> /etc/ssh/sshd_config

# Crystal 1.20+ overflows default_workers_count on high-vCPU container
# hosts (RunPod sees host's 100+ cores). Cap to 8 across all entry paths:
#   - ENV: image entrypoint + its children (sshd itself)
#   - /etc/environment: PAM-mediated sessions (login shells get this via
#     pam_env from /etc/pam.d/sshd)
#   - /etc/profile.d/crystal.sh: interactive login shells (belt + braces)
#   - sshd SetEnv (above): non-login shells from `ssh host 'cmd'` — the
#     critical path for launch.sh automation
# Without all four, some session type leaks through and `crystal --version`
# crashes with `Unhandled exception: Arithmetic overflow (OverflowError)`.
ENV CRYSTAL_WORKERS=8
RUN echo 'CRYSTAL_WORKERS=8' >> /etc/environment && \
    echo 'export CRYSTAL_WORKERS=8' > /etc/profile.d/crystal.sh && \
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
#
# PATH is exposed on three levels (mirrors CRYSTAL_WORKERS plumbing):
# /etc/environment for PAM, /etc/profile.d for interactive login shells,
# and sshd SetEnv (above) for non-login `ssh host 'cmd'` invocations.
RUN mkdir -p /opt/agpt && \
    cp -r bin /opt/agpt/ && \
    echo 'PATH=/opt/agpt/bin:/workspace/agpt/bin:/usr/local/sbin:/usr/local/bin:/usr/bin:/opt/cuda/bin' >> /etc/environment && \
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
