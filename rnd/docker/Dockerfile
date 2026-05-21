# AGPT build environment + pre-compiled binaries.
#
# Base: nvidia/cuda 12.4.1 devel on Ubuntu 22.04. CUDA must match RunPod
# host driver compatibility (550-series drivers cap at CUDA 12.4); Ubuntu
# 22.04's glibc 2.35 doesn't have the exception-spec clash with CUDA 12.4's
# crt/math_functions.h that current Arch glibc has. Earlier Arch-based
# iterations hit cascading version-pin failures fighting a rolling release;
# we switched here to stop that loop.
#
# The laptop dev environment is Arch on CUDA 13.2/glibc 2.41, which is
# explicitly NOT what this image targets — the image targets the cloud's
# 550 driver. Our agpt_train.cu uses only plain CUDA C++ + cuBLAS, no
# 12.5+-only APIs, so the version split is invisible to the code.
#
# Image size: ~5-8GB. Cold-start on a fresh pod: binaries at /opt/agpt/bin/.
#
# Filesystem layout — important for RunPod:
#   /opt/agpt/bin/  — pre-built binaries (NOT inside RunPod's /workspace
#                      volume mount, so they survive the FUSE shadow).
#   /workspace/agpt — RunPod mounts persistent storage here. Empty by
#                      default; user rsyncs source code + data per session.
#                      Anything the image COPYs into /workspace/ at build
#                      time gets shadowed at container start (verified by
#                      cyclops 2026-05-21: mfs#us-wa-1.runpod.net:9421
#                      FUSE mount).

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Base toolchain + Crystal upstream repo + openblas + sshd + just (from
# GitHub release — not in Ubuntu 22.04 universe). One layer for image size.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential pkg-config rsync git wget curl ca-certificates \
        gnupg openssh-server \
        libopenblas64-0 libopenblas64-dev libpcre2-dev libssl-dev \
        libgc-dev libyaml-dev libxml2-dev libgmp-dev && \
    curl -fsSL https://crystal-lang.org/install.sh | bash && \
    apt-get install -y --no-install-recommends crystal && \
    curl -fsSL https://github.com/casey/just/releases/download/1.36.0/just-1.36.0-x86_64-unknown-linux-musl.tar.gz \
        | tar -xz -C /usr/local/bin just && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Justfile hardcodes /opt/cuda/bin/nvcc and -L/opt/cuda/lib64. The
# nvidia/cuda image installs at /usr/local/cuda; symlink so Justfile and
# laptop-style invocations work without patching.
RUN ln -sf /usr/local/cuda /opt/cuda

# microgpt's backend.cr links with -lopenblas_64 (Arch's libopenblas_64.so
# naming). Ubuntu ships libopenblas64.so.0 (no underscore). Symlink so the
# linker resolves; matches what microgpt/src/cloud/vast_ai.cr does for
# Vast.ai pods.
RUN ln -sf /usr/lib/x86_64-linux-gnu/libopenblas64.so.0 \
           /usr/lib/x86_64-linux-gnu/libopenblas_64.so

# Configure sshd for RunPod-style key-only access (no passwords).
#
# SetEnv is critical: `ssh host 'cmd'` runs a non-login bash shell which
# sources NEITHER /etc/profile.d nor ~/.bash_profile. Without sshd setting
# the env directly, PATH and CRYSTAL_WORKERS are missing from
# non-interactive sessions — breaking launch.sh automation. /etc/environment
# (set below) covers PAM-mediated paths; SetEnv covers everything sshd spawns.
RUN mkdir -p /run/sshd /root/.ssh && \
    chmod 700 /root/.ssh && \
    sed -i 's/^#\?PermitRootLogin .*/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/^#\?PasswordAuthentication .*/PasswordAuthentication no/' /etc/ssh/sshd_config && \
    sed -i 's/^#\?PubkeyAuthentication .*/PubkeyAuthentication yes/' /etc/ssh/sshd_config && \
    printf '\n# AGPT environment for non-login SSH sessions (added 2026-05-21).\nSetEnv CRYSTAL_WORKERS=8\nSetEnv PATH=/opt/agpt/bin:/workspace/agpt/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\n' >> /etc/ssh/sshd_config

# Crystal 1.20+ overflows default_workers_count on high-vCPU container hosts
# (RunPod sees host's 100+ cores). Cap to 8 across all entry paths:
#   - ENV: image entrypoint + its children (sshd itself)
#   - /etc/environment: PAM-mediated sessions
#   - sshd SetEnv (above): non-login `ssh host 'cmd'` sessions
ENV CRYSTAL_WORKERS=8
RUN echo 'CRYSTAL_WORKERS=8' >> /etc/environment

WORKDIR /workspace/agpt

# Copy only what's needed for build.
COPY shard.yml shard.lock Justfile ./
COPY src/ ./src/
COPY rnd/runpod/microgpt_tf32.patch ./rnd/runpod/
COPY rnd/docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Fetch microgpt shard, apply TF32 patch.
RUN shards install && \
    (cd lib/microgpt && patch -p1 < /workspace/agpt/rnd/runpod/microgpt_tf32.patch || \
     echo "WARNING: TF32 patch did not apply cleanly; check microgpt version drift")

# Build all required binaries.
RUN just build-agpt-train && \
    just build-agpt-build-radix-corpus && \
    just build-microgpt-tools && \
    just build-agpt-sliding-window-perplexity

# Stash binaries at /opt/agpt/bin — OUTSIDE /workspace — so RunPod's FUSE
# volume mount doesn't shadow them at container start.
#
# PATH is exposed on three levels (mirrors CRYSTAL_WORKERS): /etc/environment
# for PAM, sshd SetEnv (above) for non-login `ssh host 'cmd'`, and ENV below
# for the entrypoint and its children.
RUN mkdir -p /opt/agpt && \
    cp -r bin /opt/agpt/ && \
    echo 'PATH=/opt/agpt/bin:/workspace/agpt/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin' >> /etc/environment

ENV PATH="/opt/agpt/bin:/workspace/agpt/bin:/usr/local/cuda/bin:${PATH}"

# Run sshd as PID 1. Keeps the container alive AND lets `ssh root@<IP>
# -p <PORT>` work for launch.sh automation (rsync + non-interactive cmds),
# which the RunPod ssh.runpod.io proxy doesn't support.
EXPOSE 22
CMD ["/entrypoint.sh"]
