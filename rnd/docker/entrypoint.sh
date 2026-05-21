#!/bin/bash
# Container entrypoint for RunPod-style cloud sessions.
#
# Does three things:
#   1. Generate SSH host keys on first start (they're persistent only if the
#      pod's persistent volume is mounted somewhere covering /etc/ssh — the
#      typical RunPod 50 GB volume isn't, so keys are regenerated each
#      container start. That's fine for short-lived pods.)
#   2. Inject the user's public key from $PUBLIC_KEY into root's
#      authorized_keys. RunPod sets $PUBLIC_KEY automatically based on the
#      account's stored SSH keys.
#   3. Start sshd in the foreground, which keeps the container alive AND
#      lets `ssh root@<IP> -p <PORT>` work for our launch.sh automation.

set -e

# 1. Generate host keys if missing.
if [ ! -f /etc/ssh/ssh_host_ed25519_key ]; then
    ssh-keygen -A
fi

# 2. Inject RunPod's PUBLIC_KEY env var as an authorized key.
if [ -n "$PUBLIC_KEY" ]; then
    mkdir -p /root/.ssh
    chmod 700 /root/.ssh
    echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
fi

# 3. sshd in foreground keeps the container alive while serving SSH.
#    Path differs: /usr/sbin/sshd on Ubuntu/Debian, /usr/bin/sshd on Arch.
#    Resolve via PATH so the entrypoint is portable.
exec "$(command -v sshd)" -D -e
