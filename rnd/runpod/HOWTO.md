# RunPod workflow — how to run AGPT experiments on a cloud GPU

Practical guide for the working pipeline as of 2026-05-21. The existing
`README.md` files in this directory and in `rnd/docker/` cover history
and rationale; this file is the day-to-day cookbook.

## What this gives you

A pre-built Docker image (`docker.io/7rans/agpt:latest`) that boots a
RunPod A100/H100 with:

- AGPT binaries already compiled at `/opt/agpt/bin/` (no Crystal
  install, no nvcc build on the pod)
- CUDA 12.4 toolkit (works on RunPod drivers ≥550)
- sshd entrypoint exposing port 22 so `rsync` and
  `ssh root@<IP> -p <PORT> 'cmd'` automation works
- `CRYSTAL_WORKERS=8` and `PATH` propagated across login shells,
  non-login shells, and PAM-mediated sessions

Plus a launcher script (`launch.sh`) for pushing data, running
experiments, and pulling results back.

## One-time setup (already done — for reference)

If you ever rebuild this from scratch on a new laptop:

1. **SSH key on RunPod.** Generate `~/.ssh/runpod`, paste the `.pub`
   into RunPod account → Settings → SSH Public Keys. Add it to your
   ssh-agent or `~/.ssh/config`.
2. **Docker Hub creds for GH Actions.** Repo → Settings → Secrets →
   Actions. Add `DOCKERHUB_USERNAME=7rans` and `DOCKERHUB_TOKEN=...`
   (token from hub.docker.com → Account → Security → New Access Token,
   Read/Write/Delete scope).
3. **RunPod pod template.** Web UI → Templates → New. Image:
   `docker.io/7rans/agpt:latest`. Container disk: 30 GB. Volume disk:
   50 GB mounted at `/workspace`. **Expose TCP Ports: `22`** (literal
   "22" in the TCP field — separate from HTTP ports).

## Daily workflow

### 1. Provision a pod

Web UI → Deploy → choose **Secure Cloud** (Community Cloud often
won't give direct TCP). Pick A100 SXM 80GB or H100, the agpt template.
Wait for status `RUNNING` (~1-3 min for image pull).

If the Connect panel only shows the `ssh.runpod.io` proxy form and no
direct `ssh root@<IP> -p <PORT>`, the container is crash-looping or
TCP exposure failed. Check container logs from the UI; destroy and
retry on a different host (the physical RunPod host has to support
direct TCP, and not all do).

### 2. Push data + tries

```sh
bash rnd/runpod/launch.sh setup-image root@<IP>:<PORT>
```

This rsyncs ~485 MB to the pod (Shakespeare + Gutenberg corpora, init
models, radix tries, holdout files). Takes ~3 min on average WiFi.

If you have **uncommitted code changes** you want to test on the pod
(rather than what's baked into the image):

```sh
bash rnd/runpod/launch.sh push-code root@<IP>:<PORT>
bash rnd/runpod/launch.sh run root@<IP>:<PORT> 'just build-agpt-train'
```

Otherwise skip — the image's binaries are already at `/opt/agpt/bin/`.

### 3. Run experiments

```sh
bash rnd/runpod/launch.sh run root@<IP>:<PORT> '<command>'
```

The script wraps your command with `cd /workspace/agpt && export
CRYSTAL_WORKERS=8 && ...`. Persist all training output to
`rnd/<experiment>/` on the pod — `/tmp` is tmpfs and will be lost on
container restart. See [feedback_persist_results.md] in memory.

Smoke test that the path works:

```sh
bash rnd/runpod/launch.sh run root@<IP>:<PORT> \
  'agpt_train --model /tmp/seed1.model \
   --trie-dir /tmp/shake_baseline_d16_radix \
   --epochs 5 --lr 3e-3 --optimizer rmsprop \
   --lr-schedule warmup-cosine --warmup-epochs 1 \
   --partition-depth 1 --mass-weight off --no-accumulate \
   --save /tmp/smoke.model'
```

Should finish in ~14 sec, loss ~2.45 → 2.28 across 5 epochs.

### 4. Pull results

Before stopping the pod:

```sh
bash rnd/runpod/launch.sh pull root@<IP>:<PORT>
```

Pulls `rnd/streaming-agpt-v1/logs/` and `findings.md`. Extend
`pull_results()` in `launch.sh` if other experiment directories need
to come back.

### 5. Stop / destroy

Stop pod from the UI. RunPod bills storage on stopped pods too — for
long-running experiments that need to resume, leave running; for
one-shots, destroy.

## Rebuilding the image

When `src/cuda/agpt_train.cu`, `Justfile`, `src/cudax/`, or anything
the image bakes in changes:

```sh
# Make the change, commit, push to main
git add ... && git commit && git push

# Trigger the GH Actions build (manual — never auto on push)
gh workflow run docker-image.yml

# Watch
gh run watch          # or: gh run list --workflow=docker-image.yml
```

Build times:

- **Cold** (first build, or changes early in the Dockerfile): ~8-10 min
- **Warm** (only source code or entrypoint changed): ~3 min, thanks to
  BuildKit's `type=gha` cache

Both tags get pushed: `:latest` (moving) and `:YYYY-MM-DD` (immutable
for that date). Use the date tag in pod templates if you need a
reproducible reference; `:latest` is fine for iterative dev.

Once pushed, **provision a fresh pod** to pick up the new image —
RunPod's running pods don't pull updates dynamically. Stopping and
restarting an existing pod does NOT re-pull; you have to destroy +
re-deploy.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ssh: connect to host ... : Connection refused` right after provision | Image still pulling (4 GB), sshd not up yet | Wait 30-60s, retry. Or use the watcher loop in `launch.sh` (`until ssh ... 'echo ok'; do sleep 15; done`) |
| Connect panel shows only proxy form (`ssh.runpod.io`) | Container crash-looping, OR Community Cloud host without direct TCP | Read container logs in UI. If "is not running": entrypoint crashed — see below. If logs look fine: destroy + redeploy on Secure Cloud A100 SXM |
| `container ... is not running` on every connect | Entrypoint failed. Common causes: missing `/run/sshd` (created in entrypoint, not Dockerfile, because `/run` is tmpfs); sshd binary path wrong (`command -v sshd`, not hardcoded); `set -e` killed a step before sshd | Check container logs in UI; `set -x` in entrypoint surfaces the failing line |
| `bash: agpt_train: command not found` | Non-login SSH session not picking up `/opt/agpt/bin` from `/etc/profile.d/` | The image sets PATH via `/etc/environment` (PAM) and sshd `SetEnv` — both should cover non-login sessions. If broken, check that `bash -lc 'echo $PATH'` works (login shell) — if so, the env-propagation regressed |
| `crystal --version` crashes with `Arithmetic overflow (OverflowError)` | `CRYSTAL_WORKERS` not set; Crystal 1.20+ overflows on high-vCPU hosts | Should be set via ENV + `/etc/environment` + sshd `SetEnv`. If missing, the image regressed |
| `CUDA driver version is insufficient for CUDA runtime version` | Image's CUDA newer than host driver. Driver ≥550 supports CUDA 12.4; we pin to 12.4 | If you see this, the image rebuild may have unpinned CUDA. Check Dockerfile `FROM` line is still `nvidia/cuda:12.4.1-...` |
| GH Actions build fails on pacman/apt step | Upstream package URL drift (e.g. Crystal repo, just GH release) | Check `Dockerfile` for hardcoded URLs; update the version pin |
| rsync hangs at 99%+ | Pod's `/workspace` is the RunPod FUSE mount and gets sluggish with many small files | Patience usually. If actually wedged: `bash rnd/runpod/launch.sh run <POD> 'sync; sleep 2'` and retry |

## Files in this workflow

- `rnd/runpod/launch.sh` — laptop-side automation
  (push-code, push-data, run, pull, full)
- `rnd/runpod/setup_pod.sh` — legacy: manual install on a stock
  PyTorch/Ubuntu pod. Use only if the agpt image is unavailable.
- `rnd/docker/Dockerfile` — image definition. Base
  `nvidia/cuda:12.4.1-devel-ubuntu22.04`. Installs Crystal upstream,
  builds binaries, configures sshd.
- `rnd/docker/entrypoint.sh` — container PID 1. Creates `/run/sshd`,
  generates host keys, injects `$PUBLIC_KEY`, execs sshd.
- `rnd/runpod/microgpt_tf32.patch` — small patch applied to the
  microgpt shard at build time to enable TF32. Carried across image
  rebuilds.
- `.github/workflows/docker-image.yml` — GH Actions build workflow.
  Manual trigger only (`workflow_dispatch`). Uses BuildKit GH cache.

## Cost notes

A100 SXM 80GB on RunPod Secure Cloud: ~$1.89/hr. Typical
single-experiment session: 1-3 hr → $2-6. The 485 MB initial data
push is 3-5 min of pod time. Stop pods when not running experiments.
