# AGPT Docker image

Self-contained build environment + pre-compiled binaries for AGPT.

## Why

The RunPod PyTorch images hit platform-mismatch issues at every step
of setup: missing rsync, Crystal CPU-count overflow, /opt/cuda vs
/usr/local/cuda path mismatch, libopenblas_64 vs libopenblas64 naming
mismatch, etc. (See `rnd/runpod/setup_pod.sh` for the laundry list.)

This image side-steps all of that by basing on Arch (matches the
laptop dev environment exactly) and pre-building the binaries.
**Cold start on a fresh pod: instant.**

## Build locally (first time, ~15-20 min)

```sh
bash rnd/docker/build.sh
```

Image will be tagged `agpt:latest`. ~5-8GB (CUDA package is the bulk).

Subsequent rebuilds after source changes: ~3-5 min thanks to layer
caching (only the `just build-*` and below layers re-run when src/
changes).

## Run locally

Interactive shell:
```sh
podman run --rm -it -v $PWD/data:/workspace/agpt/data:ro agpt:latest
```

Run a training command:
```sh
podman run --rm \
    -v $PWD/data:/workspace/agpt/data:ro \
    -v $PWD/rnd:/workspace/agpt/rnd \
    agpt:latest \
    bash rnd/streaming-agpt-v1/run_multiseed_generic.sh 100 5 100
```

Note: data is mounted read-only, rnd/ is read-write so logs persist
back to the host.

For GPU access locally (laptop), add `--device nvidia.com/gpu=all`:
```sh
podman run --rm --device nvidia.com/gpu=all -it agpt:latest
```
(requires nvidia-container-toolkit installed; podman-specific CDI setup.)

## Push to registry (after local testing)

Decide between Docker Hub vs GHCR:

**Docker Hub** (simpler):
```sh
podman login docker.io
# Push :latest (moving tag, convenient for "give me the most recent")
podman tag agpt:latest docker.io/7rans/agpt:latest
podman push docker.io/7rans/agpt:latest
# ALSO push an immutable date-tagged version (use this in RunPod for
# reproducibility — :latest is a moving target).
DATE_TAG=$(date +%Y-%m-%d)
podman tag agpt:latest docker.io/7rans/agpt:${DATE_TAG}
podman push docker.io/7rans/agpt:${DATE_TAG}
```

**GitHub Container Registry**:
```sh
echo $GH_TOKEN | podman login ghcr.io -u 7rans --password-stdin
podman tag agpt:latest ghcr.io/7rans/agpt:latest
podman push ghcr.io/7rans/agpt:latest
```

## Use on RunPod

Once pushed:
1. Create a new pod with **Custom Container Image** set to
   `docker.io/7rans/agpt:latest` (or `ghcr.io/...`)
2. SSH in — no apt-get, no Crystal install, no symlinks. Binaries
   already at `bin/`.
3. Use the existing `launch.sh` for code/data rsync. Now ~30 sec to
   fully provisioned vs the previous ~90 min.

## What's in the image

- `archlinux:latest` base
- `cuda` package: nvcc, cudart, cublas, all at `/opt/cuda/`
- `crystal` + `shards` + `just`
- `openblas64` (with the `_64` naming the Justfile expects)
- `base-devel`, `pkg-config`, `rsync`, `git`, `wget`
- Project source under `/workspace/agpt/`
- `lib/microgpt/` shard fetched and TF32-patched
- Pre-built binaries at `/workspace/agpt/bin/`:
  - `agpt_train`
  - `agpt_build_radix_corpus`
  - `microgpt`
  - `perplexity`
- `CRYSTAL_WORKERS=8` env (sidesteps Crystal 1.20 high-CPU overflow)
- `PATH` includes `bin/` for convenience

## What's NOT in the image

- Corpora (`data/`) — rsync'd at run time
- Init models (`/tmp/init_seed*.model`) — rsync'd at run time
- Experiment scripts in `rnd/*/`  except runpod/microgpt_tf32.patch —
  rsync'd at run time (they evolve faster than the image is rebuilt)
- Training outputs (`logs/`, `models/`) — generated at run time

## Image rebuild triggers

The image needs rebuild when **anything in the COPY'd paths** changes:
- `src/cuda/agpt_train.cu` (triggers `just build-agpt-train`)
- `lib/microgpt/` source (rare — it's a vendored shard)
- `Justfile`, `shard.yml`, `shard.lock`
- `rnd/runpod/microgpt_tf32.patch`

Otherwise (data, experiment scripts, notes): no rebuild needed.
