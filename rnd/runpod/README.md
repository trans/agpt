# RunPod launcher for AGPT experiments

Lets you provision an H100/H200/A100 RunPod instance, sync code+data, run
experiments, and pull results back — without manually maintaining the env.

## One-time setup on your end

1. Generate an SSH key for RunPod if you don't have one:
   ```
   ssh-keygen -t ed25519 -f ~/.ssh/runpod -N ""
   cat ~/.ssh/runpod.pub
   # add this public key to your RunPod account (Settings → SSH Public Keys)
   ```

2. (Optional) Add a config block to `~/.ssh/config` so you don't have to
   pass the key every time. RunPod will give you a command line like
   `ssh root@69.30.85.92 -p 18234 -i ~/.ssh/runpod`; the equivalent config:
   ```
   Host runpod-current
       HostName 69.30.85.92
       Port 18234
       User root
       IdentityFile ~/.ssh/runpod
   ```

## Provisioning a pod

In RunPod's UI:
1. Pick a GPU: A100 SXM 80GB (~$1.89/hr, recommended — usually available),
   H100 SXM (~$2.49/hr, when in stock), H200 (141GB, ~$3.59/hr, often gone)
2. Pick a CUDA image: their default `runpod/pytorch` works fine — we just
   need CUDA + nvcc. Avoid images that pin a specific PyTorch version we
   don't need.
3. Set persistent volume to 50 GB (corpora + tries + models)
4. Note the SSH command they give you (host, port)

## Using the launcher

From your laptop, in the agpt project root:

```sh
# First-time setup (rsync code + data, install deps, build):
bash rnd/runpod/launch.sh setup root@69.30.85.92:18234

# Run an experiment:
bash rnd/runpod/launch.sh run root@69.30.85.92:18234 \
  "CORPUS=\$PWD/data/gutenberg_5m.txt bash rnd/streaming-agpt-v1/run_multiseed_baseline.sh 500 200 300"

# Pull results back to laptop:
bash rnd/runpod/launch.sh pull root@69.30.85.92:18234

# Or all-in-one:
bash rnd/runpod/launch.sh full root@69.30.85.92:18234 \
  "CORPUS=\$PWD/data/gutenberg_5m.txt bash rnd/streaming-agpt-v1/run_multiseed_baseline.sh 500 200 300"
```

`launch.sh full` does setup + run + pull in sequence. Best for one-off
experiments. `launch.sh setup` + `launch.sh run` separately is better when
you want to leave the pod running and fire off multiple experiments.

## What gets transferred

**To the pod:**
- All Crystal/CUDA source (`src/`, `lib/microgpt` builds from shard on the pod)
- All experiment scripts (`rnd/streaming-agpt-v1/*.sh`)
- `data/input.txt`, `data/gutenberg_5m.txt` (corpora)
- `data/input.random.model` (random init checkpoint)
- `/tmp/init_seed{100,200,300}.model` (seeded init checkpoints)
- Notes (`notes/*.md`)

**Excluded (not sent):**
- `bin/`, `build/`, `lib/` — rebuilt fresh on the pod
- `rnd/streaming-agpt-v1/models/` — gitignored, regenerated locally to pod
- `rnd/streaming-agpt-v1/logs/` — pulled BACK from pod
- `rnd/seq-len-decouple/*.bin` — the 205MB position maps
- `data/wormhole_*.txt` — large research artifacts

**Pulled back:**
- All training logs and PPL eval outputs from `rnd/streaming-agpt-v1/logs/`
- Updated `findings.md` if the pod modified it

## Cost estimate per session

| GPU | $/hr | typical experiment | wall time | cost |
|---|---:|---|---:|---:|
| A100 SXM 80GB | $1.89 | 2 Gutenberg baseline seeds | ~3 hr | ~$6 |
| A100 SXM 80GB | $1.89 | 3 streaming + 3 baseline seeds (1000 SE each) | ~12 hr | ~$23 |
| H100 SXM | $2.49 | Same as above | ~7 hr | ~$18 |
| H200 | $3.59 | Same as above | ~6 hr | ~$22 |

A100 SXM is the most available tier. Bandwidth-bound AGPT workload
(KV gather) sees ~2× speedup on H200 over A100 thanks to HBM3e
(4.8 TB/s) vs HBM2e (1.94 TB/s). For our small models the matmul
speedup is limited; memory bandwidth is the bigger win when we can
get it. But total cost differences are within noise once you account
for availability — A100 SXM is the practical default.

## Notes on building on the pod

- `setup_pod.sh` installs Crystal + just + builds with the same flags as
  laptop: `-O3 --use_fast_math -gencode=sm_89,sm_90`.
- It applies `microgpt_tf32.patch` so the cuBLAS path on microgpt also gets
  TF32 (since `lib/microgpt` is fetched fresh from the shard).
- The build is fast (~2-3 min on RunPod since they have fast disks +
  recent CUDA).
- First-time setup including all deps: ~10-15 min.

## Don't forget to stop the pod

RunPod bills per second. After `launch.sh pull` succeeds:
1. Verify the results landed on your laptop (`ls rnd/streaming-agpt-v1/logs/`).
2. Stop the pod via RunPod UI (Stop, not Delete, if you want to keep the
   volume for next session).
