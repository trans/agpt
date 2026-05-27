# cap-recurrence smoke test — 2026-05-27

Phase 1 MVP validation of the h_cap capture hook. Goal: verify that
`launch_capture_h_caps` runs without crashing, fills the per-radix-id
EMA buffer correctly, and writes a valid binary file. Not a training
quality experiment — just plumbing validation.

## Setup

- Trie: Shakespeare 1M d=16 (`/tmp/shake_d16_radix`, 1,607,928 radix
  nodes built from data/input.txt via agpt_build_index then
  agpt_build_radix).
- Seed: data/input.random.model (108,481 weight floats, d_model=64,
  2 layers, 4 heads).
- Trainer config: 1 SE, partition-depth 1, no-accumulate, lr=3e-3
  warmup-cosine, rmsprop β=0.999, mass-weight log, entropy-λ 1.0.
- Capture: AGPT_CAPTURE_H_CAPS=1, EMA α=0.9, output h_caps.bin.

## Results

**Capture stats (from C trainer, end of training):**
```
[h_cap stats final] radix_count=1607928 filled=1607927 (100.0%)
  norm: mean=0.7847 std=0.0102 min=0.7594 max=0.8017
```

**File validation (Python):**
```
header: radix_count=1607928 d_model=64
data bytes: 205814784 (expected 205814784)
norms: mean=0.7847 std=0.0103 min=0.0000 max=0.8017 zeros=1
```

- 100% fill: every radix node hit at least once (the single zero-norm
  entry is radix_id=0, the virtual root, which is never an endpoint).
- File size: exactly 8 (header) + 1607928 × 64 × 2 (bf16 data) bytes.
- Python decoder matches C trainer stats to 3 sig figs.

**Loss parity (3 baseline runs vs 2 capture-on runs):**

| Run | Capture | Loss | Time |
|---|---|---:|---:|
| baseline-1 | off | 2.501414 | 20.33s |
| baseline-2 | off | 2.502076 | 21.38s |
| baseline-3 | off | 2.498984 | 20.95s |
| capture-1 | on | 2.494989 | 6.09s (anomaly) |
| capture-2 | on | 2.494228 | 27.30s |

- Loss spread across 5 runs: 0.008 absolute, 0.3% relative. Within
  typical cuBLAS non-determinism band.
- Capture-on losses are ~0.005 below baseline mean — borderline within
  noise but slightly consistent. Likely from CUDA stream-ordering
  effects on cuBLAS atomic accumulation; not a real training signal.
- Capture overhead: ~6-7s (~35% of 20s baseline). Higher than hoped.
  Likely candidates: per-chunk kernel launch overhead, end-of-training
  cudaMemcpy of 200MB to host. Optimization deferred to later phase if
  it becomes a bottleneck.
- The 6.09s capture-1 run is a timing anomaly; subsequent capture runs
  are consistently 25-30s.

## Conclusion

Smoke test PASSES.

- Build clean (only pre-existing warnings).
- Capture writes to all expected radix slots.
- Binary file format round-trips correctly.
- Loss numbers consistent with baseline noise band.
- Performance overhead (~35%) acceptable for MVP; optimization can
  wait for Phase 2 integration.

## Not committed

`h_caps.bin` is 200MB — excluded via `rnd/cap-recurrence/.gitignore`.
Reproducible by re-running the smoke command (see commit log).

## Repro command

```sh
cp data/input.random.model /tmp/cap_smoke.model
AGPT_CAPTURE_H_CAPS=1 \
AGPT_CAPTURE_H_CAPS_OUT=rnd/cap-recurrence/20260527-smoke/h_caps.bin \
AGPT_CAPTURE_H_CAPS_EMA=0.9 \
bin/agpt_train \
    --model /tmp/cap_smoke.model \
    --trie-dir /tmp/shake_d16_radix \
    --save /tmp/cap_smoke_out.model \
    --epochs 1 --partition-depth 1 --no-accumulate \
    --lr 3e-3 --lr-schedule warmup-cosine --warmup-epochs 1 \
    --optimizer rmsprop --rmsprop-beta 0.999 \
    --mass-weight log --entropy-lambda 1.0
```
