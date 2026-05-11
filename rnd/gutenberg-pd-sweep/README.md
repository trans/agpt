# Gutenberg 5M — partition-depth sweep at d=16/d=18

**Data source:** 19 logs preserved from a multi-day experiment burst on
2026-05-08 (recovered into `logs/` on 2026-05-10 from `/tmp/`, which is
tmpfs and would have been wiped on next reboot).

**Status:** This writeup recovers what we can from the preserved logs.
The saved model files are gone (also tmpfs casualties), so per-run
held-out PPL eval is not directly recoverable — only training-loss
trajectories are. A separate ~7.5-hour 300-SE run was lost entirely
(no log preserved in /tmp).

## Setup

- **Corpus:** Gutenberg 5M (`data/gutenberg_5m.txt`, vocab 65)
- **Trie:** d=16 radix at `/tmp/agpt_g5m_d16_radix` (7.07M nodes,
  34.6M edge chars), d=18 / d=12 / d=26 variants for select runs
- **Model:** d_model=64, n_layers=2, n_heads=4, d_ff=256, seq=128
  (108k params)
- **Recipe (common across all runs):**
  - optimizer: rmsprop, lr=3e-3
  - lr-schedule: warmup-cosine
  - weight-decay: 0.01 (decoupled, AdamW-style)
  - curriculum: flat
  - shuffle-order: per-SE Fisher-Yates over partition groups
  - mass-weight: not surfaced in log header (likely default = log,
    see TODO below)

## Partition-depth → optimizer-fire-density

| pd | partition groups (d=16) | fires per SE |
|---:|---:|---:|
| 1 | 65 (root-child subtrees) | 65 |
| 2 | 1,751 | 1,751 |
| 4 | 86,252 | 86,252 |
| 6 | 629,392 | 629,392 |

pd=6 fires the optimizer **~10 000×** more often than pd=1 per SE.
That density ratio is the dominant cost driver and the dominant
quality knob in this sweep.

## Per-SE-budget results (final training loss)

### d=16, varying pd, common 6 SE budget

| pd | 6SE final loss | wall (s) | fires |
|---:|---:|---:|---:|
| 0 | 4.143 | 135 | ≤ 65 (1 step / SE total fire model) |
| 1 | (not run; pd=1 100SE at SE=6 ≈ 2.25 from trajectory) | 132 | 390 |
| 2 | 1.851 | 136 | 10 506 |
| 4 | 1.549 | 408 | 517 512 |
| 6 | 1.475 | 2 468 | 3 776 352 |

### d=16 pd=1 long-trajectory (the recovery anchor)

100 SE run preserved in `agpt_g5m_d16_pd1_100se.log`:

| SE | loss |
|---:|---:|
| 1 | 3.179 |
| 5 | 2.281 |
| 10 | 2.144 |
| 20 | 1.993 |
| 30 | 1.926 |
| 40 | 1.804 |
| 50 | 1.761 |
| 60 | 1.748 |
| 70 | 1.663 |
| 80 | 1.637 |
| 90 | 1.626 |
| **100** | **1.624** |

Plateau-shaped: Δ(90→100) = 0.002. Wall: 2208 s = 37 min.

This is the *closest preserved data* to the lost 300 SE overnight
run. The 100 SE plateau training loss 1.624 corresponds reasonably
well with the d=32 pd=1 110 SE matched-recipe run's plateau training
loss of 1.598 measured 2026-05-10 (which evals at PPL@32 = 4.90).

### d=16 pd=2, 60 SE

Preserved in `agpt_g5m_d16_pd2_60se.log`:

| SE | loss |
|---:|---:|
| 1 | 2.688 |
| 10 | 2.020 |
| 30 | 1.874 |
| 60 | 1.747 |

Plateau still descending mildly at SE 60. Wall: 1339 s = 22 min.

### d=16 pd=4 / pd=6 at 12 SE

| pd | 12 SE loss | wall |
|---:|---:|---:|
| 4 | 1.523 | 818 s |
| 6 | 1.437 | 5 132 s |

### Depth sweep at pd=6, 12 SE

| d | trie size | 12 SE loss | wall |
|---:|---:|---:|---:|
| 12 | smaller | 1.580 (6SE) | 2 209 s (6SE) |
| 16 | 7.07M nodes | 1.437 | 5 132 s |
| 18 | 8.29M nodes | 1.418 | 4 756 s |
| 26 (shuffle cos) | larger | 1.351 | 2 430 s (12SE) |

## Wall-time-matched comparison (≈ 140 s budget)

| variant | SE achievable in ~140s | est. final loss |
|---|---:|---:|
| pd=0 | 6 | 4.14 |
| pd=1 | 6 | 2.25 |
| pd=2 | 6 | 1.85 |
| pd=4 | ~2 | ~1.83 |
| pd=6 | ~0.34 | (pd=6 1SE = 1.80; pd=6 0.34SE > 1.80) |

At very tight wall budget pd=2 looks competitive. Higher pd needs
proportionally more wall to outperform.

## Key observations

1. **pd=0 is essentially broken.** Loss only drops from 5.02 → 4.14
   in 6 SE and 2.96 in 12 SE. Single-fire-per-SE doesn't accumulate
   useful gradient signal for a transformer of this size.
2. **pd→quality is monotonic** at matched SE budget: pd=6 > pd=4 >
   pd=2 > pd=1 > pd=0 on training loss.
3. **pd→wall is super-linear**: pd=6 6SE costs ~18× pd=2 6SE because
   more partition groups means more chunks fired per SE. Per-SE wall
   is roughly proportional to chunks-processed.
4. **d=16 → d=18 small win** (1.437 → 1.418 at pd=6 12SE; 1.5%
   improvement). d=26 with shuffle+cosine pushed further to 1.352 at
   12SE — but with deeper trie + different schedule, not isolated.
5. **pd=1 keeps descending well past 100 SE.** From the 100SE log,
   the slope is still negative, just shallow (~0.0002/SE). Consistent
   with `project_agpt_undertrained.md`'s pd=1 plateau finding at SE≈110
   on Shakespeare.

## What's missing

- **The 7.5-hour 300-SE run.** No log preserved. Wall math (≈ 27 000s
  at d=32 pd=1 ≈ 82 s/SE) suggests it was d=32 pd=1 ~330 SE. Its
  trajectory is gone. The closest reproducible substitute: the d=32
  pd=1 110 SE Gutenberg run from 2026-05-10 at PPL@32 = 4.90.
- **Per-run held-out PPL.** Models lived in /tmp and are gone.
  Trajectories above are training loss only.
- **Mass-weight setting.** Not surfaced in log headers. The CLI may
  have used various `--mass-weight` values across runs that we cannot
  now distinguish. To be repaired by re-running with explicit logging.
- **Curriculum + shuffle ablations.** Several `curric*` and `warmup_pd*`
  logs exist in /tmp from 2026-05-09 early morning but are not
  catalogued here yet.

## Saved files

```
logs/agpt_g5m_d12_pd6_6se.log
logs/agpt_g5m_d16_pd0_6se.log       logs/agpt_g5m_d16_pd0_12se.log
logs/agpt_g5m_d16_pd1_12se.log      logs/agpt_g5m_d16_pd1_100se.log
logs/agpt_g5m_d16_pd2_6se.log       logs/agpt_g5m_d16_pd2_60se.log
logs/agpt_g5m_d16_pd4_6se.log       logs/agpt_g5m_d16_pd4_12se.log
logs/agpt_g5m_d16_pd6_1se.log       logs/agpt_g5m_d16_pd6_6se.log
logs/agpt_g5m_d16_pd6_12se.log      logs/agpt_g5m_d16_pd6_cos_6se.log
logs/agpt_g5m_d18_pd6_6se.log       logs/agpt_g5m_d18_pd6_12se.log
logs/agpt_d26_pd6_shuf_cos_12se.log
logs/agpt-gut-d16-pd1.log           (30SE; mass-weight=off)
logs/agpt-gut-d32-pd1.log           (30SE; mass-weight=off)
logs/agpt-gut-d32-pd1-110.log       (110SE; mass-weight=log; matched-context anchor)
```

## Lesson

This writeup is a recovery, not a normal experiment record.
`feedback_persist_results.md` (memory) was created 2026-05-10 to
prevent the same failure mode: every training run must land in
`rnd/<experiment>/logs/` *before* its result is reported, not after.
