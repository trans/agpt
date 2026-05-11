# AGPT Epoch Scaling — The Undertraining Discovery

> **Status (2026-04-29): MAJOR FINDING.** AGPT was severely undertrained at the
> "standard" 3 super-epoch budget across the entire prior project history.
> Pushing to 20-40 SE drops PPL@32 from 10.83 to **5.39** — half the prior best.
> This rewrites several earlier conclusions; see `findings.md` for the full
> story. Recipe is unchanged from post-fix-baseline; only the epoch count
> changed.
>
> **Update (2026-04-30):** further investigation showed pd=1 (root-child
> partition) was even more under-stepped than just being under-epoched.
> Setting `--partition-depth 6 --no-accumulate` (per-6-gram Adam steps)
> drops PPL@32 to **3.95** at just 3 SE — 27% lower than this directory's
> best of 5.39, in 1/8 the wall-clock. See `../partition-depth/` for the
> follow-up finding. The "AGPT was undertrained" story here is correct
> but partial — the deeper issue was Adam-step granularity.

## TL;DR

| SE | PPL@32 mean (3 reps) | Prior project context |
|---:|---:|---|
| 3 | 10.82 | "the AGPT plateau" — used as standard for all prior comparisons |
| 7 | 8.83 | Beats SGD seq=32 10k steps (9.77) |
| 10 | 8.24 | |
| 15 | 7.21 | |
| 20 | 6.59 | |
| 30 | 5.96 | |
| 40 (n=1) | **5.39** | Half of prior project best PPL@32 |

The marginal improvement per SE is slowing (−0.6/SE in the 3-5 range vs
−0.06/SE in the 30-40 range) but **had not plateaued at 40 SE.** Wall-clock
cost: ~10 minutes for 40 SE on Shakespeare 1M d=32.

## What this rewrites

1. **"AGPT loses to SGD on PPL"** (memory: project_compact_kv_cache.md, etc.)
   — REVERSED. AGPT 5 SE already beats SGD seq=32 10k steps. AGPT 40 SE
   substantially beats it (5.39 vs 9.77).
2. **"Per-fix recipe gives ~12.79 PPL at d=32"** (memory: project_wkwv_bias_fix.md)
   — that was at 3 SE. At 20+ SE the recipe gives <7 PPL.
3. **"Joint-mass per-position improvement of ~1%"** (rnd/trie-attention-framing/)
   — measured under undertrained AGPT. At high-SE the absolute numbers shift
   so much that the joint-mass effect's relative size and significance
   changes; needs re-measurement under proper training.
4. **"Subtree dropout helps AGPT" (~6%)** (rnd/subtree-dropout/) — the help
   shows at mid-SE budgets and disappears at high-SE budgets. Dropout is a
   regularization aid for low-budget training, not a fundamental improvement.
5. **"Smoothness-trap hypothesis from hybrid AGPT-SGD"** (project_agpt_sgd_interleave.md)
   — AGPT wasn't stuck in a smooth valley; it was simply stopped before
   convergence. Hybrid experiments showed AGPT plateauing at 10.83 because
   each AGPT round was 1 SE — undertrained.

## Caveat: context-length specialization

PPL@128 INCREASES with more SE, even as PPL@32 decreases:

| SE | PPL@32 | PPL@128 |
|---:|---:|---:|
| 3 | 10.82 | ~12.67 |
| 10 | 8.24 | 15.18 |
| 20 | 6.59 | 28.54 |

The model becomes a "seq=d specialist" — better at its trained context length,
worse at extrapolating. So AGPT trained for many SE handles seq=32 inputs
beautifully but NOT seq=128 inputs. For applications needing variable context
length, SGD's broader-context training still wins. This is consistent with the
RoPE positions 0..d-1 being the only ones AGPT trains; positions past d are
extrapolated and the extrapolation gets less coherent as the trained range
becomes more specialized.

## Recipe (unchanged from post-fix-baseline)

```sh
bin/agpt_train --model <init> --trie-dir <prefix-trie> --save <out> \
    --epochs <SE> --lr 3e-3 \
    --optimizer rmsprop --rmsprop-beta 0.999 \
    --lr-schedule warmup-cosine --warmup-epochs 1 \
    --entropy-lambda 1.0 --mass-weight log --no-accumulate
```

Only change: increase `--epochs` from 3 (the prior standard) to 20+.

## Reproduce

```sh
bash rnd/agpt-epoch-scaling/run_se_sweep.sh
```

This runs 3 SE through 40 SE × 3 reps each on Shakespeare 1M d=32, evaluating
at matched seq=32. Total wall-clock ~30-45 minutes.

## See also

- `rnd/subtree-dropout/` — where the discovery emerged. Dropout's apparent
  ~6% improvement at 5 SE was real for that budget but largely vanishes at
  high-SE training. The dropout experiment's diagnostic value was in
  exposing the undertraining, not the dropout itself being a major fix.
- `rnd/trie-attention-framing/` — joint-mass and depth-routing experiments
  measured under (now known to be) undertrained AGPT. Their effect sizes
  may shift with proper training; needs re-evaluation.

## Future directions

1. **Find the plateau.** Push to 80-100 SE to see where AGPT actually
   converges. Currently the curve at 40 SE is still descending at ~−0.06 PPL/SE.
2. **Re-measure framing experiments at 20 SE.** Joint-mass per-position,
   depth-routing, etc. — do they still help? Do their effect sizes change?
3. **Scaling.** Apply this finding to larger corpora (Gutenberg 5M),
   different d values, and larger model architectures.
4. **Why does AGPT specialize so heavily?** Investigate whether the
   positional embeddings can be made to extrapolate better (RoPE base
   tuning, NTK-aware extension, etc.). This would let AGPT compete with
   SGD on seq=128 eval.
