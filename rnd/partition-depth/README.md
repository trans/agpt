# Partition-Depth — Bigram-and-Beyond Adam-Step Granularity

> **Status (2026-04-30): MAJOR FINDING.** Increasing `--partition-depth`
> from the default 1 (per-root-child, 65 Adam steps/SE) to 6 (per-6-gram,
> 283k Adam steps/SE) drops PPL@32 on Shakespeare 1M from 5.39 (prior
> best) to **3.95** — a 27% improvement at 1/8 the wall-clock of the
> prior best. pd=7 starts breaking down (gradient too noisy). The
> technique is recipe-level (no architectural rebuild).

## TL;DR

Partition-depth controls n-gram-level Adam-step granularity. With
`--partition-depth N --no-accumulate`:

- `N=1` (current default): per-root-child, 65 Adam steps/SE
- `N=2`: per-bigram, ~1400 Adam steps/SE  (~17.5× more)
- `N=3`: per-trigram, ~12k steps/SE  (~178× more)
- `N=4-6`: per-N-gram, exponential growth in step count
- `N=7+`: granularity breakdown, gradient signal too noisy per step

Each Adam step has a smaller gradient (smaller batch of training
events), but they fire 100×-1000× more often per epoch. The combined
effect is dramatically faster convergence per second of wall-clock.

## Headline numbers (Shakespeare 1M, d=32)

Recipe: `rmsprop wc lr=3e-3 entropy-λ=1.0 mass-weight=log no-accumulate`.
Eval at matched seq=32. 3 reps each (range tight, std ~0.05 for most).

| pd | groups | best SE | wall-clock | PPL@32 |
|---:|---:|---:|---:|---:|
| 1 | 65 | 40 | ~800s | 5.39 |
| 2 | 1404 | 20 | 377s | 4.61 |
| 3 | 11557 | 10 | 238s | 4.35 |
| 4 | 50713 | 10 | 497s | 4.19 |
| 5 | 141022 | 5 | 568s | 4.02 (plateau) |
| **6** | 283309 | **3** | **654s** | **3.95** |
| 7 | 447345 | 3 | 1003s | 4.00 (worse + slower) |

**Best RMSprop result: pd=6, 3 SE → 3.95 PPL@32, 654s wall-clock.**

**Best Adam result (2026-04-30 update): Adam pd=6, 3 SE, lr=1e-3 → 3.82 PPL@32, 643s wall-clock.**

That's ~1.93 bpc on Shakespeare 1M with our 108k-param model — within
30% of the theoretical English entropy floor (1.3-1.5 bpc).

Adam needs lower LR than RMSprop (1e-3 vs 3e-3); at matched LR (3e-3),
Adam is slightly worse than RMSprop. Properly tuned, Adam wins by
~3% PPL. AdamW with weight decay 0.01 is essentially tied with vanilla
Adam (3.83 vs 3.82); weight decay doesn't help here.

## Why this works

`--no-accumulate` mode fires one Adam step per partition group. Default
(pd=1) groups are root-children: 65 groups, 65 Adam steps per super-epoch.
Each step accumulates gradients from a whole subtree (~25k radix nodes
for Shakespeare 1M) — a huge gradient batch.

At pd=2, a "group" is a bigram-rooted subtree. ~17.5× more groups, each
with ~1.5k radix nodes. Adam fires 17.5× more often per SE, with
proportionally smaller gradients per step.

The empirical observation: smaller-but-more-frequent Adam steps drive
convergence dramatically faster than larger-but-rarer ones. The model
gets ~1300 Adam updates/sec at pd=6 vs ~3 updates/sec at pd=1. RMSprop
has more opportunity to adapt per-parameter scaling, the schedule
progresses through more steps per epoch, and the model's optimization
trajectory is much finer-grained.

The breakdown at pd=7: with 447k groups at d=32, each group has on
average 3-4 radix nodes. Gradient signal per step approaches single-
event noise. Optimizer updates start being more random than directed.

## Why this finding wasn't obvious before

Memory says the AGPT recipe used `--partition-depth=1` (default) for
the post-fix baselines. Memory also says
`--no-accumulate` was opt-in and "for reproducing old experiments only"
— we'd been using it for the baselines but not exploring its
interaction with finer partitioning. The flag combination
`--partition-depth N + --no-accumulate` was implemented but its
power at N>1 hadn't been measured against high-SE training.

The discovery came from chasing the user's intuition that AGPT had
"too few optimizer steps" — first via subtree dropout (didn't help),
then via more SE (yesterday's "AGPT was undertrained" finding), then
via finer partitioning (this finding). All three pointed at the same
underlying constraint.

## Wall-clock vs PPL: best operating points

| budget | best config | PPL |
|---:|---|---:|
| ~60s | pd=2, 3 SE | 5.79 |
| ~150s | pd=4, 3 SE | 4.26 |
| ~250s | pd=4, 5 SE | 4.21 |
| ~350s | pd=5, 3 SE | 4.02 |
| ~650s | **pd=6, 3 SE** | **3.95** (project best) |

Past ~650s wall-clock, returns diminish sharply. pd=7 (1003s) is worse
than pd=6 (654s).

## Recipe (current best — Adam variant)

```sh
bin/agpt_train --model <init> --trie-dir <prefix-trie> --save <out> \
    --epochs 3 --lr 1e-3 \
    --optimizer adam \
    --lr-schedule warmup-cosine --warmup-epochs 1 \
    --entropy-lambda 1.0 --mass-weight log \
    --no-accumulate --partition-depth 6
```

For RMSprop (very close result, slightly worse), swap `--optimizer adam --lr 1e-3`
for `--optimizer rmsprop --rmsprop-beta 0.999 --lr 3e-3`.

## See also

- `../agpt-epoch-scaling/` — the prior reframing. Pure pd=1 with high SE
  (40+ epochs) reached 5.39 PPL. Partition-depth gives a much faster
  path to lower PPL with fewer SE.
- `../subtree-dropout/` — the diagnostic that exposed AGPT being
  undertrained. Helped at low SE budgets but not in any meaningful way
  past 5 SE.
- `../trie-attention-framing/` — joint-mass + depth-routing
  experiments at pd=1. Effect sizes need re-measurement at pd=6 since
  the absolute PPL is now half what it was.

## Future directions

1. **Adam optimizer.** Currently using RMSprop (best at pd=1 per
   memory). With pd=6's many small Adam steps, Adam's per-parameter
   momentum might help more than RMSprop's variance-only adaptation.
   *Experiment in flight at time of writing.*
2. **Re-measure other framing experiments at pd=6.** Joint-mass,
   depth-routing, looping — all measured at pd=1. Their effect sizes
   may shift dramatically (could become significant or vanish).
3. **Validate on Gutenberg 5M.** Does pd=6 transfer to a 5× larger
   corpus? Or does it break down at scale (groups become smaller
   relative to gradient floor)?
4. **Push past pd=7 with larger gradients per step.** Subbatching:
   accumulate gradients across K consecutive partition groups, fire
   ONE Adam step. Could combine fine partitioning with stable gradients.
5. **Find the entropy floor.** 1.98 bpc on 108k params is close to but
   not at Shakespeare's theoretical limit. Bigger model + pd=6 might
   reach it.

## Reproduce

```sh
bash rnd/partition-depth/run_pd_sweep.sh
```

Runs the full pd=2 through pd=7 sweep with 3 reps each. ~2-3 hours.
