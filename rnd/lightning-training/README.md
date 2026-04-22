# Lightning Training — empirical

Design doc: `notes/agpt/lightning-training.md`.

## Hypothesis

Stochastic variable-depth subtree sampling (L3 mass-weighted walk) can match
or beat the deterministic per-root-child sweep at matched total optimizer
steps. L3's stop-probability `p_stop` controls the expected sample depth:
small p_stop → deep/narrow samples, large p_stop → shallow/broad samples.

## Reference baselines

| config | PPL mean | source |
|---|---|---|
| d=8, 3 SE, 65 steps/SE, per-root-child, lr=3e-3 RMSProp | 17.99 | `rnd/radix-saturation/logs` |
| d=16, 3 SE, 65 steps/SE, per-root-child, lr=3e-3 RMSProp | 14.59 | `rnd/radix-saturation/logs` |
| d=32, 3 SE, 65 steps/SE, per-subtree-file, lr=3e-3 RMSProp | 13.40 | `rnd/radix-saturation/logs` |

d=16 global-radix won't fit KV cache (9.5GB). Initial sweeps live at d=8
where one full run takes ~5 sec. d=16 work will require adopting the
per-subtree file format (deferred — each subtree view is self-contained
so L3 within a subtree still works).

## Experiments

| script | what |
|---|---|
| `run_d8_pstop_sweep.sh` | L3 p_stop ∈ {0.2, 0.3, 0.5}, 65 steps × 3 SE matched to baseline |

### Result: matched-budget p_stop sweep (commit 20e8121)

| config | mean PPL | min | max | spread |
|---|---|---|---|---|
| L3 p_stop=0.2, 65×3 | 20.46 | 19.26 | 22.03 | 2.76 |
| L3 p_stop=0.3, 65×3 | 20.12 | 18.40 | 22.87 | 4.47 |
| L3 p_stop=0.5, 65×3 | 20.09 | 19.41 | 20.96 | 1.55 |
| **baseline det. 65×3** | **17.99** | — | — | — |

All Lightning configs lose to the deterministic baseline by ~2.1 PPL at
matched step count. Consistent with the design doc's hypothesis:
stochastic sampling at matched budget without LR retuning underperforms.

Observations:
- Variance anti-correlates with p_stop. Smaller p_stop = deeper, lower-mass
  samples → flakier stochastic corpus coverage.
- All runs train (no divergence). Infrastructure verified.
- Best single run (p_stop=0.3, seed=43): 18.40 PPL — within 0.4 of baseline.
  Suggests LR tuning or a higher step budget could close the gap.

### Result: extended step-budget + L1/L2 comparators (same commit)

| config | mean PPL | min | max | spread |
|---|---|---|---|---|
| **baseline det. 65×3** | **17.99** | — | — | — |
| L2 uniform-rc s65×3 | **19.54** | 18.59 | 21.12 | 2.53 |
| L3 p_stop=0.5 s65×3 | 20.09 | 19.41 | 20.96 | 1.55 |
| L3 p_stop=0.5 s130×3 | 25.83 | 21.01 | 31.17 | 10.16 |
| L3 p_stop=0.5 s260×3 | 26.13 | 24.71 | 28.45 | 3.74 |
| L1 uniform-all s65×3 | 28.55 | 28.37 | 28.90 | 0.53 |

Key findings:
- **L2 is the closest Lightning variant to baseline** (19.54 vs 17.99). L2
  samples root-children uniformly with replacement — essentially the
  deterministic sweep with stochastic noise. Expected.
- **L1 is disastrous** (28.55). Uniform sampling across 845k radix nodes
  skews heavily toward deep, low-mass leaves.
- **More L3 steps at the same LR makes PPL WORSE, not better**. 2× steps
  → 25.83 (from 20.09 at matched), 4× → 26.13. At fixed lr=3e-3 the extra
  stochastic steps are over-rotating the weights. LR has to come down.
- L3 was design-motivated by mass weighting; at this LR it lands between
  L1 (bad) and L2 (close), but can't beat baseline.

Next direction: LR sweep for L3 at higher step budgets — the "many small
steps need a smaller lr" story from earlier bigram work predicts
lr≈3e-4..1e-3 at 260 steps/SE. If L3 can't close the gap even with tuned
LR, the claim "mass-weighted stochastic sampling beats deterministic
uniform-over-root-children" is probably wrong.

## Run

```sh
rnd/lightning-training/run_d8_pstop_sweep.sh
```

Each cell dumps a config JSON (`*.json`), per-run logs (`*_r{N}.log`), and
prints `label  run N  PPL = X  time = Ys` to stdout.
