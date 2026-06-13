# Hybrid AGPT Experiments

Purpose: test whether deterministic whole-tree AGPT is best used as the broad
coverage phase, followed by a smaller stochastic refinement phase using the same
attention `f_theta` model.

The motivation is narrow: standalone stochastic traversal-stop training with
context-only ancestors now beats the clean pd1 baseline on held-out metrics, but
only after much more wall time. A hybrid run asks whether pd1 can do the cheap
global pass and stochastic sampling can do the final local refinement.

## Baselines

Clean pd1 baseline:

```text
rnd/stochastic-agpt/20260611T160456-d64l2-depth16-pd1-100ep
fixed-token PPL: 4.7929
rolling byte PPL: 5.3359
train wall: 582.1s
optimizer steps: ~6,500
```

Best standalone stochastic row so far:

```text
rnd/lightning-agpt/20260612T204543-d64l2-depth16-lightning-random-desc-u40000-r4-contextonly-co
fixed-token PPL: 4.7637
rolling byte PPL: 5.1657
train wall: 1999.2s
optimizer steps: 160,000
```

Important correction: that run's filename/config said random-descendants, but a
stale `bin/agpt_experiment` dropped `lightning.anchor_mode` from
`resolved_config.yml`. The trainer banner shows it actually ran
`anchor_mode=traversal-stop`. Literal uniform random-descendant sampling was
tested later and mostly selected tiny subtrees.

## Results

| run | actual sampler | LR | trained query passes | train wall | fixed PPL | rolling byte PPL | note |
|---|---|---:|---:|---:|---:|---:|---|
| `20260612T213817...` | traversal-stop | `3e-4` | 397.9M | 266.2s | 5.5280 | 5.7552 | stale orchestrator dropped intended `random-descendants`; too destructive |
| `20260612T215213...` | random-descendants | `3e-4` | 0.9M | 21.3s | 5.6295 | 6.2570 | literal uniform nodes, mostly tiny subtrees |
| `20260612T215441...` | traversal-stop | `3e-5` | 397.9M | 276.7s | 5.4332 | 5.7765 | gentler but still destructive |

## Interpretation

The simple hybrid did not work. Starting from the pd1-100 checkpoint and then
resetting Adam for a stochastic pass damages the solution even at `3e-5`.

The failure is still useful:

- Stochastic refinement is not plug-and-play with a trained pd1 checkpoint.
- Reset Adam state is probably part of the problem; the stochastic gradients may
  also be miscalibrated relative to the pd1 optimum.
- Literal uniform random-descendant sampling is not the policy that produced the
  good stochastic row. It gives too many tiny subtrees unless we add a
  size/coverage policy.
- If this hybrid line continues, the next credible version needs either
  optimizer-state continuity, a much smaller trust-region style update, or a
  sampler constrained to the same large-unit regime as traversal-stop.
