# SGD Ceiling

**Status**: closed

**Trainer note**: post-fix.

**Code**: commits `149b926` and `2af0a80` (2026-04-24 to 2026-04-25).

## Hypothesis

If AGPT beats plain SGD, how much of that gain really comes from subtree
aggregation, and how much comes from optimizer/recipe differences?

This experiment tried to separate:

- vanilla window-trained SGD via `bin/microgpt`
- L4 path-sampling with SGD inside `bin/agpt_train`
- post-fix AGPT subtree training with the stronger optimizer recipe

## Setup

- corpus: Shakespeare
- window baselines: `bin/microgpt`
- AGPT-side comparison path: `bin/agpt_train` with L4 sampling
- headline depth: `d=32`
- post-fix AGPT reference: deterministic subtree run, mean PPL `12.79`

## What was measured

Two complementary sweeps were run.

### 1. Real SGD window training

Constant-LR window training via `bin/microgpt`:

- seq=16
- seq=32
- later-added reference logs also for seq=64 and seq=128

### 2. L4 path-sampling with SGD

Because `bin/microgpt` did not expose LR schedules, L4 path-sampling in
`bin/agpt_train` was used to probe whether a warmer / decayed SGD recipe could
close the gap.

## Results

Headline results from commit `149b926`:

### Real SGD window training

| config | best reported PPL |
|---|---:|
| seq=16, const `lr=3e-4` | 17.02 |
| seq=32, const `lr=3e-4` | **14.72** |

The added logs in commit `2af0a80` also preserved longer-context reference
points used in later comparison:

| config | reported PPL |
|---|---:|
| seq=64, const `lr=3e-4`, 2000 steps | 12.45 |
| seq=128, const `lr=3e-4`, 2000 steps | 10.88 |

### L4 + SGD sweep at d=32

Best single-seed L4+SGD result:

- `lr=1e-1`: PPL `14.93`

Three-seed result at that same LR:

- mean `15.18`
- min `15.05`

So cosine decay / L4 did **not** beat the simpler real-SGD baseline of `14.72`.

## Interpretation

The commit's decomposition was:

| method | PPL |
|---|---:|
| real SGD window baseline | 14.72 |
| L4 + Adam + matched recipe | 13.12 |
| L4 + AGPT KL endpoint loss | 13.13 |
| full AGPT subtree training | 12.79 mean / 12.48 min |

The reading attached to those numbers was:

- optimizer/recipe choice explains the majority of the gap to SGD
- AGPT's subtree aggregation adds a smaller but real extra gain
- KL-at-endpoint versus plain CE was basically a null result here

In the wording of `149b926`:

- Adam over SGD: about `1.6` PPL
- full subtree aggregation: about `0.33` PPL

## Conclusion

This experiment closed with a more modest claim for AGPT:

- AGPT's edge over plain SGD on Shakespeare is real
- but most of that edge came from the stronger optimization recipe
- subtree aggregation still contributed, just less dramatically than the
  earlier narrative suggested

## Reproduction note

No dedicated driver script was preserved in this directory. The logs and the
commit summaries are the primary record. If revisited, rebuild the sweep from
the filename matrix plus commit `149b926`.
