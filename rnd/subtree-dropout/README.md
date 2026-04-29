# Subtree Dropout — Per-Epoch Random Root-Child Masking

> **Status (2026-04-29): closed.** Modest improvement at mid-SE budgets,
> neutral or slightly negative at high-SE budgets. The experiment's most
> important byproduct was *exposing* that AGPT was severely undertrained
> — see `../agpt-epoch-scaling/` for the major finding that came out of
> this work. As a standalone training improvement, subtree dropout is
> useful but small.

## Hypothesis

AGPT does 65 deterministic root-child subtree firings per super-epoch
(one Adam step per subtree at d=32 with no-accumulate). Each SE walks
all 65 in the same order. Hypothesis: random per-epoch subset selection
would (a) increase trajectory variety across SE, (b) provide a
dropout-like regularization, (c) effectively give more "kinds of
optimization steps" without changing AGPT's loss objective.

The framing was: AGPT might be stuck in smooth gradient valleys due to
common-prefix gradient washout. Random subtree masking is a
within-AGPT-design way to inject variety, unlike the SGD interleave
hybrid (which cross-paradigm-breaks AGPT's optimizer state and loss).

## Implementation

Added env var `AGPT_SUBTREE_DROPOUT=p` (default 0). Each super-epoch
samples a keep mask: each of n_root_children root-children is kept
with prob (1-p), dropped with prob p. Skipped subtrees contribute no
training events that epoch. Different keep mask each epoch (LCG seeded).

If everything got dropped (rare at small p), the largest-mass subtree
is force-kept.

## Results

Recipe: rmsprop wc lr=3e-3 entropy-λ=1.0 mass-weight=log no-accumulate.
Eval at seq=32 (matched).

### Initial sweep (3 reps each)

| Config | PPL@32 mean | std |
|---|---:|---:|
| p=0, 3 SE (baseline) | 10.82 | 0.07 |
| p=0.3, 3 SE (fewer steps) | 11.49 | 0.22 |
| **p=0.3, 5 SE** | **10.13** | 0.07 |
| p=0.5, 6 SE (matched-step) | 10.77 | 0.01 |

The headline read at first: p=0.3 with 5 SE beats no-dropout 3 SE by
0.69 PPL. Compensating for the dropout's per-epoch reduction by adding
more epochs gave a real ~6% improvement.

### Follow-up: dropout helps at mid-SE, NOT at high-SE

| Config | PPL@32 mean | std |
|---|---:|---:|
| p=0, 5 SE | 9.64 | 0.05 |
| p=0, 7 SE | 8.83 | 0.13 |
| p=0, 10 SE | 8.24 | 0.15 |
| p=0, 15 SE | 7.21 | 0.16 |
| p=0.3, 5 SE | 10.13 | 0.07 |
| p=0.3, 7 SE | 9.38 | 0.06 |
| p=0.3, 10 SE | 8.62 | 0.09 |
| p=0.3, 15 SE | 7.69 | 0.13 |

At 5-10 SE: dropout adds nothing meaningful when compared to **just
training pure AGPT for the same number of epochs**.
- p=0 5SE 9.64 vs p=0.3 5SE 10.13 — pure AGPT WINS
- p=0 10SE 8.24 vs p=0.3 10SE 8.62 — pure AGPT WINS
- p=0 15SE 7.21 vs p=0.3 15SE 7.69 — pure AGPT WINS

The "p=0.3 5SE beats p=0 3SE" win was just "more epochs help" plus a
small variety bonus that didn't compound at higher SE. Dropout is at
best a regularization for low-budget training; it's not a fundamental
improvement to AGPT optimization.

### Combination with joint-mass

| Config | PPL@32 mean |
|---|---:|
| p=0.3, 5 SE | 10.13 |
| p=0.3, 5 SE + joint-mass per-position | **11.42** |

Joint-mass and subtree-dropout do NOT compound. Combining them gives a
WORSE result than either alone. The two mechanisms target different
modes (joint-mass weights events; dropout removes events) and the
interaction destabilizes both.

## Why this is "closed"

Subtree dropout's apparent ~6% improvement at 5 SE was an artifact of
comparing it to under-trained pure AGPT (3 SE). When you give pure AGPT
the same total epoch count, it wins. The diagnostic value of this
experiment was in *exposing* the undertraining — see
`../agpt-epoch-scaling/` for the consequence.

Subtree dropout itself has a narrow use case: regularization for very
low-SE training budgets where the model is at risk of subtree-order
artifact memorization. For any serious training run (10+ SE), it's not
a useful knob. Leaving the code in place behind the `AGPT_SUBTREE_DROPOUT`
env var (default off — no behavior change) for future experimentation.

## Recipe (default off)

```sh
AGPT_SUBTREE_DROPOUT=0.3 bin/agpt_train ... --epochs 5 ...
```

`AGPT_SUBTREE_DROPOUT_SEED` env var also available to control the LCG
seed for reproducibility.

## Reproduce

```sh
bash rnd/subtree-dropout/run_dropout_sweep.sh
```

Runs p=0/0.3/0.4/0.5 across 3-15 SE budgets × 3 reps. ~30 min.

## See also

- `../agpt-epoch-scaling/` — the major finding that emerged from this
  experiment's follow-up sweep. Pure AGPT 40 SE → 5.39 PPL@32, far
  below anything dropout achieves.
- `../trie-attention-framing/` — the depth-routing / joint-mass line.
  Joint-mass + dropout don't compound, as documented above.
