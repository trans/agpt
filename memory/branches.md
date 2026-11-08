# Branches, tags and worktrees

*Updated 2026-11-07*

`main` is at `7ef76d6`, plus `a6f3c84`, which ignores `.ai/`. The `origin`
remote tracks `main` and `agpt-cap-recurrence`. Every line below is
**Closed**: no branch carries active development as of 2026-11-07. The
active work (gradient-population, context-tree) is on `main`.

## Not merged: the record is a stub `rnd/` README plus a tag

| Branch | Tip | Tag | What it holds |
|---|---|---|---|
| `worktree-linear-recurrence` | `f5497d5` (2026-09-28) | `exp/linear-recurrence-final` | linear, GRU and tanh recurrent trainers; overlay trainers; the vanilla attention trainer; the 2026-06-11 closure memo; a KN sweep. It is the GRU-on-prefix-trie baseline for the context-tree A/B |
| `slot-selection` | `2c4cd65` (2026-09-28) | `exp/slot-selection-final` | backoff slot selection; closed 2026-05-31 |
| `precondition` | `df75f96` (2026-06-02) | `exp/precondition` | preceding-context encoder; archived, not merged |
| `agpt-cap-recurrence` | `b760526` (2026-05-30) | `exp/cap-recurrence` | cap-recurrence, closed |
| `agpt-root-loop` | `ce4cc3e` (2026-04-21) | `exp/root-loop` | root-loop virtual tree, K>1 negative |
| `agpt-packed-varlen` | `9019fe8` (2026-04-15) | `exp/packed-varlen` | packed variable-length forward |
| `agpt-partition-kv-scoping` | `bc742de` (2026-04-22) | `exp/partition-kv-scoping` | partition-scoped KV cache plan |

`exp/linear-recurrence` and `exp/slot-selection` are the earlier tags. The
`-final` tags were added rather than moving the old ones (`4355c02`).

## Merged into `main` (no unique commits)

- `sampled-node-phase-rope`
- `successor-prefix-attention`
- `tanh-recurrence`
- `cudax-growth`
- `cudax-parity`

## Named in notes but gone

- **`ancestor-backward`:** named in gradient-population Exp 8. Its work is on
  `main` (`aa1b6ac`..`88c9335`).
- **`worktree-masked-lm-combiner`:** named in per-machine memory. It and its
  results are not in git anywhere.

## Worktrees

The development machine has checkouts of five of these branches
(`git worktree list`):

- `sampled-node-phase-rope`
- `slot-selection`
- `successor-prefix-attention`
- `tanh-recurrence`
- `worktree-linear-recurrence`

None of them holds uncommitted changes to tracked files. Three hold untracked
run outputs (checkpoints, run directories) that were left out of git on
purpose. Leave them alone unless the user asks. They are the only copies of
some checkpoints, for example the vanilla-attention runs
([open-lines.md](open-lines.md) §5).
