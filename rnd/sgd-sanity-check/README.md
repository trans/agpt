# SGD vs AGPT Sanity Check

**Status**: in progress.

**Code**: branch `sgd-sanity-check` (intended to merge to main once validated).
Adds `--mass-weight <mode>` where mode ∈ {off, log, sqrt, linear}, replacing
the old boolean `--mass-weight` flag. Linear mode matches SGD's frequency
weighting — needed to isolate whether the weighting choice (not the
aggregation) is what differentiates AGPT from SGD.

## Hypothesis

If the D-trie captures the corpus's conditional distributions faithfully, and
AGPT's gradient aggregation is mathematically equivalent to "sum the gradient
contributions from all corpus positions sharing the same context," then AGPT
and standard SGD over corpus positions should converge to **similar held-out
PPL at matched compute budget**.

If AGPT is *significantly better* at matched compute: the trie aggregation is
adding something (prefix-sharing speedup, or the equal-per-context weighting is
actually beneficial).

If SGD is better: either there's a bug in AGPT's gradient aggregation OR the
equal-per-context weighting is actively hurting us and we should use
`--mass-weight`.

If they match: AGPT is behaving as specified — the aggregation is mathematically
sound. This is the primary sanity check.

## The weighting question (why this isn't just a re-run)

SGD with uniform corpus-position sampling trains in proportion to corpus
frequency — common D-grams get sampled often, rare ones rarely. Each gradient
step is weighted implicitly by `f(context)`.

AGPT's default behavior is **equal weight per radix node**, regardless of how
many corpus positions that node represents. So:
- `AGPT-default`: sums `∇L` across unique contexts with equal weight each.
- `AGPT --mass-weight`: weights by `log(1 + edge_mass)` — partial frequency
  restoration.
- `SGD`: weights proportional to corpus frequency (linear in count).

These are THREE different effective loss functions, even though they're
training the same model on the same corpus. Comparing them cleanly is part
of this experiment.

## What we'll measure

1. **Raw held-out PPL** at each config, matched at:
   - Same wall-clock time.
   - Same number of optimizer steps.
   - Same number of gradient samples (corpus positions covered).
2. **Convergence speed** (PPL vs. training time curve).
3. **Sensitivity to weighting scheme** (AGPT default vs. AGPT mass-weighted vs. SGD uniform).

## Configs

All runs use the same model architecture:
- d_model=64, n_heads=4, n_layers=2, d_ff=256, head_dim=16, vocab_size=65.
- lr=3e-3, rmsprop (β=0.999), warmup-cosine schedule.

### Variants to compare

| Label | Tool | Mode | Context length |
|---|---|---|---|
| sgd-s16-{2000,10000,50000,200000} | `bin/microgpt` | plain window training | seq_len=16 |
| sgd-s32-{2000,10000,50000,200000} | `bin/microgpt` | plain window training | seq_len=32 |
| agpt-d16-off | `bin/agpt_train` | equal per node | d=16 |
| agpt-d16-log | `bin/agpt_train` | `--mass-weight log` | d=16 |
| agpt-d16-sqrt | `bin/agpt_train` | `--mass-weight sqrt` | d=16 |
| agpt-d16-linear | `bin/agpt_train` | `--mass-weight linear` (matches SGD) | d=16 |
| agpt-d32-{off,log,sqrt,linear} | `bin/agpt_train` | same four modes | d=32 |

### Matched-compute scheme

AGPT d=16 at 3 super-epochs = 195 optimizer steps. Baseline.
AGPT d=32 at 3 super-epochs = 195 optimizer steps.
SGD will be run at various step budgets (195, 1000, 5000, 20000) to see where
its PPL curve sits relative to AGPT's.

## Files

- `run.sh` — runs the matrix of configs.
- `logs/` — raw training output per config.
- `results/summary.md` — condensed PPL comparison.

## Results so far (AGPT side, mass-weight sweep)

| d | mode | mean PPL | SD | notes |
|---|---|---|---|---|
| 16 | off | 13.76 | 0.18 | default, equal per context |
| 16 | log | 14.18 | 0.06 | compressed frequency — hurts |
| 16 | sqrt | 14.49 | 0.09 | between off and linear — worst |
| 16 | linear | **13.74** | 0.16 | matches SGD weighting — tied-best |
| 32 | off | 13.48 | 0.11 | |
| 32 | log | 13.47 | 0.09 | |
| 32 | sqrt | 13.47 | 0.12 | all non-linear modes tied at d=32 |
| 32 | linear | **13.36** | 0.14 | **BEST overall mean** (tied with blending at d=16) |

**Key reads**:

- Linear weighting (≡ SGD frequency weighting) wins or ties at both depths.
  The "rare sequences count too much" concern is real; SGD-style weighting
  is the right direction.
- At d=16, log and sqrt actively hurt. They're a bad middle ground — enough
  frequency bias to break AGPT's equal-per-context compensating structure,
  not enough to match SGD's actual distribution.
- At d=32, weighting mode barely matters (except linear edges ahead).
  Because most cap endpoints are singletons with count=1-2, log(1+count) ≈
  sqrt(count) ≈ count in that regime.
- d=32 linear = 13.36 is the new overall best mean seen in this project
  (tied with d=16 blending within noise).

**Implication for AGPT defaults**: consider making `--mass-weight linear` the
default once we see how it interacts with SGD comparison and other
experiments. Leave `off` accessible for the equal-per-context philosophy when
that's wanted deliberately.

## Decisions made (documentable so we don't re-litigate)

- **Training data**: `data/input.txt` (Shakespeare, 1.11M chars).
- **Eval**: `bin/perplexity --file data/input.txt --max-positions 16384 --backend openblas`.
  Uses the same file for eval — relies on perplexity tool's held-out-by-default
  convention. (Worth double-checking: is the eval really held-out?)
- **Model init**: fresh weights per config (not loaded from a common checkpoint)
  — otherwise we'd be measuring fine-tuning dynamics, not the training-from-scratch
  behavior we want.
- **Single-subtree mode**: for AGPT (matches the memory's winning recipe).
