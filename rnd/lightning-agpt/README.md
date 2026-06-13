# Lightning AGPT

Status: research thread closed for now.

Conclusion: stochastic mini-tree batching is useful for update cadence and
memory, but it is not a decisive replacement for the standard pd1 whole-tree
AGPT baseline. The best stochastic traversal-stop/context-only row slightly
beats pd1 on held-out metrics, but costs much more wall time. A pd1-then-
stochastic hybrid fine-tune damaged the pd1 checkpoint rather than refining it.
Future AGPT v1 work should prioritize update geometry/calibration, Fisher or
trust-region style scaling, and suffix-side statistics over more sampler tweaks.

Purpose: recover stochastic optimizer cadence without giving up AGPT's
full-tree statistical objective. The full corpus trie remains the source of
truth; each optimizer update trains on a sampled, clustered mini-tree induced
from the full tree.

## Motivation

The fixed-depth control in `rnd/stochastic-agpt` showed that single-depth
training is not enough. The normal all-depth pd1 baseline remains materially
better than any one-depth objective:

| objective | rolling byte PPL | fixed-token PPL |
|----------:|-----------------:|----------------:|
| normal pd1 baseline, all depths | 5.3359 | 4.7929 |
| best single-depth rolling, depth 12 | 7.6428 | 7.0596 |
| best single-depth fixed, depth 16 | 8.7932 | 5.2318 |

That suggests the interior tree losses are useful. The problem is update
cadence: full-tree AGPT epochs produce too few optimizer steps for modern
optimizers. Lightning AGPT keeps all-depth mini-tree training but samples many
mini-trees from the full tree.

## Core Definition

Lightning AGPT is not ordinary sequence SGD. Each update:

1. Samples a clustered subset of paths/nodes from the full corpus trie.
2. Closes that subset under ancestors/prefixes so it forms a valid mini-tree.
3. Uses original full-tree masses and target distributions for included nodes.
4. Runs the normal AGPT forward/loss/backward over that mini-tree.
5. Applies one optimizer step to the global model weights.
6. Repeats with a new sampled mini-tree.

The mini-tree is temporary. The full tree is the fixed population/statistical
object.

## Non-Goals

- Do not sample paths proportional to mass by default. AGPT already uses mass in
  the loss; mass-proportional sampling risks double-counting.
- Do not collapse to independent fixed windows. The batch must preserve shared
  prefix structure and multi-depth supervision.
- Do not change the baseline full-tree path while developing this. Lightning
  should be behind explicit config fields or a new trainer mode.

## Sampling Shape

The first sampler should be clustered. Purely random endpoints would often split
near shallow bigram territory, producing skinny unrelated paths and little
in-batch tree aggregation.

Initial policy:

1. Start traversal at the root or a root child.
2. At each branching point, use a stop/continue draw to decide whether this
   prefix becomes an anchor.
3. If continuing, choose child branches structurally, not by mass.
4. From selected anchors, sample or expand descendant paths to max depth subject
   to a mini-tree query budget.
5. Close the sampled set under ancestors and deduplicate shared prefixes.

This is deliberately close to a Monte Carlo traversal over the tree while still
preserving local tree structure.

## Coverage

The hard part is giving every node a principled stake over time. The first
implementation should record exposure metrics before trying sophisticated
correction:

- active rows per update
- unique full-tree nodes touched
- coverage by depth
- coverage by root child
- coverage by endpoint-depth bucket
- repeat rate across recent updates

If structural sampling is too uneven, add an under-visited bias that is separate
from loss mass, for example child selection weighted by:

```text
1 / sqrt(1 + visit_count[child])
```

This is a coverage correction, not a mass-proportional objective.

## First Knobs

Expose these as config fields from the start:

| knob | meaning |
|-----:|---------|
| `lightning.enabled` | selects mini-tree training path |
| `lightning.seed` | deterministic sampler seed |
| `lightning.updates` | number of optimizer updates |
| `lightning.query_budget` | target query rows per mini-tree |
| `lightning.anchor_mode` | `traversal-stop` or `random-descendants` |
| `lightning.stop_p` | base stop probability during traversal |
| `lightning.stop_schedule` | optional depth-dependent stop schedule |
| `lightning.anchors_per_step` | anchors sampled while filling one mini-tree update |
| `lightning.sample_fanout` | random descendant branches sampled from each anchor |
| `lightning.repeats_per_sample` | optimizer steps to run on the same sampled mini-tree before sampling another |
| `lightning.coverage_bias` | `none` first; later `under_visited` |

Implemented in the first prototype:

- `lightning.enabled`
- `lightning.seed`
- `lightning.updates`
- `lightning.query_budget`
- `lightning.anchor_mode`
- `lightning.stop_p`
- `lightning.sample_fanout`
- `lightning.anchors_per_step`
- `lightning.repeats_per_sample`
- streamed sampled-unit training

Legacy aliases are still accepted:

- `lightning.fanout` aliases `lightning.sample_fanout`
- `lightning.paths_per_anchor` aliases `lightning.sample_fanout`
- `lightning.children_per_branch` aliases `lightning.anchors_per_step`

`random-descendants` is the deliberately simple random-node baseline: choose one
non-root radix node uniformly, include its ancestor path for cache correctness,
then train the full descendant subtree rooted at that anchor. It does not use
`query_budget`, `sample_fanout`, `anchors_per_step`, or `stop_p` to truncate the
sample. `train.chunk_queries` still controls CUDA chunk size only.

The trainer rejects configs that set both a canonical field and its legacy alias
to conflicting values.

Not implemented yet:

- explicit `anchor_mode`
- depth-dependent `stop_schedule`
- coverage-bias sampling
- exposure/coverage metrics

## First Experiment

Use the same model and held-out split as `rnd/stochastic-agpt`:

```text
d_model=64
n_layers=2
n_heads=4
d_ff=256
seq_len=16
max_depth=16
optimizer=Adam
lr=0.0015
```

Baseline to beat:

```text
normal pd1 baseline at 100 epochs:
rolling byte PPL: 5.3359
fixed-token PPL: 4.7929
train wall: 582.1s
optimizer steps: about 6,500
```

Initial Lightning target:

- hold wall time near the pd1 baseline
- produce many more than 6,500 optimizer steps if mini-trees are small enough
- preserve all-depth loss inside each mini-tree
- compare rolling byte PPL and fixed-token PPL at matched wall/update budgets

## Prototype Smoke

Config:

```text
rnd/lightning-agpt/d64L2-depth16-lightning-smoke.yml
```

Corrected smoke run:

```text
rnd/lightning-agpt/20260611T232319-d64l2-depth16-lightning-smoke-sampled
```

The corrected smoke confirmed that the trainer uses the sampled mini-tree plan:

| item | value |
|-----:|------:|
| sampled units / optimizer steps | 5 |
| query budget per sampled unit | 20,000 |
| total query positions | 100,044 |
| total node-visits | 19,324 |
| chunks/run | 10 |
| train wall | 1.5s |
| rolling byte PPL | 46.6326 |
| fixed-token PPL | 48.9042 |

The PPL is not meaningful as a quality result because this was only 5 optimizer
steps. The purpose was to verify the path: wrapper preflight validation,
Lightning plan construction, chunk planning, forward/backward, Adam step,
checkpoint save, rolling eval, and fixed-token eval.

Repeat smoke run:

```text
rnd/lightning-agpt/20260611T234157-d64l2-depth16-lightning-smoke-repeat2
```

This run used the same 5 sampled mini-trees but set
`lightning.repeats_per_sample=2`, so each mini-tree received two forward /
backward / Adam-update passes before advancing:

| item | value |
|-----:|------:|
| sampled mini-trees | 5 |
| repeats per mini-tree | 2 |
| optimizer steps | 10 |
| query budget per sampled unit | 20,000 |
| total trained query passes | 200,088 |
| total trained events | 15,854,156 |
| train wall | 1.5s |
| rolling byte PPL | 36.4366 |
| fixed-token PPL | 39.3464 |

The log confirms repeated updates in the form
`unit X/5 repeat 1/2` followed by `unit X/5 repeat 2/2`.

Streamed smoke run:

```text
rnd/lightning-agpt/20260612T005346-d64l2-depth16-lightning-stream-smoke
```

This run verifies the streamed implementation: the trainer builds one probe unit
for capacity/logging, then samples each mini-tree inside the update loop.

| item | value |
|-----:|------:|
| streamed sampled mini-trees | 5 |
| repeats per mini-tree | 2 |
| optimizer steps | 10 |
| cache compact slots | 983,047 |
| runtime estimate | 1.14GB |
| train wall | 1.6s |
| rolling byte PPL | 36.4117 |
| fixed-token PPL | 39.3187 |

The streamed smoke matches the earlier prebuilt repeat smoke closely while
removing the `updates`-scaled plan/cache memory behavior.

## First Short Run

Config:

```text
rnd/lightning-agpt/d64L2-depth16-lightning-u1000-q20k-r2.yml
```

Run:

```text
rnd/lightning-agpt/20260612T001917-d64l2-depth16-lightning-u1000-q20k-r2
```

This was the first non-smoke Lightning point:

| item | value |
|-----:|------:|
| sampled mini-trees | 1,000 |
| sample fanout | 16 |
| anchors per step | 1 |
| repeats per mini-tree | 2 |
| optimizer steps | 2,000 |
| query budget per sampled unit | 20,000 |
| total sampled query positions | 20,006,110 |
| total trained query passes | 40,012,220 |
| mean train loss | 2.3058 |
| train wall | 29.4s |
| rolling byte PPL | 7.5232 |
| fixed-token PPL | 8.4438 |

This is not competitive with the pd1 all-depth baseline yet, but it confirms the
Lightning path is very fast and can run many optimizer updates over sampled
mini-trees. The next question is whether sampler shape, LR schedule, query
budget, or coverage correction can close the quality gap.

Aborted 10x pre-streaming run:

```text
rnd/lightning-agpt/20260612T004110-d64l2-depth16-lightning-u10000-q20k-r2
```

This run was killed before training because the prototype prebuilt all 10,000
sampled units and sized runtime/cache from aggregate sampled plan totals. The
streaming implementation fixed that by sizing cache from global compact slots
and sampling/freeing each mini-tree inside the train loop.

Streamed 10x run:

```text
rnd/lightning-agpt/20260612T005519-d64l2-depth16-lightning-stream-u10000-q20k-r2
```

| item | value |
|-----:|------:|
| streamed sampled mini-trees | 10,000 |
| sample fanout | 16 |
| anchors per step | 1 |
| repeats per mini-tree | 2 |
| optimizer steps | 20,000 |
| query budget per sampled unit | 20,000 |
| total trained query passes | 400,117,834 |
| total trained events | 30,661,554,277 |
| mean train loss | 2.2350 |
| train wall | 282.5s |
| rolling byte PPL | 5.5939 |
| fixed-token PPL | 5.3889 |

This is much closer than the 1,000-unit point but still behind the pd1 all-depth
baseline, especially fixed-token PPL. The final LR decayed all the way to zero,
so LR schedule is now an obvious knob for this stochastic-update regime.

Constant LR comparison:

```text
rnd/lightning-agpt/20260612T010408-d64l2-depth16-lightning-stream-u10000-q20k-r2-constlr00075
```

Same streamed setup, but `lr_schedule=constant` and `lr=0.00075`.

| item | cosine-to-zero | constant 0.00075 |
|-----:|---------------:|-----------------:|
| optimizer steps | 20,000 | 20,000 |
| mean train loss | 2.2350 | 2.2435 |
| last 2k-step train loss | 2.2226 | 2.2283 |
| train wall | 282.5s | 269.5s |
| rolling byte PPL | 5.5939 | 5.6940 |
| fixed-token PPL | 5.3889 | 5.4366 |

Constant `0.00075` was worse on both train loss and held-out metrics. The issue
is not simply that LR hit zero; this setting undertrained early and did not make
up the gap late.

Cosine floor and smaller-batch comparison:

```text
rnd/lightning-agpt/20260612T012510-d64l2-depth16-lightning-stream-u10000-q20k-r2-cosfloor10
rnd/lightning-agpt/20260612T013026-d64l2-depth16-lightning-stream-u10000-q10k-r2-cosfloor10
rnd/lightning-agpt/20260612T024242-d64l2-depth16-lightning-stream-u20000-q10k-r6-cosfloor10
rnd/lightning-agpt/20260612T062224-d64l2-depth16-lightning-stream-u20000-q10k-r10-cosfloor10
rnd/lightning-agpt/20260612T072957-d64l2-depth16-lightning-stream-u20000-q10k-r6-dropnode80-cos
rnd/lightning-agpt/20260612T083109-d64l2-depth16-lightning-stream-u20000-q10k-r6-entgate10-cosf
rnd/lightning-agpt/20260612T085328-d64l2-depth16-lightning-stream-u20000-q10k-r6-entgate50-cosf
rnd/lightning-agpt/20260612T091556-d64l2-depth16-lightning-stream-u20000-q10k-r6-entgrad50-cosf
rnd/lightning-agpt/20260612T053624-d64l2-depth16-lightning-stream-u40000-q5k-r6-cosfloor10
rnd/lightning-agpt/20260612T055801-d64l2-depth16-lightning-stream-u40000-q5k-r4-cosfloor10
```

Both use `lr=0.0015`, `lr_schedule=warmup-cosine`, `min_lr_ratio=0.1`,
`sample_fanout=16`, and `anchors_per_step=1`.

| item | q10k/u10k/r2 | q20k/u10k/r2 | q40k/u10k/r2 | q10k/u20k/r2 | q10k/u20k/r6 | q10k/u20k/r10 | q10k/u20k/r6/drop80 | q10k/u20k/r6/ent10 | q10k/u20k/r6/ent50 | q10k/u20k/r6/entgrad50 | q5k/u40k/r4 | q5k/u40k/r6 |
|-----:|-------------:|-------------:|-------------:|-------------:|-------------:|--------------:|-------------------:|------------------:|------------------:|----------------------:|------------:|------------:|
| optimizer steps | 20,000 | 20,000 | 20,000 | 40,000 | 120,000 | 200,000 | 120,000 | 120,000 | 120,000 | 120,000 | 160,000 | 240,000 |
| query budget per sampled unit | 10,000 | 20,000 | 40,000 | 10,000 | 10,000 | 10,000 | 10,000 | 10,000 | 10,000 | 10,000 | 5,000 | 5,000 |
| repeats per sampled unit | 2 | 2 | 2 | 2 | 6 | 10 | 6 | 6 | 6 | 6 | 4 | 6 |
| target-node keep prob | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.8 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| entropy loss min scale | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.1 | 0.5 | 1.0 | 1.0 | 1.0 |
| entropy grad min scale | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.5 | 1.0 | 1.0 |
| total trained query passes | 198,827,441 | 400,118,547 | 800,096,882 | 396,605,272 | 1,192,651,624 | 1,997,278,613 | 954,283,222 | 1,192,107,636 | 1,196,936,439 | 1,189,619,641 | 799,964,728 | 1,200,484,590 |
| total trained events | 21,616,358,179 | 30,660,762,096 | 39,572,247,047 | 43,198,043,360 | 129,697,213,819 | 216,644,950,611 | 103,835,577,657 | 76,018,672,935 | 99,991,912,188 | 129,573,436,699 | 108,339,833,249 | 162,542,483,817 |
| mean train loss | 2.2981 | 2.2356 | 2.1436 | 2.2936 | 2.2855 | 2.2805 | 2.2857 | 2.4839 | 2.3664 | 2.2887 | 2.3110 | 2.3094 |
| final bucket train loss | 2.2874 | 2.2240 | 2.1315 | 2.2816 | 2.2731 | 2.2686 | 2.2657 | 2.4742 | 2.3523 | 2.2762 | 2.2885 | 2.2877 |
| train wall | 126.2s | 250.1s | 493.3s | 227.5s | 727.1s | 1107.3s | 708.5s | 738.7s | 665.7s | 720.5s | 504.7s | 769.2s |
| rolling byte PPL | 5.6932 | 5.5921 | 5.5660 | 5.4730 | 5.3037 | 5.3392 | 5.3739 | 5.6888 | 5.3208 | 5.3887 | 5.3371 | 5.3422 |
| fixed-token PPL | 5.4197 | 5.3617 | 5.3039 | 5.1783 | 4.9776 | 4.9511 | 5.0503 | 5.4387 | 4.9775 | 5.0645 | 4.9932 | 4.9780 |

The 10% LR floor gave a small fixed-token improvement over cosine-to-zero
without changing rolling byte PPL much. Halving `query_budget` was much faster
but clearly worse at the same update/repeat count. Doubling `query_budget`
improved PPL at the same optimizer-step count, but it also roughly doubled
trained query passes and wall time.

The normalized smaller-batch control is the important point: `q10k/u20k`
roughly matches `q20k/u10k` total trained query passes, runs slightly faster,
and is substantially better on held-out PPL. Smaller sampled units therefore
look promising when the update count is scaled up. Sampled train loss is not
comparable across these sampler shapes: `q10k/u20k` has worse sampled train loss
than `q20k/u10k` while producing better held-out PPL.

Increasing repeats from 2 to 6 on `q10k/u20k` improved both held-out metrics
again, reaching fixed-token PPL 4.9776. This is not normalized for total work:
it uses 3x the optimizer steps and trained query passes of `q10k/u20k/r2`, and
about 2.9x its training wall time. It does show that repeated local optimization
on the same sampled mini-tree is useful in this regime.

Increasing repeats further to 10 is mixed. It improves fixed-token PPL to
4.9511 and the sampled train loss is still edging down late, but rolling byte
PPL worsens to 5.3392 and wall time rises to 18.5 minutes of training. This is
probably past the practical repeat-count knee unless fixed-token PPL is the only
metric being optimized.

Target-node dropout at keep probability 0.8 did not help. It zero-weighted
random radix nodes' own query/loss rows while keeping the rows and ancestor
context in the graph. Despite ending with a lower final sampled train-loss
bucket than the no-dropout `q10k/u20k/r6` run, it was worse on both held-out
metrics: rolling byte PPL 5.3739 and fixed-token PPL 5.0503. A lighter dropout
rate might still be worth a check, but this first structural dropout point is
negative.

Entropy gating with `entropy_gate_min_scale=0.1` was also negative. It damped
deterministic rows and low-entropy endpoints by scaling their loss weight, but
it cut effective trained events from 129.7B to 76.0B and produced much worse
held-out metrics: rolling byte PPL 5.6888 and fixed-token PPL 5.4387. This
implementation appears too aggressive, especially because intra-edge rows are
treated as entropy 0. A gentler gate or endpoint-only gate would be a different
experiment.

A gentler entropy gate with `entropy_gate_min_scale=0.5` avoided the failure but
did not improve quality. It tied fixed-token PPL at 4.9775 while slightly
worsening rolling byte PPL to 5.3208. It also reduced trained events to 100.0B
and ran somewhat faster than baseline. This suggests the entropy signal is not
useless, but the current row-level gate is not a clear win.

Gradient-only entropy scaling with `entropy_grad_min_scale=0.5` preserved normal
loss/event accounting, but it also degraded held-out metrics: rolling byte PPL
5.3887 and fixed-token PPL 5.0645. This weakens the case for entropy as a
direct node movement controller in the current attention setup.

Halving the sampled unit again to `q5k/u40k/r6` did not improve the result. It
roughly matches the `q10k/u20k/r6` trained query passes, uses twice the optimizer
steps, and lands essentially tied on fixed-token PPL but worse on rolling byte
PPL. For this configuration, `q10k` looks like the better sampled-unit size than
`q5k`.

Reducing the `q5k/u40k` repeats from 6 to 4 improves rolling byte PPL slightly
but hurts fixed-token PPL. Its late train-loss buckets are also flat/noisy, not
cleanly descending. This weakens the hypothesis that `q5k/r6` only failed
because it was over-repeating the same mini-tree.

Correction: stale random-descendant labels:

```text
rnd/lightning-agpt/20260612T174136-d64l2-depth16-lightning-random-desc-u20000-r6-cosfloor10
```

This filename/config said `lightning.anchor_mode=random-descendants`, but the
then-current `bin/agpt_experiment` binary was stale and dropped `anchor_mode`
from `resolved_config.yml`. The trainer banner for this run shows
`anchor_mode=traversal-stop query_budget=20000`. The result should therefore be
read as another traversal-stop sampler row, not literal uniform random-descendant
full-subtree sampling.

| item | stale-label/u20k/r6, actual traversal | q10k/u20k/r6 |
|-----:|--------------------:|-------------:|
| optimizer steps | 120,000 | 120,000 |
| sampler query budget | 20,000 | 10,000 |
| total trained query passes | 2,399,665,541 | 1,192,651,624 |
| total trained events | 161,198,692,265 | 129,697,213,819 |
| mean train loss | 2.2112 | 2.2855 |
| train wall | 1666.4s | 727.1s |
| rolling byte PPL | 5.3539 | 5.3037 |
| fixed-token PPL | 5.0013 | 4.9776 |

This row is viable but not better than the q10k traversal sampler. It is not
evidence for literal random-descendant sampling.

Context-only ancestor closure:

```text
rnd/lightning-agpt/20260612T190826-d64l2-depth16-lightning-random-desc-u20000-r6-contextonly-co
rnd/lightning-agpt/20260612T204543-d64l2-depth16-lightning-random-desc-u40000-r4-contextonly-co
rnd/lightning-agpt/20260612T194009-d64l2-depth16-lightning-stream-u20000-q10k-r6-contextonly-co
```

The first Lightning implementation made ancestor-closure rows trainable. The
context-only version still includes those rows to populate/read the K/V cache,
but zeros their query weights. Anchors and sampled descendants remain trainable.

| item | stale-label/trainable anc | stale-label/u20k/r6/context anc | stale-label/u40k/r4/context anc | q10k/trainable anc | q10k/context anc |
|-----:|--------------------------:|-------------------------------:|-------------------------------:|-------------------:|-----------------:|
| optimizer steps | 120,000 | 120,000 | 160,000 | 120,000 | 120,000 |
| trained query passes | 2,399,665,541 | 2,386,090,689 | 3,181,425,235 | 1,192,651,624 | 1,184,754,699 |
| trained events | 161,198,692,265 | 111,729,261,425 | 148,907,627,536 | 129,697,213,819 | 81,137,713,672 |
| mean train loss | 2.2112 | 2.1151 | 2.1117 | 2.2855 | 2.1942 |
| train wall | 1666.4s | 1687.3s | 1999.2s | 727.1s | 721.2s |
| rolling byte PPL | 5.3539 | 5.2215 | 5.1657 | 5.3037 | 5.2962 |
| fixed-token PPL | 5.0013 | 4.8272 | 4.7637 | 4.9776 | 4.9720 |

The `stale-label/u40k/r4/context anc` row is the strongest stochastic result in
this thread so far, but it is a traversal-stop result. The context-only change
barely moves the q10k traversal sampler, while the larger traversal-stop
u40k/r4 row buys another small but clear gain: about `-0.064` fixed PPL and
`-0.056` rolling byte PPL for about `18.5%` more train wall than
`u20k/r6/context anc`.

Invalid smoke run:

```text
rnd/lightning-agpt/20260611T231817-d64l2-depth16-lightning-smoke
```

That run printed `lightning.enabled=true` but still built the normal pd1 plan
in the main `train-epoch` path. It trained 65 pd1 units and should not be used
as a Lightning result. The bug was fixed by routing the main trainer plan
through `build_lightning_training_plan_v2` when `lightning.enabled=true`.

## 2026-06-12 Synthesis

Lightning/stochastic mini-tree batching helps, but the benefit is modest and
metric-dependent. It is not a replacement for a stronger model/objective by
itself.

The clean deterministic baseline is the corrected pd1 all-depth run:

```text
rnd/stochastic-agpt/20260611T160456-d64l2-depth16-pd1-100ep
fixed-token PPL: 4.7929
rolling byte PPL: 5.3359
train wall: 582.1s
optimizer steps: ~6,500
peak combined runtime estimate: 1545.8 MB
```

The best balanced Lightning run is:

```text
rnd/lightning-agpt/20260612T024242-d64l2-depth16-lightning-stream-u20000-q10k-r6-cosfloor10
fixed-token PPL: 4.9776
rolling byte PPL: 5.3037
train wall: 727.1s
optimizer steps: 120,000
peak combined runtime estimate: 1136.4 MB
```

So, compared with pd1 at roughly comparable total query work:

- fixed-token PPL is worse by `+0.1847`
- rolling byte PPL is better by `-0.0322`
- train wall is `1.25x` longer
- optimizer steps are `18.5x` higher
- estimated runtime memory is `0.74x` of pd1, about `26%` lower

The performance/memory tradeoff is therefore mixed:

- Memory improves because each mini-tree chunk has a smaller runtime contract.
  The representative combined estimate drops from `1545.8 MB` to `1136.4 MB`.
- Wall time does not automatically improve. A cheap r2 Lightning run is much
  faster but underperforms; the r6 run that reaches the useful quality region is
  slower than pd1.
- The sampler buys optimizer cadence. That is the real win: many more updates
  without materializing or training the whole tree in each update.
- More updates are not free. r10 improves fixed-token PPL to `4.9511`, but
  rolling byte PPL regresses to `5.3392` and train wall rises to `1107.3s`.

Curated comparison:

| run | objective/sampler | fixed PPL | rolling byte PPL | train wall | optimizer steps | trained query passes | runtime estimate |
|---|---|---:|---:|---:|---:|---:|---:|
| `pd0-100` | full tree, one update/epoch | 12.1095 | 12.0540 | 584s | 100 | 890M | 1535 MB |
| `pd1-100` | unigram subtree units | 4.7929 | 5.3359 | 582s | 6,500 | 890M | 1546 MB |
| `light-u20k-q10-r2` | Lightning, cheap | 5.1783 | 5.4730 | 228s | 40,000 | 397M | 1136 MB |
| `light-u20k-q10-r6` | Lightning, balanced | 4.9776 | 5.3037 | 727s | 120,000 | 1193M | 1136 MB |
| `light-u20k-q10-r10` | Lightning, more repeats | 4.9511 | 5.3392 | 1107s | 200,000 | 1997M | 1136 MB |
| `light-u40k-q5-r6` | Lightning, smaller units | 4.9780 | 5.3422 | 769s | 240,000 | 1200M | 1136 MB |
| `traversal-u40k-r4-context` | Traversal-stop, context ancestors | 4.7637 | 5.1657 | 1999s | 160,000 | 3181M | 1136 MB |
| `drop80` | Lightning + node dropout | 5.0503 | 5.3739 | 708s | 120,000 | 954M | 1136 MB |
| `entgate50` | Lightning + entropy loss gate | 4.9775 | 5.3208 | 666s | 120,000 | 1197M | 1136 MB |

Interpretation:

- pd0 confirms the update-cadence problem. One Adam update per epoch is nowhere
  near enough.
- pd1 is a strong baseline because unigram-subtree units already provide about
  65 updates per epoch without losing full-tree supervision.
- Lightning gives far more optimizer steps and lower memory, but it gives up
  some deterministic coverage/aggregation. The best traversal-stop context-only
  point now beats pd1 on fixed-token PPL and rolling byte PPL, but costs much
  more wall time.
- Smaller sampled units help up to a point. `q10k/u20k` beats the matched-query
  `q20k/u10k` family, but `q5k/u40k` does not beat `q10k/u20k`.
- Repeating the same mini-tree helps from r2 to r6. Past that, returns diminish
  and metric disagreement appears.
- Dropout and entropy gates should be treated as separate regularization
  offshoots, not as evidence for or against stochastic batching itself.

Bottom line: stochastic batching is useful for update cadence and memory, but
not a decisive quality or compute-efficiency win. It should be recorded as a
promising mechanism, not the main AGPT v1 path.

## Open Questions

- Does mini-tree sampling need explicit coverage correction, or is structural
  stochasticity forgiving enough?
- What mini-tree size gives the best tradeoff between aggregation and update
  cadence?
- Should anchor traversal stop probabilities be depth-dependent?
- Should sampled descendant paths expand all the way to max depth, or only to a
  local budget/cap?
- Can suffix/backoff slots be added later as a separate architectural change
  without entangling the Lightning sampler?
