# Gradient Population — freeze, fan out, observe

Status: active (opened 2026-09-24).

Diagnostic, not a training recipe. Numbers here are **not** canonical PPL
and do not go through `bin/agpt_experiment`; they are geometry
measurements of the per-partition gradients that the trainer normally
sums before stepping.

## Idea

Big Issue #2 for AGPT is that more prefix sharing means fewer optimizer
updates per epoch. Thomas's reframing (2026-09-23): every training unit
is asked the same question, "how would you nudge *this one* frozen
model?" Because every unit answers against the same θ, aggregation still
buys full sharing — we just don't sum the answers. The un-summed set
`{θ + δ_i}` is a population, and its geometry is what tells us where
sharing is cheap (units agree) and where it is expensive (units cancel).

The quantity that directly addresses cadence is **coherence**:

```text
rho(S) = || sum_{i in S} delta_i || / sum_{i in S} || delta_i ||
```

rho ≈ 1: the units in S agree, one aggregated step loses nothing.
rho ≈ 0: the units cancel; the aggregated step is a small residual of
large disagreeing pulls, and sequential steps would have taken a path the
sum never sees.

## Mechanism

`bin/agpt_train_v2` gained an env-var probe (see `GradDumpProbeV2` in
`src/cudax/agpt_train_v2.cu`):

```sh
AGPT_GRAD_DUMP_DIR=rnd/gradient-population/<name> \
  bin/agpt_train_v2 --config rnd/gradient-population/configs/<cfg>.yml
```

In train-epoch mode every training unit's fired gradient (after
`scale_gradients_for_fire`, i.e. the **event-mean** gradient) is appended
as one float32 row to `<dir>/grads.f32`, with metadata in
`<dir>/units.tsv` and the parameter section map in `<dir>/layout.json`.
The optimizer step is **skipped**, so all rows are answers about the same
frozen weights. (`AGPT_GRAD_DUMP_APPLY=1` keeps the step, to dump the
sequential trajectory instead.) `TrainingUnit` gained an `anchor_id`
field so pd>1 units can be tied back to their trie node.

The pd=1 fired gradient should equal the event-weighted mean of its pd=2
children (`sum_i ev_i g_i / sum_i ev_i`) **restricted to targets at
depth ≥ 2** — the pd≥2 "descendant" plan makes the ancestors above the
anchor context-only, so their own next-char targets are never trained at
pd≥2. Events per root are exactly 16/15 between pd=1 and pd=2 for this
reason (events = Σ_depth mass, 16 depths vs 15). Measured with
`experimental.loss_depth_min: 2` on the pd=1 side: forward losses agree
to 1e-5, but the gradients agree only to **median 2.8% / max 15%**
relative error, spread evenly over every parameter section. It is not
the ancestor-scatter path (`anc_grad: false` gives the same residual),
not chunking (single-chunk units show it too), and not the query weights
(events match exactly). Open item; see Next steps. It does not affect
the population statistics, each row being exactly what the trainer would
fire for that unit at that pd.

Analysis: `src/tools/agpt_grad_population.py DUMP_DIR` → prints the
tables below and writes `DUMP_DIR/summary.json`.

## Setup

d=64 L=2 h=4 ff=256, depth-16 trie on the carved 95% Shakespeare train
slice (`data/.tries/64e9fe109211b366`), seq_len 16, anc_grad on, mass
weight linear. Frozen weights taken from the pd=1 Adam run
`rnd/stochastic-agpt/20260611T160456-d64l2-depth16-pd1-100ep`
(held-out byte PPL 8.04 @ ep10, 5.72 @ ep50, 5.34 @ ep100) plus the
seed. Partition depth 2 → 1401 units (65 root children × their depth-2
children), 108,481 floats per row, ~608 MB per dump. Dumps are
gitignored and regenerable in ~15 s each.

## Results (pd=2, frozen fan-out)

Coherence of the population, by checkpoint:

| checkpoint | train loss | ρ global (event-wtd) | ρ global (unwtd) | ρ per-root, event-wtd avg | ρ per-root, unwtd avg | cos within root | cos across root | pairs with cos<0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| seed  | 4.68 | 0.719 | 0.494 | 0.851 | 0.691 | 0.434 | 0.297 | 0.4% |
| ep10  | 2.04 | 0.590 | 0.212 | 0.693 | 0.404 | 0.144 | 0.112 | 18% |
| ep50  | 1.67 | 0.394 | 0.127 | 0.527 | 0.318 | 0.064 | 0.050 | 22% |
| ep100 | 1.60 | 0.130 | 0.049 | 0.351 | 0.279 | 0.008 | 0.000 | 51% |

"ρ global" is what pd=0 (one step per epoch) keeps; "ρ per-root" is what
pd=1 keeps, averaged over root children weighted by events.

Cosine to the global event-weighted mean direction, and by unit size
(event quartiles), ep100:

| events quartile | units | mean pairwise cos | ρ | mean ‖g‖ (event-mean) |
|---|---:|---:|---:|---:|
| 15 – 255       | 352 | 0.0007 | 0.068 | 10.09 |
| 255 – 1620     | 359 | 0.0008 | 0.066 | 3.24 |
| 1620 – 6960    | 352 | 0.0021 | 0.069 | 1.67 |
| 6960 – 393105  | 351 | 0.0081 | 0.102 | 1.16 |

Same table at ep10: cos 0.017 / 0.066 / 0.139 / 0.367, ρ 0.13 / 0.22 /
0.32 / 0.54. Big units agree with each other; small units are noise with
large event-mean norms (an event-mean over 15 events is ~10× the norm of
one over 100k events).

Per-root coherence at ep100 (unweighted ρ, roots with ≥5 units):
lowest `'` 0.18, `-` 0.20, ` ` 0.22, `\n` 0.22, `o` 0.23, `e` 0.23;
highest `,` 0.81, `j` 0.73, `X` 0.71, `v` 0.64, `W` 0.54, `J` 0.49. The
space root (55 units, 2.4M events, 15% of all events) is among the least
coherent at every checkpoint after init.

Covariance spectrum of the centered population (1401 rows in 108k dims):

| checkpoint | participation ratio | top-1 var | top-10 | top-100 | rank for 90% | rank for 99% |
|---|---:|---:|---:|---:|---:|---:|
| seed  | 145 | 4.7% | 18% | 58% | 429 | 1015 |
| ep10  | 158 | 2.8% | 17% | 63% | 334 | 799 |
| ep50  | 162 | 2.2% | 16% | 64% | 308 | 784 |
| ep100 | 152 | 2.1% | 16% | 67% | 269 | 707 |

Event-weighted, uncentered (Fisher-like second moment): participation
ratio 3.6 → 11.8 → 54.0 → 148.5 (seed → ep100); top-1 share 52% → 3%.

Per-section coherence is flat across sections (ep100 event-weighted ρ
0.11–0.17 for every weight matrix, 0.23 for `out_w`); the embedding
table carries ~28% of gradient norm.

## Reading

1. **Cancellation is the cost of sharing, and it grows with training.**
   At init the population is nearly a single direction (ρ_w 0.72 for one
   global step; PR 3.6). By ep100 a pd=0 step keeps 13% of the gradient
   mass and a pd=1 step keeps 35%; half of all unit pairs point in
   opposite directions. This is Big Issue #2 measured directly: late in
   training, aggregation at the root throws away most of what the units
   asked for.
2. **Coherence is not uniform over the trie**, so a single pd is the
   wrong knob. Roots with few, similar continuations (`,` `j` `X`) stay
   coherent and can be shared; the space root and vowels cancel and
   should be split. This is the same asymmetry the old `hotspot`
   experiment exploited by splitting the space subtree — and it explains
   why hotspot helped at pd=1 and hurt at pd=6.
3. **Trie proximity buys only a little agreement.** Siblings under one
   root child are more aligned than units under different roots, but by
   ~1.3–1.5× at ep10/ep50 and both are ~0 at ep100. The topology of the
   depth-1 split is a weak predictor of gradient direction; what units
   actually agree on is probably target-distribution similarity, not
   prefix similarity (untested, see next steps).
4. **The population is structured, not isotropic.** 1401 random rows in
   108k dims would give participation ratio ≈ 1401; we see ~150, with
   90% of variance in ~270–430 directions. The disagreement lives in a
   low-hundreds-dimensional subspace. That is the input a quasi-Newton /
   natural-gradient step would want, and it is the precondition for any
   "predict the attractor" attempt — met, but not strongly (not rank-10).
5. **Unit-size normalization is doing damage at fine pd.** The trainer
   fires the event-mean gradient, so a 15-event unit steps as hard as a
   400k-event unit in a direction that is mostly noise (mean pairwise cos
   0.0007). That is the mechanism behind the pd=7 breakdown in
   `rnd/partition-depth/`.

## Trainer bug found on the way (fixed 2026-09-24)

The first dumps were not reproducible run-to-run even with frozen
weights: per-row gradients differed by 2–7% (median) and >100% (max), and
a unit's *forward* mean loss differed by up to 0.5%. Stage-by-stage
tensor diffs (`AGPT_DIAG_TENSOR_DIR`, extended to all layers plus the
per-query loss vector) localized it: first LayerNorm-1 (2 of 18,170 rows
differing), then, after fixing that, the per-query loss vector (48 rows,
some off by 20×). Root cause, confirmed by `compute-sanitizer --tool
racecheck`: every two-phase shared-memory reduction did

```text
float max_val = sdata[0];      // all threads read slot 0
sdata[tid] = local_sum;        // thread 0 may overwrite slot 0 first
```

with no `__syncthreads()` between. Fixed in `src/cuda/kernels.cu`
(LayerNorm forward, softmax, fused attention softmax, fused softmax-CE,
batched varlen attention ×2 — these kernels are shared by the v1 and v2
trainers) and `src/cudax/kernels_v2.cuh` (`agpt_loss_per_query_kernel_v2`,
which runs 4 warps over the 65-token vocab and was the worst offender).
After the fix: forward loss bit-identical on 65/65 pd=1 units across
runs; gradient rows agree to ~1e-5, which is the backward atomicAdd floor
documented in `todo/deterministic-backward.md`.

All dumps and numbers in this README were regenerated with the fixed
kernels. The pre-fix population statistics agreed with the post-fix ones
to three decimals, so nothing above changed materially. **Every CUDA
AGPT training result before this fix was trained with ~0.3% of per-query
losses randomly corrupted by up to 20×.** Whether the canonical
baselines deserve a re-run is Thomas's call.

## Caveats

- pd≥2 dumps do not contain the depth < pd targets (see Mechanism).
- Rows are event-mean gradients. Multiply by `trained_events` to get the
  raw summed contribution; the analysis reports both weightings.
- The pd=1 ↔ Σ pd=2 identity holds only to ~3% (see Mechanism).
- Operational: do **not** run `compute-sanitizer` while any other GPU
  job is running on this 15 GB / 8 GB-GPU machine; the combination
  thrashed swap and froze the box on 2026-09-24. Sanitizer logs from the
  diagnosis are kept in `_sanitizer/`.


## Experiment 2 — coherence-driven cadence, static map (2026-09-24)

Question: does splitting only the *incoherent* root children deeper beat
uniform partition depth at equal epochs? Five arms, all v2, d64 L2
depth16 seq16, Adam lr 0.0015 warmup-cosine, 100 epochs from the seed,
fixed kernels, evaluated through `bin/agpt_experiment` on the carved
multi-chunk held-out split. Run dirs `2026…-cadence-*-100ep/`; configs in
`configs/cadence-*.yml`; maps in `maps/`.

Map construction: roots with ≥5 pd=2 units ranked by **unweighted** ρ at
ep10, split into tertiles. `coherence`: top tertile pd1 (share), middle
pd2, bottom pd3 (split). `reversed`: the same tertiers flipped. Roots with
<5 units → pd1 in both. The mixed plan is `experimental.partition_depth_map`
(a `<token_id> <depth>` file) handled by `build_mixed_partition_plan_v2`.

| arm | units | events/unit median | units <1000 events | ep25 fixed | ep50 fixed | **ep100 fixed** | ep100 rolling byte | train wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pd1 uniform | 65 | 83,488 | 5% | 6.114 | 5.149 | 4.810 | **5.348** | 570 s |
| pd2 uniform | 1,401 | 1,620 | 42% | 5.626 | 5.131 | 4.771 | 6.160 | 604 s |
| pd3 uniform | 11,481 | 154 | 78% | 5.167 | 4.860 | **4.591** | 7.043 | 1186 s |
| coherence map | 3,412 | 140 | 76% | 5.690 | 5.411 | 5.267 | 6.804 | 770 s |
| reversed map | 4,671 | 252 | 71% | 5.324 | 4.990 | 4.778 | 6.438 | 797 s |

"fixed" = `agpt_fixed_token_perplexity` (every scored token has the full
16-char window). "rolling byte" = canonical lm-eval rolling byte PPL
(each 16-token window scores its first tokens with 0…15 chars of
context).

### Reading

1. **The two metrics disagree, and the disagreement is a training-plan
   artifact.** Any pd≥2 plan never trains the targets at depth < pd (the
   ancestors above the anchor are context-only), so those models have
   never seen a short-context prediction. Rolling byte PPL scores 1/16 of
   positions with zero context, 1/16 with one char, etc., and punishes
   exactly that. Under the full-window metric the ordering flips. For
   cadence comparisons the fixed-window metric is the fair one; for the
   canonical number, any split plan needs the shallow targets trained
   somewhere (e.g. one extra pd1-style pass over the depth<pd stubs).
2. **The static coherence map loses.** At full window it is the worst arm
   (5.27 vs 4.81 for pd1), and it is the only arm whose curve stalls
   (5.69 → 5.41 → 5.27 while pd1 goes 6.11 → 5.15 → 4.81). The reversed
   control, which splits the *coherent* heavy roots, matches pd2 and pd1.
   pd3 uniform is best. Conclusion 2 of Experiment 1, as operationalised
   here, is refuted.
3. **What the ordering actually tracks is cadence on mass.** Splitting
   the heavy roots (reversed, pd2, pd3) helps; splitting only the light
   roots (coherence) hurts. The "incoherent" tertile holds 2.5M events
   against the coherent tertile's 6.5M, and its low unweighted ρ is
   largely *noise* from tiny depth-2 children, not disagreement — the
   very thing Experiment 1's quartile table warned about. Unweighted ρ
   over children is a poor split criterion; it flags small subtrees.
4. **Likely mechanism for the stall: Adam with heterogeneous unit
   sizes.** The trainer fires event-*mean* gradients, so a 15-event unit
   has ~10× the gradient norm of a 400k-event unit, and Adam's global
   second-moment estimate is set by whichever units fire most often. In
   the coherence arm 76% of units carry 2.8% of the events and fire
   between ~30 giant steps per epoch; the heavy steps get normalised
   against tiny-unit noise and shrink. In pd3 the same small units exist
   but the mass is also delivered in many small steps, so the moments
   match the steps. This is a hypothesis; the test is a fire
   normalisation proportional to mass (summed rather than mean
   gradient, or sqrt-mass), which the v2 trainer does not yet expose
   (`train.fire_norm` accepts only `mass`).

### What this does and does not settle

- Settled: a per-root split chosen by unweighted children-coherence,
  under Adam with event-mean firing, is worse than uniform splitting.
- Not settled: whether coherence *properly measured* (event-weighted,
  noise-corrected, at the depth actually being split) and *properly
  applied* (mass-proportional steps, or per-unit Adam state) beats
  uniform pd. Experiment 1's measurement stands; this experiment shows
  the naive translation into a plan does not.

### Next

1. Add `train.fire_norm: sum|sqrt` to v2 and rerun coherence / reversed /
   pd3 — the cheapest test of the Adam-heterogeneity hypothesis.
2. Rebuild the maps from **event-weighted** ρ with a minimum-mass floor
   per unit (never split a subtree into units under ~1k events).
3. Train the depth<pd stubs so split plans are comparable on the
   canonical rolling metric.
4. Dynamic re-measurement every k epochs remains unbuilt.


## Experiment 3 — does the pool converge? (2026-09-24)

Thomas's question, taken literally: start from one model, make a tiny
pool, apply random node updates to each member over and over. Do the
members converge to the same weights? To different weights? To weights
that are the same under some projection?

Setup: θ₀ = the clean pd=1 Adam run's epoch-25 checkpoint. Four members
per learning rate, each a plain-SGD chain (no optimizer state, constant
lr) over the 1401 pd=2 units in its own seeded random order per epoch
(`experimental.unit_order_seed`), 40 epochs, checkpoints at 1, 2, 5, 10,
20, 30, 40. Two learning rates, 0.02 and 0.1 (0.5 was unstable in a
probe). Analysis: `src/tools/agpt_pool_analysis.py`; dirs `pool/lr*-m*/`;
summaries `pool/summary-lr*.json`. Held-out function-space numbers use
8192 fixed-window targets on the carved held-out slice.

Notation: disp = ‖mean(θ) − θ₀‖; spread = RMS distance of members to the
pool mean; NLL in nats/char; pair KL = mean KL between members'
next-char distributions on the same held-out targets.

**lr = 0.02**

| epoch | disp | spread | pair weight dist | member NLL | pool-mean model NLL | pair KL | KL mean→members |
|---:|---:|---:|---:|---|---:|---:|---:|
| 1 | 1.35 | 0.65 | 1.01–1.12 | 1.817–1.858 | **1.808** | 0.080 | 0.032 |
| 5 | 3.92 | 0.95 | 1.52–1.58 | 1.802–1.820 | **1.776** | 0.087 | 0.034 |
| 10 | 6.40 | 1.30 | 2.08–2.19 | 1.765–1.795 | **1.745** | 0.095 | 0.038 |
| 20 | 10.23 | 2.12 | 3.38–3.56 | 1.723–1.785 | **1.707** | 0.115 | 0.046 |
| 40 | 15.60 | 4.01 | 6.46–6.72 | 1.675–1.690 | **1.661** | 0.109 | 0.047 |

spread ∝ epoch^0.49.

**lr = 0.1**

| epoch | disp | spread | pair weight dist | member NLL | pool-mean model NLL | pair KL | KL mean→members |
|---:|---:|---:|---:|---|---:|---:|---:|
| 1 | 4.86 | 3.48 | 5.45–5.90 | 1.907–1.994 | **1.881** | 0.216 | 0.097 |
| 5 | 11.90 | 5.87 | 9.26–9.75 | 1.843–1.877 | **1.792** | 0.210 | 0.093 |
| 10 | 17.62 | 8.76 | 13.9–14.6 | 1.777–1.806 | **1.749** | 0.228 | 0.110 |
| 20 | 25.06 | 13.74 | 21.9–22.9 | 1.705–1.762 | 1.719 | 0.249 | 0.147 |
| 40 | 34.29 | 21.77 | 34.9–36.2 | 1.654–1.670 | 1.753 (worse than every member) | 0.246 | 0.232 |

spread ∝ epoch^0.51; spread at matched epoch is 5.4× the lr-0.02 value
(lr ratio 5).

### Answers

1. **They do not converge to a point.** In weight space the members
   random-walk apart: spread ∝ √epoch at both learning rates, and
   ∝ lr, exactly the diffusion term η²·C_g per step summed over steps.
   No sign of saturation within 40 epochs. So the units do not share a
   minimizer — this is the conflict regime of the math above, where the
   pool converges in distribution (a cloud) rather than to a point.
2. **But the cloud is the same size in function space the whole time.**
   Pairwise KL between members is flat (0.08 → 0.11 nats at lr 0.02;
   0.21 → 0.25 at lr 0.1) while pairwise weight distance grows 6×.
   The growing weight-space separation is along directions the
   function does not see — it concentrates in the FFN matrices
   (`l0.l2_w`, `l1.l1_w`, `l0.l1_w`, 15–17% each by epoch 40), where a
   ReLU MLP has its scaling and permutation-like flat directions. That
   is Thomas's "different, but the same under a projection": the
   projection is onto function space, and under it the members sit at a
   constant distance from each other set by the learning rate.
3. **The center of the cloud is a better model than any member — until
   the cloud outgrows its basin.** At lr 0.02 the weight-space mean of
   the four members beats every member at every checkpoint (1.661 vs
   1.675–1.690 at epoch 40) and sits at less than half the pairwise KL
   from each. At lr 0.1 the same holds through epoch 10, then the mean
   model degrades and by epoch 40 it is worse than every member: the
   members have diffused out of a linearly connected region and
   averaging weights across it no longer gives a valid model. The
   crossover is at spread ≈ 10–14 in these units.
4. **The mean drifts ballistically.** disp grows ~linearly in epochs
   (15.6 at 40 for lr 0.02), i.e. the pool's center follows a
   deterministic flow while the members diffuse around it — the
   moment-closure picture (mean follows the modified loss, covariance
   follows the linearised recursion) is the right description, not the
   N^k tree.

### What this buys

- The explosion collapses to (mean, covariance), and the mean is a
  strictly better model than any chain — a free gain over a single
  sequential run, as long as the pool stays inside one basin (small lr
  or periodic re-centering).
- "Where is it headed" is a question about the mean's flow, which is
  low-dimensional and smooth; the members' diffusion is orthogonal to it
  in function space.
- Not yet tested: whether the pool mean's flow matches aggregated
  full-batch GD on the modified loss L̃ (i.e. whether the η² implicit
  regulariser is measurable here), and whether re-centering the pool
  every k epochs (Polyak-style averaging and restart) beats a single
  chain per gradient evaluation.


## Experiment 4 — the pool with true per-node updates (2026-09-24)

Thomas's correction to Experiment 3: a pd=2 step is already an aggregated
subtree, which hides within-subtree conflict and caps depth. Per-node
updates are, by SGD-equivalence, plain SGD on (prefix, next-char)
examples; a causal window of 16 chars is 16 nested nodes at depths 1–16
in one step. So this experiment uses no trie at all.

Setup: `src/tools/agpt_pool_sgd.py` (PyTorch, same architecture and the
same θ₀ = clean pd=1 Adam epoch-25 checkpoint, via `agpt_ppl.AGPTModel`).
M=4 members. Each step: draw a member uniformly, draw one random
16-char window from the training corpus, one plain SGD step on that
member with the mean causal cross-entropy over the window. 400,000 steps
per pool (≈100k per member ≈ 1.8 epochs of positions per member),
snapshots every 40k. Learning rates 0.002 and 0.01 (single-chain probes:
0.002 improves steadily from θ₀; ≥0.02 first worsens held-out NLL by the
SGD noise floor). Held-out numbers: 8192 fixed-window targets. θ₀ NLL =
1.8337.

**lr = 0.002**

| step | disp | spread | s/d | member NLL | pool-mean model NLL | pair KL | KL mean→members |
|---:|---:|---:|---:|---|---:|---:|---:|
| 40k | 0.76 | 0.48 | 0.63 | 1.810–1.814 | **1.787** | 0.062 | 0.024 |
| 120k | 1.50 | 0.64 | 0.43 | 1.788–1.792 | **1.766** | 0.059 | 0.023 |
| 200k | 2.13 | 0.75 | 0.35 | 1.770–1.776 | **1.749** | 0.061 | 0.024 |
| 400k | 3.45 | 0.92 | 0.27 | 1.744–1.750 | **1.722** | 0.061 | 0.024 |

spread ∝ step^0.28.

**lr = 0.01**

| step | disp | spread | s/d | member NLL | pool-mean model NLL | pair KL | KL mean→members |
|---:|---:|---:|---:|---|---:|---:|---:|
| 40k | 2.70 | 1.62 | 0.60 | 1.860–1.882 | **1.800** | 0.177 | 0.072 |
| 120k | 5.23 | 2.19 | 0.42 | 1.805–1.841 | **1.761** | 0.143 | 0.057 |
| 200k | 7.23 | 2.61 | 0.36 | 1.778–1.791 | **1.731** | 0.143 | 0.058 |
| 400k | 11.11 | 3.60 | 0.32 | 1.728–1.737 | **1.687** | 0.132 | 0.054 |

spread ∝ step^0.35.

### Answers, per-node version

1. **Still no convergence to a point, but now sub-diffusive.** Spread
   grows as step^0.28–0.35, slower than the √step random walk of the
   pd=2 chains. With true per-node steps the curvature's restoring pull
   is visible: the cloud is heading toward a saturated size rather than
   diffusing freely. It has not saturated within 100k steps per member.
   Spread at matched step scales ~lr^0.85 (3.60 vs 0.92 for 5× lr).
2. **Constant separation in function space, again.** Pairwise KL is flat
   from the first snapshot: 0.06 nats at lr 0.002, 0.13–0.18 at lr 0.01,
   while weight distance keeps growing. The separation lives in the FFN
   matrices (`layers.0.l2`, `layers.1.l1`, 14–17% each) plus the
   embedding early on — the same flat directions as in Experiment 3.
3. **The pool mean is a much better model than any member, at every
   snapshot, at both learning rates, and the gap is larger than with
   aggregated steps.** At lr 0.01, step 400k: members 1.728–1.737,
   mean 1.687 — 0.04–0.05 nats better than the best member, and better
   than the lr-0.002 pool's mean (1.722) despite far noisier members.
   No member left the basin (spread 3.6, versus the ≈10–14 where
   averaging broke in Experiment 3). The noisier pool has the better
   centre: the members explore, the mean cancels their noise.
4. **Data efficiency.** Each member here saw ≈1.8 epochs of positions
   and the pool mean reached 1.687. The pd=2 aggregated chains needed 40
   epochs per member to reach a mean of 1.661. Per position seen, the
   per-node pool is ~20× more data-efficient, which is the cadence
   effect measured end-to-end: many small steps, then average.

### Comparison with Experiment 3 (aggregated pd=2 steps)

| | pd=2 aggregated chains | per-node SGD pool |
|---|---|---|
| spread growth | step^0.5 (free diffusion) | step^0.3 (curvature-limited) |
| pair KL | flat, ~0.1 (lr .02) | flat, ~0.06 (lr .002) / ~0.14 (lr .01) |
| pool mean vs members | better until basin breaks (lr 0.1) | better throughout, larger margin |
| data per member for mean ≈ 1.66–1.69 | 40 epochs | ~1.8 epochs |

Aggregation inside the step was hiding both the restoring force and most
of the cadence benefit, as Thomas suspected.

### What this now supports

- **Population as the optimizer, first working form:** M plain-SGD
  members with random node updates, model = their weight mean. At lr
  0.01 this beats every member by ~0.05 nats with no extra gradient
  evaluations beyond what the members do anyway (averaging is free).
- The moment-closure picture holds with per-node steps too: ballistic
  mean, sub-diffusive cloud, constant function-space radius set by lr.
- Next: (a) re-centre periodically (set all members to the mean every k
  steps) and compare against one chain at the same gradient budget;
  (b) larger M (the mean's advantage should grow like the variance
  reduction, ∝ 1/M, until the drift term dominates); (c) push lr until
  the basin breaks, to find the per-node version of the ≈10–14 spread
  threshold; (d) test whether the mean's trajectory matches full-batch
  GD on the modified loss L̃.


### Experiment 4b — pool vs single chain at equal gradient budget (2026-09-25)

All runs: 400,000 single-window SGD steps in total from the same θ₀,
held-out NLL of the final model. "mean" = weight average of members.

| run | lr | steps per member | member NLL | model used | NLL |
|---|---:|---:|---|---|---:|
| single chain | 0.01 | 400k | 1.6045 | the chain | 1.6045 |
| **single chain + tail iterate average** (Polyak from step 200k) | 0.01 | 400k | 1.6045 | running mean of last 200k iterates | **1.5722** |
| 4-member pool | 0.01 | 100k | 1.728–1.737 | mean | 1.6870 |
| 4-member pool, re-centred every 40k | 0.01 | 100k | 1.724–1.741 | mean | 1.6820 |
| 16-member pool | 0.01 | 25k | 1.817–1.844 | mean | 1.7554 |
| 4-member pool | 0.03 | 100k | 1.713–1.725 | mean | 1.6777 |
| 4-member pool | 0.05 | 100k | 1.712–1.735 | mean | 1.7065 |

Reading:

1. **Averaging is variance reduction, not cadence.** At equal budget the
   single chain that keeps all its sequential steps beats every pool
   mean by 0.07–0.15 nats. Splitting the budget across members costs
   sequential time that no amount of averaging buys back (16 members are
   worse than 4). Four members at 100k steps ≈ one chain at 160k steps.
2. **Running hotter does not rescue the pool.** lr 0.03 gives the best
   pool mean (1.678) but still far behind the lr-0.01 chain; at lr 0.05
   the mean barely beats its members (spread 16, near the basin limit).
3. **The cheapest variance reduction wins outright.** A running average
   of one chain's iterates over its last 200k steps — zero extra
   gradient evaluations, no parallel members — gives 1.572, the best
   number of the whole strand, 0.03 below the raw chain and 0.10 below
   the best pool mean. Temporal averaging of one chain recovers what
   spatial averaging of M chains gives, without paying M× in steps.
4. What a pool still offers is wall-clock parallelism (M devices, each
   1/M of the steps, then average) — the Local-SGD trade — not a better
   optimizer per gradient evaluation.

Net for "population as the optimizer": the population's value is its
mean's noise cancellation, and a single chain's own history is already a
population for that purpose. The open question is therefore not the
pool but the modified loss L̃ — whether one aggregated (trie-shared)
step on L̃ tracks the sequential chain (Experiment 5, in progress).


## Experiment 5 — does one aggregated step on the modified loss track the sequential chain? (2026-09-25)

Backward-error analysis says the mean of random-order SGD with step η
follows gradient descent on L̃ = L + (η/4)·mean_i‖g_i‖². The extra term's
gradient, (η/2)·mean_i H_i g_i, is a per-example Hessian-vector product
that aggregates through the trie like a gradient. If full-batch descent
on L̃ tracks the pool mean, aggregated AGPT plus one extra backward pass
gets the effect of sequential cadence without threading.

Setup (`src/tools/agpt_igr_test.py`): from the same θ₀ as the lr-0.002
pool (Experiment 4), full-batch GD (65,536-window gradient per step,
lr 0.1 × 2000 steps = learning-rate mass 200, the same as one pool member's
100k × 0.002) in two variants: **A** on L, **B** on L̃ with the penalty
gradient estimated from 512 windows per step by finite-difference HVPs.
Compared against the four saved pool members and their mean (spread
0.924, mean displacement 3.452). Distances in parameter L2; KL on 8192
held-out fixed-window targets.

| | disp | ‖·−pool mean‖ | ‖·−members‖ | KL(·→pool mean) | held-out NLL |
|---|---:|---:|---|---:|---:|
| pool mean (reference) | 3.452 | 0 | 0.92 (=spread) | 0 | 1.7220 |
| pool members | — | 0.92 | — | ~0.024 | 1.744–1.750 |
| **A**: GD on L, lr-mass 200 | 3.670 | 1.707 | 1.94 | 0.0237 | **1.6988** |
| **B**: GD on L̃, lr-mass 200 | 3.392 | **0.949** | 1.31–1.34 | **0.0110** | 1.7126 |

Trajectory (distance to pool mean at lr-mass 50 / 100 / 150 / 200):
A 2.47 / 1.84 / 1.57 / 1.71 — B 2.44 / 1.69 / 1.14 / 0.95.
mean_i‖g_i‖² ≈ 55 throughout, so the penalty is (0.002/4)·55 ≈ 0.03 nats.

### Reading

1. **The regularizer is real and does what the expansion says.** Adding
   it halves the distance from the full-batch endpoint to the pool mean
   (1.71 → 0.95), halves the function-space KL (0.024 → 0.011), and
   brings the displacement magnitude onto the pool's (3.39 vs 3.45; plain
   GD overshoots to 3.67). The penalised endpoint sits *inside* the pool
   cloud: its distance to the mean equals one pool spread, and its
   distance to each member is √2·spread, exactly what a point at the
   cloud's centre would show. Plain GD sits two spreads outside. A term
   worth 0.03 nats accounts for half the discrepancy between one-shot
   aggregation and sequential SGD.
2. **It is computable by aggregation.** The whole correction is
   mean_i H_i g_i, one HVP per example along its own gradient, summed —
   the same algebraic family as the gradient itself, so it factorises
   through shared prefixes. This is the "branch-point operation" Thomas
   asked about: not a merge rule on the proposals, but one more
   accumulated quantity per node.
3. **But fidelity is not value, at this learning rate.** Held-out NLL:
   plain GD 1.699 < penalised GD 1.713 < pool mean 1.722 < members
   1.745. At η = 0.002 the sequential process has no advantage to
   transfer: it is worse than one-shot aggregation on held-out, and the
   regulariser faithfully makes full-batch descent worse in the same
   direction. The regulariser's *benefit* (SGD's generalisation edge)
   has to be measured where SGD actually beats GD — at higher η or later
   in training — which is the run queued next (lr-0.01 pool, lr-mass
   1000, ~2.5 h).

### Status of the strand after Experiments 1–5

- Big Issue #2 (sharing vs cadence) is now a precise statement:
  aggregation at one θ loses (i) the SGD noise, which is *harmful* at
  small η and cancelled for free by iterate averaging anyway, and (ii)
  the second-order term (η/4)·mean‖g_i‖², which is *recoverable* inside
  aggregation by one extra backward pass. Whether (ii) is worth having
  is the remaining empirical question.
- The population/pool as an optimiser reduces to iterate averaging
  (Experiment 4b) plus this regulariser (Experiment 5).


### Experiment 5b — value test at lr 0.01 (2026-09-25)

Same protocol against the lr-0.01 pool (members 100k × 0.01 = lr-mass
1000; spread 3.60, mean displacement 11.1, mean NLL 1.687, members
1.728–1.737). Full-batch GD lr 0.1 × 10,000 steps, 65,536-window
gradient, penalty from 512 windows/step. ~2 h per variant.

| endpoint | disp | ‖·−pool mean‖ | KL(·→pool mean) | held-out NLL |
|---|---:|---:|---:|---:|
| pool mean | 11.11 | 0 | 0 | 1.6870 |
| pool members | — | 3.6 | — | 1.728–1.737 |
| **A**: GD on L | 10.43 | 10.73 | 0.124 | **1.5882** |
| **B**: GD on L̃ | 10.32 | **4.92** | **0.033** | 1.6655 |

Trajectory of ‖·−pool mean‖ at lr-mass 250/500/750/1000: A 9.6 / 9.4 /
9.9 / 10.7 (diverging from the chains); B 8.0 / 6.1 / 5.1 / 4.9
(converging onto them). mean‖g_i‖² falls 28 → 24 under the penalty.

**Fidelity confirmed, strongly.** With five times the noise, plain GD
wanders off in a different direction from the sequential chains and
keeps going (KL to the pool mean *grows* along the trajectory); the
penalised run converges onto the chains' mean — KL 4× smaller, distance
halved and still shrinking, displacement matched. The (η/4)·mean‖g_i‖²
term is the fingerprint of sequential threading.

**Value negative, again.** Plain GD at lr-mass 1000 reaches 1.588 —
better than the pool mean (1.687), better than every member, and better
than the single 400k-step chain (1.6045, lr-mass 4000). Following the
sequential trajectory costs 0.08 nats. At this model/corpus scale,
one-shot aggregation is the better optimiser *per unit of learning-rate
mass*, and cadence has no second-order magic to transfer.

### The reframing this forces

The axes were being conflated. Two costs:

| | sequential chain (batch 1) | full-batch GD |
|---|---:|---:|
| steps to ≈1.59–1.60 | 400k | 10k |
| positions processed | 6.4M (0.4 epoch) | 10.5B (~650 epochs) |
| wall here (PyTorch) | 20 min | 4 h |

Per *step*, aggregation wins 40×. Per *position processed*, the chain
wins 1600×. The chain is launch-overhead bound in PyTorch; the trie
processes positions ~300× faster than that chain does, but even so one
trie-aggregated GD step ≈ one epoch of trie work (~6 s at pd=1), so
10k of them is ~16 h against the chain's 20 min. Full-batch steps are
capped by curvature (lr 0.2 oscillates, 0.5 diverges) — you cannot make
each aggregated step cover more lr-mass with first-order updates.

So Big Issue #2, final form: **the cadence advantage is a per-compute
advantage, not a per-step one, and no first-order correction to the
aggregated step changes that.** The only way an exact, expensive
aggregated step beats many cheap noisy ones is if it extracts far more
than a first-order update per pass — i.e. curvature: Newton-CG,
L-BFGS, natural gradient. That is the `optimizer-mismatch` thread, and
Experiment 5 hands it a tool: the per-example HVPs that computed the
penalty are exactly the Hessian-vector products a trie-aggregated
Newton-CG step consumes, and they factorise through shared prefixes.

What survives from the population idea:
- iterate averaging (free, +0.03 nats on a chain; Experiment 4b);
- the modified-loss term as a measurable object (Experiments 5, 5b),
  not as a training improvement at this scale;
- HVP aggregation through the trie as the route to a curvature step.


## Experiment 6 — curvature: L-BFGS passes-to-parity (2026-09-25)

The one lever left after Experiment 5b: can a curvature method make one
exact aggregated pass worth many cheap SGD steps? `src/tools/agpt_lbfgs_test.py`:
same model, same θ₀ (pd=1 Adam epoch-25 checkpoint, held-out 1.8337), a
FIXED random subset of 65,536 windows (≈1.05M positions, about one
corpus' worth) as a deterministic objective, torch L-BFGS with strong-
Wolfe line search, history 50. Every closure evaluation (line-search
trials included) counts as one gradient pass. 600-pass budget, 7.3 min.

| held-out NLL target | plain full-batch GD (Exp 5b A) | **L-BFGS** | ratio |
|---:|---:|---:|---:|
| 1.698 | 2,500 passes | **80** | 31× |
| 1.639 | 5,000 passes | **140** | 36× |
| 1.588 | 10,000 passes | **240** | 42× |

L-BFGS curve (passes → held-out): 100 → 1.664, 200 → 1.600, 300 → 1.566,
400 → 1.546, 500 → 1.532, **600 → 1.5228**, still descending at budget.
1.5228 is the best held-out number of the entire strand: below the
single chain + iterate averaging (1.572, 400k steps), below plain GD at
10k passes (1.588), and below the pd=1 Adam run's 100-epoch fixed-window
result from the seed (1.571).

### Reading

1. **Curvature buys ~40× in passes, and it is enough.** The wall-clock
   break-even against the batch-1 chain needed ~50× fewer passes than
   GD at the trie's ~6 s/pass (240 passes ≈ 24 min vs the chain's 20
   min). L-BFGS lands there, and its asymptote is better than anything
   the sequential methods reached. In PyTorch on the subset it is 0.73
   s/pass: 240 passes = 3 min to the chain's 20-minute result.
2. **Exact gradients finally pay.** This is the regime the
   optimizer-mismatch note predicted: a quasi-Newton method that is
   unusable on noisy gradients is very usable on the trie's exact ones.
   No noise, no averaging, no partition depth — one deterministic
   objective, ~600 passes.
3. **"One epoch and done" is closer than it looked.** The objective
   here is one corpus' worth of positions, and L-BFGS solved it (to
   1.52) in 600 passes over it. What it still is not: a single pass.
   The pass count is where the trie's sharing has to do its work.


### Experiment 6b — from the seed, and the same-objective control (2026-09-25)

| run | 100 | 200 | 300 | 400 | 500 | 600 passes |
|---|---:|---:|---:|---:|---:|---:|
| L-BFGS from the random seed (held-out 4.639 at start) | 2.037 | 1.831 | 1.739 | 1.678 | 1.643 | **1.611** |
| plain GD lr 0.1 on the *same* fixed subset, from ep25 | 1.783 | — | 1.769 | — | — | 1.756 |
| L-BFGS from ep25 (Exp 6) | 1.664 | 1.600 | 1.566 | 1.546 | 1.532 | **1.523** |

- **Curvature works from scratch**, at ~0.035 nats per 100 passes late
  in the run and still descending at 600. It has not yet reached the
  pd=1 Adam run's 100-epoch mark (1.571); extrapolating, ~800 passes.
  On the trie at ~6 s/pass that is ~80 min against Adam pd=1's 9.5 min:
  from scratch, partitioned Adam still wins wall-clock at this scale by
  ~8×, with L-BFGS reaching a better asymptote.
- **The same-objective control** removes any doubt about the subset:
  GD on the identical fixed 65k windows gets 1.756 in 600 passes where
  L-BFGS gets 1.523.
- **The hybrid is the interesting recipe**: 25 epochs of partitioned
  Adam (2.4 min on the trie) followed by L-BFGS on the exact gradient
  reached 1.523, below anything else measured, including 100 epochs of
  Adam (1.571). Cheap noisy steps to get near the basin, exact
  curvature steps to finish. That is a concrete proposal for the CUDA
  trainer: pd=1 Adam warm start, then pd=0 L-BFGS.

### Next
- Same test from the seed (random init) — passes to reach the Adam
  pd=1 100-epoch result (1.571); does curvature work from scratch?
- GD on the same fixed subset for 600 passes (same-objective control).
- Full-corpus objective (the trie's true gradient) rather than a subset.
- Then the real thing: L-BFGS driven by the CUDA trainer's aggregated
  gradient (pd=0, one exact pass per evaluation). The two-loop recursion
  is host-side vector algebra over 108k floats; the trainer only needs
  an "evaluate loss+gradient at θ" mode.


## Experiment 7 — L-BFGS on the trie's own aggregated gradient (2026-09-25)

`bin/agpt_train_v2` gained `optimizer: lbfgs` (train-epoch mode): every
epoch is one function evaluation — all pd=1 units run with their raw
gradient sums accumulated in a dedicated device buffer (the backward
zeroes `d_grads` at each unit's first chunk), scaled once by total event
mass. Two-loop recursion, ring-buffer history, Armijo backtracking, all
on device via cuBLAS; only scalars reach the host. Config fields
`train.optimizer.history|c1|max_backtracks` (also added to the
orchestrator schema). Run `20260925T220537-lbfgs-trie-pd1-ep25-600`:
600 passes from the pd1 Adam epoch-25 checkpoint, history 20, 55 min.

| checkpoint | fixed-token PPL | rolling byte PPL |
|---:|---:|---:|
| 100 | 5.270 | 5.787 |
| 200 | 5.088 | 5.605 |
| 300 | 4.989 | 5.512 |
| 400 | 4.975 | 5.499 |
| 600 | 4.888 (NLL 1.587) | 5.407 |
| *ref: Adam pd1 100 ep from seed* | *4.810* | *5.348* |

Worse than Adam, and much worse than the PyTorch L-BFGS curve on the same
start and near-identical objective (first-pass train loss 1.8757 trie vs
1.8732 PyTorch; at pass 600 1.674 vs 1.555). After ~pass 150 the line
search failed constantly: 264 of 600 evaluations were backtracks (steps
down to 0.002), 332 accepted, 3 history resets.

### Diagnosis: the trie gradient, not the line search

1. **Discriminator.** `agpt_lbfgs_test.py --optimizer lbfgs-armijo` is a
   line-for-line mirror of the CUDA algorithm. On the exact PyTorch
   gradient it accepted **593 of 600** steps, 6 backtracks, 0 resets, and
   reached held-out **1.5287** (torch strong-Wolfe L-BFGS: 1.5228). Same
   algorithm, same start; only the gradient source differs.
2. **Directional-derivative check** (`src/tools/agpt_grad_check.py`):
   trie loss at θ ± ε·ĝ versus ‖g‖.

| point | ‖g_anc‖ | FD / g_anc·d (ε .01 / .003) | FD / g_noanc·d | cos(g_anc, g_noanc) |
|---|---:|---|---|---:|
| ep25 start | 0.648 | 1.045 / 1.058 | 1.109 / 1.123 | 0.991 |
| L-BFGS pass 600 | 0.076 | 0.995 / 1.020 | 1.079 / 1.106 | 0.929 |

   The gradient is right to a few percent along its own direction —
   invisible to Adam — but it is not the exact gradient of the loss the
   trainer reports, and L-BFGS amplifies exactly the low-curvature
   directions where a few-percent error dominates. The ancestor-scatter
   path matters (turning it off makes the mismatch 6–10% and rotates the
   gradient 7%) but is truncated: the trainer's anc-grad (resolved
   2026-05-20, `todo/descendant-ancestor-scatter.md`) scatters the
   ancestor-slot K/V gradient into **Wk/Wv only**; it does not continue
   back through the ancestor's LN1, the earlier layer's residual stream,
   or the token embeddings, and ancestor K/V are read from a **bf16**
   cache the backward treats as exact. That note also records that the
   formal parity test against a numerical reference was never done —
   this check is that test. Truncation is also the likely cause of the
   3% pd=1 vs Σpd=2 non-additivity in Experiment 1. (Likely causes, not
   yet isolated: truncation vs bf16 can be separated by rerunning the
   check with an fp32 cache.)

### Consequence

AGPT's premise is an *exact* aggregated gradient. The CUDA trainer
delivers an approximate one, and Experiments 6/7 show that the
approximation is precisely what separates "curvature buys ~40×" from
"curvature loses to Adam". Completing the ancestor K/V backward (full
chain through ancestor positions, fp32 cache) is now the single
highest-value item: it is the prerequisite for any second-order
method on the trie, and the L-BFGS mode plus `agpt_grad_check.py` are
ready as its acceptance test (target: FD ratio 1.000 ± 0.005 and
L-BFGS backtrack rate ≈ 1%).

## Next steps (Experiment 1)

- **Coherence-driven cadence.** Replace fixed `partition_depth` with a
  per-subtree split rule: share a subtree while ρ(S) is above a
  threshold, split it otherwise. Cheapest form: compute ρ per root child
  from a fan-out pass every k epochs and choose pd per root. Test at the
  pd=6 recipe on Shakespeare 1M where the pd=6/pd=7 cliff lives.
- **pd schedule.** ρ falls monotonically with training, so early epochs
  tolerate coarse sharing and late epochs need fine splits — or the
  opposite reading via critical batch size (noise scale grows, so late
  training wants bigger aggregates per step). One run each way settles
  it.
- **Cosine vs target-distribution similarity.** Test whether units agree
  because their next-char count distributions are similar rather than
  because their prefixes are; the counts are in the trie already.
- **pd=3 dump** (11.5k units, ~5 GB) to see whether the within/across
  gap grows with depth. Needs a chunked Gram in the analysis script.
- **Explain the ~3% pd=1 vs Σ pd=2 gradient non-additivity.** Forward
  is identical, so it is in the backward's treatment of shared context
  between a unit's nodes. Matters for coherence-driven cadence only if
  it grows with depth; measure it at pd=3.
- The multi-stage population (members diverge, random node updates a
  random member) is deliberately not built yet: after stage 1 the prefix
  Jacobian is no longer shared, and the stage-1 spectrum (PR ~150) does
  not promise a cheap extrapolation.
