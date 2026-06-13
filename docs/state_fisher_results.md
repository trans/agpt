# State-Fisher Results

This note records the first useful separation between trie geometry and model
projection.

## Setup

The current tiny model is:

```text
Embedding -> GRUCell -> linear head W
```

For a trie node `v`, the GRU computes a starting hidden state `h_v` from the
prefix. The node has transition counts `n(v, x)` and empirical target
distribution `q_v`.

The direct state-Fisher diagnostic freezes the model parameters and optimizes
each node hidden state as a free variable:

```text
p_v = softmax(W h_v + b)
g_v = N_v W^T (p_v - q_v)
F_v = N_v W^T (Diag(r_v) - r_v r_v^T) W
delta_h_v = -(F_v + damping I)^-1 g_v
```

`r_v` is either the model distribution `p_v` or the empirical distribution
`q_v`. The strongest runs so far use empirical curvature.

For `step_scale=auto-node`, each node then solves a one-dimensional line search
along `delta_h_v` in logit space:

```text
L_v(eta) = -N_v q_v^T log softmax(z_v + eta W delta_h_v)
```

This is not backtracking. It uses Newton steps on the exact scalar objective,
clamped to `[0, max_auto_eta]`.

## Key Results

All runs below use Tiny Shakespeare, stride-1 circular samples, block size 8,
prefix length 1, hidden size 64, embedding size 64, damping 10, and empirical
state curvature.

| Run | Free-state iterations | max eta | Aggregate trie PPL |
| --- | ---: | ---: | ---: |
| `20260609T211252-7ddef865` | 1 | 32 | `63.52 -> 7.70` |
| `20260609T231230-d4c06161` | 3 | 32 | `63.52 -> 5.12` |
| `20260610T014014-0b465a85` | 5 | 32 | `63.52 -> 4.485` |
| `20260610T014014-c304ecdd` | 8 | 32 | `63.52 -> 4.187` |
| `20260610T014604-a1b30baf` | 8 | 64 | `63.52 -> 4.093` |
| `20260610T014604-10663d21` | 16 | 32 | `63.52 -> 3.994` |

This validates the state-space trie/Fisher geometry: with no parameter update,
node-local Fisher corrections can reach Kneser-Ney-ish perplexity and slightly
cross below it on this depth-8 objective.

## Interpretation

The trie counts are carrying real geometry. The gap is no longer whether a
node-local Fisher step is meaningful; it is how to project the corrected node
states back into a shared `f_theta`.

The earlier body-Fisher experiments tried to move shared GRU parameters
directly with a large aggregate natural step. That was mechanically valid but
not competitive, likely because one shared nonlinear recurrent model cannot
exactly realize all free node-state moves after a single aggregate update.

The next experiment is corrected-logit distillation:

1. Compute normal GRU hidden states for a prefix subtree.
2. Apply direct state-Fisher correction to get corrected hidden states.
3. Convert corrected states to target distributions with the frozen head:
   `softmax(W h'_v + b)`.
4. Train the GRU body to match those corrected distributions with a
   transition-mass-weighted KL objective.

The first implementation is `scripts/run_state_distill.py`. It freezes the head
by default so `W` remains the coordinate system used by the Fisher correction.

Initial result:

| Run | Epochs | State iters | Head | Teacher trie PPL | Seq val PPL |
| --- | ---: | ---: | --- | ---: | ---: |
| `20260610T020025-cdd69959` | 10 | 3 | frozen | `24.80 -> 4.86` at epoch 10 | `23.12` |
| `20260610T023930-fb467b9f` | 5 | 3 | trainable | `10.02 -> 4.04` at epoch 5 | `9.18` |
| `20260610T025319-0c503ea3` | 1 | 3 | trainable, 2 distill steps/subtree | `21.45 -> 4.29` | `12.03` |

The teacher remains strong, and the model projection improves monotonically
from `41.18` to `23.12` validation PPL over 10 epochs, but the GRU is not yet
absorbing the corrected-state geometry efficiently.

Training the head is a major improvement over freezing a random `W`: d128
epoch-1 validation PPL improved from `28.51` with a frozen head to `15.79` with
a trainable head. Reusing each subtree teacher for two Adam steps improved the
d128 trainable-head epoch-1 result further to `12.03`.

## Current Baselines

For context, conventional sequence training with the same GRU family reached:

| Run | Context | Updates | Validation PPL |
| --- | ---: | ---: | ---: |
| `20260609T182235-96580b8b` | 8 | 5,000 | 6.27 |
| `20260609T182825-57460691` | 8 | 50,000 | 5.78 |

The best prefix-subtree parameter-training run before direct state correction
was roughly `8.22` to `8.26` PPL at depth 8.

## Open Questions

- Can corrected-logit distillation transfer the free-state improvement into the
  GRU body?
- Does matching hidden states directly work better than matching corrected
  logits?
- Should `W` remain frozen permanently, or only while the body learns to inhabit
  the corrected state geometry?
- Can suffix/backoff overlays explain the remaining gap to stronger KN-style
  behavior without making the tree memory explode?

## Projection Notes

The first direct state-delta projection experiment implemented the more literal
bridge:

```text
Find delta_theta such that J_theta(prefix_v) delta_theta ~= delta_h_v
```

where `delta_h_v` is the direct state-Fisher move. This is implemented in
`scripts/run_state_delta_project.py`.

Initial d64 result:

| Run | Method | Seq val PPL | Note |
| --- | --- | ---: | --- |
| `20260610T030541-fa1ddd04` | state-delta projection, fixed head | `28.72` | Similar to frozen-head KL distillation |

This appears to hit the same random-head wall as frozen-head distillation. The
projection can move hidden trajectories, but if `W` is a bad fixed coordinate
system, sequence validation remains poor. A naive body-projection plus
head-Fisher update was unstable in the first attempt, so the coupled
body-plus-head natural update remains a longer-term target rather than the next
cheap experiment.

The most promising short-term direction is now a residual tree-prior model:

```text
z_final(v) = z_tree_prior(v) + z_model(v)
```

The tree should carry exact/smoothed local frequency evidence, while the neural
model learns residual corrections, smoothing, and generalization instead of
being forced to rediscover trie statistics from scratch.

## Residual Tree Prior

`scripts/run_residual_prior.py` implements:

```text
z_final(v) = z_tree_prior(v) + z_model(v)
```

The first prior is a suffix/backoff prior built from the training trie. Held-out
evaluation uses train-derived priors only, so validation does not leak
validation counts.

Initial d128, depth-8 results:

| Run | Weighting | Train prior PPL | Val prior PPL | Epoch 5 val residual PPL |
| --- | --- | ---: | ---: | ---: |
| `20260610T032416-4b549234` | raw transition mass | `7.159` | `10.123` | `9.995` |
| `20260610T032743-007894ba` | `N^0.5 * H`, entropy floor `0.05` | `7.159` | `10.123` | `10.059` |
| `20260610T033601-e47a83fb` | band-pass `N/(N+16) * 4H(1-H)` | `7.159` | `10.123` | `10.104` |
| `20260610T033915-14d79f13` | normalized band-pass | `7.159` | `10.123` | `10.104` |

The simple band-pass trust function did not beat raw mass. The hypothesis is
still useful, but the symmetric `4H(1-H)` weighting likely suppresses too much
of the high-mass, low-entropy structure that actually helps character modeling.

## Fisher-Prior Residual

`scripts/run_fisher_residual.py` tests the more direct state-space prior:

```text
h'_v = state_fisher_correct(h_v)
z_prior(v) = stopgrad(W h'_v + b)
z_final(v) = z_prior(v) + z_residual_model(v)
```

The residual head is initialized to zero, so the starting predictor is exactly
the detached Fisher prior. This first version reports train-trie PPL only,
because the Fisher prior is computed from node counts and a clean held-out
version needs train-derived priors for validation contexts.

Initial result:

| Run | Dim | State iters | Train PPL |
| --- | ---: | ---: | ---: |
| `20260610T034848-b91f6d12` | 64 | 3 | `5.157 -> 4.782` |
| `20260610T035227-e238437f` | 128 | 3 | `4.951 -> 4.587` |
| `20260610T041249-0b04aaaa` | 128 | 3 | `4.969 -> 4.326` after 5 epochs |
| `20260610T043323-fa1e265d` | 128 | 3 | `4.357 -> 4.266` with trainable Fisher head |
| `20260610T043857-b5c200bd` | 128 | 3 | `4.287 -> 4.260` with trainable Fisher head and 2 residual steps/subtree |
| `20260610T044859-d3ea6f81` | 128 | 3 | best `4.336 -> 4.165` at epoch 2; later epochs destabilized the prior |
| `20260610T050632-23cfcdec` | 128 | 3 | `4.655 -> 4.215` at epoch 5 with Fisher head LR `3e-4` |
| `20260610T052216-d98d3d80` | 128 | 5 | `4.157 -> 4.082` at epoch 1 with Fisher head LR `5e-4` |
| `20260610T052819-ec07500f` | 128 | 5 | best `4.122 -> 3.995` at epoch 3; epoch 5 ended `4.328 -> 4.041` |
| `20260610T055800-7b0380ff` | 128 | 5, block 16 | best/final `2.166 -> 2.076` at epoch 5 |

This confirms that a learned residual can improve on top of the Fisher
state-space prior without replacing the mass-weighted trie objective.

The trainable Fisher-head run is non-monotonic: the residual helps within each
epoch, but after epoch 2 the next epoch's Fisher prior worsens. This suggests
the head/body learning rate or schedule is too aggressive for repeated
recomputed Fisher priors.

Reducing the Fisher-head LR to `3e-4` stabilizes the later epochs and avoids the
sharp prior blow-up, but the best PPL so far remains the faster-head epoch-2
point.

Increasing state-Fisher iterations from 3 to 5 is a large quality improvement
at d128. The one-epoch runtime rose to `264.5s`, roughly 1.5x the iter3 path,
but the result reached the strongest train-trie diagnostic so far.

The 5-epoch iter5 run crossed below `4.0` on the train-trie diagnostic at epoch
3. This is with block size 8 and a GRU body, so the next critical step is
held-out evaluation with train-derived priors.

The first block-16 run is much slower and larger, but the train-trie diagnostic
improves dramatically: epoch 5 reaches `2.076`. Runtime is roughly 37-40 minutes
per epoch and peak RSS is about `10GB`. This confirms that extra tree depth
contains substantial train-objective signal, while also making optimization and
instrumentation necessary before routine deeper sweeps.

## Held-Out Evaluation

`scripts/eval_fisher_residual.py` evaluates Fisher-residual checkpoints on the
held-out split without using held-out counts to build the Fisher prior.

The evaluator:

1. Rebuilds train-derived context counts from the training split only.
2. Scores held-out samples from the validation split.
3. For each held-out context, looks up the longest matching train-seen suffix
   and uses that train-derived distribution as the Fisher prior evidence.
4. Uses the held-out next character only as the scored target.

Initial block-16 held-out result:

| Checkpoint | Train diagnostic | Held-out prior PPL | Held-out residual PPL |
| --- | ---: | ---: | ---: |
| `20260610T055800-7b0380ff` best | `2.076` | `13.243` | `12.782` |

This means the block-16 training diagnostic is not a held-out generalization
result. The residual improves the train-derived Fisher prior on held-out text,
but the contiguous last-10% split is much harder than the train objective.

Follow-up diagnostics separate three quantities:

| Eval slice | Count prior PPL | Fisher prior PPL | Fisher+residual PPL |
| --- | ---: | ---: | ---: |
| first 2k validation samples | `6.601` | `9.395` | `9.137` |

The backed-off train count rows are therefore useful on held-out text. The
larger failure is that the current Fisher state/head projection does not
faithfully preserve those count-row distributions when reused as a prior.

Backoff-depth breakdown on the same slice shows the same pattern. Dropping 1-7
characters still leaves count-prior PPL around `7.0`, while the Fisher prior is
around `12-13`. Using the matched backoff suffix hidden state instead of the
full held-out hidden state did not help (`9.473 -> 9.217`), so the immediate
issue is not simply a hidden-state/context mismatch.

`scripts/run_prior_fidelity.py` is the next diagnostic. It freezes the body and
residual head, then trains only the Fisher head to imitate backed-off
train-count rows under forced suffix drops. This tests whether
`head(state_fisher_correct(h, q_backoff))` can approach the raw count-prior
distribution before adding more residual/body learning.

Smoke result on prefix `Z`, forced drops 1-8:

| Run | Prefixes | Count build | Subtree runtime | Teacher PPL |
| --- | ---: | ---: | ---: | ---: |
| `20260610T123042-ed5564e3` | 1 | `83.70s` | `0.38s` | `62.872 -> 8.670` |
| `20260610T124434-ae7b692c` | 1 | `12.90s` | `0.37s` | `62.872 -> 8.670` |
| `20260610T130339-00c5967c` | 3 (`a/e/t`) | `55.82s` | `531.77s` | `57.022 -> 6.826` |

The smoke test validates the path but exposes a Python bottleneck: the first
count builder spent over a minute scanning stride-1 samples. Replacing that
with exact NumPy packed-context sorting reduces the same count build to
`12.90s`. The actual subtree/Fisher work is tiny for this case.

The first dense-prefix fidelity pass trained only prefixes `a/e/t`. It fits the
forced-backoff teacher objective on those prefixes, but the 2k held-out
evaluation barely moves:

| Checkpoint | Count prior PPL | Fisher prior PPL | Fisher+residual PPL |
| --- | ---: | ---: | ---: |
| before fidelity | `6.601` | `9.395` | `9.137` |
| after `a/e/t` fidelity | `6.601` | `9.366` | `9.115` |

This weak movement suggests the held-out projection gap is not solved by a
small amount of head-only calibration on a few prefixes.

## Direct Tree-Prior Residual

The cleaner prior architecture is:

```text
z_final(v) = log q_tree(x | context_v) + z_residual_theta(v)
```

Here the tree prior is explicit count/backoff evidence, not reconstructed
through `head(state_fisher_correct(...))`. Fisher can still be used later as an
optimizer for the residual or reconciliation path, but it is not required for
the prior to express its own distribution.

`scripts/run_direct_prior_residual.py` implements a prefix-subtree version. To
avoid the residual learning an identity problem on exact train rows, training
uses forced suffix drops: the target is the full train node row, while the prior
comes from a shorter train suffix row.

On the same first-2k validation slice:

| Run | Train prefixes | Epochs | Held-out prior PPL | Held-out residual PPL |
| --- | --- | ---: | ---: | ---: |
| `20260610T133723-bbf59635` | `Z` | 1 | `6.601` | `6.585` |
| `20260610T133748-2b5b2583` | `a/e/t` | 1 | `6.601` | `6.539` |
| `20260610T133932-926b7cfc` | `a/e/t` | 5 | `6.601` | `6.394` |
| `20260610T134344-8f27097a` | space + `etaoin` | 5 | `6.601` | `6.314` |
| `20260610T135803-d7dc0a42` | all, depth 16 | 5 | `6.601` | `6.356` |
| `20260610T142025-dc03094f` | all, depth 8 | 5 | `6.191` | `6.350` |
| `20260610T142511-281a289b` | all, depth 8, scale `0.25`, LR `3e-4` | 5 | `6.191` | `6.157` |

This is the first held-out result in this line that behaves as expected: the
explicit tree prior starts at the known count-prior baseline, and the neural
residual improves it rather than degrading it through a Fisher/head projection
bottleneck.

The space + `etaoin` run covers `493,195` of `1,003,854` training samples by
first-character subtree. It is still a partial-tree run, but it is no longer a
tiny diagnostic. Runtime after count construction is about one minute per epoch
for these seven high-mass prefixes.

The all-prefix depth-16 run improves the train forced-backoff objective strongly
but does not beat the partial high-mass prefix run on the first 2k validation
slice. The depth-8 run is much faster (`11.31s` count build and roughly
`25s/epoch`) and has a stronger direct held-out prior (`6.191`), but the
head-only residual worsens it to `6.350`. That suggests the residual schedule or
initialization needs more care at shallow depth; the explicit prior itself is
useful.

A conservative depth-8 residual schedule fixes the immediate overcorrection:
`residual_scale=0.25` and LR `3e-4` improves the held-out slice from `6.191` to
`6.157`. The gain is small, but it confirms the residual can help when kept
close to the explicit prior.

## Return To AGPT Training

The tree-prior residual path is useful as a diagnostic, but it is not
competitive with ordinary SGD on a small attention model. The active direction
returns to AGPT as a training algorithm: bounded subtries provide aggregated
training units, and Fisher/natural-gradient machinery updates the model rather
than representing the prior.

Depth-8 GRU bounded-subtree results:

| Run | Body | Head | Passes | Held-out sequential PPL |
| --- | --- | --- | ---: | ---: |
| `20260610T143550-5214fe74` | AdamW GRU | Fisher head | 1 | `15.76` |
| `20260610T144326-16e08e58` | AdamW GRU | Fisher head | 5 | `11.96` |
| `20260610T144847-c71c605d` | body Fisher GRU, `a/e/t` only | Fisher head | 3 prefix updates | `16.16` |
| `20260610T152023-b1a8dfe0` | body Fisher GRU | Fisher head | 1 | `2287.49` |

The body-Fisher smoke confirms it runs, but it is much heavier: the `a/e/t`
smoke reached roughly `9GB` RSS. The all-prefix body-Fisher pass is unstable
under the current schedule: it reaches `2287.49` held-out PPL after one pass,
while the AdamW-body run reaches about `15.76`. The main body in these runs is
still GRU; the natural-gradient part is the output head unless
`--body-optimizer fisher` is explicitly enabled.

The bridge-literal state-target variant was added to
`scripts/run_state_distill.py` with `--distill-target hidden` and
`--head-fisher-after-epoch`. This computes direct state-Fisher corrected hidden
states at trie nodes, trains the GRU body toward those corrected states with
AdamW, then applies one merged Fisher head update after the whole prefix sweep.
This avoids the failed direct pullback of Fisher through GRU parameters.

Depth-8 hidden-target state distillation:

| Run | State step | Head update | Passes | Held-out sequential PPL |
| --- | --- | --- | ---: | ---: |
| `20260610T162957-9ff6741b` | `0.05`, 1 state iter | merged head, scale cap `0.25` | 1 | `21.44` |
| `20260610T163158-0c7fff20` | `0.1`, 1 state iter | merged head, scale cap `0.25` | 1 | `20.97` |
| `20260610T163304-32b7fdbe` | `0.1`, 1 state iter | merged head, scale cap `0.25` | 5 | `13.25` |
| `20260610T164514-c749682d` | whole tree, `0.1`, 1 state iter | merged head, scale cap `0.25` | 1 | `21.96` |
| `20260610T164620-ef33f7a2` | whole tree, `0.1`, 1 state iter | merged head, scale cap `0.25` | 5 | `14.04` |

The run is stable and follows the Fisher bridge more literally, but it is still
weaker than direct AdamW-body/Fisher-head subtree training (`11.96` after five
passes). Per-prefix head Fisher was tested and is not safe: with only partial
prefix evidence it drove held-out PPL to infinity on a four-prefix smoke test.
The head evidence needs to be merged after a full prefix sweep, or guarded by a
real acceptance test.

Whole-tree mode (`--prefix-length 0`) removes the prefix-shard schedule
entirely: one depth-8 trie is materialized, state-Fisher targets are computed on
that trie, the GRU body is distilled once, and the head gets one merged Fisher
step. It is cleaner but uses more memory (`~6.4GB` RSS for depth 8) and did not
beat the prefix-sweep variant in the first 5-pass run.

## Pure Book State Model

`scripts/run_book_state_model.py` implements the literal one-head derivation:
there is no GRU body. The model is just a persistent node-state table
`E[node]`, one global output matrix `W`, and a bias. Training alternates direct
state-Fisher updates on `E` with one merged Fisher update on `W,b`. Held-out
evaluation uses longest-suffix lookup/backoff into the train trie.
The script now saves a checkpoint, and `scripts/generate_book_state.py` performs
the same longest-suffix lookup for generation.

Depth-8, d64, stride-1 whole train trie:

| Run | Epoch | Train PPL after head | Held-out PPL |
| --- | ---: | ---: | ---: |
| `20260610T170305-67e61db9` | 1 | `15.98` | `52.53` |
| `20260610T170305-67e61db9` | 3 | `8.75` | `34.95` |
| `20260610T170422-c263f4a3` | 6 | `5.86` | `25.14` |
| `20260610T170422-c263f4a3` | 20 | `4.94` | `31.50` |

This confirms the pure bridge model is easy to optimize locally: train PPL
falls rapidly and reaches the KN-like range on the materialized training trie.
But held-out PPL improves only briefly and then worsens while train PPL
continues down. The literal node table overfits without smoothing, suffix
sharing, or a shared body/reconciliation mechanism. This is useful: the clean
math works as a local optimizer, but generalization requires a prior over node
states rather than independent free states.

Generation from the epoch-6 checkpoint
`20260610T172103-678dc456_book_state_b8_d64_e6_ckpt.pt` works mechanically, but
the text is poor. A 600-token sample from prompt `ROMEO:\n` at temperature `0.8`
and top-k `20` had mean suffix depth `5.23` with max depth `8`. This confirms
that inference is presently trie lookup plus backoff; without smoothing, many
contexts use a partially matched suffix state that was optimized independently
for its own train row.

## Reconciliation Through A Transition Model

`scripts/run_reconciled_transition_model.py` is the first direct test of
child-to-parent evidence reconciliation. It uses a deliberately weak transition
model:

```text
e_child = e_parent + r[token]
```

For each node, local state gradient and diagonal state Fisher are computed from
the trie row. Evidence is then accumulated upward through the tree. Because this
transition has identity Jacobian with respect to both parent state and token
residual, child evidence can be pulled to the parent and to shared token
residuals by summation. The resulting global root/token update is accepted only
if a backtracking line search improves the actual trie loss.

The first unguarded run exploded immediately: train PPL went `64.85 -> 645.06`
after the state update in epoch 1. With line search, the same model becomes
stable:

| Run | Epoch | Train PPL after head | Held-out PPL |
| --- | ---: | ---: | ---: |
| `20260610T180842-19f69eb8` | 1 | `36.64` | `59.06` |
| `20260610T180842-19f69eb8` | 2 | `26.18` | `28.89` |
| `20260610T180842-19f69eb8` | 5 | `21.57` | `21.24` |
| `20260610T180842-19f69eb8` | 9 | `19.67` | `19.95` |
| `20260610T180842-19f69eb8` | 10 | `19.35` | `20.15` |

This is not competitive, but it is qualitatively different from the free node
table: train and held-out stay close because the transition model is heavily
shared. The current transition is too weak and mostly order-insensitive, but the
experiment validates the reconciliation pathway and shows that line-search or
trust control is required for aggregated upward Fisher messages.

## Distribution-Only Reconciliation

`scripts/run_distribution_reconcile.py` is the simplest possible version of the
original model-reconciliation idea. There are no hidden states, no `W`, no
Fisher, and no deltas. Each node starts with its empirical next-character count
row. Processing goes leaf-to-root: each child passes reconciled pseudo-counts to
its parent, and the parent combines local counts with child pseudo-counts. Every
node keeps its reconciled distribution for inference via longest-suffix trie
lookup.

Depth-8, stride-1, whole train trie:

| Run | Smoothing | Child scale | Train local PPL | Train reconciled PPL | Held-out local PPL | Held-out reconciled PPL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `20260610T185433-6880e85b` | `0.001` | `1.0` | `4.91` | `8.90` | `35.88` | `40.38` |
| `20260610T185519-6bc98b72` | `0.001` | `0.25` | `4.91` | `5.57` | `35.88` | `33.96` |
| `20260610T185519-8c939575` | `0.001` | `0.1` | `4.91` | `5.14` | `35.88` | `33.58` |
| `20260610T185601-5e151776` | `0.001` | `0.03` | `4.91` | `4.97` | `35.88` | `33.98` |
| `20260610T185601-be15fbcd` | `0.001` | `0.01` | `4.91` | `4.93` | `35.88` | `34.47` |
| `20260610T185645-b5394471` | `0.01` | `0.1` | `5.16` | `5.39` | `27.44` | `26.75` |
| `20260610T185727-846761b4` | `0.01` | `0.03` | `5.16` | `5.22` | `27.44` | `26.86` |
| `20260610T185727-3eccb600` | `0.01` | `0.25` | `5.16` | `5.83` | `27.44` | `27.10` |

The naive full child merge is harmful: it over-smooths toward descendant
distributions and worsens held-out PPL. A weak child contribution helps
slightly. Additive smoothing dominates the absolute held-out number, but the
best reconciled variant still improves over its local-count baseline
(`27.44 -> 26.75`). This confirms the reconciliation mechanics in the simplest
non-neural setting, while showing that child model influence needs a trust
weight rather than raw subtree mass.

## Progressive Logit-Model Reconciliation

`scripts/run_progressive_logit_reconcile.py` is the first minimal prototype that
reconciles actual model parameters rather than distributions or pseudo-counts.
The local model at each node is a next-character logit vector. Processing goes
leaf-to-root:

1. receive already-trained child logit models,
2. precision-merge them with a weak unigram prior,
3. locally train the parent logit vector toward the parent's own trie row,
4. pass that trained parent logit model onward.

This is intentionally not equivalent to one fixed global Fisher step, because
the parent locally adapts after receiving child models.

Depth-8, stride-1, one local natural-style logit step:

| Run | Child weight | Train PPL | Held-out PPL |
| --- | ---: | ---: | ---: |
| `20260610T193817-2722d2db` | `0.1` | `7754.04` | `23.59` |
| `20260610T194009-4b09aaec` | `0.1`, LR `0.25` | `48.52` | `23.57` |
| `20260610T194009-22855078` | `0.03` | `181.97` | `22.79` |
| `20260610T194148-6067614a` | `0.01` | `52.56` | `22.35` |
| `20260610T194331-9bb0eb27` | `0.003` | `18.61` | `21.91` |
| `20260610T194517-baad6144` | `0.001` | `12.53` | `21.74` |
| `20260610T194708-e05a16de` | `0.0` | `8.01` | `21.37` |

The prototype is useful but not yet positive evidence for child model
reconciliation: the zero-child ablation is best. The gain over the unigram
baseline (`28.43 -> 21.37`) comes from local logit-model fitting at each node,
not from child messages. Small child weights degrade gently; larger child
weights can destroy train PPL while still keeping held-out reasonable. This
suggests the next model-merge test needs a better trust/compatibility rule for
child models, not raw precision-weighted logit averaging.

## Shared-Basis Head Adapter Reconciliation

`scripts/run_shared_basis_adapter_reconcile.py` is the first LoRA-like adapter
prototype. It uses a fixed additive token state and a shared random basis. Each
node carries a small adapter matrix `A` in that shared basis; logits are:

```text
logits_v = unigram_logits + A_v phi_v
```

The model object being reconciled is therefore an actual adapter parameter
matrix, not counts or free logits. Parents precision-merge child adapters, then
locally train the parent adapter on its own row.

Depth-8, stride-1:

| Run | Rank | Child weight | Train PPL | Held-out PPL |
| --- | ---: | ---: | ---: | ---: |
| `20260610T201356-a7661a60` | 4 | `0.001` | `19.21` | `27.50` |
| `20260610T201616-7deba611` | 4 | `0.0` | `19.03` | `27.50` |
| `20260610T201831-dd03bf12` | 8 | `0.0` | `18.52` | `26.90` |
| `20260610T202711-861177d0` | 8 | `0.0`, 5 persistent passes | `18.52` | `26.90` |

This is technically closer to a scalable adapter model, but the first version is
weak. Child reconciliation does not help at rank 4, and increasing rank only
modestly improves held-out PPL. The likely bottleneck is the fixed random
additive state/basis; adapters need a useful shared feature space before model
reconciliation can matter. This points back toward either learned states or a
real base model feature extractor before LoRA-style merging.

Adding epoch persistence, where the previous root adapter becomes the prior for
the next pass, produced a flat curve for rank 8. The local adapter solve
overwhelms the weak prior, so the same per-node adapters are recovered each
epoch. A persistent adapter experiment will need either a stronger prior/trust
schedule or actual learned/global feature updates between passes.

A tiny attention baseline was added in `scripts/run_attention_sequence.py`.
Depth-8, d128, 2 layers, 4 heads, 1k ordinary AdamW updates:

| Run | Steps | Held-out window PPL |
| --- | ---: | ---: |
| `20260610T145344-34c0f074` | 1,000 | `7.333` |
| `20260610T145737-f944cbcc` | 5,000 | `6.390` |
| `20260610T150133-d8299c9c`, seq len 16 | 5,000 | `5.600` final, `5.586` best logged |

This baseline is already far stronger than the GRU bounded-subtree run, so the
next AGPT implementation target is attention-body bounded-subtree training.

## Segment Memory Prototype

`scripts/run_segment_memory_model.py` is the first prototype for the corpus-route
idea:

```text
training text -> prefix-count trie -> shortest unique-or-depth-capped segments
segment chars -> GRUCell -> segment terminal hidden state
current segment hidden -> attention over previous segment terminal states
```

This is intentionally not using Fisher, mass weighting, trie reconciliation, or
tree-derived posterior targets yet. It tests the narrower claim that a trie can
produce a useful compressed route vocabulary for a normal neural sequence model.

Initial smoke run:

| Run | Train chars | Depth | Train segments | Mean segment len | Max memory | Final val PPL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `20260611T020424-0cb022c8` | 50,000 | 16 | 7,270 | 6.88 | 64, no RoPE | 14.05 |
| `20260611T023942-9f19cac8` | 50,000 | 16 | 7,270 | 6.88 | 32, RoPE | 14.16 |
| `20260611T024450-2bae7996` | 50,000 | 16 | 7,270 | 6.88 | 32, RoPE, 5 epochs | 10.64 |
| `20260611T030307-9b13665a` | 50,000 | 16 | 7,270 | 6.88 | 32, no RoPE, 5 epochs | 10.71 |
| `20260611T031941-70765756` | 50,000 | 16 | 7,270 | 6.88 | 32, RoPE, 11 epochs | 10.58 best, 10.92 final |
| `20260611T032603-75a3d39e` | 100,000 | 16 | 13,213 | 7.57 | 32, RoPE, 10 epochs | 10.03 best, 10.34 final |
| `20260611T035653-a8cd9038` | 50,000 | 16 | 7,270 | 6.88 | 32, RoPE, fused cap state, 10 epochs | 10.80 best, 11.01 final |
| `20260611T040750-4ae8bef3` | 50,000 | 16 | 7,270 | 6.88 | 32, char-position RoPE, 10 epochs | 10.49 best, 10.99 final |
| `20260611T042115-2b4b5676` | 50,000 | 16 | 7,270 | 6.88 | 32, char-position RoPE, 2 cross-attention blocks | 10.95 best, 12.43 final |
| `20260611T043732-b785a544` | 50,000 | 16 | 7,270 | 6.88 | 32, char-position RoPE, gated residual mixer | 10.63 best, 11.10 final |
| `20260611T044404-2c63a8ca` | 50,000 | 16 | 7,270 | 6.88 | gated diagnostics, 5 epochs | 10.69 final |
| `20260611T122410-c5e5110d` | 50,000 | 16 | 7,270 | 6.88 | embedding 128, hidden 128 | 10.73 best, 11.44 final |
| `20260611T123203-c7f59a6e` | 50,000 | 16 | 7,270 | 6.88 | attention-only readout | 14.35 best, 14.39 final |
| `20260611T124132-6ea3dfdc` | 50,000 | 16 | 7,270 | 6.88 | late char-RoPE diagnostics | 10.41 final |
| `20260611T133131-0bd5c703` | 50,000 | 16 | 7,270 | 6.88 | causal current-segment attention | 10.13 final |
| `20260611T133745-96782cdf` | 50,000 | 16 | 7,270 | 6.88 | current-attn source diagnostics | 10.13 final |
| `20260611T134119-3ff0e897` | 50,000 | 16 | 7,270 | 6.88 | current-only self-attn diagnostic | 10.70 final |
| `20260611T134449-d43b7d56` | 50,000 | 16 | 7,270 | 6.88 | prefix-attn, no current self key | 10.37 final |
| `20260611T142905-3e0540f8` | 50,000 | 16 | 7,270 | 6.88 | carried GRU hidden, no attention | 8.27 final |
| `20260611T143349-1d76c6d1` | 50,000 | 16 | 7,270 | 6.88 | carried GRU hidden + late attention | 8.86 best, 9.15 final |
| `20260611T145057-51ebc8b8` | 50,000 | 16 | 7,270 | 6.88 | carried GRU hidden + prefix-attn | 8.78 best, 8.82 final |
| `20260611T152531-6dbfc801` | 50,000 | 16 | 7,270 | 6.88 | d64 carried GRU hidden, no attention | 8.83 final |
| `20260611T152838-d1763c84` | 50,000 | 16 | 7,270 | 6.88 | d64 carried GRU hidden + prefix-attn | 9.57 best, 9.58 final |
| `20260611T153946-1b5df247` | 50,000 | 16 | 7,270 | 6.88 | d32 carried GRU hidden, no attention | 10.50 final |
| `20260611T154804-c88e2ce7` | 50,000 | 16 | 7,270 | 6.88 | d32 carried GRU hidden + prefix-attn | 10.72 final |
| `20260611T161528-b61460d0` | 50,000 | 16 | 7,270 | 6.88 | carried GRU hidden + attention-only readout | 12.56 best, 12.68 final |
| `20260611T162444-ca89deec` | 50,000 | 16 | 7,270 | 6.88 | carried prefix-attn + gated boundary feedback | 9.12 best, 9.64 final |
| `20260611T163940-888f129c` | 50,000 | 16 | 7,270 | 6.88 | per-token context feedback + input RoPE | 26.92 final |
| `20260611T170619-fe5de429` | 50,000 | 16 | 7,270 | 6.88 | per-token context feedback + raw fallback | 26.90 final |
| `20260611T173040-1b74c8cd` | 50,000 | 16 | 7,270 | 6.88 | per-token gated feedback + input RoPE | NaN |
| `20260611T174321-a2cfa175` | 50,000 | 16 | 7,270 | 6.88 | stable per-token gated feedback, 2 epochs | 10.35 at epoch 2 |
| `20260611T181536-7876769b` | 50,000 | 16 | 7,270 | 6.88 | stable per-token gated feedback, 10 epochs | 8.53 best, 8.60 final |
| `20260611T191031-94883771` | 100,000 | 16 | 13,213 | 7.57 | carried GRU hidden, no attention | 7.80 final |
| `20260611T191910-edc841fc` | 100,000 | 16 | 13,213 | 7.57 | carried GRU hidden + prefix-attn | 8.32 best, 8.51 final |
| `20260611T201731-bc9158cd` | 50,000 | 16 | 7,270 | 6.88 | dual-V per-token gated feedback, 3 epochs | 9.63 at epoch 3 |

These runs are not benchmarks. They used a small slice and CPU execution, but
the validation curves moved from roughly `26.3` at step 20 to about `14.1` at
the end of epoch 1, so the segment-memory path is learning rather than being
dead on arrival.

The second smoke adds segment-age RoPE to the attention query/key vectors and
uses a 32-segment attention window. It is effectively tied with the first smoke,
so RoPE is architecturally cleaner but not an immediate quality win in this
minimal model.

The 5-epoch RoPE run continued improving:

```text
epoch 1: 14.16
epoch 2: 12.02
epoch 3: 11.18
epoch 4: 10.81
epoch 5: 10.64
```

So the one-epoch smoke was not near the floor, but the curve is flattening far
above the ordinary sequence baselines. The likely missing piece is not merely
segment-position information; it is training the inference/composition operator
that combines local segment state, previous segment states, and fallback evidence.

A matched 5-epoch no-RoPE ablation finished at `10.71`, effectively tied with
RoPE's `10.64`. Segment-age RoPE is therefore not the current limiting factor.

Vectorizing the per-segment GRU pass preserved the epoch-1 result (`14.16`) and
reduced 50k epoch runtime from roughly `142s` to `31s`. With that speedup, the
50k run plateaued around `10.6`, while a 100k run reached a better best of
`10.03` at epoch 5 before overfitting/degrading. More data helps, but the current
architecture still plateaus far above ordinary sequence baselines.

A simple fused cap-state variant stored `LayerNorm(tanh(W[terminal, context]))`
instead of the raw segment terminal state. This made memory recursive in
principle, but degraded results (`10.80` best vs `10.58` best on the 50k
terminal-state run). The likely issue is that this overwrites a clean local
terminal state with a randomly initialized transformed state. A safer variant
would need to be residual/gated and initialized near the terminal-state identity.

Changing RoPE coordinates from segment index to actual character position
helped modestly: 50k best improved from `10.58` to `10.49`. This suggests the
earlier "RoPE does nothing" result was partly caused by using the wrong position
coordinate for variable-length segment memory. It still did not break the
plateau.

A two-block cross-attention mixer was negative. It updates token states with
residual attention/MLP blocks before prediction, but predicts from the mixed
state only; best PPL worsened to `10.95` and later epochs degraded badly. This
suggests the strong direct GRU path should be preserved explicitly, with memory
used as a gated/residual augmentation rather than replacing the prediction
state.

The gated residual mixer preserved the raw GRU path and injected a learned
memory update:

```text
state = raw + sigmoid(gate(raw, context)) * update(raw, context)
logits = head([raw, state, context])
```

This was also negative (`10.63` best). Diagnostics show the gate is not simply
closed: mean gate rose from `0.0506` to `0.0648` over five epochs, and mean
gated-update norm rose from `2.81` to `7.46`. The memory branch is active, but
the learned update is not improving the plateau.

Matching embedding size to hidden size (`128/128`) was neutral-to-negative: it
started faster but plateaued worse than the `64/128` baseline. An attention-only
readout, where the GRU can only organize states and logits are predicted from
attention context, learned but plateaued around `14.35`. This shows the memory
path can carry predictive signal, but it is much weaker than direct local GRU
state access in the current design.

Late char-RoPE diagnostics show the memory path is not decorative:

```text
epoch 5 full PPL:         10.41
epoch 5 no-memory PPL:    14.85
epoch 5 context-only PPL: 22.71
epoch 5 first-token PPL:  16.22
epoch 5 later-token PPL:   9.32
```

The attention top weight rises from `0.18` to `0.33`, and mean attended
character distance falls from `83` to `75` over five epochs. The model is
learning to retrieve more focused and more local route memory. The main weakness
is segment-boundary prediction: the first token of each segment remains much
harder than later tokens, which suggests the route-memory handoff between
segments is the current failure point.

Adding current segment prefix states into the attention memory, causally, is the
first clearly positive architecture change after char-position RoPE:

```text
late char-RoPE epoch 5 PPL:           10.41
current-segment attention epoch 5:    10.13
late first-token/later-token PPL:     16.22 / 9.32
current-attn first/later PPL:         15.00 / 9.19
```

The gain is concentrated at the segment boundary, which supports the diagnosis
that the previous memory field was too impoverished. Attention also becomes much
more focused: top weight reaches `0.53`, entropy drops to `1.67`, and mean
attended character distance drops to `39` by epoch 5.

Source-split diagnostics clarify that current-attn is not simply deleting route
memory:

```text
current-attn epoch 5 PPL:    10.13
previous-memory weight:       0.599
top key is previous memory:   0.519
```

However, allowing the current token state to be a key creates a self-attention
shortcut. A current-only diagnostic, with no previous segment memories, still
reached `10.70` and had top attention weight `0.975`, meaning it mostly attended
to the current GRU state itself. A stricter prefix-attn variant excludes that
self key; it reached `10.37`, with previous-memory weight `0.694` and previous
memory as the top key `0.639` of the time. This is weaker than self-allowed
current-attn but still slightly better than previous-only late attention, and it
confirms previous segment memories are still actively used.

The larger correction is that the GRU had been reset to zero at every segment
boundary. Carrying the GRU hidden state from one segment to the next, while
still detaching at optimizer chunk boundaries, immediately beat all reset-GRU
attention variants:

```text
carried GRU, no attention:
epoch 1: 13.30
epoch 5:  8.99
epoch 10: 8.27
```

This means previous attention experiments were mostly trying to patch an
artificial recurrence break. Attention should now be evaluated only as an
incremental gain over this coherent carried-GRU route baseline.

The first incremental attention test was negative. Adding previous-segment
attention to the carried GRU helped only in early epochs, then degraded:

```text
carried GRU only epoch 10:        8.27
carried GRU + attention best:     8.86
carried GRU + attention epoch 10: 9.15
```

So on the 50k Tiny Shakespeare slice, simple terminal-state attention was mostly
acting as a patch for the old reset-GRU boundary problem. Once the GRU has a
coherent route state, this attention branch adds noise/overfitting rather than
useful long-context signal.

Carried prefix-attn, which adds current segment prefix states without allowing a
self-attention key, is slightly better than carried late attention but still
worse than carried GRU alone:

```text
carried GRU only epoch 10:          8.27
carried prefix-attn best/final:     8.78 / 8.82
```

It uses previous segment memories heavily (`prev_w` around `0.72` by epoch 10),
but that usage does not translate to better validation PPL on this slice.

Shrinking the model state from d128 to d64 did not make attention useful:

```text
d128 carried GRU only:          8.27
d64 carried GRU only:           8.83
d64 carried GRU + prefix-attn:  9.57
```

The smaller GRU has less capacity and bottoms out worse, while prefix attention
still degrades validation PPL. This argues against the simple explanation that
d128 was too large and crowded out attention.

At d32, prefix attention helps early but still does not beat the carried-GRU
baseline at the bottom:

```text
d32 carried GRU only:          10.50
d32 carried GRU + prefix-attn: 10.72
```

So shrinking state size increases the early usefulness of attention, but not the
best validation result on this slice.

Attention-only readout with carried GRU, where the prediction head sees
`context_G` but not `h_G`, bottomed around `12.56`. It continued improving until
epoch 9 and then worsened. This confirms that attention context alone carries
signal, but it is much weaker than the carried GRU state for next-character
prediction in this setup.

Gated segment-boundary feedback routes attention into the recurrent state for
the next segment:

```text
next_hidden = terminal + gate * update(terminal, context_last)
logits = head([h_t, context_t])
```

This also failed to beat the carried-GRU baseline:

```text
carried GRU only:                  8.27
carried prefix-attn:               8.78
carried prefix-attn + feedback:    9.12 best, 9.64 final
```

So this conservative feedback route does not solve the mismatch. It may be too
weak, or the terminal/context attention signal is simply not useful enough on
this data slice.

Direct per-token context feedback was also negative:

```text
next_hidden = context_t
logits = head([raw_h_t, context_t])
```

With input RoPE enabled, this collapsed because early/empty contexts are zero
and the recurrent state gets wiped out. Validation PPL stayed near `27-30`, and
mean context norm remained `0.0`. Future per-token feedback tests need a
residual/gated update or a nonzero fallback state; direct replacement is not
viable.

Adding a fallback so the first empty-context step uses `raw_h_t` did not rescue
direct context feedback. The run still ended at `26.90`, and mean context norm
remained near zero. In the per-token loop the current-prefix entries are built
from the feedback state; once direct feedback collapses to weak context states,
the prefix entries also become weak. Direct replacement remains non-viable.

The first per-token gated feedback run with input RoPE was numerically unstable
and produced NaNs. A conservative version disabled input RoPE, initialized the
feedback gate smaller, and norm-capped the recurrent delta:

```text
next_hidden = raw_h_t + clipped(gate * update(raw_h_t, context_t))
```

That version was stable and promising early:

```text
epoch 1: 12.78
epoch 2: 10.35
mean gate: 0.0033 -> 0.0054
mean delta norm: 0.47 -> 0.63
```

It is too slow in the current Python loop for broad sweeps, but it is the first
per-token feedback variant that did not collapse and that improved over the
carried-GRU-only curve at epoch 2.

The 10-epoch version kept a small early lead but bottomed above the carried-GRU
baseline:

```text
carried GRU only:                  8.27 final
stable per-token gated feedback:   8.53 best, 8.60 final
```

The gate continued opening slowly (`0.0033 -> 0.0306`) and the clipped delta
norm approached the cap (`0.47 -> 0.99`). This confirms stable feedback is
possible, but the current attention signal still does not improve the 50k-slice
floor.

Scaling the corrected carried-GRU baseline from 50k to 100k train characters was
a large win:

```text
50k carried GRU, epoch 10:   8.27
100k carried GRU, epoch 10:  7.80
```

The 100k run was still descending at epoch 10. This means the 50k slice was
holding back absolute PPL substantially. It also raises the comparison bar for
attention/feedback variants: they should be tested against the 100k carried-GRU
baseline, not the older reset-GRU runs.

The cheap 100k attention comparison was negative:

```text
100k carried GRU only:           7.80 final
100k carried GRU + prefix-attn:  8.32 best, 8.51 final
```

As on 50k, prefix attention helped early (`10.63` vs `11.05` at epoch 1) but
then flattened and degraded above the no-attention carried-GRU baseline.

The first dual-value feedback test separated prediction values from recurrent
feedback values:

```text
context_pred = weights @ V_pred(memory)
context_rnn  = weights @ V_rnn(memory)
logits = head([raw_h_t, context_pred])
next_hidden = raw_h_t + clipped(gate * update(raw_h_t, context_rnn))
```

It was stable and helped versus carried GRU early, but did not beat the simpler
single-value gated feedback:

```text
carried GRU only epoch 3:        9.87
single-V gated feedback epoch 3: 9.51
dual-V gated feedback epoch 3:   9.63
```

The recurrent feedback delta started smaller in the dual-V run, so the first
version may be too conservative, but the basic separation did not immediately
improve the curve.

The remaining compute bottleneck is Python-level segment iteration plus repeated
attention over segment memory. The remaining modeling bottleneck appears more
important: the stored memory state is still only the segment-local terminal GRU
state, not a fused cap state that writes route/context composition back into
memory.

An attention-final-say gated feedback smoke test was negative:

```text
50k carried GRU + prefix memory + token gated feedback + attn-only head:
epoch 2: 14.18
```

This removed the direct GRU prediction vote (`logits = attn_head(context)`)
while still letting the GRU produce states for attention to read. The result was
much worse than both carried GRU and gated feedback with a direct GRU path. The
interpretation is that attention cannot be given final say over raw GRU hidden
states and expected to recover useful retrieval records automatically. The next
principled variant should separate the recurrent encoder state from an explicit
memory record written for retrieval.

The first explicit memory-record split added:

```text
h_t = GRU state
r_t = memory_writer(h_t)
q_t = query(h_t)
k_i, v_i = key(r_i), value(r_i)
logits = attn_head(context_t)
```

On a 10k-character smoke test with current-prefix attention and no direct GRU
logit vote, written records beat raw GRU records but remained weak:

```text
raw records:      29.86 -> 27.28 -> 24.78 -> 22.91 -> 21.83
written records:  29.56 -> 24.59 -> 21.14 -> 19.41 -> 18.77
```

So the writer helps, but this is not yet a competitive cooperative model. It
still lacks a direct training pressure that makes memory records good retrieval
objects rather than merely hidden-state transforms.

Adding a small auxiliary next-token loss on the written record itself helped the
early curve but not the short-run floor:

```text
written records + aux 0.25: 27.22 -> 22.01 -> 20.68 -> 19.23 -> 18.99
```

This was better than plain written records at epochs 1-2, but slightly worse by
epoch 5 (`18.99` vs `18.77`). The result suggests the problem is not merely that
the writer lacks a local predictive objective. The attention model also needs a
better way to compose/read records, or the recurrent encoder needs to write a
different kind of retrieval object than a next-token predictive state.

A first retrieval-specific auxiliary objective trained each segment terminal
query to classify the immediately previous segment record within the memory
window. This was negative:

```text
written prefix-attn-only + previous-record aux 0.25:
30.77 -> 27.48 -> 24.93 -> 23.38 -> 21.63
```

It made attention too peaky and did not help the prediction objective. A weaker
version with the corrected current-token attention path was also slightly worse:

```text
written current-attn-only-final:                    27.58 -> 18.68 -> 15.84 -> 14.69 -> 14.02
written current-attn-only-final + retrieval 0.05:   28.92 -> 19.57 -> 15.95 -> 14.85 -> 14.14
```

The bigger correction was allowing the attention-final-say model to read the
current token's written record. Excluding the current record forced attention to
predict the next character from older records only, which was too severe. With
current-record access, written records became competitive with the carried-GRU
baseline on the 10k smoke test:

```text
10k carried GRU baseline:                       24.40 -> 18.66 -> 16.27 -> 15.14 -> 14.40
10k raw current-attn-only-final records:        29.80 -> 27.10 -> 23.92 -> 21.04 -> 19.12
10k written current-attn-only-final records:    27.58 -> 18.68 -> 15.84 -> 14.69 -> 14.02
```

This is the first attention-final-say variant that beat the carried-GRU baseline
in a matched smoke test. However, its attention mostly selects the current
written record (`prev_w` around `0.04` by epoch 5), so it proves the writer and
attention-head interface can work, not that long-range memory is useful yet.

The same current-record written attention-final-say model on 50k characters did
not beat the carried-GRU baseline:

```text
50k written current-attn-only-final, 3 epochs: 12.32 -> 10.55 -> 10.29
```

The earlier carried-GRU reference was about `9.87` at epoch 3 and `8.27` at
epoch 10. So the 10k win did not scale cleanly. The 50k run still mostly selected
the current written record (`prev_w` around `0.02-0.05`), which means the model is
functioning more like a transformed-GRU readout than a useful segment-memory
reader.

A softer predictive-utility auxiliary was added after the previous-record
retrieval target failed. For each token, allowed candidate records are scored by
how well each record's value vector alone predicts the next character:

```text
candidate_nll_i = CE(attn_head(value_i), target)
target_i = softmax(-candidate_nll_i / temperature)
utility_loss = KL(target || actual_attention)
```

This is aligned with predictive usefulness rather than adjacency. On the 10k
current-attn-only-final written-record setup, it successfully increased mass on
previous-memory records but did not improve PPL:

```text
no utility aux:     27.58 -> 18.68 -> 15.84 -> 14.69 -> 14.02   prev_w: 0.04
utility 0.050:      28.19 -> 24.57 -> 20.40 -> 17.10 -> 16.31   prev_w: 0.49
utility 0.010:      27.84 -> 20.35 -> 16.13 -> 15.02 -> 14.13   prev_w: 0.19
utility 0.005:      27.69 -> 19.37 -> 15.87 -> 14.78 -> 14.18   prev_w: 0.12
```

So the auxiliary can alter routing in the intended direction, but even a small
amount slightly hurts the short-run PPL. The likely interpretation is that the
available previous records are usually not useful enough yet; forcing attention
to read them trades off against the strong current-record signal.

The next correction was to stop asking one attention mechanism to handle both
short-range high-resolution decoding and long-range low-resolution memory. A new
`dual-attn-final` mode uses two separate attention paths:

```text
local_context = causal attention over current segment records
long_context  = attention over previous segment records
logits = dual_attn_head([local_context, long_context])
```

The GRU still produces the representations, but the final prediction comes from
the two attention contexts, not directly from the raw GRU state. On 10k, this
learned much faster early but did not beat the best current-record-only variant:

```text
10k dual-attn-final: 21.87 -> 16.68 -> 14.92 -> 14.40 -> 14.65
```

On 50k, however, the split was clearly positive:

```text
50k written current-attn-only-final, 3 epochs: 12.32 -> 10.55 -> 10.29
50k dual-attn-final, 3 epochs:                  11.47 ->  10.00 ->  9.50
```

The old carried-GRU reference was about `9.87` at epoch 3, so this is the first
attention/memory variant that beats the carried-GRU baseline at matched epoch
count on the 50k slice. This supports the architectural hypothesis: the useful
split is local high-resolution attention plus separate long low-resolution
segment-memory attention, rather than one attention mechanism that must do both
jobs.

A local-only ablation was added to separate "better local readout" from "useful
long memory":

```text
50k local-attn-final, 3 epochs: 11.54 -> 10.30 -> 9.95
50k dual-attn-final, 3 epochs:  11.47 -> 10.00 -> 9.50
```

Most of the early gain comes from the local high-resolution attention path, but
the long low-resolution segment-memory path still contributes about `0.45` PPL
at epoch 3 on the matched setup.

A 10-epoch dual-attention run exposed a plateau/drift problem:

```text
50k dual-attn-final, 10 epochs:
11.47 -> 10.00 -> 9.50 -> 9.29 -> 9.29 -> 9.287 -> 9.52 -> 9.78 -> 10.13 -> 10.58
```

The best value was epoch 6 at `9.287`, after which validation degraded steadily.
So the architecture learns quickly and beats the matched carried-GRU reference
early, but the current training recipe needs early stopping, regularization, or
a better head/optimizer split before longer runs are useful.

The recurrent core was then made selectable with `--rnn-core gru|tanh` to test
whether GRU gating was making poor memory records for attention. The tanh RNN
core is a plain `nn.RNN(..., nonlinearity="tanh")` in the same dual-attention
setup.

On 10k, tanh was better than GRU:

```text
10k GRU dual-attn-final:   21.87 -> 16.68 -> 14.92 -> 14.40 -> 14.65
10k tanh dual-attn-final:  20.05 -> 15.57 -> 13.93 -> 13.60 -> 13.85
```

On 50k, tanh started better but lost by epoch 3:

```text
50k GRU dual-attn-final:             11.47 -> 10.00 -> 9.50
50k tanh dual-attn-final lr=1e-3:     11.11 -> 10.54 -> 10.09
50k tanh dual-attn-final lr=5e-4:     11.63 -> 10.33 -> 9.94
```

So the simple tanh core supports the intuition on small data and early training:
it exposes a cleaner trajectory for attention to read. But with the current
setup, GRU still gives the better matched 50k result. Lowering tanh LR helped,
which suggests more tuning may close part of the gap, but the gating hypothesis
is not yet a win at scale.

The "dual attention" framing was then corrected. The intended architecture is
not two attention mechanisms. It is:

```text
RNN core handles the current/local sequence
attention reads previous segment records only
logits = head([rnn_state, long_context])
```

This is the existing `late` mode. Under that cleaner split, attention does not
recompute the local sequence and is more compatible with an eventual AGPT/RNN
state-passing design.

Results:

```text
10k tanh late: 21.94 -> 17.09 -> 15.23 -> 14.45 -> 14.19

50k tanh late: 12.47 -> 11.01 -> 10.15
50k GRU late:  12.46 -> 10.44 ->  9.46
```

This is important: the simple long-only attention split with GRU slightly beats
the earlier 50k dual-attention result (`9.46` vs `9.50` at epoch 3), while
avoiding local attention recomputation. The tanh core remains interesting on
small data, but under the matched 50k long-only-attention setup, GRU is still
better.

Scaling the clean GRU-local + long-attention setup to 200k train characters was
strong:

```text
200k GRU late, 3 epochs: 9.245 -> 8.299 -> 7.845
```

Run details:

```text
train_chars=200000
train_segments=24706
eval_chars=20000
eval_segments=3400
mean_segment_len=8.10
max_segment_len=16
peak_rss_mb=1148
runtime_sec=482
```

This is substantially better than the 50k matched run (`9.46` at epoch 3), and
epoch 1 on 200k already beats epoch 3 on 50k. More data helps this architecture
cleanly; the previous poor attention behavior was not simply an unavoidable
limitation of segment memory.

The matched 200k tanh-core comparison still lagged GRU:

```text
200k GRU late, 3 epochs:   9.245 -> 8.299 -> 7.845
200k tanh late, 3 epochs: 10.177 -> 9.094 -> 8.808
```

Tanh also scaled, but the gap widened rather than closed. Under the clean
RNN-local/attention-long setup, GRU remains the stronger recurrent core at this
scale.

Extending the 200k GRU late run to 10 epochs showed the first clear bottom:

```text
200k GRU late, 10 epochs:
9.245 -> 8.299 -> 7.845 -> 7.590 -> 7.408 -> 7.305 -> 7.254 -> 7.256 -> 7.270 -> 7.315
```

Best validation PPL was epoch 7 at `7.254`. After that the run drifted upward,
so this configuration bottoms around `7.25` with the current optimizer and eval
slice. The no-memory diagnostic continued improving (`11.637 -> 8.216`), while
the context-only diagnostic stayed poor (`27-30`), indicating the long attention
context is useful only in combination with the recurrent state, not as an
independent predictor.

An attention-interface normalization experiment added `--attn-interface-norm`.
Modes:

```text
none       current baseline
attention  LayerNorm before Q and K/V projections only
layer      LayerNorm before Q and K/V, plus LayerNorm both late-head branches
```

The full `layer` mode helped the 10k smoke but did not beat the 200k baseline:

```text
10k GRU late + layer norm:   18.97 -> 15.76 -> 14.64 -> 14.15 -> 13.98

200k GRU late baseline:      9.245 -> 8.299 -> 7.845
200k GRU late + layer norm:  9.431 -> 8.393 -> 8.003
```

It made the context-only diagnostic much better on 200k (`~22` instead of
`27-30`), but the combined model got worse. That suggests full branch
normalization made the attention representation cleaner while stripping useful
scale information from the recurrent branch/head interface.

The narrower `attention` mode was less promising even on 10k:

```text
10k GRU late + attention norm: 24.70 -> 17.74 -> 16.21 -> 15.19 -> 14.21
```

Removing `LayerNorm(h_t)` from the final head while keeping context branch
normalization was tested as `--attn-interface-norm context`:

```text
10k GRU late + context norm:   23.15 -> 17.77 -> 16.12 -> 15.22 -> 14.57

200k GRU late baseline:        9.245 -> 8.299 -> 7.845
200k GRU late + layer norm:    9.431 -> 8.393 -> 8.003
200k GRU late + context norm:  9.617 -> 8.393 -> 7.942
```

This improved over full branch normalization, so normalizing `h_t` before the
head was indeed part of the problem. But it still did not beat no norm. The
notable clue is diagnostic behavior: context-only PPL improved dramatically
(`~17` instead of `27-30`), while combined PPL still lagged. The attention branch
became much more predictive, but the late head did not combine it with the
recurrent branch better than the raw baseline.

So plain LayerNorm at the RNN/attention boundary is not an immediate win, though
normalizing the context branch exposes a stronger standalone memory signal.

## Backoff-Gated Trie Prior

A first Python prototype of the suffix-link/backoff prior was added as
`scripts/run_backoff_gate_prior.py`. It builds exact tuple-key contexts from the
training text, uses drop-oldest backoff chains:

```text
ABCD -> BCD -> CD -> D -> root
```

and evaluates a product-of-experts prior:

```text
logits = log p_root + sum_d gate(features_d) * log p_d
```

with fixed count/stat tensors and only a 5-parameter logistic gate trainable.
This prototype intentionally does not use the radix/suffix catalog yet; it tests
the math first.

The 200k prior-only baselines were:

```text
root_ppl       28.812
deepest_ppl    13.840
uniform_ppl   224.823
init_gate_ppl  10.130
```

The conservative initialized gate (`bias=-2`) is much better than deepest-only,
while uniform PoE explodes. This confirms that uncontrolled multiplication of
sharp count distributions is unsafe.

Entropy damping was tested structurally:

```text
none:          deepest 13.840, uniform 224.823, init_gate 10.130
entropy a=1.0: deepest 13.414, uniform  48.991, init_gate 13.993
entropy a=0.5: deepest 13.014, uniform  87.314, init_gate 11.949
middle  a=1.0: deepest 17.309, uniform  75.994, init_gate 11.354
middle  a=0.5: deepest 15.440, uniform 111.129, init_gate 10.750
```

Entropy damping helps deepest-only and fixes much of the uniform-PoE explosion,
but it worsens the conservative initialized gate. The simple entropy damper is
not enough by itself.

Training the 5-parameter gate overfits badly even at low LR:

```text
none, lr=0.005:
init 10.130 -> 10.917 -> 11.394 -> 12.361

entropy a=0.5, lr=0.005:
init 11.949 -> 11.099 -> 11.345 -> 11.890
```

Train PPL drops near 1, while validation worsens. The current count prior is too
easy to memorize from train statistics, so any learned gate needs stronger
regularization, heldout-tuned calibration, or a residual setting where the prior
is fixed/conservative and the neural model learns corrections.

## Transformer-Style Fusion Block

A `late-block` mode was added to test a small transformer-style fusion block
after the long-memory attention read:

```text
context = long_attention(hidden_seq, previous_segment_records)
state = hidden_seq
for block in cross_blocks:
    state = LayerNorm(state + context)
    state = LayerNorm(state + MLP(state))
logits = state_head(state)
```

This keeps the RNN as the local sequence model and adds residual+MLP processing
for the long-memory context. It did not beat the simpler `late` head:

```text
10k GRU late-block: 26.09 -> 17.91 -> 15.65 -> 14.84 -> 14.43

200k GRU late baseline:   9.245 -> 8.299 -> 7.845
200k GRU late-block:      9.491 -> 8.556 -> 8.097
```

The block was also slower and the long-context norm collapsed toward a small
value (`~2` on 200k), suggesting this simple residual block suppresses the memory
branch rather than improving the combination. The next transformer-style test
should probably be a real local transformer core or a stronger cross-attention
block, not just post-attention fusion.

An `attention-decision` mode tested an attention-owned final state:

```text
context = long_attention(hidden_seq, previous_segment_records)
state = context
for block in decision_blocks:
    state = decision_block(hidden_seq, state)
logits = state_head(state)
```

This prevents a raw `[h_t, context]` concat head, but the first version injected
the RNN signal through a gated projection. It was slower and worse:

```text
200k attention-decision: 10.012 -> 9.138 -> 8.431
200k simple late:         9.245 -> 8.299 -> 7.845
```

The cleaner LMA-token formulation treats the current RNN state as token `S0` and
previous segment records as memory tokens:

```text
tokens = [S0=h_t, S1, S2, ...]
tokens = transformer_encoder(tokens)
logits = head(tokens[0])
```

This gives LMA final say without a raw RNN logit branch. A batched implementation
was added as `--mixing lma-token`. It works but is still slower and worse than
the simple late head:

```text
10k LMA-token: 23.20 -> 16.69 -> 15.59
50k LMA-token: 12.49 -> 10.38 -> 9.99
50k simple late: 12.46 -> 10.44 -> 9.46
```

The architecture is cleaner, but current LMA-token blocks are not yet learning a
better composition than `head([h_t, context])`. They also add substantial memory
and compute overhead.

## Matched GRU Harness Check And Gated Cross-Attention

A concern was that the segment harness GRU floor might be much worse than the
standalone sequence GRU. A controlled comparison was run with:

```text
train slice:      200k chars
eval slice:       first 20k validation chars
embedding/hidden: 64/64
lr:               0.001
updates:          ~3,860
tokens/update:    ~512
```

The random-window sequence trainer used `batch_size=32`, `block_size=16`,
`steps=3860`. It reached:

```text
matched sequence GRU: 7.83 PPL
```

The segment harness used `--mixing none --carry-hidden`, with 24,706 trie
segments per epoch, mean segment length 8.10, and 10 epochs:

```text
10.587 -> 9.107 -> 8.577 -> 8.349 -> 8.192
 8.072 -> 7.966 -> 7.879 -> 7.808 -> 7.757
```

This largely exonerates the harness GRU floor under matched settings. The older
gap to `~5.x` was mostly an apples-to-oranges comparison: more/full data,
different LR/update exposure, and random-window training. Streaming segment
training is much slower in wall time, but it does not appear fundamentally
broken.

A Flamingo-style gated cross-attention block was added as
`--mixing gated-xattn`:

```text
a = MHA(LN(h), LN(mem), LN(mem))
h = h + tanh(g_attn) * a
h = h + tanh(g_mlp) * MLP(LN(h))
logits = shared_gru_head(h)
```

Both gates initialize at zero, so the initial function is exactly the carried
GRU path. On the same 200k/d64 setup with one gated block and written memory
records:

```text
gated-xattn: 10.390 -> 9.599 -> 9.575
no-memory:   10.587 -> 9.107 -> 8.577
```

The zero-gated cold start works and epoch 1 is slightly better, but the block
falls behind the no-memory GRU by epoch 2. The likely remaining issue is not
the cross-attention block shape; it is that the memory records are still not
trained as useful retrieval records for next-character prediction.

A sharper terminal-record auxiliary was added:

```text
memory_record = write_memory(segment_terminal_state)
record_logits = record_head(memory_record)
record_target = token_after_segment
loss += alpha * CE(record_logits, record_target)
```

This supervises exactly the object appended to memory, unlike the older
per-token `record_aux_weight`.

On the same 200k/d64 gated-xattn setup:

```text
no terminal aux: 10.390 -> 9.599 -> 9.575
alpha=0.10:     10.379 -> 9.575 -> 9.458
alpha=0.25:     10.453 -> 9.641 -> 9.483
no-memory GRU:  10.587 -> 9.107 -> 8.577
```

The low-weight terminal objective helps slightly and improves the epoch-3 slope,
but stronger weighting hurts. This suggests single-next-char supervision is
only weakly aligned with the role of a long-memory record. A better record
objective may need to predict a short future sketch, a local distribution, or a
contrastive retrieval target rather than only the immediate next character.

## Full-Corpus Segment-Memory Check

The 200k slice may have been too small for long-range route memory. Running the
same d64 segment harness on the full training split produced:

```text
train_chars:       1,003,854
train_segments:      108,041
mean_segment_len:        9.29
eval_chars:           20,000
```

One full no-memory GRU epoch:

```text
mixing=none, carry_hidden=true:
val_ppl = 6.988
runtime = 329.6s
peak_rss = 2768 MB
```

One full gated cross-attention epoch with written memory records, terminal
record auxiliary `alpha=0.1`, and `max_memory=16`:

```text
mixing=gated-xattn, terminal_record_aux_weight=0.1, max_memory=16:
val_ppl = 6.503
mean_gate = 0.148
runtime = 735.3s
peak_rss = 2826 MB
```

This is the first clean result where the long-memory branch beats the matched
local GRU floor. The cost is high: about 2.2x slower than no-memory even with
`max_memory=16`. The result supports the hypothesis that the 50k/200k slices
were underpowered for route-memory learning; full-corpus route scale matters.

A longer paired run answered whether the gain was only an epoch-1 artifact:

```text
full no-memory GRU:
6.988 -> 6.303 -> 5.942

full gated-xattn + terminal aux 0.1, max_memory=16:
6.503 -> 5.867
```

The gated model's epoch 2 beats the no-memory model's epoch 3. This does not
prove a lower final asymptote yet, but it does show the long-memory branch is
not merely an early-training perturbation. Its slope remains competitive after
the local GRU has had another full corpus pass.

A checkpointed 5-epoch paired run extended the curve:

```text
full no-memory GRU:
6.988 -> 6.303 -> 5.942 -> 5.744 -> 5.627

full gated-xattn + terminal aux 0.1, max_memory=16:
6.503 -> 5.867 -> 5.546 -> 5.344 -> 5.199
```

The gated model remains ahead at every matched epoch and has not bottomed out by
epoch 5. Its per-epoch improvement is shrinking (`-0.636`, `-0.321`, `-0.202`,
`-0.145`), but the no-memory curve is also flattening and remains about `0.43`
PPL worse at epoch 5. This strengthens the case that route-memory attention is
adding useful modeling capacity, not just changing early optimization.

## Segment-Memory Checkpointing And Profile

`run_segment_memory_model.py` now saves an epoch-boundary checkpoint by default
next to the CSV output. Resume is epoch-based:

```text
--resume-checkpoint runs/<id>_experiment.pt --epochs <total_target_epochs>
```

The checkpoint stores model state, optimizer state, RNG state, completed epoch,
step, run args, and vocab chars. This is enough because training memory and
carried hidden reset at each epoch boundary.

Lightweight profiling was added with:

```text
--profile --progress-every-steps N
```

On a 200k/d64 one-epoch profile:

```text
gated-xattn + terminal aux 0.1, max_memory=16:
train_sec = 155.17
tokens/sec = 1,289
forward = 58.27s / 37.6%
backward = 95.65s / 61.6%
optimizer+clip+detach < 1%
eval = 5.56s
checkpoint = 0.01s

no-memory GRU:
train_sec = 61.95
tokens/sec = 3,228
forward = 20.80s / 33.6%
backward = 40.55s / 65.5%
optimizer+clip+detach ~= 1%
eval = 1.60s
checkpoint = 0.01s
```

Checkpoint overhead is negligible. The cost is in autograd over the recurrent
and attention computation, especially backward. Gated cross-attention is about
2.5x slower than the no-memory segment GRU on the same 200k slice.

The most plausible performance improvement is to reduce the number of tiny
segment-level PyTorch graphs. For `mixing=none`, the local RNN could be run once
over the whole contiguous update chunk, then sliced at segment boundaries,
because the carried state is just the raw terminal RNN state.

For current `gated-xattn`, this is not exactly semantics-preserving: the gated
attention output becomes the carried state for the next segment. That means the
next segment's RNN state depends on the previous segment's memory read, so
blindly chunking the local RNN would remove feedback that may be part of the
observed gain. A safe speedup for gated memory likely needs either:

```text
1. a deliberate "raw carry" variant, where LMA affects logits but not next
   segment recurrence; or
2. a deeper vectorization of the recurrent/memory scan itself.
```

## Count Backoff Gate Reproduction

The old count-gate tool was ported into this repo as
`scripts/agpt_count_gate.py` and reproduced against the carved split from the
older AGPT project:

```text
train:   /home/trans/Projects/agpt/data/.splits/4fa9aec1db6b3aea/train_corpus.txt
heldout: /home/trans/Projects/agpt/data/.splits/4fa9aec1db6b3aea/heldout_corpus.txt
depth:   8
features: entropy_delta,suffix_stats
epochs:  100
```

Heldout fixed-skip results:

```text
unigram:                27.245
witten_bell:             5.433
learned_gate:            3.860
target_backoff_oracle:   2.952
```

The implementation detail that matters: this count gate is a recursive convex
mixture in probability space:

```text
q_d = w_d * p_mle_d + (1 - w_d) * q_backoff
```

That is different from the newer `run_backoff_gate_prior.py` prototype, which
used a product-of-experts/logit-sum form:

```text
logits = log p_root + sum_d w_d * log p_d
```

The product form is much sharper and likely explains why the newer prior
experiments were unstable. The count-only reproduction shows that prefix counts
plus suffix statistics already contain enough information to reach the desired
`~4` PPL range. The next integration target is therefore:

```text
final_logits = log p_count_gate(context) + neural_residual_logits
```

with the count gate frozen initially.

The frozen count prior was integrated into `run_segment_memory_model.py` as
`--count-prior frozen`. Prediction heads are zero-initialized in this mode, so
the initial model is exactly the count prior and the neural model learns a
residual correction:

```text
final_logits_t = log p_count_gate(x_{<=t}) + residual_logits_t
```

On the full current 90/10 segment split, with depth 8,
`entropy_delta,suffix_stats`, and a d64 carried GRU residual:

```text
count prior fit/precompute: 188.8s first run, 224.2s on resume
count prior fit-tail PPL:   4.648

full count-prior + GRU residual:
4.995 -> 4.876 -> 4.781 -> 4.686 -> 4.594
```

This beats the no-prior gated memory result at epoch 5:

```text
gated-xattn without count prior: 5.199
count prior + GRU residual:      4.594
```

The count-prior residual is still improving at epoch 5 and trains at roughly
the no-memory GRU speed once the prior tensor is precomputed. Peak RSS increased
to about `5.6 GB`, mostly from storing train/eval prior log-probability rows.
The immediate engineering improvement was to cache the fitted prior/precomputed
log-prob tensors across resumes and sweeps. `--count-prior-cache` now stores the
metadata, fit history, and CPU log-prob tensors. A cold full-depth-8 prior build
took `193.6s`; the matching cached resume loaded in `0.10s`.

The frozen count prior was then combined with the gated cross-attention residual:

```text
mixing: gated-xattn
memory_record: written
terminal_record_aux_weight: 0.1
max_memory: 16
hidden/embedding: 64/64
count prior: frozen depth 8, entropy_delta,suffix_stats
cache: runs/count_prior_full_d8_entropy_suffix.pt
```

Heldout PPL on the current 90/10 split:

```text
count prior + gated-xattn residual:
4.869 -> 4.675 -> 4.538 -> 4.407 -> 4.286
```

This is the best current neural segment result. It beats the count-prior +
carried-GRU residual by `0.308` PPL at epoch 5, and it beats the no-prior
gated-xattn result by `0.913` PPL at epoch 5. The gated branch appears to be
earning influence rather than acting as noise: `mean_gate` rose from `0.0422` at
epoch 1 to `0.0870` at epoch 5, while heldout PPL improved every epoch.

Profile for the cached 5-epoch continuation:

```text
per epoch train time: ~709-716s
throughput:           ~1400 tokens/s
forward/backward:     ~37% / ~62%
eval:                 ~4.3s
checkpoint:           ~0.01s
peak RSS after cache: ~3.2 GB
```

The main remaining cost is recurrent gated-attention autograd, not count-prior
setup, checkpointing, or evaluation.

The same checkpoint was continued to 10 total epochs with the cached prior:

```text
count prior + gated-xattn residual, epochs 1-10:
4.869 -> 4.675 -> 4.538 -> 4.407 -> 4.286
      -> 4.205 -> 4.153 -> 4.121 -> 4.116 -> 4.136
```

The best heldout point in this run is epoch 9 at `4.116` PPL. Epoch 10 regressed
slightly to `4.136`, so the curve has flattened rather than continuing to fall
without bound. That is a useful leakage check: the model crossed the KN-parity
region, then behaved like a normal saturating run.

Epoch-10 profile:

```text
train time:     724.1s
throughput:     1386 tokens/s
forward/back:   37.2% / 62.1%
eval:           4.09s
peak RSS:       3.18 GB
```

The next step should be validation rather than new architecture: audit split
isolation, evaluate the same checkpoint on the older carved heldout protocol,
and run count-prior-only under this exact segment harness for a direct floor.

## Initial Protocol Audits

Best-checkpoint saving was added to `run_segment_memory_model.py`:

```text
--checkpoint-output       rolling latest epoch checkpoint
--best-checkpoint-output  checkpoint updated only when heldout PPL improves
```

Checkpoint payloads now store `best_val_ppl`, `best_epoch`, and `best_step`.
This only protects future runs; the 10-epoch run above overwrote the epoch-9
state with the epoch-10 state before best-checkpoint saving existed.

Count-prior-only audit under the exact current segment harness:

```text
run: runs/audit_countprior_only_countprior_only_current_split.csv
mode: --epochs 0 --eval-initial --count-prior frozen
current 90/10 heldout PPL: 5.325
```

This is an important decomposition. The count prior alone does not explain the
`4.116` result; the trained residual/memory model is doing substantial work.

Cache metadata sanity check:

```text
current cache: runs/count_prior_full_d8_entropy_suffix.pt
train_log_probs: (1003853, 65)
eval_log_probs:  (19999, 65)
train/eval digests: distinct
```

An attempted audit against the older carved heldout produced `2.577` PPL, but it
is invalid for the current checkpoint. Direct overlap check showed that 9 of the
10 old heldout chunks lie inside the current 90% training slice:

```text
old chunks 0-8: current train split
old chunk 9:   current validation tail
```

So the older carved heldout can only be used as a fair protocol if the model and
count prior are trained on that carved train split from the start, not when
evaluating a checkpoint trained on the current contiguous 90/10 split.

The harness now supports this fair carved protocol with explicit source files:

```text
--train-input <carved train_corpus.txt>
--eval-input  <carved heldout_corpus.txt>
--vocab-input data/input.txt
```

Fair carved-split count-prior-only audit:

```text
train: /home/trans/Projects/agpt/data/.splits/4fa9aec1db6b3aea/train_corpus.txt
eval:  /home/trans/Projects/agpt/data/.splits/4fa9aec1db6b3aea/heldout_corpus.txt
cache: runs/count_prior_carved_d8_entropy_suffix.pt

count prior fit-tail PPL: 4.790
heldout PPL:              3.859
```

This reproduces the old count-gate baseline (`~3.86`) under the segment harness,
so the carved-split path is valid.

One epoch of the current gated-xattn residual on top of that carved count prior:

```text
epoch 0 initial prior: 3.859
epoch 1 residual:      3.958
```

So this residual setup improves the harder current tail-10% split, but on the
old carved split it initially hurts a count prior that is already very strong.
That points to a tuning/objective issue rather than a failure of the count prior.

`--eval-initial` now participates in best-checkpoint tracking. This matters for
prior-residual runs where the best state may be the initial zero-residual prior.

Residual trust knobs were added:

```text
--prior-residual-scale  scales residual logits before adding them to log p_prior
--prior-residual-l2     penalizes squared residual logits during training
```

The first carved split trust run used:

```text
prior_residual_scale = 0.25
prior_residual_l2    = 0.01
```

and improved over the prior:

```text
epoch 0 initial prior: 3.8594965
epoch 1 regularized:   3.8335766
```

This is the first carved-split result where the neural residual helps rather
than damaging the already-strong count prior. The interpretation is consistent
with the current hypothesis: the residual path needs enough gradient to learn,
but must be charged for overriding a strong empirical prior.
