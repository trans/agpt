# Reconciliation Design Notes

## Current Read

The experiments so far separate two facts that were easy to conflate:

1. Local Fisher geometry is real and useful.
2. Local model objects are not automatically composable.

The prefix trie gives exact local next-token statistics. For a node state `h_v`
and output head `W`, the state-space Fisher

```text
F_v = W^T Cov(q_v) W
```

is a correct local curvature object. Direct state updates and head updates can
fit trie rows very efficiently.

The problem is global composition. A node-local optimum can be very good for
that node and still be a poor prior for unseen contexts, sibling contexts, or
parents. Exact local optimization is not the same thing as generalization.

## What Worked

- **Head Fisher / merged head evidence** is stable when evidence is collected
  before mutation. It is a sound local/global boundary.
- **Direct state Fisher** rapidly improves the materialized trie objective.
- **Pure book state model** verifies the one-head derivation: persistent node
  states plus `W` are easy to optimize.
- **Ordinary learned sequence models** still provide the best shared feature
  space. Attention and GRU baselines beat the handcrafted reconciliation toys.

## What Failed Or Underperformed

- **Free node states** overfit. Train PPL reaches the KN-like range, while
  held-out PPL worsens after a few epochs.
- **Raw subtree/head updates** are order-sensitive unless evidence is merged or
  guarded.
- **Direct body Fisher through GRU parameters** is fragile; local quadratic
  prediction can be wrong after a finite nonlinear parameter update.
- **Pseudo-count/distribution reconciliation** is not model reconciliation.
  It is smoothing, not model-copy merging.
- **Free logit reconciliation** improved held-out, but the zero-child ablation
  was best. Child messages did not help.
- **Random shared-basis adapters** are too weak. Without a meaningful feature
  space, LoRA-like adapter reconciliation has little to merge.

## Condition For A Reconciliable Model

A branch-local model object must live in a shared coordinate system and must
carry uncertainty/precision. Raw weights are usually not enough.

Good message objects:

```text
gradient + Fisher/Hessian
precision + information vector
low-rank adapter delta + precision
state delta in a shared feature space + precision
Bayesian natural parameters
```

Weak message objects:

```text
raw trained neural weights
raw child logits from semantically different contexts
hidden states without a transition model
probability rows without trust/backoff structure
```

The parent should not simply average child models. It needs to combine child
messages in information form:

```text
precision_parent = precision_local + sum precision_child
info_parent      = info_local      + sum info_child
model_parent     = inverse(precision_parent) * info_parent
```

This is the Bayesian version of reconciliation.

## Transition Model With Bayesian Message Passing

The missing structure is a transition model:

```text
h_child = T_x(h_parent)
```

where `x` is the edge token.

If child evidence is about `h_child`, parent evidence must be pulled through the
transition:

```text
parent_gradient += transpose(J) * child_gradient
parent_fisher   += transpose(J) * child_fisher * J
```

where `J` is the Jacobian of `T_x` with respect to `h_parent`.

In message-passing terms:

1. Root-to-leaf pass computes states.
2. Each node computes local likelihood evidence from its trie row.
3. Leaf-to-root pass sends information-form messages through transition
   Jacobians.
4. Parent combines child messages with local evidence and a prior.
5. A later root-to-leaf pass can send posterior/prior information back to
   children.

That last bidirectional piece is not implemented yet. Current prototypes only
test fragments of this idea.

## Mass-Weighted SGD Idea

A simpler path may be to keep a normal sequence model and use the trie objective
to correct the sampling problem.

Ordinary SGD estimates frequency by repeatedly sampling common contexts over
time. The trie already knows the mass:

```text
n(prefix, next_token)
```

So instead of relying on repeated sampling, one could sample contexts with an
inverted or flattened mass probability, then weight the loss by the true mass.

Sketch:

```text
sample prefix v with probability s(v)
compute CE loss for q_v
weight by n_v / s(v)
```

This is an unbiased estimator of the AGPT trie objective if `s(v)` covers the
nodes being optimized.

The useful part is control: high-mass easy contexts do not have to dominate the
batch schedule, but their evidence still enters the gradient with the correct
weight.

## The Long-Sequence Mass Problem

A concern: long/deep contexts often have mass 1.

At depth `d`, most exact long contexts may be singleton rows. If we sample only
the deepest context, the mass weight is low and noisy. But the same token
prediction also belongs to shorter suffix/prefix contexts with higher mass.

So a long sequence probably should not be treated as only one mass-1 event. It
should expose a stack of backing contexts:

```text
context length 16
context length 15
...
context length 1
context length 0
```

Each position can contribute evidence at multiple context lengths, or choose a
context length according to a trust/backoff rule.

This starts to look like interpolated n-gram smoothing:

```text
loss(position) =
  sum_depth w(depth, mass, entropy) * CE(q_context_depth, model_prediction)
```

Open question: should the model be trained to match the deepest reliable row,
or a posterior/backoff mixture of rows?

Likely answer: use a posterior target, not raw deepest `q`.

## Near-Term Directions

### 1. Mass-Corrected Sequence Training

Use the real sequence model that already works, but train batches from trie
nodes:

```text
sample nodes with flattened mass distribution
loss = true_mass_weight * CE(row target, model(prefix))
```

Compare against ordinary sequence SGD at matched update count.

### 2. Backoff Posterior Targets

Before applying neural machinery, define the target distribution for a context
as a mass/entropy-weighted suffix mixture:

```text
q_hat(context) = sum_suffix w_s q_s
```

This is closer to KN-style generalization and should reduce singleton overfit.

### 3. Bayesian Transition Messages

Return to reconciliation only after a transition model is explicit:

```text
h_child = T_x(h_parent)
```

Then implement messages in information form, not raw model averaging.

### 4. Adapter Reconciliation Only With Learned Features

Do not continue random-basis adapters. If using LoRA/adapters, attach them to a
real learned GRU/attention feature extractor, otherwise there is little useful
geometry to reconcile.

## Working Hypothesis

The trie is best used as an exact evidence engine, not as an unconstrained
lookup-table model.

The promising form is:

```text
learned shared sequence model
+ trie-derived mass-corrected objectives
+ Fisher/natural updates where curvature is local and reliable
+ Bayesian/backoff targets for low-mass contexts
```

The original progressive reconciliation idea may still be viable, but only when
node messages are information-form objects in a shared transition geometry.
