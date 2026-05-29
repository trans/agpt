# Suffix-Weighted Curvature Proxy for AGPT

## The premise: the optimizer is the weak link

The §5 identity `J_p · (Σ g_s) = Σ (J_p · g_s)` and the §6.5 variance-reduction
argument together imply that AGPT's aggregated gradient is *qualitatively
different* from an SGD gradient — it's a count-weighted exact aggregate, not a
stochastic sample. Adam and its relatives were designed to manage gradient
noise from minibatch sampling. In AGPT, that noise is already much smaller by
construction, but the optimizer still behaves as if it needs to compensate for
it via EMA-based second-moment estimation.

The §11.3 catastrophic overfit at 50 super-epochs is consistent with this: the
optimizer takes confident-looking steps in directions whose apparent confidence
is actually masking suffix disagreement inside the aggregate. The aggregate is
moderate not because suffixes agree on a moderate direction, but because
strongly disagreeing suffixes happened to partially cancel.

## What's available for free

At each prefix node `p`, during the backward sweep, you transiently have the
per-suffix gradient contributions `{g_s : s ∈ subtree(p)}` in scope — that's
exactly the moment they're being summed into `G_suffix(p) = Σ_s g_s`. The
sum alone is what the optimizer currently sees. The *distribution* of those
contributions across the subtree is thrown away by the summation, but it
carries the structural information about whether the aggregate is well-supported.

## The curvature proxy

For each parameter, compute the count-weighted second moment of suffix
contributions alongside the existing first moment:

```
G_suffix(p) = Σ_s n_s · g_s              # what AGPT already computes
F_p         = Σ_s n_s · (g_s ⊙ g_s)      # new: diagonal empirical Fisher
```

where `n_s` is the count weight of suffix `s` and `⊙` is elementwise square.
`F_p` is the diagonal empirical Fisher *at this node*, estimated from this
node's own suffix ensemble — no EMA, no cross-step staleness, no minibatch
noise.

This is qualitatively different from Adam's `v_t`. Adam estimates curvature by
averaging `g_t ⊙ g_t` across optimization *steps*, which conflates true
loss-surface curvature with sampling noise and becomes increasingly stale as
parameters drift. `F_p` estimates curvature from the current step's own suffix
ensemble — it reflects the loss surface at the current parameters, weighted by
the empirical count distribution that already governs the loss.

## The preconditioned update

```
θ ← θ − η · G_suffix(p) / (sqrt(F_p) + ε)
```

Interpretation per-parameter:

- **Suffixes agree** (`F_p` ≈ `G_suffix² / N_s`): preconditioner is roughly
  uniform, the step proceeds at natural magnitude. This is a well-supported
  aggregate.

- **Suffixes disagree** (`F_p` ≫ `G_suffix² / N_s`): contributions pull
  strongly in different directions but sum to something moderate. The
  preconditioner damps the step, recognizing the aggregate is structurally
  fragile.

The second case is the brittleness fix. The optimizer stops treating
high-variance aggregates as if they were high-confidence aggregates.

## Where the estimate should actually live

The `g_s` discussion above is at the hidden-state-gradient level
(`∂L/∂h_p` contributions). Parameter gradients come from combining these
with activations: `∂L/∂W = h^T · ∂L/∂h` for the FFN, and a messier
attention-specific composition for QKV/output projections.

So the cleanest version computes suffix-decomposed *parameter* gradient
contributions per layer. For the FFN this is essentially free — the
suffix decomposition of `∂L/∂h` carries through the outer product with
activations directly. For attention it's more involved, and the K-side
gradients are where the per-path structure most naturally collapses
anyway (since K at an interior node is genuinely shared infrastructure
for every descendant). The V-side and FFN gradients are where the
proxy will be most informative.

## Sketch: modified aggregation at a branching node

The existing backward pass already visits each node `p` and sums child
contributions. The modification is to maintain a parallel accumulator
for the squared contributions:

````python
def backward_aggregate_at_node(p):
    """
    Called during backward sweep at branching node p.
    Children's gradient contributions are in scope here, before they
    get folded into G_suffix(p) for propagation up through J_p.
    """
    # Existing aggregation (first moment)
    G_suffix_p = zeros_like(param_grad_shape)

    # NEW: parallel aggregation (uncentered second moment, count-weighted)
    F_p = zeros_like(param_grad_shape)

    for child in p.children:
        # n_s is the count weight on this branch (mass at child)
        n_s = child.edge_mass

        # g_s is the gradient contribution from this child's subtree,
        # already aggregated by recursive descent. At the FFN/V level this
        # is the per-parameter contribution; at the hidden-state level it's
        # ∂L/∂h_p from this branch.
        g_s = child.aggregated_grad

        # First moment — what AGPT currently does
        G_suffix_p += n_s * g_s

        # Second moment — the new piece, per-parameter elementwise square
        F_p += n_s * (g_s * g_s)

    # Propagate first moment up through J_p as usual.
    # F_p is attached to the node for use at the optimizer step, or
    # accumulated into a per-parameter buffer if you'd rather apply
    # preconditioning once per super-epoch.
    return G_suffix_p, F_p


def optimizer_step(params, G_aggregate, F_aggregate, lr, eps=1e-8):
    """
    Applied once per subtree (matching AGPT's existing update cadence).
    G_aggregate and F_aggregate have been summed up from leaf to root
    via the modified backward pass.
    """
    for p, G, F in zip(params, G_aggregate, F_aggregate):
        # Diagonal preconditioned step.
        # sqrt(F) plays the role of Adam's sqrt(v), but F is THIS step's
        # within-subtree curvature, not an EMA across steps.
        p.data -= lr * G / (torch.sqrt(F) + eps)
````

## Cost

One extra elementwise multiply and one extra accumulator per parameter at each
aggregation point. Same memory layout as the existing first-moment
aggregation, doubled bandwidth at the aggregation step, no new asymptotic cost.

The harder engineering question is the one from earlier: whether the per-child
`g_s` exists as a separate tensor at the moment of aggregation, or whether the
varlen attention backward kernel sums them implicitly. If the former, the above
is genuinely a few-line change at the aggregation point. If the latter, you'd
either need a custom kernel or to expose the unfused intermediates — same
engineering hinge that gates everything we discussed about exposing
per-path structure.

## Two implementation variants worth considering

**Per-subtree preconditioning (tighter, matches current update cadence).**
`F_p` is computed within one subtree's backward pass and used immediately for
that subtree's optimizer step. Cleanest semantically — the curvature estimate
and the gradient it preconditions come from the same ensemble. Matches AGPT's
"one update per subtree" structure exactly.

**Cross-subtree accumulation (more like Adam, but informed by structure).**
`F_p` contributions accumulate across subtrees within a super-epoch, giving a
running estimate. Closer to Adam's behavior but with the noise source
fundamentally different (between-subtree variation, not minibatch sampling).
Could be combined with the bigram-partition signal — `F` accumulated across
bigram partitions is a global-scale agreement signal over the prefix space.

The first variant is the natural first experiment. It's the smallest possible
change that tests whether within-subtree suffix variance carries useful signal
for the optimizer.

## What this would falsify or confirm

If the suffix-disagreement hypothesis is right, the preconditioned step should:

1. Be more robust to over-training — the 50-super-epoch catastrophic overfit
   should soften, because the optimizer is damping the very directions where
   apparent gradient signal is actually suffix conflict.

2. Reduce sensitivity to the super-epoch count generally — §11.3's narrow
   sweet spot at 3 super-epochs should widen.

3. Possibly enable larger base learning rates, since per-parameter step sizes
   are now self-regulated by the within-step curvature estimate rather than
   set globally.

If none of these show up, the within-subtree gradient variance is probably
low-rank or path-uniform, and the mean already captures essentially all the
signal. That itself would be a clean negative result — and the cheapest thing
to check first is the variance distribution that was already on the todo list.


## Addendum: Interpreting Suffix Agreement and Structural Coherence

One subtle but important clarification is needed regarding the notion of
"suffix agreement."

At first glance, suffixes *should* disagree — they are different continuations
by definition. A branching node in the prefix tree exists precisely because
multiple suffixes diverge from the same prefix. If "agreement" merely meant
"the suffixes predict the same next token," then the metric would be trivial
and useless.

But the optimizer does not operate directly on token identities. It operates
on shared hidden representations and shared parameters.

The relevant question is therefore not:

    "Do the suffixes differ?"

but rather:

    "Do the suffixes induce compatible parameter updates?"

This is a much deeper structural property.

Different suffixes may still reinforce highly compatible latent features.
For example:

    "the quick brown fox"
    "the quick brown dog"
    "the quick brown bear"

These suffixes disagree at the token level, but their gradients may still
push the model toward similar internal structure:

- noun-like continuation
- animal semantics
- natural-language continuation
- narrative syntax
- sentence continuation priors

The gradients are not merely trying to memorize token identity. They are
trying to shape a shared representation space capable of supporting all
descendant continuations simultaneously.

This distinction is especially important for AGPT because the entire premise
of aggregation assumes that shared prefixes should admit partially shared
representation learning. If descendant suffixes are structurally compatible,
aggregation compresses learning efficiently. If they are structurally
incompatible, aggregation may become lossy or unstable.

This reframes the proposed curvature proxy:

    F_p = Σ_s n_s · (g_s ⊙ g_s)

not primarily as a Hessian approximation, but as a measure of subtree
gradient coherence.

The optimizer is effectively asking:

    "How trustworthy is this aggregate update?"

A moderate aggregate gradient can arise from two very different situations:

### Case 1: Coherent aggregate

All descendant gradients point in roughly compatible directions.

The aggregate is moderate because many suffixes independently reinforce a
shared representation update.

This is high-confidence structural evidence.

### Case 2: Cancellation aggregate

Descendant gradients pull strongly in different directions but happen to
sum to something moderate.

The aggregate appears stable numerically, but the subtree is internally
fractured.

This is low-confidence structural evidence.

The proposed second-moment accumulation distinguishes between these two
cases, which ordinary aggregation discards.

This suggests that the most informative quantity may not be the raw second
moment itself, but the ratio between coherent signal and disagreement energy.

One possible coherence metric is:

    C_p = ||G_p||² / (Σ_s ||g_s||² + ε)

Interpretation:

- C_p ≈ 1:
    Strong suffix agreement.
    Descendant updates are highly aligned.

- C_p ≈ 0:
    Strong suffix disagreement.
    Descendant updates largely cancel.

This resembles a signal-to-noise ratio over subtree structure.

Importantly, this is not measuring whether suffixes are semantically
identical. It measures whether they admit a compatible shared representation
under the current parameterization.

This may ultimately reveal something fundamental about AGPT itself:

- coherent regions compress efficiently under aggregation;
- incoherent regions resist compression and may require specialization.

The coherence structure of the tree may therefore define the true limits of
aggregation efficiency.

In that sense, the optimizer signal is no longer merely about curvature or
step scaling. It becomes a direct measurement of where shared representation
learning succeeds or fails within the corpus topology.

