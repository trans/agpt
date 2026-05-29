# Experiment: Prefix-Grouped SGD vs AGPT Smart Batching

## Purpose

This experiment tests whether AGPT’s advantage comes primarily from **prefix-aware smart batching** rather than merely from reducing redundant computation.

The central question:

> Can standard SGD-like training recover some of AGPT’s benefit by grouping training examples that share prefixes, even without full AGPT-style trie/backward aggregation?

AGPT batches over prefix-related paths and performs fewer optimizer updates. Ordinarily, fewer optimizer updates can weaken training. The hypothesis is that AGPT avoids this failure mode because its batches are not arbitrary. They are structured around shared prefix states and therefore approximate empirical conditional distributions.

This experiment compares ordinary SGD, prefix-ordered SGD, prefix-grouped SGD, entropy-gated prefix grouping, and full AGPT.

---

## Core Hypothesis

Ordinary minibatching averages gradients from mostly unrelated contexts:

```text
"the cat sat ..."       → target A
"government debt ..."   → target B
"function foo(...)"     → target C
"once upon ..."         → target D
```

This reduces gradient variance but may also reduce useful stochasticity and update cadence.

AGPT-style batching instead groups examples by a shared prefix:

```text
AB → C
AB → D
AB → E
AB → F
```

This is qualitatively different. The batch estimates the empirical continuation distribution for a specific prefix state:

```text
P_data(. | AB)
```

Instead of sampling one continuation at a time:

```text
step 1: AB → C
step 2: AB → D
step 3: AB → C
step 4: AB → E
```

the model sees the local distribution directly:

```text
one update:
AB → {C: 17, D: 9, E: 3, F: 1}
```

This is not merely “larger batching.” It is structured conditional-distribution training.

---

## Mathematical Framing

Let `p` be a shared prefix and let `s` range over its continuation targets.

For a prefix group:

```text
p → {s_1, s_2, ..., s_k}
```

the grouped loss is:

```text
L_p = Σ_s count(p, s) · CE(model(p), s)
```

or normalized:

```text
L_p = Σ_s P_data(s | p) · CE(model(p), s)
```

The gradient flowing into the shared prefix representation is:

```text
∂L_p / ∂h_p = Σ_s count(p, s) · ∂L_{p,s} / ∂h_p
```

For the prefix-side parameters:

```text
∂L_p / ∂θ_prefix = J_pᵀ · Σ_s g_s
```

where:

```text
J_p = Jacobian of the prefix computation
g_s = continuation-side gradient signal for continuation s
```

The AGPT insight is that:

```text
J_pᵀ · (Σ_s g_s)
```

can be computed more efficiently than repeatedly computing:

```text
J_pᵀ · g_1
J_pᵀ · g_2
J_pᵀ · g_3
...
```

under nearly identical prefix computation.

The lighter experiment does not need to implement full Jacobian reuse. It only needs to test whether the **grouped continuation distribution** is itself helpful.

---

## Experimental Ladder

Run the following training variants.

### Variant 1: Random Window SGD Baseline

Standard next-token training.

```text
sample random windows
forward
loss
backward
optimizer step
```

Properties:

```text
many optimizer updates
high stochasticity
no prefix locality
no explicit reuse of shared prefixes
```

This is the baseline.

---

### Variant 2: Prefix-Sorted SGD

Extract normal training windows, but sort them lexicographically by token sequence or by prefix.

Training remains per-window.

```text
ABC → update
ABD → update
ABE → update
XYZ → update
XYQ → update
```

Properties:

```text
same number of optimizer updates as baseline
same loss objective
different presentation order
better locality
no grouped loss
no prefix cache required
```

Purpose:

> Tests whether ordering alone helps.

If this improves training, then curriculum/locality matters even before true aggregation.

---

### Variant 3: Prefix-Grouped SGD

Group windows by a shared prefix of length `k`.

Example:

```text
ABC
ABD
ABE
```

becomes:

```text
AB → {C, D, E}
```

Training step:

```text
forward shared prefix AB
compute logits for next token
compute count-weighted continuation loss
backward once
optimizer step once
```

Properties:

```text
fewer optimizer updates
direct empirical continuation distribution
prefix-local batch
SGD-like optimizer behavior
no full AGPT trie backward aggregation required
```

Purpose:

> Tests whether smart batching over prefix-related paths can compensate for fewer optimizer updates.

---

### Variant 4: Prefix-Grouped SGD with Group Controls

Same as Variant 3, but with safeguards against bad aggregation.

A prefix group may be rejected, split, or downweighted based on:

```text
group mass
continuation entropy
branch count
maximum continuation probability
gradient norm
gradient conflict
depth
```

The goal is to avoid “mushy” batches where the prefix is too broad and the continuation distribution is too diffuse.

Example bad group:

```text
A → many unrelated continuations
```

Example better group:

```text
ABCD → coherent continuation set
```

Purpose:

> Tests whether disciplined smart batching beats naive prefix grouping.

---

### Variant 5: Full AGPT

The current or best available AGPT implementation.

Properties:

```text
prefix trie structure
shared prefix computation
aggregated continuation gradients
possible radix compression
possible depth curriculum
fewer optimizer updates
deeper prefix/Jacobian reuse
```

Purpose:

> Tests whether full AGPT provides additional benefit beyond prefix-grouped SGD.

---

## Core Comparison

The important comparison is not merely:

```text
Which model gets lowest loss?
```

The important comparison is:

```text
How much useful learning happens per optimizer update?
How much useful learning happens per token-equivalent?
How much useful learning happens per wall-clock second?
Does generalization improve or degrade?
Does generated text degrade into repetition/run-on artifacts?
```

---

## Implementation Plan

### Step 1: Extract Training Windows

Given a tokenized corpus:

```text
tokens = [t_0, t_1, ..., t_N]
```

For context length `D`, extract windows:

```text
prefix = tokens[i : i + D]
target = tokens[i + D]
```

Each training example is:

```text
(prefix, target)
```

For shorter prefix experiments, define grouping prefix length `k <= D`.

```text
group_key = prefix[0:k]
continuation_context = prefix[k:D]
target = target
```

For the simplest version, use `k = D`, meaning exact full-context grouping:

```text
full_prefix → next-token target distribution
```

However, full-prefix grouping may have low duplicate mass for large `D`, especially on small corpora.

So also test smaller `k`:

```text
k ∈ {2, 4, 8, 16, 32}
```

depending on corpus size and tokenization level.

---

## Variant Details

## Variant 1: Random Window SGD

Pseudocode:

```python
for epoch in range(num_epochs):
    shuffle(windows)

    for batch in make_batches(windows, batch_size):
        loss = model.loss(batch.prefixes, batch.targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

Record:

```text
train loss
held-out loss
optimizer steps
tokens processed
wall-clock time
generated samples
```

---

## Variant 2: Prefix-Sorted SGD

Pseudocode:

```python
for epoch in range(num_epochs):
    sorted_windows = sort_by_prefix(windows)

    for batch in make_batches(sorted_windows, batch_size):
        loss = model.loss(batch.prefixes, batch.targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

Important:

- Keep batch size equal to Variant 1 where possible.
- The only intended change is order/locality.
- Do not aggregate continuation counts yet.

Record the same metrics.

---

## Variant 3: Prefix-Grouped SGD

Build groups:

```python
groups = {}

for prefix, target in windows:
    key = prefix[:k]
    groups[key][target] += 1
```

For each group:

```text
prefix key p
target histogram H_p
```

Example:

```text
AB → {
    C: 17,
    D: 9,
    E: 3,
    F: 1
}
```

Loss:

```python
logits = model.forward(prefix)
log_probs = log_softmax(logits)

loss = 0
for target, count in target_histogram.items():
    loss += count * negative_log_likelihood(log_probs[target])

loss = loss / total_count
```

Then:

```python
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

Important choices:

### Count-weighted vs normalized loss

Test both if easy.

Count-weighted total loss:

```text
L = Σ count(target) · CE(target)
```

Normalized distribution loss:

```text
L = Σ P(target | prefix) · CE(target)
```

The normalized version prevents huge groups from dominating. The count-weighted version better reflects corpus mass.

A compromise is:

```text
weight = sqrt(total_group_count)
L_group = weight · normalized_loss
```

or:

```text
weight = log(1 + total_group_count)
```

This may reduce domination by extremely frequent shallow prefixes.

---

## Variant 4: Prefix-Grouped SGD with Controls

For each prefix group, compute:

```python
mass = sum(target_counts)
branch_count = len(target_counts)
probs = [count / mass for count in target_counts]
entropy = -sum(p * log(p) for p in probs)
max_prob = max(probs)
```

Optional normalized entropy:

```python
entropy_norm = entropy / log(branch_count)
```

Useful group indicators:

```text
high mass
moderate branch count
moderate entropy
not too uniform
not too deterministic
```

Bad group types:

### Too diffuse

```text
high entropy
large branch count
low max probability
```

This may produce a mushy gradient.

### Too deterministic

```text
one target dominates almost completely
```

This may still be useful, but it may not teach alternatives.

### Too shallow

```text
short prefix
huge mass
high entropy
weak semantic coherence
```

May need splitting by deeper prefix.

### Too tiny

```text
mass = 1
```

No batching benefit.

---

## Suggested Group Policy

Start simple.

For each candidate group:

```python
if mass < min_mass:
    defer to ordinary SGD or merge into fallback batch

elif entropy_norm > max_entropy_norm and k < max_k:
    split group using deeper prefix

elif branch_count > max_branch_count and k < max_k:
    split group using deeper prefix

else:
    train as prefix group
```

Initial suggested parameters:

```text
min_mass = 2
max_entropy_norm = 0.85
max_branch_count = 64
k values = 2, 4, 8, 16, 32
```

These are placeholders. Tune empirically.

---

## Variant 5: Full AGPT

Compare against the current AGPT path.

Record the same metrics, plus any AGPT-specific metrics:

```text
number of trie nodes
number of radix nodes
radix mass contribution
average subtree mass
average subtree depth
updates per epoch
forward calls per epoch
backward calls per epoch
cache hit/reuse rate if available
```

---

## Metrics to Track

For every variant, log:

```text
train loss
held-out loss
perplexity
optimizer steps
tokens processed
examples processed
wall-clock time
loss improvement per optimizer step
loss improvement per token-equivalent
loss improvement per wall-clock second
gradient norm
update norm
generalization gap
```

Also periodically generate samples and inspect:

```text
repetition
run-on words
collapsed vocabulary
overfitting artifacts
local memorization
syntax coherence
```

---

## Derived Metrics

### Loss Improvement Per Step

```text
Δloss_per_step = (loss_before - loss_after) / optimizer_steps
```

This helps test whether grouped updates are higher information-density.

---

### Loss Improvement Per Token

```text
Δloss_per_token = (loss_before - loss_after) / token_equivalents_processed
```

This helps compare against ordinary SGD.

---

### Loss Improvement Per Wall Clock

```text
Δloss_per_second = (loss_before - loss_after) / elapsed_seconds
```

This is the practical speed metric.

---

### Update Cadence

```text
updates_per_epoch
updates_per_second
tokens_per_update
```

AGPT and prefix-grouped SGD may process many tokens per update. This metric tells us whether optimizer-step starvation is occurring.

---

### Batch Information Density

For prefix-grouped variants:

```text
group_mass
branch_count
entropy
entropy_norm
max_target_probability
loss_before_update
loss_after_update
```

Possible useful score:

```text
information_density = group_mass * (1 - entropy_norm)
```

or:

```text
information_density = group_mass / (1 + entropy)
```

These are rough diagnostics, not necessarily training weights.

---

## Expected Outcomes

### Outcome A: Prefix-Sorted SGD Improves

If Variant 2 beats Variant 1:

```text
ordering/locality alone helps
```

This suggests curriculum matters independently of aggregation.

---

### Outcome B: Prefix-Grouped SGD Beats Random SGD at Similar or Fewer Steps

If Variant 3 beats Variant 1 despite fewer optimizer steps:

```text
smart batching hypothesis is supported
```

This suggests grouped continuation distributions are more useful than random window batches.

---

### Outcome C: Entropy-Gated Prefix Grouping Beats Naive Prefix Grouping

If Variant 4 beats Variant 3:

```text
batch quality matters
```

This suggests AGPT needs disciplined group selection, not maximal aggregation.

---

### Outcome D: Full AGPT Beats Prefix-Grouped SGD

If Variant 5 beats Variant 3/4:

```text
AGPT's deeper trie/Jacobian reuse matters beyond smart batching
```

This supports the full AGPT architecture.

---

### Outcome E: Large Prefix Groups Underperform

If grouped variants underperform:

```text
lost update cadence / reduced stochasticity dominates
```

Possible remedies:

```text
smaller groups
higher learning rate
different optimizer
entropy gating
group splitting
more frequent updates
hybrid SGD polish phase
```

---

## Key Risk: Optimizer-Step Starvation

A giant grouped update may contain rich information but still train poorly because the optimizer receives too few corrective steps.

Example:

```text
one root-level tree update per epoch
```

This may be statistically rich but dynamically weak.

The likely target is not maximum aggregation.

The likely target is:

```text
maximum useful aggregation before optimizer-step starvation begins
```

So we should sweep group sizes and prefix depths.

---

## Suggested Sweeps

### Prefix length sweep

```text
k = 2
k = 4
k = 8
k = 16
k = 32
k = D
```

For character-level corpora, smaller `k` may still have large useful groups.

For token-level corpora, exact full-prefix duplication may be rare unless `D` is small.

---

### Group mass cap

Test maximum group mass:

```text
max_group_mass = 8
max_group_mass = 16
max_group_mass = 32
max_group_mass = 64
max_group_mass = 128
unlimited
```

If a group exceeds the cap, split it into subgroups or sample from it.

---

### Entropy gate

Test:

```text
max_entropy_norm = 0.65
max_entropy_norm = 0.75
max_entropy_norm = 0.85
max_entropy_norm = 0.95
no entropy gate
```

---

### Learning rate sweep

Grouped updates may need a different learning rate.

Test around the current best known values:

```text
lr = 3e-4
lr = 1e-3
lr = 3e-3
lr = 1e-2
lr = 3e-2
```

Especially test whether prefix-grouped SGD benefits from the higher LR that helped AGPT.

---

### Optimizer sweep

Test:

```text
AdamW
Adam
SGD + momentum
RMSProp
Adafactor if available
```

Hypothesis:

> Adam may not be ideal for AGPT-like grouped updates because the gradient structure is less like noisy SGD and more like structured conditional-distribution fitting.

---

## Minimal First Implementation

Implement the simplest useful version first:

```text
Variant 1: Random SGD baseline
Variant 2: Prefix-sorted SGD
Variant 3: Prefix-grouped SGD with fixed k
```

Do not implement entropy gating immediately unless the first grouped version is clearly unstable or disappointing.

Recommended first `k` values:

```text
k = 4
k = 8
k = 16
```

depending on corpus/tokenization.

---

## Minimal Pseudocode

```python
def build_windows(tokens, D):
    windows = []

    for i in range(0, len(tokens) - D):
        prefix = tuple(tokens[i:i+D])
        target = tokens[i+D]
        windows.append((prefix, target))

    return windows
```

---

```python
def build_prefix_groups(windows, k):
    groups = {}

    for prefix, target in windows:
        key = prefix[:k]

        if key not in groups:
            groups[key] = {}

        groups[key][target] = groups[key].get(target, 0) + 1

    return groups
```

---

```python
def train_prefix_group(model, optimizer, prefix, target_counts):
    logits = model.forward_prefix(prefix)
    log_probs = log_softmax(logits)

    total = sum(target_counts.values())

    loss = 0.0

    for target, count in target_counts.items():
        p = count / total
        loss += p * (-log_probs[target])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
```

---

## Important Implementation Note

For the simplest implementation, `model.forward_prefix(prefix)` can simply run the model normally on the prefix and return logits for the next token.

This does not yet require an actual reusable prefix cache.

The experiment is first testing the **training signal**, not the compute optimization.

Once the signal is validated, add prefix cache reuse.

---

## Prefix Cache Version

After the basic grouped version works, optimize:

```text
compute hidden state for shared prefix once
reuse for all continuations
compute grouped loss
backward once
```

But beware:

> Cached states become stale after optimizer updates.

Therefore, cached prefix states are only safe inside a no-update interval.

Safe pattern:

```text
compute prefix state
compute all continuation losses for this prefix group
backward
optimizer step
discard cache
```

Unsafe pattern:

```text
compute prefix state
optimizer step
reuse old prefix state later
```

unless doing an intentionally stale-cache experiment.

---

## Interpretation Guide

The experiment should answer:

### Does locality alone help?

Compare:

```text
Variant 1 vs Variant 2
```

### Does prefix-local aggregation help?

Compare:

```text
Variant 1 vs Variant 3
Variant 2 vs Variant 3
```

### Does disciplined grouping help?

Compare:

```text
Variant 3 vs Variant 4
```

### Does full AGPT provide extra benefit beyond smart batching?

Compare:

```text
Variant 4 vs Variant 5
```

---

## Working Theory

AGPT should not be described as merely using larger batches.

A better description:

> AGPT replaces stochastic sampling of continuation events with direct conditional-distribution training over trie-local empirical measures.

Ordinary batching:

```text
averages unrelated samples
```

AGPT-style smart batching:

```text
aggregates continuation pressure on a shared prefix state
```

The practical question is whether that richer update compensates for fewer optimizer steps.

The likely answer is:

```text
yes, up to a point
```

Beyond that point, aggregation becomes too broad, update cadence becomes too low, and training weakens.

Therefore, the goal is:

```text
maximum useful aggregation
not maximum aggregation
```

---

## Success Criteria

The experiment is promising if prefix-grouped SGD or entropy-gated prefix-grouped SGD achieves one or more of:

```text
lower held-out loss than random SGD at equal wall-clock
lower held-out loss than random SGD at equal token-equivalent budget
better loss improvement per optimizer step
similar loss with fewer optimizer steps
reduced run-on/repetition artifacts compared with naive AGPT
better stability at higher learning rates
```

The experiment is especially important if:

```text
prefix-grouped SGD gets close to AGPT
```

because that would imply that a simpler training bridge exists between ordinary GPT training and full AGPT.

If full AGPT still clearly wins, that supports the deeper prefix/Jacobian reuse thesis.

---

## Final Summary

This experiment separates three possible sources of AGPT’s advantage:

```text
1. Training order / curriculum locality
2. Prefix-local smart batching
3. Full trie/Jacobian aggregation
```

The experimental ladder is:

```text
Random SGD
    ↓
Prefix-sorted SGD
    ↓
Prefix-grouped SGD
    ↓
Entropy-gated prefix-grouped SGD
    ↓
Full AGPT
```

The key hypothesis:

> Prefix-related examples can be batched without weakening training because the batch represents a coherent empirical continuation distribution, not an arbitrary collection of unrelated samples.

The key risk:

> Too much aggregation reduces optimizer update cadence and creates mushy high-entropy gradients.

The engineering goal:

> Find the aggregation regime where each optimizer step has maximal useful information density without starving the optimizer of corrective updates.
