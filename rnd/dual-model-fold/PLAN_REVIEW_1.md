# Dual-View Consistency AGPT

## Summary

This experiment implements a **dual-model prefix/suffix AGPT trainer**. It is not intended to test fold-as-loop or effective context extension directly. Instead, it tests whether a forward AGPT model and a backward AGPT model can be trained jointly so that each benefits from the other’s contextual view.

The core question is:

> Can prefix evidence and suffix evidence be coupled during training in a way that reduces disagreement between the two models and improves prediction quality?

The motivation comes from the observed gap between independently trained forward and backward AGPT models. Although the underlying prefix/suffix trie distributions agree structurally, the trained models disagree substantially on held-out positions, with a measured forward/backward divergence of approximately **2.38 nats per position**. The proposed experiment uses a symmetric KL-consistency loss to see whether that gap can be reduced meaningfully. :contentReference[oaicite:0]{index=0}

---

## What This Experiment Is

This is a **dual-view consistency training experiment**.

Two models are trained at the same time:

- **F**, a forward model that sees the prefix context before a target token.
- **B**, a backward model that sees the reversed suffix context after the same target token.

At each corpus position, both models predict the same held-out token:

```text
prefix      -> F -> predicts target
suffix_rev  -> B -> predicts target
```

Each model receives its ordinary cross-entropy loss against the true target. In addition, each model receives a KL-consistency loss that pulls its predicted distribution toward the other model’s predicted distribution.

The intended effect is that each model learns not only from the local target, but also from the other directional view’s belief about that target.

---

## What This Experiment Is Not

This is **not yet a folding/looping experiment**.

It does not test whether prefix paths can loop into suffix paths.
It does not extend the effective sequence length.
It does not implement fold-as-loop, wraparound, or path continuation beyond `seq_len`.

Instead, this experiment answers a prerequisite question:

> Is there useful, trainable mutual information between prefix and suffix views that can be distilled into the models?

If the answer is yes, then later folding/looping work has a stronger foundation. If the answer is no, then the larger fold-as-loop idea may need a different mechanism.

---

## Core Training Objective

For each target position `i`:

```text
prefix      = corpus[i - seq_len .. i - 1]
suffix_rev  = reverse(corpus[i + 1 .. i + seq_len])
target      = corpus[i]
```

The two models produce:

```text
P_F = F(prefix)
P_B = B(suffix_rev)
```

The losses are:

```text
ce_F = CE(target, P_F)
ce_B = CE(target, P_B)

kl_F = KL(stop_grad(P_B) || P_F)
kl_B = KL(stop_grad(P_F) || P_B)
```

The final model losses are:

```text
loss_F = ce_F + β * kl_F
loss_B = ce_B + β * kl_B
```

The `stop_grad` is critical. Each model treats the other model’s prediction as a fixed teacher distribution. This avoids a tangled gradient path where both models simultaneously chase a moving target through the same loss term.

The practical gradient contribution for the KL term is simple:

```text
grad_logits_F += β * (P_F - P_B)
grad_logits_B += β * (P_B - P_F)
```

assuming the opposite distribution is detached.

---

## Unified Prefix/Suffix Context Index

The plan proposes a unified data structure carrying both forward and backward statistics at each substring node.

Conceptually, each node represents a corpus substring `W` and stores:

```text
in_counts   = counts of tokens appearing before W
out_counts  = counts of tokens appearing after W
in_mass     = total backward-side mass
out_mass    = total forward-side mass
```

The important invariant is:

> The incoming and outgoing counts are marginals of the same underlying joint event:
> token-before, substring, token-after.

This is the structural reason the unified index is useful. It physically co-locates the two directional views at the same corpus substring.

One implementation caveat: this structure may be more accurately described as a **bidirectional substring graph** or **dual context index** rather than a pure tree. Forward expansion and backward expansion are different operations:

```text
forward child:   W + token
backward child:  token + W
```

So the storage may behave more like a bidirectional substring DAG than a conventional rooted trie. That is not a problem, but the implementation should preserve the invariant rather than force the object to behave like a normal tree.

Suggested names:

```text
UnifiedContextGraph
DualContextIndex
PrefixSuffixContextIndex
```

---

## Baselines and Controls

The most important baseline is:

```text
β = 0
```

At β=0, the trainer should behave like two independent models trained in the same executable. This must reproduce the previous independent forward/backward behavior. If it does not, then the new trainer has introduced some other coupling or data-ordering difference.

Recommended sweep:

```text
β ∈ {0.0, 0.01, 0.1, 1.0}
```

A better second-stage sweep, if 0.1 looks promising:

```text
β ∈ {0.03, 0.05, 0.1, 0.2}
```

A very useful negative control should also be added:

```text
shuffled suffix pairing
```

In this control:

```text
F sees prefix from position i
B sees suffix from random position j
target remains corpus[i]
```

If aligned suffix coupling helps but shuffled suffix coupling does not, then the improvement is probably due to real prefix/suffix information. If shuffled coupling also helps, then the KL term may mostly be acting as a generic regularizer.

---

## Recommended Diagnostics

In addition to held-out perplexity, the trainer should log:

```text
ce_F
ce_B
kl_B_to_F
kl_F_to_B
β * kl_B_to_F
β * kl_F_to_B
H(P_F)
H(P_B)
KL(P_B || P_F)
KL(P_F || P_B)
```

The entropy diagnostics are important. A reduced KL gap is only meaningful if the models remain predictive. If KL shrinks because both models become bland high-entropy distributions, that is not a real success.

A β warmup is also recommended:

```text
β_eff = β_max * min(1, step / warmup_steps)
```

Early in training, both models are weak teachers. Cross-entropy should dominate first; KL consistency should become stronger only after the models have learned useful distributions.

---

## Evaluation Metrics

The experiment should distinguish two different evaluation modes.

### 1. Causal Evaluation

Use the forward model alone:

```text
P_F(x_i | x_{i-d:i-1})
```

This is comparable to normal causal AGPT/GPT training.

This is the most important metric if the goal is still to improve a causal language model.

### 2. Bidirectional Reconstruction Evaluation

Use both prefix and suffix context:

```text
P(x_i | left context, right context)
```

This is not directly comparable to ordinary causal perplexity, because the model has access to future context. It is still a valid and useful masked-token-style reconstruction metric.

The ensemble should be evaluated several ways:

```text
Arithmetic mixture:
P = 0.5 * P_F + 0.5 * P_B

Logit average:
logits = 0.5 * logits_F + 0.5 * logits_B

Product of experts:
P ∝ P_F * P_B

Weighted product:
P ∝ P_F^α * P_B^(1 - α)
```

The product-of-experts version is closest to Bayesian conjunction, but it may become overconfident. Calibration and entropy should therefore be tracked.

---

## Success Criteria

### Tier 1: Structural Coupling Works

The forward/backward KL gap drops substantially:

```text
F/B KL: 2.38 nats -> < 1.0 nat
```

without significant degradation in held-out cross-entropy or perplexity.

### Tier 2: Causal Model Improves

The forward model alone improves over the β=0 matched baseline:

```text
F-alone PPL at β > 0 < F-alone PPL at β = 0
```

This is the key result if the goal is improving causal AGPT.

### Tier 3: Bidirectional Ensemble Improves

The combined F+B model beats either model alone:

```text
ensemble PPL < F-alone PPL
ensemble PPL < B-alone PPL
```

This would show that the two directional views contain complementary information.

### Tier 4: Prefix/Suffix Alignment Is Real

Aligned suffix coupling beats shuffled suffix coupling:

```text
aligned suffix KL/PPL improvement > shuffled suffix KL/PPL improvement
```

This is important because it distinguishes real bidirectional information transfer from generic KL regularization.

---

## Optional Branch-Gated KL Ablation

A useful follow-up is to apply the KL term only at informative branching regions.

Possible variants:

```text
KL everywhere
KL only when H(out_counts) > ε
KL only when H(in_counts) > ε
KL only when both H(out_counts) and H(in_counts) > ε
```

The reasoning is that unary/radix-cap regions may mostly encode deterministic continuation. Branching regions are where probability distributions matter most.

A reasonable first entropy threshold:

```text
ε = 0.1 nats
```

This ablation would help determine whether prefix/suffix coupling is most useful at decision points rather than along deterministic paths.

---

## Implementation Notes

The dual loss should be implemented using numerically stable log-softmax values.

For:

```text
KL(P_B || P_F)
```

compute:

```text
sum_t P_B[t] * (log_P_B[t] - log_P_F[t])
```

where:

```text
log_P_F = log_softmax(logits_F)
log_P_B = log_softmax(logits_B)

P_F = exp(log_P_F)
P_B = exp(log_P_B)
```

The detached teacher distribution should not receive gradient from that KL term.

Because the KL gradient is simple, the loss kernel can directly add:

```text
grad_logits_F += β * weight * (P_F - P_B)
grad_logits_B += β * weight * (P_B - P_F)
```

The KL term should inherit the same mass-weighting policy as the cross-entropy loss, so frequent corpus events affect both CE and consistency proportionally.

---

## Interpretation of Possible Results

### KL closes and F-alone PPL improves

This is the strongest positive result.

It would suggest that suffix-side evidence acts as useful regularization or information distillation for the causal forward model.

### KL closes but F-alone PPL does not improve

The models can be made to agree, but agreement does not improve causal prediction.

This may mean the prefix and suffix views contain genuinely different information and forcing agreement removes useful asymmetry.

### Ensemble improves but F-alone does not

This shows useful bidirectional reconstruction, but not necessarily causal LM improvement.

Still valuable, but the claim should be framed as bidirectional/masked-token benefit rather than causal AGPT improvement.

### KL does not close

The consistency objective is too weak, too noisy, or structurally mismatched.

Possible next steps would include β tuning, warmup, branch-gated KL, or a more direct fold/loop mechanism.

### Shuffled suffix performs similarly to aligned suffix

The KL term is probably acting as generic regularization.

This would weaken the prefix/suffix interpretation.

### Aligned suffix clearly beats shuffled suffix

This strongly supports the idea that the suffix view contains usable information that the forward model can absorb during training.

---

## Recommended Framing

This branch should be described as:

> **Dual-view consistency AGPT**

or:

> **Prefix/suffix consistency training for AGPT**

It should not be described as full folding or looped context extension.

A good concise description:

> This experiment jointly trains forward and backward AGPT models over a unified prefix/suffix context index. The models predict the same held-out token from opposite contexts and are coupled by a symmetric stop-gradient KL loss. The goal is to determine whether the measured forward/backward disagreement can be reduced in a way that improves causal perplexity, bidirectional reconstruction, or both.

---

## Verdict

This is worth implementing.

It has a clean hypothesis, a measurable baseline, and an interpretable failure mode. It does not yet prove the larger fold-as-loop idea, but it is exactly the kind of prerequisite experiment that can tell whether prefix/suffix coupling contains useful signal.

The most important additions before implementation are:

1. Add β warmup.
2. Log CE, KL, weighted KL, entropy, and held-out PPL separately.
3. Add a shuffled-suffix negative control.
4. Clearly distinguish causal PPL from bidirectional reconstruction PPL.
5. Define ensemble math precisely.
6. Consider branch-gated KL as a follow-up ablation.

If this experiment reduces the 2.38-nat gap and improves F-alone held-out PPL, it is a meaningful AGPT result. If it only improves the F+B ensemble, it is still useful, but the claim should be limited to bidirectional reconstruction. If it fails cleanly, that is also valuable because it tells us the next serious attempt must involve actual path-level folding/looping rather than loss-level consistency.

