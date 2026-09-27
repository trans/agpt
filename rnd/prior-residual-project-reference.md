# Prior-Residual Learning Project Reference

Status: extracted project brief from `rnd/rnn-agpt`.

Date: 2026-06-14.

## Thesis

This should become a separate project from AGPT.

The core problem is not tree training. It is how to combine an explicit prior
distribution with a neural residual model:

```text
logits_t = log P_prior(x_t | c_t) + g_theta(c_t, h_t, prior_stats_t) * R_theta(c_t, M_t, prior_stats_t)
```

The prior may come from AGPT tree statistics, Kneser-Ney, a count/backoff model,
a grammar, a retrieval index, a symbolic rule system, or domain-specific
knowledge. The neural model should learn the part the prior cannot cheaply
provide, and should learn when to stay silent.

## Why This Split Happened

The RNN-AGPT thread started with:

```text
h_child = tanh(W_h h_parent + W_x emb[x] + b)
```

inside an AGPT tree. That was useful for proving that AGPT can host non-attention
`f_theta` choices, but the best results quickly came from a learned count/backoff
prior, not from the recurrence itself.

Once we added:

```text
logits = log P_prior + alpha * residual
```

the AGPT tree stopped being the main object. The real question became prior
quality, residual calibration, and representation design.

## Current Best Numbers

All numbers are character-level Tiny Shakespeare, natural-log perplexity.

| system | eval protocol | PPL | note |
|--------|---------------|----:|------|
| learned count/backoff prior | direct full heldout fixed skip-depth | ~3.7919 | strongest pure prior |
| live count prior in history script | full legal history positions | 3.8032 | clean residual baseline |
| live count prior + current fixed-lag residual | full legal history positions | 3.8017 | tiny positive residual |
| live count prior + top-50% surprise training | full legal history positions | 3.8189 | overcorrects |
| live count prior + top-25% surprise training | full legal history positions | 3.9679 | strongly overcorrects |

The cleanest residual result is:

```text
prior:     3.803192
residual:  3.801722
gain:      ~0.0015 PPL
```

That gain is real but too small to justify the current architecture as-is.

## Learned Prior

The current prior is a learned count/backoff gate:

```text
q(root) = p_root
q(c) = w(c) * p_mle(. | c) + (1 - w(c)) * q(suffix(c))
w(c) = sigmoid(theta . phi(c))
```

Useful features included:

```text
reliability
kl_gain
entropy_norm
depth_norm
mass_norm
branch_norm
entropy_delta
suffix_branch_norm
suffix_reliability
suffix_entropy_norm
suffix_kl_gain
suffix_entropy_delta
bias
```

Key findings:

- The prior is surprisingly strong by itself.
- Suffix-side statistics matter.
- Many weak features combine better than a tiny compact set.
- Repeating `suffix_mass_norm` is conceptually wrong because it duplicates
  `mass_norm`; it was removed from the public suffix feature set.
- The learned gate is not neural history. It uses local context and suffix
  backoff only.

## Sidecar Lesson

AGTS sidecars were originally keyed by AGPT position-data substrings. That is
valid for AGPT node targets, but it is not a complete catalog for arbitrary
rolling heldout contexts.

Audit result on identical history-stream positions:

| path | positions | PPL |
|------|----------:|----:|
| live count gate on full depth-16 context | 54,736 | 3.803192 |
| live count gate on sidecar-matched suffix | 54,736 | 3.837546 |
| AGTS sidecar logits | 54,736 | 3.837559 |

Conclusion:

- The sidecar export is effectively exact for the suffix it stores.
- The loss comes from lookup coverage.
- For prior-residual work, use a live prior provider or a sidecar keyed by the
  prior model's context table, not AGPT position-data substrings.

## Residual Findings

The current residual uses:

```text
h_t = GRU state
M_t = [h_{t-1}, h_{t-16}, h_{t-32}, ..., h_{t-1024}]
a_t = attention(h_t, M_t)
logits = log P_prior + alpha_t * W_out(a_t)
```

Findings:

- With a strong prior, the learned alpha gate suppresses the residual.
- Prior-feature alpha helps calibration but still leaves only tiny gains.
- Prior dropout wakes attention but trains the wrong problem and hurts eval.
- Hard surprise filtering wakes attention more directly, but overcorrects.
- The residual can learn if the prior is muted, but it is much weaker than the
  learned count prior.

The likely architectural issue is representation. The residual attends directly
over RNN hidden states, so it inherits the RNN's local recurrence coordinate
system. It does not get a clean learned retrieval space or a rich post-attention
correction space.

## Surprise

Surprise is:

```text
surprise_t = -log P_prior(x_t | c_t)
```

Hard filtering results:

| training filter | residual PPL | avg alpha | attention entropy | result |
|-----------------|-------------:|----------:|------------------:|--------|
| none | 3.8017 | 0.0309 | 3.8666 | tiny gain |
| top 50% surprise | 3.8189 | 0.0847 | 3.0085 | overcorrects |
| top 25% surprise | 3.9679 | 0.1579 | 2.1724 | badly overcorrects |

Interpretation:

- Surprise is a real signal.
- Hard masks are too abrupt.
- Better candidates are soft surprise weighting, stronger residual pricing, or
  gating objectives that let the model spend capacity on hard rows without
  damaging easy rows.

## Recommended New Architecture

Give the residual its own representation space:

```text
h_t = local RNN/sequence state

q_t = W_q [h_t, prior_stats_t]
k_i = W_k [h_i, prior_stats_i]
v_i = W_v [h_i, prior_stats_i]

a_t = Attention(q_t, K, V)
z_t = MLP([h_t, a_t, prior_stats_t])
alpha_t = sigmoid(Gate([h_t, a_t, prior_stats_t]))

logits_t = log P_prior(. | c_t) + alpha_t * ResidualHead(z_t)
```

Important differences from the current prototype:

- RNN state is input to memory, not the memory space itself.
- Q/K/V projections should be explicit and allowed to learn retrieval geometry.
- The residual head should be an MLP, not just a thin linear readout from the
  attended state.
- Prior statistics should be first-class inputs.
- The model should report calibration diagnostics, not just PPL.

## Prior Provider Interface

A useful project abstraction:

```text
class PriorProvider:
  def log_probs(contexts) -> [B, V]
  def features(contexts) -> [B, F]
  def hit_depth(contexts) -> [B]
  def name() -> str
```

Candidate providers:

- learned count/backoff gate
- Kneser-Ney or modified Kneser-Ney
- AGPT tree target prior
- retrieval prior
- grammar/rule prior
- uniform/root prior for ablations

## Losses To Try

Baseline:

```text
loss = CE(log_prior + alpha * residual, target)
     + lambda * ||alpha * residual||^2
```

Soft surprise weighting:

```text
w_t = stopgrad(normalize(surprise_t))
loss = mean(w_t * CE(logits_t, target_t))
```

Residual-only delta loss:

```text
gain_t = CE(log_prior, target_t) - CE(log_prior + delta, target_t)
```

Trust-region:

```text
loss = CE(logits, target) + beta * KL(P_prior || P_model)
```

Calibration targets:

```text
alpha should rise when residual improves heldout likelihood,
not merely when train prior surprise is high.
```

## Evaluation Rules

Always report:

- prior-only PPL on the exact same positions
- residual PPL on the exact same positions
- gain/loss by prior-surprise bucket
- gain/loss by prior entropy bucket
- alpha by bucket
- residual norm by bucket
- attention entropy and top lags
- exact/backoff/miss counts for prior lookup

Do not compare sampled eval against full eval without labeling it. Do not
compare AGTS sidecar priors against live count priors unless the catalog
coverage is audited.

## AGPT Boundary

This project should not be called AGPT unless the AGPT tree training objective
is central.

In the new framing:

- AGPT tree data can provide a prior.
- AGPT node statistics can provide features.
- AGPT can be one prior source among many.
- The central project is prior-residual learning and calibration.

That separation should keep AGPT v1 focused on tree-based training and let this
new project explore a broader class of prior-guided neural systems.

## Handoff Plan

Treat the code in this repo as a prototype/reference, not as the new project's
final shape.

Carry over:

- the learned count/backoff prior equations and feature definitions
- the live-prior evaluation path
- the sidecar audit lesson
- the residual diagnostics: prior PPL, residual PPL, alpha, residual norm,
  attention entropy, and surprise/entropy buckets
- the exact Tiny Shakespeare split and full-eval discipline

Rebuild cleanly:

- `PriorProvider` as a first-class interface
- residual model code with explicit Q/K/V projection space
- training/eval loops around ordered corpus streaming, not AGPT position-data
  lookup
- sidecar export keyed by the prior provider's own context table, if sidecars
  are needed at all

Do not carry over as design assumptions:

- AGTS position-data sidecars for rolling residual eval
- hard surprise masking as the main objective
- direct attention over raw recurrent hidden states as the final residual
  memory design
- sampled eval numbers without a matching full heldout eval

First clean milestones:

1. Reproduce the live learned count/backoff prior at about `3.80` PPL on the
   full legal heldout history positions.
2. Add a no-history residual head and confirm it cannot materially beat the
   prior without careful pricing.
3. Add projected history attention over ordered stream states and require it to
   beat the prior-only baseline on the exact same positions.
4. Add bucketed diagnostics before sweeping architecture size.
5. Only then test softer surprise weighting or other hard-row emphasis.

## Useful Artifacts In This Repo

Reference files:

```text
rnd/rnn-agpt/README.md
src/tools/agpt_count_gate.py
src/tools/agpt_history_residual.py
src/tools/agpt_sidecar_audit.py
```

Important result files:

```text
rnd/rnn-agpt/count_prior_d16_simple_standardized_sidecar.json
rnd/rnn-agpt/sidecar_audit_simple_prior_history.json
rnd/rnn-agpt/history_stream_k64_d64_steps1000_l2_5_alpha_priorfeatures_live_prior_evalall.json
rnd/rnn-agpt/history_stream_k64_d64_steps1000_l2_5_alpha_priorfeatures_live_prior_surprise50_evalall.json
rnd/rnn-agpt/history_stream_k64_d64_steps1000_l2_5_alpha_priorfeatures_live_prior_surprise25_evalall.json
```

Temporary sidecar artifact:

```text
/tmp/agpt_rnn_prior_d16_simple_standardized.agts
```

The sidecar artifact is not the preferred path for rolling residual eval unless
the catalog coverage issue is fixed.
