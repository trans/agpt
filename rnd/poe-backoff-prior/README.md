---
title: Product-of-experts backoff prior
kind: experiment
status: concluded
outcome: negative
question: >-
  Does a product-of-experts backoff prior (log p_root plus gated log p_d along drop-oldest
  suffix chains, with a 5-parameter logistic gate) make a usable trie prior?
answer: >-
  No. Multiplying sharp count distributions is unsafe: in the 200k-char prior-only setting
  a uniform product scored legacy validation PPL 224.823 and the conservatively initialized
  gate 10.130 (deepest context alone 13.840). Entropy damping tamed the uniform product but
  worsened the initialized gate, and training the gate overfit (10.130 to 12.361 while train
  PPL fell toward 1). The recursive convex count gate replaced it.
opened: 2026-06-12
updated: 2026-06-12
code: main
eval: legacy
family: count-prior
headline:
- {label: 'Conservatively initialized 5-parameter gate (bias -2), prior only', metric: 'legacy
    validation PPL, 200k-char setting', value: 10.13}
- {label: 'Uniform product of experts, prior only', metric: 'legacy validation PPL, 200k-char
    setting', value: 224.823}
- {label: Gate trained 3 epochs at LR 0.005 (no damping), metric: 'legacy validation PPL,
    200k-char setting', value: 12.361}
tags: [priors, trie-structure]
related: [count-backoff-gate, count-prior-residual, tree-prior-residual]
superseded_by: [count-backoff-gate]
---

# Product-of-experts backoff prior

A Python prototype of a suffix-link backoff prior: exact tuple-key contexts from the training
text, drop-oldest backoff chains (`ABCD -> BCD -> CD -> D -> root`), and a product of experts in
logit space, `logits = log p_root + sum_d gate(features_d) * log p_d`, with fixed count tensors
and only a 5-parameter logistic gate trainable. It tested the math before any radix or suffix
catalog was involved.

Results (Tiny Shakespeare, 200k-char setting, prior only, legacy validation PPL): root 28.812,
deepest context 13.840, uniform product 224.823, conservatively initialized gate (bias -2)
10.130. Entropy damping helped deepest-only (13.014 at a=0.5) and the uniform product (48.991
at a=1.0) but worsened the initialized gate (11.949 at a=0.5). Training the gate overfit even
at LR 0.005: validation went 10.130 -> 10.917 -> 11.394 -> 12.361 while train PPL fell toward 1.

Conclusion: multiplying sharp count distributions is unsafe, and a learned gate memorizes train
statistics. The count-gate reproduction that followed (`count-backoff-gate`) uses a recursive
convex mixture, `q_d = w_d p_mle_d + (1 - w_d) q_backoff`, and the archive says the product form
likely explains this prototype's instability.

Code and records (under `research/ultra/`):
- `scripts/run_backoff_gate_prior.py`
- `notebook/archive/state_fisher_results.md`: "Backoff-Gated Trie Prior", and the contrast drawn in "Count Backoff Gate Reproduction"
