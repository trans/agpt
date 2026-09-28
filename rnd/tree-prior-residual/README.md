---
title: Tree-prior residual
kind: experiment
status: concluded
outcome: mixed
question: >-
  Can a trie-derived prior (backoff counts, or a state-Fisher-corrected head) plus a learned
  neural residual improve held-out prediction when the prior is built from training counts
  only?
answer: >-
  Only with an explicit count prior, and only slightly. The Fisher-prior residual reached
  train-trie PPL 2.076 at block 16 but legacy held-out PPL 12.782 (Fisher prior alone 13.243).
  On the first 2k validation samples the backed-off count prior (6.601) beat the Fisher prior
  (9.395), so the state/head projection loses count information. With the log-count prior
  used directly, a conservative depth-8 residual improved held-out PPL from 6.191 to 6.157.
  The line was judged not competitive with ordinary SGD on a small attention model.
opened: 2026-06-10
updated: 2026-06-10
code: main
eval: legacy
family: hybrid
headline:
- {label: 'Fisher-prior residual, block 16 (train-trie diagnostic 2.076)', metric: 'legacy
    held-out PPL, contiguous last-10% split, train-derived priors', value: 12.782}
- {label: Backed-off train count prior alone (depth 16), metric: 'legacy held-out PPL, first
    2k validation samples', value: 6.601}
- {label: 'Direct depth-8 count prior + conservative GRU residual, 5 epochs', metric: 'legacy
    held-out PPL, first 2k validation samples', value: 6.157}
tags: [priors, curvature, recurrence, evaluation]
related: [state-fisher-geometry, count-prior-residual, poe-backoff-prior]
---

# Tree-prior residual

Model the logits as a trie-derived prior plus a learned residual, `z_final = z_prior +
z_residual`. Three priors were tried: a suffix/backoff count prior, a detached
state-Fisher-corrected head prior, and the explicit log-count backoff distribution with forced
suffix drops during training. Held-out evaluation uses train-derived priors only (longest
train-seen suffix), never held-out counts.

Results (Tiny Shakespeare, contiguous last 10% held out; legacy PPL): the backoff prior with a
d128 residual moved held-out 10.123 to 9.995 (depth 8). The Fisher-prior residual looked strong
on the train trie (block 16: 2.166 -> 2.076, at 37-40 min per epoch and about 10 GB RSS) but
scored 12.782 held-out (Fisher prior alone 13.243). On the first 2k validation samples the
backed-off count prior (6.601) beat the Fisher prior (9.395) and Fisher + residual (9.137), and
head-only fidelity training barely moved it (9.366). With the explicit count prior, a
conservative depth-8 residual (scale 0.25, LR 3e-4) improved held-out 6.191 to 6.157.

Conclusion: the state/head projection loses the count evidence, so the Fisher prior does not
generalize; an explicit count prior plus residual behaves as expected but gains little. The
strand was dropped as not competitive with ordinary SGD on a small attention model; count
priors returned later as the recursive count gate (see `count-prior-residual`).

Code and records (under `research/ultra/`):
- `scripts/run_residual_prior.py`, `run_fisher_residual.py`, `eval_fisher_residual.py`, `run_prior_fidelity.py`, `run_direct_prior_residual.py`
- `agpt_ultra/state_fisher.py`, `flat_trie.py`
- `notebook/archive/state_fisher_results.md`: "Residual Tree Prior", "Fisher-Prior Residual", "Held-Out Evaluation", "Direct Tree-Prior Residual"
