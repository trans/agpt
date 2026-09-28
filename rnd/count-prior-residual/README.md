---
title: Count-prior residual
kind: experiment
status: concluded
outcome: mixed
question: >-
  Does a neural residual over segment memory (carried GRU or gated cross-attention), trained
  on top of a frozen recursive count-gate prior, improve held-out PPL beyond the prior alone?
answer: >-
  Yes on the contiguous tail split, only slightly on the carved split. On the 90/10 split
  (first 20k held-out chars) the prior alone scores legacy PPL 5.325 in the harness; a GRU
  residual reached 4.594 and a gated cross-attention residual 4.116 (epoch 9). On the fair
  carved split the prior alone scores 3.859: an unregularized residual hurt (3.958), and a
  charged residual (scale 0.10, L2 penalty, context alpha gate, raw memory records) reached
  at best 3.776, with most settings drifting after epoch 1 or 2.
opened: 2026-06-12
updated: 2026-06-13
code: main
eval: legacy
family: hybrid
headline:
- {label: 'Count prior + gated-xattn residual, d64, best epoch 9', metric: 'legacy held-out
    PPL, first 20k chars of the contiguous tail-10% split', value: 4.116}
- {label: 'Count-gate prior alone, same harness and split', metric: 'legacy held-out PPL,
    first 20k chars of the contiguous tail-10% split', value: 5.325}
- label: >-
    Carved split, prior + charged raw-memory residual, epoch 2 (prior alone 3.859)
  metric: legacy held-out PPL, carved split 4fa9aec1db6b3aea
  value: 3.776
tags: [priors, recurrence, attention, evaluation, reproducibility]
related: [count-backoff-gate, gated-xattn-memory, gutenberg-prior-residual, rnn-agpt]
---

# Count-prior residual

Freeze the recursive count-gate prior (depth 8, `entropy_delta,suffix_stats`; see
`count-backoff-gate`) under the segment-memory model and train a zero-initialized residual:
`logits = log p_count_gate(context) + alpha * residual_logits`, optionally with an L2 charge on
the residual and a context-conditioned alpha gate.

Contiguous 90/10 split (legacy held-out PPL, first 20k chars of the tail): the prior alone
scores 5.325 in this harness. A carried-GRU residual reached 4.594 at epoch 5; a gated
cross-attention residual reached 4.116 at epoch 9 (4.136 at epoch 10), against 5.199 for gated
cross-attention without the prior. Fair carved split (`data/.splits/4fa9aec1db6b3aea`, prior
and model trained on the carved train file): the prior alone scores 3.859 and an unregularized
residual hurt (3.958 after one epoch). Charging the residual (scale 0.10, L2 0.01) gave 3.815,
then drifted; a context alpha gate gave 3.811 with a steadier epoch 2; raw memory records
without the terminal auxiliary gave the best point, 3.776 at epoch 2. Scoring the tail-split
checkpoint on the old carved heldout gave an invalid 2.577 (9 of 10 chunks are in its train
slice). Conclusion: the residual improves a strong count prior only when it pays for overriding
it; the tail-split gain shrinks to 3.859 -> 3.776 on the carved split and fades after epoch 1-2.

Code and records (under `research/ultra/`):
- `scripts/run_segment_memory_model.py` (`--count-prior frozen`, `--count-prior-cache`, `--prior-residual-scale`, `--prior-residual-l2`, `--prior-residual-gate`, `--train-input`/`--eval-input`/`--vocab-input`)
- `agpt_ultra/count_gate.py`
- `notebook/archive/state_fisher_results.md`: "Count Backoff Gate Reproduction" (integration part), "Initial Protocol Audits"
- `notebook/archive/segment_memory_prior_residual_snapshot.md`; `segment_memory_math.md` ("Count-Gate Prior", "Carved Split Audit")
- `notebook/journal/2026-06-13.md`, entries 13:50 to 19:18
