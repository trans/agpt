---
title: Model reconciliation
kind: experiment
status: concluded
outcome: negative
question: >-
  Do child-to-parent messages in the trie (Fisher- or precision-weighted merges of child evidence,
  pseudo-counts, logit models or adapters) improve on purely local per-node fitting?
answer: >-
  No. Child messages never beat the zero-child ablation. An unguarded upward Fisher merge
  through an additive transition model exploded (train PPL 64.85 to 645.06); with line search
  it was stable but weak (legacy held-out PPL 19.95 at epoch 9). Progressive logit-model reconciliation
  was best with child weight 0 (held-out 21.37), distribution-only reconciliation gained only
  with small child weights (27.44 to 26.75), and random shared-basis adapters reached 26.90
  with no child benefit.
opened: 2026-06-10
updated: 2026-06-10
code: main
eval: legacy
family: recurrent
headline:
- {label: 'Transition model with line-searched upward Fisher messages, epoch 9', metric: 'legacy
    held-out PPL (Tiny Shakespeare, track-local)', value: 19.95}
- {label: 'Progressive logit-model reconciliation, child weight 0.0 (best)', metric: 'legacy
    held-out PPL (depth 8, stride 1)', value: 21.37}
- {label: 'Unguarded transition-model Fisher merge, epoch-1 state update', metric: legacy
    train-trie PPL (from 64.85), value: 645.06}
tags: [curvature, gradient, trie-structure]
related: [state-fisher-geometry, trie-fisher-gru, gradient-population]
---

# Model reconciliation

Tests of the original Ultra idea that each trie node makes a local model proposal and parents
reconcile the child models leaf-to-root, weighting them by Fisher or precision. Four minimal
prototypes on Tiny Shakespeare (legacy train and held-out PPL; depth 8, stride 1 where stated):

- Transition model `e_child = e_parent + r[token]`, local state gradient and diagonal Fisher summed upward: unguarded, train PPL exploded 64.85 -> 645.06 in epoch 1; with a backtracking line search it was stable, held-out 19.95 at epoch 9 (train 19.67), weak but with train and held-out close.
- Distribution-only reconciliation (pseudo-counts passed upward): a full child merge hurt (held-out 35.88 -> 40.38); small child weights helped slightly, best 27.44 -> 26.75.
- Progressive logit-model reconciliation: best with child weight 0.0 (held-out 21.37, unigram baseline 28.43); every nonzero child weight was worse.
- Shared-basis head adapters (LoRA-like, fixed random basis): rank 8 reached 26.90 with no child benefit, and epoch persistence was flat.

Conclusion: child-to-parent messages never beat the zero-child ablation. Local model objects
are not automatically composable. The design notes conclude that messages must live in a
shared coordinate system in information form (precision plus information vector, pulled
through a transition Jacobian) with a trust rule, and that pseudo-count merging is smoothing,
not model merging.

Code and records (under `research/ultra/`):
- `scripts/run_reconciled_transition_model.py`, `run_distribution_reconcile.py`, `run_progressive_logit_reconcile.py`, `run_logit_model_reconcile.py` (no recorded results), `run_shared_basis_adapter_reconcile.py`
- `agpt_ultra/reconcile.py` (Fisher-weighted merge `theta_p = (sum F_i)^-1 sum F_i theta_i`)
- `notebook/archive/state_fisher_results.md`: "Reconciliation Through A Transition Model" through "Shared-Basis Head Adapter Reconciliation"
- `notebook/archive/reconciliation_design_notes.md`
