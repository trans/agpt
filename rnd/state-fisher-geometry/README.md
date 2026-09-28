---
title: State-Fisher geometry
kind: experiment
status: concluded
outcome: mixed
question: >-
  Do node-local natural steps on trie hidden states, using a state-space Fisher built from
  each node's count row, fit the trie objective, and can the corrected states be carried into
  a shared GRU (or a node-state table) that generalizes?
answer: >-
  The geometry works; the projection does not. With no parameter update, 16 state-Fisher iterations
  took the depth-8 aggregate train-trie PPL from 63.52 to 3.994. Carrying this into the GRU
  was weak: corrected-logit distillation reached legacy sequential validation PPL 9.18 (trainable
  head, 5 epochs), state-delta projection 28.72 and hidden-target distillation 13.25. A free
  node-state table reached train PPL 4.94, but its held-out PPL bottomed at 25.14 (epoch 6)
  and rose to 31.50 by epoch 20.
opened: 2026-06-09
updated: 2026-06-10
code: main
eval: legacy
family: recurrent
headline:
- label: >-
    Direct state-Fisher correction, 16 iterations, no weight update (depth 8, d64)
  metric: legacy aggregate train-trie PPL (from 63.52; no held-out)
  value: 3.994
- {label: 'Corrected-logit distillation into the GRU, trainable head, 5 epochs', metric: legacy
    sequential validation PPL (90/10 contiguous split), value: 9.18}
- {label: 'Pure book state model (free node states), best epoch 6', metric: 'legacy held-out
    PPL, longest-suffix lookup into the train trie', value: 25.14}
tags: [curvature, recurrence, targets, trie-structure]
related: [trie-fisher-gru, tree-prior-residual, model-reconciliation]
---

# State-Fisher geometry

Freeze the GRU and its head, treat each trie node's hidden state as a free variable, and take
damped natural steps `delta_h_v = -(F_v + damping I)^-1 g_v` with the state-space Fisher
`F_v = N_v W^T (Diag(r_v) - r_v r_v^T) W` built from the node's count row, plus an exact
per-node line search in logit space. Then try to move the improvement into a shared model.

Results (Tiny Shakespeare, stride-1 circular samples, block 8, d64, damping 10, empirical
curvature; legacy aggregate train-trie PPL, no held-out): with no parameter update the trie
PPL fell from 63.52 to 7.70 (1 iteration), 4.187 (8) and 3.994 (16). Projection into the GRU
was weak (legacy sequential validation PPL, 90/10 contiguous split): corrected-logit
distillation 23.12 with a frozen head (10 epochs) and 9.18 with a trainable head (5 epochs);
state-delta projection 28.72; hidden-target distillation with a merged Fisher head 13.25
after five passes (whole-tree variant 14.04). The pure book state model (a free node-state
table plus one Fisher-updated head, longest-suffix lookup at eval) reached train PPL 4.94 by
epoch 20, while held-out PPL bottomed at 25.14 (epoch 6) and rose to 31.50.

Conclusion: the count-derived local Fisher geometry is real, but one shared nonlinear model
does not absorb the free node-state moves, and free node states overfit.

Code and records (under `research/ultra/`):
- `agpt_ultra/state_fisher.py`, `state_projection.py`, `fisher.py`
- `scripts/run_state_fisher_diagnostic.py`, `run_state_distill.py` (`--distill-target hidden`), `run_state_delta_project.py`, `run_book_state_model.py`, `generate_book_state.py`
- `notebook/archive/state_fisher_results.md`: "Setup" through "Projection Notes", hidden-target rows of "Return To AGPT Training", "Pure Book State Model"
- `docs/trie_fisher_bridge.md`, `docs/trie_fisher_bridge_revised.md` (theory)
