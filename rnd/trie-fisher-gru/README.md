---
title: Trie-Fisher GRU training
kind: experiment
status: concluded
outcome: negative
question: >-
  Can trie-aggregated Fisher natural-gradient updates (an exact merged output-head Fisher
  step, optionally a body Fisher step) train a small GRU character model competitively, over
  the whole trie or over bounded prefix subtrees?
answer: >-
  No, not in these prototypes. On bounded depth-8 subtrees an AdamW GRU body with a merged
  Fisher head reached legacy held-out sequential PPL 11.96 after five passes (15.76 after
  one); body Fisher on all prefixes diverged to 2287.49 after one pass, and per-prefix head
  Fisher on partial evidence drove held-out PPL to infinity. Sequence training of the same
  GRU family reached validation PPL 5.78 (context 8, 50,000 updates), and a small AdamW attention
  LM reached held-out window PPL 5.600 (seq len 16, 5,000 steps).
opened: 2026-06-09
updated: 2026-06-10
code: main
eval: legacy
family: recurrent
headline:
- label: AdamW GRU body + merged Fisher head, depth-8 subtrees, 5 passes
  metric: >-
    legacy held-out sequential PPL (Tiny Shakespeare, script-default 90/10 contiguous split)
  value: 11.96
- {label: 'Body Fisher GRU + Fisher head, all prefixes, 1 pass', metric: legacy held-out sequential
    PPL (same protocol), value: 2287.49}
- {label: 'Reference: sequence-trained GRU, context 8, 50,000 updates', metric: legacy validation
    PPL, value: 5.78}
tags: [optimizer, curvature, recurrence, baseline]
related: [state-fisher-geometry, model-reconciliation, linear-recurrence, gradient-population]
---

# Trie-Fisher GRU training

The first AGPT Ultra prototypes trained a small GRU character model (`Embedding -> GRUCell ->
linear head`) on the trie objective, updating the output head with an exact, matrix-free trie
Fisher natural step, `theta -= (sum_p F_p + lambda I)^-1 sum_p g_p` solved by conjugate
gradient, and the body with ordinary gradient steps (AdamW in the subtree runs) or a
body-Fisher step. Later runs used bounded depth-8 prefix subtrees as the training units.

Results (Tiny Shakespeare, script-default contiguous 90/10 split, legacy held-out sequential
PPL): AdamW body + merged Fisher head reached 15.76 after one pass and 11.96 after five. Body
Fisher on the `a/e/t` prefixes reached 16.16 at about 9 GB RSS; on all prefixes it diverged to
2287.49 after one pass. Per-prefix head Fisher on partial evidence drove held-out PPL to
infinity, so head evidence has to be merged after a full sweep or guarded. For reference,
sequence training of the same GRU family reached validation PPL 6.27 (5,000 updates) and 5.78
(50,000 updates) at context 8, and a 2-layer AdamW attention LM reached held-out window PPL
5.600 (seq len 16, 5,000 steps). An earlier prefix-subtree run is recorded only as "roughly
8.22 to 8.26 PPL at depth 8". Conclusion: the merged head Fisher step is stable, but training
the GRU this way was not competitive with ordinary sequence training.

Code and records (under `research/ultra/`):
- `agpt_ultra/head_only.py`, `hybrid.py`, `body_fisher.py`, `embedding_fisher.py`, `fisher.py`, `reconcile.py`, `baseline.py`
- `scripts/train_head_only.py`, `train_hybrid.py`, `train_baseline.py`, `run_comparison.py`, `run_hybrid_stress.py`, `sweep_block_size.py`, `run_prefix_head_exact.py`, `run_prefix_exact_hybrid.py`, `run_prefix_subtree_train.py`, `run_sequence_trainer.py`, `run_attention_sequence.py`
- `README.md` (Head-Only, Hybrid and Baseline sections), `notebook/archive/current_math.md`
- `notebook/archive/state_fisher_results.md`: "Current Baselines", "Return To AGPT Training", attention baseline at the end of "Shared-Basis Head Adapter Reconciliation"
- `tests/test_prefix_structure.py` (cached-logit equivalence and the shared-Jacobian identity)
