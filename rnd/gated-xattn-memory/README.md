---
title: Gated cross-attention memory
kind: experiment
status: concluded
outcome: mixed
question: >-
  In a matched segment harness, does a zero-initialized gated cross-attention block over segment
  memory records improve on a carried GRU, and does the answer depend on data scale?
answer: >-
  Yes at full-corpus scale, no on a 200k slice. The harness GRU floor matched a random-window
  sequence GRU at equal updates (7.757 vs 7.83, 200k chars, d64). On 200k chars the gated
  block trailed the no-memory GRU (epoch 3: 9.575 vs 8.577; 9.458 with a 0.1 terminal-record
  auxiliary). On the full 1,003,854-char training split it led at every epoch, 5.199 vs 5.627
  at epoch 5 (legacy held-out PPL, first 20k chars of the tail-10% split), at about 2.2x the
  epoch time.
opened: 2026-06-12
updated: 2026-06-12
code: main
eval: legacy
family: recurrent
headline:
- label: Full corpus, no-memory carried GRU (d64), epoch 5
  metric: >-
    legacy segment-harness held-out PPL, first 20k chars of the tail-10% split
  value: 5.627
- label: Full corpus, gated-xattn + terminal aux 0.1, max_memory 16, epoch 5
  metric: >-
    legacy segment-harness held-out PPL, first 20k chars of the tail-10% split
  value: 5.199
- label: >-
    Matched harness check, carried segment GRU, 200k chars, epoch 10 (sequence GRU 7.83)
  metric: legacy validation PPL, first 20k validation chars
  value: 7.757
tags: [recurrence, attention, context-length, scaling]
related: [segment-memory, count-prior-residual]
---

# Gated cross-attention memory

Follow-up to `segment-memory`. First a controlled check of the segment harness: at 200k train
chars, d64, LR 1e-3 and about 3,860 updates, a random-window sequence GRU reached 7.83 and the
carried segment-stream GRU 7.757 after 10 epochs (first 20k validation chars), so the harness
GRU floor is not handicapped. Then a Flamingo-style block (`--mixing gated-xattn`),
`h + tanh(g_attn) * MHA(LN(h), LN(mem), LN(mem))` plus a gated MLP, with both gates
zero-initialized so training starts exactly at the carried GRU.

Results (legacy segment-harness held-out PPL, first 20k chars of the tail-10% split): on the
200k slice the block fell behind the no-memory GRU by epoch 2 (epoch 3: 9.575 vs 8.577); a
terminal-record auxiliary at weight 0.1 helped slightly (9.458) and 0.25 did not. On the full
1,003,854-char training split (108,041 segments) it led at every epoch: 6.503 vs 6.988 after
one epoch and 5.199 vs 5.627 after five, still descending. Cost: about 2.2x the no-memory epoch
time on the full corpus and about 2.5x on 200k (backward is about 62% of train time;
checkpointing is negligible).

Conclusion: route-memory attention adds modeling capacity once there are enough routes; the
50k/200k slices were underpowered. This configuration became the residual model on top of the
frozen count prior (`count-prior-residual`).

Code and records (under `research/ultra/`):
- `scripts/run_segment_memory_model.py` (`--mixing gated-xattn`, `--memory-record written`, `--terminal-record-aux-weight`, `--max-memory`, `--profile`, `--checkpoint-output`, `--resume-checkpoint`)
- `notebook/archive/state_fisher_results.md`: "Matched GRU Harness Check And Gated Cross-Attention", "Full-Corpus Segment-Memory Check", "Segment-Memory Checkpointing And Profile"
- `notebook/archive/segment_memory_math.md`: "Controlled Harness Check" through "Full-Corpus Route Scale"
