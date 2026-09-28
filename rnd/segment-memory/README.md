---
title: Segment memory
kind: experiment
status: concluded
outcome: mixed
question: >-
  Can trie-derived variable-length segments serve as a route vocabulary for a recurrent character
  model with attention over previous segment states, and how should local recurrence and segment
  memory be combined?
answer: >-
  Partly. Resetting the GRU at every segment boundary was the main handicap: the best reset-GRU
  attention variant reached legacy validation PPL 10.13 on a 50k-char slice, while a carried
  GRU with no attention reached 8.27 (epoch 10). With the recurrence carried, attention over
  previous segment records helped early (50k, epoch 3: late head 9.46 vs 9.87) but did not
  lower the 10-epoch floor (8.86 best with attention). The simple late head, head([h_t, context]),
  was the best composition (200k chars: 7.254 at epoch 7); fusion blocks, attention-owned
  decisions, LMA-token encoders, interface LayerNorm and a tanh core did not beat it.
opened: 2026-06-11
updated: 2026-06-12
code: main
eval: legacy
family: recurrent
headline:
- label: >-
    Best reset-GRU variant (causal current-segment attention), 50k train chars, epoch 5
  metric: legacy segment-harness validation PPL
  value: 10.13
- {label: 'Carried GRU hidden, no attention, 50k train chars, epoch 10', metric: legacy segment-harness
    validation PPL, value: 8.27}
- label: >-
    GRU late head (local GRU + attention over previous segments), 200k train chars, best epoch
    7
  metric: legacy segment-harness validation PPL (20k eval chars)
  value: 7.254
tags: [recurrence, attention, context-length, rope]
related: [gated-xattn-memory, count-prior-residual]
---

# Segment memory

Segment the training text into shortest-unique trie segments (depth cap 16), run a GRU over
each segment, and let attention read previous segments' terminal states. No Fisher, mass
weighting or reconciliation: the question is whether the trie gives a useful route vocabulary
for an ordinary sequence model. Runs used 10k-200k-char Tiny Shakespeare train slices, mostly
on CPU (legacy segment-harness validation PPL; the 200k runs used 20k eval chars).

Results: with the GRU reset at every segment boundary, the best 50k variant (causal
current-segment attention) reached 10.13 and most others stayed between about 10.4 and 11.
Carrying the GRU hidden state across segments beat all of them with no attention (50k, epoch
10: 8.27; 100k: 7.80). With the carried GRU, attention over previous records helped early (50k,
epoch 3: late head 9.46 vs 9.87) but not at the 10-epoch floor (best 8.86 with attention). The
`late` head, `head([h_t, context])`, was the best composition: 200k GRU late bottomed at 7.254
(epoch 7). At 200k, epoch 3 (late: 7.845), a tanh core (8.808), interface LayerNorm (8.003), a
post-attention fusion block (8.097) and an attention-owned decision state (8.431) were worse, as
was an LMA-token encoder at 50k (9.99 vs late 9.46).

Conclusion: early attention gains mostly patched an artificial recurrence break; with a carried
GRU, segment memory added little on these slices (full-corpus follow-up: `gated-xattn-memory`).

Code and records (under `research/ultra/`):
- `scripts/run_segment_memory_model.py` (`--mixing`, `--carry-hidden`, `--rnn-core`, `--attn-interface-norm`, `--rope-positions`)
- `notebook/archive/state_fisher_results.md`: "Segment Memory Prototype", "Transformer-Style Fusion Block"
- `notebook/archive/segment_memory_math.md` (formulation, the `late` rule, open problems)
