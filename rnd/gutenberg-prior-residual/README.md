---
title: Gutenberg prior residual
kind: experiment
status: concluded
outcome: inconclusive
question: >-
  Does the count-prior + segment-memory residual transfer to a 5M-character Gutenberg corpus,
  and what does the residual need in order to add information beyond a strong prior?
answer: >-
  Only with a softened prior, and modestly. The depth-8 count gate alone scores legacy held-out
  PPL 3.57155 on the carved Gutenberg split. The Tiny Shakespeare residual recipe diverged
  and a conservative one was null (3.57491). Scaling the prior by 0.75 gave the residual enough
  gradient to learn selective attention: with a 200k-char train prior it went 5.52120 to 5.27333
  over five epochs, and with the full 4.75M-char prior one epoch reached 3.56268, just below
  the full prior. The work stopped there, compute-bound at about an hour per CPU epoch.
opened: 2026-06-13
updated: 2026-06-14
code: main
eval: legacy
family: hybrid
headline:
- {label: 'Count-gate prior alone (depth 8, entropy_delta + suffix_stats)', metric: 'legacy
    held-out PPL, carved Gutenberg 5M split (250k held-out chars)', value: 3.57155}
- label: >-
    Prior strength 0.75 + gated-xattn residual, full 4.75M-char train, 1 epoch
  metric: legacy held-out PPL, carved Gutenberg 5M split (250k held-out chars)
  value: 3.56268
- label: >-
    Same setting with a 200k-char train prior, 5 epochs (softened prior 5.52120)
  metric: legacy held-out PPL, carved Gutenberg 5M split (250k held-out chars)
  value: 5.27333
tags: [priors, scaling, data, infrastructure]
related: [count-prior-residual, count-backoff-gate]
---

# Gutenberg prior residual

Scale the count-prior residual (`count-prior-residual`) to a carved 5M-character Gutenberg
split: 4,750,000 train and 250,000 held-out chars, 65-character vocabulary. The split
(`data/.splits/4bfd5a43d446644a` in the former Ultra checkout) was not imported. All numbers
are legacy held-out PPL on the 250k held-out chars.

Results: the depth-8 count gate alone scores 3.57155 (Witten-Bell 4.60359, target-backoff oracle
2.94643). The Tiny Shakespeare residual recipe diverged (epoch 1: ~7.44e43); a conservative one
(scale 0.01, L2 0.10, LR 3e-4) was safe but null (3.57491). With a 200k-char train prior (prior
alone 5.61342), prior-state features and written memory with char RoPE gave only tiny gains
(5.60930 after five epochs) with attention near uniform. Softening the prior,
`final_logits = prior_strength * log p_prior + residual`, was the lever: strength 0.5 diverged,
while strength 0.75 (residual scale 0.25, L2 0.05) went 5.52120 -> 5.27333 over five epochs with
selective attention. With the full 4.75M-char prior the same setting went from 3.72280 (softened
prior) to 3.56268 after one epoch, just below the full prior's 3.57155, at about an hour per CPU
epoch; it was not run further. A depth-by-depth NumPy prior compiler over packed count tables
cut full-Gutenberg prior precompute from 573.73 s (Python) to 76.92 s (64.91 s with an mmap
depth cache). Conclusion: the residual adds information beyond a strong count prior only when
the prior is softened enough to leave it gradient; at full scale the gain is small so far.

Code and records (under `research/ultra/`):
- `scripts/run_segment_memory_model.py` (`--count-prior-impl packed`, `--count-prior-precompute numpy`, `--count-prior-storage mmap`, `--prior-strength`, `--prior-residual-state-features`)
- `agpt_ultra/vectorized_count_prior.py`, `agpt_ultra/count_gate.py` (`PackedCountModel`), `tests/test_count_gate_packed.py`
- `notebook/journal/2026-06-13.md`, entries "23:55 - Gutenberg 5M Prior Scaling And Packed Counts" through "2026-06-14 - Full Gutenberg Prior-Strength Transfer"
