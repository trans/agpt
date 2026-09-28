---
title: Linear and GRU recurrence
kind: experiment
status: concluded
outcome: mixed
question: >-
  Can cheaper recurrent f_theta (linear, linear+RMSNorm, GRU and GRU variants) replace attention
  inside AGPT's trie training, and how close do they come?
answer: >-
  Partly. A GRU trains through the trie to held-out PPL 5.40 (d=64, depth 8, pd=1, 500 ep,
  seq=8 sliding-window eval), against 9.13 for linear and 8.15 for linear+RMSNorm. Adding
  position (RoPE 5.58; SinPos 6.61 at 100 ep) hurt, and GRU+Wrap reached 5.49 at 100 ep. No
  variant beat a vanilla GRU LM trained without the trie (4.53) or attention AGPT (~4.3).
opened: 2026-06-06
updated: 2026-09-28
code: {branch: worktree-linear-recurrence, tag: exp/linear-recurrence-final}
eval: legacy
family: recurrent
headline:
- {label: 'GRU, d=64 depth 8 pd=1, 500 ep', metric: 'legacy held-out PPL (agpt_recur_perplexity,
    seq=8 sliding window)', value: 5.4}
- {label: 'linear, d=64 depth 8 pd=1, 500 ep', metric: 'legacy held-out PPL (agpt_recur_perplexity,
    seq=8 sliding window)', value: 9.13}
- {label: 'GRU+Wrap (k=2, detached), d=64 depth 8 pd=1, 100 ep', metric: 'legacy held-out
    PPL (agpt_recur_perplexity, seq=8 sliding window)', value: 5.49}
tags: [recurrence, position, context-length]
related: [tanh-recurrence, rnn-agpt]
---

# Linear and GRU recurrence

This strand tested AGPT as a framework with a cheap f_theta: a recurrent cell replaced
attention over the same prefix trie, mass weighting and Adam update scheme. Trainers built:
linear (`h = W_h h_parent + W_x emb + b`), linear+RMSNorm, GRU, GRU+RoPE, GRU+SinPos, and
GRU+Wrap (a second cycle from a detached end-of-chain state), plus a vanilla GRU LM without
the trie as the control. On the clean Shakespeare carve at d=64, depth 8, pd=1, with the
seq=8 sliding-window evaluator (not canonical byte PPL), the GRU reached held-out PPL 5.40
at 500 ep, against 9.13 for linear and 8.15 for linear+RMSNorm. Position inputs hurt (RoPE
5.58, SinPos 6.61 at 100 ep), and GRU+Wrap reached 5.49 at 100 ep. The trie works with any
f_theta, but no variant beat the vanilla GRU LM (4.53) or attention AGPT (~4.3). See
../tanh-recurrence for the parallel tanh trainer and ../rnn-agpt for how the line closed.

Code: branch `worktree-linear-recurrence`, tag `exp/linear-recurrence-final` (adds the overlay trainers, the vanilla attention trainer and the 2026-06-11 closure memo `notes/agpt-project-closure.md`, committed 2026-09-28; the older tag `exp/linear-recurrence` predates them). Key files, all in
`src/tools/`: `agpt_train_recur_linear.cr`, `agpt_train_recur_linear_rms.cr`,
`agpt_train_recur_gru.cr`, `agpt_train_recur_gru_{rope,sinpos,wrap}.cr`,
`agpt_train_gru_lm.cr`, and the evaluator `agpt_recur_perplexity.cr`.
