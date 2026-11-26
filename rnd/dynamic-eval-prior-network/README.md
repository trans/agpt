---
title: Dynamic evaluation on mobius's prior + stream network
kind: diagnostic
status: active
outcome: n/a
question: >-
  The user (2026-11-23, "ok, please coordinate with @mobius"; 2026-11-26 via @air, "We need
  gains closer to 0.1 not 0.01"): agpt's dynamic evaluation gave d64/L2 models 0.10-0.17
  nats/char from a book's own past (rnd/dynamic-eval). How much does it give mobius's best
  network, a 10.8M transformer that already reads the count prior and the book's stream rows
  over 1,024 characters, on mobius's 50 whole held-out books, beyond adapting on another
  book's text?
answer: >-
  Not yet measured: the slot chain (run.sh) is queued.
opened: 2026-11-26
updated: 2026-11-26
code: main
eval: none
family: hybrid
tags: [context-length, adaptation, dynamic-eval, prior, books]
related: [dynamic-eval]
---

# Dynamic evaluation on mobius's prior + stream network

Diagnostic, mobius's protocol (bits per character over its 50 whole held-out
books), not agpt's canonical lm-eval. Agreed with @mobius:
`.ai/mail/received/20261123-161956Z-@mobius--*` and `20261126-185534Z-@mobius--*`.

## Setup

- **Networks** (mobius `runs/frozen/runs/`, trained by @air's
  `prior_input_only.py` on the 20M split's stream rows):
  - `pd_s10.8m_stream_svd32_b1024.pt`: 6 layers of 384, block 1,024 (sha 1a6b9d95…).
  - `pd_s25m_stream_svd32_b1024.pt`: 8 layers of 512, the size control (sha b5d1fc93…).
- **Books and rows:** `data/pg19/imm-split-20m-books` (50 books, 18,665,334
  characters, 76k-917k) with `runs/imm/pg19split20m_books_stream_d24_pg10g`.
  Static 10.8M on them: 1.3309 bits/char (mobius `77cf9c4`).
- **Protocol** (`dyneval_prior.py`): each book is scored by imm's
  `score_windows` exactly as mobius's `score_books.py` does (windows of 1,024
  at a stride of 512; each target scored once). After each window, one
  optimizer step on the mean NLL of the targets just scored. The model is in
  eval mode throughout. Weights and optimizer state reset at every book.
  - `full`: the transformer adapts; the prior's input layer and the output
    gate stay frozen (mobius: they price the prior).
  - `all`: everything adapts. `lora16`: a rank-16 delta on each block's four
    matrices.
  - Control (`past=other`): book i's k-th step trains on the k-th window of
    books i+1, i+2, ...
- **Tune and report:** settings are chosen on the even-indexed books and read
  on the odd ones.
- **Check (CPU, 2026-11-26):** the static scorer matches mobius's
  `loss_stream.npy` to +0.0001 nats on book 0's first 2,048 characters (float32
  here, bf16 on mobius's GPU).

## Results

Pending: the slot chain (`run.sh`, slot 218).
