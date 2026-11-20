---
title: Dynamic evaluation (the deep past through the weights)
kind: diagnostic
status: concluded
outcome: positive
question: >-
  The user (2026-11-19): "the missing link is the deep past having some influence in the
  present, like a LoRA on the trie." Can text far older than an 8-character window improve
  prediction if it acts through the weights (score a block, take a gradient step on it, go
  on), and how much of that is the document's own past rather than adaptation to held-out
  text in general?
answer: >-
  Yes. On the 50 PG-19 held-out chunks (5,000 characters, each from a different book), one
  Adam step on all weights per 64 characters scored gains 0.057-0.102 nats/char over
  the static model on the report half, across four d64/L2 checkpoints. The same steps on
  another book's text gain 0.026-0.063, so the document's own past is worth 0.021-0.041 on
  average. That part grows with the amount of past: about 0 in a chunk's first 1,000 characters,
  0.062-0.094 in its last 1,000, still rising at the chunk's end. A rank-4 or rank-16 LoRA
  (no biases) recovers 56-83% of the full-weight gain. Whole books (2026-11-20, 20 PG-19 test
  books of 300,000 characters, two checkpoints): the gain keeps growing to the end. Own past
  minus another book is 0.135 and 0.100 nats/char on the report books, and 0.173 and 0.135 in
  characters 200,000-300,000, still rising; the control's gain fades to 0.016 and 0.007 there.
  The adapted 1M-text model (1.4545) beats the static 100M-text model (1.5298). Eval-time only;
  training was unchanged.
opened: 2026-11-19
updated: 2026-11-20
code: main
eval: legacy
family: attention
tags: [context-length, adaptation, dynamic-eval]
headline:
- label: own past minus another book's text, mb-a1m-ref100m, report half
  metric: diagnostic fixed-window NLL gain, nats/char (50 PG-19 chunks, tune/report split)
  value: 0.0407
- label: own past minus another book's text, mb-a1m-ref100m, characters 4,000-5,000 of each chunk
  metric: diagnostic fixed-window NLL gain, nats/char (all 50 chunks)
  value: 0.094
- label: own past minus another book's text, mb-a1m, characters 200,000-300,000 of whole books
  metric: diagnostic fixed-window NLL gain, nats/char (20 PG-19 test books)
  value: 0.173
related: [refined-trie-counts, polyrope-probe, far-channel-probe, forward-backward-coupling]
---

# Dynamic evaluation

**Status:** concluded (2026-11-19), CPU only, commit `07d1646` (scorer) and
this one (results).

## Question

The user (2026-11-19): "I can't help but think that the missing link is the deep
past having some influence in the present, like a LoRA on the trie." A window
model sees only its last `seq_len` characters. The record measured the deep
past through attention: at d64/L2 on PG-19, exact attention over offsets
128-1,023 gained only 0.0025 nats/char (`polyrope-probe`). The other way in is
through the weights: score a block of the document, take a gradient step on
it, score the next (dynamic evaluation), so the weights carry everything
earlier in the document. A low-rank delta is the LoRA form. Under the user's
SGD-equivalence rule this is legitimate: the step is a sum of the past
windows' gradients, which a trie of the recent past could aggregate.

## Setup

- `dyneval.py`: the fixed-window protocol of the runner's
  `agpt_fixed_token_perplexity` (target i from tokens[i-8:i], each held-out
  chunk scored on its own, NLL pooled). Static = exactly that number. Adaptive:
  each chunk in order in blocks of 64 targets; score the block, then one
  optimizer step on that block's mean NLL (text already scored). Weights and
  optimizer state reset at every chunk, so a chunk's past is its own earlier
  text only.
- Settings: `full` (every parameter) with Adam 3e-4 / 1e-3 / 3e-3 or SGD
  0.01 / 0.03; `loraR` (a rank-R delta B·A on Q, K, V, O, both FFN maps and the
  output head; biases, embeddings and LayerNorms stay fixed) with rank 1, 4, 16
  and Adam 1e-3 to 1e-2.
- **Tune/report split:** each checkpoint's setting is the one with the lowest
  NLL on chunks 0-24; the numbers below are chunks 25-49.
- **Control (`--past other`):** the same steps, but chunk i's steps train on
  chunk i+1's text at the same positions. Every chunk comes from a different
  test book (`data/pg19/imm-split/manifest.json`), so this adapts to held-out-like
  text without this document's past. It also catches what any small step does
  to these models, which were trained at a constant Adam rate, never annealed.
- Checkpoints (all d64/L2, seq_len 8, from `rnd/refined-trie-counts`):
  `mb-a1m` (minibatch on the 1M slice), `mb-a1m-ref100m` (the same with the 100M
  text's next-character distributions), `mb-100m-st100` (minibatch on the 100M
  text, 25 epochs), `agpt-pd1-ref100m` (AGPT pd=1 replay, 128 passes, 100M
  distributions).
- `run.sh` (two CPU streams, 2 threads each, about 55 minutes), `results/`,
  `summarize.py` → `summary.json`.

## Checks

- Static reproduces the runner's number for every checkpoint: mb-a1m 5.3678
  (result.json 5.36775917628342), mb-a1m-ref100m 4.9009, mb-100m-st100 4.9085,
  agpt-pd1-ref100m 6.7677.
- Each block is scored before the step that trains on it; the next block is
  scored with the stepped weights.

## Results

NLL in nats/char. Gain = static minus adapted, report half (chunks 25-49).

| checkpoint | static | own past: setting | gain | other book: setting | gain | own − other |
|---|---:|---|---:|---|---:|---:|
| mb-a1m | 1.6941 | full-adam-1e-3 | 0.0965 | full-sgd-0.01 | 0.0630 | 0.0335 |
| mb-a1m-ref100m | 1.5999 | full-adam-1e-3 | 0.0666 | full-adam-3e-4 | 0.0258 | 0.0407 |
| mb-100m-st100 | 1.6003 | full-adam-1e-3 | 0.0569 | full-sgd-0.01 | 0.0360 | 0.0209 |
| agpt-pd1-ref100m | 1.9173 | full-adam-1e-3 | 0.1016 | full-sgd-0.03 | 0.0618 | 0.0399 |

Gain by position in the chunk (all 50 chunks, the settings above): own / other / own − other.

| checkpoint | 0-1,000 | 1,000-2,000 | 2,000-3,000 | 3,000-4,000 | 4,000-5,000 |
|---|---|---|---|---|---|
| mb-a1m | 0.051 / 0.061 / −0.010 | 0.086 / 0.060 / 0.026 | 0.105 / 0.054 / 0.052 | 0.099 / 0.047 / 0.051 | 0.143 / 0.053 / 0.090 |
| mb-a1m-ref100m | 0.030 / 0.035 / −0.005 | 0.062 / 0.030 / 0.032 | 0.079 / 0.024 / 0.054 | 0.073 / 0.020 / 0.053 | 0.108 / 0.014 / 0.094 |
| mb-100m-st100 | 0.028 / 0.042 / −0.014 | 0.048 / 0.037 / 0.012 | 0.066 / 0.035 / 0.031 | 0.064 / 0.031 / 0.033 | 0.089 / 0.027 / 0.062 |
| agpt-pd1-ref100m | 0.074 / 0.071 / 0.003 | 0.096 / 0.067 / 0.029 | 0.112 / 0.061 / 0.051 | 0.099 / 0.056 / 0.043 | 0.138 / 0.051 / 0.087 |

Low rank (own past, report half; best LoRA and best full setting on the tune half):

| checkpoint | LoRA | gain | full | gain | LoRA / full |
|---|---|---:|---|---:|---:|
| mb-a1m | lora4-adam-1e-2 | 0.0800 | full-adam-1e-3 | 0.0965 | 83% |
| mb-a1m-ref100m | lora16-adam-3e-3 | 0.0372 | full-adam-1e-3 | 0.0666 | 56% |
| mb-100m-st100 | lora4-adam-1e-2 | 0.0413 | full-adam-1e-3 | 0.0569 | 73% |
| agpt-pd1-ref100m | lora16-adam-3e-3 | 0.0784 | full-adam-1e-3 | 0.1016 | 77% |

## Reading

- **The deep past carries information an 8-character window lacks, and the
  weights can read it at this size.** All four checkpoints agree. The
  document-specific part grows with the amount of past, from about zero in the
  first 1,000 characters to 0.06-0.09 nats/char in the last 1,000, and the last
  bin is the largest, so 5,000-character chunks cut it off while it is still
  rising.
- **Adapting on any held-out-like text also helps** (the control, 0.026-0.063),
  most in the first 1,000 characters, then less as the weights drift toward the
  other book. Part of this is the constant learning rate: these models were never
  annealed, and a few small steps on similar text act like annealing. Own minus
  other removes this part; it may slightly overstate the document's part, since
  the control also pays for drifting toward a different book.
- **The model trained on the 100M text gains least** from its own past (0.021),
  the weaker models most. The AGPT checkpoint gains as much as the minibatch ones:
  this is a property of window models, not of how they were trained.
- **Low rank gets most of it but not all.** These LoRA deltas leave biases,
  embeddings and LayerNorms fixed, and the best rank-4 rate (1e-2) is the top of
  the tested range, so low rank is under-explored here.
- Against attention: `polyrope-probe` found 0.0025 nats/char from offsets
  128-1,023 at d64/L2 on PG-19. Not a clean comparison (a 16-window model trained
  to use that reach, another split, the legacy diagnostic), and the weight route
  here also covers offsets 8-127, but the weights pick up a document-level signal
  that attention at this size barely did.

## Whole books (2026-11-20)

The pieces ended while the gain was still rising, so `run_books.sh` scores
whole books: 20 PG-19 test books (the first 20 by file name with at least
310,000 characters), characters 10,000-310,000 of each (`make_books.py`, a port
of mobius's normalization; all 50 held-out pieces appear verbatim in its
output). Books 0-9 tune, 10-19 report. The same protocol, block 64, about 4,700
steps a book. Two checkpoints: `mb-a1m` (the 1M slice) and `mb-100m-st100`
(the 100M text). Own past: static, full Adam 1e-4 / 3e-4 / 1e-3, rank-4 LoRA
without biases at 1e-2, with biases (`lora4b`) at 3e-3 and 1e-2. Control
(another book): full Adam 1e-4 / 3e-4 / 1e-3. Two CPU streams, 12:24 UTC to
finish (2.5 hours). `summarize_books.py` → `books_summary.json`.

Report books (10-19), NLL in nats/char, each side's setting chosen on books 0-9:

| checkpoint | static | own past: setting | gain | other book: setting | gain | own − other | best LoRA | gain | share |
|---|---:|---|---:|---|---:|---:|---|---:|---:|
| mb-a1m | 1.6104 | full-adam-1e-3 | 0.1559 | full-adam-1e-4 | 0.0209 | 0.1350 | lora4b-adam-3e-3 | 0.1149 | 74% |
| mb-100m-st100 | 1.5298 | full-adam-1e-3 | 0.1147 | full-adam-1e-4 | 0.0150 | 0.0996 | lora4b-adam-3e-3 | 0.0827 | 72% |

Gain by position (all 20 books): own / other / own − other.

| checkpoint | 0-1k | 1k-2k | 2k-5k | 5k-10k | 10k-20k | 20k-50k | 50k-100k | 100k-200k | 200k-300k |
|---|---|---|---|---|---|---|---|---|---|
| mb-a1m | 0.030 / 0.041 / −0.011 | 0.110 / 0.041 / 0.069 | 0.086 / 0.039 / 0.048 | 0.142 / 0.046 / 0.096 | 0.117 / 0.038 / 0.079 | 0.131 / 0.036 / 0.095 | 0.165 / 0.031 / 0.134 | 0.170 / 0.023 / 0.148 | 0.189 / 0.016 / 0.173 |
| mb-100m-st100 | 0.012 / 0.037 / −0.025 | 0.050 / 0.026 / 0.023 | 0.050 / 0.029 / 0.020 | 0.077 / 0.026 / 0.050 | 0.072 / 0.026 / 0.047 | 0.091 / 0.023 / 0.067 | 0.119 / 0.019 / 0.100 | 0.125 / 0.013 / 0.112 | 0.143 / 0.007 / 0.135 |

- **The document's part keeps growing for the whole 300,000 characters**,
  about 0.02 nats/char for each doubling of the past, with no sign of levelling
  off. In the last bin it is 0.173 (mb-a1m) and 0.135 (mb-100m-st100): perplexity
  about 16% and 13% lower than the same model without it.
- **The control fades**: adapting to another book helps at first (0.04) and
  then pulls the weights toward the wrong book; its best rate is the lowest
  tried, and at 1e-3 it ends worse than static. So the long-range gain is this
  book's text, not general adaptation or the missing anneal.
- **The better-trained model gains less, but not much less**: 0.100 against
  0.135 on the report books. Adapting the 1M-text model to its own book (1.4545)
  beats the 100M-text model without adaptation (1.5298) by 0.075 nats/char.
- **Rank-4 LoRA with biases gets 72-74%** at 3e-3, a little more than without
  biases at 1e-2 (66-70%); 1e-2 with biases is too high.
- Caveats for this section: the best own-past rate, 1e-3, is the top of this
  grid (3e-3 was worse on the 5,000-character pieces); one ordering of books for
  the control; 20 books, two checkpoints, one seed each.

## Caveats

- Eval-time only. No model was trained to be adapted; the gain is what the
  existing weights allow.
- Diagnostic protocol (fixed window, one step per 64 characters), not the
  canonical lm-eval rolling score; the static setting equals the runner's
  fixed-token number.
- One model size (d64/L2), contexts of 8, one seed per checkpoint; the grid
  is small (11 settings, block 64 only).
- Cost: one backward pass per 64 characters scored, on top of the forward
  passes.

## Next (not started)

- Done 2026-11-20: whole books and LoRA with biases (above). Not done: where the
  gain stops growing (it hasn't by 300,000 characters); rates above 1e-3 on books.
- Training for it: adapt along each training piece and train the slow weights to
  do well after adaptation (test-time-training style), or a per-document trie of
  the past so the step is AGPT's exact aggregated gradient instead of a
  sequence of block steps.
