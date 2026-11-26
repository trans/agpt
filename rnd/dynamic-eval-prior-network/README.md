---
title: Dynamic evaluation on mobius's prior + stream network
kind: diagnostic
status: concluded
outcome: positive
question: >-
  The user (2026-11-23, "ok, please coordinate with @mobius"; 2026-11-26 via @air, "We need
  gains closer to 0.1 not 0.01"): agpt's dynamic evaluation gave d64/L2 models 0.10-0.17
  nats/char from a book's own past (rnd/dynamic-eval). How much does it give mobius's best
  network, a 10.8M transformer that already reads the count prior and the book's stream rows
  over 1,024 characters, on mobius's 50 whole held-out books, beyond adapting on another
  book's text? And (the user via @air, 2026-11-26) on the 25.8M trained on 100M: do the
  text gain and the adaptation gain add up to about 0.1, and does limiting drift help?
answer: >-
  It gives 0.037 bits/char on the 10.8M and 0.067 on the 25.8M, on top of the stream rows
  (the 25 report books, paired by book: 25/25 better for both). The 10.8M goes from 1.3297 to
  1.2924, the 25.8M from 1.3408 to 1.2737. One Adam step per 512 characters scored, all
  transformer weights, lr 1e-4 (chosen on the other 25 books). A rank-16 LoRA (1.2920) and
  adapting the prior's layers too (1.2929) match full adaptation. On the networks mobius
  trained on 100M (the user via @air, slot 220; approximate, on the 20M books' rows) the
  gain is smaller: 0.029 on the 10.8M (1.3213 to 1.2922) and 0.031 on the 25.8M (1.2903 to
  1.2592, the best figure on these books). The text gain and the adaptation gain do not
  add: together they come to 0.071 below the 20M 10.8M's static 1.3297, short of the user's
  0.1. What the book's own past is worth against the same steps on another book's text is
  nearly constant over the four networks (0.063-0.069); what changes is the cost of the steps
  themselves, measured by that control (about 0 for the 20M 25.8M, which memorized its
  training text, 0.027-0.037 for the others). Limiting that cost does not raise the net gain
  (slot 221, the user via @air): decay toward the trained weights after each step (lambda
  0.003) cuts the control's cost from 0.037 to 0.022 but improves the own-past run by only
  0.001 (1.2592 to 1.2582), and Adam 3e-5 ties 1e-4 on the tune books. The own-past run was
  not paying the control's cost, so own minus control is not headroom.
headline:
- label: 10.8M (20M split), own past against static, report books
  metric: diagnostic bits/char gain, mobius's 50-book protocol (25 report books)
  value: 0.0374
- label: 25.8M (20M split), own past against static, report books
  metric: diagnostic bits/char gain, mobius's 50-book protocol (25 report books)
  value: 0.0671
- label: 25.8M (100M split) adapted, report books (approximate rows; 20M 10.8M static 1.3297)
  metric: diagnostic bits/char, mobius's 50-book protocol (25 report books)
  value: 1.2592
opened: 2026-11-26
updated: 2026-11-26
code: main
eval: none
family: hybrid
tags: [context-length, adaptation, dynamic-eval, prior, books, text-size, drift]
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

Slot 218, 2026-11-26 19:33-20:27 UTC, two runs at a time on the GPU; peak host
1.9 GB, GPU 0.7 GB per run (`out/*.json`; tables by `summarize.py`,
`summary.json`). Bits per character; "gain" is negative when the adapted run
is better. Pairs are per book against the same checkpoint's static run, over
the positions both scored.

**Check.** The static 10.8M here scores 1.3309 over all 50 books, the same as
mobius's `loss_stream.npy` (difference -0.00000 over 18,665,284 characters).
The static 25.8M: 1.3479.

**Tune books** (the 25 even-indexed books; static 10.8M 1.3321, 25.8M 1.3548):

| run | bits/char | gain by chars | per book ± s.e. | better |
|---|---|---|---|---|
| 10.8M full, Adam 1e-5 | 1.2904 | -0.0417 | -0.0365 ± 0.0035 | 25/25 |
| 10.8M full, Adam 3e-5 | 1.2773 | -0.0548 | -0.0489 ± 0.0046 | 25/25 |
| 10.8M full, Adam 1e-4 | 1.2733 | -0.0588 | -0.0533 ± 0.0059 | 25/25 |
| 10.8M full, Adam 3e-4 | 1.3059 | -0.0262 | -0.0237 ± 0.0061 | 18/25 |
| 10.8M LoRA 16, Adam 1e-3 | 1.2761 | -0.0560 | -0.0508 ± 0.0052 | 25/25 |
| 10.8M LoRA 16, Adam 3e-3 | 1.3200 | -0.0122 | -0.0174 ± 0.0056 | 17/25 |
| 25.8M full, Adam 3e-5 | 1.2608 | -0.0940 | -0.0861 ± 0.0075 | 25/25 |
| 25.8M full, Adam 1e-4 | 1.2549 | -0.0999 | -0.0933 ± 0.0090 | 25/25 |

Chosen: full at 1e-4 for both networks, LoRA at 1e-3. The 25.8M was tried
only at 3e-5 and 1e-4, and was still improving at 1e-4.

**Report books** (the 25 odd-indexed books; static 10.8M 1.3297, 25.8M 1.3408):

| run | bits/char | gain by chars | per book ± s.e. | better |
|---|---|---|---|---|
| 10.8M own past (full) | 1.2924 | -0.0374 | -0.0373 ± 0.0056 | 25/25 |
| 10.8M other-book control | 1.3564 | +0.0267 | +0.0243 ± 0.0023 | 1/25 |
| 10.8M own past, everything adapting | 1.2929 | -0.0369 | -0.0368 ± 0.0056 | 25/25 |
| 10.8M own past, LoRA 16 | 1.2920 | -0.0377 | -0.0377 ± 0.0048 | 25/25 |
| 25.8M own past (full) | 1.2737 | -0.0671 | -0.0701 ± 0.0085 | 25/25 |
| 25.8M other-book control | 1.3431 | +0.0023 | -0.0027 ± 0.0046 | 12/25 |

Own past against the control: 10.8M -0.0641 by chars (per book -0.0615 ±
0.0060), 25.8M -0.0694 (-0.0674 ± 0.0064), 25/25 books for both.

By position in the book (report books; gain against static, own past and
control, and own minus control):

| characters | share | 10.8M own | control | own - control | 25.8M own | control | own - control |
|---|---|---|---|---|---|---|---|
| 0-1k | 0.003 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1k-5k | 0.011 | -0.0183 | -0.0086 | -0.0097 | -0.0472 | -0.0285 | -0.0187 |
| 5k-25k | 0.054 | -0.0134 | +0.0060 | -0.0194 | -0.0401 | -0.0173 | -0.0228 |
| 25k-100k | 0.199 | -0.0298 | +0.0165 | -0.0463 | -0.0603 | -0.0095 | -0.0508 |
| 100k-300k | 0.400 | -0.0388 | +0.0292 | -0.0680 | -0.0686 | +0.0058 | -0.0743 |
| 300k+ | 0.333 | -0.0450 | +0.0344 | -0.0795 | -0.0749 | +0.0096 | -0.0845 |

(The first 1,024 characters are scored before any step.)

Per book, the 10.8M's gain runs from 0.003 to 0.132 bits/char and the
25.8M's from 0.025 to 0.216; neither correlates with the book's length
(r = -0.01 and 0.15 against log length).

## Reading

- **On top of the stream rows, dynamic evaluation adds 0.037 bits/char to
  the 10.8M and 0.067 to the 25.8M.** The stream rows themselves take 0.072
  from the book's own past (mobius). The adapted 25.8M, 1.2737, is the best
  figure on these books; statically it is 0.011 worse than the 10.8M.
- **The book-specific part is about the same for both networks** (own minus
  control 0.064 and 0.069). What differs is the control: adapting the 10.8M
  on other books' text costs it 0.027, adapting the 25.8M costs about 0. The
  25.8M memorizes its training rows (mobius: 1.3787 on the pieces), so text
  from any held-out book seems to repair some of that, but this run does not
  separate the two.
- **The gain grows through the book and has not levelled off** past 300k
  characters (own minus control 0.068 in 100k-300k, 0.080-0.085 after).
- **Which weights adapt matters little.** A rank-16 LoRA on the four matrices
  of each block matches full adaptation (1.2920 against 1.2924) in two-thirds
  of its time; letting the prior's input layer and output gate
  adapt as well changes nothing (1.2929).
- **Against agpt's earlier dynamic eval** (`rnd/dynamic-eval`, d64/L2 on
  whole books, no prior): own minus control was 0.100-0.135 nats/char there,
  0.044-0.048 nats/char (0.064-0.069 bits) here. The stronger network, which
  already reads the book through the stream rows, leaves less to the weights.
- **Not tried:** a 25.8M rate above 1e-4; more than one step per 512
  characters or a shorter stride; Krause et al.'s RMS-scaled steps with decay
  toward the trained weights. (The 100M-trained networks: below.)

## The networks trained on 100M (slot 220)

The user, via @air (2026-11-26, `.ai/mail/received/20261126-211810Z-@air--*`):
does adaptation on mobius's 100M-trained 25.8M bring the total to about 0.1?
`run_100m.sh`, slot 220, 21:21-21:55 UTC, one run at a time (peak host 1.9
GB). Networks: `pd_s25m_stream100m_svd32_b1024.pt` (8 × 512, sha 44744c92…,
1.3231 on mobius's 50 pieces) and `pd_s10m8_stream100m_svd32_b1024.pt` (6 ×
384, sha 4e4d5595…, 1.3525 there). Both use the same vocabulary as the rows
(the vocab files are identical). **Approximate:** the rows are the 20M
books' rows. Their counts include the 200 books dropped from the 100M split,
and four of the 100M training books share 550 held-out positions with these
books. Mobius sizes these at about 0.0003 and under 0.0001 bits.

Static, all 50 books: the 25.8M 1.2883, the 10.8M 1.3186 (against 1.3309 for
the 20M 10.8M).

Tune books (static 25.8M 1.2864): Adam 1e-4 gives 1.2369 (-0.0495, per book
-0.0444 ± 0.0054, 25/25), 2e-4 gives 1.2564 (-0.0301, 22/25). Chosen: 1e-4.
The 10.8M takes 1e-4, the 20M 10.8M's tuned rate.

Report books:

| network | static | own past | gain, per book ± s.e. | better | control | own - control |
|---|---|---|---|---|---|---|
| 10.8M, 20M | 1.3297 | 1.2924 | -0.0374, -0.0373 ± 0.0056 | 25/25 | +0.0267 | -0.0641 |
| 25.8M, 20M | 1.3408 | 1.2737 | -0.0671, -0.0701 ± 0.0085 | 25/25 | +0.0023 | -0.0694 |
| 10.8M, 100M | 1.3213 | 1.2922 | -0.0291, -0.0275 ± 0.0042 | 23/25 | +0.0342 | -0.0633 |
| 25.8M, 100M | 1.2903 | 1.2592 | -0.0311, -0.0301 ± 0.0051 | 23/25 | +0.0366 | -0.0677 |

(Own minus control is 25/25 books for all four.)

By position, the 100M networks gain little early. The 25.8M gains -0.0024
in characters 1k-5k, -0.0086 in 5k-25k, -0.0239 in 25k-100k, -0.0330 in
100k-300k and -0.0379 past 300k. Its control costs grow alongside:
+0.0047, +0.0140, +0.0257, +0.0394, +0.0449. Own minus control is the same
shape as on the 20M networks, -0.0827 past 300k.

**The two gains do not add.** Against the 20M 10.8M's static 1.3297 on the
report books, the 100M 25.8M takes 0.0394 statically and 0.0311 more by
adaptation, 0.0705 in all. Adaptation on the 20M 25.8M took it to 1.2737,
so the 100M training adds 0.0145 on top of adaptation. On the 10.8M the
100M training's static gain is used up entirely: adapted, both 10.8Ms score
1.292.

**Reading.** On all four networks the book's own past beats another book's
text by 0.063-0.069 bits/char. The net gain is smaller by what the control
measures: adapting on text that isn't the book costs 0.027-0.037 on three of
the networks, and about nothing on the 20M 25.8M, which memorized its
training text and gets something back from any held-out book. The own-past
run doesn't necessarily pay the control's cost, since its steps are on the
text it then scores. Still, a method that limits drift is the plain next
test: decay toward the trained weights (Krause et al. 2018), or a lower rate
on the 100M networks (only 1e-4 and 2e-4 tried, and 2e-4 is worse).

## Limiting drift (slot 221)

The user approved the test via @air (2026-11-26,
`.ai/mail/received/20261126-221917Z-@air--*`). `run_drift.sh`, slot 221,
22:21-22:53 UTC, one run at a time, on the 100M 25.8M. The decay arm takes
setting suffix `-dLAMBDA`: after each Adam step, w += λ(w0 − w). This is
Krause et al. 2018's decay without their RMS step scaling.

Tune books (static 1.2864):

| setting | bits/char | gain, per book ± s.e. | better |
|---|---|---|---|
| Adam 1e-4 (slot 220) | 1.2369 | -0.0495, -0.0444 ± 0.0054 | 25/25 |
| Adam 1e-4, decay 0.003 | 1.2373 | -0.0491, -0.0446 ± 0.0049 | 25/25 |
| Adam 1e-4, decay 0.01 | 1.2432 | -0.0432, -0.0399 ± 0.0042 | 25/25 |
| Adam 3e-5 | 1.2376 | -0.0488, -0.0429 ± 0.0043 | 25/25 |
| Adam 2e-4 (slot 220) | 1.2564 | -0.0301, -0.0268 ± 0.0056 | 22/25 |

The rate arm chose 1e-4, already reported. The decay arm chose 0.003; the
report books (static 1.2903):

| setting | own past | control | own - control |
|---|---|---|---|
| Adam 1e-4 | 1.2592 (-0.0311) | 1.3269 (+0.0366) | -0.0677 |
| Adam 1e-4, decay 0.003 | 1.2582 (-0.0321) | 1.3121 (+0.0218) | -0.0539 |

Decay against no decay, paired per book: own past -0.0010 by chars (per book
-0.0014 ± 0.0004, better in 19/25); control -0.0148 (-0.0127 ± 0.0010,
25/25). By position, decay saves the control 0.0057 in 25k-100k, 0.0152 in
100k-300k and 0.0227 past 300k. It helps the own-past run by at most 0.0022
(100k-300k) and costs it 0.0009 past 300k.

**Reading.** Decay removes 40% of the control's cost, but the own-past run
gains only 0.001. Steps on the book's own text don't do the harm that steps
on another book's text do, so the control's cost was never part of the
own-past run's score. Own minus control (0.065) measured how much better the
book's own text is than another book's text to adapt on. It isn't a ceiling
the net gain could reach by removing drift; I suggested it might be, and
this run says no. The lower rate doesn't help either. The 100M 10.8M was not
run: the tune books chose plain Adam 1e-4, which it already has.

Cost per character, two runs sharing the GPU in each case: the adapted 10.8M
takes 3x its static run (332 s for the 9.2M report characters against 220 s
for all 18.7M), LoRA 2x, the adapted 25.8M 6x (635 s against 222 s). Alone on
the GPU (slot 220) the 25.8M takes 333 s for the report books with adaptation
and 134 s for all 50 books static, 5x per character.
