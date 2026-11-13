---
title: Count prior vs training on its books at 1 GB (plain 10.8M on 1.02G characters)
kind: baseline
status: concluded
outcome: positive
question: >-
  The control for a count prior counted from more text than the network can read in its budget:
  is mobius's prior counted from 1 GB of PG-19 books, fed to a 10.8M char transformer trained on
  the 20M split, worth more than training the same transformer on those books plus the 20M, for
  the same 50k steps?
answer: >-
  Yes, at equal network steps, on one seed each. The plain 10.8M trained on the 1.02G characters
  reaches held-out 2.9058 PPL (1.5389 bits/char). The 10.8M with the 1 GB prior in its input and
  prior dropout 0.5 (mobius, rnd/unit-lattice) reaches 2.8559 (1.5139), 0.025 bits/char better,
  about 2.5 times the ~0.01 rerun noise. At 200M the same comparison was a tie (0.006). Going from
  200M to 1 GB of books gained the prior model 0.038 and the plain model 0.019. The prior's lead
  shrinks over training (0.137 at 5k steps, 0.040 at 15k, 0.025 at 50k) but is not gone. The
  prior's counting (about 2.6 h of CPU) is not in the comparison.
opened: 2026-11-13
updated: 2026-11-13
code: main
eval: legacy
family: count-prior
tags: [count-prior, scaling, data, pg19]
headline:
- label: plain 10.8M char transformer, 1 GB of books + 20M split, 50k steps, seed 1337
  metric: held-out PPL (air's protocol, 250k chars of PG-19 test books)
  value: 2.9058
  run: s10.8m_none_books1g
- label: 10.8M + 1 GB books prior in the input, prior dropout 0.5 (mobius) minus the plain 10.8M on 1 GB + 20M
  metric: held-out bits/char advantage at 50k steps
  value: 0.0250
related: [prior-scaling-plain-220m, polyrope-probe, gutenberg-prior-residual]
---

# prior-scaling-plain-1g

**Status:** concluded (2026-11-13). One run, in agpt at the user's request;
the second half of [prior-scaling-plain-220m](../prior-scaling-plain-220m/),
which found a tie at 200M of books.

## Question

At 200M characters of books, the count prior with prior dropout tied a
plain transformer trained on the same books (0.0057 bits/char, inside the
noise). The user's view is that the prior's use is to feed a small model
more text than it trains on. At 50k steps × 64 × 256 = 819M characters,
the plain model can read 220M characters 3.7 times, so 200M of books was
not more text than it could read. mobius has since counted a prior from
1 GB of books (`pg19split20m_gated_d16_pg1g`, 2,557 books), more than the
model can read once in the same budget. This run is the matching control:
the same plain transformer trained on those books.

## Design

- **Code:** `prior_transformer_bytes.py`, a copy of air's
  `experiments/prior_scaling/prior_transformer.py` (last change air
  `7c46036`; imm's `transformer_baseline` at `43057d9`) with two changes:
  imm's path is absolute, and the training text is held as a byte tensor
  (`np.fromfile` through a lookup table), with batches widened to int64.
  The original holds one int64 per character, 8 GB for this text. A
  30-step CPU run printed the same output, character for character, from
  both scripts. `--prior none`; the other arguments are those of air's
  `s10.8m_none` and of the 220M run: 6 layers, 6 heads, width 384, block
  256, dropout 0.2, batch 64, 50k iterations, lr 1e-3 to 1e-4, warmup 100,
  weight decay 0.1, seed 1337 (`run.sh`).
- **Config:** air's `pg19split20m_cfg.json` with only `run_id` and
  `train_input` changed (`config.json`).
- **Training text:** 1,019,779,493 characters, the first 999,779,493 of
  mobius's `data/pg19/imm-split-20m/books.txt` followed by imm's
  `train_corpus.txt` (20M). Those are its first 2,557 books, what the 1 GB
  prior counted (its `extra_counts.chars`). sha256
  `21eaf56afc24206a12648b8a72c38f39a4912ca2b2e4a1debd06c1c3f2e0fe72`; all 96
  characters are in the split's vocabulary.
- **How much it reads:** 819M target characters, 0.80 per character of
  text. Batches start at uniform random offsets, so about e^−0.80 = 45% of
  the characters are never a target. On the 220M text that share was 2.4%.
- **Evaluation:** air's protocol, unchanged (the `protocol` field matches
  air's runs): every held-out character after the first of the split's 250k
  held-out characters (50 pieces of PG-19 test books), with up to 256
  characters of context, every 2,500 steps.
- **The prior run compared with:** mobius ran @air's `pd_s10.8m_prior_in`
  (prior in the input and on the logits, prior dropout 0.5, seed 1337,
  50k steps) on the 1 GB prior's rows, reported in mobius's
  `rnd/unit-lattice/README.md` (2026-11-12 and 2026-11-13 sections). Its
  curve is read from that run's result file in mobius's working files, not
  committed anywhere.

## Results

Held-out PPL; bits/char in brackets. 10.8M parameters, seed 1337, 50k
steps, final evaluation. Rows marked air are in air's
`experiments/prior_scaling/runs/`; rows marked mobius are from its
`rnd/unit-lattice/README.md`.

| Model | Training text | Held out |
|---|---|---:|
| plain (air `s10.8m_none`) | 20M | 3.0386 (1.6034) |
| plain (agpt, `prior-scaling-plain-220m`) | 200M books + 20M | 2.9442 (1.5579) |
| **plain (this run, `s10.8m_none_books1g`)** | **1 GB books + 20M** | **2.9058 (1.5389)** |
| 200M books prior, input, prior dropout 0.5 (air `pd_s10.8m_prior_in`) | 20M | 2.9326 (1.5522) |
| 1 GB books prior, input, prior dropout 0.5 (mobius) | 20M | 2.8559 (1.5139) |
| 200M prior + document lattice rows (mobius) | 20M | 2.8557 (1.5138) |
| 1 GB prior + document lattice rows (mobius) | 20M | 2.8521 (1.5120) |

What the extra 800M characters of books bought, bits/char at 50k steps:

| | 200M books | 1 GB books | gain |
|---|---:|---:|---:|
| plain, trained on the books | 1.5579 | 1.5389 | 0.019 |
| prior counted from the books | 1.5522 | 1.5139 | 0.038 |

Over training, in bits/char (positive: the first model is ahead):

| Step | 1 GB prior over plain 1 GB | 200M prior over plain 1 GB | plain 1 GB over plain 220M |
|---:|---:|---:|---:|
| 2,500 | +0.206 | +0.164 | +0.001 |
| 5,000 | +0.137 | +0.073 | −0.011 |
| 10,000 | +0.062 | +0.004 | +0.012 |
| 15,000 | +0.040 | +0.002 | +0.017 |
| 20,000 | +0.034 | +0.004 | +0.018 |
| 30,000 | +0.050 | +0.008 | +0.016 |
| 40,000 | +0.038 | −0.008 | +0.018 |
| 45,000 | +0.031 | −0.012 | +0.019 |
| 50,000 | +0.025 | −0.013 | +0.019 |

## Reading

- **At 1 GB the prior beats training on its books.** At equal network steps
  the prior model ends 0.025 bits/char ahead. That is about 2.5 times
  @air's same-seed rerun spread at this size (up to about 0.01). It is one
  seed each.
- **The prior made twice the use of the extra books.** From 200M to 1 GB of
  books the prior model gained 0.038 and the plain model 0.019. The plain
  model's gain over the 220M model reached 0.017 by 15k steps and stayed
  near there to the end, while 45% of its text was never a target.
- **The plain model now beats the 200M prior,** by 0.013 at 50k steps, after
  trailing it until about 30k. More books trained in beat fewer books
  counted.
- **The prior's lead still shrinks.** It falls from 0.137 at 5k steps to
  0.040 at 15k and 0.025 at 50k, more slowly after 15k. At 200M the same
  lead fell from 0.062 to 0.006. Whether a longer budget would close the
  gap at 1 GB is not measured. The plain model would also reach more of its
  text in a longer run.
- **The two document-row runs land on the same 1.514,** 0.025 ahead of
  this model as well. mobius's document side is counted from the 20M
  alone, so it is a gain that does not come from more books at all.

## Caveats

- **One seed per run.** The 0.025 gap is above the measured rerun spread,
  not far above it.
- **Equal network steps, not equal compute.** The prior's rows took mobius
  about an hour of matching and about 1.6 h more of projections and folds
  on CPU, at a peak of 4.4–6.5 GB (mobius's notes). The networks trained in
  3,752 s (plain) and 3,934 s (prior), with evaluation. A plain run given
  the prior's extra time as more steps has not been run.
- **Books are joined end to end,** so a 256-character block can span two
  books, as in air's and the 220M runs.
- **The 20M split is 2% of this training text** against 9% of the 220M
  text. The held-out pieces come from PG-19's test books, not from the 20M
  split's books, so the split has no special claim on them.
- **Not lm-eval byte PPL:** air's protocol, the same for every row.
  `eval: legacy` in agpt's terms.
- `s10.8m_none_books1g.pt` (the final weights) is local and not committed.

## Files

`config.json`, `run.sh`, `prior_transformer_bytes.py`,
`s10.8m_none_books1g.json` (the result, with the curve),
`s10.8m_none_books1g.log`. Run in slot 14 of @air's queue (`tools/slots`),
10:02–11:04 UTC 2026-11-13, at nice 19, idle I/O priority, 2 CPU threads;
472 MiB of GPU memory, about 2.2 GB resident.
