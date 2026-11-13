---
title: Count prior vs training on its books (plain 10.8M on 220M characters)
kind: baseline
status: concluded
outcome: mixed
question: >-
  The user's framing of the count prior as a data resource: is a prior counted from 200M
  characters of books, fed to a 10.8M char transformer trained on a 20M split, worth more than
  simply training the same transformer on those books plus the 20M, at the same compute?
answer: >-
  At 50k steps it is a tie for the best prior form and a loss for the other. The plain 10.8M
  trained on the 220M characters reaches held-out 2.9442 PPL (1.5579 bits/char), 0.0455 bits/char
  better than on the 20M alone. @air's prior model with the prior in its input and prior dropout
  0.5 reaches 2.9326, only 0.0057 bits/char better, inside the ~0.01 run-to-run noise at this
  size; with the prior on the logits only it reaches 2.9890, 0.022 worse. The prior's lead is
  early: 0.062 bits/char at 5k steps, shrinking steadily to 0.006 at 50k, while the books-trained
  model's gain over 20M keeps growing. One seed each.
opened: 2026-11-13
updated: 2026-11-13
code: main
eval: legacy
family: count-prior
tags: [count-prior, scaling, data, pg19]
headline:
- label: plain 10.8M char transformer, books 200M + 20M split, 50k steps, seed 1337
  metric: held-out PPL (air's protocol, 250k chars of PG-19 test books)
  value: 2.9442
  run: s10.8m_none_books220m
- label: 10.8M + books prior in the input, prior dropout 0.5 (air pd_s10.8m_prior_in) minus the plain 10.8M on 220M
  metric: held-out bits/char advantage at 50k steps
  value: 0.0057
related: [polyrope-probe, gutenberg-prior-residual, count-prior-residual]
---

# prior-scaling-plain-220m

**Status:** concluded (2026-11-13). One run, in agpt for the user; the test
@air proposed (air's `.ai/memory/todo.md`, 2026-11-12) for the user's
framing that the count prior is a data resource whose value is how much
information it feeds, and how fast, compared with a model learning the same
text.

## Question

@air's prior-scaling study (air `experiments/prior_scaling/`, one seed per
run) put mobius's count prior, built from 200M characters of PG-19 books plus
the 20M training split, into imm's char transformer trained on the 20M split
alone. At 10.8M parameters the prior helped only with prior dropout 0.5:
2.9326 held-out PPL (prior in the input and on the logits) against 3.0386 for
the plain transformer. But that plain transformer never saw the books. The
fair comparison for "the prior is a data resource" is the plain transformer
trained on the books themselves, at the same compute.

## Design

- **Code:** air's `experiments/prior_scaling/prior_transformer.py`,
  unchanged (last change air `7c46036`; imm's `transformer_baseline` at
  `43057d9`), with `--prior none`. That reproduces imm's
  `transformer_baseline.py`. Arguments are those of air's `s10.8m_none`:
  6 layers, 6 heads, width 384, block 256, dropout 0.2, batch 64, 50k
  iterations, lr 1e-3 to 1e-4, warmup 100, weight decay 0.1, seed 1337
  (`run.sh`). The output's `model` field names a gated prior, but that is
  a fixed string in the script; `prior=none` is in the log.
- **Config:** air's `pg19split20m_cfg.json` with only `run_id` and
  `train_input` changed (`config.json`).
- **Training text:** 219,949,702 characters, the first 199,949,702 of
  mobius's `data/pg19/imm-split-20m/books.txt` followed by imm's
  `train_corpus.txt` (20M). Those are its first 500 books, all from PG-19's
  train part, exactly what the books prior counted
  (`pg19split20m_gated_d16_pg200m.json`: `extra_counts.chars` 199,949,702,
  `books` 500). sha256 `f5815b65bdb8354126b87ac21dd16aa2faeb832be383c050fcebda9332897280`.
  All 96 characters are in the split's vocabulary.
- **Same compute:** 50k steps × 64 × 256 = 819M characters, about 3.7 passes
  over the 220M (the 20M runs make about 41). Training took 3,748 s, against
  3,823 s (plain, 20M), 3,967 s (prior in the input) and 3,877 s (prior on
  the logits), all with evaluation. The prior's own cost, counting the 200M
  and building its rows, is not included in those times.
- **Evaluation:** air's protocol: every held-out character after the first
  of the split's 250k held-out characters (50 pieces of PG-19 test books),
  with up to 256 characters of context, every 2,500 steps.

## Results

Held-out PPL; bits/char in brackets. 10.8M parameters, seed 1337 each, 50k
steps. The other rows are air's runs (`experiments/prior_scaling/runs/`).

| Model | Training text | Last | Best |
|---|---|---:|---:|
| plain (`s10.8m_none`) | 20M | 3.0386 (1.6034) | 3.0386 |
| **plain (this run, `s10.8m_none_books220m`)** | 200M books + 20M | **2.9442 (1.5579)** | 2.9442 |
| books prior on the logits, prior dropout 0.5 (`pd_s10.8m_prior`) | 20M | 2.9890 (1.5797) | 2.9890 |
| books prior in the input and on the logits, prior dropout 0.5 (`pd_s10.8m_prior_in`) | 20M | 2.9326 (1.5522) | 2.9280 |
| books prior on the logits, no dropout (`s10.8m_prior`) | 20M | 3.0397 (1.6039) | 3.0397 |
| books prior in the input, no dropout (`s10.8m_prior_in`) | 20M | 3.0667 (1.6167) | 3.0655 |

Over training, in bits/char (positive: the first model is ahead):

| Step | plain 220M over plain 20M | prior input over plain 220M | prior logits over plain 220M |
|---:|---:|---:|---:|
| 5,000 | +0.0339 | +0.0622 | +0.0680 |
| 10,000 | +0.0079 | +0.0161 | +0.0571 |
| 20,000 | +0.0168 | +0.0224 | +0.0275 |
| 30,000 | +0.0256 | +0.0247 | +0.0104 |
| 40,000 | +0.0443 | +0.0106 | −0.0091 |
| 50,000 | +0.0455 | +0.0057 | −0.0218 |

## Reading

- **The best prior form ties with training on the books.** The prior in the
  input with dropout ends 0.0057 bits/char ahead of the books-trained model.
  @air measured same-seed reruns at 10.8M differing by up to about 0.01
  bits/char, so this is a tie on one seed each.
- **The prior feeds information faster, not more.** It is 0.06 bits/char
  ahead at 5k steps, and the lead shrinks at every later evaluation. The
  books-trained model's gain over the 20M model grows instead, from 0.008 at
  10k to 0.0455 at 50k. With the prior on the logits only, the books-trained
  model passes it at about 35k steps.
- **Where it goes next is a guess.** The books-trained model had seen its
  text 3.7 times, the prior models theirs 41 times, and every curve is still
  falling at the end of its learning-rate decay. A longer budget would
  likely favour the model with more text left. That is a reading of the
  trends, not a measurement.
- **My earlier guess was off.** On 2026-11-12 I expected the plain model on
  220M to gain more than 0.05 bits/char over the 20M model and, by
  implication, to beat the prior model. It gained 0.0455, and did not beat
  the best prior model.

## Caveats

- **One seed per run.** The decisive difference (0.0057) is below the
  rerun noise.
- **Books are joined end to end,** so a 256-character block can span two
  books; air's 20M runs join their 800 pieces the same way.
- **Not lm-eval byte PPL:** air's protocol, the same for every row of the
  table. `eval: legacy` in agpt's terms.
- **The prior's preprocessing is not counted** in the compute comparison.
- `s10.8m_none_books220m.pt` (the final weights) is local and not committed.

## Files

`config.json`, `run.sh`, `s10.8m_none_books220m.json` (the result, with the
curve), `s10.8m_none_books220m.log`. Run in a slot from @air's queue
(`tools/slots`), at nice 19, idle I/O priority, 2 CPU threads.
