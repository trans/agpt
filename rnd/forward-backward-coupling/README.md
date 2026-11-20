---
title: Forward-backward coupling
kind: experiment
status: concluded
outcome: negative
question: >-
  Does pulling a forward model toward a backward model's prediction of each character (the
  same character predicted from the 8 that follow it) make the forward model better at
  predicting the next character? The record's dual-model-fold measured only the mechanism.
answer: >-
  No, it hurts. With each target's loss (CE + beta CE(B, model)) / (1 + beta) and B fixed, beta
  0.1 is level with the plain model (+0.0008 nats canonical rolling) and beta 1 is 0.155
  nats worse, at every checkpoint. The control, the same pull toward a forward teacher of the
  same recipe (plain distillation), helps a little: -0.0079 and -0.0124. B's prediction
  depends on text the forward model never sees; averaged over what follows a given past it is
  not the right next-character distribution, so as a target it pulls the forward model off.
  One seed, d64/L2, contexts of 8, the 1M PG-19 slice.
opened: 2026-11-19
updated: 2026-11-20
code: main
eval: canonical
family: attention
tags: [targets, prefix-suffix, distillation]
headline:
- label: backward teacher, beta 1, vs the plain model
  metric: canonical rolling byte PPL (50 PG-19 held-out chunks), epoch 8
  value: 7.6848
  run: 20261120T014914-fb-b1
- label: forward teacher (control), beta 1
  metric: canonical rolling byte PPL (50 PG-19 held-out chunks), epoch 8
  value: 6.5017
  run: 20261120T014934-ff-b1
- label: plain forward model
  metric: canonical rolling byte PPL (50 PG-19 held-out chunks), epoch 8
  value: 6.583
  run: 20261120T005456-mb-a1m-base
related: [dual-model-fold, prefix-suffix-bayes, refined-trie-counts, dynamic-eval]
---

# Forward-backward coupling

**Status:** concluded (2026-11-20). Code `07d1646`; six runs through
`bin/agpt_experiment` on the CPU.

## Question

The user (2026-11-19) asked why AGPT uses only "where a character goes" and
not "where it comes from", and whether the flow could run both ways. The record
had the pieces: the forward and reversed tries hold the same counts
(`prefix-suffix-bayes`, KL 0), and a KL term between a forward and a backward
model shrinks their disagreement (`dual-model-fold`), but nobody measured
whether the forward model got better at its own job. The user: "Well feel
free."

## Setup

The refined-trie-counts minibatch recipe throughout: d64/L2 (h4, d_ff 256),
random init seed 42, contexts of up to 8 characters, the 1M PG-19 slice, 128
windows a step, every window start once an epoch, 8 epochs, Adam 0.003
constant, canonical eval on the 50 held-out chunks. Trained on the CPU (2
threads a run; the GPU queue was hours long), so `mb-a1m-base` here, not the
GPU `mb-a1m`, is the baseline.

- `mb-a1m-rev`: the backward model B, the same recipe on the reversed slice
  (`make_data.py`), scored on the reversed held-out chunks.
- `fb-b0.1`, `fb-b1`: forward models; each target's loss is
  (CE(y) + beta CE(B, model)) / (1 + beta), i.e. the data term plus beta
  KL(B || model) up to a constant, with B fixed. B's distribution for character
  p is given the 8 characters after it (`teacher_preds.py --direction backward`;
  experimental.teacher_targets_file in the replay trainer).
- `ff-b0.1`, `ff-b1`: the control, the same with a forward teacher (the GPU
  `mb-a1m` from refined-trie-counts, given the 8 characters before): plain
  distillation of the same strength. Help from B beyond this would be the other
  side's information.
- Checks: teacher_beta 0 reproduces the plain run to six decimals (smoke test,
  20k characters, CPU); the teacher files are checked by SHA-256 in each run.

## Results

Canonical rolling byte PPL / fixed-token PPL at each checkpoint; gap to the
plain model at epoch 8 in nats.

| run | epoch 1 | epoch 2 | epoch 4 | epoch 8 | gap, rolling / fixed |
|---|---|---|---|---|---|
| mb-a1m-base (plain) | 6.9941 / 5.8548 | 6.7609 / 5.5920 | 6.6492 / 5.4604 | 6.5830 / 5.3758 | 0 |
| fb-b0.1 (backward, 0.1) | 7.0566 / 5.9139 | 6.8241 / 5.6544 | 6.6802 / 5.4779 | 6.5884 / 5.3813 | +0.0008 / +0.0010 |
| fb-b1 (backward, 1) | 8.2611 / 7.1489 | 7.9014 / 6.7328 | 7.8119 / 6.6122 | 7.6848 / 6.4545 | +0.1547 / +0.1829 |
| ff-b0.1 (forward, 0.1) | 6.9320 / 5.7925 | 6.6827 / 5.5216 | 6.5906 / 5.3936 | 6.5315 / 5.3230 | −0.0079 / −0.0099 |
| ff-b1 (forward, 1) | 6.7870 / 5.6345 | 6.6261 / 5.4629 | 6.5329 / 5.3513 | 6.5017 / 5.3096 | −0.0124 / −0.0124 |
| mb-a1m-rev (backward model) | 7.0768 / 5.9503 | 6.8384 / 5.6580 | 6.6839 / 5.4895 | 6.5997 / 5.3904 | reversed held-out |

`summarize.py` → `runs.json`.

### Where each direction finds the text hard (diagnostic, training text)

`per_position.py` reads the two teacher files (the GPU `mb-a1m` forward, the
backward `mb-a1m-rev`) at every position of the 1M training slice: each
character's loss given the 8 characters before it and given the 8 after.
Training text, so not a held-out number.

- Totals: forward 1.4729, backward 1.4925 nats/char.
- Per position they differ by 1.42 nats on average (as much as the mean loss
  itself), with correlation 0.125; 55% of positions differ by more than 1 nat.
- By target, mean nats (share of the direction's total):

| target | share of text | forward | backward |
|---|---:|---:|---:|
| space or newline | 18.8% | 0.740 (9.4%) | 0.533 (6.7%) |
| first letter of a word | 16.9% | 2.881 (33.1%) | 1.119 (12.7%) |
| last letter of a word | 17.0% | 0.763 (8.8%) | 2.518 (28.8%) |
| other letters | 43.1% | 1.386 (40.6%) | 1.564 (45.2%) |
| punctuation, digits, etc. | 4.1% | 2.872 (8.0%) | 2.412 (6.7%) |

The branching difference is large and real: forward, a word's first letter is
the hard one (many ways out of a space); backward, its last letter is (many
ways into a space). It moves the difficulty from one end of the word to the
other and leaves the total where it was, as the chain rule requires: the two
directions split the same probability of the same text. It is also why B is a
bad teacher: B's confidence on a first letter comes from seeing the rest of
the word.

## Reading

- **The backward teacher hurts, in proportion to its weight.** At beta 0.1
  the forward model ends level (it trailed by 0.009 nats at epoch 1); at beta 1
  it is 0.155 nats worse at the end and worse at every checkpoint.
- **Why, in one line.** For a given past, the coupled loss is minimized by
  (p(x | past) + beta E[B(x | future) | past]) / (1 + beta). The second term
  is B's prediction averaged over the futures that follow this past, which is
  not p(x | past): B predicts x from text the forward model never sees, so
  matching it moves the forward model away from the right answer. Coupling the
  two models, as dual-model-fold did, makes them agree by making the forward
  model worse at its own task.
- **Distillation from a forward teacher helps a little** (−0.008 and −0.012
  nats), most early on (−0.03 at epoch 1 with beta 1): the soft targets of a
  same-recipe model, as in refined-trie-counts' trie targets but weaker.
- **Direction costs almost nothing:** the backward model on reversed text
  scores within 0.003 nats of the forward one (6.5997 vs 6.5830), as the
  two-way factorization of the same text predicts.
- With the masked-LM combiner (June: both sides help when both are seen, but
  do not train a better next-character model), this closes the record's
  prefix-suffix coupling line for next-character prediction at this size. The
  other side is useful where the future is available (fill-in, scoring a
  finished text), not as a teacher for a model that predicts forward.

## Caveats

One seed per run, one model size, contexts of 8, a fixed (not jointly trained)
backward teacher. Joint training would let B move toward the forward model too,
but the forward model's optimum above would still be pulled off p(x | past).
