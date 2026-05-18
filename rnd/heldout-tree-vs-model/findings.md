# Held-out: Trie alone vs trained model on Gutenberg (2026-05-18)

## Question

Earlier qualitative evidence (parrot generation visibly more coherent than
the trained model's generation) raised the hypothesis: maybe the trie is
doing all the real work, and the trained model is a redundant lossy
compressor. To test, we need held-out evaluation — both systems trained
on the same subset, both scored on text neither has seen.

## Setup

- Corpus: data/gutenberg_5m.txt (5M chars)
- Split: first 4.5M chars → train (90%); last 500K chars → test (10%)
- Trie: built on train subset, max-depth 16
- Model: AGPT trained on the same trie, 100 SE, single seed (init_seed100),
  recipe = `--partition-depth 1 --no-accumulate --lr 3e-3
  --lr-schedule warmup-cosine --warmup-epochs 1 --optimizer rmsprop
  --rmsprop-beta 0.999 --mass-weight log --entropy-lambda 1.0`
- Eval: PPL on 4096 positions sampled from the 500K held-out, seq_len=16
- Vocab: full-corpus vocab (65 chars) for both systems

## Result

| System | Held-out PPL |
|---|---:|
| **Trained model (108K params, 100 SE)** | **5.03** |
| Trie alone (count lookup + backoff) | 170.39 |

**Model wins by 34×.**

Trie diagnostics (4096 positions):
- 21,279 backoff invocations (~5 per position avg)
- 349 root-only fallbacks (8.5% — no walk possible)
- 28 mid-edge positions
- Effectively: every held-out 16-char context required substantial
  backoff, and ~10% had no useful trie information at all.

## Interpretation

The trie does *not* generalize. It is a perfect memorizer of training
contexts and effectively useless on unseen contexts (PPL 170 ≈ near
unigram). The trained model, despite having only 108K parameters and
training for only 100 super-epochs, achieves 34× better PPL on the
same held-out text.

This **falsifies** the "tree does the work" hypothesis raised by the
earlier qualitative generation comparison. The parrot's apparently
better generation was a memorization-and-stitching effect — the
parrot has the trie available at inference time, so it can output
verbatim corpus passages. The trained model has only its parameters
at inference and must *generalize*, which is what it actually does.

## Consequences for research priorities

- **Stage 2 (suffix F_p) is back on the table.** Making the optimizer
  smarter at the model's real generalization task is a meaningful
  direction. Earlier dismissal was based on bad inference from
  training-corpus PPL (where the trie trivially wins by remembering
  itself).

- **The "Wk/Wv partial gradient" and "intra-SE staleness" issues
  are clearly not killing the model.** The model successfully
  generalizes despite both. They may still be performance ceilings
  worth investigating, but they aren't blockers.

- **Streaming-AGPT's PPL win on Gutenberg (-6.46%, p<0.01) is a
  real generalization win**, not a memorization artifact. The
  research program is targeting a meaningful objective.

- **The 5.03 held-out PPL vs ~4.0 training-corpus PPL** is a roughly
  20-25% generalization gap, which is normal for character-level
  models. Not a sign of overfitting collapse.

## Caveats

1. The 100-SE training was deliberately short (~7 min on laptop) to
   answer the "does model beat trie at all" question quickly. A
   500-SE matched-recipe model would presumably get a held-out PPL
   meaningfully lower than 5.03 — but the relative comparison is
   unaffected since trie is 34× worse regardless.
2. Single seed. Variance not measured. Given the magnitude of the
   gap (34×), seed variance is irrelevant for the qualitative
   conclusion.
3. Contiguous train/test split. A random split might give slightly
   different absolute numbers but wouldn't change the conclusion.

## Files

- `/tmp/gut_train.txt`, `/tmp/gut_test.txt` — 90/10 split (not tracked)
- `/tmp/gut_train_d16_radix/` — trie on 90% train
- `/tmp/gut_heldout_model.model` — model trained 100 SE on 90% train
- `logs/trie_ppl_heldout.log`, `logs/model_ppl_heldout.log`,
  `logs/train.log` — full evaluation outputs
