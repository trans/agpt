# Proper PPL Held-out Methodology

## Background

2026-05-23 discovery: `/tmp/shake_holdout.txt` was the last 4.5% of
`data/input.txt` (i.e., a subset of training); same for
`/tmp/gut_holdout.txt`. Months of "held-out PPL" numbers were
training-set PPL.

Immediate fix: `scripts/build_proper_heldout.py` produces
`/tmp/gut_holdout_proper.txt` from the tail of war_peace.txt (which
the README confirms is the truncated-away portion of the 5M training
corpus). Relative findings survived the re-evaluation — depth-weight=log
still wins on Gutenberg, branching still doesn't generalize — but
absolute PPL was inflated 0.3 PPL on average.

The war_peace-tail approach works but has limits:

1. Single author, single style
2. Tail of a document (epilogue, philosophical prose) — possibly
   not representative of typical text
3. Only addresses Gutenberg; Shakespeare side is harder (input.txt
   is essentially all of Shakespeare)
4. Same domain as training (Tolstoy → Tolstoy) — measures
   in-distribution generalization but not cross-domain transfer

## Improved methodology (to implement when held-out work is a priority)

### Multiple held-outs by purpose

| held-out type | source | what it measures |
|---|---|---|
| **in-corpus disjoint** | random non-contiguous N-gram slices excluded from trie build | how well model fits unseen positions of in-distribution text |
| **cross-document same-author** | full book by same author, not in training | within-author generalization (Tolstoy→Tolstoy, Dickens→Dickens) |
| **cross-author same-domain** | book by different author, same genre | within-domain generalization (e.g., Austen→Brontë for 19th-c novels) |
| **out-of-domain** | text from different domain entirely | how brittle is the model to distribution shift |
| **adversarial** | text designed to expose weaknesses (long-range deps, rare n-grams) | upper-bound failure modes |

For most experiments, in-corpus disjoint + cross-document is plenty.
The rest are for paper-writing or end-of-arc validation.

### In-corpus disjoint construction

Random non-contiguous slices are the cleanest "what does the model
fail to memorize" test. Concrete recipe:

1. Pick K=20 random 1000-char windows from the corpus, evenly
   distributed.
2. Exclude those exact windows from the corpus used to build the trie.
   (Replace with a sentinel or skip the chars.)
3. Retrain (yes, this requires rebuilding the trie + retraining).
4. Evaluate PPL on the 20×1000 = 20K char held-out.

Pros: in-distribution, controls for sampling noise, no overlap.
Cons: requires trie rebuild + retrain.

### Cross-document hold-out (no retrain)

For evaluating an already-trained model:

1. Identify books/documents NOT in the training corpus.
2. Apply the same preprocessing (Unicode → ASCII, vocab filter).
3. Evaluate.

Current implementation: `scripts/build_proper_heldout.py` does this
for Gutenberg using war_peace's tail (which the truncation removed
from training). Same logic could pull from other Project Gutenberg
books not in the training set.

### Shakespeare-specific challenge

`data/input.txt` is the full TinyShakespeare corpus = essentially
all of Shakespeare. Options for proper held-out:

1. Use a NON-Shakespeare Elizabethan text (Marlowe, Jonson) —
   measures cross-author generalization but distribution match isn't
   perfect.
2. Hold out one full Shakespeare play from input.txt, rebuild the
   trie, retrain. Requires re-running the entire Shakespeare
   experiment family.
3. Accept that absolute Shakespeare PPL can't be cleanly tested with
   what we have; report Gutenberg as the canonical generalization
   measure going forward, treat Shakespeare as "small-scale fast
   iteration corpus."

(3) is probably the right pragmatic move; Shakespeare's role is
quick experimentation, Gutenberg's role is real generalization
measurement.

## Files

- `scripts/build_proper_heldout.py` — current war_peace-tail
  Gutenberg held-out (200K chars, disjoint from gutenberg_5m.txt)
- `/tmp/gut_holdout_proper.txt` — output of the above; the current
  proper Gutenberg held-out
- `/tmp/gut_holdout.txt` and `/tmp/shake_holdout.txt` — the OLD
  training-set-tail held-outs. Should be deprecated for new
  experiments; relative comparisons against existing results
  remain valid.

## Effort

- War_peace-tail (current): ~30 min, done.
- Cross-document Gutenberg (any other PG book, not in training):
  ~1 hour to fetch + preprocess + integrate.
- In-corpus disjoint with retrain: ~4-8 hours (rebuild tries, retrain
  key experiments).
- Multiple-held-out infrastructure (a held-outs registry tool):
  half a day.

## Trigger

Pick this up when:
- We're writing up findings for external consumption (paper, blog)
- A specific finding hinges on absolute (not relative) PPL
- We want to compare AGPT to external baselines (KN, transformer
  baselines) on the same held-out
- Generalization-vs-memorization becomes a research question in itself
