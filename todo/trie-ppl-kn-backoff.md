# Trie-PPL evaluator: proper Kneser-Ney backoff

## Status

Open. Created 2026-05-18 alongside `agpt_trie_perplexity` to track
the known limitation.

## What's wrong

`bin/agpt_trie_perplexity` uses naive backoff: walks the trie with
held-out context, finds deepest match, computes log-prob. If target
has zero count at the deepest match, backs off ONE char and tries
again. Stops at the first context length where target has any
non-zero count.

The bug: stops too early. A deep context where target has count=1
out of total_mass=1000 (prob=1e-3, NLL=6.9 nats) "wins" over a
shorter context where target has count=100 out of total_mass=200
(prob=0.5, NLL=0.69). My loop picks the deep-rare option.

## Impact

On held-out Gutenberg 5M (90/10 split, seq_len=16):
- Naive backoff reported PPL 170
- Estimated PPL with proper KN: ~15-30
- Model PPL: 5.03 (unaffected)

The qualitative finding "model >> trie on held-out generalization"
is robust to this issue. The quantitative magnitude (34× claim) is
overstated by ~5-10×.

## What it would take to fix

Three options, ranked by effort:

**Option 1 — Less-naive backoff (~30 min):**
Don't stop at first non-zero count. Continue backing off as long as
each shorter context gives a HIGHER log-prob. Stop when the next
backoff would make things worse (or when context exhausts to root).

**Option 2 — Kneser-Ney interpolation (~2-3 hr):**
Implement modified Kneser-Ney smoothing. Discount the counts of
observed n-grams; redistribute discounted mass to a backoff
distribution. Standard textbook algorithm. Would give the proper
"how well does the trie ALONE predict next char" baseline.

**Option 3 — Use KenLM (~30 min install + 30 min wrap):**
Install kenlm via `paru -S kenlm` (AUR has it). Convert corpus to
space-separated chars. Build character-level n-gram model with
`lmplz -o 16`. Score held-out via the kenlm Python API or
command-line scoring.

Option 3 is the strongest baseline (KenLM is the industry-standard
modified KN). Option 2 is interesting if we want to understand the
algorithm. Option 1 is the cheap fix that probably captures most of
the win.

## Why we deprioritized it

Once we established the model beats the trie meaningfully on held-out
(by some factor ≥ 3-5×), the exact magnitude doesn't change the
research direction. We've moved on to "the model is doing real work,
let's improve it" rather than "is the model doing anything at all."

A proper KN baseline is still worth having for:
- Grant-pitch credibility ("we beat the canonical n-gram baseline by X")
- Sanity-checking that scaling experiments aren't degrading below
  the trivially-strong baseline
- Quantifying the actual generalization gap precisely

So this is a "do before the next major writeup" task, not an urgent
blocker.

## Acceptance criteria

- Trie-only PPL on held-out Gutenberg matches what a third-party KN
  library (like KenLM) reports, within a few percent
- The relative comparison vs trained-model PPL is consistent across
  multiple corpora / setups
