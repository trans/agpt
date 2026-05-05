# Forward vs Backward Model Divergence on the Same Corpus

**Date**: 2026-05-04
**Source**: `rnd/cap-folding/` — measurement gathered while exploring
cap-fold mechanisms.

## TL;DR

Two AGPT models trained on the same Shakespeare 1M corpus — one on the
prefix radix-trie, one on the suffix radix-trie (reversed corpus) —
disagree heavily at held-out positions:

- **Symmetric KL = 2.38 nats** between their predictions
- **JS = 0.33 nats** (about half-saturated of the 0.69 max)
- **Top-1 agreement = 33.2%** — they pick the same most-likely char only a third of the time

This is despite `bayes_probe.cr` confirming that the underlying *trie
distributions* agree perfectly (KL = 0 between forward and suffix-tree
empirical distributions). The disagreement is at the model level: same
data, different learned models, very different per-position predictions.

## Setup

- Corpus: `data/input.txt` (Shakespeare 1M chars, V=65)
- Both tries: `--max-depth 32`, suffix trie via `--reverse`
- Same recipe for both trainings: `pd=6 RMSprop --no-accumulate
  lr=3e-3 weight-decay=0.01 mass-weight=off`, 6 SE, fresh from
  `data/input.random.model`
- Comparison: 4096 held-out positions evaluated by
  `bin/prefix_suffix_compare`

At each position p, forward model F predicts c at p given chars
`p-32..p-1`; backward model B predicts c at p given the *reverse* of
`p+1..p+32`. Both target the same c.

## Numbers

| metric | value |
|---|---:|
| KL(F ‖ B) | 2.37 nats |
| KL(B ‖ F) | 2.40 nats |
| Symmetric KL | 2.38 nats |
| JS divergence | 0.33 nats |
| Forward NLL on true target | 1.57 (PPL 4.80) |
| Backward NLL on true target | 1.63 (PPL 5.12) |
| Top-1 agreement | 33.2% |
| Top-1 disagreement | 66.8% |

## Why this matters

The corpus's underlying conditional distributions are mathematically
consistent in both directions (Bayes inversion holds; bayes_probe
verified KL=0 over the trie empirical statistics). So in principle, F
and B *should* agree on average over many positions.

The fact that they disagree by 2.4 nats per position quantifies the
**per-position information asymmetry** between prefix and suffix
evidence. Two interpretations, neither exclusive:

1. **Different inductive factorings.** F learns continuation patterns
   (what follows a prefix). B learns precedence patterns (what precedes
   a suffix). Same data, different views. The model parameters encode
   different aspects of the corpus and the disagreement at any single
   position is the gap between those views.

2. **Information-content asymmetry per position.** At any individual
   held-out position, the prefix and suffix carry structurally different
   information about c. "the ___ ran" — prefix "the" leaves the noun
   open, suffix "ran" narrows it differently. Both views average to the
   same corpus marginal over many positions, but at any single position
   they point in different directions.

Both views recover the corpus marginal in the limit. They diverge
strongly at finite scale because they're different lossy compressions
of the same underlying joint distribution.

## Implications

### For folding / dual-model architecture (`prefix-suffix-fold-architecture.md`)

The KL_suffix consistency loss term in §3 of the broader architecture
proposal is what would close this gap. The 2.4-nat per-position measurement
is exactly what that consistency loss has to fight against. Without
explicit coupling at training time, the two models will continue to
encode different views, and any fold/loop mechanism that crosses
prefix↔suffix information has to navigate this divergence.

### For cap-folding (target-substitution version, shipped)

The cap-fold we shipped *does not* bridge this gap — it stays entirely
within the prefix-tree view. The small PPL win (-2 to -3%) we measured
is independent of the prefix↔suffix divergence. The structural
opportunity that *does* address the divergence is dual-model training,
not target substitution within one direction.

### For any future "extend seq_len via fold-loop" idea

If looping crosses from a deep cap (prefix-side, end of identity tail)
into the corresponding shallow internal node (suffix-side equivalent),
the model has to interpret state from both sides. The 2.4-nat
disagreement is the magnitude of "things the two sides see
differently." A loop without bridging trains the model to ignore this
asymmetry; a loop with bridging (KL_suffix or similar) trains it to
align both views.

## Reproduce

```sh
just build-prefix-suffix-compare
just build-agpt-build-radix-corpus

# Suffix trie (one-time)
./bin/agpt_build_radix_corpus --corpus data/input.txt --max-depth 32 --reverse \
                              --out /tmp/agpt_input_d32_suffix_radix

# Backward model
./bin/agpt_train --model data/input.random.model \
                 --trie-dir /tmp/agpt_input_d32_suffix_radix \
                 --save /tmp/backward_6se.model \
                 --epochs 6 --partition-depth 6 --no-accumulate \
                 --optimizer rmsprop --lr 3e-3 --weight-decay 0.01 --mass-weight off

# Comparison (assumes /tmp/baseline_6se.model exists from cap-folding work)
./bin/prefix_suffix_compare \
  --forward /tmp/baseline_6se.model --backward /tmp/backward_6se.model \
  --file data/input.txt --seq-len 32 --max-positions 4096 --backend openblas
```

## Related

- `rnd/cap-folding/README.md` — the cap-folding work that motivated this
  measurement
- `notes/agpt/prefix-suffix-fold-architecture.md` — the broader proposed
  dual-model architecture
- `rnd/prefix-suffix-bayes/` — the underlying corpus-level math
  validation (KL=0 for trie distributions)
- `bin/prefix_suffix_compare` — the comparison tool itself
- `bin/bayes_probe` — the trie-distribution agreement probe (KL=0 reference)
