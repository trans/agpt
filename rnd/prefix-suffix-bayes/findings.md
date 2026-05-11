# Prefix-Suffix Bayesian Probe — First Iteration Findings

**Date:** 2026-05-02
**Tool:** `bin/bayes_probe`
**Tries:** Shakespeare 1M, d=32 forward + suffix radix tries

## Result: math is sound, raw trie statistics agree exactly

For every prefix tested, the forward-tree direct lookup and the
suffix-tree Bayesian inversion produce **bit-exactly identical**
distributions (KL = 0.00000000 nats).

| prefix | mass | distribution highlights | KL |
|---|---:|---|---:|
| `"the kin"` | 162 | g 96.91%, d 2.47%, s 0.62% | 0.0 |
| `"the"` | 10495 | r 19.2%, e 7.6%, i 4.4%, m 4.4%, n 4.4%, y 4.2%, ... | 0.0 |
| `"and"` | 4931 | space 80.5%, , 4.5%, s 4.2%, '\n' 2.7%, e 1.6%, ... | 0.0 |
| `"to be"` | 283 | a 9.2%, d 6.4%, g 4.2%, ... | 0.0 |
| `"ROMEO:"` | 163 | '\n' 100% | 0.0 |
| `"Hamlet"` | — | not in corpus | — |

## What this validates

1. **Trie construction is symmetric.** The suffix trie is a true mirror
   of the forward trie — same data, opposite indexing.
2. **Bayesian inversion math is correct.** The decomposition
   `P_s(t|p) ∝ P_s(p|t)·P_s(t)` is mathematically equivalent to direct
   joint-mass lookup, both giving the same empirical distribution.
3. **The architecture's mathematical foundation is sound.** When
   training a forward model and a backward model jointly, the
   constraint `P_forward ≈ P_backward` is well-defined — both
   distributions are unbiased estimators of the same conditional.

## What this *doesn't* test

This probe uses raw trie counts. By the equivalence, they trivially
agree. The interesting question — does the **trained-model
regularizer** `KL(P_backward(suffix) || P_forward(prefix))` carry
useful training signal? — requires actual trained models, which can
extrapolate differently even when their empirical training data agrees.

The trained-model comparison is a separate experiment, requiring:
1. A backward model trained on the suffix radix-trie (~10 min wall
   under current AGPT recipe at pd=6 3 SE)
2. A dual-model loss probe tool (~80 LOC) that runs both models on a
   sample of held-out positions and reports CE + KL_suffix per position

## Next steps

- **Phase 1 (this run):** ✓ math validated.
- **Phase 2:** train backward model, run trained-model probe.
- **Phase 3:** add the `KL(P_backward || P_forward)` term to AGPT
  training and measure whether PPL improves vs forward-only baseline.
- **Phase 4:** add `KL(P_fold || P_model)` once fold map is built.
