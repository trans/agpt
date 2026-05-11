# Prefix–Suffix Bayesian Consistency

**Status:** investigation in progress (2026-05-02).

## Hypothesis

The forward radix-trie (built from corpus) and the suffix radix-trie
(built from `reverse(corpus)`) both encode the same empirical
distributions, viewed from opposite directions. By Bayes' rule, the
"next-token-given-prefix" distribution is recoverable from either tree.

This investigation tests:
1. Whether raw trie statistics agree exactly (they should, by counting)
2. Where the magnitudes of `KL(P_forward || P_suffix)` land for
   real prefixes — sets a baseline scale for the `KL(P_s || P_p)`
   regularizer term in the unified loss.
3. Whether trained-model predictions diverge from raw trie statistics
   in a way that would make the regularizer informative.

## Design context

This connects to the larger architecture vision developed 2026-05-02:

```
L = CE(y, P_model)                       # data term
  + α · KL(P_fold(p)    || P_model(p))   # cap-fold consistency
  + β · KL(P_suffix(p)  || P_model(p))   # Bayesian self-consistency
```

For the **first iteration probe** we compute the simplest sub-question:

> Pick a single prefix `p`. Compute `P(t | p)` from the forward tree
> directly. Compute `P(t | p)` via Bayesian inversion on the suffix
> tree. Compare.

If the two distributions are equal (they should be — both are unbiased
estimators of the same conditional, computed from the same corpus
counts), then the inversion math is correct and we can move on to
training a backward model and computing the full loss.

If they differ, we either have a math bug or an edge-case in trie
construction (corpus boundaries, depth limits, etc.).

## Math

Forward tree at path `p` gives us the empirical distribution directly
from edge-mass ratios:

```
P_p(t | p) = mass_forward(p ++ [t]) / mass_forward(p)
```

Suffix tree at path `reverse(p)` ++ [t] gives us a joint count. By
the trie-construction equivalence,

```
mass_suffix([t] ++ reverse(p))    counts in suffix tree of substring
                                  [t][reverse(p)] in reverse(corpus)

  = # corpus positions where original[i..i+|p|] = p ++ [t]
  = mass_forward(p ++ [t])
```

So the suffix-tree-derived `P_s(t | p)` equals the forward-derived one
exactly when both trees have full coverage. Where they differ:

- If `|p|+1 > forward_tree_max_depth` but suffix tree still has
  capacity → forward tree is truncated, suffix wins
- Vice versa for opposite-side truncation
- Corpus boundary positions can introduce small asymmetries

## Files

- `bayes_probe.cr` — the analysis tool (Crystal)
- `bayes_probe` — built binary
- `findings.md` — results from running on Shakespeare 1M

## Reproduction

```sh
crystal build src/tools/bayes_probe.cr -o bin/bayes_probe --release
bin/bayes_probe \
  --forward /home/trans/agpt-tries/shakespeare_d32_radix_corpus \
  --suffix  /home/trans/agpt-tries/shakespeare_d32_suffix_radix \
  --corpus  data/input.txt \
  --prefix "the kin"
```
