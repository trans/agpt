# TODO

## AGPT

- Replace the current `--agpt-max-starts` prefix-of-file safety cap with a full-corpus strategy.
- Candidate directions:
  - distribute capped starts across the whole corpus instead of taking only the first `N`
  - support randomized or epoch-rotated start subsets
  - remove the need for the cap with a more compact prefix/suffix index

## Wrap-around synth

- Space-aligned wrapping in `bin/synth_wrap_corpus`: prefer bridging at
  a word boundary instead of mid-word, to eliminate artifacts like
  "bishhanged". Sketch: when sampling a bridge token from the leaf's
  endpoint counts, prefer ' ' if it is in the top-k options; otherwise
  walk one more edge before retrying. Expected: small PPL improvement
  on top of the seq=128 / 10k / 7.17 baseline.
- Apply the same wrap-around effect inside AGPT itself (subtree-aware,
  not just sequence generation). Open question: how to compose
  leaf→root continuation with subtree-aggregated gradients.
