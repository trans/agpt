# Suffix-Depth Blending at Radix Endpoints

**Status**: complete — helps at d=16, no effect at d=8, hurts at d=32.

**Trainer note**: pre-fix, needs reassessment.

The idea and implementation remain useful, but the reported gains were measured
on the pre-fix AGPT trainer. Re-check before treating the d=16 improvement as a
current project conclusion.

**Code**: branch `agpt-root-loop` (shared with root-loop experiment).
Key commits: 178e28d (initial), 6ebac06 (count-aware), ce4cc3e (endpoint-only).

## Hypothesis

At each branching point in the radix trie, the target distribution can be
softened by blending in distributions from shorter suffixes of the same
path. If deep-path counts are sparse (high-variance estimate), shallow
well-estimated distributions act as a smoothing prior.

## What we built

- CLI flag `--blend-alpha F` on `bin/agpt_train`. F > 0 enables count-aware
  blending; F = 0 is plain AGPT.
- At each endpoint query (last character of a radix edge), replace the
  target distribution with:

      target[v] = Σ_k λ_k · P_k(v)

  where `P_k` is the D-trie count distribution at the k-token suffix of
  this query's path (k = 1..d), and `λ_k ∝ log(1 + count_k)` normalized.
- Intermediate queries (inside a radix edge) keep their deterministic
  singleton targets unchanged.
- `state_index[D-tuple → radix_id]` built once per training run over
  endpoint paths only (intra-edge extension was an anti-pattern — reverted).

## Results (Shakespeare per-subtree, 3 super-epochs, rmsprop+warmup-cosine+entropy)

| D | Baseline mean (n=6 at d=16, n=3 elsewhere) | Blending mean | Signal |
|---|---|---|---|
| d=8  | 15.13 | 15.19 | ~tied (no help) |
| d=16 | 13.66 | **13.34** | +0.32 PPL (best overall mean) |
| d=32 | **13.41** | 13.47 | slight hurt |

Best single run: **PPL 12.94** at d=16 + blending.

## Interpretation

Blending has a "Goldilocks" depth range:
- d=8: deep counts already dense (thousands of observations per node). Smoothing
  redundant.
- d=16: deep counts moderately sparse with real branching. Smoothing reduces
  target-side noise.
- d=32: deep counts near-singleton (most 32-grams unique in Shakespeare).
  Smoothing introduces noise around a correct deterministic target.

## Files

- `run.sh` — reproduces the n=6 d=16 comparison.
- `results.md` — full variance data.
