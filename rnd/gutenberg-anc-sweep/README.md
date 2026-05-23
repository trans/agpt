# Gutenberg Weighting Sweep at the anc-grad Baseline

**Date:** 2026-05-23
**Status:** Closed. **No recipe-level wins.** Best cell is borderline; the entire weighting design space looks like corpus-specific tuning, not architectural improvement.

## Question

Following 2026-05-22's headline finding that `--anc-grad` is a substantial cross-corpus win, we wanted to test which (if any) of the per-event weighting flags still adds value on top of anc-grad — specifically, whether the Shakespeare-validated `--branching-weight log` finding (−0.58 PPL, p<0.05 n=3) generalizes to Gutenberg, and what other knobs become net-positive once anc-grad is doing the gradient-routing heavy lifting.

The historical pattern was: `--mass-weight log` was useful at d=8/16 with no anc-grad (compensation for starved K/V gradient at shallow positions); turned net-negative once anc-grad routes those gradients properly. We expected some other signal might be analogously revealed at the new baseline.

## Setup

- Corpus: Gutenberg 5M, d=16 radix at `/tmp/gutenberg_5m_baseline_d16_radix`
- Init: shared Kaiming (`/tmp/agpt_init_kaiming_s{1,2,3}.model`); same as Shakespeare sweep
- Recipe per cell: rmsprop, lr=3e-3, warmup-cosine 1 epoch, `--partition-depth 1`, `--no-accumulate`, `--mass-weight off`, **`--anc-grad`**
- 10 super-epochs per cell
- Eval: Crystal `agpt_sliding_window_perplexity --pool deep_only` (byte-identical to Python `agpt_ppl --mode fixed`)
- 4 signals × 5 modes × 3 seeds = 60 cells + 3-seed baseline = 63 cells

| signal | --flag |
|---|---|
| mass | `--mass-weight {log,sqrt,linear,inv-log,inv-linear}` |
| entropy-w | `--entropy-weight {log,sqrt,linear,inv-log,inv-linear}` |
| branching | `--branching-weight {log,sqrt,linear,inv-log,inv-linear}` |
| depth-weight | `--depth-weight {log,sqrt,linear,inv-log,inv-linear}` |

Ran on RunPod A100-SXM4-80GB. ~95s/cell, ~100 min total wall.

## Results

Baseline (depth+anc, mw=off): **7.629 ± 0.086** (n=3)

| signal\mode | log | sqrt | linear | inv-log | inv-linear |
|---|---|---|---|---|---|
| mass | 7.560 (−0.07) | 7.485 (−0.14) | 7.928 (+0.30) | 7.805 (+0.18) | 7.702 (+0.07) |
| entropy-w | 7.790 (+0.16) | 7.760 (+0.13) | 7.651 (+0.02) | 7.619 (−0.01) | 7.683 (+0.05) |
| branching | 7.655 (+0.03) | 7.761 (+0.13) | 7.634 (+0.01) | 7.673 (+0.04) | 7.818 (+0.19) |
| **depth-weight** | 7.590 (−0.04) | **7.445 (−0.18)** | **7.422 (−0.21) ⭐** | 7.927 (+0.30) | 8.682 (+1.05) |

(Δ relative to baseline; n=3 per cell; std per cell typically 0.05-0.20)

Top wins:
- depth-weight linear: −0.21 PPL (p<0.05 paired-t, but the absolute effect is ~3%)
- depth-weight sqrt: −0.18 PPL
- mass sqrt: −0.14 PPL

## Cross-corpus comparison

| signal | Shakespeare 1M (n=3) | Gutenberg 5M (n=3) | survives? |
|---|---|---|---|
| branching=log | **−0.58 ⭐ (p<0.05)** | +0.03 (n.s.) | **NO** |
| depth-weight=log | −0.24 | −0.04 | partial |
| depth-weight=linear | (untested at Shakespeare) | −0.21 | new finding |
| mass=log | (historical: useful pre-anc-grad) | −0.07 | small at best |
| entropy=1.0 (icing) | +0.13 (n.s.) | +0.01 | null both |

The Shakespeare branching=log win collapsed completely on Gutenberg. **This is the same pattern as the historical mass-weighting story** — a corpus-specific gradient bias that doesn't survive cross-corpus.

## Interpretation

**No finding here rises above "tuning to corpus quirks".** The best win is −0.21 PPL = ~3% improvement, which is in the same range as the seed-to-seed noise floor. Across two corpora, this isn't a generalizable architectural signal; it's tuning the gradient-amplitude profile to a specific data distribution.

The pattern across today's sweep:

1. **`inv-log` and `inv-linear` modes are consistently bad.** Worst cell of the entire matrix is `depth-weight inv-linear` at +1.05 PPL — actively breaks the model. These modes up-weight low-mass / deep / rare nodes, which (per the 2026-05-22 RoPE leaf-end finding) is *not* where extra gradient signal should go. Useful "don't do this" observation; not actionable as a recipe addition.

2. **`linear` and `sqrt` modes consistently outperform `log` for depth-weight.** Log compresses too much; the model can use the stronger differentiation that linear/sqrt give. Limited to depth-weight though; for other signals the mode choice is mostly noise.

3. **Branching is the most corpus-specific signal.** All 5 modes are within ±0.2 of baseline on Gutenberg (no signal); but branching=log was a 0.58-PPL win on Shakespeare. Whatever branching captures, Gutenberg doesn't have it in the same way Shakespeare does.

4. **anc-grad remains the only architectural win that cross-generalizes.** Weighting heuristics tune the gradient *amplitude profile*; anc-grad changes the gradient *flow topology*. Different category of intervention, different generalization properties.

## Conclusion

**No update to the canonical recipe from this sweep.** `depth-weight=linear` (or sqrt) is the closest candidate, but the effect size is too small relative to seed/corpus variance to canonize.

The takeaway is methodological as much as empirical: **per-event weighting flags are mostly corpus-specific tuning, not architecture**. We shouldn't expect them to generalize unless we find one that does — and so far we haven't.

The next real win is likely to come from architectural changes (something in the class of anc-grad), not from a better weighting heuristic.

## Files

- `run.sh` — the sweep script (env-var-driven; supports focused sub-sweeps)
- `results.txt` — per-cell PPL with train wall time
- `summary.txt` — aggregated mean ± std per signal/mode

## Open follow-ups

- Eventually re-run depth-weight linear/sqrt at L=4 to see if the small effect holds at larger model
- d=32 trie + this sweep would be more discriminating, but expensive (d=32 trie at Gutenberg is 1+ GB to build)
- The "inv-log / inv-linear breaks model" finding is interesting on its own — could be the basis for a "what NOT to do" note in the trainer's CLI help
