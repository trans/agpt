# anc-grad — corrected normalizer (no knob)

## What

`--anc-grad` enables descendant→ancestor gradient flow for Wk/Wv. Without
it, the trainer drops the ancestor-slice of dK/dV from the packed attention
backward, so K[ancestor_position] only ever receives gradient from queries
that attend to it within its **own** chunk (via the own-edge path) — the
contributions from later descendant queries (in subsequent chunks of the
same subtree) are discarded.

With `--anc-grad`, those ancestor-slice gradients are scatter-added into a
per-(compact-char-position) accumulator, then chain-ruled via cuBLAS at the
fire-end against the saved ln1_out of each ancestor position. The
descendants→ancestor pathway through Wk/Wv is restored.

## The bug that was hiding inside the prior version

Earlier `--anc-grad` builds shipped with an `--anc-grad-scale F` knob.
Sweeping F on Gutenberg produced a non-monotonic curve (best at 0.5, also
good at 2.0, bad at 1.0 and 0.0) — a "data smell."

Diagnosis: the prior fire-end gemm used `1 / chunks_processed` as its
scalar, while own-edge uses per-chunk `1 / T_q_chunk`. Same chunk
contributes to dW_kw via two paths with **incompatible per-event
weights**. The "scale" knob was a global multiplier trying to compensate
for the mismatch, but no single value works because the ratio
`T_q_chunk / chunks_processed` varies per subtree.

## The fix

Pre-scale at scatter time by `grad_scale = 1/T_q_chunk` — the **same**
per-event weight own-edge uses. Each ancestor event is added to the
accumulator with the weight its descendant query would have given its own
events. Fire-end GEMM uses scalar 1.0 — no further normalization.

Result: the `--anc-grad-scale` knob is gone. The flag is now binary: off =
own-edge only, on = own-edge + descendant scatter, both consistently
per-event-weighted by `1/T_q_chunk`.

(Sidenote: own-edge itself uses per-chunk `1/T_q` rather than per-fire
`1/subtree_events`. That bakes chunk-as-memory-artifact into gradient
math. Worth a follow-up cleanup; left alone here to keep this fix
surgical.)

## Validation — Shakespeare 1M, d_model=64, 10 SE, n=3 seeds

Recipe: `--lr 3e-3 --optimizer rmsprop --lr-schedule warmup-cosine --warmup-epochs 1 --partition-depth 1 --mass-weight off --no-accumulate`.

Trie: `/tmp/shake_baseline_d16_radix`. Models: `/tmp/seed{1,2,3}.model`.

**Training-set (exp of epoch-10 mean loss):**

| seed | off PPL | on PPL | %Δ |
|---|---|---|---|
| 1 | 7.93 | 7.25 | -8.5% |
| 2 | 7.90 | 7.23 | -8.5% |
| 3 | 7.55 | 7.40 | -2.1% |
| **mean** | **7.79** | **7.29** | **-6.4%** |

**Held-out PPL** (sliding-window, d=16, 10k positions, last 50k chars of `data/input.txt`):

| seed | off | on | %Δ |
|---|---|---|---|
| 1 | 9.07 | 8.31 | -8.4% |
| 2 | 8.46 | 8.08 | -4.5% |
| 3 | 8.43 | 8.21 | -2.6% |
| **mean** | **8.65** | **8.20** | **-5.2%** |

3/3 seeds favor anc-grad on. Paired Δ has t ≈ -2.8 (n=3, two-sided p ≈ 0.11
— marginal at n=3 but directionally unanimous, and effect size is large
relative to seed noise).

## Validation — Gutenberg 5M, n=3 seeds (same recipe)

Trie: `/tmp/gutenberg_5m_baseline_d16_radix`. Holdout: last 200K chars of `data/gutenberg_5m.txt`.

**Training-set:**

| seed | off PPL | on PPL | %Δ |
|---|---|---|---|
| 1 | 8.33 | 8.15 | -2.1% |
| 2 | 8.11 | 7.89 | -2.7% |
| 3 | 8.33 | 7.82 | -6.1% |
| **mean** | **8.25** | **7.95** | **-3.6%** |

**Held-out PPL** (10k positions):

| seed | off | on | %Δ |
|---|---|---|---|
| 1 | 10.03 | 9.87 | -1.6% |
| 2 | 9.31 | 9.33 | +0.2% |
| 3 | 9.80 | 9.31 | -5.0% |
| **mean** | **9.71** | **9.50** | **-2.2%** |

Direction holds. Smaller magnitude than Shakespeare and seed 2 is a
tie, but no regression — anc-grad is corpus-portable.

## Comparison to prior buggy-normalizer result

Prior runs used divisor `1/chunks_processed` at fire-end, paired with the
`--anc-grad-scale` knob to compensate. Result there:

| corpus | n | direction | magnitude |
|---|---|---|---|
| Shakespeare | 6 | suggestive win | -1.6% |
| Gutenberg | 3 | suggestive loss | +28% (paired Δ noisy) |

After the fix:

| corpus | n | direction | magnitude |
|---|---|---|---|
| Shakespeare | 3 | 3/3 win | -5.2% held-out |
| Gutenberg | 3 | 2/3 + tie | -2.2% held-out |

The bug created an artificial "corpus dependence" that was actually
inherent to the broken normalizer interacting with Gutenberg's chunk-size
distribution. With the principled per-event normalization, both corpora
benefit from anc-grad.

The corrected normalizer ~3× amplifies the Shakespeare effect — consistent
with the prior implementation applying anc-grad gradient at ~5× too low a
magnitude relative to own-edge.

## Files touched

- `src/cuda/agpt_train.cu`
  - `scatter_anc_dkv_to_subtree_kernel`: added `float grad_scale` arg,
    multiply during atomicAdd
  - Per-fire init: moved d_dkv_subtree_{k,v} + h_subtree zero inside the
    splits loop (each split is a separate Adam fire)
  - Fire-end gemm: scalar = 1.0 (no chunks_processed, no scale knob)
  - Removed `cfg.anc_grad_scale`, `--anc-grad-scale` CLI flag
