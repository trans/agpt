# d_model and Depth Scaling at 100 SE — Gutenberg

**Date:** 2026-05-24
**Status:** Closed. **d_model=128 is the sweet spot**; doubling to 256 regresses PPL. Doubling **L** at d=128 (L=4 → L=8) drops PPL by 1.5% to **3.6899 ± 0.016** — new project best on Gutenberg with proper held-out.

## Question

The L=6 d=64 200 SE result (3.7544 ± 0.014) seemed to be hitting a ceiling along the "stack more layers" axis. Does widening d_model push past it at L=4, holding training budget at 100 SE?

## Setup

- Training corpus: `data/gutenberg_5m.txt` (5M chars, 65-char vocab)
- Trie: `/tmp/gutenberg_5m_baseline_d16_radix` (d=16)
- Held-out: `/tmp/gut_holdout_proper.txt` (war_peace tail, 200K chars, disjoint by construction)
- Recipe: `rmsprop lr=3e-3 warmup-cosine warmup-epochs=5 --partition-depth 1 --no-accumulate --mass-weight off --anc-grad`
- All configs: L=4, n_heads scaled to keep head_dim=16, d_ff=4×d_model, 100 SE
- Hardware: RunPod A100-80GB
- n=3 seeds each config

## Results

| config | params (approx) | wall/seed | per-seed PPL | mean ± std |
|---|---|---|---|---|
| L=4, d_model=64, ff=256, h=4 | ~280K | 1551s | 3.96, 4.04, 3.98 | **3.99 ± 0.04** |
| L=4, d_model=128, ff=512, h=8 | ~870K | 3204s | 3.73, 3.75, 3.76 | 3.7450 ± 0.012 |
| L=4, d_model=256, ff=1024, h=16 | ~3.0M | 5451s | 3.74, 3.78, 3.82 | 3.7802 ± 0.032 |
| L=8, d_model=128, ff=512, h=8 | ~1.7M | 6271s | 3.69, 3.67, 3.71 | 3.6899 ± 0.016 |
| L=12, d_model=128, ff=512, h=8 | ~2.5M | 9357s | 3.70, 3.79, 3.73 | 3.7383 ± 0.040 |
| **L=8, d_model=128, 200 SE** | **~1.7M** | **12527s** | **3.63, 3.65, 3.60** | **3.6274 ± 0.018** ⭐ |

**Reference:** L=6 d=64 200 SE landed at 3.7544 ± 0.014, ~75 min/seed.

## Headline

**Width and depth (in layers) are capped at L=8, d=128 — but epochs still pay.** L=4 d=128 → L=4 d=256 regressed PPL (width dead). L=4 → L=8 at d=128 gave +0.055 PPL (depth alive). L=8 → L=12 at d=128 went backwards (depth past L=8 hurts at 100 SE). But **L=8 d=128 100 SE → 200 SE dropped another 0.063 PPL** (3.6899 → 3.6274), tight variance. The architectural ceiling for the current recipe is **L=8 d=128 200 SE = 3.6274 ± 0.018**, not the 100 SE number. 400 SE would likely continue the trend; haven't run yet.

## Interpretation

The d=256 regression at fixed 100 SE is the key signal. Three competing hypotheses:

1. **Under-trained capacity.** d=256 has ~3.5× the params of d=128 but the same number of training events. The optimizer hasn't moved the weights far enough for the extra capacity to pay off. **Likely correct** — loss at d=256 epoch 100 was 1.382-1.474 across seeds, vs d=128 final loss ~1.33-1.36 (small but consistent gap on training loss too).
2. **Optimizer mis-tuned.** Same lr=3e-3 across all d_models. Bigger d_model may want a smaller lr; RMSprop's adaptive scaling doesn't fully compensate for the wider matmuls.
3. **AGPT-specific quirk.** Per-fire updates are mass-weighted differently when d_model changes effective per-position gradient magnitude through the attention softmax. Plausible but speculative.

Variance widening (0.032 vs 0.012) supports under-training: incomplete convergence amplifies seed-init differences.

## What this means for next steps

- **Don't naively go bigger.** L=8 / d_model=512 with 100 SE will almost certainly regress further.
- **If we want to test d=256 properly: 200 SE.** Match the "events per parameter" budget d=128 100 SE had.
- **For headline runs at fixed budget: d_model=128.** Current best at ~50 min/seed.
- The "trie taps out at d=10-20" problem is now the active constraint. Bigger d_model can't extract more value because the gradient signal is bounded by what the d=16 trie carries. To break this ceiling we need context extension beyond what the trie naturally supports — the per-node position distribution work in `notes/seq-len-extension/per-node-position-distributions.md`.

## Files

- `summary.txt` — Tabular results
- `train_logs/` — Per-seed train.log files (pulled from pod)

## Reproduce

```sh
for dmodel in 64 128 256; do
    case $dmodel in
        64)  heads=4;  ff=256  ;;
        128) heads=8;  ff=512  ;;
        256) heads=16; ff=1024 ;;
    esac
    for s in 1 2 3; do
        bin/agpt_train --init --init-seed $s \
            --init-d-model $dmodel --init-n-heads $heads --init-n-layers 4 --init-d-ff $ff \
            --epochs 0 --trie-dir /tmp/gutenberg_5m_baseline_d16_radix \
            --save /tmp/init_d${dmodel}_s${s}.model
        bin/agpt_train --model /tmp/init_d${dmodel}_s${s}.model \
            --trie-dir /tmp/gutenberg_5m_baseline_d16_radix \
            --epochs 100 --lr 3e-3 --optimizer rmsprop \
            --lr-schedule warmup-cosine --warmup-epochs 5 \
            --partition-depth 1 --no-accumulate --mass-weight off --anc-grad \
            --save /tmp/trained_d${dmodel}_s${s}.model
        bin/agpt_sliding_window_perplexity --model /tmp/trained_d${dmodel}_s${s}.model \
            --file /tmp/gut_holdout_proper.txt --vocab-file data/input.txt \
            --d 16 --max-positions 10000 --backend openblas --workers 8 --pool deep_only
    done
done
```

## Open follow-ups

- L=4 d=256 200 SE (or 400) — does extra training redeem d=256, or has the trie capped what any width can deliver?
- L=4 d=192 100 SE — interpolate between 128 and 256 to confirm 128 is the actual peak, not a hump.
- L=8 d=128 200 SE — does more compute at the new best architecture push lower still? (open; L=12 100SE regressed, but 200SE at L=8 might still help)
- **Architectural changes are now the real path forward.** Pure scaling along L, d_model, or SE at this trie/recipe is exhausted. See `notes/seq-len-extension/position-distributions-plan.md`.

## Closed via this experiment

- L=12 d=128 100 SE: regressed vs L=8 (3.7383 ± 0.040 vs 3.6899 ± 0.016) with 2.5× wider variance. Depth past L=8 hurts at this epoch budget.
