# AGPT Scaling vs Kneser-Ney Baseline (Proper Held-out)

**Date:** 2026-05-23 (evening)
**Status:** Closed. AGPT at moderate scale (L=6, 200 SE) edges out classical KN order 6 on properly disjoint held-out. Modest win, monotone trajectory, well-conditioned for further scaling.

## Question

Earlier 2026-05-23 we discovered our existing held-out files (`/tmp/shake_holdout.txt`, `/tmp/gut_holdout.txt`) were strict subsets of training. All "held-out PPL" we'd reported across the project was actually training-set tail PPL. Two problems followed:

1. **KN comparison was artifactual.** KN at high order was memorizing the held-out (every 8-gram in heldout was seen during count-collection), producing impossibly-low PPL (1.78 at order 8). Made AGPT look 4× worse than it actually was.
2. **We hadn't measured generalization at all.** Just training-set fit.

Built `scripts/build_proper_heldout.py` to produce `/tmp/gut_holdout_proper.txt` — the last 200K chars of preprocessed war_peace, which the `gutenberg_5m.txt`-construction truncation excluded from training. Provably disjoint.

This experiment re-anchored the KN vs AGPT comparison on the proper held-out and tested whether AGPT can beat KN at moderate scale.

## Setup

- Training corpus: `data/gutenberg_5m.txt` (5M chars, 65-char vocab)
- Trie: `/tmp/gutenberg_5m_baseline_d16_radix` (d=16, built from full training corpus)
- Held-out: `/tmp/gut_holdout_proper.txt` (war_peace tail, 200K chars, disjoint by construction)
- Recipe: `rmsprop lr=3e-3 warmup-cosine warmup-epochs=5 --partition-depth 1 --no-accumulate --mass-weight off --anc-grad`
- Eval: `bin/agpt_sliding_window_perplexity --pool deep_only --d 16 --max-positions 10000`
- Hardware: laptop RTX 4070 for L=2 (small); RunPod A100-80GB for L=4/L=6

## Results

**KN baselines** (nltk's KneserNeyInterpolated, char-level, trained on `data/gutenberg_5m.txt`, evaluated on proper held-out):

| KN order | context (chars) | PPL |
|---|---|---|
| 3 | 2 | 7.21 |
| 4 | 3 | 4.78 |
| 6 | 5 | **3.96** ⭐ (KN winner) |

(Order 8 not run on Gutenberg — extrapolated to be ~3-3.5 PPL, but takes >1h with nltk's pure-Python KN at 5M training chars.)

**AGPT at varying scale** (n=3 seeds each, proper held-out):

| config | wall (A100) | per-seed PPL | mean ± std |
|---|---|---|---|
| L=2, 10 SE, depth+anc | ~5 min | 7.07-7.29 | 7.29 ± 0.09 |
| L=4, 100 SE, depth+anc | ~26 min | 3.96, 4.04, 3.98 | **3.99 ± 0.04** |
| **L=6, 200 SE, depth+anc** | **~74 min** | **3.76, 3.76, 3.73** | **3.7544 ± 0.014** ⭐ |

**AGPT at L=6 200 SE beats KN order 6 by 0.21 PPL on proper held-out.** Variance is tiny (std 0.014); the win is robust across seeds.

## Loss curve (training, not held-out)

Per the per-seed training logs, loss converges monotonically:

| epoch | seed 1 loss | seed 2 loss | seed 3 loss |
|---|---|---|---|
| 1 | 3.28 | 3.17 | 3.26 |
| 50 | 1.494 | 1.498 | 1.484 |
| 100 | 1.380 | 1.396 | 1.378 |
| 150 | 1.344 | 1.345 | 1.337 |
| 200 | 1.337 | 1.336 | 1.328 |

Most learning happens in the first 50 SE (loss drops from ~3.2 to ~1.5). Diminishing returns: SE 50→100 drops loss 0.11; 100→150 drops 0.04; 150→200 drops 0.007. We're at the soft training-loss ceiling for this architecture.

## Interpretation

1. **AGPT is no longer behind KN at moderate scale.** The earlier "AGPT crushed by KN" framing was a methodology artifact (training-tail eval). At proper-held-out, AGPT at L=6 200 SE edges KN out — modestly but unambiguously.
2. **The improvement is from scale, not from any architectural intervention.** Same recipe at L=2→L=4→L=6, doubling-then-doubling layers and increasing SE 10x→100x→200x, gave PPL 7.29 → 3.99 → 3.75.
3. **Diminishing returns are kicking in.** From L=4 100 SE (3.99) to L=6 200 SE (3.75) we used roughly 3× more compute for 0.24 PPL gain. The next L=8 / 400 SE step would likely yield another 0.15 PPL at best. We're approaching the architecture's soft ceiling at this (d_model=64, d=16) configuration.
4. **The "3.3-ish" historical best** is still 0.45 PPL below where we landed today. Required different architectural choices (larger d_model probably, possibly larger d). We hit a wall along the "just scale L and SE" axis.

## Why this isn't a great result, despite beating KN

KN order 6 uses only 5 characters of context. AGPT uses 15 characters (d=16). AGPT *should* beat KN by a wide margin given the 3× longer context — instead we beat it by 0.21 PPL. This suggests AGPT isn't extracting full value from its longer attention window. KN at order 8 (7-char context) would likely beat us; we ran out of time/compute to confirm.

The narrow win is consistent with the long-running architecture story: **the trie taps out around d=10-20 because most leaves go mass=1 past that point**, and the model can't usefully learn from one-hot targets at unreachable mass-1 leaves. To get meaningful context-extension beyond ~d=20 needs a mechanism other than just deeper tries — per-fire corpus-position, suffix-side flow, or some other axis we haven't tried.

## What this changes about the recipe

Nothing structural. The canonical recipe (`--anc-grad`, depth-RoPE, mass-weight off) is unchanged. The new data points are:

- L=6 200 SE on Gutenberg is reproducible at ~3.75 PPL with std 0.014 (any future "is this run normal?" check has this number)
- Scale-to-saturation point for the L stacking axis is somewhere between L=6 and L=8 at d=16/d_model=64

## Files

- `l6_200se_training_curve.txt` — per-seed loss snapshots at epochs 1/50/100/150/200
- `logs/L4_100SE_seed{1,2,3}.train.log` — full L=4 training logs
- `logs/L6_200SE_seed{1,2,3}.train.log` — full L=6 training logs

## Reproduce

```sh
# Proper held-out (one-time):
python3 scripts/build_proper_heldout.py /tmp/gut_holdout_proper.txt 200000

# L=6 200 SE 3 seeds on Gutenberg:
for s in 1 2 3; do
    bin/agpt_train --init --init-seed $s --init-n-layers 6 --epochs 0 \
        --trie-dir /tmp/gutenberg_5m_baseline_d16_radix \
        --save /tmp/init_s${s}.model
    bin/agpt_train --model /tmp/init_s${s}.model \
        --trie-dir /tmp/gutenberg_5m_baseline_d16_radix \
        --epochs 200 --lr 3e-3 --optimizer rmsprop \
        --lr-schedule warmup-cosine --warmup-epochs 5 \
        --partition-depth 1 --no-accumulate --mass-weight off \
        --anc-grad \
        --save /tmp/trained_s${s}.model
    bin/agpt_sliding_window_perplexity --model /tmp/trained_s${s}.model \
        --file /tmp/gut_holdout_proper.txt --vocab-file data/input.txt \
        --d 16 --max-positions 10000 --backend openblas --workers 8 \
        --pool deep_only
done

# KN order 6 baseline:
python3 src/tools/agpt_kn_baseline.py \
    --train data/gutenberg_5m.txt --heldout /tmp/gut_holdout_proper.txt \
    --orders 6 --max-positions 10000
```

## Open follow-ups

- KN order 8 on Gutenberg with proper held-out (need faster KN — KenLM or our own implementation)
- L=8 200 SE to see if the scaling curve has bottomed out
- d_model sweep at L=6 (the "3.3-ish" historical best probably needed bigger d_model)
- The real research arc remains *context extension beyond what the trie naturally supports* — per-fire corpus-position is the next probe
