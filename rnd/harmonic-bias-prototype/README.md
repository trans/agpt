# harmonic-bias-prototype

**Status:** closed (inconclusive single-seed result)

## Hypothesis

The asym-DFT harmonic-filter bias term (from
`notes/seq-len-extension/harmonic-filter-asymmetric.md`) added to attention
scores improves byte_perplexity on a small char-LM trained on Shakespeare.
Specifically: `score(Q, K) += β_h · (1/C_K) · Σ_j Re[z_K[j] · exp(-i p_Q ω_j)]`
where `z_K` is the DFT chord of K's position-mod-W distribution and β_h is
a learnable per-head scalar.

The diagnostic in `rnd/harmonic-filter-diagnostic/` showed the operator
separates on/off-path pairs by 1.6 IQR-units at mass=2-9 on both
Shakespeare and Gutenberg. This run tests whether that separation
translates into measurable PPL improvement during actual training.

## Scope

Prototype only. Uses `src/tools/agpt_pytorch_train.py` (NOT the canonical
Crystal/CUDA trainer). Training algorithm differs from AGPT — dense
sliding-window batched cross-entropy rather than trie-based subtree
fires. Result speaks to "does adding this bias to a vanilla causal-LM
help?", not directly to "does it help AGPT's training cadence?".

Existed because the CUDA-kernel version of the bias is multi-day work
(chord ingestion into trainer, per-query position passing, per-K
substring lookup, fused-softmax bias forward/backward, β as a saved
model parameter). This prototype lets us A/B before committing to that.

## Setup

- Model: `data/input.model` (d_model=64, n_heads=4, n_layers=2, d_ff=256, vocab=65)
- Corpus: Shakespeare (`data/input.txt`, 1.115M chars)
- Train slice: first 95% (1,059,624 chars); heldout: last 5% (55,770)
- d_window=16 (matches trie depth)
- Trainer: `src/tools/agpt_pytorch_train.py` (PyTorch sliding-window)
- Optimizer: RMSprop lr=3e-3, warmup-cosine (0.5 warmup-epochs)
- Epochs: 10
- Batch size: 128
- Seed: 42 (single seed)
- Eval: PyTorch-direct `byte_perplexity_pytorch` on the heldout 55,770 tokens,
  rolling fixed-window context d=16. β is exercised at eval (lm-eval
  can't, since .model has no slot for β).

Run B adds: harmonic bias enabled, W=64, n_freq=16, β init=0, per-(layer,head).

## Reproduce

```sh
# Run A — baseline
python3 src/tools/agpt_pytorch_train.py \
    --model data/input.model --corpus data/input.txt \
    --save /tmp/agpt_pytorch_ab_baseline.model \
    --epochs 10 --batch-size 128 --growth-max-depth 16 \
    --warmup-epochs 0.5 --optimizer rmsprop --lr 3e-3 \
    --device cuda --seed 42 --quiet --eval-heldout-frac 0.05

# Run B — bias
python3 src/tools/agpt_pytorch_train.py \
    --model data/input.model --corpus data/input.txt \
    --save /tmp/agpt_pytorch_ab_bias.model \
    --epochs 10 --batch-size 128 --growth-max-depth 16 \
    --warmup-epochs 0.5 --optimizer rmsprop --lr 3e-3 \
    --device cuda --seed 42 --quiet \
    --harmonic-bias --bias-window 64 --bias-n-freq 16 \
    --eval-heldout-frac 0.05 \
    --save-beta /tmp/agpt_pytorch_ab_bias.beta.pt
```

## Artifacts

- `run_a_baseline.log` — full stdout of Run A
- `run_b_bias.log` — full stdout of Run B
- `run_b_bias.beta.pt` — learned β tensor (shape (n_layers, n_heads)) + chord meta

## Results

| Run | byte_perplexity | bits_per_byte | wall (s) |
|---|---:|---:|---:|
| A — baseline | 5.4587 | 2.4486 | 369 |
| B — bias     | 5.4080 | 2.4351 | 375 |

**Δ = −0.93% byte_perplexity, −0.55% bits/byte.**

Learned β (per layer, per head):

```
L0: +0.1057  +0.1813  +0.2020  +0.1922      (all positive, ~0.1–0.2)
L1: −0.0259  +0.1219  +0.0891  −0.0462      (mixed sign)
```

The model moved β away from 0 across most heads, so the optimizer
genuinely found the bias useful. But the magnitude of the PPL win is
within what could plausibly be single-seed noise.

## Conclusion

**Inconclusive.** The signal is real (β ≠ 0 learned) but the
byte_perplexity improvement is below 1% and was measured at a single
seed. Going from "interesting hint" to "ship the CUDA kernel" needs at
minimum:

1. Multi-seed variance estimate. If 0.9% is the mean of, say,
   0.5%/0.7%/1.3% across 3 seeds with overlapping CIs, the signal isn't
   real. If it's 0.7%/0.9%/1.1% tight cluster, more interesting.
2. Larger model. d_model=64 L=2 is at the very low end; the bias may
   express more (or less) signal at d_model=128 L=8, which is the
   canonical AGPT scale.
3. Different W / n_freq. Diagnostic suggested W=64 HD=48 (= n_freq=24)
   might give slightly stronger signal than n_freq=16 used here.

What this run is NOT evidence for/against:
- Whether the bias would help **AGPT's** training (different schedule,
  trie-based fires, partition_depth 1 + anc-grad). The prototype trains
  a vanilla transformer-LM, not AGPT.
- Whether the bias would help at a held-out source the model never saw
  in any form (this corpus is Shakespeare; the held-out 5% is the END
  of Shakespeare, which has Shakespeare's idiosyncratic vocabulary
  distribution already learned in the first 95%).

## Next steps

Decision: **do not start the CUDA kernel work yet.** Instead, before
committing 2+ days to that, run:

- 3-seed repeat of the A/B at the same scale to see if the −0.93% is
  inside or outside variance.
- One run at the larger AGPT scale (d_model=128 L=8 if local laptop can
  handle 10 SE — probably ~30–60 min wall).

If both come back positive (multi-seed gap > 1.5%, larger-model gap >
0.5%), then committing to the CUDA version is justified. If either is
null, park the harmonic-bias direction and pick up
`notes/seq-len-extension/` from another angle.

## Related

- `notes/seq-len-extension/harmonic-filter-asymmetric.md` — operator spec
- `rnd/harmonic-filter-diagnostic/stratified/` — the diagnostic that
  motivated this prototype (1.6 IQR-unit on/off separation at mass=2-9)
- `src/tools/agpt_pytorch_train.py` — prototype trainer
- `src/tools/agpt_pytorch_bias.py` — chord precompute + bias attention + PyTorch eval
