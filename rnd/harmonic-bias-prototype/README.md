# harmonic-bias-prototype

**Status:** active (variance check in progress)

## Hypothesis

The asym-DFT harmonic-filter bias term (from
`notes/seq-len-extension/harmonic-filter-asymmetric.md`) added to attention
scores improves byte_perplexity on a small char-LM trained on Shakespeare:

```
score(Q, K) = (Q · K)/√d + β_h · (1/C_K) · Σ_j Re[ z_K[j] · exp(-i p_Q ω_j) ]
```

where `z_K` is the DFT chord of K's position-mod-W distribution, ω_j =
2π(j+1)/W, and β_h is a learnable per-head scalar (init 0).

The diagnostic in `rnd/harmonic-filter-diagnostic/stratified/` showed the
operator separates on/off-path pairs by 1.6 IQR-units at mass=2-9 on
both Shakespeare and Gutenberg. The question this experiment answers:
does that separation translate into measurable PPL improvement during
actual training?

## Scope

Prototype only. Uses local PyTorch trainer (`tools/train.py`) — NOT the
canonical Crystal/CUDA trainer. Training algorithm is dense sliding-
window batched cross-entropy, not AGPT's trie-fire schedule. Result
speaks to "does adding this bias to a vanilla causal-LM help?", not
directly to "does it help AGPT's training cadence?".

Existed because the CUDA-kernel version of the bias is multi-day work
(chord ingestion into trainer, per-query position passing, per-K
substring lookup, fused-softmax bias forward/backward, β as a saved
model parameter). This prototype lets us A/B before committing.

A single-seed pilot at seed=42 (before this experiment was wired into
the orchestrator) showed Δ=-0.93% byte_perplexity for the bias variant
— under the 1% threshold and not enough signal to commit to CUDA work.
This variance check runs 3 seeds × {baseline, bias} through the
orchestrator to see if the signal is real or single-seed noise.

## Setup

- Model: `data/input.model` (d_model=64, n_heads=4, n_layers=2, d_ff=256, vocab=65)
- Corpus: Shakespeare (`data/input.txt`, 1.115M chars)
- Train slice: first 95% (1,059,624 chars); heldout: last 5% (55,770)
- d_window=16
- Trainer: `tools/train.py` (PyTorch sliding-window)
- Optimizer: RMSprop lr=3e-3, warmup-cosine (1 warmup-epoch)
- Epochs: 10
- Batch size: 128
- Seeds: 1, 2, 3 (three each per variant)
- Bias variant: W=64, n_freq=16, β init=0, per-(layer, head)
- Eval: `tools/lm_eval_with_bias.py` (bias variant — loads β sidecar
  and exercises bias at eval) vs `src/tools/agpt_lm_eval.py` (baseline)

## Tools (local to this experiment)

- `tools/train.py` — PyTorch trainer, accepts `--harmonic-bias` etc.
- `tools/bias.py` — chord precompute + bias-attention forward +
  PyTorch byte_perplexity
- `tools/lm_eval_with_bias.py` — β-aware byte_perplexity evaluator that
  writes lm-eval-shape JSON so the orchestrator picks it up unchanged

## Reproduce

```sh
just build-agpt-experiment  # one-time
for cfg in rnd/harmonic-bias-prototype/configs/seed{1,2,3}-{baseline,bias}.yml; do
    bin/agpt_experiment --config "$cfg"
done
```

Each run produces a `<UTC-stamp>-<slug>/` subdir with full provenance
(config.yml + meta.json + result.json + train.log + eval.log).

## Results

<!-- agpt-experiment-table:start -->
(populated by orchestrator as runs complete)
<!-- agpt-experiment-table:end -->

## Conclusion

(to be filled in once all 6 runs land)

**Decision criterion:** if mean Δ across 3 paired-seed (bias − baseline)
comparisons is < 1% AND/OR the per-seed Δ has the same sign less than 3
of 3 times → null result, park the bias direction. Otherwise → larger-
model run (d_model=128 L=8) as the next gate before CUDA work.

## Related

- `notes/seq-len-extension/harmonic-filter-asymmetric.md` — operator spec
- `rnd/harmonic-filter-diagnostic/stratified/` — diagnostic that
  motivated this prototype
