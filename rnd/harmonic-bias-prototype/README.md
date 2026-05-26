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
| Run ID | fixed_token_ppl | rolling_byte_ppl | bits/byte | wall (s) |
|--------|----------------:|-----------------:|----------:|---------:|
| `20260526T064017-seed1-baseline` | 5.2246 | 8.2629 | 3.0467 | 441.0 |
| `20260526T064738-seed1-bias` | 5.2982 | — | 2.4055 | 542.0 |
| `20260526T065640-seed2-baseline` | 5.4632 | 9.0589 | 3.1793 | 499.0 |
| `20260526T070459-seed2-bias` | 5.5147 | — | 2.4633 | 504.0 |
| `20260526T071324-seed3-baseline` | 5.3488 | 8.9022 | 3.1542 | 331.0 |
| `20260526T071854-seed3-bias` | 5.3452 | — | 2.4182 | 367.0 |
<!-- agpt-experiment-table:end -->

## Conclusion

**Null result. Bias direction parked at this scale.**

Paired comparisons (baseline byte_perplexity vs bias-model with β
applied at eval), fixed-token protocol:

| seed | baseline | bias-on | β=0 (bias model) | Δ(bias − base) | Δ(β applied vs β=0) |
|-----:|---------:|--------:|-----------------:|---------------:|--------------------:|
| 1    | 5.2246   | 5.2982  | 6.2650           | **+1.41%**     | −15.43%             |
| 2    | 5.4632   | 5.5147  | 6.2355           | **+0.94%**     | −11.56%             |
| 3    | 5.3488   | 5.3452  | 6.2425           | **−0.07%**     | −14.37%             |
| mean | —        | —       | —                | **+0.76%**     | −13.79%             |

The bias term IS used by the model — when applied at eval, it
consistently lowers PPL by ~14% on the bias-trained model. But the
underlying weights of the bias-trained model are ~14% worse than the
baseline's at β=0 evaluation. The two cancel to roughly zero net
benefit (slightly negative on 2 of 3 seeds, tie on 1).

Interpretation: the bias term provides predictive signal the model
learns to use, but it doesn't unlock any net capacity the architecture
couldn't already get from RoPE + standard attention at this scale.
The −0.93% seed=42 single-seed result that originally motivated this
variance check was inside ~1% noise.

Per the decision criterion (mean Δ < 1% AND signs not consistently
negative): mean Δ = +0.76% (wrong sign), 2:1 unfavorable across seeds.
**Do not start CUDA kernel work.**

### What could change the picture

1. **Larger model.** d_model=64 L=2 is small. The bias may need more
   capacity to provide net uplift; the diagnostic's 1.6 IQR-unit
   separation suggests there IS signal to learn — just nowhere to put
   it at this scale.
2. **AGPT's actual training cadence**, not vanilla sliding window.
   AGPT's trie-fire schedule has different gradient flow that may
   interact with the bias differently. But measuring that requires the
   CUDA kernel — circular.
3. **Different β init / W / n_freq.** This sweep was β=0 init, W=64,
   n_freq=16. The diagnostic suggested HD=48 (= n_freq=24) had stronger
   separation; at this small model that's likely noise but at scale it
   might matter.

None of these are tonight-sized. Parking the direction; revisit when
the project has both a larger working trainer AND a clear motivating
reason (e.g., a different position-aware op family showing similar
diagnostic-level promise).

### Bonus: what this experiment validates regardless of outcome

The new orchestrator + per-experiment-dir + custom-tools-per-experiment
pattern worked end-to-end on the second try (caught and fixed a
vocab-mismatch bug between iterations). 6 paired runs, full provenance
per run, auto-generated table, single-seed pilot correctly contextualized
by multi-seed follow-up. That's the system functioning as designed.

## Related

- `notes/seq-len-extension/harmonic-filter-asymmetric.md` — operator spec
- `rnd/harmonic-filter-diagnostic/stratified/` — diagnostic that
  motivated this prototype
