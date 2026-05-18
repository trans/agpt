# Sliding-Window AGPT v1 — inference prototype

**Status:** in progress, started 2026-05-11
**Design doc:** `notes/sliding_window_agpt.md`
**Phase 0 dependency:** `rnd/seq-len-decouple/` (position→node map; not
strictly required for v1 inference but useful for verification)

## Goal

Build the smallest possible inference-time prototype of
sliding-window AGPT and answer one question: **does pooling
activations from overlapping d-windows and running sequence-level
attention on the pooled representations produce a better PPL than the
d=16 model evaluated at seq_len=16?**

Pass/fail:
- **Pass:** PPL at sliding-window seq_len > 16 *beats* PPL@16 = 8.01
  on Gutenberg 5M with the d=16-trained AGPT model.
- **Fail:** PPL ≥ PPL@16. Either the existing model's activations
  aren't useful as pooled features, or uniform pooling is too lossy.
  Move to v1.1 with depth-weighted pooling, or escalate to v2
  (stack-attention or per-node K vectors).

## Setup

- **Model:** `/tmp/agpt-gut-d16-pd1.model` — 30 SE pd=1 d=16 Gutenberg,
  PPL@16 = 8.01 baseline. (If gone from tmpfs we retrain — at
  ~22 s/SE × 100 SE = 37 min.)
- **Trie:** `/home/trans/agpt-tries/gutenberg_5m_d16_radix_corpus`
- **Eval corpus:** `data/gutenberg_5m.txt`
- **Eval scope (v1):** small held-out window, say 1000–5000 positions.
  Sufficient for a PPL number with reasonable noise.

## Test variants

For each, compute mean per-token NLL and PPL on the eval window:

1. **Baseline (no sliding window):** standard model inference at
   `seq_len = 16`. Should reproduce PPL = 8.01.
2. **Sliding-window seq_len = 32, uniform pool, coverage-normalized.**
   At each query position q, attend to 32 pooled positions.
3. **Sliding-window seq_len = 64, uniform pool.**
4. **(Optional)** Sliding-window seq_len = 32, depth-weighted pool
   (`w_k ∝ k+1`). Tests whether favoring deeper contributors helps.

## Implementation outline

The inference loop for one query position q at sliding-window
seq_len = S:

```
1. Gather window forward passes
   For each w in [q-S+1 ... q]:                     # S start positions
     - Run model.forward(corpus[w..w+d-1]) but with hooks to capture
       per-layer residual at each within-window depth j ∈ [0..d-1].
     - Save the *post-final-block* residual at depth j for use later.
     - Note: window w only produces useful contributions for positions
       p = w+j where p ∈ [q-S+1, q]. Other depths can be discarded.

2. Pool by global position
   For each p in [q-S+1 ... q]:
     - Collect contributors {(w, j) : w + j = p, w ∈ [q-S+1, q],
                              j ∈ [0, d-1], 0 ≤ p < N}
     - h_p = mean(contributor residuals)
     - (Edge positions: divide by actual contributor count for
        coverage normalization. Interior positions get d contributors.)

3. Sequence-level attention
   - Compute K_p = RoPE(W_K · h_p, p), V_p = W_V · h_p,
     Q_q = RoPE(W_Q · h_q, q) for p ∈ [q-S+1, q].
   - Standard softmax attention over S keys/values, multi-head.
   - (Reuse the *final-block's* W_K, W_Q, W_V — they're the
     projections trained at chunk-internal attention. Architectural
     question: do we want the final block's *whole* attention here,
     or just its projections? v1 just uses the projections; the rest
     of the block is skipped.)

4. Output logits and loss
   - att_out at q → W_unembed → softmax → log-prob of corpus[q+1].
   - Accumulate NLL.
```

The expensive part: step 1 needs S forward passes through the
transformer per query position. For S = 32 that's 32× the cost of
standard inference. Acceptable for a 1000-position eval (32k chunk
forwards in total ≈ minutes on GPU).

## Implementation language

**Decision needed:** Crystal (extends `bin/perplexity`) or Python
prototype (NumPy + manual forward pass)?

- **Crystal:** keeps everything in one toolchain. Existing perplexity
  tool has the model loading and forward-pass scaffolding. Need to
  add: per-layer residual capture, the pooling step, a small custom
  attention pass.
- **Python:** faster to prototype, easier debugging, but requires
  porting model weights (or using onnx export). Adds a dependency
  layer to the project.

Recommendation: Crystal. Less moving parts, prototype lives next to
the production tooling, results stay reproducible inside the existing
build system.

## Files this experiment will produce

```
rnd/sliding-window-v1/
├── README.md                        (this file)
├── findings.md                      (results once we have them)
├── logs/
│   ├── baseline_ppl16.log           (sanity recheck)
│   ├── slidingwin_s32_uniform.log
│   ├── slidingwin_s64_uniform.log
│   └── slidingwin_s32_depthweighted.log
└── (binary outputs gitignored, in /tmp if any)
```

The tool itself will live at `src/tools/agpt_sliding_window_perplexity.cr`
(name TBD), built via Justfile.

## Open questions for v1 implementation

1. **Which residual to capture?** Post-final-block (after all
   transformer layers but before unembedding)? Post-attention but
   pre-FF in the final block? Some earlier layer? v1 default:
   post-final-block (the "final hidden state" of the chunk forward
   pass at each within-window depth).
2. **Do we run the model's full chunk forward at each window, or just
   layer 1?** The full forward exposes the layer hierarchy already
   trained; layer 1 only is cheaper but loses depth. v1 default: full
   forward.
3. **Coverage normalization at edges.** Position p has fewer
   contributors when |p − N/2| > N/2 − d/2. Just divide by actual
   contributor count.
4. **What's the "final block's W_K, W_Q, W_V" we should use at
   step 3?** The model has multiple transformer blocks. Each has its
   own W_K, W_Q, W_V. For v1 we use the *last* block's projections,
   since those operate on the deepest residuals — but this is a
   design choice worth revisiting.

## Followup if v1 succeeds

- Train end-to-end with the pooling step in the training loop, not
  just at inference.
- Try depth-weighted, then learned-gating pooling.
- Try seq_len ∈ {64, 128, 256} for the scaling curve.
- Move to v2: stack-attention (multi-key per position).

## Followup if v1 fails

- Try depth-weighted pool before giving up on pooling entirely.
- If still failing: pooled activations from a chunk-trained model
  probably aren't carrying useful seq-level signal. Move to per-node
  K-vector training (option B in shared_key_rope.md) — the bigger
  lift, but architecturally more principled.
