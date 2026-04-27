# Lightning L3 — mass cap + ancestor warmup on Gutenberg 5M (d=32)

> **Status (2026-04-26): closed.** Architecture validated, PPL ceiling at this
> model size identified. Pivoting to prefix-to-suffix attention
> (`rnd/p2s-attention/`). Re-open if revisiting the leaf-to-root wrapping
> formulation.

## Hypothesis

Lightning L3 (mass-walk + per-sample Adam fire) on Gutenberg 5M was blowing up
to ~5 hours wall-clock because samples landed at root-children with
million-node subtrees. Two-part fix:

1. **Mass cap** (`--lightning-max-mass N`): force the mass-walk to keep
   descending past nodes with `mass > N`. Bounds per-step subtree size.
2. **Ancestor warmup**: prepend `r_sample`'s root-to-parent ancestor chain to
   the BFS so their K/V (computed from current weights) is scattered into the
   cache before any descendant attention reads from those positions. Without
   this, the cache holds zeros (epoch-init) or stale K/V from prior steps —
   descendant attention reads garbage prefix.

Goal: make L3 produce L4-comparable PPL at much faster wall-clock, then push
past L4 by exploiting Lightning's larger optimizer-step budget.

## What worked

- **Mass cap mechanism**: bounds wall-clock as designed. 10K capped L3 steps
  in 8 sec at cap=10, vs 5+ hr uncapped.
- **L3 cap=10 ≈ L4 in PPL**: 30.83 vs 29.15. Confirms the architecture is sound
  and that "small L3 batches approximate L4" in the limit. **130× wall-clock
  speedup** for ~1.7 PPL gap.
- **CLI footgun fix** (`--save` defaults to `--model`): caught silently
  discarded weights from multiple prior runs. Pre-existing model files in
  `/home/trans/agpt-tries/` (e.g. `g5m_d32_l3_10k.model`) are bit-identical
  to random init from this bug.

## What didn't work

- **Naive warmup at loose cap (50K)**: PPL 65.2, *worse* than no warmup
  (PPL 56.3). The cap forces deep r_samples; ancestors appearing in many BFS
  sets per epoch get over-trained. Same shallow rootID 149117 dominated
  residuals across many root-children.
- **Masked-ancestor warmup (anc loss = 0)**: PPL 59.4 — better than naive
  (65.2) but worse than no warmup (56.3). The K/V backward path still routed
  gradient into shallow ancestor `Wk/Wv` weights via descendant attention's
  K/V grad — masking first-order CE wasn't enough. Reverted.
- **Tighter cap doesn't help if architecture is capacity-bound**: see ceiling
  finding below.

## Why no further recipe tuning helps

| Method | Wall | Adam fires | PPL |
|---|---:|---:|---:|
| Deterministic AGPT 3ep (unigram) | 4:15 | 195 | **29.01** |
| L4 path-sampling 10K | ~17 min | 10K | 29.15 |
| L3 cap=10 1ep | 0:08 | 10K | 30.83 |
| SGD seq=32 10K | 4:45 | 10K | 35.14 |
| L3 cap=50K 1ep | 17:48 | 10K | 56.30 |

**All three AGPT-family approaches converge to PPL ~29 at d_model=64,
n_layers=2.** SGD at the same context length is ~6 PPL worse, validating that
prefix-context aggregation buys real PPL — but the ceiling at this model size
is the capacity, not the training signal.

## "Sorted SGD" framing (the theory that fits)

Per-step batch grows → optimizer step direction averages over more
conflicting prefixes → training behaves like sorted-batch SGD with no
shuffling between → known-bad for SGD convergence. The cap=50K result is the
extreme case (~10K positions per Adam fire = sorted big-batch SGD). The
cap=10 result lives in the small-batch SGD regime that works.

This also explains:
- **Bigram split underperforms unigram split**: more partitions means smaller
  per-partition batches but each partition is still locally-coherent / globally
  -correlated. 1139 sorted updates ≠ 1139 random updates.
- **Shakespeare worked, Gutenberg doesn't (at loose cap)**: 5× corpus → 7×
  larger root-child subtrees → larger per-step batches → more sorted-SGD-like.

The smart-batching hypothesis has an inherent **per-step batch ceiling**
independent of compute budget.

## Code preserved on main

| Change | Why |
|---|---|
| `--lightning-max-mass N` flag | Bounds wall-clock, useful for fast experiments |
| `subtree_n_anc[]` + ancestor warmup in BFS (naive variant) | Required for cap≥1 to produce sensible PPL |
| `d_mass_weights` always-allocated | Needed for ancestor handling |
| `--save` defaults to `--model` | CLI footgun fix |
| `size[min/mean/max]` histogram | Replaces meaningless `mass[]` print (sum was depth-inflated) |

## Code reverted

| Change | Why reverted |
|---|---|
| Masked-ancestor warmup (anc loss=0) | PPL worse than naive warmup; masking first-order CE doesn't gate the backward K/V grad path that over-trains shallow nodes |

## Open thread (re-open if revisiting)

The ancestor over-training under loose cap is a real problem if we want to
exploit the cap mechanism for many-step Lightning training. The **leaf-to-root
wrapping** angle (run Lightning from the leaf perspective with explicit
ancestor sharing across samples) might address it more cleanly than per-sample
prepending. Out of scope for now.

## Reproduce headline result (L3 cap=10 ≈ L4)

```sh
cp data/input.random.model /tmp/g5m_l3_cap10.model
bin/agpt_train --model /tmp/g5m_l3_cap10.model \
  --trie-dir /home/trans/agpt-tries/gutenberg_5m_d32_radix_corpus \
  --epochs 1 --lr 3e-3 --d-model 64 --n-layers 2 \
  --lightning-steps 10000 --lightning-sampler l3 --lightning-p-stop 0.3 \
  --lightning-max-mass 10 \
  --no-accumulate --rmsprop --weight-clip-mode off \
  --schedule warmup-cosine --warmup-steps 1000

bin/perplexity --model /tmp/g5m_l3_cap10.model \
  --file data/gutenberg_5m.txt --max-positions 4096 --backend openblas
# → PPL 30.83 (NLL 3.43), wall ~8s + 30s eval
```
