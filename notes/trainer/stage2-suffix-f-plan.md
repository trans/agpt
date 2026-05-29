# Stage 2: Within-Step Suffix-Ensemble F_p — Plan

Stage 2 of topological optimizer state. Builds on the suffix-weighted
curvature proposal in `notes/optimization/suffix_weighted_curvature.md` and the
Stage 1 closure documented in `rnd/per-rc-adam-v1/findings.md`.

## Why Stage 2 and not more Stage 1 variants

Stage 1 (`--per-rc-adam`) closed cleanly with two findings:

1. **The topological localization signal exists.** The diagnostic dump
   showed structured per-rc `v` variation: ‖v‖₂ spanning 8 orders of
   magnitude, cosine similarities 0.23-0.83 (structured, not uniform).
   Different rcs have measurably different curvature profiles.

2. **Temporal EMA at ~50 fires/bucket can't extract it.** Bias from
   slow β₂=0.999 warmup AND variance from small per-bucket sample sizes
   compound. Mass-weighting ablation didn't help (the gap stayed
   ~20%). Stage 1's failure mode is intrinsic to *temporal* accumulation.

Stage 2 sidesteps both problems by construction. F_p is estimated from
the **spatial ensemble of within-fire suffix contributions** — every
descendant of node p contributes one sample, all available
simultaneously in one backward pass. No warmup. No sample-size scaling
with super-epoch count.

## The mechanism

At each backward aggregation point (interior node p where descendants
contribute gradients):

```
G_p = Σ_s n_s · g_s       (existing, count-weighted first moment)
F_p = Σ_s n_s · (g_s ⊙ g_s)   (NEW, count-weighted diagonal second moment)
```

`g_s` is the gradient contribution from descendant subtree s; `n_s` is
the count weight on that branch. F_p is the diagonal empirical Fisher
at p, estimated from p's own suffix ensemble — no temporal EMA, no
cross-step staleness.

Used as a per-fire preconditioner:
```
θ ← θ − η · G_p / (sqrt(F_p) + ε)
```

Per the curvature note's "Interpretation": when suffixes agree
(F_p ≈ G_p² / N_s) the preconditioner is roughly uniform; when they
disagree (F_p ≫ G_p² / N_s) the preconditioner damps the step,
recognizing the aggregate as structurally fragile.

## Where in the code

The hot path is `kv_uncopy_own_edge_kernel` in
`src/cuda/agpt_train.cu` (line ~2158). This kernel scatters
descendant K/V gradient contributions back to ancestor slots via
`atomicAdd`. The **per-suffix `g_s` is in scope at exactly the moment
the atomicAdd happens**, before they get summed away.

### Stage 2a (minimal viable surgery)

Add a parallel squared-contribution accumulator at the K/V level. The
backward pass already visits each node p and accumulates K/V grad
contributions from descendants; adding F_kv is one extra atomicAdd
per existing atomicAdd:

```cuda
float val = packed_grad[...];
atomicAdd(&d_grad_kv[ancestor * d_model + col], val);     // existing
atomicAdd(&d_F_kv  [ancestor * d_model + col], val*val);  // NEW
```

This gives us F at the K/V *position* level: for each ancestor's
(K, dim) and (V, dim), the count-weighted sum of squared descendant
gradient contributions.

**Use F_kv to precondition the K/V gradient at aggregation time:**
```cuda
grad_kv_scaled[p, d] = grad_kv[p, d] / (sqrt(F_kv[p, d]) + ε)
```
before propagating back through the chain rule to FFN/projection
weights. Preconditioning naturally translates to parameter-level
preconditioning via the standard backward pass.

### Why 2a over a full parameter-level F (Stage 2b)

Parameter-level F_W = h^T · ∂L/∂W requires unfusing the per-suffix
decomposition all the way to the parameter gradients — at the FFN
this is straightforward (the outer-product carries through) but at
the attention block it's messier because the K-side gradients
genuinely collapse at the node (each interior K is shared
infrastructure for every descendant).

Stage 2a applies preconditioning at the position level (where the
per-suffix structure is most naturally available) and lets the chain
rule propagate the resulting scaled gradient to parameters. It's
the smallest possible test of "does within-step F help" with a
clean integration point.

If 2a works (matches baseline or beats it at 50 SE), 2b becomes
optional. If 2a is flat, 2b might still help (parameter-level F is
the principled quantity); reassess at that point.

## Files to modify

| File | Change | Lines |
|---|---|---:|
| `src/cuda/agpt_train.cu` | Add `d_F_kv` allocation alongside existing `d_dkv_keys[l]`, `d_dkv_values[l]` per layer | ~5 |
| `src/cuda/agpt_train.cu` | New kernel `kv_uncopy_own_edge_F_kernel` (or modify existing to take F output) | ~20 |
| `src/cuda/agpt_train.cu` | New kernel `kv_precondition_kernel`: scale `d_grad_kv /= sqrt(d_F_kv + ε)` | ~15 |
| `src/cuda/agpt_train.cu` | Zero `d_F_kv` between subtree fires | ~3 |
| `src/cuda/agpt_train.cu` | CLI flag `--suffix-f` to opt in | ~5 |

Total: ~50 lines of CUDA. Plus a `--dump-suffix-f` debugging flag
modeled on Stage 1's `--dump-per-rc-v`.

## Memory budget

`d_F_kv` is same shape as one of the existing K-grad or V-grad
accumulators per layer: `[total_nodes × d_model]`. For Shakespeare 1M
d=16:
- 1M nodes × 128 dims × 4 bytes = 512 MB per layer
- × 4 layers × 2 (K and V) = **~4 GB total**

Borderline on the laptop's 4070 (8GB VRAM). Options if pressure hits:
- **BF16 the F accumulator**: halves memory to ~2GB, at small accuracy
  cost (F values are squared so range is wide; might bias low for
  small values). Worth ablating.
- **Per-layer freed-between-layers**: only one layer's F lives at a
  time. Halves to ~2GB.
- **Compact-cache-aware**: skip F for mass=1 caps (same logic as the
  existing K/V compact cache from `project_compact_kv_cache`). Could
  save 90%+ at d=32.

First experiment: just allocate the full thing. Optimize later if it
OOMs.

## Verification plan

### V.1 Sanity: gradient parity test

Before any PPL measurement, verify that the new code path produces
the *same gradients* as baseline when `--suffix-f` is off (control)
and that turning it on produces grad/sqrt(F) values that match a
CPU-computed reference for a tiny trie.

Implementation: take a known small trie (~10 nodes), compute gradients
both via `agpt_train` (with `--suffix-f`) and a Python NumPy reference
that does the explicit Σ_s n_s · (g_s ⊙ g_s) loop. Compare F_kv values
elementwise to within 1e-5. Catches kernel bugs before any
expensive run.

### V.2 No-op parity

Run baseline (without `--suffix-f`) before and after the patch. PPL
should be bitwise identical. Confirms the new code path is dead when
unused.

### V.3 Headline experiment

Same 6-run setup as Stage 1: Shakespeare d=16, 50 SE, 3 seeds × 2
variants (baseline vs `--suffix-f`). Compare PPL means with Welch's t.

**Pass criterion**: F-based preconditioning matches or beats baseline
at 50 SE. Magnitude doesn't have to be huge — even a 1-2% PPL
improvement validates the within-step approach over Stage 1's
temporal approach.

**Failure interpretations:**
- Flat at 50 SE: F signal is real but too weak at K/V level alone.
  Consider Stage 2b (parameter-level F) before closing.
- Regression: preconditioning at the position level disrupts the
  chain-rule scaling unexpectedly. Investigate where (FFN vs
  attention vs LM head) and possibly apply F more surgically.

### V.4 F sanity diagnostic

Dump F_kv at end of a run. Compute per-bucket stats analogous to
Stage 1's dump analysis:
- Distribution of F_kv values across (node, dim)
- Coherence metric: `C_p = ‖G_p‖² / Σ_s ‖g_s‖²` per node
- Cosine similarity of F across nodes at the same depth

If F_kv is essentially uniform across positions (low coherence
variance), F isn't carrying signal. If F_kv has structure that
correlates with node depth, node mass, or bucketed similarity to
other nodes, signal is present.

### V.5 Streaming compatibility

Run `--suffix-f` × streaming-AGPT (100×5 SE Gutenberg) for 1 seed.
We don't expect interaction effects, but worth verifying that
Stage 2 doesn't break streaming's incremental fresh-warmup pattern.

## Open design questions

1. **Bias correction.** The note's F_p doesn't have an Adam-style
   bias correction because it's not a temporal EMA. But the
   sample size N_s (number of descendants) varies wildly across
   nodes — deep nodes have few descendants, shallow nodes have
   many. Should F_p be divided by N_s (to make it variance-like)
   or kept as count-weighted sum? The curvature note doesn't say
   explicitly. **Default to no division for first experiment;**
   N_s normalization would change the per-fire step magnitude in
   ways that could confound the PPL signal.

2. **Interaction with `--mass-weight log`.** Mass-weighting
   scales each event's gradient by w_i. F_p computed from
   already-weighted gradients double-counts (gradient g_s already
   has w_s baked in, then F gets w_s² effectively). Run with
   mass-weight on (current recipe) for direct comparability to
   the streaming-AGPT findings. Ablate with mass-weight off as
   a separate experiment.

3. **F per-layer vs F shared.** Each transformer layer has its
   own K/V projection. Should F be computed per-layer or shared?
   First experiment: per-layer (simplest, matches the natural
   data layout). Shared F would require aggregating across
   layers somewhere; not obviously helpful.

4. **Numerical stability.** F values can be very small (squared
   tiny gradients) leading to huge effective LR scaling. The ε
   in `sqrt(F) + ε` matters; default 1e-8 might be wrong. Plan:
   start with 1e-8 (Adam default), monitor for explosions, tune
   if needed.

## Connection back to broader topological optimizer program

Stage 2 is the first real instance of "topological optimizer state":
optimizer state computed from trie structure (the suffix ensemble at
each node) rather than from optimization history (temporal EMA).

If Stage 2 works, the next stages are:

- **Stage 3: block-diagonal Fisher**. Replace diagonal F with
  per-layer block-diagonal C_p = Σ_s n_s · g_s g_s^T. Stronger
  preconditioner; quadratic memory in d_model per node so feasible
  only for small dims or restricted to specific layers.

- **Stage 4: pure-topology curvature surrogate**. Compute curvature
  prior from trie structure alone — entropy at endpoint, branch
  divergence, subtree mass distribution — without any gradient
  computation. The trie *already encodes* curvature-like information
  (high entropy = high curvature region, branch divergence =
  representational competition). Use as a static prior for the
  optimizer; no kernel surgery needed.

Stages 3 and 4 sit downstream of Stage 2 working. They get planned
in detail only after Stage 2 produces a real result.

## Time estimate

- Implementation (Stage 2a kernels + CLI + zero-between-fires): ~1 day
- V.1 + V.2 verification (parity tests): ~half day
- V.3 headline experiment: ~12 min compute + ~half day to write up
- V.4 + V.5: ~half day total

Total: **2-3 days of focused work** for Stage 2a end-to-end. Possibly
faster if no surprises in the gradient parity test (V.1).

The 50-line CUDA estimate could be wrong — kernel modifications often
surface subtle issues with thread layout, shared memory, or atomic
contention that take longer than expected. 2-3 days is the realistic
estimate.
