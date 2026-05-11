# TODO — L-BFGS Optimizer (implemented but broken)

**Status:** Implemented 2026-05-03. After bug-fix pass (curvature-condition,
pushed_count tracking, gamma fallback, cleanup), **no longer NaNs** but
**no longer learns either** — the L-BFGS step collapses to ~lr² magnitude
and becomes a no-op. PPL stays at ~157 (worse than random ~64) at pd=6
across all K and LR values tested.

**Fundamental cause confirmed (Hypothesis 4 from review):** AGPT's
per-partition fire structure makes consecutive gradients near-orthogonal
(each fire uses a different partition group's events). L-BFGS's H scaling
assumes correlated consecutive gradients (shared loss surface) and
collapses gamma when this assumption is violated. The s/y history pairs
become structurally inconsistent with each other.

**Implementation is technically correct but the algorithm doesn't fit
the data distribution.** Needs structural restructure, not bug fixes.

## What was built

- `OptimizerKind::LBFGS` enum variant added to `agpt_train.cu`
- `LBFGSState` struct with K-history buffers (s_hist, y_hist, ρ_hist,
  alpha, q scratch, g_prev, step)
- `cuda_lbfgs_step()` function implementing standard two-loop recursion
  using cuBLAS Saxpy/Sdot/Sscal/Scopy
- CLI flags: `--optimizer lbfgs` and `--lbfgs-k N` (default 10)
- Allocation block in `run_radix_training` (skipped unless optimizer = LBFGS)
- Wired into both optimizer-step switch sites (per-subtree and accumulate)

About 250 LOC total. Compiles clean (only pre-existing warnings). Memory
budget: K=10 × 108k params × 2 buffers × 4 bytes = ~8.6 MB. Trivial.

## What works

- Builds cleanly with `just build-agpt-train`
- `--optimizer lbfgs --lbfgs-k 10` CLI accepted; state allocation prints
  correctly (e.g., "lbfgs state: K=10, n=108481, 9.52 MB")
- **Epoch 1 trains successfully** at all tested LRs (1e-3 down to 1e-7)
- First-step SGD (no history yet) runs without error

## What's broken

**Divergence after epoch 1**, even at very low LR (tested down to lr=1e-7
at pd=1).

Concrete trajectory (lr=1e-7, pd=1, d=16, 3 SE):

| epoch | loss | nodes processed |
|---:|---:|---:|
| 1 | 5.19 | 9.3M (clean) |
| 2 | 11.58 | 4.4M (degraded — half the nodes NaN-skipped) |
| 3 | 0.00 | 0 (full NaN — silently zeroed) |

Pattern: model trains normally during the first super-epoch (which uses
mostly first-step SGD per fire when lbfgs history is empty/just starting).
By epoch 2, history is built and the L-BFGS curvature step kicks in,
producing weight updates that slowly NaN the model. By epoch 3 every
forward pass produces NaN.

## Diagnosed failure modes

### Hypothesis 1: lax curvature condition (most likely)

```cpp
if (ys > 1e-10f && std::isfinite(ys)) {
    // push to history...
}
```

When `ys = y^T s` is barely positive (e.g., 1e-9), `ρ = 1/ys ≈ 1e9`. In the
two-loop recursion, `α[i] = ρ · sᵀq` becomes huge. `q -= α · y` then
swings q by enormous amounts. Subsequent iterations explode.

**Fix candidate:** strengthen the curvature threshold. Standard L-BFGS
uses `ys > ε · ||y|| · ||s||` (a relative threshold) rather than an
absolute threshold. Recommend testing `ys > 1e-6 * sqrt(yy * ss)` or
similar.

### Hypothesis 2: cuBLAS async ordering (less likely)

`cublasSdot` with HOST_PTR mode is documented as synchronous (blocks
until result is in host memory). My code assumes this and immediately
uses dot results. If there's a stream/handle subtlety where the call
isn't actually synchronous, race conditions could give zero-results
that propagate.

**Fix candidate:** explicitly call `cublasSetPointerMode(cublas,
CUBLAS_POINTER_MODE_HOST)` once at handle creation. Add explicit
`cudaStreamSynchronize` after each Sdot if the issue persists.

### Hypothesis 3: Sign error in two-loop (re-checked, looks right)

I re-read the implementation against standard L-BFGS pseudocode three
times and the indexing/signs look correct:

- First loop newest-to-oldest: ✓
- Second loop oldest-to-newest: ✓
- α stored at slot index, retrieved at same slot index: ✓
- `θ -= lr * (H * g)` direction: ✓
- γ = (s_lastᵀy_last) / (y_lastᵀy_last) = 1/(ρ · yy): ✓

If this is the issue, it's subtle and I missed it.

### Hypothesis 4: AGPT's per-fire gradient distribution incompatible with L-BFGS assumptions

L-BFGS assumes gradients come from a roughly stationary distribution
(consecutive fires sample similar gradient surfaces). AGPT's fires across
DIFFERENT partition groups (different prefixes, different events)
intentionally sample different surfaces. The K-history may contain
(s, y) pairs from semantically unrelated optimization regions, making
the H approximation meaningless or destructive.

**Fix candidate:** maintain SEPARATE L-BFGS histories per partition
group (or per root-child). Heavy memory cost: 65 root-children × K=10 ×
n × 4 = 280 MB. Or use much smaller K (=1, =2). Or implement per-fire
reset after every Adam step (degenerates to fancy SGD).

### Hypothesis 5: First-step SGD magnitude (ruled out — still NaNs at lr=1e-7)

Originally suspected, but lr=1e-7 still NaN'd, so SGD step magnitude
isn't the root cause. The first epoch trains fine even at lr=1e-3 in
the smoke test.

## Suggested debugging plan (next session)

1. **Add diagnostic prints** to `cuda_lbfgs_step`:
   - Per-step: |g|, |g_prev|, |y_new|, |s|, ys, ρ, γ, |q|_after_first_loop, |q|_after_gamma, |q|_final
   - Identify when values first become abnormal

2. **Test Hypothesis 1 first** (curvature threshold):
   - Tighten to `ys > 1e-6 * sqrt(yy * ss)`
   - If divergence delays or stops, that was it

3. **Test Hypothesis 2** (sync):
   - Add `cudaDeviceSynchronize()` after every cuBLAS scalar call
   - If divergence stops, race condition was it

4. **Test Hypothesis 4** (gradient stationarity):
   - Reset L-BFGS state at every super-epoch boundary
   - Or, even more aggressive: K=1 (one history entry only)
   - If even K=1 NaNs, the issue is in the basic mechanics

5. **Compare against a reference impl**:
   - Write a tiny CPU-only L-BFGS test on the Rosenbrock function
   - Verify the two-loop produces correct steps for a known problem
   - Once verified, port back to the AGPT trainer

## Code locations

- Enum: `src/cuda/agpt_train.cu:132` — `OptimizerKind::LBFGS`
- Config field: `src/cuda/agpt_train.cu:113` — `int lbfgs_k = 10`
- `LBFGSState` struct: `src/cuda/agpt_train.cu:~190`
- `cuda_lbfgs_step()` function: `src/cuda/agpt_train.cu:~210`
- State allocation: `src/cuda/agpt_train.cu:~3635`
- Switch case: `src/cuda/agpt_train.cu:~5870` (per-subtree) and second site
- CLI: `src/cuda/agpt_train.cu:~6645`

## Memory state

- Nothing on disk persisting wrong state — fresh runs start fresh
- `/tmp/dr_lbfgs_*.model` files contain NaN'd weights from failed runs;
  safe to delete

## When this is fixed

Once L-BFGS produces sane PPL at lr=1e-3 pd=6 3 SE:

- Compare against current best Adam pd=6 3 SE = 3.71 PPL@32
- Test pd=1 (where coherent gradients should suit L-BFGS best)
- Test 60 SE / 120 SE (does L-BFGS reach lower plateau than Adam?)
- Compare wall-clock per equivalent PPL

L-BFGS theoretically should help more at pd=1 (low-noise gradients) than
at pd=6 (noisy small batches). But that's a hypothesis until measured.

## Reference

- Architecture design: `memory/project_lbfgs_optimizer.md` (in user's
  Claude memory, has the original design rationale and pseudocode)
- Implementation commit (when made): TBD

## Effort estimate

- Fix + verify: ~half day
- Probe matrix at d=16 to find best LR/K: ~2 hours
- Headline runs at d=32: ~1-2 hours
- Total to "L-BFGS evaluated as a real recipe option": ~1 day focused work

---

## Update 2026-05-03 (post bug-fix testing)

### What we tested

After applying the 5-finding bug-fix pass (pushed_count tracking, relative
curvature threshold, gamma fallback to last-pushed pair, cleanup
allocations, and others), ran the matrix at d=16 pd=6 3 SE no-wd:

| recipe | PPL@16 | wall |
|---|---:|---:|
| Adam baseline | **4.14** | 497s |
| L-BFGS pd=6 K=10 lr=1e-3 | 157.79 | 816s |
| L-BFGS pd=6 K=10 lr=1e-4 | 157.81 | 785s |
| L-BFGS pd=6 K=5 lr=1e-3 | 157.79 | 817s |
| L-BFGS pd=6 K=1 lr=1e-3 | 157.79 | 582s |
| L-BFGS pd=1 K=10 lr=1e-3 | 69.15 | 17s |
| L-BFGS pd=1 K=10 lr=3e-3 | 738.04 | 18s |

### What we learned

1. **NaN is gone.** All v2 runs completed cleanly without numerical
   blowup. The Finding 1/3/5 fixes work as intended.

2. **The optimizer no-ops instead.** Loss across 3 epochs at pd=6 K=10
   lr=1e-3:
   - Epoch 1: 5.117764
   - Epoch 2: 5.117749 (Δ -0.000015)
   - Epoch 3: 5.117740 (Δ -0.000009)
   Identical PPL between lr=1e-3 (157.79) and lr=1e-4 (157.81) — LR
   doesn't matter because the step is essentially zero.

3. **The K-history doesn't help.** K=1, K=5, K=10 all produce identical
   PPL (157.79) at pd=6. The collapse isn't a multi-history bug.

4. **At pd=1, the FIRST SGD step does substantial damage** (lr · |g|
   for huge AGPT pd=1 gradients). lr=1e-3 → PPL 69 (essentially random).
   lr=3e-3 → PPL 738 (catastrophic). After the first step, the L-BFGS
   step also no-ops, freezing weights at the post-SGD state.

### Why: gamma collapse for orthogonal-gradient partitions

For AGPT pd=6, consecutive Adam fires train DIFFERENT partition groups
(different prefixes). The gradients at consecutive fires are nearly
orthogonal, so:

- s_k = -lr · g_{k-1}    (small magnitude, |s| ≈ lr · |g|)
- y_k = g_k - g_{k-1} ≈ g_k  (full magnitude, |y| ≈ |g|)
- ys = -lr · (g_kᵀg_{k-1} - |g_{k-1}|²) ≈ lr · |g|²  (positive but small)

Then in the two-loop:
- ρ = 1/ys ≈ 1/(lr · |g|²)  (LARGE if lr small)
- α_i = ρ · sᵀq ≈ (1/lr · 1/|g|²) · (-lr · g_oldᵀq) = -g_oldᵀq/|g|²  (~0 for orthogonal)
- γ = ys/yy = (lr · |g|²)/|g|² = **lr**  (small!)

After two-loop: q ≈ γ · g_curr ≈ **lr · g_curr** (instead of expected H · g_curr).

Then: θ -= lr · q = θ -= lr · (lr · g_curr) = θ -= **lr² · g_curr**.

**Effective step is lr², not lr.** With lr=1e-3, that's 1e-6 — barely
moves weights. Hence the no-op.

### Verdict

L-BFGS doesn't compose with AGPT's per-partition fire structure. The
algorithm assumes consecutive gradients sample a coherent loss surface;
AGPT's per-partition fires intentionally sample different surfaces.

**Fixing this isn't a one-line change.** Options:

1. **Per-partition history**: maintain separate K-history per partition
   group. Memory cost at pd=6 = 283k × K × n bytes ≈ 28 GB at K=10 —
   infeasible. At K=1: 2.8 GB — tight but possible.

2. **--accumulate mode + L-BFGS**: AGPT's existing `--accumulate` sums
   gradients over a whole epoch and fires ONCE. With 1 fire per epoch,
   consecutive fires share trajectory; L-BFGS history is meaningful.
   But this loses AGPT's per-partition optimizer-fire structure (which
   is the key insight from the partition-depth work). Might still be
   worth testing as a hybrid.

3. **Fix gamma scaling**: replace γ = ys/yy with γ = 1 (always identity
   H_0). Loses the "calibration" property of L-BFGS but might avoid the
   collapse. Worth a single experiment.

4. **Compensate**: scale d_q up by 1/lr after the two-loop to restore
   the "expected step ≈ lr" property. Hacky but effective.

### Recommendation

L-BFGS as currently implemented is a dead end for AGPT pd>1. Worth
ONE more experiment before fully closing: option 3 (γ=1 fixed) at
pd=6 3 SE — just to check if the collapse vanishes. If yes, that's a
working L-BFGS variant. If not, defer indefinitely in favor of other
optimizer ideas (like the user's top-k subtree training, which doesn't
have this assumption mismatch).

Findings 2 (cross-call persistence) and 4 (weight-decay s_k inconsistency)
are still unfixed but are no longer the priority — they're issues for
when L-BFGS actually works.

---

## Update 2026-05-03 (γ=1 experiment, FINAL)

Tried `AGPT_LBFGS_GAMMA_ONE=1` env var to force γ=1.0 (skip the
1/(ρ·yy) scaling that was collapsing to lr).

| recipe | PPL@16 |
|---|---:|
| pd=6 K=10 lr=1e-3 γ=1 | NaN |
| pd=6 K=10 lr=1e-4 γ=1 | NaN |
| pd=1 K=10 lr=1e-3 γ=1 | NaN |
| pd=1 K=10 lr=1e-4 γ=1 | NaN |

**All four NaN.** γ=1 unmasks the divergence that γ-collapse was hiding.
The two-loop's H * g direction (built from orthogonal-gradient history
pairs) is unreliable — when γ-scaled small, it's a no-op; when γ=1, it
explodes.

**Two failure modes, no middle ground:**
- γ-scaling collapses → no-op (PPL ~157)
- γ=1 forced → full magnitude H * g direction is destabilizing → NaN

### Final verdict — L-BFGS closed

L-BFGS requires that consecutive gradient samples come from a coherent
loss surface (the algorithm's K-history pairs are meaningful only when
they describe the same function's curvature). AGPT's per-partition fire
structure intentionally samples DIFFERENT surfaces (each partition group
is a distinct prefix-conditioned distribution). The two are mathematically
incompatible in their core assumptions.

**To make L-BFGS work for AGPT, one of:**
1. Restructure AGPT to fire one optimizer step per epoch (--accumulate),
   making consecutive fires share the same surface. But this loses the
   per-partition optimizer-fire structure (which is the source of pd=6's
   gains).
2. Maintain per-partition L-BFGS history. At pd=6, that's 283k separate
   K=10 histories = ~28 GB. Infeasible.
3. Accept L-BFGS's no-op behavior and use it as a regularizer rather
   than primary optimizer. Doesn't justify the complexity.

**None of these are practical.** Closing L-BFGS as a dead end. Implementation
remains in the codebase (gated by `--optimizer lbfgs`) for future
reference but is not a recommended recipe option.

For the "second-order curvature would help AGPT" intuition that motivated
this investigation: the user's **top-k subtree training** idea is more
promising. Top-k operates within AGPT's existing per-partition structure
without requiring cross-partition gradient coherence.

### Alternative reorganization (idea, 2026-05-03 — not pursued)

User suggested: train by **suffix token** rather than by prefix depth.
Partition events by their target c_p (V=65 partitions, one per
vocabulary token) instead of by N-char prefix.

Within each suffix-token partition: events share the same TARGET but
have different prefixes. Per-target L-BFGS history would be coherent
(each fire pushes the same output direction). Memory cost: 65 × K × n
× 4 ≈ 280 MB at K=10 — tractable.

**Pros**: gradients within a target-partition are more correlated
(all push W_unembed[target]); per-target K-history records meaningful
curvature; L-BFGS would have a chance of working.

**Cons**: abandons AGPT's distributed-target framework (each fire
becomes single-target CE; equivalent to standard supervised training
grouped by output class); we measured `--ce-only` costs ~0.4 PPL at
pd=6 and ~1 PPL at pd=1; significant code reorganization (trie
traversal, partition structure, chunk batching) — probably a week.

**Verdict**: filed as "interesting alternative if the fold direction
hits a wall." Not the next experiment. Suffix-token-grouped training
would essentially abandon AGPT for an L-BFGS-friendly alternative,
which only makes sense if the resulting PPL is substantially better
than the current pd=6 + Adam recipe (3.30 at 120 SE on Shakespeare).
