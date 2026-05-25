# KN-Distilled Soft Targets for AGPT Training — Plan

**Status:** PARKED 2026-05-24. Per-node position distributions take priority — they attack the d=16 trie ceiling directly, where this would only smooth around it. Pipeline tools (KenLM build, ARPA parser, trie context dumper) are still in tree for future use.
**Date:** 2026-05-24

## What we're testing

Replace AGPT's one-hot training targets with **KN-smoothed conditional distributions** at every endpoint context, using KenLM's modified-KN as the teacher. The trie structure stays the same; only the loss-side target changes.

For each radix-trie endpoint (representing some context `c_1..c_k`), the training loss currently uses either:
- The trie's `counts` distribution (at endpoints with mass > 1), or
- A one-hot (at mass-1 endpoints)

With KN distillation, both are replaced with `P_KN(w | c_1..c_k)` — KenLM's smoothed conditional distribution.

## Hypothesis

KN's smoothing fills in plausible-but-unseen continuations at low-mass nodes. The model should learn this smoothing structure rather than memorize the noisy one-hot signal. Expected PPL gain: **uncertain, possibly 2-5% on Gutenberg held-out**. Possibly zero — `--anc-grad` may already cover the same statistical structure from the gradient side.

## Why this is worth considering

1. **Direct attack on the noisy one-hot at low-mass leaves.** Mass-1 endpoints are the most common type in our trie (it's a power-law tail), and their one-hot targets are pure noise relative to the true conditional.
2. **Cheap teacher.** KN at order 8 trains in seconds, queries in microseconds. No expensive teacher-forward-pass.
3. **Orthogonal to anc-grad?** Anc-grad smooths along the trie axis (descendant→ancestor). KN distillation smooths along the backoff axis (long-context→short-context). Different statistical structure; could compound.
4. **Tools are already built.** KenLM is installed and tested. ARPA parser exists. Trie context dumper exists. Implementation is plumbing, not invention.

## Why it might fail

1. **Anc-grad may have absorbed all the value.** Our 8% PPL win from anc-grad is structurally similar to KN backoff (route information from longer-context-but-noisier to shorter-context-but-cleaner). Distillation might add nothing on top.
2. **KN with `--discount_fallback` isn't a great teacher.** Char-vocabs are too small for proper modified-KN; we use a single fixed discount of 0.5. The teacher is imperfect.
3. **High-mass endpoints would be degraded.** At endpoints with counts spread across many chars from thousands of observations, the trie's `counts` distribution IS the true distribution. KN's additional smoothing pulls these toward the corpus marginal — net negative. (Position-aware blending could fix this but adds complexity.)
4. **The trie's d=16 ceiling is the real bottleneck.** No amount of target-side smoothing changes the fact that we can only condition on 15 chars. The headline research direction (per-node position distributions) attacks this directly; distillation does not.

## Architecture decisions (settled)

- **Endpoint-only**, NOT all-positions. Preserves radix compression. Memory: ~1.8 GB at fp32 / 0.9 GB at fp16 for Gutenberg's 7M endpoints. Comparable to current K/V cache scale.
- **Pure KN target**, no blending knob. KN already incorporates the training observations; blending with one-hot un-smooths what KN smoothed. If pure KN underperforms, *then* reconsider position-aware blending.
- **Lazy cache** by default. Populate on first encounter of each endpoint; ~1 epoch to warm. Avoids precompute wait.
- **Runtime KN query via Crystal bindings to KenLM C++ lib.** No giant offline side-table; trainer queries on demand and caches.
- **KN order 8.** Matches the empirical plateau on Gutenberg (PPL 4.09).

## Implementation scope (~14 hours / ~2 days)

Four PRs:

1. **KenLM C wrapper + Crystal bindings** (~6h)
   - `tools/kenlm_c_wrapper.{h,cpp}` — minimal C-facing shim
   - `src/agpt/kn_model.cr` — Crystal binding class with `distribution(ctx)` method
   - Standalone test: query "the", "qu", "and" → compare to Python tool output

2. **KNCache** (~3h)
   - `src/agpt/kn_cache.cr` — hashmap, lazy populate
   - Parallel precompute path (optional via `--kn-precompute`)
   - Standalone exerciser tool

3. **Chunk metadata + loss kernel** (~3h)
   - Add `kn_targets : Float32*` to chunk packing
   - Loss kernel reads KN distribution as target (replaces existing target tensor when `--kn-targets` set)
   - `--kn-targets <arpa>` CLI flag

4. **Validation + experiment** (~2h + wall)
   - Smoke test: confirm baseline reproduction when KN cache is bypassed
   - Small-scale Shakespeare L=2 3-seed (~15 min)
   - Headline Gutenberg L=8 d=128 100 SE 3-seed (~5h pod time)

## Cost / value

| dimension | KN distillation | per-node position distributions (the other queued direction) |
|---|---|---|
| Implementation time | ~2 days | ~1 week+ |
| Memory cost | ~1 GB endpoint cache | ~600 MB sparse table + GPU-side per-batch lookup |
| Risk it adds nothing | Medium-high (anc-grad overlap) | Low (attacks a different axis: position not backoff) |
| Risk it makes things worse | Low | Low |
| Attacks the trie's d-ceiling | No | Yes |
| Generalizes beyond char-LM | Standard distillation, well-understood | Novel architecture, harder to predict |

## Decision criteria

This is a "tactical" smoothing experiment, not a strategic architectural pivot. Worth doing if:

- You want a 2-day diversion before committing to the multi-day position-distribution work
- You want a clean comparison "does anc-grad already cover this?" data point
- You believe even a small distillation gain stacks usefully with the eventual position-distribution work

NOT worth doing if:

- You're confident the per-node-position-distribution direction is the right next bet (in which case spend the 2 days getting that started instead)
- You're skeptical that target-side smoothing has any room left after anc-grad's gradient-side smoothing

## Go/no-go criteria for the experiment itself (if we proceed)

- **Smoke test must pass:** baseline-equivalent loss when KN cache is no-op'd.
- **Small-scale Shakespeare:** if `--kn-targets` produces PPL ≥ baseline + 0.05 (no improvement), abandon before Gutenberg run.
- **Gutenberg headline:** ≥ 2% PPL reduction (3 seeds, p < 0.05) to count as a win. < 2% → mark as "small effect, parked," don't merge to default recipe.

## Open questions

- Do we use the existing `counts` distribution at endpoints with mass > some threshold (as a hybrid), or always KN regardless of mass? Defer until we have first-pass data.
- Does KN order 6 or order 10 differ from order 8 in distillation utility? Cheap to test once the pipeline exists.
- Could we use the Crystal bindings + KN cache *for inference too* (KN-weighted ensemble with the AGPT model)? Out of scope for now but a natural follow-up.

## Files (when implemented)

- `tools/kenlm_c_wrapper.{h,cpp}`
- `src/agpt/kn_model.cr`
- `src/agpt/kn_cache.cr`
- `src/cuda/agpt_train.cu` (CLI + loss kernel changes)
- `notes/kn-distillation-plan.md` (this file)
- `rnd/kn-distillation/` (experiment results when run)
