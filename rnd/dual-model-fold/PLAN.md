# Dual-Model AGPT with Unified Prefix/Suffix Tree

**Branch:** `dual-model-fold`
**Status:** design

## Context

The cap-folding experiment (`rnd/cap-folding/`) shipped target-substitution
fold as a small regularizer (~-1 to -2% PPL on Shakespeare 1M d=32) but
explicitly does not address the seq_len-extension goal that motivated
folding in the first place. The user's original intent — fold-as-loop
to bridge prefix and suffix views — requires a structurally different
mechanism.

Measurement (`notes/agpt/prefix-suffix-model-divergence.md`):
independently-trained forward F and backward B AGPT models on the same
Shakespeare 1M corpus disagree by **KL = 2.38 nats** per held-out
position despite the underlying trie distributions agreeing perfectly
(KL=0). That 2.4-nat gap is the per-position information asymmetry
between prefix and suffix evidence — the gap any meaningful "fold"
must close.

This experiment builds the dual-model trainer that explicitly couples
F and B via a KL_suffix consistency term during training, on top of a
unified prefix/suffix-tree data structure that physically pairs the two
views per corpus position.

## Goal

Train F and B jointly on a unified tree such that:

1. The 2.4-nat F-vs-B divergence shrinks meaningfully (target: < 0.5
   nats — a 5× reduction would be strong evidence the coupling works).
2. Held-out PPL of either model alone improves (each gets the benefit
   of the other's view as regularization).
3. The ensemble F+B (Bayesian-inverted at inference) outperforms either
   alone. This is the deliverable that distinguishes "real fold" from
   "side-table fold."

If (1) closes but (2)/(3) don't improve, the coupling is structurally
present but the corpus's per-position information asymmetry is genuine
and not bridgeable beyond a certain level.

## Architecture

### Unified prefix/suffix tree

A single data structure per corpus, replacing the current pair of
separate prefix-trie + suffix-trie files. Each node corresponds to a
substring of the corpus and carries both forward (out) and backward
(in) views.

**Per-node payload:**

| field | type | meaning |
|---|---|---|
| `token_id` | int32 | token at this position in the substring |
| `out_counts` | sparse (token_id, count) | next-char counts (forward) |
| `in_counts`  | sparse (token_id, count) | preceding-char counts (backward) |
| `out_children` | edge pointers | radix-compressed forward edges |
| `in_children`  | edge pointers | radix-compressed backward edges |
| `out_mass`, `in_mass` | int32 | mass at this node from each side |

The forward and backward radix structures are independently compressed
(unary chains in one direction may not be unary in the other), so the
same physical node can be a branching point in one direction and inside
a unary edge in the other.

**Key invariant** that the user observed (`notes/agpt/prefix-suffix-fold-architecture.md`,
§1.5): every node represents a unique corpus substring, and the node's
in_counts and out_counts are marginals of the same underlying joint
P(c_before, substring, c_after). This is structurally guaranteed by
construction; we don't need to enforce it.

**Wire format:** new file format `unified_tree_meta.bin` +
`unified_depth_NNN.bin` files. Each record extends the existing
`RadixTrieReader::LoadedRecord` struct with `in_counts` and `in_edge`
fields.

**Builder:** new tool `bin/agpt_build_unified_tree`. Takes corpus + max-depth,
builds both forward and backward views in one pass over the corpus,
emits unified file format. Memory cost: ~2× current radix trie (since
we're storing both views), so ~300 MB for Shakespeare 1M at d=32.
Reasonable.

**Reader:** new class `UnifiedTrieReader` that exposes both
`forward_walk(W)` and `backward_walk(W)` plus `dual_node_at(corpus_pos)`
which returns the (in, out) pair at that position.

### Trainer architecture

Two separate `MiniGPT` instances loaded simultaneously (F and B). No
shared parameters. Each has its own Adam state.

**Per-event flow** (pseudocode):

```
for each AGPT partition group g (visiting unified-tree nodes):
    F.zero_grads(); B.zero_grads()

    for each query q in g:
        prefix     = corpus[q.pos - seq_len .. q.pos - 1]
        suffix_rev = reverse(corpus[q.pos + 1 .. q.pos + seq_len])
        target     = corpus[q.pos]

        P_F = F.forward(prefix)        # predicts target
        P_B = B.forward(suffix_rev)    # predicts target

        ce_F = CE(target, P_F)
        ce_B = CE(target, P_B)
        kl_F = KL(stop_grad(P_B) || P_F)   # F is pulled toward B
        kl_B = KL(stop_grad(P_F) || P_B)   # B is pulled toward F

        F.grad += ce_F + β · kl_F
        B.grad += ce_B + β · kl_B

    F.adam_step()
    B.adam_step()
```

The `stop_grad` is critical: each model's KL term uses the OTHER
model's prediction as a fixed teacher. Symmetric KL (both directions)
ensures neither is privileged. The two models pull each other toward
agreement, not toward one direction.

**Why both queries fire at the same corpus position:** the unified
tree's natural training unit IS a corpus position, with both views
co-located. Each event computes both losses and aggregates per
partition exactly like single-model AGPT. F and B see identical
event-counts and identical partition boundaries.

**Hyperparameter β (KL weight):** start at 0.1 per the architecture-notes
recommendation. Sweep {0.0 = no coupling baseline, 0.01, 0.1, 1.0} once
the trainer works. Higher β risks dominating CE; β=0 gives back two
independent models.

### Memory budget

At Shakespeare 1M d=32, d_model=64, n_layers=2:

| component | single-model | dual-model |
|---|---:|---:|
| weights | ~430 KB | ~860 KB |
| Adam state | ~860 KB | ~1.7 MB |
| K/V cache | 569 MB | 1138 MB |
| forward-pass buffers | ~3 GB | ~3 GB (shared) |
| trie (unified) | ~150 MB | ~300 MB |
| **total** | ~3.7 GB | ~5.4 GB |

Fits in 8 GB GPU comfortably for our default architecture. At larger
architectures (d=96 n_layers=6) we'd be tight — would need to be careful
about buffer sharing.

### Compute cost

Per training step: 2× forward, 2× backward, single Adam-step-per-model.
Wall-time per SE roughly 2× single-model. At our Shakespeare 1M
baseline of ~215 s/SE, dual-model SE would be ~430 s. 6 SE = 43 min.
Tractable.

## Implementation phases

### Phase 1: Unified tree data structure + builder + reader

- [ ] `src/agpt/unified_tree_reader.cr` — Crystal class, mirrors
      `RadixTrieReader` but with dual in/out fields per record
- [ ] `src/tools/agpt_build_unified_tree.cr` — builder, walks corpus
      to construct both views in one pass
- [ ] Test: round-trip — build, load, verify each node's in_counts +
      out_counts sum to the same total mass (corpus-position count)
- [ ] Test: at a known substring W, `forward_walk(W).out_counts`
      should match what current `agpt_build_radix_corpus` produces;
      `backward_walk(reverse(W)).in_counts` should match the
      `--reverse` build's analog

### Phase 2: Dual-model trainer

- [ ] `src/cuda/agpt_dual_train.cu` — new trainer entry point.
      Loads unified tree, two models, two Adam states. Reuses kernels
      from `agpt_train.cu` for individual model forward/backward.
- [ ] New kernel: `agpt_dual_loss_per_query_kernel` — extends
      `agpt_loss_per_query_kernel` to take both models' logits and
      compute the joint loss including KL terms
- [ ] Per-pass dual forward (run F on prefix, B on reversed suffix,
      get both logits)
- [ ] Per-pass dual backward (gradients through both networks
      independently)
- [ ] CLI: `bin/agpt_dual_train --corpus PATH --unified-tree PATH
      --kl-beta F [...standard agpt_train flags]`

### Phase 3: Inference / evaluation

- [ ] `bin/prefix_suffix_compare` (already shipped from cap-folding
      work) — measure post-training KL gap to confirm convergence
- [ ] New tool: `bin/dual_ensemble_perplexity` — computes ensemble
      PPL via Bayesian inversion of F and B at each position. Compare
      to F alone and B alone.

### Phase 4: Verification & sweep

- [ ] **Sanity** — train at β=0 (no coupling) and verify the result
      matches independent F + independent B from prior work (they
      should reproduce the 2.38-nat divergence). This validates the
      dual trainer doesn't accidentally couple at β=0.
- [ ] **Coupling sweep** — train at β ∈ {0.01, 0.1, 1.0}. Measure:
  - Final F-vs-B KL (the 2.4-nat gap; does it shrink?)
  - F-alone PPL, B-alone PPL, ensemble PPL
  - Wall-time
- [ ] **Pick β* from the sweep**. Use that β for the headline run.
- [ ] **Headline experiment** — 6 SE dual-model at β*, compare to:
  - Independent baseline_6se (F-alone, no coupling)
  - Cap-fold variant fold_orig_6se (target-substitution version)
  - Best PPL from cap-folding work (4.728 at 12 SE)
- [ ] **Effective-context probe** — same as D.4 in cap-folding,
      see if dual training changes the seq_len profile

## Open design questions

1. **Should KL_suffix terms apply at every event or only at branching
   points?** Architecture notes don't say. Applying everywhere is
   simplest. Applying only at branching nodes (skip cap intermediates)
   would couple less aggressively. We saw earlier that suppressing
   training at cap intermediates hurts; so coupling there might
   matter too. Default: apply everywhere.

2. **Symmetric KL or one-sided?** Architecture notes prescribe
   symmetric (both `KL(B||F)` for F and `KL(F||B)` for B). Default:
   symmetric. One-sided might be a cheaper variant to test if
   compute is tight.

3. **Should the KL term mass-weight match each event's loss
   weight?** Currently AGPT's loss has optional mass-weighting (corpus
   frequency). The KL term should follow the same weighting for
   consistency. Default: yes, KL term inherits mass-weighting.

4. **Initialization** — should F and B start from the same random
   weights, or different ones? Different gives them more independent
   factorings; same gives them shared starting point. Default:
   different (use distinct seeds), since independence of factorings
   is the whole experimental premise.

## Risks

| risk | likelihood | mitigation |
|---|---|---|
| KL term dominates and both models collapse to a single learned distribution | moderate | Start β small (0.01), sweep up |
| KL gradient is too noisy and slows convergence | low-moderate | Use stop_grad strictly; verify gradient direction |
| Memory blows past 8 GB at default arch | low | Measured ~5.4 GB above; cushion exists |
| Unified tree build is slow | low | Single pass over corpus, similar to current builder |
| No PPL improvement despite KL gap closing | moderate | Documents as null result; the corpus may genuinely have irreducible per-position asymmetry |
| The fold mechanism really needs forward-pass loops, not just KL coupling | moderate | This is the bigger architectural question; addressing it would be Phase 5 if Phase 4 plateaus |

## Out of scope (for this branch)

- **Forward-pass loops / path extension.** This experiment couples F
  and B at the loss level; it does not extend training paths past d.
  Loop-based path extension is a separate larger architectural
  change.
- **Wrap-around / position encoding for d > seq_len.** Same as above;
  not addressing the long-seq question.
- **Larger architecture / corpus.** Stay at d_model=64 n_layers=2,
  Shakespeare 1M, d=32 for the headline experiment. Scale up as
  Phase 5 if results warrant.
- **Cap-fold + dual-model combined.** The cap-fold mechanism (target
  substitution at caps) and dual-model coupling are independent.
  Combining them might compound or interfere; out of scope here.

## Estimated effort

| task | time |
|---|---|
| Unified tree builder + reader | 1 day |
| Dual trainer (CUDA) | 2-3 days |
| Verification + sweep | 1 day |
| Headline + writeup | 1 day |
| **Total** | **~5-6 days focused work** |

Significantly more than cap-folding's half-day. Worth it because the
result has clean interpretation regardless of outcome — either we close
the 2.4-nat gap (positive: coupling works), or we don't (negative: gap
is irreducible, and that's a structural truth about prefix↔suffix
asymmetry in language).

## Pointers

- `notes/agpt/prefix-suffix-fold-architecture.md` — original design
  (§3.1, §3.3 spec the dual-model trainer)
- `notes/agpt/prefix-suffix-model-divergence.md` — the 2.4-nat
  measurement this experiment targets
- `rnd/cap-folding/README.md` — what we already shipped (target-sub
  fold) and how it differs from this work
- `bin/prefix_suffix_compare` — measurement tool, will be reused
  for verification
