# Dual-View Consistency Training (Forward + Backward Models)

**Branch:** `dual-model-fold`
**Status:** design (revised after review #1 + AGPT-fit reconsideration)

## What this experiment is

Two independent models train simultaneously on the same corpus:
- **F** sees the prefix before each target token.
- **B** sees the reversed suffix after each target token.

Both predict the same target c_p. They're coupled by a symmetric
stop-gradient KL consistency loss that pulls each toward the other's
prediction. The experiment measures whether explicit cross-direction
coupling can shrink the **2.38-nat per-position F-vs-B divergence**
measured in `notes/prefix-suffix-model-divergence.md`.

## What this experiment is NOT

- Not fold-as-loop or path extension — does not reach beyond the
  trained seq_len.
- Not a wrap-around or position-encoding generalization mechanism.
- Not strict AGPT in the "distribution-CE-per-trie-event" sense — see
  "Relationship to AGPT" below.

## Relationship to AGPT — important clarification

The original architecture notes (`prefix-suffix-fold-architecture.md`)
specified per-corpus-position CE (`ce_F = CE(c_p, Q_F)`, c_p a single
token), NOT trie-aggregated distribution-CE. So:

- **What's preserved as AGPT-flavored:** per-partition gradient
  batching, one Adam step per partition, deferred optimizer fires.
  This is the operational AGPT property that matters for gradient
  variance and convergence dynamics.
- **What's NOT preserved:** the existing AGPT trainer's per-trie-event
  distribution-CE (where the target at a branching node is the
  empirical multinomial over children). With per-position dual
  coupling, each event is at a corpus position with a one-hot CE
  target. This is closer to SGD-with-batching than to AGPT-with-
  trie-aggregation.

So calling this "dual-model AGPT" overclaims. It's **dual-view
consistency training** with AGPT-style partition batching for the
Adam-fire schedule. The cap-folding work we already shipped (target
substitution at trie events) genuinely IS AGPT-with-coupling because
it adds a term to the existing distribution-CE loss; the dual-model
work is a structurally different training formulation.

## Hypothesis

The 2.38-nat F-vs-B divergence comes from each model learning a
different lossy compression of the same corpus joint distribution.
Coupling them via KL_suffix during training should:

1. (Tier 1) Shrink the F-vs-B KL gap meaningfully.
2. (Tier 2) Improve F-alone causal PPL via cross-direction
   regularization.
3. (Tier 3) Make the F+B ensemble outperform either model alone.
4. (Tier 4) The improvement should require *aligned* prefix/suffix
   pairing — a shuffled-suffix control should NOT show similar
   benefit. If shuffled also helps, the KL is just a generic
   regularizer.

## Training objective

For each corpus position i:

```
prefix     = corpus[i - seq_len .. i - 1]
suffix_rev = reverse(corpus[i + 1 .. i + seq_len])
target     = corpus[i]

P_F = softmax(F(prefix))
P_B = softmax(B(suffix_rev))

ce_F = CE(target, P_F)
ce_B = CE(target, P_B)

kl_F = KL(stop_grad(P_B) || P_F)   # F pulled toward B's view
kl_B = KL(stop_grad(P_F) || P_B)   # B pulled toward F's view

loss_F = ce_F + β_eff(step) · kl_F
loss_B = ce_B + β_eff(step) · kl_B
```

Both models receive their own gradient through their own forward pass.
The `stop_grad` on the teacher-side is critical — without it, the
gradient flows through both models simultaneously and the KL term
becomes a tangle.

The KL gradient is simple in closed form:

```
∂kl_F / ∂logits_F = β · (P_F - P_B)
∂kl_B / ∂logits_B = β · (P_B - P_F)
```

When the teacher distribution is detached.

### β warmup

Early in training both models are noisy teachers. Pulling each toward
the other's noise actively hurts. Use linear warmup:

```
β_eff = β_max · min(1, step / warmup_steps)
```

Default `warmup_steps` = total_partition_groups (one full SE of warmup).

## Training loop

```
for SE in 1..n_epochs:
    partitions = group_corpus_positions_by_depth_d_prefix(corpus, partition_depth)
    for partition g in partitions:
        F.zero_grad(); B.zero_grad()
        for position i in g:
            # forward both models
            P_F = F(corpus[i-seq_len..i-1])
            P_B = B(reverse(corpus[i+1..i+seq_len]))

            # accumulate joint loss
            ce_F = CE(corpus[i], P_F);  ce_B = CE(corpus[i], P_B)
            kl_F = KL(stop_grad(P_B) || P_F)
            kl_B = KL(stop_grad(P_F) || P_B)

            F.grad += ce_F + β_eff · kl_F
            B.grad += ce_B + β_eff · kl_B

        F.adam_step()  # one Adam step per partition for both models
        B.adam_step()
```

Partition grouping uses the same `--partition-depth` semantics as
existing AGPT — positions with identical depth-N prefix go into the
same partition. Adam fires per partition. This is the
"AGPT-flavored" part: per-partition gradient batching.

## Architecture choices

- **Two independent models**, no shared parameters. Same architecture
  for clean comparison (d_model=64, n_layers=2, default).
- **Different random init seeds** for F and B so their independent
  factorings have full freedom.
- **Same tokenizer / vocabulary** so their predictions are over the
  same V-dim space.
- **Coupled only via KL terms.** Everything else is independent.

## Memory + compute

At Shakespeare 1M d_model=64 n_layers=2 d=32 seq_len=32:

| component | dual cost |
|---|---:|
| Two model weights | ~860 KB |
| Two Adam states | ~1.7 MB |
| Two K/V working buffers | ~6 GB at chunked forward |
| Corpus + partition table | <100 MB |
| **Total** | **~6 GB** |

Fits in 8 GB. Wall-time per SE roughly 2× single-model SGD pass
(both F and B forward+backward per event). On the order of ~10-15 min
per SE on our GPU.

## Implementation phases

### Phase 1 — new dual-trainer

- [ ] `src/cuda/agpt_dual_train.cu` — new trainer entry point.
      Loads corpus directly (not a trie). Builds partition table by
      sorting positions by depth-N prefix. Loads two models.
      Per-partition: dual forward, joint loss kernel, dual backward,
      dual Adam.
- [ ] New loss kernel: `dual_loss_per_position_kernel` — takes both
      models' logits at each position, computes joint CE+KL loss and
      writes gradient to both models' logit grad buffers.
- [ ] CLI: `bin/agpt_dual_train --model-f F.model --model-b B.model
      --corpus PATH --epochs N --partition-depth N --kl-beta F
      --kl-warmup-steps N [--shuffle-suffix] [--branch-gated]`
- [ ] Logging per partition: `ce_F`, `ce_B`, `kl_F`, `kl_B`,
      `H(P_F)`, `H(P_B)`, β_eff. Dump to a TSV alongside the standard
      SE summary line.

### Phase 2 — eval tools

- [ ] `bin/prefix_suffix_compare` (already exists from cap-folding
      work) — gives F-vs-B KL post-training. Compute pre and post,
      track Tier 1 metric.
- [ ] `bin/dual_ensemble_perplexity` — new tool. Computes ensemble
      PPL via several mixtures: arithmetic mean, logit average,
      product-of-experts, weighted product. Reports each so we can
      see which mixture strategy actually wins.
- [ ] Causal-only PPL: existing `bin/perplexity` on F alone gives
      Tier 2. Same on B alone for completeness.

### Phase 3 — verification

- [ ] **β=0 sanity** — train at β=0 from same random init as our
      existing `baseline_6se` and `backward_6se`. PPL should match
      those models within fp tolerance. Confirms the dual trainer
      has no implicit coupling.
- [ ] **Shuffled-suffix negative control** — `--shuffle-suffix` flag
      that pairs F's prefix at position i with B's suffix from a
      different random position j (target stays c_i). Train at β > 0.
      If aligned-suffix improvement vanishes under shuffling, alignment
      is the source of signal. If shuffled improves similarly,
      the KL is a generic regularizer.

### Phase 4 — sweeps + headline

- [ ] **Coarse β sweep**: β ∈ {0.0, 0.01, 0.1, 1.0}. Track Tier 1
      (KL gap) and Tier 2 (F-alone PPL) at each.
- [ ] **Refined β sweep** at the best coarse point: β ∈
      {best/3, best/2, best, 2·best, 3·best}.
- [ ] **Headline run** at β*: full 6 SE, log all diagnostics, run
      all eval modes.
- [ ] **Shuffled-suffix re-run** at β* — distinguishes Tier 4.
- [ ] **Branch-gated KL ablation** at β* — apply KL only when
      ε(in_counts) > 0.1 nats AND ε(out_counts) > 0.1 nats.
      Tests whether coupling matters at decision points only.

### Phase 5 — writeup

- [ ] `rnd/dual-model-fold/README.md` — full results, all four
      tiers, sweep tables, mechanism interpretation.
- [ ] Memory file update.
- [ ] Decision: does this branch's mechanism justify further work
      toward fold-as-loop? Or is the per-position information
      asymmetry irreducible at our scale?

## Eval modes — explicit definitions

**Causal F-alone:** Standard left-to-right perplexity. F predicts c_p
from prefix only, no suffix access. Comparable to single-model AGPT
PPL. This is the metric for Tier 2.

**Bidirectional ensemble:** P(c_p | prefix, suffix) computed by
mixing P_F and P_B at each position. Multiple mixture options:

```
arithmetic     = 0.5 · P_F + 0.5 · P_B
logit_avg      = softmax(0.5 · logits_F + 0.5 · logits_B)
product        = normalize(P_F · P_B)
weighted_prod  = normalize(P_F^α · P_B^(1-α)),  α ∈ [0, 1]
```

This sees future context, so PPL is not directly comparable to causal.
It IS comparable to MLM-style reconstruction. Tier 3 metric.

## Diagnostic logging

Per partition (or per N partitions to control log volume):

| field | meaning |
|---|---|
| `step` | partition counter |
| `β_eff` | current KL weight after warmup |
| `ce_F`, `ce_B` | mean CE per partition |
| `kl_F`, `kl_B` | mean unweighted KL |
| `wkl_F`, `wkl_B` | β_eff·kl values added to grad |
| `H_F`, `H_B` | mean prediction entropy |
| `top1_agree` | fraction of positions where F and B argmax agree |

Per SE summary:

| field | meaning |
|---|---|
| F-alone NLL on held-out | causal PPL proxy |
| B-alone NLL on held-out | for completeness |
| Symmetric KL on held-out | the 2.38-nat gap, post-SE |
| Ensemble NLL (arithmetic) | bidirectional PPL |

## Success criteria (from review)

| Tier | Criterion | Implication |
|---|---|---|
| 1 | Symmetric KL drops from 2.38 → < 1.0 nat without entropy collapse | Coupling structurally works |
| 2 | F-alone PPL at β > 0 < F-alone PPL at β = 0 | Causal model improves from cross-coupling |
| 3 | Ensemble PPL < min(F PPL, B PPL) | Bidirectional combination has real signal |
| 4 | Aligned-suffix improvement > shuffled-suffix improvement | The signal is genuine prefix↔suffix coupling, not generic regularization |

Tier 4 is the *most important* control — without it we cannot
distinguish "dual coupling reveals real bidirectional information"
from "the KL term acts like dropout."

## Out of scope

- Fold-as-loop / path extension past d.
- Larger architectures (try after Tier 2 succeeds).
- Larger corpora (Gutenberg 5M/10M; defer until Shakespeare 1M
  shows the mechanism works).
- Combining dual-model with cap-fold (target substitution at
  caps). They're independent; combining is a follow-up if both win
  individually.

## Risks (revised)

| risk | likelihood | mitigation |
|---|---|---|
| KL dominates and both models collapse to high-entropy bland distributions | moderate | β warmup; track entropy as a guard; abort if H_F or H_B grows monotonically |
| Aligned and shuffled controls perform identically | moderate | This is *information* — the KL is acting as a regularizer not a bridge. Document and adjust. |
| Per-position training is slower than expected | low | Profile; if a problem, switch to chunked positions (process N positions per kernel launch) |
| KL gap closes but F-alone PPL doesn't move | moderate | Tier 1 success without Tier 2 is interesting but limited. Document. |

## Estimated effort

| task | time |
|---|---|
| Phase 1 dual trainer | 2-3 days |
| Phase 2 eval tools | 1 day |
| Phase 3 verification | 0.5 day |
| Phase 4 sweeps + headline | 1 day |
| Phase 5 writeup | 0.5 day |
| **Total** | **5-6 focused days** |

## Pointers

- `notes/prefix-suffix-fold-architecture.md` — original design,
  §3.1 and §3.3 specify the dual-model architecture
- `notes/prefix-suffix-model-divergence.md` — the 2.38-nat
  measurement this experiment targets
- `rnd/cap-folding/README.md` — the target-substitution fold work
  that motivated this; the dual-model approach is structurally
  different
- `bin/prefix_suffix_compare` — existing tool that measures the
  F-vs-B divergence; will be reused for verification
- `rnd/dual-model-fold/PLAN_REVIEW_1.md` — external review that
  drove this revision; tier criteria, β warmup, shuffled control
  all came from there
