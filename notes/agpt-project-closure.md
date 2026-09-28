# AGPT Project Closure

**Date:** 2026-06-11
**Status:** Closed. The framework as designed does not deliver on its core promise at the scale and conditions tested.

This note is the project's closing memo. It is the canonical reference for "what did AGPT turn out to be?" — written for future agents (including a returning Claude) so they can pick up the project's conclusions cold without re-deriving them.

## What AGPT was meant to be

A framework, not a model. The framework defines:

1. **Trie aggregation.** Build a prefix trie of the training corpus. For each unique prefix path, sum the gradient contributions of all corpus occurrences of that prefix into a single node-level gradient.
2. **Gradient factorization.** Each trie node's gradient is computable from the parent's hidden state and the edge transitions, so the whole loss is differentiable end-to-end via a single trie traversal per epoch.
3. **f_θ is the hole.** Any per-step recurrence can plug in: linear, tanh-Elman, GRU, attention, etc.

The mathematical promise:
- **SGD-equivalence.** The trie's count-weighted gradient sum equals what plain SGD over all corpus positions would compute. So AGPT is "SGD with a different compute strategy," not a different optimization target. See `feedback_agpt_sgd_equivalence.md`.
- **Compute savings.** The trie de-duplicates work: a prefix appearing m times in the corpus is processed once instead of m times. For corpora with high prefix redundancy (most natural language), this is a sizeable theoretical win.

The empirical hope: this would let AGPT either (a) reach the same PPL as vanilla SGD with much less compute, or (b) reach a *lower* PPL by exploiting structure SGD can't see.

Neither materialized on Shakespeare at d=64 L=2.

## The bar that matters

Classical **Kneser-Ney smoothing** is the long-standing baseline for char-level Shakespeare. KenLM modified-KN sweep on the `data/.splits/4fa9aec1db6b3aea` carved heldout (see `rnd/kn-4fa9aec/kn_sweep.txt`):

| KN order | PPL |
|---:|---:|
| 5 | 4.43 |
| 6 | 4.23 |
| **7** | **4.19** ← plateau |
| 8 | 4.21 |
| 9 | 4.22 |

KN order 7 lands at **PPL 4.19**. This is the bar: any neural method on the same data and protocol that doesn't beat ~4.2 is, on the merits, no better than a 1980s-era smoothing technique.

## What we measured

All numbers on the same carved heldout (`data/.splits/4fa9aec1db6b3aea/heldout_chunks`), in the rolling-per-token family of evals (within ~5% of each other across KenLM PPL, lm-eval canonical byte_PPL, and Crystal seq=8 sliding window):

| Method | PPL | Wins KN? |
|---|---:|---|
| **KN order 7** | **4.19** | — |
| Vanilla SGD attention seq=32 ep50 | **4.17** | ties |
| Vanilla SGD attention seq=16 ep10+ | 4.45 | no |
| Vanilla GRU LM ep50 | 4.53 | no |
| Canonical AGPT pd=1 attention seed+wrap 512ep | 4.85 | no |
| AGPT pd=6 (legacy stale path, 25 SE) | 6.0+ | no |
| Mixture-overlay GRU ep10 (KN-style learnable λ) | 5.86 | no |
| Overlay-GRU unweighted ep10 | 6.00 | no |

**Only vanilla attention with seq_len=32 (mini-batch SGD, no trie, from random init) matches KN.** Every AGPT variant tried — pd=1, pd>1 stale, overlay-AGPT in three forms — loses to KN by 0.7-1.7 PPL.

## Why the framework didn't deliver

Three distinct failure modes, each independently sufficient:

### 1. Optimizer mismatch

AGPT's trie aggregation computes an *exact* full-corpus (or full-subtree under pd=1) gradient per update. That's effectively very large-batch SGD. The empirical large-batch generalization gap (Keskar 2017 and many followups) is well-documented: removing the gradient noise that small-batch SGD relies on for implicit regularization biases the optimizer toward sharper minima that train well but generalize worse.

Adam was designed under the assumption that the true gradient is too expensive to compute, so you use noisy stochastic estimates. Every design choice — second-moment normalization, momentum, weight decay calibration — flows from that assumption. AGPT inverts the assumption: the trie makes the exact gradient cheap. Adam doesn't know how to exploit that precision; it just gets a single big "batch" per pd=1 update and suffers the generalization gap.

The pd-sweep finding from April 2026-04-30 actually fits this: pd=6 (massively more updates, each on tiny subtrees → high gradient noise) outperformed pd=1 (4.19 KN-equivalent on PPL@32; though we couldn't reproduce that on canonical-PPL today, see "Numbers that didn't survive" below). What was being measured was the optimizer's preference for noisy mini-batch regime, not the trie's structural value-add.

See `feedback_optimizer_mismatch_for_exact_gradient.md`.

### 2. Mass double-counting in sampled training

When you switch from deterministic trie traversal to sampled corpus windows ("AGPT applied to samples"), the sampler already supplies the mass: a prefix appearing m times in the corpus naturally shows up m times in random sampling over an epoch.

If you also weight the gradient by m (or log m, as we did in the trust-weighting variant), you're counting mass twice. Net effect: frequency-squared overweighting of common patterns.

This was identified late in the project. The clean rule:

> Mass belongs in the forward pass, not the gradient.

The mixture variant (`agpt_train_overlay_mix.cr`) implements this correctly — mass enters via `score_j = lambda_proj · h_j + lambda_alpha · log(m_j)` and softmaxes to λ_j weights that mix per-depth predictions. This is Kneser-Ney discounting made learnable.

It worked as designed but didn't help enough — heldout PPL 5.86, still well above KN's 4.19.

### 3. Depth ≠ context

We discovered late that "trie depth" and "model context length" had been conflated in early framing. The trie of depth d gives the model d-character rooted prefix contexts during training. At inference, the model can use up to d characters of context (or up to whatever RoPE/positional encoding it has).

The April pd=6 result (3.82 PPL@32) had been read as "the trie aggregation at depth 6 is winning." On closer inspection, depth=32 trie was used and the eval was at seq=32 — so what was actually winning was **the longer attention context window**, not the trie's deeper aggregation.

Vanilla SGD at matching seq=32 (4.17 canonical) confirms this: longer context wins, the trie isn't doing the work.

## What got tried (so future agents don't redo it)

This is the list of variants explored, with their failure modes, so the project can avoid relitigating them.

### Direct AGPT variants on attention f_θ
- **pd=0** (one Adam step per epoch): catastrophically under-trained, 6.4+ canonical at 512ep
- **pd=1** (~65 Adam updates per epoch, root-child partitions): canonical 4.85 — this is the published AGPT result. Beaten by vanilla SGD at seq=16 (4.45) and KN (4.19) on the same data and protocol.
- **pd>1 with anc-grad**: explicitly forbidden by the v2 trainer; cross-group cache staleness would confound the new gradient flow.
- **pd>1 with legacy --no-accumulate --ablate-anc-grad** (the path that produced the April 3.82 PPL@32): on canonical lm-eval at d=64/trie-depth=16, gets to 6.0+ canonical at 5 SE and *increases* with more training (clear overfitting). The original April result was on a depth=32 trie with eval also at seq=32, which our reproduction showed was the attention-context effect, not the trie depth.
- **wrap (virtual cycles, root-loop extension)**: the canonical 4.85 result used wrap; comparison to no-wrap at 256ep showed wrap-no-wrap difference is within noise (4.94 vs 4.96).

### Cheap-f_θ variants (recur strand, GRU and cousins)
See `project_recur_agpt.md` for the full table. Linear, tanh, GRU, GRU+RoPE, GRU+SinPos, GRU+wrap. None beat vanilla GRU LM at the same architecture.

### AGPT-on-samples (overlay variants, the final pass)
- **Overlay-GRU, unweighted** (`agpt_train_overlay_gru.cr` without `--trie`): per non-overlapping window of length T+d, every (target position × backoff depth) pair gets a separate rooted GRU walk and contributes an unweighted CE loss term. Per-window single Adam step. Heldout PPL 6.00 — better than AGPT pd=1's per-protocol equivalent, but worse than vanilla GRU LM and well above KN.
- **Overlay-GRU, trust-weighted** (`agpt_train_overlay_gru.cr` with `--trie`): same structure but weights each (target, j) gradient by `log(m) / (1 + H)`. **Has the mass-double-counting bug**: heldout 6.16, *worse* than unweighted.
- **Mixture-overlay GRU** (`agpt_train_overlay_mix.cr`): the closing variant. Mass enters the forward pass only via the score net `s_j = lambda_proj · h_j + lambda_alpha · log(m_j)`, λ_j = softmax over j, mixture `p_final = Σ λ_j p_j`. Single loss per target. Heldout PPL 5.86 (mixture eval) — best AGPT variant, still 1.7 below KN.

### Architectural extensions (Codex strand)
- **Cap-recurrence** (`project_cap_recurrence_null.md`): null across 3 forms tested.
- **Precondition** (`project_precondition_closed.md`): null at all tested scales.
- **Slot-selection Step 0** (`project_slot_selection_step0.md`): closed for structural incompatibility with pd=1.
- **Masked-LM combiner** (`project_masked_lm_combiner_probe.md`): closed null.
- **Phase tree** (`project_phase_tree_works.md`): the only architectural extension that moved the needle (~3.9 PPL on a non-canonical protocol). But when we tried to put it on canonical lm-eval today, no run in `rnd/` reproduces sub-4 canonical PPL — best phase tree canonical is 4.50. The 3.9 number was on a protocol that didn't survive scrutiny.

## Numbers that didn't survive scrutiny

Several long-held "AGPT does well" numbers turned out to be protocol artifacts, leaks, or non-reproducible when we put them on canonical lm-eval:

- **"AGPT pd=6 → 3.82 PPL@32" (April 2026-04-30):** trained AND evaluated on the same `data/input.txt` (no carved heldout); the 3.82 was likely train-fit, not generalization. Today's reproduction on the carved split lands at ~6 canonical and overfits as training proceeds (the legacy `--no-accumulate --ablate-anc-grad` path needed for pd>1 has documented K/V staleness).
- **"Phase tree → 3.9 PPL" (June 2026-06-04):** on a non-canonical protocol. Best canonical phase tree number we could find or reproduce is 4.50.
- **"Wrap-around is a meaningful improvement":** at 256ep wrap (4.94) vs no-wrap (4.96) shows it's within noise.
- **"AGPT beats KN" (rnd/scale-vs-kn, 2026-05-23):** the win was on Gutenberg 5M at L=6 200 SE; the README itself noted "the improvement is from scale, not from any architectural intervention" (L=2→4→6 gave 7.29 → 3.99 → 3.75). And on a different eval protocol than the one used for AGPT-vs-vanilla today.

In each case, the original number was honest within its protocol — but the protocol wasn't the canonical one we ended up needing for cross-method comparison.

## What WOULD be worth trying (if anyone picks this back up)

The framework isn't logically refuted — only empirically demonstrated to fail at the scale and configuration tested. Three avenues that could plausibly change the conclusion:

1. **AGPT + L-BFGS or natural gradient.** The optimizer-mismatch hypothesis predicts that quasi-Newton or natural-gradient methods, which assume exact-gradient inputs, would exploit AGPT's trie-aggregated gradient much better than Adam does. Not tested today.
2. **Larger model and corpus scale.** The large-batch generalization gap shrinks with model and data size. At d_model > 1024 and corpora > 100M tokens, the trie's compute amortization might pay off and the generalization gap might vanish. Untested.
3. **AGPT as preconditioner for plain SGD.** Use the exact-gradient computation to compute per-parameter learning rates (Fisher diagonal or similar), then apply many small mini-batch SGD steps. Best of both worlds, in principle.

None of these were tried because we ran out of headroom to keep iterating on a framework that wasn't beating classical baselines at the small scale.

## Files

Trainers (Crystal):
- `src/tools/agpt_train_recur_gru.cr` — AGPT with GRU f_θ (recur strand)
- `src/tools/agpt_train_recur_gru_wrap.cr` — k=2 detached wrap-around
- `src/tools/agpt_train_gru_lm.cr` — vanilla GRU LM baseline (no trie)
- `src/tools/agpt_train_overlay_gru.cr` — sample-overlay AGPT (with optional trust weighting)
- `src/tools/agpt_train_overlay_mix.cr` — sample-overlay AGPT with learnable λ-mixture

Trainer (Python):
- `src/tools/agpt_vanilla_attn_train.py` — vanilla mini-batch SGD on canonical AGPT attention architecture; this is the one that ties KN at seq=32

Eval:
- `bin/agpt_recur_perplexity` — Crystal sliding-window eval for `.recur` formats
- `src/tools/sliding_window_ppl.py` — Python equivalent for HF-loadable `.model` checkpoints
- `src/tools/agpt_lm_eval.py` — canonical lm-eval-harness wrapper
- `src/tools/kenlm_baseline.sh` — KN baseline via KenLM

Run directories of record:
- `rnd/vanilla-attn-torch/d64L2-seq{8,16,32}-scratch/` — the vanilla attention runs
- `rnd/overlay-gru/d64-nt16-d8-trust/` and `rnd/overlay-mix/d64-nt16-d8/` — the final overlay-AGPT runs
- `rnd/kn-4fa9aec/kn_sweep.txt` — the canonical KN baseline on our heldout split
- `rnd/baseline-calibration-v2-static-sampled/20260531T021212-d64l2-depth16-pd1-adam-lr0015-512ep-wrap/` — the canonical AGPT 4.85 reference (verified-reproducible)

Memory entries (per-project memory, not in-repo):
- `project_vanilla_attn_beats_agpt.md` — the empirical finding with the consolidated table
- `feedback_optimizer_mismatch_for_exact_gradient.md` — the optimizer-mismatch principle
- `feedback_agpt_sgd_equivalence.md` — the design principle the framework rested on
- `feedback_agpt_is_a_framework.md` — the distinction between framework-level and f_θ-level claims
- Various closed-strand notes (`project_cap_recurrence_null.md`, etc.)

## Closing

The framework had a clean mathematical foundation, an interesting design space, and a principled SGD-equivalence guarantee. Empirically at the scale and conditions we tested, it didn't beat classical KN smoothing — and the gap was large enough (0.7-1.7 PPL across every variant) that the failure mode is structural, not a matter of tuning. The most defensible thing the project produced was a clear demonstration that **char-level Shakespeare is harder for neural methods to beat with frequency-statistical baselines than the field's optimism would suggest** — vanilla attention at seq=32 just ties KN, and AGPT in every form we tried doesn't reach that bar.

For anyone returning to this project: read `feedback_optimizer_mismatch_for_exact_gradient.md` first. The framework's theoretical promise is intact; what failed is the execution path that uses Adam to ride that promise. If you have a quasi-Newton or natural-gradient optimizer handy, the picture might look different.
