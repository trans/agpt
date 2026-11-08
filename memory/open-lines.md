# Open lines of research

*Updated 2026-11-07*

Each line has a state, where its plan lives, and what it needs next.

## 1. Context-tree orientation (suffix tree instead of the prefix trie)

**Open.** The experiment's front matter says `concluded`/`negative` for the one
GRU test. The todo and the user (2026-10-03) say to run it to ground.

- **Plan:** `todo/context-tree-orientation.md`, "Next session", in this order:
  1. Mechanism tests: mixed units, interference between units, and a no-trie
     control with minibatches sorted by last character, by first character,
     and at random.
  2. An attention f_θ in both orientations.
  3. 3 seeds, a warmup-cosine LR, depth 16, pd=2/3.
  4. Re-read the history: `--shuffle-order` helped about 2% at pd>1 in
     `rnd/cap-folding`.
  5. Canonical eval.
- **Tool:** `src/tools/agpt_context_tree_coherence.py` (PyTorch; exact
  level-by-level backward, checked against autograd).
- **Caveats on the first runs:** one seed, a constant LR of 3e-3, depth 8, and
  a diagnostic held-out metric. Any canonical claim needs an HF wrapper that
  re-encodes the context newest-first at each position.
- **Baseline for the recurrent A/B:** GRU AGPT on the prefix trie (tag
  `exp/linear-recurrence-final`).

## 2. Curvature that keeps partitioned cadence

**Open.** Full-trie L-BFGS works on the exact gradient but loses to pd=1 Adam at
equal traversals (gradient-population Exp 8).

- **Routes recorded there:**
  - per-subtree L-BFGS histories (65 at pd=1, about 1 GB);
  - partitioned steps preconditioned by curvature learned once per epoch from
    the full-trie gradient, which is the free sum of the epoch's unit
    gradients (stochastic quasi-Newton).
- **Seed idea from the Ultra track:** use the exact output-layer Fisher block
  from counts as L-BFGS's initial Hessian instead of γI
  (`research/ultra/docs/trie_fisher_bridge_revised.md`).
- **Unmeasured:** where L-BFGS levels off (it was still descending at 600
  passes, with no schedule), and anything beyond d64/L2.

## 3. v2 trainer speed (kernel work)

**Open, not started.** No commit to `src/cudax` or `src/cuda` since `88c9335`
(checked 2026-11-07). Per per-machine memory, the user is interested in this
kernel work as research (2026-09-28).

- **Profile** (nsys, 2026-09-28, pd=1 d64/L2 depth 16, about 5.6 s/epoch):
  the trainer is GPU-bound at a small fraction of the GPU's peak. GPU time
  splits as:

  | Share | What |
  |---:|---|
  | ~35% | tree-attention forward/backward (scalar loops, a block per query×head) |
  | ~30% | unfused elementwise ops (LayerNorm, bias/ReLU) |
  | ~12% | cuBLAS GEMMs |
  | ~11% | device-to-device "save" copies and memsets |
  | ~7.5% | bias-grad accumulation |

  The estimated gain is 3–5× per epoch, unmeasured.
- **Planned order:**
  1. Rewrite the tree-attention kernels, with the exact ancestor path designed
     in.
  2. Drop the activation save copies in `forward_stages_v2.cuh`.
  3. Use cuBLASLt epilogues for bias/ReLU/bias-grad, and a warp-shuffle
     LayerNorm.
  4. In the exact second pass, skip the output head, the loss, and the
     all-zero top-layer backward.
- **Acceptance for every change:**
  - the frozen-weight gradient dump matches the current build to about 1e-5
    (the atomics floor);
  - `agpt_grad_check.py` gives an FD ratio of 1.000 ± 0.005;
  - the L-BFGS backtrack rate stays around 1%;
  - re-profile after each step.
- Prefix-shared attention kernels exist only for inference (cascade-style or
  radix attention). No standard training kernel computes the gradient into
  ancestors.

## 4. Coherence-driven cadence, leftovers

**Open.** From gradient-population Exp 1's "Next steps" and Exp 2's "Next":

- **`train.fire_norm: sum|sqrt`:** v2 accepts only `mass` today (verified in
  `yaml_config_v2.cuh`). This is the cheapest test of the hypothesis that Adam
  is hurt by unit sizes varying widely.
- **Better split maps:** build them from event-weighted ρ, with a floor of
  about 1k events per unit.
- **Shallow targets:** train the depth<pd targets so that split plans can be
  compared on the rolling metric.
- **Dynamic re-measurement:** re-measure coherence during training to adjust
  the split.
- **Target-distribution similarity:** test it against prefix similarity as the
  thing units actually agree on.
- **The ~3% gap between pd=1 and the summed pd=2 gradients:** this was blamed
  on the truncated backward. It has not been re-measured with the exact
  backward.

## 5. Baselines with weak provenance

**Open.** These numbers are needed to frame any AGPT claim:

- **Vanilla attention, no trie** (4.45 at seq 16; 4.17 at seq 32): no
  `result.json` exists, and the seq-32 number has no surviving artifact
  (`rnd/vanilla-attn-torch` caveats). The trainer is on
  `exp/linear-recurrence-final`. The seq-16 checkpoints may survive locally
  on the machine that ran them, outside git.
- **KN order 7 on the canonical split:** two records disagree, 4.114 against
  4.19 ([measurements.md](measurements.md)).
- **The best d128/L6 result** (phase-conditioned RoPE, 4.497 rolling) predates
  both the race fix and the exact backward. A rerun would give the current
  headline at that size. This is a suggestion; nobody has decided it.

## 6. Planned, not started

- **Skip-horizon AGPT:** design note `rnd/skip-horizon-agpt` (2026-06-18).
- **Deterministic backward:** `todo/deterministic-backward.md`, **Open**.
  Forward determinism landed in `ef3a8e9`.
- **Kneser-Ney backoff in the trie-only evaluator:**
  `todo/trie-ppl-kn-backoff.md`, **Open**.
- **Measure the one-pass gap:** how close one corpus pass gets, in quality and
  in time.
