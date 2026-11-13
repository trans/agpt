# Open lines of research

*Updated 2026-11-12*

Each line has a state, where its plan lives, and what it needs next.

## 1. Context-tree orientation (suffix tree instead of the prefix trie)

**Called on 2026-11-10 by the user: a minor breakthrough, with a proviso to
double-check and polish.** The prefix trie is the right orientation, because
its pd=1 units group by the least predictive character and train like shuffled
minibatches. Suffix-tree units are target-sorted. Record and numbers:
`rnd/context-tree-orientation/README.md` (Conclusion; Grouping control).

- **Proviso checks, not yet done** (the README's list, in order):
  1. seeds for the random and 2×2 cells;
  2. an attention f_θ (cheapest: the no-trie control with attention over the
     window);
  3. v2's fixed unit order (`experimental.unit_order_seed` defaults to −1);
  4. depth 16, pd=2/3, warmup-cosine;
  5. canonical eval;
  6. interference between unit steps.
- **Open puzzle from the control:** shuffled minibatches at a constant 3e-3 got
  worse after epoch 100 (train 1.846 → 1.893 at epoch 300), while
  start-position-labelled random batches did not. One seed each; the
  cosine-schedule runs avoid it.
- **Tools:**
  - `src/tools/agpt_context_tree_coherence.py`: the trees, with an exact
    level-by-level backward;
  - `src/tools/agpt_unit_grouping_control.py`: the no-trie grouping control,
    with node and event groupings and `--schedule cosine`.

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

## 3a. Multi-positional (superposed) RoPE — next direction

**Open, not started.** The user named it on 2026-11-10 as this project's next
direction, once the orientation strand closed. The other direction they named,
a Möbius count prior plus imm's neural residual models, belongs to @mobius
and @imm (see [air-projects.md](air-projects.md)).

- **Look back first (the user, 2026-11-10):** past `rnd/` experiments touched
  this, so review them before designing anything new. Found by grepping the
  front matter on 2026-11-10:
  - `rope-position-substitution` (positive, 2026-05-22)
  - `dist-rope-smoke` (negative, 2026-05-25)
  - `harmonic-filter-diagnostic` (n/a, 2026-05-25)
  - `harmonic-bias-prototype` (negative, 2026-05-29)
  - `sampled-node-phase-rope` (positive, 2026-06-05)
  - `seq-len-decouple` (negative, 2026-05-11): shared-key bookkeeping
  - the head-split occurrence-position result, which exists only in commit
    `81ed16f`'s message
  - design notes in `notes/seq-len-extension/`
- **Survey of what agpt already tried:** @imm sent one on 2026-11-10
  (`mail/received/20261110-094438Z-@imm--survey-of-agpt-s-multi-positional-rope-line-a4a30c40.md`).
  Read it before planning. Its main points, all @imm's unless marked:
  - **Things that collapse a node's occurrences into one summary broke RoPE**
    (`rnd/dist-rope-smoke`: 18–30% worse train loss). Substituting depth with
    mass or log-mass tied (`rnd/rope-position-substitution`).
  - **The chord bias was probed only on ancestor keys,** which share the
    query's position trivially (`rnd/harmonic-filter-diagnostic`).
  - **The phase-tree gain is not positional. Settled 2026-11-10** by a
    control the user chose ("D") from the RoPE review. One phase serves the
    whole epoch, and the KV cache is zeroed at each epoch start, so cached
    ancestor keys share the phase and the shift cancels in Q·K. The new flag
    `experimental.rope_phase_positions: false` drops the shift and keeps the
    phase weights. Paired d64/L2 8-epoch runs scored rolling byte PPL 10.170
    with the shift and 10.175 without, with per-epoch losses equal to about
    1e-3 (`rnd/sampled-node-phase-rope`, "Control" section). What the phase
    modes change is the loss weights. Each epoch trains on the windows
    starting in one residue class mod 64, so 64 epochs make one pass in 64
    strided minibatches, and `phase-conditioned` also takes its targets from
    that minibatch. **Open:** whether minibatching explains the d128/L6 gain
    over static (4.50 vs 4.71, unmatched, pre-race-fix). The test would be a
    post-fix d128/L6 pair, static vs phase-weighted with depth positions, on
    the same schedule; nobody has decided to run it.
  - **Never built:** shared-key RoPE
    (`notes/seq-len-extension/shared_key_rope.md`), the user's H1 matched
    filter (`harmonic-filter-plan.md`, 2026-05-25), chord forms C and D
    (`chord-position-encoding-design.md`), sliding-tree RoPE.
  - **The open question:** can one key standing for many occurrences, scored
    bilinearly with relative offsets kept, carry anything attention uses
    beyond the plain relative offset?
- **Step 1, the superposed tail (2026-11-10): inconclusive, and closed in
  this form.** The test (`rnd/superposed-tail-rope`) added a linear,
  character-keyed channel over the text older than the 16-character window.
  It went almost unused, but its control (plain softmax windows of 32 and 64)
  also showed no gain. The runs were undertrained (~20M scored targets
  against vanilla's ~170M), so the test had no power. The user called it
  worthless (2026-11-10): ordinary models are known to gain from 64
  characters, and this had been tested before. **Lesson:** check the record
  with the user before building a test, and treat a control that fails to
  reproduce a known effect as a broken setup.
- **Superposition's softmax limit: derived and measured, 2026-11-10** (the
  user asked for grounded math, then said "go ahead"). Note:
  `notes/seq-len-extension/superposition-softmax-limit.md`. Record:
  `rnd/superposition-moment-order`.
  - An exact superposed lookup needs exponential atoms λ^Δ, or exponential
    polynomials with a vector state. True wavelets break it.
  - Summing keeps the mean of the occurrence cloud on the RoPE torus. Softmax
    needs all its cumulants, and hard max needs every occurrence.
  - Measured on a regular d64/L2 model with 124-character windows: heads are
    sharp (window max weight 0.84, logit range 14), and a polynomial readout
    of their attention takes moment order 8 (p90 KL 0.012) to 16 (p99 KL
    < 1e-3, Chebyshev; 24 with centred Taylor). That is 2.7e5 to 3.9e7
    combination frequencies per vocabulary symbol (about half independent
    complex numbers, shared by the heads), against ≤ 108 occurrences. At
    order 16 the layer state is ~1.3e9 complex numbers (10 GB), and a query
    reads it all: far slower than attending over the 108 occurrences.
  - **Revised the same day, at the user's request:** the first version said
    superposed keys serve "only a linear readout". Too strong. At infinite
    precision the sum is lossless (Lindemann–Weierstrass), which is the
    user's intuition, exactly. At finite precision a nonlinear decoder is
    bounded by compressed sensing: stable recovery needs ~k log(L/k)
    measurements, so the state is never smaller than the occurrence list.
    Where it would compress (history ≳ 1,000 chars), it can't decode.
  - **Settled:** no cheap readout of a superposed RoPE state reproduces a
    sharp softmax. What survives: shared-key RoPE (compressing compute),
    exact merging (the trie), and a **polynomial positional kernel**
    (note §5). With w(Δ) = |Σ_j b_j λ_j^Δ|², softmax over every older
    occurrence is exact and recurrent in layer 1, with J complex states per
    symbol (the superposed-tail state's size), at a positional resolution of
    ~L/J. **Derived and checked numerically, not tested in training.** A
    test would need a budget at which the window-64 control shows the known
    gain.
- **Far-channel probe: concluded 2026-11-11, positive** (`rnd/far-channel-probe`).
  The user's point: blurry position is enough if it stays ordinal; they
  agreed to linear content (2026-11-10). In layer 2 of a d64/L2 model,
  beside an exact 16-char window, a channel over offsets 16–127 with linear
  content (elu+1) and a query-steered constant-Q kernel |Σ b_j λ_j^d|²
  recovers **65%** of exact softmax's gain over the window. It falls short
  of exact by 0.0104 ± 0.0009 nats on every one of 4 seeds. A recency-only
  kernel recovers 48% (cq against decay: t = −1.7, not established). The
  control (exact beats w16 by 0.030) held on every seed, and the runs were
  converged (512-char blocks, all positions scored, 30k steps ≈ 246M
  targets). **Open:** no efficiency win at reach 128 (cq's state ≈ 70k
  numbers against 14k for the keys and values); a longer reach is the
  untested case. On the trie, the channel conflicts with per-node
  aggregation (a node's occurrences share 16 chars, not their older text).
  No next step agreed.
- **PolyRoPE, the user's design (passed on by @mobius 2026-11-12,**
  `mail/received/20261112-094401Z-@mobius--the-user-s-polyrope-*.md`; the
  design doc itself is in mobius's Intake).
  - **The form the user means:** after @mobius's test of the corpus form,
    the user chose a window. Each symbol v gets
    H_v(ω) = Σ_{p in the last W} e^{iω(t−p)}, which is rotated, added to and
    dropped from at each step.
  - **The corpus form is measured and set aside.** A signature over the
    whole corpus mixes every distant pair into each frequency; bigrams at
    K = 1,024 have 20% median error, at 330× the table's storage (@mobius).
    That is the compressed-sensing bound in
    `notes/seq-len-extension/superposition-softmax-limit.md` §4.
  - **Relation to the far channel:** the window form is the far channel
    with one-hot content instead of learned content. Its state is fixed
    whatever W is. With the |Σ_j b_j λ_j^d|² reader over K = 8 atoms it
    needs the pair moments H_v(λ_j λ̄_k): K² = 64 reals per symbol, so
    65 × 64 ≈ 4.2k reals for characters. (An earlier version of this note
    said 65 × 8 ≈ 1k; that holds only for a reader linear in the atoms.)
    `cq` needs about 70k per layer at any reach. So the window form is cheap
    exactly where the far channel was slow. It reads only a positionally
    weighted bag of symbols, though.
  - **Probe (the user asked for it 2026-11-12):** `src/tools/agpt_polyrope_probe.py`
    (`af81726`), arms in `rnd/polyrope-probe/`. **Controls done, seed 42,
    2026-11-12** (score on targets with ≥ 1,040 preceding chars, nats/char):
    `w16` 1.3519; `exact` at reach 1,024 1.3137. The far-channel probe's
    reach-128 checkpoints rescored on the same targets: `w16` 1.3515 mean
    of 4 seeds (s42 1.3440), `exact` 1.3224 mean (s42 1.3200). So 129–1,023
    is worth 0.006–0.009 to exact attention when trained for it. Cutting the
    reach-1,024 model's attention at eval (CPU, `rescore/exact-r1024-s42-at*`)
    costs 0.0021 at 512, 0.0076 at 256, 0.0161 at 128; the last exceeds the
    trained gap because the cut model was trained to use those keys.
    The headroom beyond 128 is about a single seed's noise, so the user
    chose (2026-11-12) to measure it on PG-19 before any channel arm.
  - **PG-19 headroom (seed 42, 2026-11-13,** main `e21bf38`; imm's 20M
    split, blocks kept inside one 25k-char book piece, 50 held-out pieces,
    ~198k targets with ≥ 1,040 preceding chars): `w16` 1.4893, `exact` at
    reach 128 1.4274, at 1,024 1.4249. So 17–128 is worth 0.062 (twice
    Shakespeare's) and 129–1,023 only 0.0025, steady at −0.002 over the last
    10k steps. The eval-time cuts cost 0.0010 (512), 0.0038 (256), 0.0087
    (128). At d64/L2 and 30k steps, nothing beyond 128 is left for a far
    channel to recover, on either corpus. **Open:** whether the headroom
    grows with model size, which is the only case where the channel arms
    could pay. No channel arm has run.
- **imm's data point:** in imm a query has one real position, so H1 is well
  defined there. Recurrence statistics suggest a per-type chord buys reach,
  not compression of repeats already in view.

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
  both the race fix and the exact backward. Its gain comes from strided
  minibatch loss weights, not position (§3a, 2026-11-10). A rerun would give the current
  headline at that size. This is a suggestion; nobody has decided it.

## 6. Planned, not started

- **Skip-horizon AGPT:** design note `rnd/skip-horizon-agpt` (2026-06-18).
- **Deterministic backward:** `todo/deterministic-backward.md`, **Open**.
  Forward determinism landed in `ef3a8e9`.
- **Kneser-Ney backoff in the trie-only evaluator:**
  `todo/trie-ppl-kn-backoff.md`, **Open**.
- **Measure the one-pass gap:** how close one corpus pass gets, in quality and
  in time.
