# Open lines of research

*Updated 2026-11-18*

Each line has a state, where its plan lives, and what it needs next.

## 1. Context-tree orientation (suffix tree instead of the prefix trie)

**Called on 2026-11-10 by the user: a minor breakthrough, with a proviso to
double-check and polish. The user asked on 2026-11-17 to finalize it; the
attention check confirms it.** The prefix trie is the right orientation, because
its pd=1 units group by the least predictive character; suffix-tree units are
target-sorted (skew 0.86 nats at any depth: the mutual information of adjacent
characters). Record: `rnd/context-tree-orientation/README.md` (Conclusion;
Attention grouping control; Proviso status table).

- **Attention control, done 2026-11-17** (`16f73e7`..`bd9535a`, slots 66, 68,
  70; 16 canonical runs): d64/L2 attention at depth 16, 65 steps per pass.
  Suffix units trail prefix units by 0.91 nats after one pass and 0.59 after 16
  (best rates). Prefix units trail size-matched random windows by 0.09–0.28 in
  passes 1–4 and are level from pass 16 (constant rate and cosine). So the
  user's suspicion that the prefix grouping costs something holds early in
  training (one-pass training's regime) and fades, as for the GRU.
- **Next (the user asked for it on 2026-11-17, when there is a chance): check 3,
  v2's fixed unit order at pd=1.** No v2 run at pd=1 has ever had
  `experimental.unit_order_seed` set. The fixed-order arm already exists: the
  `rnd/one-pass-gap` grid `agpt-e{1,2,4,8}-lr{0.003..0.024}` (v2's radix-id
  order, warmup-cosine per step, 1 pass 15.734 at 0.012). The new arm is the
  same configs with `unit_order_seed: 1`, plus seed 2 at the best rates: about
  20 short runs, a few GPU minutes in a slot. It says how much of AGPT pd=1's
  early cost (0.150 nats against size-matched random batches after one pass)
  comes from the order rather than the grouping. Expect it to be small: at
  pd=3 the order cost 0.024 nats.
- **Proviso checks still open:** GRU seeds (check 1); v2's fixed unit order at
  pd=1 (check 3; at pd=3 it costs 0.024 nats after one pass); suffix units at
  pd=2/3 (check 4); interference between unit steps (check 6). The suffix arm's
  best rate at 16 passes was the grid's bottom (0.0015).
- **Open puzzle from the GRU control:** shuffled minibatches at a constant 3e-3
  got worse after epoch 100 (train 1.846 → 1.893 at epoch 300), while
  start-position-labelled random batches did not. One seed each; the
  cosine-schedule runs avoid it. With attention, random windows' pass-mean
  train loss rises more often than prefix units' at every rate and under
  cosine, on the same passes for one order seed, without showing in held-out.
  Not explained.
- **Tools:**
  - `src/tools/agpt_context_tree_coherence.py`: the trees, with an exact
    level-by-level backward;
  - `src/tools/agpt_unit_grouping_control.py`: the no-trie grouping control,
    with node and event groupings and `--schedule cosine`.
  - `src/tools/agpt_minibatch_train.py` `experimental.trie_plan: group`
    (`first | last | window`): the same control for the attention model,
    through the orchestrator.

## 1a. Refined trie counts (a small trie with a larger corpus's counts)

**Concluded positive 2026-11-18 (slot 86, 11 runs, canonical).** Opened at the
user's word ("let's test it", 2026-11-17), from their observation that in the
imm and mobius work most of the useful data lives in the shallow tree, and
mobius refines a small corpus's counts with a larger corpus's. Record:
`rnd/refined-trie-counts/README.md`; code `35b4ee8`
(`experimental.trie_targets_corpus`), results `9d4ef74`.

- AGPT pd=3 replay, d64/L2, depth 8, on a 1M PG-19 slice's trie: the 20M or
  100M text's counts beat the slice's own by 0.051–0.057 nats after 8 passes
  (three pairs, two order seeds), same steps and nodes. The 4.75M text's
  counts give nothing (39% of the slice's contexts still occur once there).
- A better objective, not a speed-up: in minibatch training own counts as
  soft targets fade to −0.008 by 8 epochs, the 100M counts add −0.058 on top
  and are still growing; the slice with 100M counts (6.150) beats 2 epochs of
  the 4.75M text itself (6.314).
- Keeping the small trie beats growing it: AGPT on the 4.75M text's own trie
  trails the slice's trie at equal steps (bigger units).
- Seed caution: the baseline's two order seeds differ by 0.065 nats, as much
  as the effect; the paired gaps agree within 0.006.
- **Plan-depth follow-up done (the user, 2026-11-18: "I am curious to see
  pd=1 and pd=0 as well"; pd=6 added as the deepest plan on record and its
  best score; slot 88, 8 runs, results `f5e9d7b`, `trie_plan` pd=0 `0485dce`).**
  The gain holds at every depth: pd=0 (full batch, 512 passes) −0.033, pd=1
  (128 passes) −0.073 / −0.033 on two seeds, pd=6 (2 passes) −0.052. In
  full batch it appears only from step 256. At equal passes deeper plans
  lead (pd=6 at 2 passes 9.459 beats pd=3 at 8); at equal steps the order
  reverses (pd=0's 512 steps 6.841). Best AGPT here pd=0 + 100M counts,
  6.621; minibatch with the same counts 6.503 after one epoch. The slot ran
  3 h 22 min against the 110 min asked: pd=6 with soft targets took 99 min a
  pass in the Python replay.
- **Shakespeare pd=6 rerun done (the user, 2026-11-18: "yes, queue the
  Shakespeare pd=6 rerun"), slot 91, record `rnd/partition-depth-rerun/`
  (`8bb58f8`).** Asked after the user objected to setting today's 8.98
  (PG-19, depth 8) against `partition-depth`'s 3.82. The 2026-04-30 code
  (`a8ddb15` + microgpt `bdd8e82`) reproduces 3.836 on the sweep's own trie
  and scorer. On the canonical split: old scorer on the start of
  `data/input.txt` 3.80/3.82, fixed 32-char window on the held-out chunks
  3.97, canonical rolling 4.89/4.95. The user was right that held-out text
  wasn't what made 3.82 look good (4%); the scorer is the difference.
  Corrections made to the user: the record's "d=32" is depth 32 (model
  d64/L2); I had wrongly implied 3.82 was close to a training-set score.
- **Sharing with the larger text's counts: not tested (the user asked,
  2026-11-18).** The runs took each context's distribution from the larger
  text but kept its weight in A, so sharing stayed at 5.2x. Counted
  (`eff0ed6`, `superload_sharing.py`): weighted by the 100M text, A's 1.54M
  depth-8 nodes cover 83% of its 800M events, 433x sharing (22x at 4.75M,
  88x at 20M); what A lacks is deep contexts (half of depth 8). I had left
  "weighting by the larger count was not tried" in the README without raising
  it with the user. Offered: the weighted arm in the replay against minibatch
  on the 100M text at equal compute (the replay tests the objective only).
- **Next (not decided by the user):** the weighted arm; refined counts in v2
  (needed for the compute saving); the gain at depth 16; minibatch streaming
  the larger text for the same steps.
- Related record: agpt `rnd/prior-scaling-plain-1g` (a 1 GB prior beat
  training on the 1 GB by 0.025 bits/char), `rnd/count-backoff-gate`
  (same-corpus smoothed targets hurt AGPT: 6.70 against 5.30; raw counts from
  a larger text keep the objective's form and help).

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
- **Full precision costs nothing measurable (the user: "want to do that",
  2026-11-18; slot 97, `rnd/exact-build-speed/`, `18c104c`).** fp32 K/V
  cache + `NVIDIA_TF32_OVERRIDE=0` against the normal bf16/TF32 build, two
  one-pass-gap runs, twice each: pd=1 4 passes 29.3 s vs 28.3 s (+3.7%),
  pd=3 1 pass 54.1 s vs 54.4 s (-0.6%), inside the 1-3 s repeat spread;
  scores within run-to-run noise. Runner: `--trainer v2-kvfp32`, and
  `NVIDIA_TF32_OVERRIDE` recorded in `meta.json` (`18284a0`). **The user chose
  fp32 as the default (2026-11-18: "I agree, fp32"): `fe3a4c8`.** v2 builds
  fp32 cache + fp32 cuBLAS math; `-DAGPT_V2_REDUCED_PRECISION` (`just
  build-agpt-train-v2-reduced`, `--trainer v2-reduced`) is the old default;
  v1 unchanged (TF32). Check pending (`switch_check.sh`, slot 103): dumps vs
  the timed binaries, FD on the new default with no env var, 4 runs. Holds at d64/L2;
  re-measure at a wider model.

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
    could pay. No channel arm has run. Write-up: `rnd/polyrope-probe/README.md`
    (main `4074ae2`).
  - **What has and hasn't been tested (the user asked, 2026-11-13):** only
    the channel approach ran (far-channel probe, and this probe's controls).
    The design's original, corpus-level notion was tested only in its
    Phase 1 form, by mobius (bigrams from corpus signatures, failed on
    storage). Phases 5–7 (signatures as embeddings, structural queries,
    vocabulary-only training) are untested. For characters the notion can
    be tested exactly, without spectra: corpus displacement counts R_ab(d)
    (96² × 1,024 ≈ 9.4M on PG-19) as an attention bias, which is mobius's
    proposed distance co-occurrence. Not agreed.
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
