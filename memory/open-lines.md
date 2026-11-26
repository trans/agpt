# Open lines of research

*Updated 2026-11-26*

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
  it with the user. **The user approved the weighted arm (2026-11-19, "ok
  go-ahead"):** `trie_weights_corpus` in the replay (`5db567f`, CPU-checked),
  4 runs (`run3.sh`: weighted pd=3, pd=1, minibatch on A; minibatch on the
  100M text with window_stride 100 as the equal-compute control). **Done,
  `c449019` (slot 108): weighting doesn't help.** AGPT +0.12 nats at pd=3,
  +0.05 at pd=1 against A's weights (one order seed); minibatch level (6.163
  vs 6.150). Minibatch on A with the 100M counts beats minibatch on the 100M
  text at equal steps (6.150 vs 6.329 at 62,504) and after 3x the steps
  (6.170). AGPT at equal compute 0.23-0.47 nats behind the 100M stream. Not
  run: pd=0 weighted (~75 min).
- **Depth 16 — Done 2026-11-22, mixed (agpt `10a066f`,
  `rnd/doped-counts-depth16`, slot 170, 12 runs, 2 seeds each).** The user
  asked for it with 1G and 10G count texts ("we would need to bring in a 1G
  and 10G corpus to really fill in the 16 depth tree"). Minibatch on the 1M
  slice, seq_len 16: 100M / 1G / 10G counts gain 0.053 / 0.064 / 0.072 nats
  over hard targets after 8 epochs, but minibatch on the 100M text itself
  gains 0.086 (5.397 vs the 10G-doped 5.471): the doped slice no longer
  comes out ahead, as it did at depth 8. The slice runs short of contexts,
  not only targets. Count text `data/pg19/imm-split-10g-dedup` (built here;
  a first build held held-out pieces 20 and 23 whole, other editions of
  their test books; now only books mobius counts, minus 54 more). Counter
  `src/tools/doped_counts.py` (`7418d09`, GPU, 10G in 12 min).
- **The bigger-text version was already run at 20M** (air slot 90,
  2026-11-18, air `5b007ee`): soft targets from mobius's d24 rows with 10G
  books, 1.84M transformer, scored without the prior: no transfer. But
  prior dropout did transfer once (air `bef9142`, 2026-11-12): the 1.84M
  trained with the prior on its logits, scored with no prior, 0.029
  bits/char ahead of plain (one seed); the input form and 10.8M level or
  worse. Untested difference: air's
  rows condition on <= 24 chars while its network reads 256.
- Related record: agpt `rnd/prior-scaling-plain-1g` (a 1 GB prior beat
  training on the 1 GB by 0.025 bits/char), `rnd/count-backoff-gate`
  (same-corpus smoothed targets hurt AGPT: 6.70 against 5.30; raw counts from
  a larger text keep the objective's form and help).

## 1b. The deep past and the other side (2026-11-19/20)

- **Dynamic eval — Done, positive (agpt `6e0f516`, `rnd/dynamic-eval`).** The
  user (2026-11-19): "the missing link is the deep past having some influence
  in the present, like a LoRA on the trie." Eval-time test, CPU: score each
  PG-19 held-out chunk (5,000 chars, a different book each) in order, one
  step per 64 chars on the text just scored, reset per chunk. Report half,
  four d64/L2 seq-8 checkpoints: own past gains 0.057-0.102 nats/char; the
  same steps on another book's text 0.026-0.063 (partly annealing, the models
  were trained at constant lr); own minus other 0.021-0.041, rising from ~0 in
  the first 1,000 chars to 0.062-0.094 in the last, still rising at 5,000.
  LoRA rank 4/16 without biases: 56-83% of full. Static = the runner's
  fixed-token PPL exactly.
  **Whole books — Done (agpt `6f8d8bb`, 2026-11-20, the user approved).** 20
  PG-19 test books of 300,000 chars, two checkpoints: own minus other 0.135
  and 0.100 nats/char on the report books, 0.173 and 0.135 in chars
  200,000-300,000, still rising (about +0.02 per doubling of the past); the
  control fades. The adapted 1M-text model beats the static 100M-text one.
  Rank-4 LoRA with biases: 72-74% of full. Published relatives: Krause et
  al. 2019 (Transformer-XL), TTT-E2E (arXiv 2512.23675). @imm told.
  **Next (not decided by the user):** training for adaptation (TTT-style, or
  a per-document trie of the past so the step is AGPT's aggregated gradient).
  **On mobius's prior + stream network — Concluded, positive (agpt `6cf3294`,
  slot 218, 2026-11-26; the user, 2026-11-23: "ok, please coordinate with
  @mobius").**
  - **Result (report books, bits/char, 25/25 books better):** one Adam step
    per 512 characters, transformer only, lr 1e-4 (tuned on the even books).
    - 10.8M: 1.3297 → 1.2924 (0.037). The other-book control is 0.027 worse.
    - 25.8M: 1.3408 → 1.2737 (0.067). Its control is level.
    - Own past minus control: 0.064 and 0.069. It grows through the book,
      to 0.080-0.085 past 300k characters.
    - LoRA 16 matches full. Letting `inp`/`gate` adapt changes nothing.
    - Short of the user's 0.1. Untried: a 25.8M lr above 1e-4 (still
      improving at 1e-4), more steps per window, Krause-style decay, the
      100M network.
    - Mailed mobius the results, the `.npy` paths and the correction below
      (`mail/sent/20261126-203049Z-@mobius--*`).
    - Slot 218 ran two jobs at ~1.9 GB host each, over mobius's 3 GB limit
      (told @air, `mail/sent/20261126-212131Z-@air--*`). An adaptive 25.8M is
      1.6 GB anonymous on the CPU alone, so run one job at a time.
  - **Next, the user via @air (2026-11-26, `mail/received/20261126-211810Z-@air--*`):
    Running, slot 220 (agpt `1288c55`, `run_100m.sh`).** Dynamic eval on the
    100M-trained 25.8M (`pd_s25m_stream100m_svd32_b1024.pt`, 1.3231 on the
    pieces) and the 100M 10.8M for the pair, on the 20M books' rows
    (approximate). The question: do the text gain and the adaptation gain add
    up to ~0.1? Mail air and mobius the results when the chain ends.
  - Dynamic eval of
  `pd_s10.8m_stream_svd32_b1024.pt` on mobius's 50 whole held-out books
  (static 1.3309 bits/char; prior-rows-only network 1.3635), in
  `rnd/dynamic-eval-prior-network/`. Mobius's answers (`mail/received/20261123-161956Z-@mobius--*`,
  `20261124-161618Z-@mobius--*`): agpt runs it, in a slot.
  - **Files:** `prior_input_only.py` (sha c9a82dd8…; it imports two imm files, below) and the
    checkpoint (sha 1a6b9d95…) are now in mobius `runs/frozen/` (durable).
    `score_books.py` is committed at mobius `rnd/experts-leave-one-out/` (6b75bb6).
    The rows are in `runs/imm/pg19split20m_books_stream_d24_pg10g`, the split
    is `data/pg19/imm-split-20m-books`.
  - **Protocol:** adapt in `model.eval()` (no dropout, no prior dropout).
    Keep the checkpoint's gate-feature mean/std. Headline: freeze `inp` and
    `gate` (they price the prior) and adapt the transformer; everything
    adapting is reported apart. LoRA on attention and MLP is the middle
    setting. Control: the same steps on another book; reset per book.
  - **Pairing:** `score_books.py` scores 1,024-character segments with the
    preceding 1,024 as context. Pair position by position from per-book
    `.npy` losses, and send mobius the `.npy`.
  - **Cost:** size one backward step at block 1,024 before the slot; keep
    the host side under 3 GB; use `slots add ... --run CMD`, since a grant
    lapses after 30 minutes if the session is idle.
  - **The user's goal (via @air, 2026-11-26, `mail/received/20261126-184907Z-@air--*`):**
    "We need gains closer to 0.1 not 0.01, or we will never even get close
    to 1.0." So the figure that matters is the gain on top of the stream rows,
    which already take 0.072 bits/char from the book's own past (prior 1.6194,
    with the stream 1.5473 on the 100M split's held-out text).
  - **Newer checkpoints (mobius `decf7c2`):** `pd_s10m8_stream100m_svd32_b1024.pt`
    is trained on the 100M split (1.3525 vs 1.3610 on its 250k held-out
    characters). A 25.8M on the 100M split was training on 2026-11-26.
    `pd_s25m_stream_svd32_b1024.pt` is a 25.8M on the 20M split.
    There are no 50-book rows from the 100M network's prior (`pg10g_wo100m`).
    Four of the 110 books mobius dropped for sharing text with the 50 books
    are in the 100M split's training text (38554, 42410, 7137, 8145; agpt's
    check against `books.json` and the 100M manifest, 2026-11-26).
  - **Checkpoints, settled with mobius (2026-11-26, `mail/received/20261126-185534Z-@mobius--*`):**
    - **First:** the 20M 10.8M on the 20M books' stream rows, paired against
      1.3309.
    - **Size control:** the 25.8M on the 20M split
      (`runs/frozen/runs/pd_s25m_stream_svd32_b1024.pt`, sha b5d1fc93…, 8
      layers × 512, 8 heads). It fits the same rows (same SVD basis
      `runs/frozen/svd/rows_stream_d24_pg10g.npz`, same gate
      standardization). It is 1.3787 on the pieces (it memorizes the 20M) and
      not yet scored on the books: run `score_books.py` on it in our slot
      first (~3 min).
    - **100M 10.8M:** `pd_s10m8_stream100m_svd32_b1024.pt` (sha 4e4d5595…,
      its own SVD basis carried in its buffers). It's a fair approximate
      figure on the existing book rows. Mobius sized the two caveats as
      negligible: 200 books out of the counts ≈ 0.0003 bits of prior; the
      four shared books give 550 of 18.7M positions a shared 64-gram, longest
      run ~73 characters (quotations), < 0.0001 bits. Rows under its own
      counts would cost ~5 h of Xeon; mobius won't build them unasked.
    - The GPU was slot 217's until ~20:10 UTC on 2026-11-26.
  - **Code and slot (agpt `1022410`, 2026-11-26):**
    `rnd/dynamic-eval-prior-network/` holds `dyneval_prior.py`, `run.sh` (the
    chain: sizing; static on both networks; the 10.8M tune grid on the even
    books; report own/other, all, LoRA on the odd books; the 25.8M tune and
    report) and `summarize.py`. Ran as slot 218 with `--run`, 19:33-20:27 UTC;
    log in `chain.log` (gitignored).
    - **CPU check:** static matches mobius's `loss_stream.npy` (+0.0001 nats
      on 2,048 characters).
    - **Correction, sent to mobius:** `prior_input_only.py` is not
      self-contained. It imports imm's `scripts/transformer_baseline.py` and
      `imm/data.py` (last changed at imm `43057d9`), so their hashes go into
      each run's JSON.
- **Trie growth — Concluded, mixed (agpt `55549fb`, `rnd/trie-growth`).** The
  user (2026-11-20): train while the trie grows, "cadence in time, not space";
  aggregate as counts arrive, update after N(trie) changes, recompute after
  each update. 44 canonical runs, slot 132. Doubling beats full-trie AGPT at
  equal compute at pd=1 and pd=3 (pd=3: 5.51 in 54M positions against 5.77 in
  71M), from cheap early steps, not better ones; minibatch still leads (4.66 at
  2 epochs, 4.58 at 4, against 5.51 at 3.2 epochs' worth, with 4.4x fewer
  steps). pd=0 at 1% growth per update: best steps (697 give 7.02) but 110
  full-trie passes. The first check found v2 growth mode had never applied
  `anc_grad_exact` (fixed `03f3a49`). Random-order growth since tested in
  `rnd/noise-cadence` (below).
- **Noise cadence — Concluded, positive: parity, from the adaptive batch, not the trie
  (agpt `54c7b4d`, `5e9057b`; code `0a6f4de`, `7bf11e0`, `6a435d6`, `e88bf88`;
  `rnd/noise-cadence`).** The user (2026-11-21) asked for a
  more nuanced rhythm than count changes; "go ahead" to a split-half norm test,
  then (2026-11-22) "add the resampling variant", then free rein. With a fresh
  sample each step the cadence is level with minibatch at equal compute within
  seed noise: 4.567 vs 4.552 at 136M positions (3 seeds each), 4.535 vs 4.522 at
  271M (2 each), with 6-9x fewer steps. The earlier "0.04 gap" was against
  minibatch's best seed (spread 0.078 / 0.044 / 0.023 at 1 / 8 / 16 epochs).
  v2's sample gradient equals PyTorch's (1e-6); v2 trained like minibatch equals
  the PyTorch trainer over 3 seeds. The trie merges only 13-18% at these sample
  sizes: the cadence carries it. Closed: cumulative sample (overfits, 6.12),
  constant LR, sample caps, depth-stratified sampling (noise even over depth).
  The user (2026-11-22): after free rein, if no strong fruit, a long pause.
  **No-trie control done (2026-11-22, the user asked; agpt `e88bf88` code,
  `5e9057b` results, slot 166):** the same cadence on plain windows in PyTorch
  (`agpt_minibatch_train.py` `noise_theta`) is level with the trie version:
  4.578 vs 4.567 at 136M (3 seeds each), 4.536 vs 4.535 at 271M, and 4.557 at
  equal data. The trie saves 15-18% of positions, worth <= 0.01 here; the gain
  is the adaptive batch. Pooled, both cadences trail minibatch by 0.014-0.021
  on the means (~1-1.5 s.e.). **Next (not decided by the user):** more seeds;
  where the trie could still add: a setting that merges far more, or refined
  counts (which so far help minibatch equally).
- **Forward-backward coupling — Done, negative (agpt `e6c3620`,
  `rnd/forward-backward-coupling`).** The user asked why AGPT uses only "where
  a character goes", not "where it comes from"; "Well feel free" to the open
  dual-model-fold measurement. A fixed backward model's prediction as a soft
  target: beta 0.1 level (+0.0008 nats canonical rolling), beta 1 +0.155,
  worse at every checkpoint; a forward teacher of the same recipe (control)
  helps a little (-0.008, -0.012). The coupled optimum is pulled off
  p(x | past) by B's prediction averaged over futures. With the June
  masked-LM null, closes prefix-suffix coupling for next-char prediction at
  this size. CPU runs, one seed.

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
  scores within run-to-run noise. Runner then: `--trainer v2-kvfp32` (gone
  since the switch), and `NVIDIA_TF32_OVERRIDE` recorded in `meta.json` (`18284a0`). **The user chose
  fp32 as the default (2026-11-18: "I agree, fp32"): `fe3a4c8`.** v2 builds
  fp32 cache + fp32 cuBLAS math; `-DAGPT_V2_REDUCED_PRECISION` (`just
  build-agpt-train-v2-reduced`, `--trainer v2-reduced`) is the old default;
  v1 unchanged (TF32). **Checked, `02fa484` (slot 103, 2026-11-19):** the new
  default's frozen-weight gradients match the old fp32 cache with TF32 off to
  the atomics floor (1.2e-7 whole pass, against 1.1e-7 between two of its own
  dumps), the reduced build matches the old default the same way, FD ratio
  0.9998-1.0001 with no env var, and the 4 runs fall in the grid's spread. Holds at d64/L2;
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
