# Handoff: where AGPT stands

*Updated 2026-11-25*

## Resume here (2026-11-26)

- **Dynamic eval of mobius's prior + stream networks: done** (agpt
  `6cf3294`, `rnd/dynamic-eval-prior-network`). On top of the stream rows it
  gains 0.037 bits/char on the 10.8M and 0.067 on the 25.8M. The adapted
  25.8M, 1.2737, was then the best figure on the 50 books. On the
  100M-trained 25.8M (slot 220, `2e17076`) adaptation adds only 0.031
  (1.2903 → 1.2592), 0.071 in all, against the user's bar of 0.1. Results
  are mailed to air and mobius. Next steps are not chosen (open-lines.md §1b).
- **Direction (decisions.md, 2026-11-22/23):** regular training instead of
  pd=N trie training; agpt serves the prior + residual line. Depth-16
  doping concluded (`rnd/doped-counts-depth16`, `72b4a40`).

## Back on the main path (2026-11-14)

- **The drift:** from 2026-11-10 to 2026-11-13 no work used the trie. The
  superposed-RoPE, far-channel and PolyRoPE probes were plain windowed
  transformers (PolyRoPE concluded inconclusive, `4074ae2`), and the 220M
  and 1 GB plain runs were controls for @air's count-prior study
  (`e1a5a33`, `044d249`). The user asked on 2026-11-14 whether we had
  drifted, and set the rule for what follows: small, fast comparisons,
  bigger runs only to demonstrate (decisions.md).
- **The one-pass gap, measured** (`rnd/one-pass-gap`, `f547847`, 48 runs,
  canonical eval): AGPT pd=1 after one pass 15.73; minibatch training of
  the same model from the same weights 4.83 after one pass (31 s), 4.66
  after two, better than AGPT's 625-pass best (4.764). The 65-steps-per-pass
  random-batch control gives 12.56 after one pass: update count is most of
  the gap, but AGPT's steps are worth about a third of the control's (8.71
  against 5.67 after 8 passes).
- **AGPT at minibatch cadence** (the user chose pd=3 on 2026-11-14, saying
  "the main issue is the number of updates"; 47 runs, `bd3acc9`):
  - One pass, crown trained: pd=2 9.65 (1,463 updates), pd=3 7.81
    (12,675, shuffled; 8.00 in radix-id order), pd=4 6.74 (58,721, 185 s a
    pass).
  - Random equal batches at the same counts: 5.18 (724 windows) and 4.83
    (84 windows).
  - Updates help, but past about 1,500 per pass what each step holds
    matters more than how many there are.
  - At pd=3, step count and mean step size both match `mb84`. Random
    windows in batches of pd=3's unit sizes score 6.33, so the 0.48-nat
    gap is about half unequal sizes and half the trie's prefix grouping.
  - `fire_norm: sum` is worse (8.53 at pd=3; 23.4 against 15.7 at pd=1
    after one pass).
- **New v2 options:** `experimental.train_crown` (`b1019b3`) trains the
  depth<pd nodes that pd≥2 plans skipped (12.4% of events at pd=3), and v2
  prints `plan-coverage`. `train.fire_norm: sum` (`e59b10c`). The minibatch
  trainer gained `experimental.step_events_file` (`21ec506`).
- **Conventional epochs and AGPT's plans in the minibatch trainer**
  (follow-up 2, 29 runs, 2026-11-15, `7c911d7`):
  - Every "epoch" on record (v2, the minibatch tool, the old vanilla
    trainer) is every window start with 16 targets each. A conventional
    epoch (each character a target once, 518 steps, 2 s) scores 6.94,
    better than a pd=3 pass (8.4× the positions, 24× the steps). Eight
    epochs, about a pd=3 pass's compute, score 5.12; 16 epochs 4.85,
    the same as one every-start pass (4.83).
  - The pd=3 plan replayed event by event in PyTorch scores 7.76 against
    v2's 7.81, so the gap is the plan, not v2.
  - Packed plans with 12,675 equal steps: the more of the trie's sharing
    a step keeps, the worse it scores (65% 5.39, 88% 5.86, 97% 6.34;
    random 12% 4.83). No point on the curve beats conventional epochs
    at equal positions. If the full sharing cost nothing it would be
    worth about 0.05 nats at T=16; keeping 65% costs 0.11.
- **The user's question after these results (2026-11-15): back to the
  drawing board.** They asked how AGPT's one equivalence (gradient sharing
  across repeated prefixes) can benefit any model in training or
  inference, and if it can't, why. My analysis, given in the session and
  not yet answered by the user:
  - The equivalence regroups the loss by distinct context, so it saves
    compute only where contexts repeat. Text stops repeating once a
    context's block entropy reaches log2 N (about 10–12 characters
    here). The saving is then a fixed number of positions per token,
    whatever f_θ is, while the cost grows with context length.
  - The aggregated gradient equals the sum of instance gradients at one
    θ, not the path of N steps. Stability caps the step, and Adam
    normalizes the factor N away, so an exact gradient is worth about
    one step.
  - Sharing needs a context's occurrences in the same step. With S steps
    per pass that happens mostly for contexts with N(v) > S, so keeping
    sharing means grouping, which correlates steps. That is why pd>1
    turns into minibatching (the user's own observation).
  - Where it does pay: workloads that are trees by construction (RL
    group rollouts and preference pairs sharing a prompt, prefix caching
    in serving, tree speculative decoding, beam and tree search), where
    the shared part is already in one step. Those are known techniques.
  - One cheap untested variant: conventional training with the trie's raw
    next-token distribution as the target at each position (same expected
    gradient, lower variance, no step loss). The count-backoff-gate test
    used smoothed targets and hurt; raw ones don't change the objective.
    It only touches positions within ~d* of a window start.
  - **Open: the user's call** on where AGPT goes from here.
- **@air's cross-project writeup, "The trie as data, not as a training
  schedule"** (mail `49514c7d`, 2026-11-15): the trie's counts are the
  gradient's only data term, so a count prior is the cacheable part of
  the gradient. It pays as input data (agpt's 1 GB control: 0.025
  bits/char at 50k steps) and loses as a schedule (one-pass-gap). The user
  called it "deferred AGPT". I sent corrections (the prior runs use the
  input and the logits, fixed in `5f243e4`; one misread sentence; a stale
  quote) and the follow-up 2 numbers, reply in `mail/sent/`.
- **Trie targets at minibatch cadence** (chosen by the user 2026-11-15;
  follow-up 3, 21 runs, `9456c87`): ordinary minibatch training with each
  position's target the trie's next-character distribution for its
  context. That's AGPT's node loss without its schedule: same expected
  gradient (checked to 1.8e-15), lower variance. In the same batches as
  one-hot training it gains about 0.025 nats at 1–8 conventional epochs
  (6.94 → 6.77, 5.12 → 5.00), worth 1.13–1.37× the training, and
  0.014–0.017 at 16 epochs. It costs ~10% per epoch in this
  implementation. I predicted ~1.5× from the measured noise cut
  (19–38%); early steps are less noise-limited than that. Bound: only
  repeated contexts change (about half the events at T=16, fewer at
  longer T).
- **Follow-up 4, trie targets at longer windows** (asked by the user
  2026-11-15; 36 runs, slot 45, `ab47026`): at 2048 targets per step the
  paired gain is ~0.014 nats at T=32 and ~0.013 at T=64 (6/6 pairs each),
  none at T=128 (3/6). Predicted 0.011 / 0.005 / 0.002. Bounded like the
  trie's compute saving: ~8.7 changed targets per window whatever T.
- **Follow-up 5, exact emulation** (the user, via @air, 2026-11-15: the
  equivalence is exact, so a standard transformer emulating AGPT's run
  should match; compare gradients at frozen weights). **Done, `ab47026`.**
  With the fp32 cache and TF32 off, v2's per-unit gradients match float64
  PyTorch to rounding (as float32 PyTorch does). Exact replays (v2's units,
  v2's order, the trie's tail-wrapped events: `wrap_tail`,
  `trie_plan_order_file`) reproduce v2 within 0.4% (pd=1 15.72 vs 15.73,
  pd=3 7.80 vs 7.81). The old replay's neighbouring-rate gaps came from its
  own unit order. **The normal v2 build is not exact:** bf16 K/V cache and
  TF32 put 1–4% into unit gradients (8% on the trained whole-pass pd=1
  gradient, 67% in some small shallow units); every AGPT run on record used
  it. Full precision costs no measurable speed at d64/L2
  (`rnd/exact-build-speed`), and it is v2's default build since `fe3a4c8`
  (the user, 2026-11-18), checked in `02fa484`; `just
  build-agpt-train-v2-reduced` builds the old one. Open: whether the old
  build's error mattered for long runs or L-BFGS.
- **The user's contrast reading** (2026-11-15): prefix-shared steps give no
  contrast at the root, so root learning would lag. First reported wrong:
  at equal steps the gap grows with context length, but a random
  conventional epoch trails `mb84` in the same shape (any run behind is
  behind at long contexts). **Corrected in `87dd07f`** (`depth_matched.py`,
  random-batch runs at the same loss over contexts 5–16): AGPT pd=2/3/4 lag
  at 1 char by +0.26/+0.32/+0.63, match from 4 on; packed plans and `mbu3`
  don't. Cause: a unit's mean step gives each shallow context one step per
  pass however common (space: 1 step vs 121 in equal random batches at
  pd=3); packed chunks restore frequency weight, so it's weight, not
  within-step variety. ~0.03 of pd=3's 0.48 nats. pd=1 has no reference
  (less trained than any random run). Mass-weighted units (`fire_norm: sum`,
  `bd9d1fe`) cut the root lag to +0.08 but score worse overall (8.53): Adam
  can't weight one step against another, so equal-mass steps are the fix.
  The user judged this a significant finding, the core problem identified
  (2026-11-15). Still separate: the prefix-grouping cost at equal mass
  (packed 97%, 6.34 vs 4.83).
- **Follow-up 6** (`76dafea`, slot 47, 24 runs): equal-mass packed plans at
  pd=1's 65 steps. All sharing kept, equal mass: 14.01 vs AGPT 15.73 (half
  of 0.225 nats to `mbL` 12.56). More pieces per step at fixed steps: 13.41
  (166/step, 99.8% sharing), 13.12 (1,191, 98.4%), 12.66 (8,989, 83.7%).
  `mbu1` (random windows, pd=1 sizes) 13.54: prefix grouping costs 0.150
  after one pass, like the GRU early. Random batches of 16k windows keep
  52.6% of the sharing themselves. No root lag. Sharing and variety go
  together only in big steps (few updates).
- **Follow-up 7: SAG** (`5eab514`, slot 61, 9 runs, 2026-11-16). The user
  asked what SAG can do and chose to start at the lowest pd because of the
  table's memory (28 MB pd=1, 0.6 GB pd=2, 5.5 GB pd=3). `experimental.grad_average: sag | ema` in the minibatch
  trainer's replay (`63bbb29`). Result: SAG on pd=1's units (momentum off)
  scores 24.94 after one pass and 14.49 after eight, against AGPT's 15.73 and
  8.71. Cause measured (`sag_staleness.py`, CPU, 20k chars): the table's
  mean points uphill at the current weights on 45–53% of steps, while one
  fresh unit's gradient has cosine +0.8 with the full gradient. The stored
  gradients are a pass old, and with Adam a pass moves the weights a lot;
  in one pass nothing is ever replaced. Mass-weighted momentum (`ema`)
  was worse than plain steps too (CPU calibration). Conclusion: averaging
  across steps can't buy back a grouped step's variety, so the big-step
  trade-off stands. pd=2/3 not run. **Open: what next (the user's call).**
- CPU PyTorch runs open a CUDA context (~128 MiB) at the first
  `torch.optim.Adam` step (its graph-capture check). The minibatch trainer
  now hides the GPU when on the CPU (`64ecd85`); for other CPU work, set
  `CUDA_VISIBLE_DEVICES=` outside a slot.
- `src/tools/agpt_minibatch_train.py` (`84054bc`) is the minibatch baseline
  as an orchestrator custom trainer, so baselines get `result.json` too.
- **Orientation finalized with attention** (2026-11-17, `16f73e7`..`bd9535a`,
  slots 66, 68, 70). The user asked whether the minor breakthrough was ever
  finalized (it wasn't: none of the six proviso checks had run) and approved
  both the attention control and the README polish, then asked for the
  cosine check. `trie_plan: group` (`first | last | window`) in the
  minibatch trainer. Suffix units trail prefix units by 0.91 nats after one
  pass, 0.59 after 16; prefix units trail size-matched random windows by up
  to 0.28 in passes 1–4 and are level from 16. Details in
  [open-lines.md](open-lines.md) §1. Slot lesson: three d64 streams do not
  overlap on this GPU (100% busy; a pass took 3× as long), so estimate slots
  from the sum of the runs.

## Where we were (as of 2026-11-10)

- The last research commit on `main` is `7ef76d6` (2026-10-03). Nothing landed
  between then and the `.ai/` setup (`a6f3c84`, 2026-11-07), and the main
  checkout had no uncommitted changes on 2026-11-07.
- The project was declared closed in a 2026-06-11 memo (see
  [closed-lines.md](closed-lines.md)) and reopened on 2026-09-23/24. Since then
  almost all work has been one strand, `rnd/gradient-population` (opened
  2026-09-24), the only experiment whose front matter still says `active`. It
  asks what prefix sharing costs, since more sharing means fewer optimizer
  updates per epoch (the notes call this "Big Issue #2"; #1 is context length).
- Its arc, in one line per step (details in that README):
  - **Exp 1:** gradient coherence collapses as training goes on.
  - **Exp 2:** a static per-root split map is refuted.
  - **Exps 3–5b:** the cadence advantage turns out to be per unit of compute,
    not per step.
  - **Exp 6:** curvature (L-BFGS) pays on exact gradients in PyTorch.
  - **Exp 7:** L-BFGS on the CUDA trie gradient stalled, because that gradient
    was not exact.
  - **Exp 8:** the exact ancestor backward fixed it.
- Two CUDA correctness fixes came out of this strand. They change how every
  older number reads (see [methodology.md](methodology.md)):
  - the shared-memory reduction race, `27ee367`, found on 2026-09-24;
  - the exact ancestor backward, `aa1b6ac`, on by default since `88c9335`
    (2026-09-28).
- 2026-09-27..29: the agpt-ultra history was imported into `research/ultra`
  (`bd680ff`), every `rnd/` README got validated front matter (`eaf4c6c`,
  `c81ab8f`), and the static site and its experiment archive were built
  (`e1da9b7`).
- 2026-10-03: the context-tree orientation test (`cac6c86`, `7ef76d6`). The
  user asked for it to be run to ground.
- 2026-11-10: a no-trie grouping control settled the mechanism, and the user
  called the strand a **minor breakthrough**, with a proviso to double-check
  and polish. The finding: the prefix trie's pd=1 units group by the least
  predictive character and train like shuffled minibatches, while suffix-tree
  units are target-sorted (`rnd/context-tree-orientation`, Conclusion).

## Latest results

- The best d64/L2 result on record: Adam pd=1 with the exact gradient for 625
  epochs, rolling byte PPL **4.764** (canonical).
- At an equal number of traversals, pd=1 Adam beats full-trie L-BFGS by
  5–6% (`829a7b4`). Curvature rescues pd=0 (12.05 → 5.02) but does not win.
- The exact gradient improves Adam pd=1 at 100 epochs by 2.3% (5.348 → 5.225)
  for 15% more wall time. It does nothing for pd=0 Adam (`7857983`).
- Numbers and how to reproduce them: [measurements.md](measurements.md).

## What's next

The user named item 1 on 2026-11-10. No order is agreed for the rest.

1. **Multi-positional (superposed) RoPE.** The review is done, and the
   phase control settled that phase RoPE isn't positional. Step 1 (the
   superposed tail, `rnd/superposed-tail-rope`) was inconclusive (the runs
   were undertrained), and the user judged the test worthless, 2026-11-10.
   The softmax limit of superposition is now derived and measured
   (`notes/seq-len-extension/superposition-softmax-limit.md`): superposed
   keys can't stand in for softmax over occurrences at the sharpness trained
   heads use. Revised the same day: the sum is lossless, the limit is
   decoding it (compressed sensing), and an exact polynomial-kernel family
   exists (note §5). A far-channel probe (2026-11-11,
   `rnd/far-channel-probe`) put a blurry ordinal channel with linear content
   in layer 2: it recovers 65% of exact attention's gain over the window,
   and is short of exact by a steady 0.010 nats. No next step agreed. The other direction the user named, a Möbius prior plus imm's
   residual models, belongs to @mobius and @imm.
2. **The rest of the orientation proviso** ([open-lines.md](open-lines.md)
   §1): attention, depth 16, canonical eval and seeds for attention are done
   (2026-11-17); GRU seeds, v2's fixed order at pd=1, suffix units at pd=2/3
   and interference are open, whenever the result is built on.
3. **v2 trainer speed:** correctness is done; the kernel work has not started.
   Resume point and acceptance tests: [open-lines.md](open-lines.md).
4. **Curvature that keeps partitioned cadence:** per-subtree L-BFGS histories,
   or stochastic quasi-Newton ([open-lines.md](open-lines.md)).

## Open questions

- Does a curvature method beat partitioned Adam at equal compute in any regime
  (larger models, partitioned histories)?
- What explains AGPT's remaining gap to vanilla minibatch attention? At pd=1,
  mostly the step count (65 per pass). At pd=3, with more steps than
  minibatch training, about half is unequal unit sizes and half the
  grouping by a shared 3-character prefix (one-pass-gap follow-up,
  2026-11-14). The orientation control's tie at pd=1 with a GRU does not
  carry over. Follow-up 2 (2026-11-15): with equal step sizes the
  grouping alone costs 0.27 nats, and the cost grows with the sharing a
  plan keeps; even a conventional epoch beats a pd=3 pass.
- Is the phase-tree gain positional? **No (control, 2026-11-10):** the shift
  cancels in Q·K, and depth positions score the same. The phase modes are
  strided minibatches of 1/64 of the windows per epoch. Whether minibatching
  explains their d128/L6 gain over static is open ([open-lines.md](open-lines.md) §3a).
- Should canonical baselines trained before the 2026-09-24 race fix be re-run?
  **Open: the user's call** (gradient-population README).
- How far one corpus pass gets: measured 2026-11-14 and 2026-11-15 (`rnd/one-pass-gap`).

## Docs that lag the code (checked 2026-11-07)

- `README.md` ("Where the research stands", row 3) and the site's "next" list
  (items 01 and 03) still describe the CUDA backward as incomplete. It was
  fixed in `aa1b6ac`/`88c9335`.
- `rnd/README.md`'s index lists 18 experiments and uses an older status
  vocabulary. The live index is the front matter
  (`python3 src/tools/rnd_front_matter.py index`).
- `CLAUDE.md` points to the Claude memory directory under the project's
  pre-move path.
- `notes/optimization/trust_hpyp_literature.md` still calls the trie's parent
  the HPYP backoff context. The context-tree todo corrects this: that is true
  of the suffix tree, not the prefix trie.
