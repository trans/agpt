# Handoff: where AGPT stands

*Updated 2026-11-14*

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
- **Next, offered, not chosen:** equal-mass units (split heavy subtrees,
  pack light ones) for the size half; mapping the grouping half against
  the compute that prefix sharing saves.
- `src/tools/agpt_minibatch_train.py` (`84054bc`) is the minibatch baseline
  as an orchestrator custom trainer, so baselines get `result.json` too.

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
2. **The orientation proviso checks** ([open-lines.md](open-lines.md) §1),
   whenever the result is built on.
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
  carry over.
- Is the phase-tree gain positional? **No (control, 2026-11-10):** the shift
  cancels in Q·K, and depth positions score the same. The phase modes are
  strided minibatches of 1/64 of the windows per epoch. Whether minibatching
  explains their d128/L6 gain over static is open ([open-lines.md](open-lines.md) §3a).
- Should canonical baselines trained before the 2026-09-24 race fix be re-run?
  **Open: the user's call** (gradient-population README).
- How far one corpus pass gets: measured 2026-11-14 (`rnd/one-pass-gap`).

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
