# Handoff: where AGPT stands

*Updated 2026-11-10*

## Where we were

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
   No next step agreed. The other direction the user named, a Möbius prior plus imm's
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
- What explains AGPT's remaining gap to vanilla minibatch attention? Not the
  units' target correlation: at equal steps they tie shuffled minibatches
  (orientation grouping control). Step count is the plainest candidate.
- Is the phase-tree gain positional? **No (control, 2026-11-10):** the shift
  cancels in Q·K, and depth positions score the same. The phase modes are
  strided minibatches of 1/64 of the windows per epoch. Whether minibatching
  explains their d128/L6 gain over static is open ([open-lines.md](open-lines.md) §3a).
- Should canonical baselines trained before the 2026-09-24 race fix be re-run?
  **Open: the user's call** (gradient-population README).
- How far one corpus pass gets is still unmeasured (item 04 of the site's
  "next" list).

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
