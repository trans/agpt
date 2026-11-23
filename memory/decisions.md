# Decisions and conventions in force

*Updated 2026-11-23*

These are settled unless marked otherwise. The person who decided is named
only where a source says so.

## Trainer

- **The exact ancestor backward is the default** (`88c9335`, 2026-09-28).
  - `anc_grad_exact` is on whenever `train.anc_grad` is.
  - In configurations it does not support, it switches itself off with a
    notice. An explicit `true` there is an error.
  - `experimental.anc_grad_exact: false` exists only to reproduce older runs.
  - Verified in `src/cudax/types.cuh` and `agpt_train_v2.cu`.
- **The microgpt severance** (started 2026-05-22, `ef3a8e9`).
  - The CUDA kernels and stubs in `src/cuda/` are AGPT's own copies.
  - The `lib/microgpt` shard is frozen and used only for Crystal primitives.
  - Kernel work happens here (see the `Justfile` header).

## Experiment records

- **Reportable numbers need a `result.json` from `bin/agpt_experiment`**
  (commitment of 2026-05-25). See `CLAUDE.md`.
- **Front matter on every `rnd/<exp>/README.md`** (`eaf4c6c`, `c81ab8f`,
  2026-09-28).
  - Schema: `notes/operations/experiment-front-matter.md`.
  - Check it with `python3 src/tools/rnd_front_matter.py validate`.
  - Write it when the experiment is created.
  - The `family` field keeps one experiment set across attention, recurrent,
    count-prior and hybrid models.
  - The trainer-era caveats are derived from run dates by
    `src/tools/build_experiment_pages.py`. Never type them by hand.
- **Branch-only experiments** get a stub README with `code: {branch, tag}` and
  an `exp/<name>` tag.
  - Never move a tag that has been pushed; add a new one. The `-final` tags
    were added this way on 2026-09-28 (`4355c02`).
- **Site, section 03 ("The challenges"):** each card states the problem and
  what success would look like, with no current approaches or results.
  Results belong in the experiment archive.
  - The user decided this on 2026-09-29 (`f31309f`).

## Research direction

- **Context-tree orientation is called: a minor breakthrough, with a
  proviso.** On 2026-10-03 the user asked that it not be settled on one GRU
  run. On 2026-11-10 they called it, labelled it a minor breakthrough, and
  attached a proviso to double-check and polish it. The double-check list is
  in `rnd/context-tree-orientation/README.md` ("Proviso"). Their reading:
  the prefix trie was the right choice from the start, for reasons nobody had
  seen, and further gains need something past the tree, though they suspect
  something is still being missed.
- **Next directions (the user, 2026-11-10):** a Möbius prior with imm's
  residual models (the @mobius and @imm projects), and, for this project,
  multi-positional (superposed) RoPE ([open-lines.md](open-lines.md) §3a).
- **Small, fast comparisons first (the user, 2026-11-14).** Compare model
  designs with few parameters on a small corpus, so comparison runs and
  tuning take minutes. Read scale from the learning curves. Run a larger
  model only to demonstrate something that already looks significant at
  small scale. This followed the 220M and 1 GB plain-transformer runs of
  2026-11-13 (about an hour each), which the user judged too much time on
  ever-larger models.
- **Regular training, not pd=N trie training (the user, 2026-11-23).** The
  user: "yes, record it", to moving from AGPT's pd=N trie training to
  ordinary transformer training augmented with what AGPT taught. The record
  behind it: at equal wall time pd=N trails minibatch (`rnd/one-pass-gap`:
  8.71 after 54 s against 4.83 after 31 s), and the transferable results
  run without the trie (noise cadence with no trie, level with minibatch in
  half v2's time, `rnd/noise-cadence`; doped counts; dynamic evaluation,
  `rnd/dynamic-eval`). The trie stays as the data structure behind the count
  prior (`src/tools/doped_counts.py`), not as the trainer.
- **agpt serves imm's prior + residual (the user, 2026-11-22).** imm is
  agpt's spin-off that keeps the (doped) trie at inference; its mission is
  `prior + residual`, and a regular transformer, an agpt-improved one, or
  imm's long-range scans and stats inputs are welcome if they help it. agpt
  ideas are judged by whether they help the residual over the doped prior.
  If training targets and prior use the same doped counts, the residual has
  nothing to learn: the doping goes in the prior, cross-fitted as mobius
  does its training rows. The prior's SVD code in the input (as imm and
  mobius use) can wait (the user).
- **The precondition branch was archived, not merged** (`df75f96`,
  2026-06-02).

## Pending

- **Re-running the canonical baselines trained before the 2026-09-24 race
  fix.** **Open, the user's call** (`rnd/gradient-population/README.md`, "Trainer
  bug found on the way").
- **The REMOVE list in `rnd/TRIAGE.md`.** **Open, undecided.** Its review note
  of 2026-09-28 leans toward keeping those directories as provenance.
