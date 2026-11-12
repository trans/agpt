# AGPT and the other AIR projects

*Updated 2026-11-12*

AGPT is one of the research projects overseen by @air.

## @ugpt (microgpt)

- **History:** AGPT was developed inside microgpt until the severance, which
  started 2026-05-22 (`ef3a8e9`). Coupling map:
  `notes/operations/microgpt-separation.md` (a 2026-04-28 snapshot).
- **What AGPT still takes from it:**
  - the `microgpt` shard (`shard.yml`, from GitHub `trans/microgpt` `main`),
    for Crystal primitives such as `Mat`, `MiniGPT`, `Config` and
    `TextDataset`. AGPT treats the shard as frozen; its CUDA kernels and stubs
    are its own copies in `src/cuda/`;
  - the reference binaries `bin/microgpt` and `bin/perplexity`, which
    `just test` (`tests/test_agpt_fundamentals.sh`) needs. Build them in
    microgpt, then copy or symlink them into `agpt/bin/` (`Justfile`).
    `just test-crystal` and the Ultra tests don't need them.
- **Files that may belong to microgpt:** `todo/cublas-generation-bug.md` and
  `todo/cublas-bug-findings.md` are about microgpt's cuBLAS backend
  (generation producing NaN logits), not AGPT training.
- **Memory:** Claude's session memory from the microgpt era holds AGPT history
  from April–May 2026, mostly superseded. @ugpt left the AGPT findings to this
  project.

## @imm

- **Origin:** imm split out of the agpt-ultra prior-residual thread. The brief
  that proposed the split is `rnd/prior-residual-project-reference.md`
  (2026-06-14).
- **Which records here are its history:** `rnd/count-backoff-gate`,
  `rnd/count-prior-residual`, `rnd/gutenberg-prior-residual`, `rnd/rnn-agpt`
  and `rnd/segment-memory`, plus the Ultra track in `research/ultra`
  (imported in `bd680ff`).
- **Where new work goes:** new work on learned count/backoff priors plus
  neural residuals belongs in imm.

## @mobius

- **What it reads:** mobius's examples use AGPT's `data/input.txt` (Tiny
  Shakespeare, 1,115,394 bytes, tracked in git). Moving or changing that file
  affects mobius.
- **What mobius does:** it computes exact corpus counts. These are relevant to
  AGPT's count-prior and backoff ideas, although no shared code exists.
- **Does a count prior still help at scale? (2026-11-11)** The user's view:
  no, because large models learn what the prior holds quickly, so the prior
  is a small fraction of training. @air's test
  (`mail/received/20261111-055621Z-@air--mobius-at-scale-*.md`): imm's char
  transformer, with and without mobius's rows on its logits through a gate.
  Sizes 121k–10.8M, 20M-character PG-19 split, 41 passes. It uses the
  200M-character book prior, and a train-text-only prior as the control.
  - **Results** (`mail/received/20261111-142039Z-@air--mobius-at-scale-*.md`,
    one seed each): the gain over the plain model shrinks with size, to
    between −0.08 and 0.00 bits/char at 10.8M for every form. That covers
    both priors, and the prior on the logits only or also fed into the input.
  - **One curve:** let m be the plain model's held-out bits/char minus the
    prior alone's. Across all 8 runs, the gain is close to linear in m:
    0.176 + 0.706 m on the logits (r 0.985), crossing zero near m = −0.25.
    The books help only by making the prior better.
  - My first prediction, drafted and never sent, said the books prior would
    still gain +0.03 to +0.10 at 10.8M. It was wrong (−0.001).
  - **Sent to @air 2026-11-11:** a reply with the curve, a prediction for more
    text, and a test for the 10.8M overfitting (prior dropout). The first
    draft was dropped unsent.
  - **With the prior taken away (@air, 2026-11-12,**
    `mail/received/20261112-013330Z-@air--the-prior-taken-away-*.md`): the
    user asked whether the network had learned what the prior offers. It
    hadn't: it learned only what the prior gets wrong. A fresh output layer
    on the frozen state of a books-prior network, with no prior, scores
    held-out PPL 10.57 at 121k and 5.94 at 10.8M. The plain 10.8M scores
    3.0386, so the state is 0.97 bits/char behind. Same-seed reruns differ by
    about 0.01 bits/char at 10.8M.
  - **Prior dropout 0.5 helped** (air `bef9142`; read from
    `experiments/prior_scaling/runs/pd_*` and `probe_*_pd*`, 2026-11-12;
    books prior, one seed). At 10.8M it gained +0.024 bits/char (logits)
    and +0.051 (input) over the plain model, where without dropout the
    gains were −0.001 and −0.013. The network alone recovered: a fresh
    output layer with no prior scores 3.0567 at 10.8M (plain 3.0386; 5.94
    without dropout). At 1.84M it scores 3.4802, better than the plain
    1.84M (3.5678). At 1.84M the full-system gains fell (0.161 → 0.133,
    0.204 → 0.169). My prediction (a tie at 10.8M) was wrong. **Open:** no
    run yet separates the books' extra text from regularization (a no-books
    dropout run at 10.8M would). @imm is trying dropout too (the user,
    2026-11-12).
    - @air's own write-up: `mail/received/20261112-044356Z-@air--prior-dropout-*.md`.
      Its 1.84M plain rerun from the same script (3.5678, air `aae6754`)
      confirms that the dropout network alone beats the plain 1.84M
      (`mail/received/20261112-050938Z-*.md`).
    - **The user's framing (2026-11-12):** the prior is a data resource, not
      a model. Its value is how much more information it can feed, how much
      faster, than an equivalent model could learn. @air's matching test,
      not queued: a plain 10.8M trained on the prior's 200M book characters
      plus the 20M, for the same 50k steps, against the dropout model.

## @semtrace

No known relation.

## Codex sessions

- **What Codex did here:** a Codex agent worked in this same checkout:
  - much of the v2 CUDA trainer (`src/cudax`);
  - the phase-RoPE line;
  - the site and its experiment-archive generator.
- **How to tell its commits apart:** Claude's commits carry a
  `Co-Authored-By` trailer. A commit without one may be Codex's or the
  user's own; for example, `2b913be` landed the codex-agpt v2 work. Of the
  33 commits touching `src/cudax`, 8 carry Claude's trailer.
- **How it was reached:** as of 2026-09-29, site edits went to it over the
  Arcana bus as `@agpt-codex` (per-machine memory). Check that this still
  holds before relying on it.
