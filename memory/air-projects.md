# AGPT and the other AIR projects

*Updated 2026-11-11*

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
  @air asked us for a prediction. **Draft held for the user.**

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
