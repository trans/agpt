# Project split: µGPT and AGPT cleanup inventory

µGPT is a components kit for building LLMs in Crystal/C. AGPT is a separate
research project that depends on µGPT. The two have grown together in one
repo and need to be separated.

Plan: `cp -a microgpt agpt` to a new path, change agpt's origin to a fresh
GitHub repo, then trim each side. This document is the working inventory we
trim against.

## µGPT-only (kept in microgpt/, removed from agpt/)

### Source
- `src/microgpt/` — model, backend, lookahead, router, cooperative
- `src/microgpt.cr`
- `src/cuda/kernels.cu`, `src/cuda/stubs.c` — shared CUDA kernels; µGPT owns them, AGPT links against them via the shard
- `src/cloud/` — Vast.ai GPU rental wrapper
- `src/construction_kit/` — visual SPA + Svelte frontend
- `src/tools/perplexity.cr` — eval any MGPT checkpoint

### Binaries
`microgpt`, `perplexity`, `cloud`, `construction-kit`, `construction-kit-cli`

### Specs
- `spec/microgpt_spec.cr`

### Justfile targets
`build`, `build-release`, `build-cuda`, `build-perplexity`, `build-cloud`,
`build-kit`, `build-kit-cli`, `build-kit-cuda`, `kit`, `kit-cli`, `kit-dev`,
`kit-build-frontend`, `run`, `cloud`, `docs`, `docs-api`, `bench`

### Notes
- `notes/construction-kit/`

## AGPT-only (kept in agpt/, removed from microgpt/)

### Source
- `src/agpt/` — entire directory (trie, radix, samplers, KV store, walkers)
- `src/agpt.cr`
- `src/cuda/agpt_train.cu` — CUDA training engine
- `src/tools/bayesian_posterior.cr`
- `src/tools/convergence.cr`
- `src/tools/radix_verify.cr`
- `src/tools/synth_wrap_corpus.cr`
- `src/tools/trie_profile.cr`
- `tools/check_weights.c` — used by AGPT foundational tests

### Binaries
`agpt_train`, `synth_wrap_corpus`, `radix-verify`, `trie-profile`,
`bayesian-posterior`, `convergence`, `check_weights`

### Specs
- `spec/agpt_backward_attention_spec.cr`
- `spec/agpt_chain_compression_spec.cr`
- `spec/agpt_leveled_trie_spec.cr`

### Tests
- `tests/test_agpt_fundamentals.sh`

### Justfile targets
`build-radix-verify`, `build-trie-profile`, `build-bayesian-posterior`,
`build-agpt-train`, `build-check-weights`, `build-synth-wrap-corpus`,
`test-agpt`, `compare-agpt`

### Notes / data
- `notes/agpt/` — design notes
- Most of `notes/TODO.md` (AGPT-related)
- `data/input.agpt.model` — trained AGPT checkpoint
- `data/synth_wrap_*.txt` — synthesized wrap-around corpora
- `rnd/` — entire research log directory
- `results.tsv` (root) and `data/results.tsv` — research logs

## Cross-cuts (the actual entanglement)

1. **`src/microgpt/main.cr`** — 129 references to AGPT. The single biggest
   cross-cut. All `--agpt-*` flags and the `require "../agpt"` come out on
   the µGPT side. The AGPT side gets a fresh CLI that owns those flags.

2. **`src/microgpt/backend.cr`** — one TODO comment mentions `bin/agpt_train`
   as a counterexample. Trivial reword.

3. **`spec/spec_helper.cr`** — shared. Stays as-is on both sides.

4. **`Justfile`** — `test-crystal` runs all specs; splits naturally with the
   spec files. `test` aggregates `test-crystal` + `test-agpt`; on µGPT side
   becomes just `test-crystal`, on AGPT side stays as the combined runner.

5. **`data/`** — corpora (`input.txt`, `gutenberg*`, `addition.txt`,
   `eval_add.txt`) and base checkpoints (`input.model`, `input.random.model`)
   are useful to both. Copy to both, no shared lineage needed.

## Stale / junk

- `bin/agpt_train_ngram` — binary with no source. Delete from both.

## Open decisions

1. **Dep mechanism (µGPT shard for AGPT)**: Crystal shard via `shard.yml`
   git dep. AGPT's Justfile compiles CUDA kernels from
   `lib/microgpt/src/cuda/kernels.cu`. Verify this works (cross-shard CUDA
   compile is novel) before fully relying on it; could prototype with a
   local path dep first.

2. **AGPT CLI shape**: today is 7 binaries + flags inside `bin/microgpt`.
   Options:
   - Many small binaries (current style).
   - One `bin/agpt` with subcommands (`build-leveled`, `build-radix`,
     `train`, `synth-wrap`, `verify-radix`).

   Leaning toward the subcommand CLI for a cleaner public surface.

3. **Old feature branches** (`agpt-packed-varlen`, `agpt-partition-kv-scoping`,
   `agpt-root-loop`, `agpt-sibling-attention`, `flat-graph`,
   `sgd-sanity-check`, `worktree-agent-*`): which migrate to the new AGPT
   repo, which get archived, which die?

4. **`results.tsv` at repo root + `data/results.tsv`**: dedupe and move
   exclusively to AGPT.

5. **GitHub repo for AGPT**: needs to be created; user creates it, we point
   origin at it.
