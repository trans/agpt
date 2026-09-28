# Experiment front matter

Every experiment directory `rnd/<exp>/` has a `README.md` that begins with a
YAML front-matter block. The block is the machine-readable summary the site
generator turns into a per-experiment page and an index; the prose below it
stays free-form. GitHub renders the block as a table.

Experiments whose code lives only on a branch get a stub directory
`rnd/<exp>/README.md` with the same block and `code: {branch, tag}`.

## Schema

```yaml
---
title: Partition depth                  # required. Short display name.
kind: experiment                        # required. See vocabulary.
status: concluded                       # required. See vocabulary.
outcome: positive                       # required. See vocabulary.
question: >-                            # required. One sentence: what the experiment asks.
  Does firing Adam once per depth-N prefix group speed convergence?
answer: >-                              # required when status is concluded; "" otherwise.
  Yes, up to pd=6 (PPL@32 5.39 -> 3.82); pd=7 breaks down.
opened: 2026-04-30                      # required. First commit touching the directory.
updated: 2026-05-01                     # required. Last substantive change.
code: main                              # required. "main", or {branch: <name>, tag: exp/<name>}.
eval: legacy                            # required. See vocabulary.
family: attention                       # required. Which f_theta the experiment is about. See vocabulary.
headline:                               # optional. At most three numbers the site shows.
  - label: Adam pd=6, 3 epochs          #   what was run
    metric: legacy PPL@32               #   metric AND protocol, so it is never ambiguous
    value: 3.82
    run: <run-dir>                      #   optional: orchestrator run directory it comes from
tags: [optimizer, partitioning]         # optional. Topic tags for filtering.
related: [granularity-redundancy]       # optional. Other rnd/ directory names.
superseded_by: []                       # optional. rnd/ directory names.
---
```

## Vocabularies

`kind`
- `experiment` — tests a hypothesis with training runs.
- `diagnostic` — measures a property of the trie, trainer or model; no hypothesis about training quality.
- `baseline` — establishes a reference number other work compares against.
- `design` — a written design or plan; nothing run yet.
- `infrastructure` — tooling, environments, runners (docker, runpod, smoke tests).

`status`
- `planned` — designed, not run.
- `active` — runs in progress or the question is still open.
- `concluded` — the question was answered (for infrastructure: the tool is done).

`outcome`
- `positive` — the hypothesis held / the change helped.
- `negative` — it did not.
- `mixed` — helped under some conditions, hurt or null under others.
- `inconclusive` — ran, but could not decide (noise, confound, abandoned mid-way).
- `n/a` — not a hypothesis test (design, diagnostic, baseline, infrastructure), or still active.

`family` — the model family (AGPT is a framework; f_theta is a choice)
- `attention` — transformer f_theta (the CUDA v1/v2 trainers, vanilla attention LMs).
- `recurrent` — recurrent f_theta: linear, tanh-Elman, GRU, and recurrent state models.
- `count-prior` — count/n-gram/backoff models with no learned f_theta (KN, KenLM, trie priors).
- `hybrid` — a count prior combined with a learned model (prior + neural residual, gated mixes).
- `n/a` — no model involved: trie/corpus statistics, infrastructure, pure design.

Experiments from the recurrent PyTorch track (formerly agpt-ultra) live in the same
`rnd/` set as everything else; their code stays in `research/ultra/` and their entries
point into it. Its historical numbers use track-local protocols, so they are `eval: legacy`.

`eval`
- `canonical` — numbers come from `bin/agpt_experiment` `result.json` (lm-eval byte perplexity).
- `legacy` — numbers from pre-orchestrator tools (`bin/perplexity`, `agpt_ppl.py`,
  sliding-window PPL); not comparable with canonical.
- `none` — no perplexity numbers (design, diagnostic without PPL, infrastructure).

## Derived caveats (not typed in)

The generator derives these from run dates in `result.json` / `meta.json` and the
directory's dates, so READMEs need not repeat them:

- **pre-race-fix** — trained before 2026-09-24 (commit 27ee367): ~0.3% of per-query
  losses were corrupted by a missing `__syncthreads()`.
- **truncated-ancestor-gradient** — attention AGPT trained before 2026-09-28
  (anc_grad_exact default): the ancestor K/V gradient stopped at Wk/Wv.
- **pre-loss-fix** — trained before 2026-05-26 (commit 816f7d0), the loss-normalization
  fix tracked in `rnd/TRIAGE.md`.
- **legacy-eval** — see `eval: legacy`.

## Rules

- `question` and `answer` are plain statements, not marketing. Put numbers in
  `answer` only with their metric and protocol.
- `headline` values must appear in a text file of the directory (README, notes,
  result/eval json, tables, logs); `rnd_front_matter.py validate` checks this.
  Never compute new numbers for the front matter.
- Unknown is written as `""` plus a `# VERIFY` comment, never guessed.
- Rerunning an experiment on newer code adds a run directory; update `updated`,
  `answer` and `headline` if the conclusion changes, and say so in the prose.
