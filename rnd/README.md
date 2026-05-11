# rnd/ — Research Experiment Policy

`rnd/` is where experiment history lives on `main`.

The goal is not to keep every artifact. The goal is to keep each experiment's
story reproducible and understandable:

- what was tried
- what code it depended on
- how to rerun it
- what was learned

Use [EXPERIMENT_TEMPLATE.md](/home/trans/Projects/agpt/rnd/EXPERIMENT_TEMPLATE.md:1)
when starting or backfilling an experiment directory.
See [CHRONOLOGY.md](/home/trans/Projects/agpt/rnd/CHRONOLOGY.md:1) for the
approximate order in which the experiment threads appeared.

## Required structure

Each experiment should live in its own directory:

```text
rnd/
  <experiment-name>/
    README.md
    reproduce.sh          # preferred if rerunning is nontrivial
    findings.md           # preferred for closed experiments
    logs/                 # optional raw stdout/stderr
    results/              # optional derived summaries / tables / plots
```

### Required files

- `README.md`
  This is required for every real experiment directory.
- `reproduce.sh` or equivalent script(s)
  Required if the run recipe is longer than a few commands.
- `findings.md`
  Strongly preferred when the experiment is closed and the README is no longer
  the best place for the conclusion.

### README minimum contents

Every experiment `README.md` should answer these four questions:

1. What was the hypothesis?
2. What code / branch / commit did it depend on?
3. How do I reproduce it?
4. What did we learn, or what remains open?

If a directory does not answer those four questions, it is not in good shape.

### Trainer-era note

For experiments that depend on AGPT trainer behavior, add a short note near the
top clarifying one of:

- `post-fix`
- `pre-fix, needs reassessment`
- `not obviously trainer-dependent`
- `unclear`

This is about the AGPT trainer state after commit `1c858c0` (2026-04-23),
which fixed frozen `Wk` / `Wv` / bias gradients. It is not a lifecycle status;
it is an epistemic note about how much confidence to place in old
trainer-dependent results.

## Status vocabulary

Use one of these statuses near the top of each experiment README:

- `active`
  Current line of work.
- `closed`
  Investigation completed; conclusion written down.
- `incomplete`
  Started but not properly concluded or reproducible yet.
- `log-only`
  Raw outputs exist, but the experiment still lacks a proper front-door writeup.
- `archived`
  Kept for record, not expected to be resumed.

## Branch / code policy

- Code changes for an experiment may live on a branch.
- The experiment `README.md` should name the branch and relevant commits when
  that matters.
- `rnd/` on `main` is for notes, scripts, summaries, and reproducibility
  metadata, even if the code itself lived elsewhere.

## Artifact policy

Default rule: commit the explanation, not the bulk output.

- Keep small summary artifacts when they are the clearest record:
  tables, tiny plots, compact findings documents, short result summaries.
- Do not commit large generated artifacts by default:
  checkpoints, synthesized corpora, big temporary tries, bulk logs, cache-like
  intermediates.
- If a large artifact is committed anyway, mark it clearly in the README as one
  of:
  - `canonical`
  - `hard-to-reproduce`
  - `temporary / should be removed`
- Prefer reproducible scripts plus ignored output paths over committed bulk data.

## Current index

This is the current top-level map of `rnd/`.

| Directory | Status | Trainer note | Summary |
|---|---|---|---|
| [agpt-optimizers](agpt-optimizers/) | closed | post-fix | Optimizer sweep showing subtree AGPT needs adaptive optimization; SGD/momentum plateau far above RMSProp. |
| [blending](blending/) | closed | pre-fix, needs reassessment | Suffix-depth blending at radix endpoints; reported a d=16 win under the pre-fix trainer. |
| [convergence](convergence/) | closed | not obviously trainer-dependent | Trie path probability convergence work. |
| [gutenberg-5m](gutenberg-5m/) | closed | not obviously trainer-dependent | Wrap-around scaling and larger-corpus builder notes. |
| [hotspot-curriculum](hotspot-curriculum/) | incomplete | likely post-fix, but needs writeup to confirm scope | Adaptive subtree-splitting curriculum showed a modest win at moderate coverage, but the thread did not fully close out. |
| [lightning-cap-warmup](lightning-cap-warmup/) | closed | post-fix | Lightning L3 cap / warmup investigation; explicitly closed. |
| [lightning-training](lightning-training/) | closed | pre-fix, needs reassessment | Lightning training empirical sweep and scaling notes. |
| [mass-conservation](mass-conservation/) | closed | not obviously trainer-dependent | Suffix-tree mass conservation findings. |
| [p2s-attention](p2s-attention/) | closed | not obviously trainer-dependent | Prefix-to-suffix attention investigation; architecture closed out. |
| [post-fix-baseline](post-fix-baseline/) | closed | post-fix | Baseline re-establishment after trainer fixes. |
| [radix-saturation](radix-saturation/) | active | pre-fix, needs reassessment | PPL vs radix saturation / depth behavior. |
| [root-loop](root-loop/) | closed | pre-fix, needs reassessment | Virtual-tree / root-loop experiment; negative at K>1 under the pre-fix trainer. |
| [sgd-sanity-check](sgd-sanity-check/) | incomplete | pre-fix, needs reassessment | Important experiment question; old run set should be rerun later on the corrected trainer. |
| [sgd-ceiling](sgd-ceiling/) | closed | post-fix | Comparison separating optimizer effects from aggregation effects; most of AGPT's edge over SGD came from the stronger recipe, with a smaller extra aggregation gain. |
| [sparsity-profile](sparsity-profile/) | closed | not obviously trainer-dependent | Depth-by-depth sparsity characterization. |
| [trie-attention-framing](trie-attention-framing/) | closed | post-fix | Decision/identity decomposition of the trie; descriptive predictions held, but routing and decision-only operationalizations stayed neutral-to-negative. |
| [unary-pruning](unary-pruning/) | closed | not obviously trainer-dependent | Mass-1 unary-path pruning investigation. |
| [wrap-around](wrap-around/) | closed | not obviously trainer-dependent | Wrap-around synthesis experiment family and headline recipe. |

## Practical rule

Before starting a new experiment, create the directory and fill in the template
first. It is cheaper to enforce structure up front than to reconstruct it from
logs later.
