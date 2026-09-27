# Experiment records and comparison boundary

This directory was imported from the former `agpt-ultra` repository at commit
`a3f339d` (2026-06-14), with its full Git history. The code and
[append-only journal](notebook/journal/2026-06-13.md) are part of the research
record. The original checkout's ignored `runs/` directory and generated
`data/.splits/` were not imported; they are large, local outputs.

The Python scripts usually write run-ID-prefixed CSV files and checkpoints to
`runs/`. The journal records configurations, selected results, failures, and
revisions to interpretations. It predates AGPT's current
[`bin/agpt_experiment`](../../docs/yaml-schema.md) workflow, which stores a resolved
configuration, source and corpus hashes, split details, and canonical evaluator
metrics in each run's `result.json`.

Treat Ultra's historical numbers as **exploratory within that track**. Some
used carved or sampled validation; others used full held-out text, different
context lengths, or different scoring rules. Their perplexities must be read
with the protocol beside each result and should not be placed on the same axis
as CUDA AGPT `lm-evaluation-harness` rolling byte perplexity without a fresh,
matched evaluation.

For future reportable comparisons, keep the exact command or configuration,
code commit, corpus and split hashes, model seed, evaluator definition, output
artifact, and hardware/runtime together. The existing orchestrator is the
default for CUDA AGPT runs. The Python track needs an adapter or an equivalent
run record before its new results can use that same standard.

Start with the [track README](README.md) for runnable examples and the
[notebook](notebook/README.md) for the history of hypotheses and results.
