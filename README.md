# AGPT — Aggregated Gradient Pre-training

> **One epoch and done.**

That is the research goal: could a language model learn enough from one pass
over a corpus to match a conventional training run, while using substantially
less time? AGPT investigates whether sharing computation across repeated text
prefixes and using more informative updates can move us toward that goal.
**One-epoch parity and a 10× speedup have not been demonstrated.**

This is an active research repository by Thomas Sawyer. It contains a
Crystal/CUDA trainer, a smaller PyTorch research track, experiment records,
and a [paper draft](docs/paper.md). A [static project page](site/README.md) is
also in progress. The experiments include negative results and corrections to
earlier evaluation methods.

## The idea

Many corpus windows share a beginning. For example, `cast`, `case`, and `cash`
all reuse `cas` before branching. A prefix trie stores that common path once.
At each node, next-token counts determine a count-weighted prediction loss;
descendant gradients can be summed before passing through shared prefix
computation. The [paper draft](docs/paper.md) develops the factorization and
the proposed training regimes.

```text
separate paths                  shared prefix
cas → t                         cas ┬→ t
cas → e                             ├→ e
cas → h                             └→ h
```

The mathematical reorganization does not by itself make a trainer fast or
accurate. Optimizer update frequency, sparse deep contexts, memory use, and
correct gradient propagation all matter in practice.

## Where the research stands

| Finding | Evidence | What it means |
| --- | --- | --- |
| Partitioning the trie changed learning markedly in a historical control. | In a matched depth-16 Shakespeare control, 100 epochs of one update per root child reached **5.34 rolling byte PPL**; one update per full trie reached **12.05**, with about the same training time. [Original runs](rnd/stochastic-agpt/README.md) predate a CUDA kernel race fix. A [post-fix root-child run](rnd/gradient-population/20260924T205951-cadence-pd1-100ep/result.json) reached **5.348**; the full-trie arm has not been rerun. | Sharing work also reduces optimizer updates. The full-trie comparison remains provisional and is not a comparison with a conventional trainer. |
| Gradient directions increasingly cancel when large subtrees are aggregated. | Event-weighted coherence fell from **0.719 at initialization** to **0.130 after 100 epochs** in a frozen-gradient diagnostic. [Method and results](rnd/gradient-population/README.md) | The one-step-per-epoch optimizer needs a better update rule. This diagnostic is not a perplexity benchmark. |
| Exact gradients make curvature methods promising in a controlled PyTorch test, but the CUDA implementation has a backward mismatch. | [Experiments 6–7b and finite-difference check](rnd/gradient-population/README.md) | Completing the ancestor backward path is a prerequisite for testing that promise in the trie trainer. No end-to-end speed claim follows yet. |

The paper's original empirical section was retracted after evaluation and
training-objective corrections. Its current version is a **theory and systems
draft**, with empirical claims still being rebuilt under the standard rolling
evaluation protocol. The [experiment index](rnd/README.md) and
[methodology triage](rnd/TRIAGE.md) preserve the history behind these changes.

## Research tracks

- **[CUDA AGPT](src/cudax/README.md):** radix-trie training with a transformer,
  partitioned optimizer steps, and a provenance-aware experiment runner.
- **[AGPT Ultra](research/ultra/README.md):** a Python/PyTorch track imported
  from a separate repository with its Git history. It explores exact prefix
  reuse in small recurrent models, Fisher-based updates, and count-prior
  residuals. Its [experiment record](research/ultra/EXPERIMENTS.md) explains
  why historical Ultra and CUDA perplexity numbers are not directly comparable.
- **[Research notes](notes/README.md) and [experiments](rnd/README.md):** design
  decisions, tests, failures, and follow-up questions.

## Build and verify

The Crystal tools require Crystal 1.19.1 or newer and `shards`. The CUDA
trainer additionally requires `nvcc`, cuBLAS, and a supported NVIDIA GPU.
The Python research track requires Python 3.11 or newer with PyTorch.

```sh
shards install
just build-agpt-train-v2 build-agpt-experiment
just test-crystal
(cd research/ultra && python3 -m unittest discover -s tests -q)
```

`just test` also runs the foundational parity tests, which require the
reference `microgpt` and `perplexity` binaries from the
[µGPT](https://github.com/trans/microgpt) project. For reportable CUDA runs,
use `bin/agpt_experiment` with a YAML configuration; see the
[schema and runner workflow](docs/yaml-schema.md) and a
[documented run](rnd/stochastic-agpt/README.md).
The runner saves configuration, provenance, and evaluator results together.

## License

Code in this repository, including the imported AGPT Ultra track, is released
under the [PolyForm Noncommercial License 1.0.0](LICENSE). See
[commercial licensing](COMMERCIAL_LICENSE.md) for other uses.
