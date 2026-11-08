# Methodology: evaluation pitfalls and working rules

*Updated 2026-11-07*

Each rule here cost something to learn. `CLAUDE.md` has the settled rules on
the runner, the canonical PPL and the splits; they are not repeated here.

## Reading a number

- **Rolling and fixed-window PPL can rank arms differently.**
  - Canonical rolling byte PPL scores the first tokens of each window with
    little context. Fixed-window PPL gives every scored token the full window.
  - A pd≥2 plan never trains the targets above depth pd, so rolling PPL
    punishes it. In gradient-population Exp 2, pd=3 had the best fixed-window
    PPL (4.591) and the worst rolling PPL (7.043).
  - Use fixed-window to compare cadences. A canonical claim needs the shallow
    targets trained as well.
- **Trainer eras.** Compare numbers only within an era, or after a rerun:

  | Before | What was wrong | Fixed in |
  |---|---|---|
  | 2026-04-23 | Wk, Wv and all biases frozen | `1c858c0` |
  | 2026-05-26 | loss and normalizer deviated from the paper | `0252072` (v1), `816f7d0` (v2) |
  | 2026-09-24 | about 0.3% of per-query losses corrupted by a reduction race | `27ee367` |
  | 2026-09-28 | attention runs used a truncated ancestor gradient | `aa1b6ac` |

  The site generator derives all but the first of these as caveats.
- **Match compute by traversals.** An L-BFGS pass, an Adam epoch and a
  full-trie gradient each cost one traversal, and warm-start epochs count. Two
  claims were withdrawn on 2026-09-28 for skipping this (`1acf75c`,
  `829a7b4`), including Exp 6's "L-BFGS 1.523 vs Adam 1.571".
- **Diagnostic numbers are not canonical.** This covers the PyTorch tools
  (context-tree, L-BFGS test, pool) and the seq=8 sliding-window number from
  `agpt_recur_perplexity`. Label them as diagnostic. In one check on the same
  checkpoint, the seq=8 number came out about 12% below the canonical one
  (2026-06-08).
- **Held-out hygiene.** In May 2026, held-out files turned out to be slices of
  the training text (`todo/proper-ppl-heldout-methodology.md`). Use carved
  splits (`bin/agpt_carve`) or an external file, and check the split SHA in
  `result.json`.
- **Vocabulary comes from text.** Set `corpus.vocab_source: data/input.txt`,
  so that every tool assigns the same token ids (`docs/yaml-schema.md`).
- **Don't judge a model change by PPL alone.** Sample generations as well; the
  two have disagreed before.
- **Small units are mostly noise.** The trainer fires each unit's event-mean
  gradient, so a 15-event unit steps as hard as a 400k-event one. Unweighted
  per-unit statistics, coherence among them, are dominated by tiny units. Use
  event weighting with a mass floor (Exp 1, quartile table).

## Gradient work

- **Check gradients numerically before trusting any analytic argument.**
  - `src/tools/agpt_grad_check.py` compares finite differences along the
    gradient and along chosen directions. Target an FD ratio of 1.000 ± 0.005
    at ε 0.01.
  - TF32 adds about 3e-5 of noise to the loss. For small-ε checks use
    `bin/agpt_train_v2_kvfp32` (`just build-agpt-train-v2-kvfp32`) with
    `NVIDIA_TF32_OVERRIDE=0`.
- **The L-BFGS backtrack rate is a sensitive end-to-end check.** A gradient
  error of a few percent, invisible to Adam, made trie L-BFGS backtrack on 44%
  of its steps (Exp 7).
- **Frozen-weight dumps should agree to about 1e-5.** That is the floor set by
  the backward's atomics (`todo/deterministic-backward.md`). Anything larger is
  a bug: the reduction race was found this way.
- **A fix that needs a swept scale factor means the math is wrong.** A correct
  gradient has α = 1 (the anc-grad normalizer, `f6366bb`).
- **No chunk statistics in the loss or gradient math.** Chunks are a memory
  layout (the per-fire normalizer, `609e7ab`).
- **When optimizing execution, keep the update cadence fixed.** The execution
  unit (a chunk or a batch) is not the update unit (a partition). This matters
  for the planned speed work.
- **When data smells, audit the code before adding seeds.** Examples are a
  same-seed inconsistency or a non-monotonic curve. A "protocol mismatch" that
  happens to close a gap is usually a bug.

## Interpreting results

- **Name a negative result after the version that was tested.** One failed
  design does not close the idea behind it; the front matter's `answer` fields
  are written this way.
- **The per-node update in pool or population experiments is plain SGD on one
  window** (`src/tools/agpt_pool_sgd.py`). Pre-aggregated pd=k units hide the
  conflict inside each subtree and cap the depth.

## Operations

- **Persist every run under `rnd/<exp>/` before describing it.** Runs and
  session context have both been lost before.
- **Run `compute-sanitizer` with nothing else on the GPU.** Run alongside
  another GPU job, it froze the development machine on 2026-09-24.
- **RunPod (`rnd/runpod`, `rnd/docker`):**
  - never `rsync --delete` to a pod with experiments running;
  - test the sshd image over ssh, not with a plain container run.
- **Writing:** say rootward and leafward, never up or down the tree. The user
  asked for this on 2026-10-03, and the context-tree todo follows it.
