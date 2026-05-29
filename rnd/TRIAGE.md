# rnd/ triage — 2026-05-26

Working doc. User cleans up by hand; this file is the reference for
what's in each category and why.

## Why we're triaging

The AGPT loss kernel deviated from paper §2 in two places, only fixed
on 2026-05-26:

- v1 (`bin/agpt_train`) — fixed in `0252072`
- v2 / CUDAX (`bin/agpt_train_v2`) — fixed in `816f7d0` (by Codex)

The deviations were:

1. Loss kernel computed `-Σ_x (cnt/N_p) log p_x` (per-occurrence
   normalized) instead of `-Σ_x cnt log p_x` (count-weighted).
2. Fire-level normalizer divided d_grads by `fire_events` (query
   count = T_q) instead of `fire_mass` (Σ N_p = total event mass).

Net result of both bugs together:
```
old (buggy):  grad = Σ_q (softmax − empirical) / T_q          ← per-prefix-averaged
new (paper):  grad = Σ_q N_p · (softmax − empirical) / Σ_q N_p ← per-event-mean
```

Standard transformer training divides total loss by total token count
(per-event mean). Pre-fix AGPT was dividing by total unique-prefix
count instead — a much smaller number, biased against high-mass
shallow prefixes that have many corpus occurrences but only one trie
node. The fix restores the per-event-mean form.

Empirical confirmation that the bug was real:

| run | ctx=1 PPL | ctx=16 PPL |
|---|---:|---:|
| bigram baseline | 12.06 | — |
| v1 100 SE buggy | 45.55 | 6.36 |
| v1 100 SE fixed (`--mass-weight linear --fire-norm-mass`) | 13.63 | 8.01 |

Shallow PPL was ~4× worse than even a bigram with the bug. Fix
brought it to the empirical floor.

**Everything that reported a PPL number before 0252072/816f7d0 is
measuring a different objective than the paper's.** It's not
necessarily "wrong" — it tells you what the model learned under the
pre-fix loss — but it's not directly comparable to numbers under the
fixed loss, and several "AGPT beats baseline" claims may shrink or
reverse once measured under the corrected objective.

## Categories

### REMOVE

One-offs, smoke tests, superseded scaffolding. Safe to `git rm -r`.

| Dir | Note |
|---|---|
| `_smoke/` | smoke tests (gitignored anyway; clean any tracked configs/README) |
| `cudax-exposure-fix/` | Codex's in-flight v2-fix work — superseded by `816f7d0` |
| `cudax-static-epochs/` | Codex fix-verification |
| `cudax-section2-progressive/` | Codex fix-verification |
| `cudax-growth-heldout-rerun/` | Codex earlier rerun against unfixed code |
| `post-fix-baseline/` | Codex fix-verification |
| `lr1e-5-probe/` | one-off LR probe |
| `microgpt-cublas-verify/` | one-off cublas determinism check |
| `sgd-sanity-check/` | one-off |
| `v2-compare/` | one-off Codex v1-vs-v2 comparison |

### AFFECTED-BY-BUG (results invalidated)

PPL numbers were measured with the pre-fix unweighted / per-prefix
loss. Optional: tag each README with a "RESULTS INVALIDATED
2026-05-26 — see TRIAGE.md" header pointing at the fix commits. Most
of these don't need re-running; the few that ARE worth re-running
(direct-relevance experiments on weighting and fire-norm) are flagged.

| Dir | Note |
|---|---|
| **dmodel-scaling-100se/** | "project best 3.6899 on Gutenberg" — invalidated |
| **progressive-growth-sgd-comparison/** | the comparison-table investigation; the paper's headline-equivalent |
| **shake-small-baseline/** | first orchestrator baseline |
| **v1-vs-v2-comparison/** | today's investigation; mw-linear runs ARE on the fix path but had wrong fire-norm — see note below |
| `gutenberg-5m/`, `gutenberg-anc-sweep/`, `gutenberg-pd-sweep/` | Gutenberg PPL claims |
| `legacy-rebaseline/`, `agpt-epoch-scaling/` | older PPL comparisons |
| **depth-weight/** | **directly relevant — redo under fixed loss** |
| **composite-weights/** | **directly relevant — redo** |
| **per-fire-norm/** | **directly relevant — redo (fire-norm-mass now default)** |
| **mass-conservation/** | **directly relevant — redo** |
| `anc-grad/`, `cudax-anc-grad-parity/` | anc-grad PPL claims |
| `cap-folding/`, `blending/` | architectural variants with PPL claims |
| `streaming-agpt-v1/` | streaming results |
| `granularity-redundancy/`, `partition-depth/` | training-cadence sweeps |
| `per-rc-adam-v1/` | optimizer variant |
| `subtree-dropout/`, `hotspot-curriculum/` | training schedules |
| `root-loop/` | K=2 virtual-tree result |
| `convergence/`, `beta2-diagnostic/`, `sparsity-profile/` | diagnostics with PPL |
| `lightning-cap-warmup/`, `lightning-training/` | lightning sampling |
| `overnight-2026-05-18/`, `radix-saturation/` | older study runs |
| `sliding-window-v1/`, `seq-len-decouple/`, `sgd-ceiling/` | older comparisons |
| `agpt-optimizers/` | optimizer sweep |
| `cudax-growth/` | earlier growth work |

#### Note on `v1-vs-v2-comparison/`

This experiment was run on 2026-05-26, BEFORE the default flip in
`0252072`. The mw-linear runs (e.g. `v1-100SE-mwlinear`) had
`--mass-weight linear` but the default fire-norm at that time was
`--fire-norm-events`. So those runs were effectively training with
LR scaled by `mean_mass_per_query` (~10–30×). The depth-1 = 13.63
result still held (and the directional signal was real) but a
re-run with the new defaults (both `--mass-weight linear` and
`--fire-norm-mass` on) is the clean baseline. The unweighted runs
in the same dir (without `-mw*` suffix) are the pre-fix legacy.

### KEEP

Not bug-affected (corpus diagnostics, external baselines, architectural
notes, infra) or null results where the bug-affected number doesn't
change the conclusion.

| Dir | Note |
|---|---|
| `kenlm-baseline/` | external corpus statistic, not model-side |
| `scale-vs-kn/` | KenLM external baseline reference |
| `harmonic-filter-diagnostic/` | corpus diagnostics (chord-correlation stats over fixed counts) |
| `harmonic-bias-prototype/` | null result; the null likely holds under fixed loss, but listed as an open follow-up |
| `dist-rope-smoke/` | architectural probe; regression was decisive (>10% PPL) — survives bug correction |
| `heldout-tree-vs-model/` | "tree does the work" hypothesis test |
| `p2s-attention/` | closed architectural study |
| `prefix-suffix-bayes/`, `dual-model-fold/` | architectural design dirs |
| `rope-position-substitution/` | RoPE probe |
| `trie-attention-framing/` | notes |
| `unary-pruning/` | radix structure analysis |
| `docker/` | infra |
| `runpod/` | infra |

## Suggested next "redo" priority order

For the "directly relevant — redo" entries (depth-weight,
composite-weights, per-fire-norm, mass-conservation): these tested
hypotheses that the new defaults now embody or that the new defaults
make trivial. Probably the right move is:

1. Add a "RETIRED 2026-05-26: superseded by paper-correct defaults"
   line to each of those READMEs.
2. Don't re-run; the result of those experiments under the new
   regime would be "yes, the default-on form helps" — which is
   already established by the depth-1 PPL collapse.

The first headline re-run after the cleanup is probably the
`dmodel-scaling-100se/` re-evaluation under fixed loss. Once a single
new clean baseline at d_model=64 L=2 (Shakespeare) and d_model=128
L=8 (Gutenberg) is established under the fixed loss, everything else
can be compared against it.

## Fix commits for reference

```
0252072 agpt_train (v1): mass-weight linear + fire-norm-mass default-on
816f7d0 Fix CUDAX Section 2 event weighting   (Codex)
86ce536 paper.md: retract §11 Empirical Evaluation pending re-measurement
```
