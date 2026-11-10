# Closed lines and negative results: don't retry as-is

*Updated 2026-11-10*

Each entry gets one line. The record is the experiment's README (its front
matter `answer`). "As-is" means the version that was tested. Where a source
names a way back in, it is given.

## From the gradient-population strand (September 2026)

- **A static coherence map choosing pd per root** (built from unweighted ρ
  over children): worse than uniform pd (Exp 2). It mostly flagged small
  subtrees.
- **A pool of SGD members as an optimizer:** it reduces to iterate averaging.
  At an equal budget, one chain with tail averaging beat every pool mean
  (Exp 4b).
- **Modified-loss correction of the aggregated step:** it follows SGD's path
  but does worse on held-out (Exps 5, 5b).
- **Full-trie L-BFGS as a drop-in for pd=1 Adam:** loses by 5–6% at equal
  traversals (Exp 8). L-BFGS for partitioned cadence is still open.
- **One L-BFGS history shared across partitions** (May 2026,
  `notes/optimization/l-bfgs.md`): its curvature pairs mix different
  subtrees' losses. `7444685` links it to the full-pass version that works.

## Tree structure and cadence

- **Suffix-tree (context-tree) orientation for partitioned training:**
  called on 2026-11-10 by the user. Its units are target-sorted batches, and
  the prefix trie's are not, which is the strand's positive finding (a minor
  breakthrough). The proviso checks (seeds, attention f_θ, v2 unit order,
  depth 16, canonical eval) are in the README.
- **Stochastic and hybrid cadence:**
  - pd1 followed by stochastic refinement made everything worse
    (`rnd/hybrid-agpt`);
  - lightning mini-tree sampling was not a decisive win (`rnd/lightning-agpt`);
  - pd=0 and single-loss-depth controls lost (`rnd/stochastic-agpt`).
- **Per-event loss weighting** (by mass, branching, entropy or depth, and their
  composites): closed on 2026-05-23. Nothing generalized across corpora beyond
  anc-grad (`rnd/depth-weight`, `rnd/composite-weights`,
  `rnd/gutenberg-anc-sweep`).

## Add-ons outside the aggregation family

All three were null, as the SGD-equivalence test predicts
([principles.md](principles.md)).

- **Precondition encoder over preceding characters:**
  - null at d64/L2, slightly worse at d128/L6, and 7.2× slower;
  - archived on `precondition` (tag `exp/precondition`, `df75f96`).
- **Cap-recurrence (injecting state from the preceding cap):**
  - all three forms were flat. An oracle input proved the wiring works, so the
    real input simply carries no signal;
  - branch `agpt-cap-recurrence` (tag `exp/cap-recurrence`).
- **Masked-LM combiner as an auxiliary loss** (2026-06-06):
  - it tied at λ=0.1 and hurt at λ=1, although probes had shown the joint
    signal exists;
  - **no committed record or branch survives** (checked 2026-11-07); only
    per-machine session memory describes it.
- **Backoff slot selection, step 0:**
  - the design forces pd=0, about 65× fewer Adam steps, and it came out
    negative;
  - tag `exp/slot-selection-final`;
  - the one stated way back is the context tree, where backoff contexts are a
    node's own ancestors (`todo/context-tree-orientation.md`).

## Position and attention variants

- **Distribution and expected-position RoPE:** `rnd/dist-rope-smoke`.
- **Harmonic attention bias:** `rnd/harmonic-bias-prototype`.
- **Wrapped and sampled-unit phase variants:** these failed. The unwrapped
  phase variants in `rnd/sampled-node-phase-rope` are the positive result,
  but not as position encodings: the shift cancels in Q·K (control,
  2026-11-10). Phase RoPE as a positional idea is closed; what helped there
  was the per-epoch strided sampling of loss weights.
- **Successor prefix attention:** inconclusive. Its 5.23 is a leakage
  diagnostic, and a clean version needs causal masking
  (`rnd/successor-prefix-attention`).

## Recurrent, Fisher and count-prior (June 2026, the Ultra era)

- **Negative:**
  - trie-Fisher GRU training (`rnd/trie-fisher-gru`);
  - child-model reconciliation, where messages never beat the zero-child
    ablation (`rnd/model-reconciliation`);
  - a product-of-experts backoff prior (`rnd/poe-backoff-prior`).
- **Overlay "AGPT on samples":** the GRU overlay, trust-weighted and mixture
  variants all lost to a plain GRU LM by 1.3–1.7 PPL. Code is on
  `exp/linear-recurrence-final`; the eval was legacy seq=8.
- **The count/backoff-prior residual line now continues in imm**
  ([air-projects.md](air-projects.md)). Don't restart it here.

## The June 2026 closure

- **What it is:** the memo `notes/agpt-project-closure.md`, on
  `exp/linear-recurrence-final` and not on `main`. It declared the project
  closed on 2026-06-11.
- **What it claimed:** neither the compute savings nor the beating of the KN
  bar had materialized at d64/L2 on Shakespeare.
- **What happened next:** the project reopened in September. Read the memo as
  history: its AGPT numbers predate the race fix and the exact backward, and
  its baselines have weak provenance ([measurements.md](measurements.md)).

## Older lines (April–May 2026)

Root-loop, virtual tree, wormhole, blending, lightning L3 and others are in
`rnd/` front matter. Most of them predate the loss fix and use legacy eval.
Rerun before citing any of them.
