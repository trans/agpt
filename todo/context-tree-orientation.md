# Recency-ordered context tree instead of the prefix trie

## Status

Mostly closed. Idea from Thomas, 2026-10-03.

Tested 2026-10-03 (`rnd/context-tree-orientation`), GRU at depth 8: negative.
Diagnostic held-out PPL is 4.75 prefix vs 5.09 suffix at pd=0 (2000 epochs), and
5.09 vs 6.29 at pd=1 (500 epochs). The residual readout helps at pd=0 (4.96)
but breaks at pd=1 (9.22). Suffix-tree pd=1 units each hold one target class,
so partitioned steps act like class-sorted minibatches. The prefix trie's
grouping by the least predictive character is what keeps its units mixed.
Open: an attention f_θ at pd=0 only.

## The observation

The prefix trie groups contexts by their **oldest** characters. Two
contexts share a path only if they agree from the start of the window,
so contexts that end the same way but began differently share nothing:

```
prefix trie (root = oldest char)        context tree (root = newest char)
  o─f─ ─t─h─e  → ?                        e─h─t─ ─f─o   → ?
  i─n─ ─t─h─e  → ?                          │       └─n─i → ?
  (no shared node)                          (shared " the" path, then split)
```

The character that predicts the next one best is the newest, and it is
the last thing the prefix trie branches on. Siblings in the prefix trie
differ in their newest character.

## What this is not

- **The prediction itself is not backward.** Every node predicts the next
  character from its whole path, and the newest character is the node's own
  edge. The high mass at shallow nodes is mass on short contexts, and a short
  context consists of the newest characters.
- **The tree is the May 2026 suffix trie; the prediction is not.**
  `build_radix_corpus --reverse` already builds this tree, and root-to-leaf in
  it runs into the past. The May backward model
  (`src/tools/agpt_dual_train.cr`,
  `notes/prefix-suffix/prefix-suffix-model-divergence.md`) predicted
  **leafward**, the next character further into the past. Here each node
  predicts **rootward**, the character that would become a new root, which is
  the future. `bin/bayes_probe` (`rnd/prefix-suffix-bayes`) showed that the
  rootward next-character counts can be read off the suffix trie exactly.

## What changes and what does not

The two trees have the same nodes: every substring up to the depth limit,
with the same occurrence and next-character counts. The loss
`Σ_s Σ_x n(s,x)·(−log p(x|s))` is identical, and the summed gradient is still
the full-batch gradient, so SGD-equivalence holds (see the
agpt-sgd-equivalence note). Only the edges differ.

1. **Parent = backoff.** In the context tree a node's parent is the same
   context with its oldest character dropped, and it predicts the same target.
   The root-to-leaf path is the backoff chain that KN, PPM, CTW and the
   Sequence Memoizer use. In the prefix trie the parent drops the *newest*
   character and predicts a different position. Two consequences:
   - The parent's prediction is a natural prior for the child, so backoff
     smoothing can be built in, e.g. `logits(s) = logits(parent(s)) + Δθ(s)`.
     Sparse deep nodes (over 99% of depth-16 contexts have one observed
     continuation) would fall back on a well-estimated parent instead of
     being fit alone.
   - `rnd/slot-selection-step0` tried to give the prefix trie its backoff
     contexts as extra attention slots and failed partly because, in the
     prefix trie, they live in other root-child subtrees (the design forced
     pd=0). In the context tree they are the node's own ancestors.
2. **Shared computation follows the newest characters.** The cost: f_θ must
   read the context **newest-first**, with each node extending its parent's
   state by one *older* character. A left-to-right model shares computation
   only along prefixes, and that is why AGPT uses the prefix trie today. Any
   f_θ fits the framework (see the agpt-is-a-framework note).
3. **Partition units change.** At pd=1 a unit becomes "all contexts ending in
   character c" instead of "all contexts starting with c". The root children
   have the same masses (each is that character's count), but the units are
   grouped by what predicts the target. Gradient coherence per unit (the
   Experiment 1 measurement in `rnd/gradient-population`) may differ a lot.

## Mass

Mass belongs to a string (its occurrence count), so both trees carry the same
masses, mass falls leafward in both, and the per-node loss weights are
identical. What the orientation decides is **which end of each context the
mass sits at**.

- **Prefix trie: mass and relevance point in opposite directions.** A depth-16
  context runs from its heaviest node (the oldest character) to its rarest
  (the newest). Trie sharing and the rootward gradient sums (the anc-grad
  scatter) are concentrated on the characters that matter least to the
  prediction, and the characters that matter most sit in nodes usually seen
  once.
- **Suffix tree: mass and relevance point in the same direction.** The path
  runs from the heaviest node (the newest character) to the rarest (the oldest).
  Two consequences:
  1. Mass along a path becomes a confidence schedule for a single prediction:
     how far into the past the counts can be trusted. Witten-Bell and KN
     compute this from mass. The count gate learns it: its heldout gate weight
     falls from about 0.66 at depth 1 to 0.17 at depth 8
     (`rnd/count-backoff-gate`).
  2. Thomas's May 2026 point that mass ≠ relevance (`rnd/per-fire-norm`) says
     shallow nodes carry high mass but are predictions never made at inference.
     That holds in the prefix trie. With a residual readout in the suffix
     tree, a shallow node's prediction is the first term of every deeper
     prediction of the same target. Its mass then trains part of every
     prediction that *is* made.
- **What the suffix tree gives up.** In the prefix trie a node's mass splits
  over its children by next character, so the split is the label and it is
  local. In the suffix tree the split is by *preceding* character, and the
  counts for "s followed by x" sit under root child x, in another subtree.
  Each tree keeps one thing local: the prefix trie keeps the label, the suffix
  tree keeps the backoff chain. A label is easy to precompute. Bringing the
  backoff chain into the prefix trie is not (slot-selection).

## In terms of the Jacobian reduction (paper §4–5)

```
G_p = g_p + Σ_x J_{p→px}ᵀ G_{px}          J·(Σ_s g_s) = Σ_s (J·g_s)
```

The identity is linearity, so it holds in both trees, and the cost is the same:
the same nodes, with one Jacobian application per edge. What changes is
**what the sum at each node adds up**.

- **Prefix trie.** The gradients summed into `G_p` come from losses on
  *different* targets: the character 1, 2, … up to d−|p| positions after p.
  Their only common ground is that they all read `h_p`. The labels relate as
  `n_{p,x} = N_{px}`: a parent's label is its children's mass, so the subtree
  of p is a set of other prediction problems. The heaviest sums sit at the
  node farthest from the targets they serve. For a recurrent f_θ they also
  cross the most Jacobians to get there.
- **Suffix tree.** Every gradient summed into `G_p` comes from a loss on *one*
  target, the character after p, seen through a longer context. The labels
  relate as `n_p = Σ_y n_{yp}`: a parent's label is the sum of its children's
  labels, so the subtree of p is p's own prediction problem split by older
  history. The heaviest sum sits next to the target.

**Prediction (testable).** Aggregation coherence `ρ = ‖Σg‖ / Σ‖g‖` at the
heavy nodes is much higher in the suffix tree. Big Issue #2 measured that the
prefix trie keeps only 13–35% of gradient mass through aggregation at ep100
(`rnd/gradient-population` Exp 1). The same measurement on the suffix tree is
the first thing to run in experiment 2.

**With the residual readout** `z_s = z_{parent(s)} + Δθ(h_s)`, the Jacobian on
the logit path is the identity, and the reduction becomes count arithmetic.
Ignoring window edges:

```
E_p = Σ_{s ∈ subtree(p)} (N_s π_s − n_s)
    = Σ_{levels j} ( Σ_{s at level j} N_s π_s − n_p )
```

Each level of p's subtree is checked against p's own counts. If older history
adds nothing (`π_s = π_p` throughout), `E_p = k·(N_p π_p − n_p)`, p's own
error counted once per context length. The heavy node trains hardest, and
deeper nodes learn only what older history adds. Backoff falls out of the
reduction instead of being added on.

Correction found while checking this: `notes/optimization/trust_hpyp_literature.md`
says the trie's parent is the HPYP backoff context. That holds for the context
tree, not for the prefix trie.

## Costs and risks

- **Inference.** Each new character becomes the new root, so the context is
  re-encoded at every step and there is no incremental state or KV cache. This
  costs nothing at depth 16 but matters for long contexts (Big Issue #1).
- **Recency inversion in RNNs.** A newest-first RNN reads the most predictive
  character first, and by depth 16 the most recently read character is the
  oldest. The backoff-residual readout avoids this, because the parent's
  logits already carry the newest-character information.
- **Trainer targets.** In the context tree a node's children are older
  characters, not next characters, so the targets need a table separate from
  the child structure: the forward trie's child counts, matched by string
  (what `bin/bayes_probe` does). The count-gate tool already builds both tables
  (`rnd/count-backoff-gate`, "suffix stats"). The v2 trainer's
  `experimental.target_sidecar` (soft targets keyed by substring id) may be
  reusable.
- **Eval.** Canonical eval needs an HF wrapper that re-encodes newest-first at
  each position.

## Next session: run it to ground (Thomas, 2026-10-03)

Thomas predicted months ago that a tree grouped by what predicts the target
would make partitioned training behave like sorted, non-shuffled batches. The
first runs suggest the prefix trie we built avoids exactly that by grouping on
the oldest character. One GRU, one seed, depth 8 and a diagnostic eval are
not enough to settle it. In order:

1. **Test the mechanism directly.**
   - **Mixed units.** On the suffix tree at pd=1, make each step a random
     mix of depth-2 suffix subtrees across roots, with the same events per
     step as a root unit. If class-sorting is the cause, mixing recovers most
     of the gap to the prefix trie. Run the prefix trie with mixed units as
     the control.
   - **Interference.** After each unit step, measure the loss change on the
     other units: how much one step undoes the others.
   - **No-trie control.** Plain mini-batch SGD with batches sorted by last
     character, by first character, and random. This shows whether the
     effect exists with no tree at all.
2. **Attention f_θ.** A small PyTorch attention model over the path in both
   orientations, at pd=0 and pd=1. The GRU's recency inversion confounds
   pd=0, while the pd=1 units come from the tree, whatever the f_θ.
3. **Robustness.** 3 seeds, warmup-cosine LR (these runs used a constant
   3e-3), depth 16, and pd=2/3. Suffix units at pd≥2 are even narrower: all
   contexts ending in the same 2–3 characters.
4. **History.** `--shuffle-order` helped ~2% at pd>1 (`rnd/cap-folding`) and
   not at pd=1 (`todo/agpt-trainer-structure-and-staleness-analysis.md`).
   Also re-read Thomas's earlier notes on the expected sorting problem.
5. **Canonical eval** for anything reported. A newest-first model needs its
   own HF wrapper, or the claim stays labelled diagnostic.

## Experiments, cheapest first

1. **Model-only check (no trie).** Train under plain mini-batch training at
   window 8: a GRU reading oldest-first, a GRU reading newest-first, and the
   newest-first GRU with the backoff-residual readout. All score the same
   held-out split. This shows whether reading direction and the residual
   readout give a better f_θ, separate from the tree.
2. **Trie A/B (recurrent).** GRU AGPT on the prefix trie (branch
   `worktree-linear-recurrence`, tag `exp/linear-recurrence-final`, legacy
   5.40 at depth 8 pd=1 500 ep) against newest-first GRU AGPT on the context
   tree, at the same depth, d_model, pd and compute. This shows whether
   context-tree aggregation trains better or cheaper. Also measure per-unit
   gradient coherence at pd=1 for both trees.
3. **Attention version in the v2 trainer**, only if 1 or 2 is positive.

**Success:** lower held-out byte PPL than the prefix-trie model with the same
f_θ at matched compute, measured through `bin/agpt_experiment`.

**Null:** if the newest-first model ties, tree orientation matters only for
which computation is shared, not for what is learned. That is still a useful
fact for the context-length work.
