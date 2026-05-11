# AGPT Prefix-Suffix Folding Architecture

**Date:** 2026-05-02 (updated 2026-05-03 with §1.5 unified-position framing)
**Status:** design synthesis — math validated, implementation not yet started.

## TL;DR

A unified architecture for AGPT-style training that combines three
mutually-reinforcing structural mechanisms:

1. **Prefix-suffix Bayesian self-consistency** — train a forward model
   F (on the forward radix-trie) and a backward model B (on the suffix
   radix-trie) jointly; constrain them to predict the same distribution
   for held-out positions via a KL term.

2. **Cap folding** — radix-trie endpoints (caps) at d=32 are H=0
   identity-memorization slots with no branching information of their
   own. Fold each cap into a structurally-matching internal node, so
   the cap's training positions become additional data for that
   internal node's already-learned distribution.

3. **AGPT-aggregated training** — all three loss terms (data CE, fold
   KL, suffix KL) accumulate per-query and fire one Adam step per
   AGPT partition group. The optimizer-fire structure is unchanged
   from current AGPT.

Mathematical foundation validated 2026-05-02 via
`rnd/prefix-suffix-bayes/` — the inversion math is consistent and the
forward/suffix tries encode equivalent empirical distributions. The
training-time signal lives in the *model-vs-empirical* gap, which
requires actual trained models to measure.

---

## 1. Motivation: why the current architecture leaves money on the table

### 1.1 Caps are ignorance

In a Shakespeare 1M d=32 radix-trie:
- 1,114,330 cap nodes (depth 32)
- 99.99% are singletons (one observed continuation)
- Average cap edge length 23.29 chars

Each cap is a unique 23-character corpus tail with a single observed
next token. From the model's perspective:
- H_cap ≈ 0 (no branching, no information about what follows)
- The cap's "training signal" is just: memorize this 23-char string and
  its single continuation
- Capacity allocated to caps is essentially wasted on position-
  specific memorization, not learning generalizable patterns

This validates the K=decision/V=identity framing
(`notes/agpt/trie-attention-framing.md`): past identity-depth (~21),
nodes are position-pointers with no shared-pattern structure to learn.

### 1.2 Forward and suffix tries are duals

A forward radix-trie and a suffix radix-trie built from the same
corpus encode *the same empirical distributions* read from opposite
directions. For any prefix p and candidate next-token t, both trees
produce identical empirical estimates of P(t|p).

Verified empirically: KL(P_forward || P_suffix) = 0.00000000 nats for
every prefix tested.

This means the suffix tree carries the *same data* as the forward
tree — but indexed by suffix structure rather than prefix structure.
Trained models extrapolate differently from the same data depending
on their indexing structure, so a forward model F and a backward
model B can disagree on held-out positions even though their
training data is equivalent.

That disagreement is *signal*: positions where F and B disagree are
positions where one of them is fitting a spurious correlation visible
only from one side.

### 1.5 Unified-position framing (added 2026-05-03)

The two-tree framing can be collapsed into a single structural view of
the corpus. **Each position p has two distributions:**

- **Outgoing** (from forward tree at the prefix path to p):
  `out_p[t] = P(c_{p+1} = t | corpus position p)`
- **Incoming** (from suffix tree at the reverse path through p):
  `in_p[t] = P(c_{p-1} = t | corpus position p)`

Both are V-dim vectors (V=65 on Shakespeare). They sum to 1.

The full bidirectional context is captured by the **65×65 joint matrix**
`M_p[a][b] = P(c_{p-1}=a AND c_{p+1}=b | corpus position p)`. Its
marginals recover the two vectors. The matrix carries strictly more
information than the pair of vectors *unless prev and next are
conditionally independent given p*, which they are NOT in general
(knowing what came before tells you something about what comes next,
beyond what the position itself tells you).

#### Memory cost at our scale (1.7M nodes, V=65, fp32)

| representation per node | total |
|---|---:|
| token only | ~7 MB |
| token + 2 vectors (520 B/node) | ~880 MB |
| token + 65×65 matrix (16.9 KB/node) | ~28.7 GB (too much) |

Two vectors per node fits in our 8 GB GPU; full matrix per node does
not. **Most cap-level positions have nearly-rank-1 matrices** (in and
out are roughly independent given a 23-char tail), so the matrix would
be wasteful at caps anyway.

A practical compromise: **two vectors per cap, full matrix per
branching internal node**. ~880 MB + a small extra for branching nodes
(there are <100k internal-with-branching at d=32). Fits.

#### What this unifies

- The "two trees" become **two views of one structure**: a sequence of
  typed positions, each carrying (token, in_vec, out_vec, optionally M).
- The forward radix-trie is just an efficient encoding of the
  out_vec column; the suffix radix-trie encodes the in_vec column.
- Training predicts both vectors jointly per position; the loss is two
  KL terms, one per direction.
- **Folding becomes vector-similarity matching, not string matching.**
  A cap's near-zero-information out_vec can fold to an internal node
  whose out_vec resembles the cap's prefix-context-derived expected
  out_vec. The structural fold-target search becomes a nearest-neighbor
  query in 65-dim distribution space — clean and well-defined.
- **Joint inference is natural**: at any prediction time, the model has
  both directional contexts simultaneously.

#### Implications for the loss

Replace the section 2 loss with the equivalent two-direction form:

```
L(p) = α · KL(out_emp_p   || out_pred_p(θ))    # outgoing prediction loss
     + β · KL(in_emp_p    || in_pred_p(θ))     # incoming prediction loss
     + γ · KL(out_fold(p) || out_pred_p(θ))    # cap-fold consistency (only at caps)
```

Where `out_emp_p` and `in_emp_p` are the empirical vectors from the
prefix and suffix tries respectively, and `out_pred_p(θ)`,
`in_pred_p(θ)` are the model's two predictions at this position.

The earlier formulation `KL(P_backward || P_forward)` (forward-suffix
self-consistency) is implicit here: if the model's two prediction
heads share a backbone, agreement between in_pred and out_pred is
enforced structurally rather than via an extra loss term.

#### What the matrix M would add

If we did keep M per node (or per branching internal node), the model
could predict M directly. The training target becomes:

```
L_M(p) = KL(M_emp_p || M_pred_p(θ))
```

This subsumes both KL(in) and KL(out) (since the matrix marginals are
the vectors), AND captures coupling between in and out. For high-
branching shallow nodes — exactly where the corpus structure has the
most predictively-useful information — M_pred could give meaningfully
better generalization than the marginals alone.

**Practical recipe:** start with vector-only (KL(in) + KL(out)) for v1.
Add per-branching-node matrix prediction in v2 if v1 plateaus.

#### Why this is the right abstraction

The user's insight (2026-05-03):

> "the prefix tree and the suffix tree are easily represented as two
> vectors — one is the distribution coming in and the other is the
> distribution over the vocab of going out. So does that mean there is
> a token t and two vectors that make up a node? Or is it just the two
> vectors... do they connect — as a probability from every A to every
> B? is that a matrix?"

Yes to all three. Token + two vectors is the minimal complete
representation; matrix M is the maximal one; vectors are M's marginals.
The fold mechanism, the bidirectional consistency, and the seq_len
decoupling all become more natural in this framing because positions
are now first-class objects with explicit distributional content,
rather than implicit endpoints of trie paths.

---

## 2. The unified loss

For each training position p with target c_p:

```
L(p) = CE(c_p, P_model(p))                          # data term
     + α · KL(P_fold(p)    || P_model(p))           # cap fold consistency
     + β · KL(P_backward(p) || P_forward(p))        # forward-backward consistency
```

Where:
- `P_model(p)` — forward model F's predicted distribution given the
  prefix at p
- `P_fold(p)` — for cap positions: the distribution at the cap's fold
  target (an internal radix node); for non-cap positions, this term
  is zero (no fold)
- `P_backward(p)` — backward model B's predicted distribution given
  the suffix at p (predicting c_p from the right-context)
- `α, β` — small KL weights, tuned (probably 0.01-0.1)

### 2.1 The fold term `KL(P_fold || P_model)`

For each cap C in the forward radix-trie:
- Identify a fold target — an internal radix node I whose edge text
  matches a substring of C's path AND whose distributions yield a
  high-probability trajectory through the rest of C's tail
- Store the fold map as a precomputed (cap_id → target_id) hash table

At training time, when a query is at a capped position:
- Look up the cap's fold target I (if exists)
- P_fold = the distribution at I in the trie (or ideally, F's prediction
  at I — so the fold target's *learned* distribution drives the cap)
- KL term drives F's prediction at C toward F's prediction at I

Effect: caps stop being dead-end memorization slots. Their training
positions become additional data that trains the relevant internal
node's parameters via the KL constraint. **The internal node
accumulates training signal from all caps that fold to it.**

### 2.2 The suffix term `KL(P_backward || P_forward)`

For every training position p:
- F runs on the prefix → produces P_forward
- B runs on the suffix (reversed) → produces P_backward
- Both predict c_p; KL between them drives them to agree

Effect: F and B are mutually-distilling. Each model becomes a
regularizer for the other. Predictions visible from only one side
(spurious correlations) get penalized; predictions consistent from
both sides (real linguistic regularities) survive.

This is **co-training/cycle-consistency**, not GAN adversarial. They
share an objective (predict c_p correctly); the KL forces them to
achieve it via the same predicted distribution.

### 2.3 Why CE + 2× KL gets us a stronger model

Pure CE training: the model finds parameters that minimize prediction
error on observed data. May overfit to spurious correlations (especially
at deep trie nodes with sparse observations).

CE + KL_fold: caps inherit learned-internal-node distributions, so the
model's parameters generalize via the internal pattern rather than
memorizing tails. Capacity reallocated from memorization to pattern
learning.

CE + KL_suffix: the model is constrained to be consistent with the
backward model's view. Spurious correlations only visible from one
direction get penalized. Real bidirectional regularities survive.

All three together: data-anchored, capacity-efficient, bidirectionally
consistent.

---

## 3. Architecture

### 3.1 Two AGPT-trained models, coupled

- Forward model F: standard AGPT trainer pointed at the forward radix-
  trie. Architecturally identical to current AGPT models.
- Backward model B: standard AGPT trainer pointed at the suffix radix-
  trie. Same architecture.

The two models share *no parameters*. They couple only through the
KL_suffix loss term that compares their predictions on each training
query.

### 3.2 The fold map (precomputed)

Built once at trie load:

For each cap node C in the forward trie:
1. Extract C's full path text (root-to-cap concatenation of all edge
   tokens, including the compressed cap edge — typically 25-30 chars
   total at d=32)
2. Find candidate internal nodes I whose edge text matches a substring
   of C's path (or some prefix of the cap's edge text)
3. For each candidate I: simulate walking from I along the rest of
   C's text, compute the path's joint probability under I's
   distributions
4. Pick the candidate with highest path probability (above some
   threshold, e.g., 0.05) as C's fold target
5. If no candidate exceeds threshold, C is a "dead end" — keep it
   as standalone, zero out KL_fold for C's positions

Storage: 4 bytes per cap → ~4.4 MB for 1.1M caps. Trivial.

Optional refinement (post-v1): cluster similar internal nodes
together and store cluster embeddings as fold targets, with caps
folding to clusters rather than individual nodes. Improves robustness
and pools training signal across similar internal patterns.

### 3.3 Training loop

Pseudocode (per AGPT super-epoch):

```
for each AGPT partition group g:
    F.zero_grads()
    B.zero_grads()
    for each query q in g:
        # Forward model
        Q_F = F.forward(q.prefix)
        ce_F = CE(c_p, Q_F)

        # Backward model
        Q_B = B.forward(reverse(q.suffix))
        ce_B = CE(c_p, Q_B)

        # Fold term (only if q is at a cap with a fold target)
        kl_fold = 0
        if q.position.is_cap and q.fold_target_id is not None:
            P_fold = F.distribution_at(q.fold_target_id)
            kl_fold = KL(stop_grad(P_fold), Q_F)

        # Suffix consistency term
        kl_suffix_F = KL(stop_grad(Q_B), Q_F)
        kl_suffix_B = KL(stop_grad(Q_F), Q_B)

        # Aggregate
        F.grad += ce_F + α·kl_fold + β·kl_suffix_F
        B.grad += ce_B + β·kl_suffix_B

    F.adam_step()
    B.adam_step()
```

Note: KL gradients use `stop_grad` on the "target" side so each
model is trained against the other's *current* prediction, treated
as a teacher signal. Symmetric KL avoids one-sided distillation.

### 3.4 Inference

Three deployment options, all valid:

**Forward-only**: deploy F; use it standalone. Trained with KL
regularization → better generalization than vanilla AGPT.

**Backward-only**: deploy B; same thing from the suffix side.

**Ensemble** (best PPL): combine F and B via Bayesian inversion at
inference:
- Forward gives P_F(t|p) directly
- Backward gives P_B(t|p) via the V·d Bayesian inversion (one-time
  cost per evaluation position)
- Geometric mean: P_ens(t|p) ∝ √(P_F(t|p) · P_B(t|p))
- Or weighted: log P_ens(t|p) = γ·log P_F(t|p) + (1-γ)·log P_B(t|p) + const
- The ensemble PPL should be lower than either individual model

---

## 4. Properties and consequences

### 4.1 The `seq_len` decoupling

For folded caps: F effectively learns the cap's distribution via its
fold target I (depth k where k is much smaller than 32). At inference,
F doesn't need to attend deeply for capped positions — the
distribution information is already learned at depth k.

This means **a model trained with folding at trie depth d=32 can be
deployed with smaller effective context windows for capped
positions**. Not a hard architectural decoupling, but a learned
generalization that gives shorter attention windows enough information
to predict well.

This is the same effect as longer-context models: the network learns
to compress long-range dependency into short-range pattern matches.
Folding makes that compression *structural* rather than emergent.

### 4.2 Capacity reallocation

Without folding: at d=32, ~99% of trie nodes are caps (memorization).
The model spends most of its parameters on position-specific tails.

With folding: capped positions train internal nodes' parameters via
the KL_fold constraint. Internal nodes accumulate signal from many
folded caps. Parameter budget shifts from memorization toward shared
pattern learning.

### 4.3 Compute cost

Per training step:
- 2× forward passes (F and B)
- 1× fold-map lookup per query (cheap)
- 2× KL computations (cheap, just 65-dim distribution divergences)
- 1× backward pass per model (existing AGPT cost)

Total: ~2× current AGPT wall-clock. Tractable.

Per inference position (ensemble mode only):
- Standard F forward pass
- V × d backward queries to construct P_B(t|p) via Bayesian inversion
- For Shakespeare V=65, d=32: ~2080 backward queries per eval position
- Sub-second per position; not viable for streaming generation but
  fine for held-out PPL eval

For non-ensemble inference: standard cost (one model, one forward pass).

---

## 5. Implementation roadmap

| phase | what | effort |
|---|---|---|
| 0 | Math validation (✓ done — `rnd/prefix-suffix-bayes/`) | done |
| 1 | Train backward model B on suffix trie (existing AGPT) | trivial |
| 2 | Dual-model probe tool — measure KL_suffix magnitudes on held-out positions | ½ day |
| 3 | Fold-map builder (CPU Crystal tool) | ½ day |
| 4 | Dual-model AGPT trainer (CUDA `agpt_dual_train.cu`) | 1-2 days |
| 5 | Ensemble inference tool | ½ day |
| 6 | Empirical validation: PPL on Shakespeare 1M vs current best (3.82 PPL) | 1-2 days |
| total | | **~5-7 days focused** |

Phase 6 is the moment of truth: does the unified architecture
actually beat pure-AGPT-pd=6 PPL on Shakespeare 1M?

Optimistic prediction: meaningful PPL improvement at the cost of ~2×
training wall-clock. The constraints are individually well-motivated;
the question is whether they compound or interfere.

---

## 6. Connections to other notes

- `notes/agpt/invariants.md` — the bounded-subtree training object;
  this architecture preserves it (each AGPT subtree fires a single
  Adam step including all three KL terms)
- `notes/agpt/trie-attention-framing.md` — K=decision/V=identity
  framing that motivates the cap-fold idea
- `notes/meaning-as-suffix-distribution.md` — semantic foundation
  for why the suffix tree carries equivalent information to the
  forward tree
- `rnd/prefix-suffix-bayes/` — empirical math validation
- `rnd/granularity-redundancy/README.md` — why pd=6 is the base
  recipe; the new architecture extends rather than replaces partition-
  depth
- `rnd/partition-depth/README.md` — the partition-depth mechanism
  that this architecture sits on top of

---

## 7. Open questions

- **Dead-end caps**: what fraction of caps will fail to find a fold
  target? Estimate 10-30% on Shakespeare 1M. Need to measure during
  fold-map builder development.
- **α and β weighting**: best values for the two KL terms? Likely
  small (0.01-0.1) so CE dominates. Tune empirically.
- **Symmetric vs one-sided KL_suffix**: does symmetric distillation
  beat one-sided (e.g., F always teaches B)? Probably worth testing
  both.
- **Fold-target distribution source**: should P_fold come from raw
  trie counts (fixed empirical) or from F's prediction at the fold
  target (learned, evolves with training)? The latter creates a
  dynamic teacher; the former is a fixed regularizer. The latter
  is probably better for generalization but more compute.
- **Ensemble inference cost**: V × d Bayesian-inversion per position
  is too expensive for streaming. Can we approximate it?
  - Sample t (not all V): stochastic ensemble
  - Cache common-prefix backward computations: amortize cost across
    queries
  - Distill the ensemble into a single model (post-hoc): get
    ensemble-quality predictions at single-model cost
