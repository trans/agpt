# Trust-Weighted AGPT — Literature Pointers and HPYP Connection

**Origin:** user idea 2026-05-16, after seeing the multi-seed results.
The "trust" framework is the conceptual unifier for our cap-folding,
predictive-certainty-weighting, and shared-key-RoPE threads.

## Background — the core conceptual move

A trie node is not just "P(next char | this prefix)". It is a
**distribution over scales of contextual certainty**:

- depth-1 ('T'): high-mass, low-confidence (many things follow 'T')
- depth-2 ('Th'): medium
- ...
- depth-d (long unique prefix): low-mass, high-context-confidence
  but single-sample (high estimator noise)

A reasonable training target / prediction at any position should be
a **trust-weighted blend** across these scales, not a hard pick of one.
That's the trust framework.

## Three regimes for using trust

(User's framing, 2026-05-16)

1. **Distribution interpolation** (probability-level)
   D_eff(p) = Σ_k T_k(p) · D_k(p)
   Generalized smoothing. Cheapest. Target-side change only.

2. **Hidden-state interpolation** (representation-level)
   h_eff(p) = Σ_k T_k(p) · h_k(p)
   Architectural. Resembles hierarchical attention / multi-scale
   residual. Most powerful long-term.

3. **Gradient trust weighting** (optimizer-side)
   g'_p = T(p) · g_p
   Confidence-weighted aggregation. Currently mass-weight does a
   crude version of this. Stabilizes training.

Pairing guess: (3) always-on for stability, (1) for loss target,
(2) for the architecture commitment.

## What AGPT currently does (the honest baseline)

Important correction (Claude was initially wrong on this; corrected
2026-05-16): AGPT does **not** pick a specific depth and ignore the
others. It trains on the distribution at **every** depth from 1 to d
in every chunk, with parameters shared across all depths. **Implicit**
cross-depth smoothing happens via parameter sharing — gradients from
depth-1 targets affect depth-32 predictions and vice versa.

What's missing vs. the trust framework: **explicit** combination of
scales, with principled (or learned) T_k weights.

## The literature anchor — HPYP and friends

The closest formal framework is **hierarchical Bayesian language
modeling with backoff priors**. The trie's backoff structure (each
node has a parent = shorter context) is exactly the hierarchical
context tree these models use.

### Reading list, in recommended order

#### 1. Goodman 2001 — "A Bit of Progress in Language Modeling"
- **Why first:** historical overview of smoothing techniques. Sets
  context for everything else. Builds intuitions about what kinds of
  smoothing work and when.
- **Where:** Computer Speech & Language journal, but the long-form
  version is the Microsoft tech report MSR-TR-2001-72. Easy to find.
- **Effort:** ~1-2 hours. Skimmable.

#### 2. Kneser & Ney 1995 — "Improved Backing-off for M-gram Language Modeling"
- **Why second:** the canonical pre-neural smoothing method. State of
  the art for ~15 years. Its "discount" is essentially a per-context
  trust score derived from leave-one-out estimates. The interpolated
  Kneser-Ney form is *exactly* D_eff = Σ T_k · D_k with specific T_k
  from count statistics.
- **Effort:** ~1 day. Short paper, but the absolute-discounting
  derivation rewards careful reading.
- **What you'll get:** concrete formulas for trust weights derived
  from count data, no Bayesian machinery needed. After this you'll
  know what classical "smart smoothing" looks like.

#### 3. Teh 2006 — "A Hierarchical Bayesian Language Model Based on Pitman-Yor Processes"
- **Why third:** the Bayesian generalization that subsumes KN. Each
  context's distribution is a draw from its parent context's
  distribution + a sampled discount. Derives trust weights from first
  principles. KN turns out to be a special case of HPYP with specific
  hyperparameter choices.
- **Effort:** ~2-3 days. Heavier — requires comfort with Bayesian
  nonparametrics (Dirichlet/Pitman-Yor processes). But it's the
  conceptual capstone.
- **What you'll get:** the theoretical foundation. After this you'll
  understand WHY KN's discounts work and HOW to learn better ones.

#### 4. Wood et al. 2009 — "A Stochastic Memoizer for Sequence Data"
- **Why optional:** extends HPYP to unbounded context. Their
  construction is a *random* trie that naturally encodes backoff
  hierarchy. Mathematically very close to AGPT's trie.
- **Effort:** ~1 day. Read after HPYP.
- **What you'll get:** an efficient computational scheme that may
  inform AGPT implementation.

#### 5. Teh et al. 2006 — "Hierarchical Dirichlet Processes"
- **Why optional:** the underlying machinery, more general than
  language modeling. HPYP is essentially HDP + Pitman-Yor extension.
- **Effort:** ~3-5 days. Heavy Bayesian nonparametrics paper.
- **What you'll get:** the most general framework. Useful if you want
  to extend trust-weighting beyond language modeling.

### Recommended starting point

**Start with Kneser-Ney (#2).** Reasons:

- Concrete and empirical. You can read it and immediately understand
  what trust-based smoothing looks like in practice.
- The intuitions are crisp without heavy Bayesian machinery.
- It works empirically — HPYP only marginally improves on KN in
  practice.
- Once you grok KN, HPYP becomes "KN but with rigorous justification
  and learnable discount parameters."

If you want the historical sweep first, read Goodman (#1) before KN.
That's ~3 hours total. Then KN gives you the working framework, and
HPYP gives the theoretical capstone.

For our project specifically: KN is enough to design experiments
(implement (1) distribution interpolation with KN-style discounts).
HPYP is the long-term grounding when we eventually want to learn the
trust weights from data.

## Connection to existing project work

- `project_cap_folding.md` — one specific instance of (1)
  distribution interpolation. Uses suffix-trie-derived composite
  targets instead of cap one-hots. Mixed results so far; trust
  framework gives a principled way to derive the weights.
- `project_predictive_certainty_weighting.md` — the U-shape weighting
  intuition. Trust framework subsumes this: U-shape is what falls out
  when you derive trust from sample-size AND predictive-certainty.
- `notes/shared_key_rope.md` — option B (per-node K-vectors) is
  essentially (2) hidden-state interpolation. The "shared key" is the
  contextual representation at one scale; combining across scales is
  the trust step.
- Streaming AGPT (in progress) — orthogonal mechanism. Trust would
  apply within a single training step, regardless of cadence.

## Concrete next experiments enabled by trust framework

1. **Implement KN-discounted cap targets.** Replace cap one-hots
   with KN-interpolated distributions. Compare to current AGPT and to
   cap-folding's composite targets.
2. **Add per-node "trust" feature** as input to the loss kernel.
   Start with f(mass) = simple sample-size confidence. Use it to
   scale gradients (option 3). Most invasive but most directly tests
   the trust-weight idea.
3. **Multi-scale attention layer.** Feed the model representations at
   every depth, let it learn trust weights via attention. The
   architectural commitment of (2). Bigger but highest-ceiling.

All three are testable. KN-discounted targets is the cheapest first.
