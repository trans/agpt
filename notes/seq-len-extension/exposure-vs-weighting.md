# Exposure vs weighting — rethinking the harmonic filter

**Date:** 2026-05-26 morning (post the variance-check null result).
**Status:** active thinking; resume here.

This is the place to pick up the conversation that followed tonight's
harmonic-bias variance-check null result (`rnd/harmonic-bias-prototype/`).
The variance check showed the bias direction is null at d=16
in-distribution. The conversation that followed reframed *why*, and
the rethink is more useful than the result itself.

## The reframe

The harmonic-bias work was a **weighting** operator: it modified
attention scores between K-slots the model already had. The problem we
actually need to solve is **exposure**: get far-away K-slots into the
attention computation at all. Without exposure, no weighting can buy
long-range — there is no long-range K to weight.

ALiBi and RoPE are also weighting operators. They presume exposure
exists (the K-side reaches far enough that "near vs far" is a real
distinction). At AGPT's d=16 within a single fire, there is no
far. The model can only directly see d positions of K per layer; the
only way to go further is composition across layers (effective reach
≈ L·d) — transitive attention.

So the two problems we have to solve together:
1. **Multi-position encoding** — each K node represents a substring
   that occurs at many far-flung corpus positions. What is the right
   mathematical object to encode this multi-position fact, in a way
   the model can use it?
2. **Transitive attention and/or selective sampling** — bring far
   substrings into the K set per fire (effectively "localizing" far
   attention).

AGPT's structural advantage: every trie K-node IS already aggregating
far-flung corpus occurrences. The local trie is the far-corpus
mapping. We have the ingredients; we need to figure out how to compose
them.

## What we already have

- Per trie-node K: the count f_K(p) over corpus positions p where K
  appears as the d-prefix ending at p.
- Derived from f_K, all lossy compressions:
  - C_K = Σ_p f_K(p)  (scalar mass)
  - ẑ_K[j] = Σ_p f_K(p) e^{i·p·ω_j}  (chord = DFT of K's mod-W histogram)
  - mod-W histogram itself (W floats)
  - Full position list (variable)

- The trie itself encodes ancestor-descendant relationships, which
  trivially mean shared corpus positions (Q's positions ⊂ K-ancestor's
  positions). This is the only structural relationship current AGPT
  attention exploits.

## RoPE's defining property

For K_p = R(p)k, Q_{p'} = R(p')q where R(p) is a block-diagonal
rotation by p·ω_j per dim-pair:

  Q_{p'} · K_p = q^T R(p)^T R(p') k = q^T R(p' − p) k

The score depends only on (p' − p). Absolute p, p' drop out.

**Any multi-position encoding we devise must preserve this property:**
the score between two multi-position objects has to depend only on the
relative structure between them, not on absolute positions. Otherwise
we're not building a position encoding, we're building a
position-conditional embedding (which doesn't extrapolate).

## Where the operators we've tried land algebraically

**asym-DFT bias** (last night's prototype):
```
score += β · (1/C_K) · Σ_j Re[ẑ_K[j] · e^{−i p_Q ω_j}]
```
Absolute p_Q appears. NOT translation-invariant. Algebraically wrong
shape. This was the operator we ran the variance check on. It "worked"
in the sense that the model used β, but the underlying weights
degraded ~14% in compensation, netting to zero. The lesson is partly
operator-shape (wrong invariance), partly that there's nothing in the
K-pool to discriminate among (K-pool is Q's ancestors at p_Q,
trivially all "present at p_Q").

**Chord-chord inner product** (the variant we drifted away from):
```
score += β · ⟨ẑ_Q, ẑ_K⟩
```
Under a joint shift Δ, both chords pick up phase e^{iΔω}; the
inner-product phase cancels. Translation-invariant. CORRECT
algebraic shape.

**Depth-shifted chord-chord** (for prefix-suffix specifically):
```
g(Δ) = ⟨ẑ_Q, ẑ_K · e^{iΔω}⟩ = Σ_j conj(ẑ_Q[j]) · ẑ_K[j] · e^{iΔω_j}
```
Also translation-invariant (phases shift in tandem). Targets the
specific offset Δ. For prefix-suffix where K should be d positions
after Q, evaluate at Δ = d. This is the operator we dismissed earlier
as "depth-shift didn't matter" but the diagnostic that dismissed it
was in the wrong test framing (off-path comparison).

## What last night's diagnostic actually measured

The 1.6 IQR-unit on-/off-path separation at mass=2-9 was the chord
detecting **ancestor-descendant relationship**. On-path = same trie
path = Q's positions ⊂ K's positions ⇒ chord overlap. The diagnostic
showed the chord can RECOGNIZE a relationship the trie structure
already gives us. It did NOT show that chord-correlation picks out
*non-ancestor* pairs that are predictively meaningful. That's an open
question.

## The two roles the chord could play in attention

1. **As augmenting K's key.** K_key ← W_k·x_K + W_chord·ẑ_K. Adds a
   position-fingerprint dimension to the existing inner product.
2. **As an additive bias parameterized by Q's position.** What we
   tried; wrong invariance.
3. **As a selector for which non-ancestor K's enter the K-pool.**
   Construct a "chord-neighbor" list per K via ⟨ẑ_K, ẑ_K'⟩, augment
   each fire's K-set with these neighbors. The chord doesn't appear
   in the score; it appears in the K-set construction.

Roles 1 and 2 are weighting; only role 3 is exposure. Roles 1 and 2
can't help unless role 3 has put something interesting in the K-set
first.

## The big skeptical question for role 3 (chord-correlation as selector)

Just because two substrings have similar position-mod-W histograms
does NOT make them good joint attention candidates. Positional
co-occurrence is not semantic relevance. Two substrings could both be
high-mass and roughly uniform mod W (so chord-correlation ~ small but
non-zero), with no real predictive relationship between them.

Conversely, semantically related substrings ("the king" and "Macbeth")
might appear at quite different corpus positions (different scenes)
and have low chord-correlation.

**The chord as a similarity is at best a weak signal. We have not
demonstrated it picks meaningful attention partners. Yet.**

## What needs to be measured next (paper + small Python, no training)

Before any model-side work, derive on paper / measure on real data:

### M1 — Sanity check ẑ_K on toy distributions

Hand-construct synthetic substring position distributions:
- uniform over [0, N)
- periodic with period 32
- delta at p=0
- two deltas at p=0 and p=d

Compute ẑ_K and the operators ⟨ẑ_Q, ẑ_K⟩, ⟨ẑ_Q, e^{iΔω}·ẑ_K⟩.

Confirm intuitive answers: uniform×uniform ≈ 0; periodic with matched
period peaks at the right frequency; shifted delta pairs peak at the
right Δ.

This is a small Python notebook. 30 minutes.

### M2 — The "DC dominance" question

For every trie node K on Shakespeare (or a sample), compute |ẑ_K[j]|
across j. How dominated is the chord by the DC bin (j=0, which is
just C_K)?

**Hypothesis:** for most substrings, the mod-W histogram is ~uniform,
so ẑ_K[j] for j>0 is small relative to ẑ_K[0]. If this is broadly
true, then the chord has *discriminating signal* only for the small
subset of substrings that ARE positionally structured. For most pairs,
⟨ẑ_Q, ẑ_K⟩ is dominated by C_Q·C_K, which is just "both common."

This is the actual prerequisite for any chord-correlation-as-selector
direction. If chord is DC-dominated for most K, role 3 is doomed.

### M3 — If M2 confirms, look for non-DC-dominated substrings

For the subset of K's where ẑ_K is meaningfully non-DC, what are
they? Line-end characters? Scene-marker substrings? Things tied to
verse meter? Eyeball a few. Are they linguistically meaningful?

### M4 — Depth-shifted operator on actual Shakespeare prefix-suffix pairs

For a sample of real (prefix, suffix) pairs from Shakespeare where
suffix is the d-step continuation of prefix in the corpus, compute
g(Δ) = ⟨ẑ_prefix, e^{iΔω}·ẑ_suffix⟩ as a function of Δ. Does it peak
at Δ = d? Is the peak meaningful (separated from background)?

This tests whether the depth-shifted chord operator picks out the
prefix→suffix relationship at corpus scale.

## What we are NOT doing yet

- Any model-side work (training, kernel, integration into agpt_train).
- Any decision about whether to "pressure" the model. The model can't
  be pressured about a structure it has no exposure to.
- Picking among roles 1/2/3 for the chord. Role 3 is the only one
  that buys exposure; but role 3 only makes sense if M2 and M3 say
  the chord HAS discriminating signal for meaningful substrings.

The order of operations is: confirm the math (M1), confirm the
data has the structure the math expects (M2, M3), confirm the
specific operator (depth-shift) does what we think on real prefix-
suffix pairs (M4). Only then is there any point putting it in a
model.

## Open structural questions to come back to

- The chord captures *positional co-occurrence*, not *predictive
  co-occurrence*. What's the analogous operator for the predictive
  question? Probably joint-pair counts, conditional probabilities
  from the trie — not derivable from per-node position histograms
  alone.
- Whether prefix-suffix loop (temporal transitivity) and trie-prior
  tables (spatial transitivity, of which chord is one example)
  compose, or whether one supersedes the other.
- Whether the d-offset prefix-suffix relationship is even the right
  framing for chord-based selection, or whether we should look for
  other structural offsets / cross-substring relationships first.

## Related

- `notes/seq-len-extension/harmonic-filter-asymmetric.md` — operator
  brief that motivated the wrong-shape operator.
- `notes/seq-len-extension/harmonic-filter-brief.md` — earlier
  chord-chord framing.
- `rnd/harmonic-filter-diagnostic/` — diagnostic that showed the
  on/off-path separation (now reread as detecting structural
  ancestor-descendant relationship, NOT discovering similarity).
- `rnd/harmonic-bias-prototype/` — variance check confirming the
  asym-DFT operator is null in-distribution.
