# Position-Encoding Alternatives for AGPT Multi-Position Nodes

**Context.** In AGPT a trie node represents a context that occurs at up to K
corpus positions (K can reach thousands for high-mass nodes). The open problem:
how to give a node a *positional summary* over its K occurrences without paying
O(K) storage or attention cost, in a way attention can actually exploit.

The governing principle for ranking these: **Q·K attention is bilinear.** An
encoding is good not when it merely *preserves* positional information, but when
it places the relevant relationships where attention can read them *bilinearly*.
Rotations are bilinear-friendly; arbitrary nonlinear decodings are not.

---

## The naive baseline: concatenated mini-embeddings

Represent the node's positions as `e_1 ⊕ e_2 ⊕ ... ⊕ e_K`, one mini-embedding
per occurrence, each in its own vector slice.

- Attention resolves variants because each occupies a distinct subspace.
- Cost is O(K) in both storage and per-query attention scan.
- Feasible for K ≈ 4–8; catastrophic for K in the hundreds or thousands.
- Wastes capacity: most of the K positions are near-redundant and collapse into
  a few positional regimes, but this scheme treats each as distinct.

This is the thing every alternative below is trying to beat.

---

## Option A — Fourier position summary

Encode position `c` in an orthogonal-frequency basis
`(sin(c/T_1), cos(c/T_1), sin(c/T_2), cos(c/T_2), ...)` for a set of periods
`T_j`. Store, per node, the summary (e.g. coefficients) of the position
distribution over `{c_i}` in this basis.

**Why it fits attention.** Relative offset Δ becomes a *rotation* in each
frequency band, and rotation is exactly what Q·K computes natively — this is
RoPE's entire justification. The relationship "k is Δ away from q" is directly
bilinearly readable.

- Fixed cost per node, independent of K.
- Cleanest bilinear structure of all the candidates.
- Basis functions are *global*: every position contributes to every frequency,
  so local structure is smeared across the whole spectrum.
- Less aligned with the fact that language is locally periodic but globally
  non-stationary (sentence rhythm varies across paragraphs, etc.).

**Verdict:** mathematically cleanest, most attention-friendly. The conservative
choice if you want one fixed-cost positional summary.

---

## Option B — Wavelet position summary

Same idea as Fourier, but in a *localized* multi-scale basis. Each wavelet
basis function has bounded support in both position and frequency. Store, per
node, the wavelet coefficients of the position distribution.

**Why it fits language.** Captures fine-scale ("which token in this clause"),
medium-scale ("where in the sentence"), and coarse-scale ("where in the
document") position simultaneously, each *localized* so individual positional
contexts don't smear into one another. This matches how text structure actually
behaves: locally periodic, globally variable.

- Fixed cost per node, independent of K.
- Multi-scale + localized — best alignment with real language structure.
- Higher-dimensional than equivalent-resolution Fourier (more basis functions
  for the same coverage).
- Wavelet inner products are only *partially* bilinear-friendly — they do not
  reduce to a single clean rotation the way Fourier does.

**Verdict:** best fit for "this node tends to appear in these positional
regimes." Top pick if position turns out to be the bottleneck and a single
global frequency basis proves too smeared.

---

## Option C — Chinese Remainder Theorem (CRT) residues

Encode position `c` as a tuple of residues against pairwise-coprime moduli:
`(c mod p_1, c mod p_2, ..., c mod p_r)`. With small primes (7, 11, 13, 17, 19)
the product 323,323 positions are uniquely encodable while storing only the
residues. Per node, store the *count distribution over each residue class*
(Σ p_j small counts, independent of K).

**Key correction to an earlier misstatement:** CRT does **not** lose positional
information. Positions are unique up to `lcm(p_j)`, which can be made arbitrarily
large. The limitation is representational, not informational.

**The real tradeoff.** Relative offset has clean residue arithmetic:
`(Δ mod p_j) = ((q - k) mod p_j)`, computable from absolute residues without the
absolute positions. So a relative-residue attention bias is cheap and exact:

```
bias(q, k) = Σ_j f_j( (q mod p_j) - (k mod p_j) )
```

where each `f_j` is a tiny lookup table. But: *periodic* relationships are easy
to express, while a specific *aperiodic* offset (e.g. "437 tokens back") shows
up as a residue tuple with no clean bilinear interpretation. CRT guarantees the
model *can* reconstruct Δ, but attention's bilinear Q·K structure does not do
that reconstruction natively — it must be learned through deeper nonlinear
processing.

- Fixed cost per node; genuine *compression* for high-K nodes (1000 positions →
  ~67 counts), though an *expansion* for low-K nodes.
- Multi-periodicity falls out naturally — useful if language has structure at
  small-prime scales (worth checking empirically before committing).
- Residue differences are **not** bilinear-friendly: this is the reason it ranks
  below Fourier/wavelet despite the elegant compression.

**Verdict:** clever and compact, but the basis is misaligned with attention's
native operation. Periodicity-aware, not offset-aware.

---

## Ranking for the AGPT multi-position problem

| Rank | Scheme | Cost/node | Bilinear-friendly | Language fit |
|------|--------|-----------|-------------------|--------------|
| 1 | Wavelet summary | fixed | partial | best (localized, multi-scale) |
| 2 | Fourier summary | fixed | yes (rotations) | good (but global/smeared) |
| 3 | CRT residues | fixed | no (residue diffs) | periodic-only |
| 4 | K mini-embeddings | O(K) | yes | n/a (infeasible at scale) |

**The unifying frame:** the multi-position problem reduces to *choosing a basis
for representing position distributions such that attention can read the relevant
features bilinearly.* Fourier and wavelets are the natural answers; CRT is a
clever but less attention-friendly alternative; raw concatenation is the naive
O(K) baseline everything else improves on.

---

## Suggested experiment order (whole thread)

This sequence is principled — each step assumes the previous step's answer:

1. **RoPE base-offset sweep** (cheapest). Vary the chunk's RoPE base offset
   across training on the fixed d=32 trie. Tests whether position-awareness is
   the bottleneck at all. Compare PPL to trie d=32 fixed-RoPE (13.17) and window
   seq=128 (7.00). A few hours of work, zero new memory.
2. **Trie-ALiBi substitution.** Replace RoPE with a trie-distance linear bias
   (parent-pointer walk, radix edge of length L contributes L·m). Cleaner
   substrate, preserves radix compression, no d-barrier on the encoding itself,
   sets up long-range. Test vs. RoPE at fixed d=32.
3. **Multi-position summary** (only if 1–2 show single-position-per-node is
   insufficient). Use a Wavelet or Fourier positional summary per node; CRT as
   the compact periodicity-only fallback.

**Sanity check before step 3 (CRT specifically):** compute `(position mod p_j)`
for each token on the corpus and test whether next-token distributions vary
meaningfully across residue classes. If they do, residue encoding picks up real
signal. If they don't, it encodes noise compactly.

**Cross-cutting design principle (from Shiv & Quirk 2019):** every attention
path you want the model to use should correspond to a composable linear operator
in position space, with a clean algebraic meaning (parent = U, child-of-token-x
= D_x, cluster-transition = C). If a long-range mechanism needs a query↔key
relationship that *isn't* expressible as such an operator, that's a smell — the
position encoding isn't carrying the right structure for what attention is being
asked to do.

