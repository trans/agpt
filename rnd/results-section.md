# Results: Trie Path Probability Convergence

## Experimental Setup

We split the complete works of Shakespeare (1,115,394 characters, 65-character vocabulary)
into two independent halves using a 20-block interleaved partition: the corpus was divided
into 20 contiguous blocks of approximately 55,769 characters each, with even-indexed blocks
assigned to corpus A and odd-indexed blocks to corpus B. This interleaving was chosen over a
clean temporal split to balance genre and period distribution across both halves, yielding
corpora A and B of 557,690 and 557,704 tokens respectively.

Character-level prefix tries were built for each half at maximum depth D ∈ {4, 8, 16, 32},
with Laplace smoothing (ε = 10⁻⁶) applied to transition probability estimates. Shared paths
— sequences present in both tries with prefix count ≥ 5 in each — were enumerated via
simultaneous depth-first search. For each shared path of length d, we computed the
log-probability product:

    log π_A = Σ_{i=1}^{d} log P̂_A(tᵢ | t₁…t_{i-1})

and likewise log π_B for corpus B, using log-sum-exp to compute the log-mean log π_mean,
and recording the absolute log-space divergence |log π_A − log π_B|.

---

## Path Survival and Depth Horizon

The number of shared paths qualifying under the count ≥ 5 threshold peaks at depth 5
(16,831 paths) and falls rapidly thereafter, reaching 155 at depth 16 and a single path at
depth 26 (Table 1). This survival curve reflects count-starvation: as prefix length grows,
the vast majority of corpus positions generate unique sequences that appear too rarely in
either half to yield reliable probability estimates.

**Table 1. Shared path counts and log-space divergence by depth (D=32 run, count ≥ 5
in both halves).**

| Depth | Shared paths | Median |log π_A − log π_B| | 90th pctile | Between-path spread (σ) |
|------:|-------------:|----------------------------------:|------------:|------------------------:|
| 1     | 63           | 0.026                             | 0.19        | 1.8                     |
| 2     | 1,082        | 0.105                             | 0.79        | 5.4                     |
| 3     | 5,551        | 0.244                             | 2.36        | 8.3                     |
| 4     | 12,840       | 0.454                             | 13.96       | 11.1                    |
| 6     | 15,012       | 0.987                             | 14.71       | 16.5                    |
| 8     | 7,177        | 1.786                             | 15.32       | 20.6                    |
| 12    | 676          | 5.194                             | 18.63       | 27.8                    |
| 16    | 155          | 6.898                             | 19.51       | 18.9                    |
| 20    | 41           | 7.680                             | 14.62       | 21.1                    |
| 26    | 1            | 6.846                             | —           | —                       |

The deepest surviving path — the unique sequence that appears at count ≥ 5 in both corpus
halves at a prefix length of 26 characters — is the stage-direction fragment:

    " lord.\n\nKING RICHARD III:\n"

occurring 5 times in corpus A and 7 times in corpus B. This is not a surprising passage: the
formal repetition of Shakespearean stage conventions (character address, blank line, speaker
label) produces the most statistically invariant patterns in the corpus, more so than any
word or phrase.

---

## Error Growth and Failure of the √d Hypothesis

The median log-space divergence grows substantially faster than the √d scaling predicted
by an independent-edges model. Measured at depths 1, 4, 8, and 16, the median divergence
increases approximately as d^1.5 to d^2 relative to depth 1:

| Depth | Median |Δ log π| | Ratio vs d=1 | √d predicts |
|------:|------------------:|-------------:|------------:|
| 1     | 0.026             | 1×           | 1×          |
| 4     | 0.454             | 17×          | 2×          |
| 8     | 1.786             | 69×          | 2.8×        |
| 16    | 6.898             | 265×         | 4×          |

This super-linear growth indicates that successive edges are not contributing independent
errors. Shared high-frequency patterns across the interleaved blocks (character names,
formulaic phrases, repeated scene structures) create positive correlation between edge
estimates, so divergences compound rather than partially cancel.

---

## Coverage Collapse

When probability mass is weighted by exp(log π_mean) and paths are classified by whether
their log-divergence falls below an agreement threshold, usable coverage collapses rapidly
with depth:

| Max depth | Mass with |Δ log π| < 0.10 | Mass with |Δ log π| < 0.05 |
|----------:|-------------------------------:|-------------------------------:|
| 4         | 27.9%                         | 13.2%                         |
| 8         | 2.4%                          | 2.3%                          |
| 16        | 0%                            | 0%                            |

By depth 8, fewer than 3% of the weighted probability mass is covered by paths where the
two halves agree to within 10% in log-space. The hypothesis that coverage remains above 90%
across all depths is not supported for a corpus of this size.

---

## Path Products as Stable Fingerprints: Ranking Agreement

The absolute divergence results above address a stricter question than is necessary for
fingerprinting. A path probability product serves as a reliable identifier if paths can be
consistently *ranked* by probability across two independent observations — not if their
exact values agree. We evaluated this directly via Spearman rank correlation and Kendall's
τ between log π_A and log π_B across all shared paths at each depth:

| Depth | n paths | Between-path σ | Within-path noise | S/N ratio | Spearman ρ | Kendall τ |
|------:|--------:|---------------:|------------------:|----------:|-----------:|----------:|
| 4     | 12,840  | 11.08          | 0.45              | 24.4×     | 0.895      | 0.771     |
| 8     | 7,177   | 20.62          | 1.79              | 11.5×     | 0.869      | 0.711     |
| 16    | 155     | 18.91          | 6.90              | 2.7×      | 0.815      | 0.624     |

At depth 4, the spread of log-probabilities *between* distinct paths (σ = 11.1) exceeds the
within-path measurement noise (median 0.45) by a factor of 24. The Spearman correlation of
0.895 confirms that if corpus A assigns a higher probability to path X than path Y, corpus B
agrees in 90% of cases. This ranking stability persists even at depth 16 (ρ = 0.815),
despite the absolute values diverging substantially.

The interpretation is that path probability products occupy well-separated positions on the
log-probability axis — spanning 8 or more orders of magnitude — while measurement noise from
finite corpus size shifts each path's estimate by a comparatively small amount. The products
are not precise, but they are *discriminating*: two paths that differ in log-probability by
more than ~2 units at depth 4, or ~7 units at depth 16, will be ranked correctly by an
independent half-corpus with high probability.

This constitutes the central empirical result: **the cumulative product of character
transition probabilities along a trie path functions as a stable ordinal fingerprint for
that path, with a signal-to-noise ratio that remains above 2:1 through depth 16 on a 1.1M
character corpus.** The fingerprint is ordinal rather than cardinal: absolute probability
values are not reliable past depth 6, but relative ordering is preserved well beyond that
horizon.

---

## Implications for Corpus Size

The N_eff sanity check from the theoretical model (N_eff = 2d / (rel_err² · π_min))
yields estimates in the range of 4M–500M tokens to match the observed agreement at depths
4–8 — one to three orders of magnitude larger than the actual 557k-token halves. This
discrepancy signals that the two halves are not behaving as independent draws from a
stationary process, as assumed by the model. Shakespearean text is highly non-stationary:
formulaic stage directions, character-specific idiom, and repeated scene structures create
strong cross-block correlation even under the interleaved split. A well-mixed corpus drawn
from diverse authors or periods would likely yield N_eff estimates closer to the actual
token count and tighter absolute agreement at the same depth.

For practical fingerprinting applications on corpora of this scale (~1M tokens), these
results suggest that **depths 4–6 provide the most reliable absolute probability estimates**,
while **depths up to 16 remain useful for ordinal discrimination** with a signal-to-noise
ratio above unity.
