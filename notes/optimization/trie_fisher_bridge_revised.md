# From Counts to Curvature — Revised Bridge

### How a Prefix Trie Materializes Empirical Fisher Information, with Estimation Built In

*A revision of the original companion note. The changes are not cosmetic. Three
errors and one omission in the original framing are corrected here:*

1. *Backoff was anchored to the **parent**; it must be anchored along **suffix
   links** (the recency-preserving cousins KN/Goodman actually use).*
2. *Smoothing was presented as a **condition-then-adjustment** mechanism; it is
   folded into the target distribution as a **posterior mean** from the start, so
   it never appears as a heuristic.*
3. *Fisher damping and smoothing strength were **two knobs**; they are **one
   prior**, viewed in two coordinate systems.*
4. *The trust weights down the backoff chain were left implicit; they are pinned
   here to a **precision (inverse-variance) form** derived from the sampling
   covariance, with the one genuinely open problem isolated and stated.*

---

## 1. The matrix that does two jobs

Everything turns on a single object,

$$
\Sigma(r) \;=\; \mathrm{diag}(r) - r\,r^\top,
$$

evaluated at different distributions $r$. Keeping straight *which* distribution,
and *which job*, is the whole discipline of this note:

| Role | Distribution | Meaning | Depends on sample size? |
|---|---|---|---|
| **Curvature** | $p$ (model) | exact Hessian of softmax cross-entropy w.r.t. logits | No — a population statement |
| **Estimation uncertainty** | $q$ (counts) | sampling covariance of the empirical estimate is $\Sigma(q)/N$ | Yes — scaled by $1/N$ |

The original note's identities are all the first role. All smoothing lives in the
second. They felt like different species because they *are* — but the same matrix
$\Sigma(\cdot)$ governs both, scaled by $N$. Once that is visible, smoothing stops
being a patch and becomes the precision weighting that the counts already imply.

---

## 2. Setup, with the estimator in the target

At node $v$ in the trie, with embedding $e_v \in \mathbb{R}^d$:

- **Raw empirical distribution:** $q(c \mid v) = M[v,c] / N_v$, where
  $N_v = \sum_c M[v,c]$.
- **Model distribution:** $p(c \mid v) = \mathrm{softmax}(W e_v)_c$, with
  $W \in \mathbb{R}^{|V|\times d}$ the shared output projection.

We do **not** use the raw $q$ as the target. The target is a trust-weighted
overlay of the node and its backoff chain — defined in §3. Call it $\hat q_v$.
The local loss is cross-entropy of that overlay under the model:

$$
\mathcal{L}_v = -\sum_c \hat q(c \mid v)\,\log p(c \mid v),
\qquad
\text{equivalently } \min \mathrm{KL}(\hat q_v \,\|\, p).
$$

---

## 3. The backoff chain runs along suffix links, not parent links

**This is the central correction.** In a prefix trie the *parent* of the context
$t_1 t_2 \cdots t_n$ is $t_1 \cdots t_{n-1}$ — it drops the **oldest** token and
keeps the most ancient one. But Kneser–Ney / Goodman backoff drops the **oldest
conditioning token while preserving recency**: it shortens the history from the
front, $t_1\cdots t_n \to t_2 \cdots t_n \to t_3 \cdots t_n$. Those shorter
histories are **not ancestors on the path to $v$** — they are *cousins* reached
from the root down a *different* first edge.

The data structure that makes "shorten the recent context" an $O(1)$ hop is the
**suffix link**: an auxiliary edge from each node to the node representing its
longest proper suffix. Let $s_k(v)$ denote the suffix-link ancestor at backoff
depth $k$, with $s_0(v) = v$, $s_1(v)$ the one-token-shorter cousin, and so on up
to the root.

> **Consequence for the architecture.** Correct backoff is *non-local in the tree*
> but *local along suffix links*. The original note's clean "everything is local
> to a node, $W$ carries the global part" holds — but the local edges are the
> **suffix links**, not the parent edges. The optimizer's prior structure follows
> suffix links.

The target is the trust-weighted overlay over the suffix chain (a **meta-node**):

$$
\boxed{\;
\hat q_v \;=\; \sum_{k\ge 0} w_k \, q_{s_k(v)},
\qquad
w_k = \frac{\tau_k}{\sum_j \tau_j}.
\;}
$$

A meta-node attends and updates as a unit: it has already absorbed its entire
backoff ladder into one distribution. The coarse, widely-shared structure (short
suffixes, high in the chain) is *inside* $\hat q_v$, not a separate set of
targets.

---

## 4. Pinning the weights: precision, not preference

The $q_{s_k}$ are estimates of the same underlying quantity — the next-token
distribution — observed through lenses of differing reliability. The
variance-optimal combination of estimates-of-the-same-thing is **inverse-variance
(precision) weighting**; this is not a modeling choice but the BLUE result. So
$w_k \propto \tau_k$, the precision of level $k$, and the entire question reduces
to: *what is $\tau_k$?* Only two inputs are available — depth and the frequency
distribution — and only these may enter.

**Numerator — count (derived, not free).** From the sampling-covariance identity,
$q$ from $N$ samples has covariance $\Sigma(q)/N$. Precision scales as $N$. As we
climb the suffix chain the suffix shortens, pools more contexts, and $N_{s_k}$
grows: the specific node is the *least* count-trusted, the generic ancestor the
most. This inversion — specificity is what we want, trust runs the other way — is
the whole tension of smoothing, and it falls straight out of $\tau \sim N$.

**Denominator — distributional shape via entropy.** Two nodes with equal $N$ are
not equally reliable. The same $N$ spread over many effective continuations buys
a noisier per-component estimate. Reading effective support off the entropy
($\approx 2^{H(q)}$, the perplexity):

$$
\boxed{\;
\tau_k \;=\; \frac{N_{s_k}}{1 + 2^{\,H(q_{s_k})}}.
\;}
$$

**Depth does *not* get its own term.** Depth's effect is *already* carried by $N$
(climbing = pooling = higher count). An explicit depth decay $\beta^k$ would
double-count. The *only* legitimate extra role for depth is a deliberate
**preference for specificity** beyond what counts justify — and that is a prior,
not an estimation fact, so it lives in $\alpha$ (§6), never in $\tau$.

### 4.1 Validation against the qualitative regimes

Reading the formula back against the cases a practitioner cares about (entropy
here is standard information entropy: high = spread, low = peaked):

| Node | $N$ | $H$ | $\tau$ | Reading |
|---|---|---|---|---|
| Shallow, peaked (micro-rule) | high | low | **very high** | trust the node; it speaks for itself |
| Shallow, spread (goes anywhere) | high | high | **mid** | neutral — confidently undetermined |
| Deep, singleton / low mass | tiny | — | **≈ 0** | defer entirely to backoff |
| Deep, high mass (common saying) | high | low | **doubly high** | trust the long idiom completely |
| Shallow, weakly "backed" by prefix | high | any | (high) | high *estimation* quality, low *relevance* → relevance belongs in the §6 prior, not here |

The salience of the two inputs **trades off**: when a node is peaked, entropy
dominates and depth barely matters (the node decides). When a node is spread,
entropy cancels and the decision **delegates up the suffix chain** — the meta-node
overlay automatically lets whichever higher level has the sharpest, best-counted
shape set its character. A diffuse node does not decide; it lets its ancestors
vote.

---

## 5. Gradient, curvature, and the estimated Fisher

With logits $z = W e_v$, only the target changes from the original — $\hat q$
replaces $q$ — and the smoothing now enters *through the target itself*, not as a
separate anchoring term:

$$
\frac{\partial \mathcal{L}_v}{\partial z} = p - \hat q,
\qquad
\boxed{\;\frac{\partial \mathcal{L}_v}{\partial e_v} = W^\top (p - \hat q).\;}
$$

The exact embedding Hessian is unchanged (it is a property of the model $p$):

$$
\boxed{\;H_{e_v} = W^\top \Sigma(p)\, W = W^\top\big(\mathrm{diag}(p) - p\,p^\top\big)W.\;}
$$

The quantity we can actually *form at the node* is the empirical Fisher built from
the overlay target:

$$
F_v = W^\top \Sigma(\hat q_v)\, W
    = W^\top\big(\mathrm{diag}(\hat q_v) - \hat q_v\,\hat q_v^\top\big)W.
$$

As $p \to \hat q$, the two coincide. **Honest caveat:** in the regime where
smoothing matters, $\hat q$ is *deliberately* held away from $p$, so $F_v$ and
$H_{e_v}$ do **not** coincide there — $F_v$ is a prior-regularized *estimate* of
the curvature. That is the point, not a defect, but the "they coincide" framing
of the original applies only at convergence.

---

## 6. Damping is the prior, not a second knob

The natural-gradient step needs an invertible preconditioner:

$$
e_v \;\leftarrow\; e_v - \eta\,(F_v + \lambda I)^{-1}\, W^\top (p - \hat q).
$$

The ridge $\lambda I$ is **not free**. The prior anchoring $\hat q$ toward its
backoff chain contributes exactly a prior-precision term; conjugated through $W$
this *is* the damping. The prior strength $\alpha$ (how far $\hat q$ sits from raw
counts) and the Fisher damping $\lambda$ are the **same knob** in two coordinate
systems. Two knobs in the original program collapse to one.

The diag-minus-rank-one structure of $\Sigma(\hat q)$ lets $(F_v + \lambda I)$ be
inverted in $O(d^2)$ rather than $O(d^3)$ via Woodbury — this is what makes
"computed for free from the trie row" literally true rather than aspirational.

**On $\alpha$.** This framing removes the *procedure* (no regime test, no mixing
schedule — one closed-form posterior weighted by counts) but it does **not** remove
the *choice* of prior strength. That is the same decision the n-gram smoothing
literature never settled, now stated honestly as a prior. The arbitrariness moved
from two procedural decisions to one interpretable knob; it did not vanish. Any
claim that the Dirichlet framing makes $\alpha$ disappear is overclaiming.

---

## 7. The precise theorem-shaped claim

> The prefix trie, read along suffix links, materializes per-node empirical
> Fisher information as the covariance $\Sigma(\hat q_v)$ of a precision-weighted
> backoff overlay, conjugated into embedding space by the shared projection $W$.
> Node-local updates using $(F_v + \lambda I)^{-1} W^\top(p - \hat q)$ are
> natural-gradient descent with a Bayesian prior whose strength and the optimizer's
> damping are the same quantity.

$W$ is the only piece carrying global information — how embedding directions
translate to logit-space curvature. Everything else ($p$, $\hat q$, entropy, the
covariance, the trust weights) is genuinely local to the suffix chain. The trie
holds the local geometry; $W$ provides the shared coordinate system.

---

## 8. Summary table

| Quantity | Closed form | What the trie holds |
|---|---|---|
| Backoff overlay target | $\hat q_v = \sum_k w_k\, q_{s_k(v)}$ | counts along **suffix links** |
| Trust weight | $w_k \propto \tau_k = N_{s_k}/(1+2^{H(q_{s_k})})$ | count + entropy per level |
| Gradient w.r.t. $e_v$ | $W^\top(p - \hat q)$ | $\hat q$ via the overlay |
| Logit Hessian | $\Sigma(p)$ | exactly the categorical covariance |
| Embedding Hessian | $W^\top \Sigma(p)\, W$ | local geometry + global $W$ |
| Empirical Fisher | $W^\top \Sigma(\hat q)\, W$ | materialized per meta-node |
| Damping $\lambda$ | prior precision via $W$ | = the smoothing prior $\alpha$ |

---

## 9. The one clean open problem: diffuse vs. merged

Everything above is solid for the regimes in §4.1. There is exactly **one** place
the per-node trust formula is *blind*, and it is worth stating sharply because it
is far better-posed than the original note's "path momentum" ever was.

**The blindness.** A frequent, highly diffuse node (high $N$, high $H$) is treated
as neutral. But two physically different situations produce the *identical*
marginal $q$ — same counts, same entropy:

- **Honestly diffuse:** the true continuation distribution is genuinely broad.
- **Merged:** the suffix is short enough to lump together two (or more) distinct
  sub-populations, each of which is individually *peaked* in a different
  direction.

These are **indistinguishable from the node's own row of $M$.** The
distinguishing information does not exist at the node — the marginal threw it
away. This is *why* the question resisted a single-node scalar and why one is
tempted to make trust learned.

**Why it is probably intrinsic anyway.** The information is not gone from the
*trie* — only from that one row. The tell is at the level **below**:

- merged ⇒ the node is diffuse but its **children are peaked** (in conflicting
  directions). The answer is *down*, not up — do not delegate to backoff.
- honestly diffuse ⇒ the node is diffuse and its **children are also diffuse** —
  the breadth is real all the way down. Neutral is correct; delegating up is fine.

So the discriminator is not a property of the node but a **comparison of the
node's entropy to its continuations' entropy** — roughly the conditional mutual
information between the next token and the next-next context, *which the trie
already holds* because it has the deeper counts. It is still only $N$ and the
shape of $q$ — just compared across two adjacent levels rather than read at one.

**Two named risks (for the secondary analyst).**

1. **Cross-level correlation.** A node and its suffix neighbours share samples, so
   the level estimates are not independent — which strictly violates the
   inverse-variance weighting assumption in §4 *and* may confound the two-level
   entropy comparison proposed here. Whether this matters quantitatively or washes
   out is genuinely unknown.
2. **Termination of the recursion.** Comparing a node to its children pushes the
   question down a level; the children may themselves be merged. It is not obvious
   the recursion bottoms out before the leaves rather than running all the way
   down.

**Status.** This is the *shape* of an intrinsic answer (cross-level, counts-only,
no learning) — not a solved one. It should be handed over as such: the trust
formula is $N/(1+2^H)$ per level and correct everywhere *except* it cannot see
merged sub-populations; the fix is almost certainly a two-level entropy comparison
the trie can already compute; the two risks above are where to push.

---

## 10. What is settled vs. what is open

**Settled (derived, defendable):**

- Hessian $=\Sigma(p)$, Fisher $=\Sigma(\hat q)$, both conjugated by $W$.
- Backoff runs along **suffix links**; the target is a precision-weighted overlay.
- $\tau \sim N$ is forced by the sampling covariance.
- Damping $=$ prior; one knob $\alpha$, not two.

**Open (flagged, not faked):**

- The entropy *denominator* form ($2^H$ as effective support) is one defensible
  reading among a few; it is motivated, not forced.
- Cross-level sample correlation is real and ignored in the BLUE weighting.
- Diffuse-vs-merged (§9) — the one clean open problem.
- Emission-direction trust (forward lookahead, multiple emissions) is a *separate*
  backoff axis from the suffix chain; its weights are plausibly **learned**, not
  derived, and must be flagged as such rather than presented as falling out of the
  categorical-covariance identity.
