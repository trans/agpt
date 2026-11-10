# Superposed RoPE keys and the softmax limit

*2026-11-10. This derivation came out of the multi-positional RoPE review with
the user. The measurement is in `rnd/superposition-moment-order/`.*

The idea under test: superpose a symbol's occurrences into one RoPE key (or
state) that stands for all of them. In the limit of unlimited resolution,
this is the user's observation that training could reduce to one
multi-position embedding per vocabulary symbol. This note records exactly
what such a key can and cannot do, so that later variants start from the
constraint.

## 1. What an exact superposed lookup requires

RoPE's score between a query at t and a key at p, in complex pairs:

```
s(t, p) = Re Σ_j  q_j · conj(k_j) · φ_j(t − p)
```

Superposing all occurrences of a symbol c needs two properties:

1. **Linearity in the key.** Then Σ_{p: x_p=c} s(t, p) = Re Σ_j q_j conj(k_{c,j}) Σ_p φ_j(t − p).
2. **Per-coefficient shift:** φ_j(a + b) = φ_j(a) · φ_j(b). Then the summed
   state is carried forward by one multiply per step,
   S_{t+1} = λ ⊙ S_t + (new occurrence), and the query reads it with one
   multiply per coefficient.

Property 1 restricts exact superposition to keys and values that depend only
on the symbol, which means layer 1 (no positional embedding; RoPE only).

## 2. Property 2 forces exponentials

On integer offsets, φ(Δ) = φ(1) · φ(Δ − 1) = … = φ(1)^Δ. So each coefficient
is λ^Δ for a single complex number λ, and nothing else satisfies property 2.

- **|λ| = 1 is RoPE:** a rotation, norm-preserving and valid in both
  directions.
- **|λ| < 1 is a damped rotation** (as in xPos and RetNet). It still depends
  only on t − p, but it only makes sense causally. On the key side the factor
  λ^(−p) blows up, so it must be computed in recurrent or chunked form.
- **The constant-Q family**, |λ_j| tied to arg λ_j, consists of dilations of
  one causal mother function, e^(−u/c) e^(2πiu) for u ≥ 0. It is wavelet-*like*
  in how it tiles time and frequency, and it keeps the lookup because it is
  still exponential.

With a vector state advanced by a fixed linear map M (s_{t+1} = M s_t + B x_t,
the most general way to carry a summary forward), each occurrence contributes
M^Δ B. Its entries are **exponential polynomials** Δ^k λ^Δ (Jordan form).
These are the complete set of atoms that keep an exact finite-state lookup
(finite-dimensional shift-invariant spaces are spanned by exponential
polynomials). The most wavelet-like exact atom is a Jordan block,
M = [[λ, 1], [0, λ]], whose atoms Δ^n r^Δ e^{iωΔ} have a gamma envelope peaking
near Δ ≈ nτ.

**True wavelets break the lookup.** Haar, Daubechies and Gaussian-envelope
Morlet atoms are localized, and a localized atom is not an eigenfunction of
shift. A shift moves a coefficient onto a different atom, so property 2
fails. The decimated discrete wavelet transform isn't even shift-covariant.
An exact sliding box sum needs the raw history inside the box.

## 3. The softmax limit

Summing commutes with a readout g only when g is linear:
Σ_p g(s_p) = g(Σ_p s_p) only for linear g. Softmax applies exp to each
occurrence's score before summing. The geometry makes the cost exact.

- **The cloud.** Each occurrence is a point φ(Δ) ∈ C^J ≅ R^{2J} on the RoPE
  torus, and every point has norm √J. A query's score is linear in the point,
  s = ⟨a, φ(Δ)⟩ with a = q · conj(k_c). The superposed state is the sum of the
  cloud, which is its mean times the count.
- **Linear attention** is ⟨a, S_c⟩. It reads only the mean, so it is exact.
- **Softmax** is log Σ_p exp⟨a, φ_p⟩, the cumulant generating function of the
  cloud: log n + ⟨a, mean⟩ + ½ aᵀ Cov a + … It needs every cumulant.
- **Hard max** (zero temperature) is max_p ⟨a, φ_p⟩, the support function of
  the cloud's convex hull. All points lie on a sphere, so every one of them is
  a vertex. The query a = φ(Δ_p) picks out occurrence p uniquely, because
  ⟨φ(x), φ(y)⟩ ≤ J with equality only when x = y. **Sharp attention admits no
  compression.**
- **What sharpness costs.** The k-th Taylor term of exp is bounded by
  x^k / k!, where x = |a| √J is the query's logit range. Those terms fade
  only past k ≈ e · x.
- **Each order is itself an exact superposed state.** The k-th moments of
  torus points are the cloud's Fourier transform at all signed combinations
  ±ω_{j1} ± … ± ω_{jk}. That is Jacobi–Anger. Those atoms are still
  exponentials, so each order follows the same recurrence. With J = 8, the
  orders ≤ m need D(8, m) complex coefficients per symbol (a Delannoy number):
  17, 145, 3,649 and 265,729 at m = 1, 2, 4 and 8.

So exact softmax over superposed occurrences is an infinite tower of
superposed states. That is the "no resolution limit" case. Finite resolution
truncates the tower, which softens the attention it can represent: as a rule
of thumb, order m supports logit ranges up to about m / e.

## 4. Why the trie is exact

AGPT's trie merges only occurrences with identical context. Those are the
**same point** on the torus, since their keys and their offsets to every
query in the window are identical. The cloud is one point with a count, the
mean is everything, and softmax over it is exact at any temperature.
Multi-position superposition merges occurrences at **different** offsets,
which are different points. A softmax over them then needs the shape of the
cloud, which the sum throws away.

For AGPT's on-path keys (a query's own ancestors) there is nothing to
superpose: each one sits at a single offset. The limit bites wherever a
design asks one key to stand for occurrences at several offsets.

## 5. Measurement

`rnd/superposition-moment-order/` measures this on trained layer-1 heads.
Take a regular d64/L2 model trained on 124-character windows, keep its
16-position window exact, and superpose offsets 16–123.

- **The heads are sharp:** median window max weight 0.84, median older-offset
  logit range 14.
- **Matching their attention distribution takes order 8** for p90 KL
  0.012 nats, and **order 24** for p99 KL below 1e-3.
- **That costs 2.7e5 to 8.7e8 coefficients per character,** against at most
  108 occurrences.
- **Expanding exp at 0 is worse:** at m = 12 its median KL is 2.3 nats,
  because scores around −14 need m ≳ 38.
- **AGPT's depth-16 heads are just as sharp within their window** (median max
  weight 0.86).

## 6. Consequences

- **Superposed keys can serve only a linear readout,** and trained heads
  aren't linear. Replacing attention over occurrences with a superposed key
  or state costs more than it saves at the sharpness real heads use. This is
  the mechanism behind `rnd/dist-rope-smoke` (collapsing positions broke
  RoPE) and behind `shared_key_rope.md`'s rule: do not collapse an occurrence
  set into one spectral fingerprint.
- **What survives is compression of compute, not of keys.** Share the
  content projection across occurrences and keep each occurrence's position
  (shared-key RoPE), or merge only occurrences that coincide (the trie).
- **A positional design that merges occurrences** has to say what readout it
  serves. If the answer is softmax, it must carry about e × (logit range)
  moment orders, or it must keep the occurrences.
