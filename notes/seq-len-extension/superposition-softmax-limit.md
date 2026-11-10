# Superposed RoPE keys and the softmax limit

*2026-11-10. This derivation came out of the multi-positional RoPE review with
the user. The measurement is in `rnd/superposition-moment-order/`. Revised
the same day: the first version said sharp attention "admits no compression"
and that superposed keys serve "only a linear readout". Both were too strong.
The sum loses no information, and the real limit is decoding it (§4). There
is also an exact family with a polynomial positional kernel (§5).*

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
  ⟨φ(x), φ(y)⟩ ≤ J with equality only when x = y. In one layer-1 head only
  65 query directions occur at a given step, but as t advances each
  occurrence passes the offset where the head's score peaks, so every
  occurrence decides some future query. **So a state that serves sharp
  attention must determine the whole cloud.** §4 shows the sum already
  does, and that the limit is decoding it.
- **What sharpness costs.** The k-th Taylor term of exp is bounded by
  x^k / k!, where x = |a| √J is the query's logit range. Those terms fade
  only past k ≈ e · x. The best polynomial does better, but not by much. On
  [−x, x] the Chebyshev coefficients of exp are 2 I_n(x) ≈ 2 (x/2)^n / n!,
  which fade past n ≈ e · x / 2 in uniform error, and later where relative
  error matters. On a trained head, Chebyshev interpolation needed about two
  thirds of the Taylor order (§7).
- **Each order is itself an exact superposed state.** The k-th moments of
  torus points are the cloud's Fourier transform at all signed combinations
  ±ω_{j1} ± … ± ω_{jk}. That is Jacobi–Anger. Those atoms are still
  exponentials, so each order follows the same recurrence. With J = 8, the
  orders ≤ m need D(8, m) complex coefficients per symbol (a Delannoy number):
  17, 145, 3,649 and 265,729 at m = 1, 2, 4 and 8.

So a polynomial readout of exact softmax over superposed occurrences is an
infinite tower of superposed states. That is the "no resolution limit" case.
Finite resolution truncates the tower, which softens the attention it can
represent: as a rule of thumb, order m supports logit ranges up to about m / e
with the Taylor series, and up to about 2m / e with the best uniform
polynomial.

## 4. Information is not the limit; decoding is

**At infinite precision the sum loses nothing.** Two different occurrence
sets P ≠ P′ of one symbol give states whose difference is Σ_p ε_p z^{Δ_p},
with ε_p ∈ {−1, 0, 1} not all zero and z = e^{iω}. That is a nonzero
polynomial with integer coefficients, evaluated at z. Every RoPE frequency
here is algebraic (ω_j = 10000^{−j/8} = 10^{−j/2}), so z is transcendental
(Lindemann–Weierstrass) and can't be a root. So one complex number per symbol
determines all of that symbol's occurrences. The user's unlimited-resolution
picture is exact as an encoding.

**At finite precision the question is decoding.** The state is a linear
sketch of the symbol's occurrence indicator x_c ∈ {0, 1}^L, with 2J real
measurements. A decoder may be any nonlinear function of the state. Even so,
stable recovery of a k-sparse vector from a linear sketch (tolerating noise)
needs Ω(k log(L/k)) measurements, whatever the decoder (Ba, Indyk, Price and
Woodruff, *Lower bounds for sparse recovery*, SODA 2010). Sharp softmax needs
the whole cloud (§3), so a state of J frequencies serves it for at most about
2J / log(L/k) occurrences per symbol. Beyond that, decoding rests on the
low-order digits the transcendence argument uses. Any noise in a trained
model destroys those digits, and gradients can't use them.

- **Decoding beats the moment tower but not the list.** A nonlinear decoder
  (sparse recovery, then an exact softmax over the recovered points) needs
  far fewer numbers than the D(J, m) of a polynomial readout. But it
  reconstructs the occurrence list, so up to the log factor its state is
  never smaller than that list of k positions.
- **Where the state compresses, it can't decode.** One head's state is
  V × 2J = 65 × 16 = 1,040 real numbers, whatever the history length. It
  holds fewer numbers than the text it summarizes only once the history
  passes ~1,000 characters. By then the average symbol has ~16 occurrences,
  and k ln(L/k) ≈ 16 × ln 64 ≈ 67 is well past 16 (with the bound's constant
  set to 1). Even at L = 108, the five most frequent characters (space, e, t,
  o, a: 41% of the text) have expected counts of 16.5, 9.2, 6.5, 6.4 and 5.4.
  For all five, k ln(L/k) ≥ 16.

## 5. The exact family: polynomial positional kernels

Exponential polynomials are closed under multiplication: λ^Δ μ^Δ = (λμ)^Δ.
So a finite polynomial of exponential atoms is again a finite sum of
exponential atoms, and it keeps the recurrence. Degree d costs up to about
J^d atoms, and the moment tower is the case d = m. exp has infinite degree and
leaves the class. So instead of approximating exp of a trigonometric
polynomial, which is what a RoPE head computes, make the positional factor of
the attention *weight* a polynomial.

In layer 1, keys and values depend only on the symbol. For the query at t and
an older occurrence of c at p = t − Δ:

```
logit(t, p)    = content(x_t, c) + log w(Δ),     w(Δ) = |Σ_j b_j(x_t, c) λ_j^Δ|²  ≥ 0

Z_c(t)         = Σ_{p<t: x_p=c} e^{logit(t, p)}
               = e^{content(x_t, c)} · Σ_{j,l} b_j conj(b_l) · S_c^{(j,l)}(t)

S_c^{(j,l)}(t) = Σ_{p<t: x_p=c} (λ_j conj λ_l)^{t−p},     S(t+1) = λ_j conj λ_l · (S(t) + [x_t = c])
```

- **It is exact.** Z_c is symbol c's share of the softmax normalizer over
  every older occurrence, with no approximation. The head's output adds
  Σ_c Z_c v_c, and its normalizer adds Σ_c Z_c. Checked numerically
  (2026-11-10, `src/tools/agpt_poly_kernel_check.py`): the recurrent state
  matched the direct sum over occurrences to a relative 3e-14 (8 atoms,
  400 steps, with kernels drawn at random for each step and key symbol).
- **It is the same size as a superposed RoPE state.** With equispaced
  frequencies λ_j = r e^{ijω₀}, the products depend only on n = j − l, and
  S^{(−n)} = conj S^{(n)}. That leaves J complex numbers per symbol, as in
  the superposed-tail probe.
- **It is complete for its bandwidth.** By the Fejér–Riesz theorem, every
  nonnegative trigonometric polynomial of degree J − 1 in ω₀Δ is
  |Σ_j b_j e^{ijω₀Δ}|². So w ranges over all nonnegative positional kernels
  of that bandwidth, times the envelope r^{2Δ}. Like RoPE's, the kernel may
  depend on the (query, key) symbol pair. J = 1 is ALiBi. Damped atoms and
  Jordan blocks give localized, wavelet-like bumps that stay exact.
- **The price is resolution.** Over a range of L offsets without aliasing
  (ω₀ ≤ 2π/L), the kernel's peaks are about L/J offsets wide: about 14 for
  J = 8 and L = 108. It can't be sharp in position the way a RoPE head is.
- **It works in layer 1 only.** In deeper layers, keys vary per occurrence,
  so the content factor e^{q·k_p} no longer comes out of the sum, and the
  tower returns in content space. That is where Performer's random features
  and Based's Taylor feature map work, and both pay a cost that grows with
  sharpness. Based pairs its second-order Taylor attention with an exact
  sliding window.

**Status: derived, not tested.** It fits the role the superposed-tail probe
tried to fill, a far-past channel beside an exact window, with an exact
softmax in place of the probe's linear readout. A test needs a training
budget at which the window-64 control reproduces the known gain from
context, which `rnd/superposed-tail-rope` did not have.

## 6. Why the trie is exact

AGPT's trie merges only occurrences with identical context. Those are the
**same point** on the torus, since their keys and their offsets to every
query in the window are identical. The cloud is one point with a count, the
mean is everything, and softmax over it is exact at any temperature.
Multi-position superposition merges occurrences at **different** offsets,
which are different points. A softmax over them then needs the shape of the
cloud, which a readout of the sum can recover only at the costs above.

For AGPT's on-path keys (a query's own ancestors) there is nothing to
superpose: each one sits at a single offset. The limit bites wherever a
design asks one key to stand for occurrences at several offsets.

## 7. Measurement

`rnd/superposition-moment-order/` measures this on trained layer-1 heads.
Take a regular d64/L2 model trained on 124-character windows, keep its
16-position window exact, and superpose offsets 16–123.

- **The heads are sharp:** median window max weight 0.84, median older-offset
  logit range 14.
- **Matching their attention distribution takes order 8** for p90 KL 0.012
  nats (centred Taylor estimator). **It takes order 16** for p99 KL below
  1e-3 with Chebyshev interpolation, and order 24 with centred Taylor.
- **That costs 2.7e5 to 3.9e7 coefficients per character,** against at most
  108 occurrences.
- **Chebyshev helps less than one might hope.** Below order 10 it is worse
  than centred Taylor, because its interval is the score's whole range over
  the offsets, fixed in advance and wide.
- **Expanding exp at 0 is worse still:** at m = 12 its median KL is 2.3 nats,
  because scores around −14 need m ≳ 38.
- **AGPT's depth-16 heads are just as sharp within their window** (median max
  weight 0.86).

## 8. Consequences

- **No cheap readout of a superposed RoPE state reproduces a sharp
  softmax.** A polynomial readout needs the moment tower (§3). A nonlinear
  decoder needs a state at least as large as the occurrence list (§4).
  Trained heads are sharp, so replacing attention over occurrences with a
  superposed state costs more than it saves. This is the mechanism behind
  `rnd/dist-rope-smoke` (collapsing positions broke RoPE) and behind
  `shared_key_rope.md`'s rule: do not collapse an occurrence set into one
  spectral fingerprint.
- **Three things survive.**
  - **Compression of compute, not of keys:** share the content projection
    across occurrences and keep each occurrence's position (shared-key RoPE).
  - **Exact merging:** merge only occurrences that coincide (the trie).
  - **A polynomial positional kernel (§5):** exact softmax over superposed
    occurrences, paid for in positional resolution.
- **A positional design that merges occurrences has to say what readout it
  serves.** If the answer is softmax over RoPE logits, it must carry
  between e/2 and e times the logit range in moment orders, or keep the
  occurrences. The alternative
  is to replace the RoPE logit's positional part with a polynomial kernel.
