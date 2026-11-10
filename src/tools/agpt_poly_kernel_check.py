#!/usr/bin/env python3
"""Exactness check for the polynomial positional kernel
(notes/seq-len-extension/superposition-softmax-limit.md, section 5).

Softmax over every older occurrence with weight exp(content) * |sum_j b_j lam_j^D|^2
is computed exactly by J complex recurrent states per symbol (equispaced frequencies).
Prints the worst relative error of the recurrent readout against the direct sum.
"""
import numpy as np
rng = np.random.default_rng(0)
V, J, T, r, w0 = 5, 8, 400, 0.99, 2 * np.pi / 108
x = rng.integers(0, V, T)
mu = r**2 * np.exp(1j * np.arange(J) * w0)          # atoms for n = j - l = 0..J-1
S = np.zeros((V, J), complex)
worst = 0.0
for t in range(T):
    b = rng.normal(size=(V, J)) + 1j * rng.normal(size=(V, J))   # kernel may depend on (query, key char)
    content = rng.normal(size=V)
    # readout from the state: sum_{j,l} b_j conj(b_l) S^{(j-l)}, S^{(-n)} = conj S^{(n)}
    Sfull = lambda n: S[:, n] if n >= 0 else np.conj(S[:, -n])
    Zrec = np.exp(content) * sum(b[:, j] * np.conj(b[:, l]) * Sfull(j - l) for j in range(J) for l in range(J)).real
    # direct: sum over every older occurrence p < t of exp(content) * |sum_j b_j (r e^{i j w0})^(t-p)|^2
    Zdir = np.zeros(V)
    for p in range(t):
        D = t - p
        w = abs((b[x[p]] * (r * np.exp(1j * np.arange(J) * w0)) ** D).sum()) ** 2
        Zdir[x[p]] += np.exp(content[x[p]]) * w
    if t:
        worst = max(worst, np.max(np.abs(Zrec - Zdir) / np.maximum(Zdir, 1e-300)))
    S = mu[None, :] * (S + (np.arange(V)[:, None] == x[t]))     # S(t+1) = mu (S(t) + [x_t = c])
print('max relative error, recurrent vs direct:', worst)
