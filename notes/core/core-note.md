# Core Note: Prefix-Trie Gradient Factorization

## 1. Trie Formulation

Let:
- \( \mathcal{T} \) be the set of prefix nodes
- \( n(p,x) \) be the count of token \(x\) following prefix \(p\)

Define:
\[
N_p = \sum_x n(p,x)
\]

State recurrence:
\[
h(\epsilon) = h_0
\]
\[
h_{p\cdot x} = f_\theta(h_p, x)
\]

---

## 2. Model Prediction

Logits:
\[
z_{p,x} = W_x \cdot h_p
\]

Softmax:
\[
\pi_{p,x} = \frac{e^{z_{p,x}}}{\sum_{x'} e^{z_{p,x'}}}
\]

---

## 3. Loss (Trie Form)

\[
L = -\sum_{p \in \mathcal{T}} \sum_x n(p,x)\log \pi_{p,x}
\]

---

## 4. Local Gradient

Define edge error:
\[
e_{p,x} = \pi_{p,x} N_p - n_{p,x}
\]

Node-local gradient:
\[
g_p^{\text{local}} = \sum_x e_{p,x} W_x
\]

---

## 5. Backpropagation over Trie

\[
G_p = g_p^{\text{local}} + \sum_x J_{p\to p\cdot x} \cdot G_{p\cdot x}
\]

Where:
\[
J_{p\to p\cdot x} = \frac{\partial h_{p\cdot x}}{\partial h_p}
\]

---

## 6. Core Identity (Gradient Factorization)

Let:
- \(J_p\) = product of Jacobians along a prefix
- \(g_s\) = gradient contribution from suffix \(s\)

Then:
\[
\boxed{
J_p \cdot \left(\sum_{s} g_s\right)
=
\sum_{s} \left(J_p \cdot g_s\right)
}
\]

---

## 7. Interpretation

- Right-hand side: per-path gradient accumulation (standard training)
- Left-hand side: aggregated suffix gradient (trie training)

---

## 8. Key Result

\[
\boxed{
\frac{\partial L}{\partial h_p}
=
J_p \cdot \left(\sum_{s \in \text{subtree}(p)} g_s\right)
}
\]

---

## 9. Consequence

Shared prefix computation allows:

- aggregation before transformation
- elimination of redundant Jacobian applications
- larger and lower-variance gradient updates

---

## 10. Summary

The trie formulation enables:

> factoring shared prefix derivatives out of the sum over suffix gradients

which yields:

- computational reduction  
- structural simplification  
- improved gradient efficiency  
