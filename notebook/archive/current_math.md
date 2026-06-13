# Current Math

## Trie Objective

Samples are fixed-length character blocks. We build a prefix trie. For each node
`v`, let prefix `p_v` have transition counts:

```text
n(v, x) = number of samples where token x follows prefix v
N_v     = sum_x n(v, x)
q_v(x)  = n(v, x) / N_v
```

The trie objective is:

```math
L(\theta) = -\sum_{v \in \mathcal T}\sum_x n(v,x)\log p_\theta(x \mid h_v)
```

where:

```math
h_v = f_\phi(p_v)
```

```math
z_v = W h_v + b
```

```math
p_v = \operatorname{softmax}(z_v)
```

Current parameters are split as:

```text
phi   = GRU body parameters
theta = output head parameters = vec(W, b)
```

Embeddings are frozen. The GRU body is updated with AdamW. The output head is
updated with a matrix-free natural-gradient step.

## Per-Node Head Gradient

For one node:

```math
\ell_v(\theta) = -\sum_x n(v,x)\log p_v(x)
```

Let:

```math
r_v = p_v - q_v
```

Then:

```math
\nabla_W \ell_v = N_v r_v h_v^\top
```

```math
\nabla_b \ell_v = N_v r_v
```

Flattened:

```math
g_v = \operatorname{vec}(\nabla_W \ell_v, \nabla_b \ell_v)
```

Global gradient:

```math
g = \sum_v g_v
```

## Per-Node Fisher Action

For logits:

```math
C_v = \operatorname{Diag}(p_v) - p_v p_v^\top
```

For a candidate head step:

```math
\Delta\theta = (\Delta W, \Delta b)
```

the induced logit step is:

```math
u_v = \Delta W h_v + \Delta b
```

and:

```math
C_v u_v = p_v \odot (u_v - p_v^\top u_v)
```

The node Fisher action is:

```math
F_v \Delta\theta =
N_v
\begin{bmatrix}
(C_v u_v)h_v^\top \\
C_v u_v
\end{bmatrix}
```

Global Fisher action:

```math
F\Delta\theta = \sum_v F_v\Delta\theta
```

We do not materialize dense `F`; we compute exact matrix-vector products.

## Natural Head Step

We solve:

```math
(F+\lambda I)\Delta\theta = -g
```

with conjugate gradient, then apply:

```math
\theta \leftarrow \theta + \eta \Delta\theta
```

## Quadratic Predictor

For a candidate scale `eta`, the local quadratic model is:

```math
L(\theta+\eta\Delta\theta)
\approx
L(\theta)
+ \eta g^\top\Delta\theta
+ \frac{1}{2}\eta^2 \Delta\theta^\top F\Delta\theta
```

Define:

```math
l = g^\top\Delta\theta
```

```math
q = \Delta\theta^\top F\Delta\theta
```

Predicted improvement:

```math
\widehat{\Delta L}(\eta)
=
-\left(\eta l + \frac{1}{2}\eta^2 q\right)
```

The unconstrained quadratic step scale is:

```math
\eta_\text{quad} = -\frac{l}{q+\epsilon}
```

## Trust Radius

The trust region is expressed in Fisher norm:

```math
\|\eta\Delta\theta\|_F
=
\eta\sqrt{\Delta\theta^\top F\Delta\theta}
=
\eta\sqrt{q}
\le \tau
```

So:

```math
\eta_\text{trust} = \frac{\tau}{\sqrt{q+\epsilon}}
```

The suggested step scale is:

```math
\eta_\text{suggested}
=
\min(\eta_\text{quad}, \eta_\text{trust}, \eta_\max)
```

If no trust radius is configured, `eta_trust` is omitted.

## Backtracking

If line search is enabled, we test:

```text
eta_suggested
eta_suggested / 2
eta_suggested / 4
...
```

until the chosen target loss improves. The recommended default is post-body
training loss, with validation used only for diagnostics.

If no tested scale improves:

```math
\eta = 0
```

## Calibration

For the accepted scale, we record:

```math
\Delta L_\text{train}
=
L_\text{train before} - L_\text{train after}
```

```math
\rho_\text{train}
=
\frac{\Delta L_\text{train}}{\widehat{\Delta L}(\eta)}
```

If a validation trie is provided:

```math
\Delta L_\text{val}
=
L_\text{val before} - L_\text{val after}
```

```math
\rho_\text{val}
=
\frac{\Delta L_\text{val}}{\widehat{\Delta L}(\eta)}
```

`rho_train` is the right signal for trust-region adaptation. `rho_val` is a
generalization diagnostic and should not be used as the primary line-search
acceptance target in larger experiments.

## Current Caveat

This is still not the full original AGPT program where every node owns a local
full model copy and child models reconcile into parent models. The current
implementation accumulates all node evidence into one global head update:

```math
g = \sum_v g_v
```

```math
F = \sum_v F_v
```

```math
\Delta\theta = -(F+\lambda I)^{-1}g
```

one global head update per epoch.
