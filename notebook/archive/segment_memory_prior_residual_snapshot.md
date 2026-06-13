# Segment Memory + Count Prior Residual

Active thread: frozen recursive count-gate prior plus neural residual over
segment-memory states.

## Current Form

The count prior owns local empirical next-character statistics:

```text
q(root) = p_root
q(c) = w(c) * p_mle(next | c) + (1 - w(c)) * q(suffix(c))
w(c) = sigmoid(theta . features(c))
```

The neural model learns a residual correction:

```text
logits = log p_prior + alpha * residual_logits
loss = CE(logits, target) + lambda * ||alpha * residual_logits||^2
```

## Fair Carved Split Results

Baseline count prior:

```text
prior only: 3.8594965
```

Residual trust/capacity sweep:

```text
model  alpha  residual_L2  curve
d64    0.10   0.01         3.859 -> 3.815 -> 3.880 -> 3.910
d96    0.10   0.01         3.859 -> 3.852 -> 3.939 -> 4.030
d96    0.05   0.01         3.859 -> 3.855 -> 3.845 -> 3.884
d96    0.10   0.05         3.859 -> 3.822 -> 3.823 -> 3.855
d64    0.05   0.01         3.859 -> overflow during epoch-1 eval
```

Current best:

```text
d64, alpha=0.10, residual_L2=0.01, epoch 1: 3.815
```

## Interpretation

The residual can improve a strong count prior, but only if it pays for changing
the prior. Width alone is not the next lever: d96 needed stronger residual
regularization and still did not beat d64.

The next useful tests should tune around the winning d64 trust setting:

```text
d64 alpha=0.10 L2=0.02
d64 alpha=0.10 L2=0.05
d64 alpha=0.15 L2=0.05
```

