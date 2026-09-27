# AGPT Ultra — Python research track

This directory contains the former `agpt-ultra` repository, imported with its
Git history into [AGPT](../..). It explores exact prefix reuse with small
recurrent models, Fisher-based updates, and count-prior residual models. Run its
scripts from this directory so their relative data and output paths resolve.

The [research notebook](notebook/README.md) records the investigation. See
[Experiment records](EXPERIMENTS.md) before comparing any numbers with the CUDA
AGPT track. The former repository's paper was identical to the [AGPT paper
draft](../../docs/paper.md), so this tree points to the shared copy.

Prefix-structured language model experiments.

The first prototype is intentionally small: a PyTorch character RNN trained on
fixed-length character samples. Samples are prepared up front, sorted
lexicographically, and represented as a prefix trie so shared prefixes like
`ABCDEF` in `ABCDEFQRS` and `ABCDEFXYZ` have one parent node.

This gives us two equivalent views:

- a real trie of prefix nodes with counts and terminal markers
- a stack schedule over sorted samples: keep the common prefix state, pop the old
  suffix, then push the new suffix

For the first model, prefix caching is exact because an RNN hidden state after a
prefix is the complete parent state needed by all children.

## Installation

Use Python 3.11 or newer with PyTorch installed. Run the commands from
`research/ultra/`; the scripts import the local `agpt_ultra` package directly.

The included `data/input.txt` and `data/smoke.txt` are sufficient for the
commands below. GPU access is optional for the small tests and examples.

## Usage

Put Tiny Shakespeare at `data/input.txt`, then run:

```sh
python3 scripts/train_char_rnn.py --input data/input.txt --steps 200
```

Run the Python tests:

```sh
python3 -m unittest discover -s tests -v
```

The tests verify that trie/stack cached logits match naive evaluation and that:

```text
J_p * sum_s(g_s) == sum_s(J_p * g_s)
```

for a shared parent Jacobian `J_p`.

## Local Fisher

Each trie node exposes an outgoing count row `M[v, .]` through its child counts.
That row parameterizes the empirical next-token distribution `q` at the node.
From `q` we can materialize:

```text
categorical covariance = diag(q) - q q^T
empirical Fisher       = W^T (diag(q) - q q^T) W
```

The additive object in the trie is the count row. The local Fisher is then
derived exactly from aggregated counts. This matters because covariance is not,
in general, a plain sum of child covariances without a between-distribution
correction term.

For the node-local natural-gradient experiment, the cleaner state-space form is:

```text
p_v       = softmax(W h_v)
g_v       = W^T (p_v - q_v)
F_v       = W^T (diag(p_v) - p_v p_v^T) W
delta h_v = -(F_v + lambda I)^-1 g_v
```

This avoids K-FAC at first by working in hidden-state space instead of full
parameter space. Child evidence can be reconciled upward as summed state
evidence:

```text
F_p = F_p_local + sum_c F_c
g_p = g_p_local + sum_c g_c
```

## Trie Objective

The node-local training objective is:

```text
L = -sum_{p in T} sum_x n(p, x) log pi_theta(x | h_p)
```

Here `n(p, x)` is the outgoing child count from prefix node `p` to token `x`.
Each node can start from an immutable snapshot of the global `f_theta` weights,
make a local child-conditioned proposal, and pass a local model `theta_i` plus
its evidence geometry upward.

At the parent, child models are reconciled as a Fisher-weighted quadratic merge:

```text
theta_p = argmin_theta sum_i 1/2 (theta - theta_i)^T F_i (theta - theta_i)
```

Equivalently:

```text
theta_p = (sum_i F_i)^-1 sum_i F_i theta_i
```

with damping and optional prior terms added in practice.

Equivalently, if each child passes loss-gradient evidence `(F_i, g_i)` instead
of an already-updated `theta_i`, the parent sums evidence first:

```text
F = sum_i F_i
g = sum_i g_i
delta theta = -(F + lambda I)^-1 g
```

This repo uses `g = grad L`, so the descent step has a minus sign. If `g` is
defined as the descent force `-grad L`, the same update is written
`delta theta = (F + lambda I)^-1 g`.

## Epoch Flow

One training epoch is a bottom-up tree pass:

```text
global theta_t is broadcast immutably to every node
each node creates a local proposal theta_i from its children/statistics
children pass (theta_i, F_i) upward
each parent reconciles child models into theta_p
the root's theta_root becomes global theta_{t+1}
```

The next epoch repeats the same process using `theta_{t+1}` as the immutable
starting model for all nodes.

## Head-Only Prototype

The first runnable end-to-end trainer freezes the recurrent body and updates
only the output head:

```text
theta = flatten(W, b)
pi(x | h_p) = softmax(W h_p + b)
g_p = n_p * flatten(outer(p - q, [h_p, 1]))
F_p = n_p * kron(diag(p) - p p^T, [h_p, 1] [h_p, 1]^T)
theta_next = theta - (sum_p F_p + lambda I)^-1 sum_p g_p
```

Run it on the smoke corpus:

```sh
python3 scripts/train_head_only.py --input data/smoke.txt --block-size 16 --stride 4 --epochs 3
```

## Hybrid Prototype

The next trainer makes `f_theta` a trainable recurrent model while keeping the
Fisher block simple:

```text
embeddings: frozen
GRU cell: ordinary gradient step on trie objective
output head: exact trie Fisher natural step
```

Run it on the smoke corpus:

```sh
python3 scripts/train_hybrid.py --input data/smoke.txt --block-size 16 --stride 4 --epochs 3
```

For a bounded Tiny Shakespeare run:

```sh
python3 scripts/train_hybrid.py \
  --input data/input.txt \
  --block-size 16 \
  --stride 16 \
  --max-train-samples 512 \
  --max-val-samples 128 \
  --epochs 2 \
  --hidden-size 8 \
  --embedding-size 8 \
  --head-damping 50 \
  --head-step-scale 0.5 \
  --max-cg-iter 32
```

The head Fisher update is matrix-free by default: it preserves the exact Fisher
vector product but solves the damped natural step with conjugate gradient instead
of materializing the dense matrix.

## Baseline

The comparison baseline uses the same frozen-embedding GRU model and the same
trie objective, but trains both the GRU cell and output head with AdamW:

```sh
python3 scripts/train_baseline.py \
  --input data/input.txt \
  --block-size 16 \
  --stride 16 \
  --max-train-samples 512 \
  --max-val-samples 128 \
  --epochs 2 \
  --hidden-size 8 \
  --embedding-size 8
```

## Development

```sh
python3 -m unittest discover -s tests -v
```

The imported code is licensed under the repository's [PolyForm Noncommercial
License 1.0.0](../../LICENSE). Thomas Sawyer is the author of this track.
