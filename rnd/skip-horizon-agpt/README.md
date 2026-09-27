# Skip-Horizon AGPT

Status: design note.

Date: 2026-06-18.

## Motivation

Standard AGPT attaches one empirical next-token target distribution to each
prefix node:

```text
P_out(p) = P(x_t | context ending at t - 1)
```

That gives the tree a strong local objective, but the node only sees the
immediate next-token distribution inside the chosen tree horizon. Attempts to
add longer context through successor paths, phase trees, or history attention
were either awkward or moved away from the clean AGPT objective.

The skip-horizon idea keeps the AGPT object intact but gives each prefix node a
family of shifted empirical distributions. For a target token `x_t`, ordinary
next-token prediction uses the context ending at `t - 1`. Skip-horizon evidence
also uses contexts ending earlier:

```text
context ending at t - 1  -> target x_t   horizon h = 1
context ending at t - 2  -> target x_t   horizon h = 2
context ending at t - 4  -> target x_t   horizon h = 4
context ending at t - 8  -> target x_t   horizon h = 8
...
```

The intervening tokens are deliberately hidden from that source. The question is
not "what future token follows this prefix after seeing the in-between text?"
It is:

```text
given only this earlier prefix, what distribution does it imply over the same
target token?
```

This gives AGPT a way to expose longer-range empirical structure without
changing the shared-prefix gradient factorization.

## Definition

Let `p` be a prefix node ending at corpus position `e`. For horizon `h >= 1`,
define:

```text
n_h(p, y) = count of occurrences where prefix p ends at e
            and corpus token x[e + h] = y

N_h(p) = sum_y n_h(p, y)

E_h(y | p) = n_h(p, y) / N_h(p)
```

`h = 1` is the usual AGPT next-token distribution:

```text
E_1(y | p) = P_out(y | p)
```

`h > 1` is a blind skip-horizon expert:

```text
E_h(y | p) = P(x_{e+h} = y | p)
```

Each horizon also has its own backoff ladder. For a fixed target `x_t`,
horizon `h` sets the context endpoint:

```text
endpoint = t - h
```

Depth/backoff then decides how much context before that endpoint is used:

```text
depth d:      x[t-h-d : t-h]       -> x_t
backoff d-1:  x[t-h-(d-1) : t-h]   -> x_t
backoff d-2:  x[t-h-(d-2) : t-h]   -> x_t
...
root:         unigram/root context -> x_t
```

So the more precise expert index is `(h, b)`, where `h` is the target horizon
and `b` is the context/backoff depth:

```text
E_{h,b}(y | p_b) = P(x_t = y | context p_b ending at t - h)
```

This matters because a deep context at a horizon can be sparse and noisy while
a shallower backoff at the same horizon can be better calibrated.

The same target token can therefore receive evidence from several shifted
contexts. Operationally, for a fixed target `x_t`, the model can consult:

```text
P(x_t | context ending t - 1)
P(x_t | context ending t - 2)
P(x_t | context ending t - 4)
P(x_t | context ending t - 8)
...
```

## Empirical Expert View

Each node owns a bank of empirical experts:

```text
E_{1,b}(p_b)      = ordinary next-token expert and its backoffs
E_{h,b}(p_b)      = skip-horizon expert h with backoff depth b
E_suffix_{h,b}(.) = optional reverse/suffix-side evidence for that source
```

Most skip experts, especially deep `(h, b)` sources at larger `h`, are expected
to be noisy:

```text
high entropy
high branch factor
low gain over backoff/unigram
```

The useful skip experts are the rare ones where the distribution collapses or
changes sharply:

```text
low entropy
sufficient mass
high reliability
high KL gain over backoff/count prior
strong agreement or useful disagreement with local evidence
```

These are the likely "concept hooks": earlier prefixes that constrain a later
target even while being blind to the intervening tokens.

## AGPT Objective

The simplest auxiliary objective keeps horizon/backoff sources separate:

```text
L = - sum_p sum_h sum_b lambda_{h,b}(p_b) N_{h,b}(p_b)
        sum_y E_{h,b}(y | p_b) log pi_theta(y | h_p, h, b)
```

where:

```text
h_p                    model state at prefix node p
p_b                    suffix/backoff context of p at depth b
pi_theta(.|h_p,h,b)    horizon/backoff-conditioned prediction head
lambda_{h,b}(p_b)      trust/weight for this source
```

This is a direct extension of the standard AGPT objective:

```text
L_standard = - sum_p N_1(p) sum_y E_1(y | p) log pi_theta(y | h_p)
```

The model can implement `pi_theta(. | h_p, h, b)` in several ways:

```text
shared head + horizon/depth embeddings
horizon-specific or source-specific output adapters
low-rank horizon adapter
separate auxiliary heads for diagnostics
```

An alternative is to first calibrate experts into a single target prior:

```text
P_cal(. | p) = sum_s w_s(p) E_s(. | p)

L = - sum_p N_p sum_y P_cal(y | p) log pi_theta(y | h_p)
```

where a source `s` may be a particular `(horizon, backoff depth)` pair.

The separate-head form is cleaner for diagnosing whether each horizon contains
usable information. The calibrated-prior form is closer to the IMM prior
mixture work.

## Gradient Compatibility

Skip-horizon targets do not break the aggregated-gradient identity. They only
change the local loss attached to a node.

For each `(p, h, b)`, define the local prediction error:

```text
e_{p,h,b,y} = lambda_{h,b}(p_b)
              * (N_{h,b}(p_b) * pi_theta(y | h_p, h, b)
                 - n_{h,b}(p_b, y))
```

The local hidden-state gradient becomes:

```text
g_p_local = sum_h sum_b sum_y e_{p,h,b,y}
              * d logit_{p,h,b,y} / d h_p
```

The recursive AGPT backward pass is unchanged:

```text
G_p = g_p_local + sum_child J_{p -> child} G_child
```

So the same factorization remains:

```text
shared prefix Jacobian applied once to aggregated descendant signal
```

The only difference is that the node's local empirical target is now a family
of horizon/backoff-indexed empirical targets rather than a single next-token
table.

## Relationship To Backoff Priors

The count/backoff gate learned in the prior-residual work can be viewed as a
calibrator over local and suffix evidence:

```text
q(c) = w(c) P_mle(. | c) + (1 - w(c)) q(suffix(c))
```

Skip-horizon AGPT generalizes the same idea to shifted evidence sources:

```text
P(. | p) = mixture over:
  local next-token expert
  local suffix/backoff experts
  skip horizon experts
  skip horizon backoff experts
```

The calibrator should see per-source features, not just a collapsed aggregate:

```text
source type
horizon h / H
backoff depth b / D
mass
branch count
entropy / sharpness
reliability
KL gain over local/backoff prior
agreement with count prior
agreement with other horizons
```

The IMM experiments suggest that collapsing horizons too early loses signal.
AGPT should preserve the `(horizon, backoff depth)` channel until the calibrator
or model can decide whether it is useful.

## Evidence From IMM

This idea was prototyped in the IMM prior/residual project and summarized in:

```text
~/Projects/imm/notes/skip-horizon-prior.md
```

Key observations from that note:

- Scalar skip-profile features helped slightly, showing reliability information
  is present.
- A single aggregate skip distribution improved initial PPL in one setting,
  proving actual token-vote information exists.
- Aggregating horizons was weaker on the larger 50k setting, suggesting that
  horizon identity matters.
- Keeping horizons as separate channels was the best current version.
- A gate-bias sweep showed a clear trust/noise tradeoff:
  small nonzero skip trust helped, while high skip trust hurt.

Representative 50k Tiny Shakespeare, `lr=1e-4`, `W=8`:

| run | epoch 2 PPL |
|-----|------------:|
| baseline | 10.083 |
| profile W=8 | 10.058 |
| aggregate distribution, manual | 10.075 |
| aggregate distribution, learned scalar gate | 10.079 |
| learned horizon distribution | 10.056 |

Gate-bias sweep:

| gate bias | initial PPL | epoch 1 PPL | epoch 2 PPL | mean skip alpha epoch 2 |
|----------:|------------:|------------:|------------:|------------------------:|
| -5 | 10.197 | 10.139 | 10.060 | 0.0082 |
| -4 | 10.196 | 10.137 | 10.056 | 0.0223 |
| -3 | 10.226 | 10.176 | 10.094 | 0.0600 |
| -2 | 10.360 | 10.358 | 10.294 | 0.1514 |

The evidence is not yet a large PPL win, but it is structurally meaningful:

```text
skip horizons contain predictive signal,
the signal is sparse/noisy,
and unconditional trust is wrong.
```

That is exactly the shape expected if skip-horizon experts are a long-range
signal source that needs calibration.

## First AGPT Experiment

Start diagnostic, not architectural:

1. Build skip tables for sparse horizons:

   ```text
   h in {1, 2, 4, 8, 16, 32, 64}
   ```

2. For each full-depth `(p, h)` source, compute:

   ```text
   mass
   branch count
   entropy
   max probability
   reliability = mass / (mass + branch_count)
   KL(E_h || E_1 or backoff)
   agreement with local prior
   ```

3. Evaluate the empirical experts directly on heldout target positions:

   ```text
   prior-only PPL
   horizon-only PPL by h
   calibrated mixture PPL
   bucketed gain by entropy/reliability/KL
   ```

4. Extend each horizon with its own suffix/backoff ladder and repeat the same
   feature extraction and direct eval by `(h, b)` source:

   ```text
   mass_{h,b}
   branch_count_{h,b}
   entropy_{h,b}
   reliability_{h,b}
   KL(E_{h,b} || local/backoff prior)
   horizon-only PPL by h
   best-backoff PPL by h
   calibrated mixture PPL over (h, b)
   bucketed gain by entropy/reliability/KL
   ```

5. Only after direct evidence survives full heldout eval, add the auxiliary
   AGPT loss:

   ```text
   L = L_next + beta * L_skip
   ```

6. Compare whether adding `L_skip` improves ordinary `h=1` heldout rolling PPL.

## Open Questions

- Should `lambda_{h,b}(p_b)` be learned offline as a count/stat calibrator,
  learned jointly with AGPT, or fixed from diagnostics?
- Should skip horizons train the same output head, horizon adapters, or only an
  auxiliary representation head?
- Should the model predict from the deepest node state `h_p` while supervising
  with shallower backoff sources, or should the backoff source use its own
  ancestor state `h_{p_b}`?
- How far can useful horizons extend before count sparsity dominates?
- Should horizons be dense `1..W`, powers of two, or selected by entropy/change
  events along corpus paths?
- Does skip-horizon training improve generation quality and long-range
  coherence, or only local PPL?
- Can suffix-side `P_in` statistics improve horizon calibration the same way
  they improved the learned count/backoff prior?

## Working Claim

Skip-horizon AGPT extends each prefix node from a single next-token empirical
target into a calibrated family of shifted empirical experts. Because each
expert is still a count-weighted local loss attached to the same prefix state,
the AGPT aggregated-gradient identity remains intact.

This may be the cleanest path for AGPT to receive long-range signal without
turning the framework into ordinary sliding-window attention.
