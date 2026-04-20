# Learned Suffix-Depth Blending for a Depth-Limited Prefix Tree

## Goal

We have:

- a prefix tree / suffix-state graph with maximum depth `D`
- a model sequence length `L`, where `L` may be greater than `D`
- a shift rule that updates the active tree state by dropping the oldest token and appending the newest token

The problem is:

- deepest suffix states are the most specific
- but they are also the sparsest
- so using only the deepest match is often too brittle

The solution is:

- **trace** using the deepest valid suffix state
- **predict** using a **learned blend** over all suffix depths from that state back to root

---

## Core State Update

Let the token sequence be:

    x_1, x_2, ..., x_T

At timestep `t`, define the active suffix state of depth at most `D`:

    s_t^(D) = suffix_D(x_1:t)

This is the deepest suffix state available at time `t`.

Then update by shift:

    s_(t+1)^(D) = shift(s_t^(D), x_(t+1))

Interpretation:

- do NOT restart at root
- do NOT trace a length-`L` path through the tree
- only maintain the rolling depth-`D` suffix state

---

## Suffix Chain

From the active deepest state, collect its ancestry back to root:

    s_t^(D), s_t^(D-1), ..., s_t^(1), s_t^(0)

where:

- `s_t^(k)` is the suffix state of length `k`
- `s_t^(0)` is the root / unigram-style fallback state

Each suffix state defines a local empirical next-token distribution:

    P_hat_k(. | t) = P_tree(. | s_t^(k))

These are fixed distributions from the tree counts.

---

## Learned Blending

Instead of hand-designing weights from count / entropy / branching with many knobs, learn the blending weights.

Define per-depth score:

    a_k(t) = Gate(z_k(t), h_t)

where:

- `z_k(t)` = small feature vector for suffix depth `k`
- `h_t` = optional long-range hidden state
- `Gate` = small learned function (linear layer or tiny MLP)

Then normalize with softmax:

    lambda_k(t) = exp(a_k(t)) / sum_j exp(a_j(t))

and define the blended tree prior:

    P_blend(. | t) = sum_{k=0..D} lambda_k(t) * P_hat_k(. | t)

Properties:

- all `lambda_k(t) >= 0`
- `sum_k lambda_k(t) = 1`
- deeper suffixes can dominate when useful
- shallower suffixes can dominate when deeper ones are weak

---

## Minimal Gate Features

A simple first feature vector for each depth `k`:

    z_k(t) = [
      k,
      log(1 + count_k),
      entropy_k,
      max_prob_k
    ]

where:

- `count_k` = visit count of node `s_t^(k)`
- `entropy_k` = entropy of `P_hat_k`
- `max_prob_k` = largest next-token probability at that node

A smaller first-pass version is also reasonable:

    z_k(t) = [
      k,
      log(1 + count_k)
    ]

That keeps overhead low.

---

## Optional Long-Range Residual

The blended tree prior can be used directly, or corrected by a learned model.

Let:

    h_t = long-range hidden state summarizing x_1:t

Let:

    b_t = W h_t

where:

- `h_t` is a hidden vector
- `W` maps hidden state to vocab-sized bias logits

Then final logits are:

    logits_t = log(P_blend(. | t) + eps) + b_t

and final probabilities are:

    P_t = softmax(logits_t)

Interpretation:

- `P_blend` handles local context via the tree
- `b_t` is a learned residual correction from longer-range context

If desired, start without the residual and add it later.

---

## Training Objective

Given target next token `y_t`, define cross-entropy loss:

    L_t = -log P_t(y_t)

If using only the blended tree prior:

    P_t = P_blend(. | t)

If using the residual model:

    P_t = softmax(log(P_blend(. | t) + eps) + W h_t)

---

## Gradient Flow

### Case A: Learned blend only

The tree distributions `P_hat_k` are fixed.

Gradients flow into:

- gate parameters
- optional hidden state used by the gate

The gate learns:

- when to trust deep suffixes
- when to fall back toward shallower suffixes

### Case B: Learned blend + residual

Let:

    delta_t = P_t - one_hot(y_t)

Then for the residual map:

    dL/dW = delta_t * h_t^T
    dL/dh_t = W^T * delta_t

The tree remains fixed; learned components only correct or reweight it.

---

## Why This Helps

This separates two concerns cleanly:

### 1. Tracing
Use the rolling deepest suffix state:

    shift + longest valid suffix

### 2. Prediction
Use a learned blend over the full suffix chain:

    depth D down to root

So:

- tracing stays cheap and deterministic
- prediction becomes robust to sparse deep nodes
- no manual backoff formula is hard-coded

---

## Computational Overhead

Main extra costs:

### 1. Collect suffix-chain distributions
Need `D + 1` suffix-level distributions per position.

Mitigation:
- keep `D` modest
- store ancestor links
- compute only for active path
- reuse cached node statistics

### 2. Gate evaluation
Need one small score per depth.

Mitigation:
- use a tiny linear gate first
- use only 2-4 scalar features per depth

### 3. Mixture formation
Need weighted sum over `D + 1` vocab distributions.

Mitigation options:
- do it only over top-k child support plus fallback bucket
- keep sparse support from tree nodes
- or blend logits / counts before densifying if implementation allows

In practice, the gate should be tiny compared with the rest of the model.

---

## Important Design Principle

Use the tree for:

- exact local structure
- reusable prefix statistics
- cheap suffix-state transitions

Use learned blending for:

- deciding which suffix depth to trust

Use optional residual model for:

- long-range disambiguation
- context not captured by depth `D`

---

## Minimal First Version

Recommended first experiment:

1. Maintain rolling active suffix state with shift.
2. Collect suffix ancestry back to root.
3. Compute `P_hat_k` for each depth.
4. Learn `lambda_k` with a tiny gate from:
   - depth
   - log count
5. Blend:
   
       P_blend = sum_k lambda_k P_hat_k

6. Train with cross-entropy on `P_blend`.
7. Add residual `W h_t` only after the blend-alone version is working.

This minimizes moving parts and overhead.

---

## Pseudocode

```python
state = shift(prev_state, x_t)              # deepest active suffix state
chain = ancestors_to_root(state)            # [s^(D), s^(D-1), ..., root]

dists = []
scores = []

for k, node in enumerate(chain):
    p_hat = node.next_token_distribution()  # fixed from tree counts
    count = node.count
    z = [k, log1p(count)]                   # minimal version
    a = gate(z, h_t)                        # tiny learned score
    dists.append(p_hat)
    scores.append(a)

lambdas = softmax(scores)

p_blend = 0
for lam, p_hat in zip(lambdas, dists):
    p_blend += lam * p_hat

if use_residual:
    logits = log(p_blend + eps) + W @ h_t
    p_final = softmax(logits)
else:
    p_final = p_blend

loss = cross_entropy(p_final, y_t)
```

---

## One-Line Summary

**Trace with the deepest rolling suffix state; predict with a learned mixture over all suffix depths; optionally add a learned long-range residual on top.**

---

# Addendum: separating the three ideas

The document above reads as an integrated proposal, but it is actually
*three structurally independent ideas* that should be evaluated
separately. Tangling them together makes the proposal sound novel in
aggregate while obscuring which component does the work.

## Idea 1: Depth-gradient blending vs. single-depth lookup

When the trie depth `D` is smaller than the model's `seq_len`, we
have to decide how to consult the trie. Three options:

- **Back to root each step** — discards most context, wasteful.
- **Longest-matching suffix** (classical) — always lands in the
  sparsest, noisiest part of the trie, because by construction it
  keeps trimming until the match is as deep as possible, which is
  exactly where mass-1 paths dominate.
- **Blend across all depths of the suffix chain** — proposed here.

The objection to longest-match is structural, not a mere preference.
Longest-match is a *selection bias* toward the regime where the
trie's signal is least reliable. Blending doesn't just "use more
information" — it actively corrects that bias by not always trusting
the deepest match. That is the real argument for Idea 1.

## Idea 2: Learned blending weights vs. fixed backoff formula

Given that we blend, *how* do we weight the levels? Two options:

- **Fixed formula** based on count, entropy, depth (classical
  smoothing).
- **Learned gate function** producing context-dependent `λ_k`.

This is a completely orthogonal axis from Idea 1. Blending can be
done with fixed weights (a hand-coded formula over depth and count)
and it still addresses the selection-bias problem.

## Idea 3: Residual training vs. full model training

Given a blended tree prior, what does the learned model do?

- **Option A**: predict directly with `P_blend` (no residual).
- **Option B**: predict with `P_blend + learned correction`
  (residual model).

This is orthogonal to both Ideas 1 and 2. You can blend without a
residual, you can use a residual without blending (just pick one
depth, add residual on top), or you can do both or neither.

## Classical NLP precedent worth naming

The depth-blending proposal is structurally identical to
**interpolated n-gram smoothing** — Jelinek-Mercer from the 1980s is
the canonical reference. JM interpolates across n-gram orders:

```
P_JM(w | context) = λ_1·P(w | n-gram) + λ_2·P(w | (n-1)-gram) + ... + λ_k·P(w | unigram)
```

where the λ's can be fixed, learned from held-out data, or made
context-dependent. Kneser-Ney smoothing (the standard for n-gram LMs
before neural LMs) is a more sophisticated member of the same family.

What is potentially novel in the present proposal:

- **Learned context-dependent `λ_k` via a small neural gate.**
  Most classical interpolation learned `λ` as context-independent
  from held-out counts, or used simple rules. A neural gate
  conditioned on context plus per-depth features is a modern twist
  without direct classical precedent.
- **Applied to a neural setup**, with the interpolated tree serving
  as a prior for a neural model. The classical work did not do this.
- **Depth up to D=16 or D=32**; classical n-gram smoothing rarely
  went past 5-gram.

The blending-plus-residual combination has a clean
modern-classical-hybrid character: classical smoothing as the prior,
neural as the residual. Worth naming this explicitly in any writeup;
reviewers familiar with classical NLP will recognize it immediately.

## Gotcha on Idea 3 (residual training)

Residual training is cleaner than full training *only if the trie
prior is good*. For regimes where even the blended trie prior is
weak (e.g., very rare prefixes where the blend is noise even at its
best depth), the residual has to do all the work AND cancel out the
bad prior. At that point the system has reinvented a transformer
with extra steps.

This should be tested empirically: compare residual-only training
against full training at matched compute. If the prior is
informative, residual training should converge faster and match or
beat baseline; if not, the residual will do disproportionate work
and the architecture's motivation weakens.

## Clean ablation design

A 2×2 design isolates the contribution of each idea:

|              | No blend (single depth) | Blend (fixed λ) | Blend (learned λ) |
|--------------|:-:|:-:|:-:|
| Full training | baseline | (+Idea 1, fixed) | (+Ideas 1+2) |
| Residual     | (+Idea 3 only) | (+Ideas 1+3, fixed) | (+Ideas 1+2+3) |

Minimum-interesting experiment: **baseline vs. all-three-combined**.
The clean story from that is "the combination beats baseline at
matched compute; ablating blending, learned gate, or residual each
degrade the win."

## Framing for a writeup

- State Idea 1 as a correction to longest-match's selection bias
  toward sparse tail nodes, not as a mere "smoother."
- Cite Jelinek-Mercer/Kneser-Ney explicitly. Not citing would invite
  a reviewer to flag the omission.
- Present Idea 2 as a modest-but-novel extension: the neural gate
  enables context-dependent interpolation, not just count-dependent.
- Present Idea 3 as an optional decomposition of the learning
  problem, with its own success criterion (faster convergence at
  matched compute vs. baseline).

Read together, the three ideas form a coherent system with a
defensible classical grounding. Read separately, each is a testable
claim. Both framings are legitimate; the paper should make the
separation explicit so reviewers can evaluate the pieces.
