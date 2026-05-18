# AGPT Note: Measure Loss During Subtree Training

## Goal

When training an AGPT subtree, we should measure the model's loss for that subtree **as a byproduct of the normal forward/backward pass**.

We do **not** want a separate full traversal just to compute curriculum statistics.

The training pass already computes token-level cross-entropy losses. We should accumulate those losses into lightweight statistics attached to the subtree root and, optionally, its immediate children.

---

## Core idea

When training a subtree rooted at prefix `p`, we already visit many descendant nodes:

```text
p
├── p+a
├── p+b
│   ├── p+b+c
│   └── p+b+d
└── p+e
```

For every prediction made inside this subtree, we compute a token loss:

```text
loss = cross_entropy(model_logits, target_token)
```

Instead of throwing that scalar away after backprop, accumulate it:

```text
loss_sum[p] += loss * count
token_sum[p] += count
```

Then the average subtree loss is:

```text
avg_loss[p] = loss_sum[p] / token_sum[p]
ppl[p] = exp(avg_loss[p])
```

This gives us:

> “How well does the current model predict continuations under prefix `p`?”

That becomes the curriculum signal.

---

## Important weighting rule

Use corpus/count weighting.

If a trie edge or prediction represents `count` occurrences in the corpus, then its loss contribution should be weighted by that count:

```text
loss_sum += loss * count
token_sum += count
```

Do not average nodes equally. A rare node and a frequent node should not contribute the same amount.

---

## Minimal data structure

Each active subtree root should have something like:

```crystal
struct SubtreeStats
  property loss_sum : Float64 = 0.0
  property token_sum : Int64 = 0
  property avg_loss : Float64 = 0.0
  property ppl : Float64 = 0.0
end
```

During training:

```crystal
stats.loss_sum += token_loss * count
stats.token_sum += count
```

After training the subtree:

```crystal
stats.avg_loss = stats.loss_sum / stats.token_sum
stats.ppl = Math.exp(stats.avg_loss)
```

---

## What counts as a token loss?

For each trie node representing prefix `q`, the model predicts the next token distribution from that prefix.

If `q` has children:

```text
q -> q+a count 10
q -> q+b count 3
q -> q+c count 1
```

Then the loss at `q` is the weighted cross-entropy over the empirical next-token distribution:

```text
loss(q) =
  - Σ_child [ count(q+child) / count(q) ] * log P_model(child | q)
```

Equivalent count-weighted form:

```text
loss_sum += - count(q+a) * log P_model(a | q)
loss_sum += - count(q+b) * log P_model(b | q)
loss_sum += - count(q+c) * log P_model(c | q)

token_sum += count(q+a)
token_sum += count(q+b)
token_sum += count(q+c)
```

Since:

```text
count(q) = Σ_child count(q+child)
```

this is equivalent to accumulating one loss per represented corpus occurrence.

---

## Why this is useful

After training subtree `p`, we know whether it is still “hard” for the model.

For curriculum scheduling:

```text
if avg_loss[p] remains high:
    split p into child subtrees
else:
    keep p aggregated
```

This preserves the AGPT advantage:

- easy regions stay large and aggregated
- hard regions get refined into smaller subtrees
- loss measurement is essentially free because it happens during training

---

## Optional child-level stats

While training subtree `p`, also accumulate stats for its immediate children.

Example:

```text
training subtree "A"

also collect stats for:
  "Aa"
  "Ab"
  "Ac"
  ...
```

Then if `"A"` remains high-loss, we already know which child regions are responsible.

This allows selective splitting:

```text
split only high-loss children
```

instead of blindly expanding every child.

Implementation approach:

```text
For every prediction inside subtree p:
  update stats[p]

  let child = immediate child of p that contains this prediction
  update stats[child]
```

For example, while training `"A"`:

```text
prediction at prefix "And"
belongs to:
  subtree root: "A"
  immediate child bucket: "An"
```

So update both:

```text
stats["A"]  += loss contribution
stats["An"] += loss contribution
```

---

## Suggested first version

Start simple:

```text
For each active subtree root p:
  zero stats[p]

  train subtree p normally

  during loss computation:
    stats[p].loss_sum += token_loss * count
    stats[p].token_sum += count

  after subtree:
    stats[p].avg_loss = stats[p].loss_sum / stats[p].token_sum
    stats[p].ppl = exp(stats[p].avg_loss)
```

Then after an epoch, rank active subtrees by residual loss:

```text
residual[p] = count(p) * max(avg_loss[p] - global_avg_loss, 0)
```

Split the highest residual subtrees.

---

## Pseudocode

```pseudo
for subtree_root in active_subtrees:
    stats = SubtreeStats.new

    train_subtree(subtree_root) do |prefix_node, logits|
        # prefix_node has children representing observed next tokens
        for child in prefix_node.children:
            target_token = child.token
            count = child.count

            log_prob = log_softmax(logits)[target_token]
            token_loss = -log_prob

            # normal AGPT loss accumulation / backward contribution
            training_loss += token_loss * count

            # curriculum stats
            stats.loss_sum += token_loss * count
            stats.token_sum += count
        end
    end

    stats.avg_loss = stats.loss_sum / stats.token_sum
    stats.ppl = exp(stats.avg_loss)

    subtree_root.cached_loss = stats.avg_loss
    subtree_root.cached_ppl = stats.ppl
end
```

---

## Notes / cautions

1. Use the same loss definition as training.
   If training uses weighted empirical next-token distributions, stats should use the same weighting.

2. Do not average per node unless nodes are weighted by corpus count.

3. Early in training, all subtrees will have high loss. Do not split immediately based on random-initialization loss.

4. Use loss after one or more broad AGPT passes as the signal.

5. Cached loss does not need to be perfectly fresh. It is a curriculum heuristic, not an exact objective.

6. This should add very little overhead because the loss is already computed during the subtree training pass.
