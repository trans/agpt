# Minimal AGPT Prototype
## Trie-Memory + Carried State Residual Model

This document specifies a **small, testable prototype** for AGPT as a **state-carrying prefix machine**.

The goal is to test this core hypothesis:

> A trie can handle exact local corpus continuation memory, while a small carried state vector handles dynamic/global correction, so we do not need transformer-style KV cache or full-context replay at inference.

This prototype intentionally avoids advanced features for now:

- no entropy features
- no suffix-link optimization
- no multi-state decomposition
- no attention
- no disk paging
- no complicated training tricks

It is meant to be the **smallest serious experiment**.

---

# 1. High-level idea

At any time step, the model keeps only:

- a **current trie node** representing the current prefix path
- a **continuous state vector** representing carried dynamic information

So the model state is:

\[
(n_t, s_t)
\]

where:

- \(n_t\) = trie node after consuming tokens up to time \(t\)
- \(s_t \in \mathbb{R}^d\) = learned carried state

Prediction does **not** replay the whole prefix.
Instead:

1. trie gives a local next-token distribution
2. learned state reweights that distribution
3. chosen token advances both trie node and carried state

---

# 2. Design goals

This prototype should:

- use a corpus trie as exact local memory
- use one carried state vector instead of KV cache
- train with standard recurrent-style backprop initially
- support candidate restriction via trie top-K
- be simple enough to implement and benchmark quickly

This prototype is **not** trying to beat transformers yet. It is trying to answer:

- does this architecture basically work?
- how much does trie memory reduce burden on the learned state?
- can a small carried state compete with a larger baseline on a small corpus?

---

# 3. Core architecture

The model has three main parts:

## 3.1 Trie
A prefix trie built from the corpus up to max depth \(L\).

Each node stores at minimum:

- child links by token
- next-token counts

Optional later additions can be layered in, but not needed for v1.

---

## 3.2 Token embeddings
Each token \(v \in V\) has embedding:

\[
e(v) \in \mathbb{R}^d
\]

---

## 3.3 Carried state update cell
A simple recurrent update:

\[
s_{t+1} = U(s_t, e(x_t))
\]

Use a gated update so state can retain or overwrite information.

---

# 4. Mathematical spec

## 4.1 Trie distribution

Let the current trie node be \(n_t\). It represents the current prefix path.

Define the trie next-token probability:

\[
P_{\text{trie}}(v \mid n_t)
=
\frac{\text{count}(n_t, v) + \epsilon}
{\sum_{u \in V} (\text{count}(n_t, u) + \epsilon)}
\]

where:

- \(\text{count}(n_t, v)\) = number of times token \(v\) followed this prefix in the corpus
- \(\epsilon > 0\) = small smoothing constant

For the minimal version, if a token has never been seen from this node, it just gets smoothing mass.

If the node has no outgoing edges, fall back to root or longest available suffix later. For v1, root fallback is fine.

---

## 4.2 Carried state update

Let current state be:

\[
s_t \in \mathbb{R}^d
\]

Let current input token embedding be:

\[
e_t = e(x_t)
\]

Use a gated recurrent update:

\[
g_t = \sigma(W_g s_t + U_g e_t + b_g)
\]

\[
\tilde{s}_{t+1} = \tanh(W_s s_t + W_x e_t + b_s)
\]

\[
s_{t+1} = g_t \odot s_t + (1 - g_t) \odot \tilde{s}_{t+1}
\]

where:

- \(W_g, U_g, W_s, W_x \in \mathbb{R}^{d \times d}\)
- \(b_g, b_s \in \mathbb{R}^d\)
- \(\sigma\) = sigmoid
- \(\odot\) = elementwise multiply

This is a minimal gated recurrent cell.

---

## 4.3 Residual scoring

For each candidate token \(v\), define a learned residual score:

\[
r(v \mid s_t) = s_t^\top e(v) + b_v
\]

where:

- \(e(v)\) = token embedding
- \(b_v\) = learned scalar output bias for token \(v\)

This is the simplest reasonable scorer.

---

## 4.4 Hybrid logits

Combine trie log-probability with learned residual score:

\[
\ell(v \mid n_t, s_t)
=
\alpha \log P_{\text{trie}}(v \mid n_t)
+
\beta\, r(v \mid s_t)
\]

where:

- \(\alpha\) = weight on trie prior
- \(\beta\) = weight on learned residual

Then normalize over candidate tokens:

\[
P(v \mid n_t, s_t)
=
\operatorname{softmax}(\ell(v \mid n_t, s_t))
\]

This is the final next-token distribution.

---

# 5. Candidate restriction

Scoring the entire vocabulary is optional but may be wasteful.

For v1, use trie top-K candidate restriction:

\[
C_t = \operatorname{TopK}(P_{\text{trie}}(\cdot \mid n_t), K)
\]

Then compute logits only for \(v \in C_t\).

Important implementation note:

- if the true target token is not in top-K during training, include it anyway
- this avoids impossible training targets

So practical training candidate set is:

\[
C_t = \operatorname{TopK}(P_{\text{trie}}(\cdot \mid n_t), K) \cup \{y_t\}
\]

where \(y_t = x_{t+1}\) is the true next token.

---

# 6. Trie transition

The trie node updates by the observed token:

\[
n_{t+1} = T_{\text{trie}}(n_t, x_t)
\]

For v1:

- if child exists, follow it
- otherwise fall back to root and try token from root
- if still missing, use root

This fallback can be improved later, but is enough for the first prototype.

---

# 7. Training objective

For each sequence \(x_1, x_2, \dots, x_T\), use teacher forcing.

At each step \(t\), predict \(x_{t+1}\) from \((n_t, s_t)\).

Loss:

\[
\mathcal{L}_t = -\log P(x_{t+1} \mid n_t, s_t)
\]

Total loss:

\[
\mathcal{L} = \sum_{t=1}^{T-1} \mathcal{L}_t
\]

Train with ordinary backprop through time over sequence chunks.

For the first version, keep this simple and standard.

---

# 8. State initialization

At start of a sequence/chunk:

- trie node starts at root
- carried state starts as zeros or a learned initial vector

Recommended initial version:

\[
s_0 = 0
\]

---

# 9. Inference behavior

At inference, the system only carries:

- current trie node
- current state vector

No replay of old tokens is needed once state is established.

Inference loop:

1. compute trie next-token distribution from current node
2. get top-K candidates
3. compute residual scores from current state
4. combine into logits
5. sample or argmax next token
6. update trie node with chosen token
7. update carried state with chosen token embedding
8. repeat

This is the main architectural point of the experiment.

---

# 10. Parameters

Minimum learnable parameters:

## Embeddings
- token embedding matrix \(E \in \mathbb{R}^{|V| \times d}\)

## Recurrent update
- \(W_g, U_g, W_s, W_x \in \mathbb{R}^{d \times d}\)
- \(b_g, b_s \in \mathbb{R}^d\)

## Output bias
- \(b \in \mathbb{R}^{|V|}\)

Optional:
- make initial state learned
- separate output embeddings from input embeddings

For v1, tied input/output embeddings are fine.

---

# 11. Recommended default choices

For a small first experiment:

- embedding/state dimension: `d = 64` or `128`
- trie max depth: same as chunk length or a modest fixed value like `16`, `32`, or `64`
- trie smoothing epsilon: `1e-6` or `1e-5`
- top-K candidates: `K = 32` or `64`
- alpha: start with `1.0`
- beta: start with `1.0`

Possible ablations later:

- alpha-only (pure trie)
- beta-only (pure learned recurrent model over restricted vocab)
- hybrid alpha+beta

---

# 12. Data pipeline

## Step 1: tokenize corpus
Convert corpus to token IDs.

## Step 2: build trie
Build a prefix trie over token sequences up to depth \(L\).

Each node should record:

- child token -> child node
- next-token counts

Important:
- the trie represents local corpus continuation memory
- it does **not** need to store KV or activations

## Step 3: create training chunks
Split tokenized corpus into chunks for recurrent training.

For each chunk:
- reset root/state, or
- optionally continue state across chunk boundaries later

For v1, resetting is fine.

---

# 13. Pseudocode

## 13.1 Trie node interface

Suggested minimal trie API:

```python
class TrieNode:
    children: dict[int, TrieNode]
    next_counts: dict[int, int]
```

Suggested methods:

- `transition(token_id) -> TrieNode | None`
- `next_probs(epsilon) -> dict[token_id, float]`
- `topk_next(k, epsilon) -> list[(token_id, prob)]`

---

## 13.2 Model forward step

```python
def step(node, state, input_token, target_token):
    # 1. trie distribution
    trie_probs = node.next_probs(epsilon)

    # 2. candidate set
    candidates = topk_from_probs(trie_probs, K)
    if target_token not in candidates:
        candidates.append(target_token)

    # 3. residual scores
    logits = []
    for v in candidates:
        trie_logp = math.log(max(trie_probs.get(v, epsilon), epsilon))
        residual = dot(state, E[v]) + out_bias[v]
        logit = alpha * trie_logp + beta * residual
        logits.append(logit)

    probs = softmax(logits)
    loss = -log_prob_of_target(probs, candidates, target_token)

    # 4. update carried state with current input token
    emb = E[input_token]
    gate = sigmoid(Wg @ state + Ug @ emb + bg)
    proposal = tanh(Ws @ state + Wx @ emb + bs)
    next_state = gate * state + (1.0 - gate) * proposal

    # 5. update trie node with current input token
    next_node = node.transition(input_token)
    if next_node is None:
        next_node = root.transition(input_token)
        if next_node is None:
            next_node = root

    return next_node, next_state, loss
```

---

## 13.3 Training loop

```python
state = zeros(d)
node = root
total_loss = 0.0

for t in range(len(tokens) - 1):
    input_token = tokens[t]
    target_token = tokens[t + 1]

    node, state, loss = step(node, state, input_token, target_token)
    total_loss += loss

total_loss.backward()
optimizer.step()
optimizer.zero_grad()
```

In practice this should be batched and chunked, but keep the first implementation simple.

---

# 14. Recommended implementation plan

## Phase 1: trie only baseline
Implement pure trie next-token prediction.

At each node:
- use smoothed trie probabilities directly
- no learned state

Measure:
- next-token accuracy
- perplexity
- failure modes

This gives the baseline memory-only model.

---

## Phase 2: learned recurrent baseline without trie
Use the same carried state cell, but remove trie term:

\[
\ell(v) = \beta\, r(v \mid s_t)
\]

Measure same metrics.

This gives the pure learned-state baseline.

---

## Phase 3: hybrid model
Use the full hybrid:

\[
\ell(v) = \alpha \log P_{\text{trie}}(v \mid n_t) + \beta\, r(v \mid s_t)
\]

This is the actual AGPT prototype.

Compare:
- convergence speed
- perplexity
- next-token accuracy
- parameter efficiency

---

# 15. What to log

Please instrument the experiment well.

Recommended metrics:

- training loss
- validation loss
- perplexity
- next-token accuracy
- top-K target inclusion rate
- average trie branch size
- average candidate set size
- average fallback rate
- comparison of pure trie / pure learned / hybrid

Optional:
- norm of carried state over time
- gate statistics
- how often residual overrides trie top choice

---

# 16. Important conceptual note

This prototype is **not** a transformer and should not be forced into transformer assumptions.

It is closer to:

- a recurrent model
- plus exact local corpus memory
- plus shared prefix structure via trie

That is intentional.

The main thing being tested is:

> whether a carried continuous state plus trie-local empirical memory is enough to produce a useful predictive system without full context replay or transformer-style KV cache

---

# 17. Expected strengths

This prototype may show strengths in:

- parameter efficiency
- local empirical grounding
- cheap inference state
- fast candidate narrowing
- clear division of labor between memory and generalization

---

# 18. Expected weaknesses

This prototype will likely be weak at:

- long-range exact order-sensitive dependencies beyond what the state can carry
- large unseen recombinations
- complex reasoning
- very sparse trie branches if corpus is small/noisy

That is acceptable for the first experiment.

---

# 19. Optional future upgrades

Do **not** build these in first, but keep them in mind:

- root fallback -> suffix fallback
- multiple carried states
- separate semantic/global and ordered/local state channels
- learned trust gate between trie and residual
- fallback candidate proposer from learned model
- truncated BPTT
- online/local learning rules
- richer node metadata
- state caching at shared trie nodes

---

# 20. Deliverables requested

Please implement:

1. corpus tokenizer
2. prefix trie builder with next-token counts
3. pure trie baseline
4. pure learned recurrent baseline
5. hybrid trie + carried-state model
6. simple training/eval loop
7. comparison metrics and basic reporting

Preferred first target:
- small corpus
- small vocab
- quick turnaround
- correctness and clarity over optimization

---

# 21. Final summary

The prototype state is:

\[
(n_t, s_t)
\]

Prediction is:

\[
\ell(v \mid n_t, s_t)
=
\alpha \log P_{\text{trie}}(v \mid n_t)
+
\beta \big(s_t^\top e(v) + b_v\big)
\]

State update is:

\[
g_t = \sigma(W_g s_t + U_g e(x_t) + b_g)
\]

\[
\tilde{s}_{t+1} = \tanh(W_s s_t + W_x e(x_t) + b_s)
\]

\[
s_{t+1} = g_t \odot s_t + (1 - g_t)\odot \tilde{s}_{t+1}
\]

Trie update is:

\[
n_{t+1} = T_{\text{trie}}(n_t, x_t)
\]

Train with ordinary next-token loss.

That is the full minimal AGPT prototype.
