# Idea note: subtree-structured attention (future work)

**Origin:** exploratory discussion, 2026-04-19.
**Status:** hypothesis, not validated; noted for possible follow-up.

---

## The question that kicked it off

"Instead of nodes processed by a transformer, what if the edges of the
prefix tree became the attention layer?"

The starting intuition: nodes of a prefix trie encode frequency
distributions over the vocab. Those distributions have structure — a
distribution at a node is probably similar to distributions at
structurally nearby nodes. The raw trie is expensive to store; the
"compressed" form of that structure is some kind of learned model.
Natural question: could the compression mechanism be attention, and
if so, operating on what?

## Two distinct versions of the idea

### Version 1 — per-edge mini-transformers

Each edge (or each node) carries its own small transformer. Compute
is organized as many tiny transformers running at each node rather
than one big transformer over the context window.

**Problems identified:**

- If weights are per-edge: parameter count scales with number of
  trie edges. Shakespeare d=32 is ~27M edges; billion-token BPE is
  ~billions of edges. Infeasible.
- If weights are shared across edges: this is functionally a
  tree-structured neural network, which is the
  Tree-LSTM / Graph-Neural-Network (GNN) pattern — a studied
  subfield, not a new primitive.
- Hardware: many small matmuls is poor GPU utilization (GPUs love
  huge GEMMs, hate small ones). Tree-aware accelerators exist
  (Graphcore's IPU was partly motivated by this) but relying on new
  silicon is a long bet.

**Verdict:** a 5-10 year hardware-contingent bet. Interesting, but
out of reach as a near-term project.

### Version 2 — subtree-scoped attention (the tractable version)

Constrain attention to operate within a single subtree at a time.
This brings the idea into range of AGPT's existing per-subtree
training infrastructure.

Concretely: today, the transformer's attention at node `the q` sees
its ancestor chain `the `, `the q`. Subtree-structured attention
would additionally let it see **siblings** and **cousins** — other
children of `the ` such as `the man`, `the day`, `the woman` — or
more generally, the structural neighborhood of the current node
within its subtree.

**What this buys (hypothesis):**

- **Explicit branching-context.** Ancestor-chain attention encodes
  "what came before"; sibling attention encodes "what other paths
  diverged from the same point." Current transformers have to
  reconstruct the latter statistically from many windows; a
  subtree-structured attention could get it natively in one pass.
- **Out-of-distribution generalization.** When a user types a novel
  prefix at inference time, the current model routes to a similar
  training prefix implicitly through embedding space. A tree-aware
  attention could explicitly identify "the structurally nearest
  seen prefix" and route predictions accordingly. The claim would
  be that explicit tree-aware routing beats implicit
  embedding-similarity routing on prefixes that diverge from
  training distribution.

**What has to hold for it to work:**

- Generalization has to actually improve, not just recapitulate
  what ancestor-chain attention already does. This is the
  non-trivial empirical question.
- The subtree's "neighborhood" has to be small enough that attention
  remains tractable. In AGPT's per-subtree regime this is already
  naturally bounded.
- The training signal has to reach the new attention pattern, which
  requires the loss to actually benefit from sibling information.

## Testable experiments

If this were picked up as a project:

1. **Held-out PPL on structurally-adjacent prefixes.** Construct a
   held-out set by making single-character edits to training-set
   prefixes (so the novel prefix has a known structural neighbor in
   the training trie). Compare PPL on these to PPL on random novel
   prefixes. If subtree attention helps via the generalization
   argument, it should win disproportionately on the structured
   set.

2. **Ablation: ancestor-only vs. subtree-scoped attention.** Same
   model, same corpus, same training recipe, swap only the
   attention pattern. Any PPL delta is isolated to the mechanism.

3. **Scaling curve comparison.** Train at several depths with both
   attention patterns. If the subtree pattern's advantage grows
   with depth (as structural neighbors become more numerous and
   informative), that would be a positive signal for scaling.

## Where this sits relative to current AGPT work

- Not a replacement for the factorization theorem, which is
  independent.
- Natural extension of the per-subtree training regime — the K/V
  cache is already scoped to a subtree; extending the attention
  pattern to include sibling/cousin positions within that subtree
  is additive work, not a new system.
- Probably 2-4 weeks of careful implementation + a few weeks of
  experiments to produce a defensible result.
- Would likely warrant a follow-on paper or a substantial §N in a
  revised AGPT paper, depending on the outcome.

## Related prior work to check before pursuing

- Tree-LSTM (Tai et al. 2015) — tree-structured recurrent networks.
- Tree Transformer (Nguyen et al. 2020) — hierarchical attention.
- Graph Neural Networks with tree-structured inputs (many).
- AGPT repo's own `agpt-sibling-attention` branch (partial prior
  attempt in the same direction — worth re-reading before starting
  fresh).

The sibling-attention branch in particular is relevant: a version
of this was prototyped earlier in the codebase's history and may
have generated empirical intuition worth recovering.
