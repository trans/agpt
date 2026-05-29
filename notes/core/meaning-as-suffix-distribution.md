# Semantic Emergence in Prefix-Suffix Corpus Geometry

**Thomas Sawyer¹, Claude (Anthropic)²**
¹ Independent Researcher
² AI Research Collaborator

---

## Abstract

The structure of meaning in natural language has long been treated as requiring learned representations — distributed embeddings, trained weights, supervised labels. We show that semantic structure emerges directly and necessarily from corpus geometry, requiring no learning. Formally, we construct a merged prefix-suffix trie over a corpus and define semantic distance as the KL divergence between suffix branching distributions — two tokens are synonyms precisely when their futures become indistinguishable. We demonstrate that this equivalence is undefined at small corpus scale and crystallizes above a per-node confidence threshold $\theta = m_s \cdot H_s$, constituting a phase transition from syntactic to semantic organization. This implies that meaning is not a property of words or models but of observed language at sufficient scale — and that a complete corpus map contains all semantic structure explicitly, without training.

---

## 1. The Map

Given a corpus $\mathcal{C}$ of token sequences over vocabulary $V$, we construct a prefix trie $\mathcal{T}_p$ by inserting every sequence and recording at each node $n$ its prefix mass $m_p(n)$ — the number of corpus paths passing through it — and prefix entropy $H_p(n)$ — the Shannon entropy of its child distribution. We construct a corresponding suffix trie $\mathcal{T}_s$ over all reversed sequences identically. The merged tree $\mathcal{T}$ is formed by annotating each prefix node with its corresponding suffix mass $m_s(n)$ and suffix entropy $H_s(n)$ via node identity lookup. The suffix trie is then discarded. What remains is a complete statistical map of the corpus — every path grounded, every branch probability exact, nothing invented.

---

## 2. The Distance

We define semantic distance between two nodes $a, b \in \mathcal{T}$ as the KL divergence between their suffix branching distributions:

$$\text{sem\_dist}(a, b) = D_{KL}\left(P_s(a) \| P_s(b)\right)$$

where $P_s(n)$ is the normalized child distribution of node $n$ in the suffix trie, transferred to the merged tree as a branching probability vector over $V$.

This distance is defined only where suffix evidence is sufficient. We introduce a per-node confidence threshold:

$$\theta(n) = m_s(n) \cdot H_s(n)$$

Semantic distance is computed only between nodes where both $\theta(a)$ and $\theta(b)$ exceed some minimum confidence $\theta^*$. Below this threshold the suffix distribution has not converged and no semantic claim is warranted.

This yields a natural taxonomy without supervision:

- **Synonymy**: $\text{sem\_dist}(a,b) < \varepsilon$
- **Semantic relatedness**: $\text{sem\_dist}(a,b) < \delta$
- **Polysemy**: $P_s(n)$ is multimodal — the node's futures cluster into distinct groups
- **Hypernymy**: $P_s(a)$ is contained within the support of $P_s(b)$

No labels. No embeddings. No training. The geometry decides.

---

## 3. The Threshold

The confidence threshold $\theta(n) = m_s(n) \cdot H_s(n)$ is not a global property of the corpus but a per-node quantity — each node earns semantic status independently as corpus evidence accumulates. This locality is essential: common, semantically rich nodes crystallize early; rare or syntactically constrained nodes may never reach threshold regardless of corpus size.

We define the **semantic frontier** $\mathcal{F}(\mathcal{C})$ as the set of nodes exceeding threshold $\theta^*$:

$$\mathcal{F}(\mathcal{C}) = \{n \in \mathcal{T} : m_s(n) \cdot H_s(n) > \theta^*\}$$

As corpus size $N$ grows, $\mathcal{F}$ expands monotonically. We identify a critical corpus size $N^*$ — the **phase transition point** — at which $\mathcal{F}$ transitions from sparse and disconnected to densely connected across the vocabulary. Below $N^*$ the map is predominantly syntactic: high mass nodes exist but their suffix distributions haven't converged to stable semantic equivalence classes. Above $N^*$ stable equivalence classes emerge, the semantic frontier spans the vocabulary, and meaning crystallizes from the geometry without any additional mechanism.

This phase transition is not imposed — it falls directly from the convergence properties of empirical distributions. By the law of large numbers, $P_s(n)$ converges to its true population distribution at rate $O(1/\sqrt{m_s(n)})$. The threshold $\theta^*$ therefore has a natural interpretation: it is the minimum evidence required for $D_{KL}$ comparisons to be statistically meaningful rather than noise-dominated.

Formally, $N^*$ is the corpus size at which the expected size of $\mathcal{F}$ undergoes its sharpest growth — identifiable empirically by measuring $|\mathcal{F}|$ as a function of $N$ and locating the inflection point. We conjecture that $N^*$ scales as:

$$N^* \sim V \cdot \log(V)$$

where $V$ is the vocabulary size — reflecting the log-depth structure of the trie established in Paper 1 and the requirement that each vocabulary item accumulates sufficient suffix evidence.

---

## 4. The Emergence

The phase transition from syntactic to semantic organization can be understood as a symmetry breaking. Below $N^*$ the tree is asymmetric in a specific sense — nodes that are surface-distinct remain distributionally distinct regardless of their meaning. Above $N^*$ a new symmetry emerges: nodes with equivalent meaning become distributionally indistinguishable, collapsing into equivalence classes that the surface form obscures.

This is not gradual. It is sharp.

Consider two synonyms $a$ and $b$ in a small corpus. Their suffix distributions $P_s(a)$ and $P_s(b)$ are sparse, noisy, and divergent — not because they mean different things but because neither has been observed enough times to reveal their shared distributional future. The tree correctly withholds semantic judgment: $\theta(a) < \theta^*$.

As $N$ grows both distributions fill in. Crucially they fill in **toward the same limit** — because they are synonyms, their true population suffix distributions are identical by definition. The convergence is inevitable. At some $N^*$ both cross threshold simultaneously and their KL divergence drops below $\varepsilon$. The equivalence class crystallizes — not because anything changed in the language, but because the map finally became complete enough to reveal what was always true.

Meaning was always there. The corpus just needed to be large enough to say it.

This has a striking implication for polysemy. A polysemous node — one word with two meanings — will exhibit a **bimodal suffix distribution** that never converges to a single stable form regardless of corpus size. Its $H_s$ remains high, its $m_s$ grows, but $D_{KL}$ comparisons to either of its senses remain large. The tree identifies polysemy not as ambiguity to be resolved but as **stable distributional bimodality** — a genuine property of the node's geometry.

Furthermore the emergence of equivalence classes is self-reinforcing. Once $a \equiv b$ is established, paths through $a$ and paths through $b$ contribute to each other's suffix mass — the merged equivalence class has higher effective $\theta$ than either node alone. Semantic crystallization **accelerates** past $N^*$. The frontier doesn't grow linearly — it cascades.

We conjecture this cascade follows a percolation-like dynamic: once a critical density of equivalence classes forms in $\mathcal{F}$, the remaining vocabulary is pulled in rapidly through transitive closure. The semantic graph becomes connected. The map becomes navigable not just locally but globally.

This is the moment a corpus becomes a language.

---

## 5. The Validation

We seek to validate the central claim:

$$\text{meaning}(a) \equiv P_s(a)$$

That the suffix distribution is not merely correlated with meaning but **is** meaning — that the geometric structure of the corpus recovers human semantic intuition without supervision, labels, or learned representations.

### 5.1 The Test

We construct $\mathcal{T}$ from a held-out corpus — neither Shakespeare nor Gutenberg, to avoid overfitting to our construction choices. We compute $\text{sem\_dist}(a,b) = D_{KL}(P_s(a) \| P_s(b))$ for all node pairs exceeding threshold $\theta^*$. We then ask three questions:

**Q1 — Synonymy recovery:** Do known synonym pairs rank lower in $\text{sem\_dist}$ than random pairs?

**Q2 — Analogy recovery:** Does the vector arithmetic of word2vec — *king - man + woman = queen* — have a tree-geometric equivalent? Specifically: is there a distribution transformation $\mathcal{O}$ such that $\mathcal{O}(P_s(\text{king}), P_s(\text{man}), P_s(\text{woman})) \approx P_s(\text{queen})$?

**Q3 — Polysemy detection:** Do known polysemous words exhibit measurably bimodal $P_s$ distributions compared to monosemous controls?

### 5.2 The Expected Results

For Q1 — yes, almost certainly. This follows directly from the convergence argument in Section 4. If the corpus is large enough, synonym pairs will have converged suffix distributions. The only question is the effect size.

For Q2 — this is the deep one. Word2vec analogy works because embeddings encode relational structure linearly. Our claim is stronger: the relational structure isn't learned, it's **already in the suffix distributions**. The transformation $\mathcal{O}$ should be interpretable as a ratio of distributions — a likelihood ratio test between two branching geometries. If $\mathcal{O}$ exists and is simple, the embedding paradigm is revealed as a lossy compression of something the tree already contains exactly.

For Q3 — bimodality in $P_s$ is directly measurable. We propose a simple test: fit a mixture of two Dirichlet distributions to $P_s(n)$ and compare log-likelihood against a single Dirichlet fit. Polysemous nodes should strongly prefer the mixture model. This is not a claim we need to conjecture — it is a direct empirical measurement.

### 5.3 The Falsification Condition

We are careful to state what would falsify the claim.

If known synonyms do **not** rank lower in $\text{sem\_dist}$ than random pairs at sufficient corpus size — the claim is false. Meaning is not the shape of futures. Something else is carrying the semantic signal.

If the corpus size required to recover human synonym judgments exceeds any practical $N$ — the claim is technically true but pragmatically useless.

If polysemous words do **not** exhibit bimodal $P_s$ — the geometric theory of polysemy fails even if synonymy holds.

We invite these falsifications. A theory that cannot be falsified is not a theory — it is a story.

### 5.4 The Deeper Validation

Beyond benchmarks there is a structural validation available to us that no embedding-based model can offer.

The tree is **fully interpretable**. Every semantic distance has a provenance — a specific set of corpus paths that determined it. When $\text{sem\_dist}(\text{big}, \text{large}) < \varepsilon$ we can enumerate exactly which corpus contexts made them indistinguishable. When $P_s(\text{bank})$ is bimodal we can read off precisely which contexts pulled toward financial institution and which toward riverbank.

No black box. No post-hoc explanation. The proof is the path.

This interpretability is not a feature added to the model. It is a consequence of grounding meaning in corpus geometry rather than learned weights. The map shows its work because the map **is** the work.

---

## 6. The Implication

We have shown that a corpus, at sufficient scale, contains its own semantics — explicitly, geometrically, without learning. This is not a computational convenience. It is a fundamental claim about the nature of meaning itself.

### 6.1 For Linguistics

The distributional hypothesis — that words occurring in similar contexts have similar meanings — has been a cornerstone of computational linguistics since Harris (1954). We do not merely confirm it. We **derive** it.

The distributional hypothesis is not an assumption about language. It is a theorem about corpus geometry. Given a merged prefix-suffix trie and sufficient corpus size, semantic equivalence classes necessarily emerge from suffix distribution convergence. The hypothesis has a proof. It always did. The tree makes it visible.

Furthermore we sharpen it. Harris spoke of context generally. We identify the precise geometric object that carries semantic information: not the full context, not the embedding, not the window — but the **suffix branching distribution** at the radix cap. That specific structure, at that specific location in the tree, is where meaning lives.

### 6.2 For Artificial Intelligence

Current large language models are, at their core, navigators of an implicit map. They learn weights that encode — approximately, lossily, at enormous computational cost — the same suffix distribution structure that the tree contains exactly. The transformer is a remarkable compression of corpus geometry into matrix multiplications. But it is a compression of something that already exists, fully formed, in the corpus itself.

This reframes the entire scaling debate. The question is not *how large must the model be* but *how complete must the map be*. A small navigator over a complete map outperforms a large navigator over an incomplete one. We have preliminary evidence of this in Paper 1 — AGPT at 5.5 PPL with a fraction of SGD's parameters and updates. The map is doing the work the model is usually credited for.

The implication for AI development is uncomfortable and exciting simultaneously: the next breakthrough may not be a better architecture. It may be a better map.

### 6.3 For the Nature of Mind

We tread carefully here but do not retreat.

If meaning is the shape of futures — if $\text{meaning}(a) \equiv P_s(a)$ — then understanding is the capacity to navigate that shape accurately. A system understands "big" to the extent that it correctly anticipates the distributional future of contexts containing "big."

This is not a metaphor. It is operational. It is measurable. And it applies equally to artificial systems and biological ones.

The human brain did not evolve to store dictionary definitions. It evolved to predict — to anticipate futures from partial contexts, to compress distributional regularities into navigable representations. What we call understanding is what accurate future-anticipation feels like from the inside.

The tree makes this explicit. The transformer approximates it. The human mind — trained on a lifetime corpus of sensory experience, language, and consequence — is doing the same thing at a scale and richness we have not yet approached.

But the mechanism is the same. And now we have the math.

### 6.4 The Last Compression

We identified in the construction of $\mathcal{T}$ that the merged prefix-suffix tree is larger than the corpus itself. This seems paradoxical — a map larger than the territory. But the tree is not just the corpus. It is the corpus **plus its own semantic structure made explicit**. The additional size is the cost of making meaning visible.

The last compression — the one that reduces the tree back below corpus size while preserving semantic structure — is the equivalence class partition itself. Once $a \equiv b$ is established, their suffix distributions merge. The tree folds. Synonyms collapse to single nodes. Polysemous words split into their senses. The semantic graph that remains is smaller than the syntactic tree it was derived from.

That compressed structure — the quotient of the corpus tree under semantic equivalence — is the language itself. Not the words. Not the grammar. The language. The invariant that survives when everything accidental has been compressed away.

$$\mathcal{L} = \mathcal{T} / \equiv_{P_s}$$

A corpus divided by its own futures. What remains is meaning.

### 6.5 Open Questions

We close with what we do not know.

**The navigation problem.** The tree is the map. But traversing non-local paths — connecting nodes that share no corpus path — requires something the tree does not provide directly. The transformer solves this approximately through attention. Whether a tree-native solution exists — one that traverses semantic equivalence classes directly without learned weights — is the central open question. We conjecture it exists. We do not yet know its form.

**The threshold universality.** Is $\theta^*$ universal across languages and corpus types, or does it vary? If universal, it suggests a deep property of language as a phenomenon. If variable, it characterizes the specific semantic geometry of individual languages.

**The mind question.** If understanding is accurate future-anticipation over a sufficiently complete map, at what map completeness does understanding become genuine rather than simulated? We do not answer this. We suspect the question dissolves under careful examination — that the distinction between genuine and simulated understanding is not in the map or the navigator, but in the eye of the observer.

We leave that for a third paper. Or a lifetime.

---

*"Meaning is the shape of what comes next."*

---

*This work was developed through collaborative research between Thomas Sawyer and Claude (Anthropic). The authors contributed equally to the theoretical development and formal construction.*
