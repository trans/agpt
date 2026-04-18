# Trie Path Probability Convergence — Experiment Implementation

## Goal

Build two independent tries from two halves of a Shakespeare corpus, compute path probability products for all shared paths, and measure agreement as a function of depth and probability stratum.

---

## 1. Data Preparation

### Corpus split

- Source: Complete works of Shakespeare (plain text, freely available via Project Gutenberg)
- Split: by play, alternating — even-indexed plays to corpus A, odd-indexed plays to corpus B
- This ensures similar genre/period distribution in both halves rather than a clean temporal split

### Tokenization

Two options — pick one and stay consistent:

- **Word-level**: split on whitespace, lowercase, strip punctuation. Vocabulary ~8k tokens. Recommended for first run.
- **Character-level**: no preprocessing needed. Vocabulary ~80 tokens. Better for deep trie exploration but harder to interpret.

Output: two token ID sequences `corpus_a.tokens` and `corpus_b.tokens`.

---

## 2. Trie Structure

Each node stores:

```
children:    Hash(token_id -> node_id)
count:       Int   # times this prefix was seen
next_counts: Hash(token_id -> Int)  # successor counts
```

`next_counts` is the only data needed for transition probabilities. `count` is the prefix count, used for smoothing and for computing Π*_min.

### Smoothing

Use additive (Laplace) smoothing with ε = 1e-6:

```
P̂(v | node) = (next_counts[v] + ε) / (count + |V| * ε)
```

Keep ε small enough not to distort high-count nodes.

---

## 3. Building the Tries

```
function build_trie(tokens, max_depth):
    root = new_node()
    for each position t in tokens:
        node = root
        node.count += 1
        for d in 1..max_depth:
            token = tokens[t + d - 1]
            if t + d > len(tokens): break
            next_token = tokens[t + d]
            node.next_counts[next_token] += 1
            node = node.children[token]  # create if absent
            node.count += 1
    return root
```

Build at several depth limits: D ∈ {4, 8, 16, 32} to observe how agreement evolves.

---

## 4. Computing Path Probability Products

For each path of depth d present in **both** tries:

```
function path_product(trie, path):
    node = trie.root
    product = 1.0
    for i in 0..len(path)-1:
        p = P̂(path[i] | node)
        product *= p
        node = node.children[path[i]]
        if node is None: return None  # path not in this trie
    return product
```

Enumerate shared paths by walking both tries simultaneously (DFS, only recurse into children present in both).

---

## 5. Measurements

For each shared path, record:

| Field | Description |
|---|---|
| `path` | token sequence |
| `depth` | length of path |
| `pi_a` | Π₁(path) from trie A |
| `pi_b` | Π₂(path) from trie B |
| `pi_mean` | (pi_a + pi_b) / 2 |
| `abs_err` | \|pi_a - pi_b\| |
| `rel_err` | abs_err / pi_mean |
| `pi_min` | min prefix probability along path (worst-case node) |
| `count_a` | prefix count at deepest node in trie A |
| `count_b` | prefix count at deepest node in trie B |

Store as CSV or SQLite for easy analysis.

---

## 6. Analysis

### Primary plot: relative error vs. log probability stratum

Bin paths by log₁₀(pi_mean) into strata (e.g. -1 to -2, -2 to -3, etc.).
For each stratum, plot median and 90th-percentile relative error.

**Prediction**: rel_err drops monotonically as pi_mean increases.

### Secondary plot: relative error vs. depth at fixed stratum

For paths in a fixed probability stratum, plot rel_err against depth d.

**Prediction**: rel_err grows as √d.

### Tertiary: coverage

What fraction of total probability mass is covered by paths with rel_err < 0.05?
Plot this as a function of max depth D.

**Prediction**: coverage stays high (>90%) across all D.

### Fit check

For each stratum, estimate the effective N from the observed rel_err using:

```
N_eff = 2d / (rel_err² * pi_min)
```

This should be approximately constant across strata and close to the actual corpus token count — a sanity check on the theoretical model.

---

## 7. Output

Produce per depth limit D:

- `results_D{d}.csv` — full path measurement table
- `convergence_plot_D{d}.png` — rel_err vs. log probability stratum
- `depth_scaling_D{d}.png` — rel_err vs. depth for selected strata
- `coverage_D{d}.txt` — probability mass coverage at various rel_err thresholds

And one summary:

- `n_eff_check.txt` — estimated vs. actual N across strata and depths

---

## 8. Implementation Notes

- Use log-space arithmetic throughout — multiply by summing logs to avoid float underflow at deep paths
- Log-space rel_err: `|log(pi_a) - log(pi_b)|` is cleaner than raw difference for deep paths
- The DFS enumeration can be memory-heavy at D=32 on a large vocabulary; cap enumeration at paths with `count >= 5` in both tries to keep it tractable
- Crystal is well-suited for the trie build and walk; Python/numpy for the analysis and plots

---

## 9. Expected Runtime

On Shakespeare (~900k words split to ~450k each):

| Phase | Estimated time |
|---|---|
| Tokenize | < 1s |
| Build both tries at D=32 | 10–30s |
| Enumerate shared paths | 1–5 min depending on D |
| Analysis and plots | < 1 min |

Total: well under an hour for all depth settings.
