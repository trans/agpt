# AGPT Side Experiment: Compressing Node→Vocab Distributions

## Goal

Test whether the empirical next-token distributions stored at trie nodes are compressible.

This is **not** an AGPT training change yet. It is an **offline analysis experiment**.

We want to know whether the node→vocab matrix has low intrinsic dimension.

---

## Data to Export

Construct a matrix:

\[
P \in \mathbb{R}^{N \times V}
\]

Where:

- \(N\) = number of trie nodes included in the experiment
- \(V\) = vocab size
- row \(P_i\) = next-token distribution (or counts) for node \(i\)

### Recommended export fields per node

For each node:

- `node_id`
- `depth`
- `total_count`
- `counts[0..V-1]` or normalized `probs[0..V-1]`

### Prefer normalized probabilities

For node \(i\):

\[
p_i(x) = \frac{n_i(x)}{\sum_y n_i(y)}
\]

Store one row per node as a length-\(V\) float vector.

---

## Scope

Start with the current small setup:

- vocab size \(V = 65\)
- current trie subset / current run configuration
- if full node export is easy, export all nodes
- otherwise export a representative sample (e.g. 50k–250k nodes)

Expected size is manageable.

Example:
- 250k nodes × 65 floats ≈ 16.25M floats
- float32 ≈ 65 MB

---

## Experiment 1: Truncated SVD

### Purpose

Measure whether the node→vocab matrix is approximately low-rank.

### Procedure

Given matrix \(P\):

1. center rows optionally (or leave as probabilities)
2. compute truncated SVD at ranks:

```text
r = 2, 4, 8, 16, 32
```

3. reconstruct:

\[
P_r = U_r \Sigma_r V_r^T
\]

4. measure reconstruction quality

### Metrics

For each rank \(r\), report:

- Frobenius relative error:

\[
\frac{\|P - P_r\|_F}{\|P\|_F}
\]

- mean row-wise KL divergence (if rows are normalized as probabilities)
- mean row-wise cross-entropy
- top-k overlap (k = 1, 3, 5) between original and reconstructed rows
- explained variance ratio

### Interpretation

If low ranks (e.g. 8–16) already reconstruct well, then the distribution field is highly compressible.

---

## Experiment 2: Tiny Autoencoder

### Purpose

Test nonlinear compression of the node distributions.

### Input / Output

- input: row \(p_i \in \mathbb{R}^V\)
- bottleneck: small latent dimension
- output: reconstructed row \(\hat{p}_i \in \mathbb{R}^V\)

### Suggested architectures

For \(V=65\), try small models only.

#### Model A
```text
65 → 32 → 8 → 32 → 65
```

#### Model B
```text
65 → 16 → 4 → 16 → 65
```

#### Model C
```text
65 → 32 → 16 → 32 → 65
```

### Output handling

Two options:

#### Option 1 (simplest)
- decoder outputs raw values
- apply softmax to produce probability distribution

#### Option 2
- decoder outputs logits directly
- train with cross-entropy / KL against true distribution

### Loss

Preferred:

\[
L = \mathrm{KL}(p_i \,\|\, \hat{p}_i)
\]

or row-wise cross-entropy.

Secondary option:
- MSE on probabilities

### Train/validation split

Use:
- 90% train
- 10% validation

Shuffle nodes randomly.

### Metrics

Report:

- validation KL divergence
- validation cross-entropy
- top-k overlap
- entropy error:

\[
|H(p_i) - H(\hat{p}_i)|
\]

---

## Experiment 3: Clustering Baseline (Optional)

### Purpose

See whether simple clustering already compresses distributions.

### Procedure

- run k-means on node probability rows
- use cluster centroids as compressed prototypes

Try:
```text
k = 8, 16, 32, 64
```

### Compare against SVD / autoencoder

This gives a very useful baseline:
- if clustering works well, distributions fall into a few common shapes
- if autoencoder beats clustering strongly, there is richer nonlinear structure

---

## Evaluation Questions

We want to answer:

1. Is the node→vocab matrix low-rank?
2. Can a tiny bottleneck reconstruct distributions well?
3. How many latent dimensions are needed?
4. Are the distributions closer to:
   - low-rank structure
   - clustered prototypes
   - or essentially full-rank / hard to compress?

---

## Important Notes

### This experiment does NOT change AGPT yet

Do not integrate into training for now.

This is only to test whether compression is real.

### Keep exact trie behavior unchanged

The current trie still stores exact counts/distributions.

### Use this experiment to decide whether a second compression layer is worth pursuing

---

## Possible Outcomes

### Outcome A: strong compression
Example:
- rank 8–16 works well
- tiny autoencoder reconstructs accurately

Interpretation:
- node distributions lie on a low-dimensional manifold
- promising for future memory compression

### Outcome B: moderate compression
Example:
- some gain, but only at higher rank / larger latent

Interpretation:
- useful for approximate statistics
- maybe not good enough for exact replacement

### Outcome C: weak compression
Example:
- poor reconstruction unless latent is large

Interpretation:
- distributions are too context-specific
- not worth pursuing further for now

---

## Recommended Deliverables

Please produce:

### 1. Export tool
- writes node rows to a simple format (`.npy`, `.csv`, or binary matrix + metadata)

### 2. SVD analysis script
- computes reconstruction metrics for ranks 2/4/8/16/32

### 3. Tiny autoencoder script
- trains 2–3 small bottleneck models
- reports validation metrics

### 4. Summary report
Include:
- matrix dimensions
- reconstruction metrics table
- brief interpretation

---

## Suggested Output Table

| Method | Rank / Bottleneck | Rel. Error | KL | CE | Top-1 | Top-3 | Top-5 |
|--------|-------------------|------------|----|----|------|------|------|
| SVD    | 4                 |            |    |    |      |      |      |
| SVD    | 8                 |            |    |    |      |      |      |
| SVD    | 16                |            |    |    |      |      |      |
| AE     | 4                 |            |    |    |      |      |      |
| AE     | 8                 |            |    |    |      |      |      |
| AE     | 16                |            |    |    |      |      |      |
| KMeans | 16                |            |    |    |      |      |      |

---

## One-Line Summary

> Export the trie node→vocab distribution matrix and test whether it admits strong low-rank or tiny-autoencoder compression before attempting any integration into AGPT.
