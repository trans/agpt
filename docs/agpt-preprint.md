# AGPT: Branching-Aware Training for Character-Level Language Models

**Author:** Thomas Sawyer &nbsp; `transfire@gmail.com`
**Status:** Technical preprint, v1.2 &nbsp; · &nbsp; **Date:** 2026-04-19
**Code:** https://github.com/trans/microgpt

---

## Abstract

Standard next-token language-model training draws random windows from the
corpus and presents them to the optimizer as independent examples. Each
character of the corpus contributes many near-identical gradients —
once per window it appears in — and the optimizer's job is largely to
average over these redundancies.

We propose **AGPT** (Aggregated-Gradient Pretraining): train instead on
a character-level prefix trie of the corpus. Every unique prefix appears
exactly once; each training example is a branching endpoint, weighted
by its corpus frequency. The trie's structure becomes the training
schedule, and a single Adam step per root-child subtree takes the place
of many redundant window-based steps.

On a 4.6 MB Shakespeare corpus with a 108k-parameter transformer, AGPT
trained on a radix-compressed trie at depth 32 reaches held-out
perplexity **13.17** in **195 Adam steps**. Standard window-based
training on the same model/data, step-sweep-tuned to saturation at
matched context (seq_len=32), reaches PPL **14.51** in 2000 SGD steps —
a **9% lower PPL with 10× fewer optimizer updates** at matched depth.

The advantage is specifically per-step gradient efficiency at matched
depth. At longer context, standard window training reaches lower PPL
than AGPT d=32 (seq=1024, 10k steps: PPL 6.30) — because the 108k-param
model benefits substantially from more context, and extending AGPT's
depth costs memory in a way that extending window context does not.
The infrastructure to push AGPT past d=32 (per-subtree training +
bigram partitioning + count-based pruning) is the main engineering
contribution of this work.

The contributions are: (1) a formulation of trie training with a
subtree-scoped Adam step as the factorization-aware training unit;
(2) a radix-compressed + per-subtree file format that makes depth-32
training fit in a 15 GB-RAM consumer workstation; (3) a bigram subtree
partition that further reduces peak memory 6×, unlocking deeper d;
(4) an auto-LR scaler that lets a single base-LR hyperparameter hold
across subtree granularities. All results are reproducible from the
referenced code.

## 1. Motivation

Consider a character-level corpus of N tokens. Window training with
sequence length L samples many windows that share long prefix substrings.
For each sampled window, the optimizer computes a gradient at the
next-token position and averages it with the rest of the batch.

For a fixed prefix `p` that occurs `k` times in the corpus, the
expected gradient contribution over a full epoch of uniformly-sampled
windows is `k / N` times the per-occurrence gradient — linear in corpus
frequency. The optimizer is effectively being asked to reconstruct a
count-weighted sum by stochastic sampling.

**The trie makes that sum explicit.** Every unique prefix is a single
node; the count of next-tokens at that node is exactly `k`. Computing
the gradient *once* at that node, weighted by `k`, produces the same
expected contribution as sampling all `k` windows — but without the
sampling variance, and with one forward/backward pass instead of `k`.

For a corpus with heavy long-range redundancy (Shakespeare has many
repeated stock phrases; BPE-tokenized web text has many repeated
common sequences; code corpora have repeated boilerplate) the
compute-and-variance savings compound. AGPT computes the gradient at
each unique prefix-continuation pair once, then lets the optimizer take
one step against the count-weighted sum over all pairs in a related
subtree.

This is adjacent in spirit to *importance-weighted* and
*deduplication-aware* training — rather than letting the optimizer
rediscover the corpus's redundancy statistically, give it the
redundancy-free structure directly.

## 2. The AGPT training unit

### 2.1 Trie as training schedule

Build a character-level prefix trie of the corpus to depth d. Each
internal node N stores:

- The token emitted at N
- The count distribution over next-tokens observed at N
- Children (one per distinct next-token actually seen)

A **branching endpoint** is a node with ≥ 2 distinct next-tokens in its
count distribution. Non-branching (unary) positions carry no training
signal — the next token is deterministic given the prefix — so they
can be collapsed.

### 2.2 Radix compression

Collapse every unary chain into a single radix node with a
multi-character edge. The node's metadata becomes:

```
radix_id, parent_radix_id, first_char_depth,
edge_len L, edge_tokens[L],
edge_mass (prefix count at head of edge),
entries[] (next-token counts at endpoint)
```

Radix compression is lossless: the expanded character sequence is
identical to the original trie. At d=32 on our corpus the radix trie
has 1.67M nodes representing a 27M-node uncompressed trie — a 16.2×
reduction in node count.

### 2.3 Subtree-scoped Adam steps

The **AGPT training unit** is a root-child subtree. A root-child is a
depth-1 radix node (its edge starts at character position 0); its
subtree is itself plus all descendants. Each subtree is trained in
isolation:

1. Forward pass: compute Q/K/V, RoPE-position, attention, FFN, and
   next-token logits at every branching endpoint in the subtree.
2. Loss: cross-entropy weighted by endpoint count (i.e., prefix
   frequency in the corpus).
3. Backward pass: accumulate gradients across all chunks of the
   subtree. Model parameters stay fixed throughout.
4. **One Adam/RMSProp step** per subtree — applied *after* all
   gradients have been accumulated.

The subtree-scoped step is not an engineering convenience — it is what
makes the factorization valid. Within the subtree, every endpoint's
gradient is computed against the *same* weight snapshot. Accumulating
them and taking one step produces the count-weighted gradient for the
whole subtree, which is the gradient that a standard-training full-data
pass would have accumulated (up to minibatch noise).

If the optimizer fires partway through the subtree (e.g., between
chunks), later endpoints in the same subtree see *different* weights
than earlier ones — the K/V cache used for attention at depth `d` was
computed with an older weight snapshot than the query at depth `d+1`.
This is the "K/V staleness" problem: the subtree is no longer
factorable into one gradient; it's a mid-optimization inconsistency.
Prior iterations of this codebase tried finer-grained updates and
measured convergence degradation; the one-step-per-subtree rule is
what the paper-bracketed recipe restored.

## 3. Memory scaling

### 3.1 The KV cache is the bottleneck

A transformer's K/V cache must cover every character position attended
to by any query in the subtree. The global cache size is:

```
KV_bytes = total_edge_chars × d_model × 2 (K+V) × n_layers × bytes
```

At Shakespeare d=32: 27M × 64 × 4 × 2 × 4 = 27.6 GB. Doesn't fit on
consumer hardware (15 GB RAM + 15 GB swap). Deeper d scales linearly
in `total_edge_chars`.

Radix compression reduces *node* count by ~16× but does not reduce
`total_edge_chars` — each character in an edge still consumes a KV
slot.

### 3.2 Per-subtree training

The insight: at training time, attention at a query position only
reaches characters that are *ancestors of that position*. Different
root-child subtrees share no character positions — the 'A' tree and the
'B' tree never look at each other's character positions.

Therefore: allocate the KV cache **sized to one subtree's edge
characters**, one subtree at a time. On Shakespeare d=32, peak
per-subtree KV drops from 27.6 GB to **4.0 GB** (6.9× reduction).

### 3.3 Bigram subtree partition

The unigram per-subtree scheme bottlenecks on the largest
root-child — in Shakespeare that is the space character (' '),
responsible for ~30% of corpus occurrences. At d=48+ the space subtree
alone would exceed the available RAM budget.

Bigram partitioning splits each root-child subtree by the second
character of the path. For vocab V, there are up to V² subtrees.
At Shakespeare d=32: 1,465 bigram subtrees (vs. 65 unigram),
peak per-subtree KV 677 MB (vs. 4 GB unigram, 6× further reduction).

Because each bigram subtree is self-contained (contains its own
ancestor chain), no cross-subtree coordination is needed at training
time — trivially parallelizable across a cluster.

### 3.4 Auto-LR scaling

A subtler consequence of increasing subtree count: each super-epoch
now fires more Adam steps. With 1465 bigram subtrees instead of 65
unigram, naive lr=3e-3 overshoots in a single super-epoch. Empirically
the product `lr × steps_per_super_epoch` is approximately invariant at
a fixed depth. We therefore scale:

```
lr_effective = base_lr × (REFERENCE_STEPS / steps_per_super_epoch)
```

with REFERENCE_STEPS = 65 (the unigram-d=16 calibration point). This
lets a single base_lr hold across unigram, bigram, and future
finer-grained partitions without re-tuning per config.

## 4. Empirical results

### 4.1 Setup

- Corpus: Shakespeare `input.txt`, 4.62 MB, 65-character vocabulary.
- Model: 2-layer transformer, d_model=64, 4 heads × 16 head-dim,
  d_ff=256, seq_len=128. 108,481 parameters.
- Hardware: single 8 GB GPU + 15 GB system RAM, 15 GB swap.
- Eval: held-out corpus tail, 32,768 positions scored with sliding
  seq_len=128 window. Backend: OpenBLAS (bit-identical to CUDA in
  current build).

### 4.2 Comparison vs. standard window training

All numbers from the same codebase; same model architecture; same
corpus. For each window-training seq_len we swept step counts
{1k, 2k, 3k, 5k, 10k} and report the saturated best; AGPT results are
from the recipe described in §3.

| Model | Context | Optimizer steps | Held-out PPL |
|---|---:|---:|---:|
| Random init | — | 0 | 144.2 |
| Window seq=16 (saturated) | 16 | 2,000 | 16.92 |
| **AGPT d=16**, RMSProp+warmup-cosine | 16 | 50 | **15.28** |
| Window seq=32 (saturated) | 32 | 2,000 | 14.51 |
| **AGPT d=32**, per-subtree, 3 super-epochs | 32 | 195 | **13.17** |
| **AGPT d=32 bigram**, auto-LR, 1 super-epoch | 32 | 1,465 | **13.30** |
| Window seq=128 (saturated) | 128 | 10,000 | 7.00 |
| Window seq=512 (saturated) | 512 | 10,000 | 7.13 |
| Window seq=1024 (saturated) | 1024 | 10,000 | 6.30 |

**At matched context, AGPT wins by ~10%.** At d=16: 15.28 vs 16.92 (10%
lower PPL) in 50 Adam steps vs. 2000 SGD steps (40× fewer). At d=32:
13.17 vs 14.51 (9% lower PPL) in 195 Adam steps vs. 2000 (10× fewer).

**At longer context, window training wins.** The 108k-parameter window
model absorbs substantially more context: seq=128 reaches PPL 7.00 and
seq=1024 reaches 6.30 — well below AGPT d=32's 13.17. This is the
natural boundary: AGPT's advantage is per-step gradient efficiency at
matched depth, but depth costs memory in a way that context-length for
window training does not. Scaling AGPT to d≥128 is the natural next
experiment; the per-subtree + bigram + pruning infrastructure in §3
was built for exactly this.

The step-count sweep shows seq=16 and seq=32 peak at 2k steps and
degrade with more training — short-context models memorize specific
windows rather than learn the distribution — while seq=128+ need 10k
steps to saturate. The numbers in this section are post-fix
measurements taken after resolving a backend-save bug in the Crystal-
side window trainer that had been silently masking whether gradients
reached the checkpoint at all. See the repository's
`docs/known-bugs-cublas-training.md` for the history.

### 4.3 Super-epoch sensitivity (d=32 unigram)

| Super-epochs | Adam steps | Held-out PPL |
|---:|---:|---:|
| 1 | 65 | 13.71 |
| 2 | 130 | 13.18 |
| **3** | **195** | **13.17** |
| 5 | 325 | 14.84 |
| 10 | 650 | 18.27 |
| 50 | 3,250 | 37.34 |

The AGPT objective converges faster than window training, and is
correspondingly easier to over-train: 50 super-epochs catastrophically
overfits. We provide an external best-checkpoint wrapper
(`agpt_train_best.sh`) that writes per-super-epoch checkpoints and
selects the best by held-out PPL — removing the need to hand-tune the
super-epoch count.

### 4.4 Radix + pruning numbers

| Config | radix nodes | total edge chars | Peak KV (d=32) | PPL |
|---|---:|---:|---:|---:|
| Uncompressed leveled trie | 27.0 M | — | n/a | n/a |
| Radix compression | 1.67 M | 27.0 M | 4.0 GB | 13.17 |
| Radix + prune m<2, d≥4 | 0.55 M | 1.17 M | 175 MB | 16.07 |

Radix compression gives a 16× node reduction for free. Pruning by
prefix mass trades quality for memory: 23× additional reduction at
~3 PPL cost. On small corpora the cost is real (rare paths are a
significant fraction of branching signal). On billion-token corpora
we expect the cost to collapse — long-tail mass=1 paths become
essentially noise — but this is not yet empirically verified.

## 5. Implementation notes

Full implementation in Crystal (trie builder, CLI) and CUDA C++
(training engine) at the referenced repository. Key files:

- `src/agpt/streaming_radix_builder.cr` — radix trie + per-subtree
  file format + bigram partition + pruning.
- `src/cuda/agpt_train.cu` — training engine, including per-subtree
  loop, auto-LR scaling, and full GPU-state cleanup on each
  subtree boundary.
- `scripts/agpt_train_best.sh` — external best-PPL wrapper.

## 6. Limitations and future work

Honestly stated:

1. **Scale**. All results are on a 4.6 MB char-level corpus with a
   108k-parameter model. Modern LLM training is ~9 orders of magnitude
   larger in parameter count. This paper's claims are that AGPT beats
   window training *on a small corpus at matched context*; generalization
   of the advantage to BPE vocabularies and billion-token corpora is an
   open empirical question.

2. **KV-cache scaling to BPE**. At Opus-family hyperparameters
   (d_model ≈ 16k, layers ≈ 100) a 1B-token corpus would require
   petabytes of aggregate KV memory across all subtree files.
   Per-subtree training makes this distributable, but the per-node
   memory budget still requires aggressive pruning + finer partitions
   (trigram or beyond).

3. **Pruning quality at scale**. Section 4.4's pruning results on
   Shakespeare cost 3 PPL. We hypothesize near-free pruning at larger
   scale (most long paths truly unique = noise), but this is unvalidated.

4. **Generation quality**. This paper reports PPL only; subjective
   quality of generated text has not been systematically compared.

5. **Compute-matched comparison**. Our baseline is step-saturated at
   each context length. AGPT uses ~10× fewer optimizer steps but each
   step does more forward/backward work (it covers an entire
   subtree). A wall-clock compute-matched comparison would quantify
   how much of the per-step advantage translates to real training-
   time savings. On the same 108k-param model and 8 GB GPU, the
   observed wall-clock times are: AGPT d=32 at best-PPL = 21 minutes;
   window seq=32 2k steps at best-PPL ≈ 45 seconds. AGPT spent more
   compute per-PPL-unit here, which tilts the honest framing away
   from "faster" toward "different operating point."

6. **Peer comparison**. No head-to-head with tokenizer-free neighbors
   (ByT5, MambaByte, etc.) has been run. These are the natural peer
   group for a char-level method; a proper comparison requires running
   them on the same corpus at the same model scale.

7. **Data-structure scaling**. The radix trie is a straightforward
   representation but not the most compact. For billion-token BPE
   corpora, a suffix array + LCP would give O(N) storage (8 bytes per
   token instead of ~20-40 bytes per radix record) and enable
   streaming enumeration of branching events without materializing
   the trie. Moving to an SA-based builder is a clear next engineering
   step for scale experiments; the radix trie in this paper served its
   role as a reference implementation for validating the core idea.

The natural next validation is **WikiText-103 scale with a real BPE
tokenizer on a cluster-grade GPU** — approximately 2 orders of magnitude
larger than the experiment here, achievable on a single A100 for
< $100 of cloud compute. This would establish whether the AGPT
advantage holds at BPE vocab sizes and 100M-token corpora, which
would be the meaningful first step toward frontier scale.

## 7. Summary of contributions

1. **Subtree-scoped Adam-step invariant.** Treating a root-child
   subtree as a single training unit (forward-backward-step) makes
   the one-shot gradient equal to the count-weighted sum of all
   corpus events with that prefix, without mid-subtree K/V
   staleness.

2. **Radix-compressed, per-subtree-file trie format.** Collapses
   unary chains and scopes KV-cache allocation to one subtree at a
   time, making depth-32 training fit in a 15 GB consumer RAM
   budget.

3. **Bigram subtree partition.** Splits each root-child by the
   second character of the path, attacking the dominant-root-child
   memory ceiling (the space character's subtree in English).
   6× further peak-memory reduction at Shakespeare d=32.

4. **Auto-LR scaling for subtree granularities.** The empirical
   invariant `lr × steps_per_super_epoch ≈ constant` lets a single
   base LR hold across unigram, bigram, and (prospectively) deeper
   partitions. A single `--lr-scale-by-steps` flag handles
   cross-granularity calibration.

5. **Frequency-pruned trie construction.** A `(min_mass, min_depth)`
   pair gates out paths whose prefix count is below threshold past a
   safety depth. Large memory savings (23×) at modest quality cost
   (3 PPL) on the small corpus; the tradeoff ratio is expected to
   improve at scale.

Full code is published at the referenced repository under an
MIT license; the git history provides the disclosure trail for
priority purposes.

---

*Corrections, discussion, and collaboration inquiries welcome.*
