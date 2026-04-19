# Emergent Ventures application — AGPT (Aggregated-Gradient Pretraining)

**Applicant:** Thomas Sawyer &nbsp;·&nbsp; `transfire@gmail.com`
**Draft:** 2026-04-19
**Repo:** https://github.com/trans/microgpt
**Paper (preprint):** https://github.com/trans/microgpt/blob/main/docs/agpt-paper.md

---

## The project in one paragraph

Standard language-model training draws random windows from a corpus
and has the optimizer stochastically reconstruct the count-weighted
sum of gradients implicit in the corpus. That sum is statically
computable from a prefix trie of the corpus — one gradient per
distinct prefix, not one per sampled window. I have implemented this
end-to-end (CUDA trainer, radix-compressed trie, per-subtree
KV-cache scoping, bigram partition, auto-LR scaling) and validated
the idea on a Shakespeare corpus. In early experiments at matched
context length, the trie-based method reaches lower held-out
perplexity with roughly 10× fewer optimizer steps than a standard
window-training baseline. The next-stage experiment is a stricter
matched-compute comparison at larger scale.

**This is not a vague proposal awaiting a team; it is a working
system awaiting scale validation.**

## What's demonstrated vs. what's unknown

**Demonstrated** (small-corpus experiment, 108k-parameter char-level
model, consumer GPU):

- Gradient factorization is real and implementable — the trie
  gradient is mathematically equivalent to the window-training
  gradient sum (§5 of the paper derives this).
- At matched context, trie-based training reaches lower held-out
  perplexity than a conventional window baseline with roughly 10×
  fewer optimizer steps (trie runs use RMSProp; window uses Adam —
  both adaptive).
- The method fits depth-32 training in a consumer-memory budget
  through per-subtree KV allocation, and scales to deeper depth
  through bigram partitioning.

**Unknown** (what the grant enables validating):

- Does the per-step advantage hold at BPE vocabulary sizes?
  Char-level vocab is 65 tokens; BPE is 50k+. The branching structure
  at the trie root changes qualitatively.
- Does it hold at 100M-token and billion-token corpora? At scale,
  the mass-1 long tail becomes noise and pruning becomes close to
  free — a possible asymmetric advantage for trie-based training.
- Can AGPT compete with conventional window training at matched
  model size and compute on a mid-scale benchmark?

## What the grant buys

Three months of full-time research to take the method from
char-level validated to BPE-scale tested, culminating in an arXiv
paper and a workshop submission. Concretely:

**Phase 1 — BPE integration and a pilot run** (WikiText-2 scale,
~2M tokens). Validates that the trie formulation survives the jump
from a 65-token character vocabulary to a 50k-entry BPE vocabulary.
Small enough to iterate quickly; big enough to surface the real
engineering issues.

**Phase 2 — Mid-scale comparison** (WikiText-103 scale, ~100M
tokens, ~10M parameters). Head-to-head against a saturated
window-training baseline at matched model size and compute. Scaling
to 100M BPE tokens on cloud hardware will require the per-subtree
infrastructure + bigram/trigram partitioning + aggressive pruning;
single-GPU may be sufficient or may need a small cluster,
resolved during Phase 1.

**Phase 3 — Writeup and ablations.** Per-component contribution
analysis, failure cases, peer-method comparison (ByT5 or similar
tokenizer-sensitive baseline), paper revisions.

Estimated cloud compute across phases: **$500-2,500** — depending
on whether mid-scale fits a single 80GB GPU or needs a 2-4 GPU node.
The larger cost is time, not compute.

## Funding ask

**$15,000** for three months of full-time focused research. This
covers living expenses, cloud compute, and software/infrastructure
costs while I execute the plan above. I am presently without other
income; this grant would let me work on the project full-time rather
than split attention with contract work.

Deliverables at three months:

1. Phase 1 pilot result (BPE integration validated, negative or
   positive reported honestly).
2. Phase 2 mid-scale comparison result on WikiText-103.
3. Updated paper on arXiv.
4. Workshop submission (NeurIPS ML-Sys, ICLR Tiny Papers, or
   similar).
5. Paper, trained models, and evaluation scripts released publicly;
   training pipeline released under a source-available license
   suitable for academic use and reproduction.

## Why me, why now

I'm an independent software engineer and researcher who has been
working on this for several months. The full implementation —
radix-compressed trie builder, CUDA training engine, per-subtree
memory scaling infrastructure, bigram partition, auto-LR, held-out
PPL evaluation, and the theoretical paper — is already working and
public. What's missing is the compute and runway to run the scale
validation that would determine whether this is a genuine
methodological advance or a small-corpus artifact.

This is a good fit for EV's model of supporting independent
researchers who've shipped something concrete but need a small,
targeted push to answer an important open question.

## What success looks like

Even a negative result would be publishable, because the method is
concrete and the experiment cleanly identifies whether the advantage
survives scale. The memory-scaling infrastructure (per-subtree
allocation, bigram partitioning, and trie construction tooling) is a
reusable contribution either way.

## Why this is EV-shaped

- Independent researcher without institutional backing or current
  funding.
- Working implementation, not a pitch deck.
- Clear, cheap experiment that would resolve the key open question.
- Small-dollar ask with concrete deliverables.
- If the method scales, it could become a new training primitive for
  large language models.
  
