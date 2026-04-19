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
on a 1.1 MB Shakespeare corpus: **9% lower held-out perplexity at
matched context with 10× fewer optimizer steps** versus a
step-saturated window baseline. Working code, reproducible numbers,
and a theoretical paper with the full gradient-factorization
derivation are published.

## What's demonstrated vs. what's unknown

**Demonstrated** (1.1 MB corpus, 108k-param char-level model,
consumer GPU):

- Gradient factorization is real and implementable — the trie
  gradient is mathematically equivalent to the window-training
  gradient sum, §5 of the paper proves this.
- It converges faster per optimizer step at matched context — PPL
  13.17 in 195 Adam steps vs. 14.51 in 2000 SGD steps (trie d=32
  vs. window seq=32).
- It scales to d=32 on consumer hardware through per-subtree KV
  allocation (4 GB peak vs. 27.6 GB global), and to d=48+ through
  bigram partitioning.

**Unknown** (what the grant enables validating):

- Does the per-step advantage hold at BPE vocabulary sizes?
  Char-level vocab is 65 tokens; BPE is 50k+. The branching structure
  at the trie root changes qualitatively.
- Does it hold at 100M-token and billion-token corpora? At scale
  the mass=1 long tail becomes noise and pruning becomes close to
  free — an asymmetric advantage for trie-based training vs. window.
- Can it compete head-to-head with the peer group (ByT5, MambaByte)
  at matched model size + compute?

## What the grant buys

A one-month cloud experiment at ~100× current scale:

- **Corpus**: WikiText-103 (~100M tokens) with a standard BPE
  tokenizer (GPT-2 vocab).
- **Model**: ~10M parameters, 6-layer transformer.
- **Hardware**: single A100-class GPU on spot pricing.
- **Comparison**: AGPT vs. a saturated window baseline vs. ByT5
  at matched model size and compute.
- **Output**: updated paper with scale-10× results, arXiv v2,
  submission to a workshop (NeurIPS ML-Sys, ICLR Tiny Papers, or
  similar).

Estimated compute cost: **$500-1,500** (cloud spot, ~100-300
A100-hours).

## Funding ask

**$15,000** to cover:

- Cloud compute budget ($1,500)
- 3 months of personal runway to focus on the experiments and
  writeup full-time ($13,500 — bare-bones; I am currently without
  other income)

Deliverables in 3 months:

1. Mid-scale experiment result (BPE, 100M tokens, ~10M params).
2. Updated paper posted to arXiv.
3. Workshop submission.
4. Open-source release of the cloud-ready training code.

## Why me, why now

I'm a software engineer who has been working independently on this
for several months. The full implementation — radix-compressed trie
builder, a CUDA training engine, per-subtree memory scaling
infrastructure, bigram partition, auto-LR, held-out PPL evaluation,
and the theoretical paper — is already working and public. What's
missing is the compute and runway to run the scale validation that
would determine whether this is a genuine methodological advance or
a small-corpus artifact.

I'm self-taught in ML and work better as a solo implementer than
embedded in a lab. This is a good fit for EV's model of supporting
independent researchers who've shipped something concrete but need a
push to go to the next step.

## What success looks like

Best case: the mid-scale experiment shows trie-based training
matching or beating window training at BPE vocabularies and
100M-token corpora. The paper gets cited; a big lab (Anthropic,
Together, EleutherAI) picks it up for frontier-scale validation and
I either collaborate or get hired.

Realistic case: the advantage partially holds at scale, with
caveats about specific corpus types or model sizes. A publishable
result that contributes a new training primitive to the community.

Worst case: the advantage does not survive the BPE / 100M-token
jump. Still a publishable negative result with clear diagnosis of
why, and the memory-scaling infrastructure (per-subtree, bigram,
SA-based builders) remains a reusable contribution for trie-based
training in general.

## Why this is EV-shaped

- Independent researcher without institutional backing or current
  funding.
- Working implementation (not a pitch deck).
- Clear, cheap experiment that would resolve the key open question.
- Small-dollar ask with concrete deliverables.
- Upside is asymmetric: if the method scales, it changes how
  frontier models are trained.
