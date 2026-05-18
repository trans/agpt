# Training depth vs. inference context: they're different levers

**Short version:** the depth at which the trie is built for AGPT
training and the context length the trained transformer uses at
inference are independent choices. Our earlier intuition — that they
should match, because "why would the model use context past what it
was trained on?" — assumed the trie and the transformer measure the
same quantity. They don't.

---

## The puzzle that surfaced it

Empirical branching analysis (see the convergence paper) shows the
trie's branching signal saturates around **d ≈ 12-16** on Shakespeare.
Past that depth, most paths are mass-1 — unique single-occurrence
sequences carrying essentially no distributional information.

By that reasoning, a transformer with seq_len = 16 should be enough
to capture all the learnable distributional signal, and longer
context should give diminishing returns.

But empirically this is not what we observe. Our own measurements:

- Window seq=16, saturated: PPL 16.92
- Window seq=32, saturated: PPL 14.51
- Window seq=128, saturated: PPL 7.00
- Window seq=1024, saturated: PPL 6.30

PPL improves substantially with longer context, well past where the
trie's branching signal is exhausted. Something is getting learned
from longer context that the branching analysis doesn't account for.

## Resolution: the trie and the transformer measure different things

The trie is a **nonparametric density estimator over exact
sequences**. Its "saturation point" says: past depth d, there is no
more *exact-prefix* information to be extracted from the corpus.

The transformer is a **parametric pattern learner over compositional
features**. Its context is not used to memorize exact prefixes — it
is used to learn a distributed representation of the context that
captures:

1. **Generalization via similarity.** A transformer can predict from
   a unique long prefix by relating it to similar (not identical)
   seen prefixes through its learned embedding space and attention
   patterns. The trie has no such mechanism — a mass-1 prefix is
   just noise to it.

2. **Long-range dependencies the branching structure doesn't
   encode.** Coreference ("He" resolving to "the boy" 30 tokens
   earlier), discourse structure, topic continuity, paragraph-level
   patterns. The trie's branching at depth 16 says nothing about any
   of these; they depend on features that only become visible across
   longer spans.

3. **Compositionality at scales the trie can't represent.** Complete
   phrases, sentence structure, punctuation patterns at sentence or
   paragraph scale. A d=16 trie literally cannot represent "a full
   English sentence at character-level"; a d=128 transformer can.

4. **Smoothing by a different mechanism.** The trie uses additive
   (Laplace) smoothing — weak, uniform-prior-based. The transformer
   smoothes via its learned embedding geometry and attention weights
   — a richer mechanism informed by what the data actually looks
   like. This is why a transformer can make useful predictions from
   never-before-seen prefixes while a trie can only back off.

The trie's branching saturation says "no new *exact-prefix*
information past depth d." It does **not** say "no new *learnable
pattern* past depth d." Those are different claims.

## Implication for AGPT design

AGPT uses the trie structure to organize training gradients. The
trained model is a standard transformer. Therefore:

- **Training trie depth** should be chosen based on where the
  branching signal is informative — the regime where the trie's
  count-weighted gradient formulation provides real advantage over
  random-window training.
- **Inference context length (seq_len)** should be chosen based on
  the transformer's capacity to extract generalizable patterns at
  scale — which can be much larger than the training trie depth.

These are independent levers. A reasonable design point might look
like: trie depth 32 (training side — branching still meaningfully
non-trivial at d=32 and per-subtree infrastructure handles it), with
transformer seq_len 512 or larger (inference side — leveraging the
transformer's pattern-learning capacity on longer spans).

Our current recipe already does this implicitly (d_trie = 16 or 32,
seq_len = 128), but the framing was wrong. It was not "seq_len has
to cover the trie depth." It's "training gradient organization and
inference context are picked for different reasons, and the best
choice for each is not the same number."

## Why this matters for scaling

When planning the BPE / mid-scale follow-up experiments, resist the
instinct to make trie_depth and seq_len scale together. They
shouldn't. Two separate curves:

- Trie depth scales until branching signal saturates for the given
  corpus. Past that, the trie adds mostly noise-dominated
  mass-1 paths.
- Inference context length scales as far as the transformer can
  make productive use of it — likely much further than the trie's
  saturation point, given the mechanisms listed above.

A transformer trained with AGPT at a moderate trie depth can still
benefit from a long inference-time context. The "best d" question
becomes "best training d," which is a smaller number than "best
inference context."

## Concise form for the paper

> The depth at which the trie provides informative branching signal
> and the context length at which the transformer can extract
> useful predictive features are independent quantities. The first
> saturates relatively early (~d=12-16 on Shakespeare) because past
> that depth most paths are mass-1 and carry no distributional
> information. The second extends much further because the
> transformer's context is used for pattern generalization, not
> exact-prefix memorization — these are different estimation
> problems.

---

*Captured as a design-level framing note for AGPT follow-up work.
Origin: discussion of "why d=128 transformers beat d=16 transformers
when trie-branching analysis suggests d=16 is enough."*
