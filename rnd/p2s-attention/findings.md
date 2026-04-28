# Prefix-to-Suffix Attention — Final Findings

> **Status (2026-04-27): closed.** Full architectural investigation complete.

## TL;DR

We set out to build a new prediction architecture using structural matches
between prefix-trie leaves and suffix-trie leaves. After thorough empirical
investigation, the architectural ambition didn't pan out — but we
characterized the corpus's structural properties precisely and found a
small but real inference-time win.

| Approach | Corpus PPL | Notes |
|---|---:|---|
| AGPT family ceiling | ~29 | starting reference |
| Cross-attention p2s (D=32, ctx=32) | 10.84 | bottleneck/leak issues |
| Cross-attention p2s, scaled (256/6, ctx=128) | 9.45 | better recipe, same arch |
| **Direct transformer ctx=128** | **6.50** | the winning baseline |
| **Direct + inference-time tree mask** | **6.35** | small but real arch win |
| Wrap-around-synth SGD seq=128 ref | 6.78 | comparable prior baseline |

## What worked

### Direct transformer training (PPL 6.50)
Standard transformer trained with corpus[p-127..p] as input, predict
corpus[p+1]. Same recipe as wrap-around-synth SGD seq=128, just on real
corpus. 875K-param model trained in ~10 minutes. Beats the prior reference
(6.78) cleanly.

### Inference-time mask (PPL 6.35)
At held-out positions where the structural matching produces multiple
distinct candidate next-chars, mask the model's logits to that candidate
set. Applied to ~1.9% of positions (the genuinely-ambiguous ones).
**Crucially**: applies only when there's real ambiguity in the candidate
set, avoiding the leak that killed cross-attention modes.

### Word-aligned matching is 12× more useful at word boundaries
The structural matching's value is concentrated at word boundaries (where
corpus[p] is a space):
- All-position PPL improvement from mask: -0.13
- Space-position PPL improvement from mask: -1.56

This connects to the prior `--space-cut` observation about cleaner
generation: word boundaries are where real linguistic decisions happen,
and where the structural matching has work to do.

## Real corpus property characterizations

### Branching depth ≈ 11
For each mass-1 prefix-tree leaf, computed `d*` = depth at which path becomes
unique. Distribution is bell-shaped centered at d*=10-11, with mean 11.23 and
median 11. **Empirically matches the theoretical prediction**
`log₂(N)/per-char-entropy ≈ 22.3 / 2 ≈ 11`. The corpus has an intrinsic
information-theoretic structure independent of architecture.

### D=32 is the right operating point for Gutenberg 5M
- D < 22: matching loses meaning (overlaps too short)
- D = 30-32: flat plateau (~95% single-cand, mode k=21)
- D = 48: saturation (radix barely grows past 32)

D-vs-PPL is also flat across 30-32. The "sweet spot" isn't a knife-edge.

### 95% deterministic continuation at D=32
- 4.99M prefix leaves
- 95.4% have exactly 1 structural candidate (mass-1 single-cand)
- Mode overlap k = 21 (phrase-level chunks)
- 4.6% multi-candidate, of which most are deterministic at next-char level

### Word-aligned matching is structurally cleaner
Truncating leaf-edges at the last space increases single-cand rate from
95.4% to 99.3%. Mode k jumps from 21 to 32. The remaining 0.7% multi-cand
cases are *real linguistic branch points*: verb/adjective/noun choices at
phrase positions ("Prince Andrew, [watched/glanced]", "with the most
[exasperated/lively/awful]", etc.).

## What didn't work

### Cross-attention as prediction backbone — a structural dilemma
The natural design — prefix encodes, σ candidates encode (same transformer),
cross-attention picks, output decodes — has a structural problem that
manifested as a leak in our implementation:

- **σ = corpus suffix at p+1 (our setup)**: σ's input contains
  σ.fwd[overlap_k] = corpus[p+1] (the answer). Feeding σ into the encoder
  leaks the answer through self-attention. PPL 3.09 (aux mode), or PPL
  ~10 with the bottleneck partially limiting the leak.

- **σ with last char masked (root-LOO)**: no longer leaks corpus[p+1] at
  exact position k, but σ.fwd[:-1] still covers corpus[q..q+30] which
  includes corpus[p+1..p+10] for overlap=21. The "target" σ.fwd[-1] is
  now corpus[p+11], not corpus[p+1]. So root-LOO trained on **far-char
  prediction**, not next-char prediction. PPL 11.5 on this (different
  task than the 6.50 baseline measures).

The dilemma underneath: σ-as-corpus-suffix is defined to start AT or near
p+1, so it inherently contains future chars. Either σ contains corpus[p+1]
(leak) or it doesn't (then it can't predict corpus[p+1]). To break the
dilemma you'd need σ to be something other than the corpus suffix —
e.g., representing context BEFORE p — but that's a different architecture
entirely, not p2s as we've defined it.

**The implementation-level "leak" was a bug, fixable by masking** (we did,
via root-LOO). What's **architecturally fundamental** is that the fixed
version doesn't predict next-char; it predicts a far-future char. There's
no clean way to use σ-as-corpus-suffix for next-char prediction without
either leaking the answer or solving a different prediction problem.

### Wrap-around / chained training
Tried building chained training examples that use σ's content past the
overlap as additional context for predicting deeper into the chain. Gives
wrong target alignment — predicts ~11 chars ahead instead of 1. Reverted.

### Mass-weighted, d*-weighted, skip-K=1 sampling
None of these training-signal modifications produced meaningful PPL
changes at scale. Position-based sampling (the natural distribution) is
already optimal for next-char prediction.

### Word-aligned inference mask
Applied at all positions, gave zero PPL change. The word-aligned matching
predicts "next char after current word ends" not "corpus[p+1]" — the
prediction targets don't align. Word-aligned matching implies word-level
prediction, not char-level.

## Why the cross-attention architecture was capped against SGD

Once the leak issue is set aside (root-LOO removes it), two structural
constraints prevent the architecture from beating direct transformer for
next-char prediction:

**1. The matching at low overlap is noise.** At k=1 (~0.03% at D=32),
the prefix's last char matches some random suffix's first char with no
corpus-positional alignment. The "concatenation" pairs unrelated text
fragments. There's no useful signal there.

**2. The matching at high overlap is corpus-positional.** At k=21+, the
matching pins to a single corpus position whose continuation is already
visible in the model's input context. Whatever the matching tells us,
SGD with the same context window has already seen.

**3. (the dilemma above)** σ-as-corpus-suffix can't be used for next-char
prediction without leaking the answer. The non-leaky variant (root-LOO)
predicts a different target (far-future char) which our PPL eval doesn't
measure.

There's no regime where the matching strictly beats SGD on the next-char
PPL task. At low k it's worse (adds noise); at high k it's redundant
(provides what SGD has); a clean non-leaky version doesn't predict the
right target.

## Artifacts produced

### Tools (`bin/`)
- `agpt_build_radix_corpus --reverse` — suffix-tree builder
- `agpt_p2s_match` — match-index builder (Crystal, ~3min, ~6GB RAM)
- `agpt_p2s_inspect` — match index visualizer

### Python prototype (`rnd/p2s-attention/proto/`)
- `p2s_train.py` — parameterized training, supports {direct, cross-attn,
  aux, root-LOO} modes; mass-weighted, d*-weighted sampling; CONTEXT_LEN,
  position-based, leaf-based options
- `p2s_eval_corpus.py` — corpus-walk evaluator with optional inference mask
- `branching_depth.py` — d* analyzer per leaf
- `word_aligned_match.py` — word-aligned matching variant
- `word_aligned_multi.py` — multi-candidate decoder for word-aligned
- `listing.py` / `k1_listing.py` — visualization tools

### Data artifacts (`/home/trans/agpt-tries/`)
- `gutenberg_5m_d{16,24,30,32,64}_radix_corpus/` — prefix trees at each D
- `gutenberg_5m_d{16,24,30,32,64}_suffix_radix/` — suffix trees
- `g5m_d{16,24,30,32}_p2s_match.bin` — match indices

### Logs (`rnd/p2s-attention/logs/`)
Run logs and analysis outputs preserved for all experiments.

## Reproduce headline results

```sh
# 1. Direct mode baseline (PPL 6.50)
P2S_D=32 P2S_TAG=direct_ctx128 P2S_CONTEXT_LEN=128 P2S_DIRECT=1 \
  P2S_D_MODEL=128 P2S_N_HEADS=4 P2S_N_LAYERS=4 \
  P2S_BATCH=32 P2S_LR=3e-4 P2S_STEPS=10000 P2S_WARMUP=500 \
  P2S_RECORDS=500000 \
  python3 rnd/p2s-attention/proto/p2s_train.py
P2S_D=32 P2S_TAG=direct_ctx128 P2S_DIRECT=1 \
  python3 rnd/p2s-attention/proto/p2s_eval_corpus.py
# → corpus PPL 6.50

# 2. With inference-time mask (PPL 6.35)
P2S_D=32 P2S_TAG=direct_ctx128 P2S_DIRECT=1 P2S_USE_PRIOR=1 \
  python3 rnd/p2s-attention/proto/p2s_eval_corpus.py
# → corpus PPL 6.35

# 3. d* analysis
python3 rnd/p2s-attention/proto/branching_depth.py
# → mean d* = 11.23, median 11

# 4. Word-aligned matching analysis
python3 rnd/p2s-attention/proto/word_aligned_match.py
# → 99.3% single-cand
```

## Lessons

1. **Architectural ambition needs to clear the SGD baseline first.** We spent
significant time on cross-attention variants before realizing that direct
mode (no matching) was the right baseline — and our cross-attention setup
was *worse* than this baseline.

2. **Beware of "good results" that turn out to be leaks.** The aux-mode PPL
3.09 result looked transformative until we traced where σ.fwd[overlap_k]
was coming from. Always check that the prediction target isn't reachable
through the input.

3. **Corpus properties scale with theoretical bounds.** The branching depth
matched `log₂(N)/per-char-entropy` precisely. For future work on
similar architectures, this lets us predict the operating regime from
corpus statistics rather than empirical sweep.

4. **Inference-time augmentation > training-time integration** for static
structural information. The matching gave no training-time win in any
variant we tried, but did give a small inference-time win when applied
selectively to genuinely-ambiguous positions.

5. **Word boundaries are where the action is.** Mid-word positions are
predictable from local context (SGD does fine); word boundaries are where
linguistic decisions happen and where structural information helps most.

## Possible future directions (not pursued)

- **Word-level model**: predict next *word* (variable-length token) given
  word-aligned context. Word-aligned matching gives small candidate sets
  per position — could form the basis of an efficient word-level prediction
  architecture.
- **Generation-time use**: use the structural matching as a *constraint*
  during sampling (post-hoc filter on model outputs) to ensure outputs
  stay structurally valid. Doesn't help training PPL but could improve
  generation quality measurably.
- **Retrieval-augmented application**: use the match index as a corpus
  retrieval primitive for tasks other than next-char prediction.

These are all sensible follow-ups but require different problem framings
than the one this experiment investigated.
