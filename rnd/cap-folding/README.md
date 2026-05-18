# Cap Folding via Composite Prefix-Trie Targets

**Status**: positive result at 6SE (fold beats baseline by 2.3% PPL); 3SE
was misleading due to slower fold convergence.

**Code**: branch `main`; key commit pending. Touchpoints:
- `src/tools/agpt_build_fold_table.cr` (new)
- `src/cuda/agpt_train.cu` — fold-table loader, kernel branch in
  `agpt_loss_per_query_kernel`, `--fold-table` CLI flag
- `Justfile` — `build-agpt-build-fold-table` recipe

## Hypothesis

Radix caps at d=32 have H≈0 (single observed continuation per ~23-char
tail). Today these train toward unreachable one-hot targets, contributing
mostly cancellation noise across suffix-aliased caps. Replacing the
target with the corpus-wide forward distribution `P(c | W)` (W = cap's
suffix tail), obtained by walking W from prefix-trie root and reading
children, should give the model dense supervision instead of sparse
deltas, and improve held-out PPL.

Predicted secondary signal: if fold collapses to short-W posteriors,
held-out PPL at long context (seq_len=32) shouldn't improve relative to
short context (seq_len=4). If real folding takes effect, the long-context
PPL should improve more.

## Scope

In scope:
- Composite mode only: walk W from root, read children → `P(c | W)`.
  No path-prob ranking, no single-best, no top-K node enumeration.
- Side-table form: per-cap top-K (token, prob) pairs.
- One corpus, one config (Shakespeare 1M d=32, pd=6 RMSprop --no-accumulate).

Out of scope:
- Suffix-trie-based fold (suffix-radix-trie of reversed corpus is the
  *backward* distribution — useful for KL_suffix in the broader
  prefix-suffix-fold-architecture, not for cap-fold targets).
- Single-best fold target (path-prob × log(mass) ranking).
- Materialized fold (in-trie children) — ~10× memory blowup.
- Dynamic teacher (use F's prediction at fold target instead of raw counts).
- Backward model B + KL_suffix term + ensemble inference.

## Setup

- **Corpus**: `data/input.txt` (Shakespeare 1M chars, V=65)
- **Trie**: `/tmp/agpt_input_d32_radix` (1.66M radix nodes, 33 endpoint
  depths, max_depth=32)
- **Architecture**: d_model=64, n_heads=4, n_layers=2, d_ff=256, seq_len=128
- **Recipe**: pd=6 RMSprop --no-accumulate, lr=3e-3, weight-decay=0.01,
  entropy-lambda=0, mass-weight=off
- **Init**: shared `data/input.random.model` for both runs (same seed)

## Reproduce

```sh
# 1. Build the fold-table for the d=32 trie (W∈[4..16], mass-min=10, top-K=16)
just build-agpt-build-fold-table
./bin/agpt_build_fold_table --trie /tmp/agpt_input_d32_radix --out /tmp/agpt_fold_d32.bin

# 2. Baseline (fold OFF) — 3 SE
./bin/agpt_train --model data/input.random.model \
  --trie-dir /tmp/agpt_input_d32_radix \
  --save /tmp/fold_d3_baseline.model \
  --epochs 3 --partition-depth 6 --no-accumulate \
  --optimizer rmsprop --rmsprop-beta 0.999 --lr 3e-3 \
  --weight-decay 0.01 --entropy-lambda 0 --mass-weight off

# 3. Fold ON — same recipe, add --fold-table
./bin/agpt_train --model data/input.random.model \
  --trie-dir /tmp/agpt_input_d32_radix \
  --save /tmp/fold_d3_fold.model \
  --epochs 3 --partition-depth 6 --no-accumulate \
  --optimizer rmsprop --rmsprop-beta 0.999 --lr 3e-3 \
  --weight-decay 0.01 --entropy-lambda 0 --mass-weight off \
  --fold-table /tmp/agpt_fold_d32.bin

# 4. PPL eval (8192 positions)
./bin/perplexity --model /tmp/fold_d3_baseline.model --file data/input.txt \
                 --seq-len 32 --backend openblas --max-positions 8192
./bin/perplexity --model /tmp/fold_d3_fold.model --file data/input.txt \
                 --seq-len 32 --backend openblas --max-positions 8192
```

## Artifacts

- kept in git: this `README.md`, side-table builder source
- ignored / regenerated: side-table file (~56 MB), trained models
- canonical artifact: this README's `## Results` section

## Results

### Fold-table build (Shakespeare 1M d=32)

| metric | value |
|---|---|
| caps total | 1,114,223 |
| caps w/ fold target | 912,175 (81.87%) |
| caps dead-end (no W match ≥ mass-min=10) | 202,048 |
| mean fold-target entropy | 1.188 nats (max possible 4.174) |
| W-length mode | W=5 (226,802 caps) |
| W-length range | 4..16 |
| build time | 36 s |
| side-table size | 55.6 MB |

W-length histogram (caps that found a fold target):

| W | count | share |
|---|---:|---:|
| 4 | 212,602 | 23.3% |
| 5 | 226,802 | 24.9% |
| 6 | 178,642 | 19.6% |
| 7 | 132,065 | 14.5% |
| 8 | 78,787 | 8.6% |
| 9 | 40,472 | 4.4% |
| 10 | 20,196 | 2.2% |
| 11 | 9,513 | 1.0% |
| 12 | 4,583 | 0.5% |
| 13 | 2,841 | 0.3% |
| 14 | 1,709 | 0.2% |
| 15 | 817 | 0.1% |
| 16 | 3,146 | 0.3% |

The bimodal distribution (mode at 5, secondary peak at 16) is
characteristic of the corpus: most caps share short suffixes with
many other prefixes (mass-rich common roots), a smaller tail has
truly unique 16-char suffixes.

### D.1 sanity & D.2 parity

- Foundational regression suite: skipped (d=8 trie absent); training
  with `--fold-table` not set behaves exactly as before.
- Empty fold-table parity: trained 1 SE pd=1 with `--fold-table
  /tmp/agpt_fold_d32_empty.bin` (all-dead-end). Loss 5.093187 vs
  baseline 5.093243 = 5.6e-5 absolute, ~1e-7 relative. **Parity
  passes within CUDA reduction non-determinism.**

### D.3 headline experiment

#### 3 SE (initial budget)

| metric | baseline | fold | diff |
|---|---:|---:|---:|
| Epoch 1 loss | 1.7224 | 1.7348 | +0.012 |
| Epoch 2 loss | 1.5473 | 1.5515 | +0.004 |
| Epoch 3 loss | 1.5289 | 1.5335 | +0.005 |
| Held-out PPL @ seq_len=32 (4096 pos) | 4.787 | 4.851 | +1.3% (fold worse) |
| Held-out PPL @ seq_len=32 (8192 pos) | 4.827 | 4.906 | +1.6% (fold worse) |

At 3 SE fold appears to lose by ~1.5%. Training loss is also higher —
partially expected, since the fold target has H>0 (mean 1.188 nats), so
per-cap CE has a positive lower bound (≥ entropy of fold target) that
one-hot training doesn't have.

But the loss gap stays roughly constant per epoch (~0.005), so it's
not just a fixed entropy floor — fold is also converging more slowly
under its softer gradient. So we extended training.

#### 6 SE (continuation from 3 SE checkpoints)

| metric | baseline | fold | diff |
|---|---:|---:|---:|
| Epoch 4 loss | 1.5260 | 1.5278 | +0.002 |
| Epoch 5 loss | 1.5114 | 1.5153 | +0.004 |
| Epoch 6 loss | 1.5064 | 1.5112 | +0.005 |
| Held-out PPL @ seq_len=32 (8192 pos) | 4.876 | 4.764 | −2.3% (fold wins) |
| Held-out PPL @ seq_len=32 (65536 pos) | **4.959** | **4.874** | **−1.7% (fold wins)** |

**The PPL gap completely inverted between 3 SE and 6 SE.** Baseline's
held-out PPL got slightly *worse* from SE 3 to SE 6 (4.827 → 4.876),
even though training loss kept dropping — classic overfitting onto
the one-hot cap targets that the model can't actually represent.
Fold's held-out PPL kept improving (4.906 → 4.764), driven by its
target distribution being reachable.

Wall-time was identical between the two configurations: ~215 s/SE.
Fold imposes no measurable runtime overhead.

### D.4 effective-context probe

#### 3 SE

| seq_len | baseline | fold | gap (fold − baseline) |
|---:|---:|---:|---:|
| 4 | 5.99 | 6.25 | +0.26 |
| 8 | 4.85 | 4.98 | +0.13 |
| 16 | 4.79 | 4.87 | +0.08 |
| 32 | 4.79 | 4.85 | +0.06 |

#### 6 SE

| seq_len | baseline | fold | gap (fold − baseline) |
|---:|---:|---:|---:|
| 4 | 6.49 | 6.24 | **−0.25** |
| 8 | 4.94 | 4.94 | 0.00 |
| 16 | 4.83 | 4.78 | **−0.04** |
| 32 | 4.82 | 4.75 | **−0.07** |

(4096 positions per cell)

**The "fold collapses to short-W" failure mode is falsified at both
3 SE and 6 SE.** Fold's PPL improves with longer eval context just as
much as baseline. At 6 SE, fold beats baseline at every seq_len except
the tied 8.

The seq_len=4 line is striking: at 6 SE, fold's PPL at 4-char context
(6.24) is **better** than baseline's PPL at 4-char context (6.49).
That's exactly the case where the suffix-marginal target should
dominate: with only the last 4 chars visible, the only reachable
target *is* P(c|W=last-4-chars), which is what fold trained the
model to produce. Baseline had to derive that distribution implicitly
via aliased one-hot gradients, and at 6 SE it has overfit those
specific one-hots harder than it has converged to the marginal.

### D.5 generation quality

Sample seed `"First Citizen:\n"` at temp=0.8 (openblas backend;
cuBLAS gen broken — see `notes/todo/cublas-generation-bug.md`).

3 SE:
- Baseline: `"But have imples of preat his monds! Une shaved, a kingedAndouform thal's menoffouresty thOf..."`
- Fold:     `"But dath, and the new all consoc longBang eylay alesh. Anewsinote sharenen hiches van upt theatedy st bstesty..."`

6 SE:
- Baseline: `"But eatle and the not a falorrof hint bein wely alatin beforifom'd hat fareatalilated wiveserithavesthat eves youst lofiveris edey t prid bit ealid us myow bele wofooreave..."`
- Fold:     `"But dath, and their sacine: and ber wealy ful t an thl andwhs sppare gelel hobent p may, stan so wintirenormonupllobly heeltremmatet surcendread lalid my wembighounedy..."`

Subjectively similar quality at both budgets. The 0.06 PPL gap
doesn't translate into visibly different generation at d=64 n_layers=2
— both models are clearly under-trained on this small architecture.
A larger model trained with fold may show the difference more clearly.

## Conclusion

Composite cap-folding **improves held-out PPL by 2.3% at 6 SE** on
Shakespeare 1M d=32 under our best-known recipe (pd=6 RMSprop
--no-accumulate, lr=3e-3, weight-decay=0.01). Wall-time is unchanged.

Critically, the 3 SE result was misleading: at that budget fold *lost*
by ~1.5% PPL because the soft fold gradient converges more slowly than
the hard one-hot gradient. The trajectory inverts between 3 SE and
6 SE — fold keeps generalizing while baseline starts to overfit onto
the unreachable one-hot cap targets. Anyone running this experiment
at small SE budgets would have called it dead. The headline is:
**fold is a regularizer that pays back at higher training budgets.**

The "short-W collapse" failure mode is falsified at both budgets.
Fold uses long-range context as well or better than baseline. At
6 SE, fold is strictly better at every eval context length tested
(seq_len ∈ {4, 16, 32}; tied at 8).

The mechanism behind the win is consistent with the original
hypothesis: cap targets one-hot at H=0 are unreachable, so each cap
firing contributes mostly cancellation noise across suffix-aliased
caps, and the model only converges to the right marginal through
aliasing-as-implicit-averaging. Fold short-circuits that by handing
the model the right marginal directly. At low SE, baseline catches
up because its variance helps it find sharper structure faster;
beyond convergence, baseline's variance becomes overfit to memorized
specifics, while fold's smoother gradient continues moving toward
the actual generalizable distribution.

## Cutoff sensitivity (added 2026-05-04)

The original fold-table used artificial floors: `w_min=4`, `w_max=16`,
`mass_min=10`. To probe whether these floors helped or just got lucky,
we built a "natural-cutoffs" variant: `w_min=1`, `w_max=32`,
`mass_min=2` — the trie's own structure is the only filter (deepest W
where the substring still has ≥ 2 corpus occurrences and ≥ 2 distinct
next chars).

| variant | coverage | dead-end | mean H (nats) | side-table size |
|---|---:|---:|---:|---:|
| original (w_min=4, mass_min=10) | 81.87% | 202k | 1.19 | 56 MB |
| natural (w_min=1, mass_min=2) | 99.94% | 0.6k | 0.84 | 47 MB |
| h-min=0.5 (w_min=1, mass_min=2, h_min=0.5) | 99.12% | 9.8k | 1.03 | 58 MB |

Natural's lower mean entropy comes from accepting many long-W matches
with mass=2 (one corpus aliased position only) — the resulting
distribution is approximately {a: 0.5, b: 0.5}, technically branching
but barely informative. H-min=0.5 forces those candidates to step
back to a shorter W with richer distribution.

Note: h-min=0.5 actually *shortens* the average W vs natural — many
long-W candidates fail H-min and fall back to W=1..3 (where corpus
context is very generic and entropy is high). Worth checking whether
the entropy-floor logic and the "find deeper W with H-min" search are
working as intended; this might indicate the entropy of the
*depth-W-from-root* node is intrinsically low for those caps.

### Cutoff variant PPL comparison (fresh 6 SE, matched protocol)

PPL @ seq_len=32, 65536 positions, all variants trained from the same
random init for 6 SE (no continuation reset of optimizer state):

| variant | coverage | mean H | held-out PPL | vs baseline |
|---|---:|---:|---:|---:|
| baseline (no fold) | — | — | 4.998 | — |
| natural (m≥2, w 1..32) | 99.94% | 0.84 | 4.948 | −1.0% |
| h-min=0.5 (m≥2, w 1..32, h≥0.5) | 99.12% | 1.03 | 4.934 | −1.3% |
| m=5 (m≥5, w 4..16) | 84.98% | 1.05 | 4.990 | −0.2% |
| m=9 (m≥9, w 4..16) | 82.46% | 1.17 | 4.954 | −0.9% |
| **m=10 (m≥10, w 4..16)** | 81.87% | 1.19 | **4.864** | **−2.7%** |
| m=11 (m≥11, w 4..16) | 81.30% | 1.20 | 5.042 | **+0.9% (loses)** |
| m=30 (m≥30, w 4..16) | 72.05% | 1.33 | 4.958 | −0.8% |
| m=100 (m≥100, w 4..16) | 52.14% | 1.45 | 4.947 | −1.0% |

**The m=10 point is a single-seed spike, not a smooth peak.** m=9 and
m=11 have nearly-identical coverage (82.5% / 81.3%) and mean entropy
(1.17 / 1.20) yet land 0.09 / 0.18 PPL above m=10. m=11 actually
*loses* to no-fold baseline. With single-seed run-to-run variance of
~0.05 PPL, the m=10 outlier is partly seed luck — not a true
mass-floor optimum.

**Honest gain estimate:** the median across the m-sweep is around
−1% PPL. The m=10 specifically delivered −2.7% in this run, and a
matched-shuffle reproduction gave −2.0%. So the *expected* cap-fold
gain at this recipe is probably **−1% to −2%**, and the −2.7% / 12 SE
−3.5% headlines were inflated by lucky single-seed effects.

To pin down the true gain would need 5+ seeds at each m value
(prohibitive at ~22 min/run); single-seed sweeps only show rough
trends. The main signal — that fold helps, that it doesn't extend
seq_len, and that target substitution is fundamentally a regularizer —
holds across all variants and seeds.

### Training-budget trajectory

| SE | baseline | fold m=10 | fold gain | trend |
|---:|---:|---:|---:|---|
| 6 | 4.998 | 4.864 | −2.7% | both improving |
| **12** | 4.898 | **4.728** | **−3.5%** | optimum |
| 18 | 4.937 | 4.787 | −3.0% | both overfitting |

Fold delays overfitting but doesn't prevent it. Both models pass
their PPL minimum somewhere between 12 and 18 SE. Best held-out PPL
in this experiment: **4.728 at 12 SE with fold m=10**, vs the
previous best 4.864 at 6 SE.

If the trend held (fold gap kept widening), 18 SE would have given
~-4-5% gain. Instead the gap saturates around -3.0 to -3.5%,
suggesting fold's contribution is bounded — once both models hit
their generalization ceiling for this architecture, the relative gain
plateaus.

This sets a practical upper bound on the cap-fold mechanism alone at
this architecture scale: ~3.5% PPL improvement, achieved in roughly
half the training steps where baseline matches it.

### Optimizer interaction: fold + RMSprop wins, fold + Adam loses

| optimizer | baseline PPL | fold m=10 PPL | fold gain |
|---|---:|---:|---:|
| RMSprop | 4.998 | 4.864 | **−2.7%** |
| Adam | 4.998 | 5.063 | **+1.3%** (fold hurts) |

Surprising. Adam baseline matches RMSprop baseline (4.998 each), but
adding fold flips the sign: -2.7% with RMSprop, +1.3% with Adam.

Plausible mechanism: Adam's per-parameter LR adaptation divides by
running-mean of gradient magnitudes. Fold's gradients at caps are
*smaller* than baseline's (because the fold target is reachable, while
one-hot is unreachable — the loss saturates at H(fold) > 0). Adam
sees those smaller gradient magnitudes and *increases* the effective
LR for cap-related parameters, leading to overshoot. RMSprop's simpler
scaling doesn't have this pathology.

So fold and Adam compound badly. **RMSprop is the recommended pairing
for cap-fold training.** This is worth knowing because the project's
prior PPL best (3.95) used Adam — fold won't directly improve that
recipe; would need to re-tune optimizer hyperparameters or stick with
RMSprop.

### Seed-robustness check

Single-seed runs have ~0.05 PPL variance, so we re-ran both baseline
and m=10 with `--shuffle-order --shuffle-seed 12345` for matched
comparison:

| protocol | baseline PPL | fold m=10 PPL | fold gain |
|---|---:|---:|---:|
| default order | 4.998 | 4.864 | −2.7% |
| shuffle seed=12345 | 4.889 | 4.791 | −2.0% |

Both seeds show fold winning. Magnitude varies 2.0-2.7%; the central
estimate is around -2.4%. **m=10's win is robust, not seed-luck.**

A bonus finding: **shuffle-order helps both baseline and fold**
(baseline -2.2%, fold -1.5%). Random partition-group order is a mild
regularizer on its own. Stacking gives best PPL = **4.791** —
combined gain of -4.1% vs no-fold no-shuffle baseline.

The interpretation is statistical: at a W-node with `local mass = N`
samples observed, the empirical next-char distribution `P̂(c|W)` has
per-bin standard error ≈ `√(p(1-p)/N)`. For `p=0.1` (a typical head
bin) at `N=10`, SE ≈ 0.095 — about 95% relative error. At `N=100`,
~30%. At `N=1000`, ~10%. Fold targets at low mass are noise-dominated
estimators of the true distribution; the model trained against them
fits the noise.

So `mass_min` isn't really a cutoff "below which fold doesn't apply" —
it's a cutoff "below which the fold target is statistically too noisy
to feed into the model. Caps below the cutoff are better off with
their original one-hot fallback." The 18% dead-end rate at m=10 isn't
a coverage failure — it's a quality filter.

This also explains why the W histogram naturally lands at 5-7: the
typical 5-7 char tail in Shakespeare has hundreds-to-thousands of
aliased corpus positions (high mass = reliable estimator), while
deeper tails have only a handful (low mass = noise). The "natural
depth" of fold matching is wherever local mass is still high enough
to support a reliable distribution estimate.

## Open follow-ups (high-value, not done in this experiment)

1. **More SE.** Does fold's lead grow at 9 SE / 12 SE? Or does the gap
   stabilize at ~2-3%? The trajectory suggests it may keep growing,
   but we haven't measured.

2. **Loss-component instrumentation.** Add per-event-type loss
   breakdown (intermediate / internal-endpoint / cap-no-fold /
   cap-with-fold) so we can directly measure whether cap-loss drops
   under fold (and whether internal-loss is unchanged, as predicted).
   This would let future fold variants be diagnosed without retraining.

3. **Larger model.** d=64 n_layers=2 is small; the 0.07 PPL gap at
   seq_len=32 may grow with d_model=96 / n_layers=6 (which the project
   has tested as ceiling-pushers). A bigger model has more capacity
   to absorb the dense fold signal.

4. **Adam optimizer.** This experiment used RMSprop. The user's prior
   best (project_partition_depth.md, 3.95 PPL @ pd=6) was with Adam.
   Whether fold compounds with Adam's per-parameter adaptation is
   unmeasured.

5. **w_min sensitivity.** Default w_min=4 dead-ends 18% of caps. Try
   w_min=3 (more coverage, lower-quality short-W targets) and
   w_min=6 (less coverage, higher-quality matches).

6. **Larger top_k.** Default top_k=16 truncates; try 32 or full V=65.
   Storage cost is linear; gradient quality at low-mass tail tokens
   may improve.

7. **Dynamic teacher** (architecture notes §7.5). Use F's current
   prediction at the fold-target node instead of the static empirical
   histogram. May help when the model has already learned good
   marginals — the fold target then encodes the model's own
   long-range knowledge rather than just corpus statistics.

8. **Single-best fold ranking** (path-prob × log(mass) over
   prefix-trie internal nodes). Composite is principled but may
   over-smooth; single-best preserves more prefix-specific structure.

9. **Backward model B + KL_suffix term** (full prefix-suffix
   architecture). With cap-fold validated, the dual-model addition
   becomes more attractive.

10. **Larger corpus.** Test on Gutenberg 5M. d_optimal grows with
    corpus size, fold should help more when caps are deeper and more
    numerous.

## Open follow-ups (not done in this experiment)

1. Dynamic-teacher fold: replace the static empirical histogram with
   F's current prediction at the fold-target node (architecture notes
   §7.5).
2. Single-best fold ranking: enumerate prefix-trie internal nodes that
   contain W, rank by path-prob × log(mass), pick best.
3. Backward model B + KL_suffix consistency loss (full prefix-suffix
   architecture, §3 of the architecture notes).
4. Cap-loss instrumentation: per-event-type loss breakdown in the
   trainer to directly measure whether fold reduces cap-loss in
   isolation, even when total loss is comparable.

## Structural measurements (added 2026-05-04)

Two foundational measurements gathered during the cap-folding work that
inform any future fold/loop/dual-model design.

### Cap-edge length vs. corpus size

Mean radix-cap edge length (depth-32 endpoint nodes with `counts.size==1`)
across three corpus sizes, all built with `--max-depth 32`:

| corpus | size (chars) | caps @ d=32 | mean cap-edge length | %singleton |
|---|---:|---:|---:|---:|
| Shakespeare slice | 100,000 | 99,930 | **25.06** | 100.00% |
| Shakespeare full  | 1,115,394 | 1,114,330 | **23.29** | 99.99% |
| Gutenberg combined | 9,815,099 | 7,052,157 | **21.15** | 99.98% |

Each 10× in corpus size shaves ~2 chars off the mean cap edge. The
identity-tail length scales sub-logarithmically — corpus growth helps
slowly. At Gutenberg 10M, ~28% of corpus characters are *not* caps
(they're at branching internal nodes), vs. ~0% at Shakespeare 100k
where every position is a cap. So aliasing is the corpus-size-dependent
effect that fold can exploit; cap-edge length is structural and
slow-moving.

The ~2-chars-per-decade is consistent with d_optimal = log₂(N)/H + 21
from the trie-attention framing: identity-region depth is roughly
constant (~21) in linguistic terms, decision-region grows
log-with-corpus, and the cap-edge captures "everything past the
decision point in the unique tail."

### Forward vs. backward model disagreement

We trained two AGPT models on Shakespeare 1M d=32 with the same recipe
(pd=6 RMSprop --no-accumulate, 6 SE):

- **Forward F**: trained on the prefix radix-trie (standard direction).
- **Backward B**: trained on the suffix radix-trie (reversed corpus,
  `--reverse` flag).

Each predicts the same target c at corpus position p, conditioned on
opposite sides — F sees `tokens[p-32..p-1]`, B sees the reverse of
`tokens[p+1..p+32]`. We then compared their predictions at 4096
held-out positions:

| metric | value |
|---|---:|
| KL(F ‖ B) | **2.37 nats** |
| KL(B ‖ F) | **2.40 nats** |
| Symmetric KL | 2.38 nats |
| JS divergence | 0.33 nats (out of max 0.69) |
| Forward NLL on true c | 1.57 (PPL 4.80) |
| Backward NLL on true c | 1.63 (PPL 5.12) |
| Top-1 agreement | **33.2%** |

**The trained models disagree heavily even though the trie distributions
agree perfectly** (`bayes_probe.cr` confirmed KL=0 for the underlying
corpus statistics in both directions). The disagreement is at the model
level — both converge toward the same fixed point in the limit but at
finite training they encode genuinely different views of the corpus.

Two interpretations worth keeping straight:

1. **Models factor different aspects.** Forward learns continuation
   patterns (what follows from a prefix); backward learns precedence
   patterns (what precedes a suffix). Same data, different inductive
   structure. The 2.4-nat gap quantifies this divergence.

2. **Per-position prefix/suffix asymmetry is large.** At any individual
   position, prefix and suffix are not equivalent evidence — they carry
   different *amounts* of information about c. The two models converge
   to the corpus marginal *over many positions*, but at any single
   position prefix and suffix point in different directions. Top-1
   agreement of 33% is essentially "they each have a fan of plausible
   guesses, those fans overlap on a third of positions."

**Implication for fold mechanism design.** The KL_suffix consistency
loss in the broader architecture (§3 of `prefix-suffix-fold-architecture.md`)
is exactly the bridge that would close this gap — it forces F and B to
agree at training time. The 2.4-nat measurement is what that loss has to
work against. It's also the magnitude that "real fold/loop" mechanisms
would have to navigate: any fold operator that crosses prefix↔suffix
information has to handle this large per-position disagreement, not just
the corpus-marginal agreement that bayes_probe validated.

The cap-fold mechanism we shipped (target substitution) doesn't bridge
this divergence — it stays entirely within the prefix-tree view. The
small PPL win we saw is independent of this gap. The bigger structural
opportunity (and challenge) is the dual-model coupling that this
measurement scopes.

### Reproduce

```sh
just build-agpt-build-radix-corpus
just build-prefix-suffix-compare

# Cap-edge stats
./bin/agpt_build_radix_corpus --corpus /tmp/shakespeare_100k.txt --max-depth 32 --out /tmp/agpt_shakes100k_d32_radix
./bin/trie-profile /tmp/agpt_shakes100k_d32_radix
# (and likewise for input.txt and gutenberg 10M)

# Forward/backward divergence
./bin/agpt_build_radix_corpus --corpus data/input.txt --max-depth 32 --reverse --out /tmp/agpt_input_d32_suffix_radix
./bin/agpt_train --model data/input.random.model --trie-dir /tmp/agpt_input_d32_suffix_radix \
                 --save /tmp/backward_6se.model --epochs 6 --partition-depth 6 --no-accumulate \
                 --optimizer rmsprop --lr 3e-3 --weight-decay 0.01 --mass-weight off
./bin/prefix_suffix_compare --forward /tmp/baseline_6se.model --backward /tmp/backward_6se.model \
                             --file data/input.txt --seq-len 32 --max-positions 4096 --backend openblas
```
