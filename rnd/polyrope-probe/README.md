---
title: PolyRoPE probe (window form)
kind: experiment
status: concluded
outcome: inconclusive
question: >-
  The user's PolyRoPE design in its window form: if every symbol carries a damped harmonic
  signature of where it occurred in the last 1,024 characters, can a small model read it for
  prediction, and how much of exact attention's gain over the same reach does it recover?
answer: >-
  Not tested, because the controls left nothing to recover. At d64/L2, exact attention gains
  only 0.0025 nats/char from layer-2 offsets 128-1,023 on PG-19 books (0.006-0.009 on Tiny
  Shakespeare), against 0.0619 (0.03) from offsets 16-127. That is about one seed's noise, so no
  channel reading those offsets could show reach at this size, and the channel arms (built and
  checked) were not run. This tests only an in-context reading of PolyRoPE. The design's
  original notion, signatures of a token's positions across the whole corpus, is not tested
  here.
opened: 2026-11-12
updated: 2026-11-13
code: main
eval: legacy
family: attention
tags: [position, attention, context-length, polyrope, harmonic]
headline:
- label: exact attention, reach 1,024 vs 128, PG-19 (imm's 20M split), seed 42
  metric: diagnostic held-out NLL gain from layer-2 offsets 128-1,023, nats/char
  value: 0.0025
  run: 20261113T032614-pg19-r1024-exact-s42
- label: exact attention, reach 128 vs the 16-window, PG-19, seed 42
  metric: diagnostic held-out NLL gain from layer-2 offsets 16-127, nats/char
  value: 0.0619
  run: 20261113T032614-pg19-r128-exact-s42
- label: exact attention, reach 1,024 vs 128, Tiny Shakespeare
  metric: diagnostic held-out NLL gain from offsets 128-1,023, nats/char (s42 vs the 4-seed reach-128 mean)
  value: 0.0087
  run: 20261112T100835-r1024-exact-s42
related: [far-channel-probe, superposition-moment-order, harmonic-bias-prototype, superposed-tail-rope]
---

# polyrope-probe

**Status:** concluded (2026-11-13). Controls only: exact attention at reaches
128 and 1,024 and the 16-window, on Tiny Shakespeare and on PG-19. The
channel arms were not run, because the controls showed there was nothing
beyond offset 128 for them to recover at this size.

## What this probe is, and what it isn't

The user's design document ("PolyRoPE: Learning Semantics from Corpus
Structure", 2026-11-12, passed on by @mobius; not in this repo) proposes
giving each vocabulary token a spectral signature of **all its positions in
the corpus**,

```
H_v(ω) = Σ_{p : C[p] = v} e^{iωp},
```

so that pair structure comes out of products: conj(H_a(ω)) H_b(ω) holds every
displacement between an occurrence of a and one of b, and its inverse
transform is their cross-correlation R_ab(d). Its principle: don't make the
network rediscover structure that can be extracted directly. The document's
programme runs from checking the mathematics (Phases 1–3) through signatures
as enriched embeddings in ordinary training (Phase 5) to training on the
vocabulary of signatures alone (Phases 6–7).

| Form | Where the structure comes from | Tested? |
|---|---|---|
| Corpus form (§3.2, Phases 1–3) | H_v over every occurrence in the corpus; pairs from cross-spectra | @mobius, 2026-11-12: character bigrams recovered from K random frequencies of 4.2M PG-19 characters have a median error of 20% at K = 1,024 and 10% at K = 16,384, which is 330× the bigram table's storage. That fails Phase 1's criterion (useful pairs in less storage). |
| Window form | H_v over the last W characters, at each position | This probe. Built and checked; controls run; channel arms not run. |
| Enriched embedding (Phase 5) | E_poly(v) = Combine(E(v), H_v) in ordinary training | Not tested. |
| Structural queries, vocabulary-only training (Phases 6–7) | training on (v, H_v) entries | Not tested. |

After mobius's corpus-form result, the user chose a window instead
(2026-11-12), and this probe tests that. The window form is a far channel
with one-hot content. So what has actually run is the **channel approach**:
`rnd/far-channel-probe` (learned content, reach 128, 65% of exact
attention's gain recovered), plus this probe's controls. Those controls cap
what any in-context reader can gain at this size, whatever its readout. They
say nothing about the corpus-level notion, which supplies statistics from
text the model never sees in its context. That makes it the same kind of
resource as the count prior.

**Why the corpus form fails, and what would test the notion.** Give attention
the signatures as features with RoPE's rotation by the displacement Δ. A query
on a at t and a key on b at t − Δ then score

```
Re Σ_ω c_ω conj(H_a(ω)) H_b(ω) e^{iωΔ} = Σ_d R_ab(d) κ(d + Δ),   κ(x) = Σ_ω c_ω cos(ωx).
```

That is the corpus count of b at Δ before a, blurred by a kernel κ. With K
frequencies, κ's sidelobes also collect the pairs at every other displacement
up to the corpus length. That is mobius's error, and the decoding limit in
`notes/seq-len-extension/superposition-softmax-limit.md` §4. For characters
the spectral compression isn't needed: the exact R_ab(d) for d < 1,024 is
96² × 1,024 ≈ 9.4M counts on PG-19. So the notion itself, corpus displacement
structure supplied to attention, can be tested exactly with counted tables.
mobius has proposed distance co-occurrence counted from the books (mail to
@agpt, 2026-11-12). The compression matters only at word-level vocabularies.
By the results below, such a bias would matter mostly at offsets below 128.
That is where the context carries information, and there it would act as a
data resource, as the count prior does.

## Design

`src/tools/agpt_polyrope_probe.py` (`af81726`; `fd592ad` adds the PG-19
options). It is the far-channel probe's harness: no trie, and a plain
minibatch-Adam transformer with the AGPT d64/L2 layout (4 heads, d_ff 256,
pre-LN, RoPE with interleaved pairs, base 10000, ReLU, untied head). Layer 1
attends over offsets 0–15. The arms differ only in layer 2:

| Arm | Layer 2 | Params (Shakespeare) | Run? |
|---|---|---:|---|
| `w16` | softmax over offsets 0–15 (baseline) | 108,481 | yes |
| `exact` | softmax over offsets 0 .. R−1 (upper bound); R = 128 or 1,024 | 108,481 | yes |
| `hidden` | 16-window, plus a far channel over 16 .. R−1 with keys from the layer input (far-channel probe's `cq`) | 120,965 | no |
| `char` | the same far channel, keys and values from a learned 16-dim embedding of the character only | 120,037 | no |
| `unit` | the same, one entry per complete unit at its last character | 184,533 | no |

**The far channel** (the arms not run), per head, for the query at t and an
entry at p, Δ = t − p, 16 ≤ Δ < R:

```
a[t, p] = φ(qf_t) · φ(k_p) · w_t(Δ),     w_t(Δ) = |Σ_j b_{t,j} λ_j^(Δ−16)|²,
λ_j = e^{−1/P_j} e^{2πi/P_j},  P = 8, 16, ..., 1,024   (8 constant-Q atoms)
far_t = Σ_p a[t, p] v_p / Σ_p a[t, p]
```

For `char` and `unit` the content depends only on the symbol s_p, so the
channel's exact recurrent state can be kept per symbol type:
H_v(μ) = Σ over entries p of v of μ^(t−p−16), for the 64 products
μ = λ_j conj(λ_l). That is the user's windowed signature, damped. The state
is 64 reals per symbol type, fixed whatever the reach: 65 × 64 ≈ 4.2k for
Shakespeare's characters.

**Units** (the `unit` arm) are cut as mobius's UnitCuts cuts them (ported in
`unit_ends`): at the boundaries between letters, punctuation and whitespace,
with an apostrophe after a letter counted as a letter and a single space
joining the unit after it. The 4,095 commonest training units get their own
ids. On Shakespeare's training text there are 15,800 distinct units, 270,701
in all, 3.91 characters on average, and the ids cover 93.6% of them.

## Protocol

- **Training:** Adam, lr 1.5e-3, 500 warmup steps, then a cosine decay to
  10%. Each step is 4 blocks of 2,048 characters (8,192 targets) with every
  position scored, for 30k steps (246M targets). On PG-19, imm's 20M split
  (800 pieces of 25,000 characters from as many train books), blocks stay
  inside one piece (`--piece-chars 25000`). That is 232 passes over
  Shakespeare's 1.06M training characters and 12 over PG-19's 20M.
- **Evaluation (diagnostic, not lm-eval):** each held-out target is scored
  once, with at least 1,040 preceding characters where its chunk has them,
  whatever the arm's reach, so every run is scored on the same targets.
  Shakespeare: the canonical split's 10 chunks of 5,576 characters (55,750
  targets). PG-19: 50 pieces of 5,000 characters from as many test books
  (249,950). NLL is also reported by the target's position in its chunk: 1–30,
  31–1,038, and 1,039 to the end (the main comparison: 45,370 targets on
  Shakespeare, 198,050 on PG-19).
- **Order of runs:** controls first. The channel arms were to run only if
  exact attention at reach 1,024 clearly beat reach 128 (the far-channel
  probe's rule: a control that can't show a gain makes the test
  meaningless).

**Checks** (at start-up of every run):
- Changing inputs more than 1,038 characters back changes no arm's output
  (max change 0.0). Changing offsets 31–1,038 leaves `w16` unchanged (0.0)
  and changes `exact` at reach 1,024.
- The per-symbol recurrence H_v(μ) matches the far channel's parallel form to
  a relative 4e-7 (float32). Beyond the reach, the far arms' outputs change by
  at most 1.5e-7 in float32 and by exactly 0 in float64 (2026-11-12 CPU smoke
  tests).
- The byte encoder used since `fd592ad` gives the same ids as the
  per-character dictionary on both corpora.

## Results

Diagnostic held-out NLL in nats per character, from each run's
`metrics.json` or `rescore/*/eval.json`. One seed (42) for this probe's runs.

### Tiny Shakespeare

| Model | Runs | 1–30 | 31–1,038 | 1,039–end |
|---|---|---:|---:|---:|
| `w16` | this probe, s42 (`20261112T100835-r1024-w16-s42`) | 1.3381 | 1.3047 | 1.3519 |
| `exact`, reach 1,024 | this probe, s42 (`20261112T100835-r1024-exact-s42`) | 1.3391 | 1.2772 | **1.3137** |
| `w16` | far-channel probe checkpoints, s42–45, rescored here | | | 1.3515 mean (1.3440–1.3611) |
| `exact`, reach 128 | far-channel probe checkpoints, s42–45, rescored here | | | 1.3224 mean (1.3200–1.3250) |

The reach-128 references come from the far-channel probe's trainer (16
blocks of 512 characters per step), rescored on this probe's targets
(`rescore/far-channel-*`). The two trainers agree on `w16`: this probe's
1.3519 against a mean of 1.3515. Going from offsets 16–127 to 128–1,023 gains
0.0063 (s42 against s42) to 0.0087 (against the 4-seed mean). Offsets 16–127
gain 0.029 (4-seed means).

### PG-19 (imm's 20M split)

| Model | Run | 1–30 | 31–1,038 | 1,039–end | Train loss |
|---|---|---:|---:|---:|---:|
| `w16` | `20261113T032614-pg19-r1024-w16-s42` | 1.6399 | 1.4662 | 1.4893 | 1.4356 |
| `exact`, reach 128 | `20261113T032614-pg19-r128-exact-s42` | 1.6763 | 1.4052 | 1.4274 | 1.3765 |
| `exact`, reach 1,024 | `20261113T032614-pg19-r1024-exact-s42` | 1.6868 | 1.4066 | **1.4249** | 1.3764 |

Offsets 16–127 gain 0.0619, twice Shakespeare's. Offsets 128–1,023 gain
0.0025, and the gap held steady over the last 10k steps (reach 1,024 minus
reach 128 at 20k, 22k, …, 30k: −0.0024, +0.0001, −0.0021, −0.0019, −0.0027,
−0.0025). All three PG-19 runs took 25.8–25.9 ms per step, about 13 minutes.

### Cutting the reach-1,024 model's attention at evaluation

The same weights, with layer 2's span cut (CPU, `--eval-only`, `rescore/exact-r1024-s42-at*`
and `rescore/pg19-exact-r1024-s42-at*`), on targets 1,039 to the end:

| Layer-2 span at evaluation | Shakespeare | PG-19 |
|---|---:|---:|
| 1,024 (as trained) | 1.3137 | 1.4249 |
| 512 | 1.3158 (+0.0021) | 1.4259 (+0.0010) |
| 256 | 1.3213 (+0.0076) | 1.4287 (+0.0038) |
| 128 | 1.3298 (+0.0161) | 1.4336 (+0.0087) |

Each octave of distance is worth less than the one before: on Shakespeare
0.0085 (offsets 128–255), 0.0055 (256–511) and 0.0021 (512–1,023); on PG-19
0.0049, 0.0028 and 0.0010. At 128 the cut costs more than the trained gap (0.0161 against 0.006–0.009 on Shakespeare,
0.0087 against 0.0025 on PG-19), because the cut model was trained to rely
on keys it then loses.

## Reading

- **At this size, almost nothing past offset 128 is used.** Exact softmax
  over the same offsets, the probe's upper bound, gains 0.0025–0.009 nats/char
  from offsets 128–1,023. The trained-at-1,024 model uses every octave, but
  with value falling off with distance.
- **Books make near context worth more and far context worth less.** On
  PG-19, offsets 16–127 are worth twice what they are on Shakespeare, and
  128–1,023 less than half. This fits Khandelwal et al. (2018, "Sharp
  Nearby, Fuzzy Far Away"): LSTM language models are sensitive to word order
  only in the last ~20 tokens and use the distant past only as a rough sense
  of topic.
- **So the channel arms could not have shown reach.** Their best case beyond
  offset 128 is a fraction of 0.0025–0.009, at or below the run-to-run
  spread of a single seed (`w16`'s 4 seeds span 0.017 on Shakespeare). The
  user chose (2026-11-13) to measure the headroom on PG-19 before spending
  GPU time on them, and it came out smaller still.
- **This bounds the window form, not PolyRoPE.** See "What this probe is,
  and what it isn't" above.

## Caveats

- **Diagnostic evaluation,** not lm-eval byte PPL. It is the same for every
  run, but its absolute values aren't comparable with canonical PPL.
- **One size, one seed.** d64/L2 (about 110k parameters) at 30k steps. The
  PG-19 curves were still falling at 30k (12 passes), so a longer or larger
  run may use far context more. Whether the headroom grows with model size
  is the open question for the window form.
- **The Shakespeare reach-128 comparison crosses two trainers** (the
  far-channel probe's 512-character blocks and this probe's 2,048). The PG-19
  comparison doesn't: all three runs share this probe's trainer, seed and
  batches.
- **The short-context bucket** (targets 1–30 of each chunk) reads worse for
  the long-reach models on PG-19 (1.6868 and 1.6763 against 1.6399 for
  `w16`), on 1,500 targets. It doesn't bear on the comparison.

## Files

- `run_arms.sh`: Shakespeare arms at reach 1,024 (used for the controls).
- `run_pg19_controls.sh`: the PG-19 controls and the CPU cut rescoring.
- `rescore_far_channel.sh`: rescoring the far-channel probe's `w16` and
  `exact` checkpoints on this probe's targets.
- Run directories hold `args.json` (including the start-up checks and git
  SHA) and `metrics.json` (the curve and final numbers). `model.pt` and
  `train.log` are not committed.

All heavy runs went through @air's slot queue (`tools/slots` in AIR), one
heavy job at a time, at nice 19 with idle I/O priority and 2 CPU threads.

## Next, if wanted

- **The original notion, exactly:** corpus displacement counts R_ab(d) from
  the books, given to attention as a bias by displacement. This is mobius's
  proposed distance co-occurrence; see above.
- **The window form at a larger size:** the same exact-attention controls at
  reach 128 and 1,024 on a model large enough that offsets 128–1,023 are worth
  more than a seed's noise. Only then can the channel arms show reach.
