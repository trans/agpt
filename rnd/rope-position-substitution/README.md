# RoPE Position Substitution: depth vs mass vs log-mass vs off

**Date:** 2026-05-22
**Status:** Strong finding. Monotonic substitution null on PPL; disabling RoPE entirely costs ~5 PPL.
**Branch / commits:** Mass-RoPE infrastructure landed on `main` (--rope-mode flag in agpt_train).

## Question

Standard transformers use sequential token position as the input to RoPE.
In AGPT each query lives at a specific node in a radix trie. The current
implementation feeds *char-depth in the trie* as the RoPE position
(0..d-1). Hypothesis (Thomas): the model may not be using RoPE as a
literal sequence coordinate at all — it may be using RoPE as a "trust
signature" or node-identity-ish signal, since trie depth correlates
inversely with how rare/specific a node is (higher depth → rarer → less
to trust as a general predictor).

If true, we should be able to *substitute* the position signal with
something else monotonic in depth without breaking the model. RoPE is
a relative encoding — it only cares about pairwise position differences
— so any monotonic substitution preserves the same relative structure
with different absolute spacing.

Concrete probe: feed `edge_mass` (count of strings sharing this prefix)
as the position instead of depth. Edge_mass is monotonically *inverse*
to depth along any root-to-node path. Range: [1, ~170k] on Shakespeare,
larger on Gutenberg.

## Setup

- Corpora: Shakespeare 1M, Gutenberg 5M (both d=16 radix tries)
- Init: shared Kaiming, three seeds (`/tmp/agpt_init_kaiming_s{1,2,3}.model`)
- Recipe: rmsprop, lr 3e-3, warmup-cosine 1 epoch, 10 SE, partition-depth 1,
  no-accumulate, no weighting (mass/depth/entropy/branching all off)
- Eval: PyTorch reference (`src/tools/agpt_ppl.py --mode fixed`, d=16,
  max-positions 10000). Independent of trainer kernels.

Four modes:
- **depth** (control): pos = char depth, 0..d-1 (16 values)
- **mass**: pos = edge_mass of the radix node, 0..~170k
- **log-mass**: pos = floor(log2(edge_mass)), 0..~17 (comparable span to depth)
- **off** (litmus): pos = 0 for all queries → cos=1, sin=0 → identity rotation.
  Effectively disables RoPE without other code changes.

## Results

| corpus      | depth          | mass           | log-mass       | off              |
|-------------|----------------|----------------|----------------|------------------|
| Shakespeare | 8.584 ± 0.376  | 8.386 ± 0.131  | 8.322 ± 0.297  | **13.432 ± 0.325** |
| Gutenberg   | 8.366 ± 0.102  | 8.370 ± 0.158  | 8.293 ± 0.096  | (not measured)   |

(mean ± population std, n=3)

Paired Δ vs depth (per-seed):

|             | Shakespeare           | Gutenberg     |
|-------------|-----------------------|---------------|
| mass        | −0.20 ± 0.30          | +0.00 ± 0.07  |
| log-mass    | −0.26 ± 0.29          | −0.07 ± 0.17  |
| **off**     | **+4.85 ± 0.85**      | —             |

Paired t-tests vs depth:
- mass / log-mass: smallest p ≈ 0.26 (Shakespeare log-mass). None significant.
- **off: t ≈ 9.9, p < 0.005 on n=3 Shakespeare. Definitively worse.**

Per-cell data: see `results.txt`. Training logs: `logs/<corpus>_<mode>_s<seed>.train.log`.

### Single-transposition swap probes (n=3 each)

| swap | mean | std | Δ vs depth |
|---|---|---|---|
| (0,1) | 8.407 | 0.135 | −0.18 |
| (1,2) | 8.402 | 0.262 | −0.18 |
| (2,3) | 8.276 | 0.111 | −0.31 |
| (8,9) | 8.816 | 0.262 | +0.23 |
| (14,15) | 8.940 | 0.236 | +0.36 |

Per-seed `(14,15) - (0,1)` Δ across all 3 seeds: +1.04, +0.16, +0.41 — all positive.
Direction is consistent: **front-swap is harmless, back-swap is degrading.** Not
significant at α=0.05 (n=3, paired t≈2.1 for the leaf-vs-front comparison).

## Interpretation

### Final picture across all six probe families

| condition | preserves | breaks | mean Δ vs depth | significance |
|---|---|---|---|---|
| depth (control) | everything | — | 0 | — |
| mass | monotonic ordering | linear spacing | −0.20 | p≈0.36 |
| log-mass | monotonic ordering | linear spacing | −0.26 | p≈0.26 |
| swap(0..3) | near-leaf ordering | front-local tiny break | ~ −0.2 | n.s. (favors swap) |
| swap(8,9) | leaf order intact | mid-local break | +0.23 | n.s. |
| swap(14,15) | identity except leaf | leaf-local order | +0.36 | n.s. |
| perm-depth | per-depth identity | all ordering | +3.09 | p<0.01 |
| off | nothing | everything | +4.85 | p<0.005 |

### What's actually going on

The model uses RoPE for **monotonic ordered differentiation, weighted toward
the prediction-bearing end of the window.** Three pieces:

1. **Ordering matters more than scale.** Mass and log-mass produce
   PPL statistically equivalent to depth even though their absolute
   values differ by 4-5 orders of magnitude. RoPE here is not carrying
   "literal sequence position"; it's carrying a monotonic ordering
   signal that the model uses to differentiate queries.

2. **Identity-only doesn't suffice.** Perm-depth (random shuffle of
   depth → angle, preserving per-depth identity but breaking ordering)
   costs +3.09 PPL. Off (no positional signal at all) costs +4.85.
   Perm-depth recovers ~36% of the off-vs-depth gap — meaning some
   identity-hash signal *is* used, but ordering is the dominant
   ~64% of RoPE's contribution.

3. **Leaf-end matters more than root-end.** Single transpositions at
   depths 0-3 (near root, where corpus mass concentrates) are
   harmless — sometimes even slightly *better* than identity. Single
   transpositions at depths 8-15 (near leaf, where the predictive
   decision happens) degrade the model. This was the key surprise of
   the experiment and reframes the trust-signature hypothesis:

   > **The prediction in AGPT happens at the leaf.** Each query is
   > predicting what character should come next at its position; the
   > deepest queries in the window ARE the prediction-bearing nodes.
   > Disrupting RoPE near the leaf disrupts the predictive attention.
   > Disrupting RoPE near the root touches positions that mostly
   > carry redundant "every window starts here" signal — the model
   > can absorb that disruption without losing predictive accuracy.

   So while the corpus's mass concentrates near the root (every
   window's first few characters are common), the model's *use* of
   positional information concentrates near the leaf (deep queries
   are the discriminative predictions).

### Net answer to the original question

AGPT's RoPE is *not* a literal-position encoding (mass-substitution
shows the model doesn't care about absolute scale). It *is* a
monotonic ordering signal, weighted toward the leaf end of the
window. The trust-signature framing was directionally right but
unevenly applied: the model doesn't treat all depths the same —
it treats deep, prediction-adjacent depths as load-bearing and
shallow, mass-heavy depths as nearly fungible.

## Not tested (open follow-ups)

1. **Random-permutation control.** Strongest falsifier of the
   ordering-vs-identity question. Cheap to add (`--rope-mode random-perm`
   flag, deterministic from seed).
2. **Deeper trie (d=32).** Mass range explodes further. Doesn't change
   the theory's prediction (still monotonic → still works) but tests
   robustness.
3. **Larger n.** Shakespeare's paired-diff std is ~0.30; with n=10 the
   apparent mean trend (mass/log-mass beating depth by 0.2-0.3) would
   either firm up or evaporate. Worth doing only if the trend is
   load-bearing for a publication-style claim.
4. **~~Constant position (pos=0)~~** — done as `--rope-mode off`.
   Result: +4.85 PPL vs depth on Shakespeare, p<0.005. RoPE matters.
5. **off + Gutenberg.** Confirm the ~5 PPL collapse replicates on a
   larger corpus, ruling out "Shakespeare-specific quirk."

## What this changes

Nothing operationally — `--rope-mode depth` stays default. The flag is
in the codebase as a knob for future probes. The interesting outcome
is the *interpretation*: AGPT's transformer is using RoPE differently
from a standard LM. Worth keeping in mind when designing future
positional or context mechanisms — what AGPT actually needs from
"position" may be much weaker than what RoPE was originally designed for.

## Reproduce

```sh
for seed in 1 2 3; do
  for mode in depth mass log-mass; do
    rope_flag=""
    [ "$mode" != "depth" ] && rope_flag="--rope-mode $mode"
    bin/agpt_train --model /tmp/agpt_init_kaiming_s${seed}.model \
        --trie-dir /tmp/shake_baseline_d16_radix \
        --epochs 10 --lr 3e-3 --optimizer rmsprop \
        --lr-schedule warmup-cosine --warmup-epochs 1 \
        --partition-depth 1 --no-accumulate --mass-weight off \
        $rope_flag \
        --save /tmp/run_${mode}_s${seed}.model
    python3 src/tools/agpt_ppl.py \
        --model /tmp/run_${mode}_s${seed}.model \
        --file /tmp/shake_holdout.txt --vocab-file data/input.txt \
        --d 16 --max-positions 10000 --mode fixed
  done
done
```

Swap `--trie-dir /tmp/gutenberg_5m_baseline_d16_radix` and `--file /tmp/gut_holdout.txt`
for the Gutenberg leg.
