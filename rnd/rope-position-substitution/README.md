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

## Interpretation

**Two findings, one consistent picture.**

### Finding 1: monotonic substitution is null

A standard transformer that depends on RoPE as a literal sequence
coordinate should *break* under mass-substitution. Mass values are
4-5 orders of magnitude larger than depth values; the RoPE rotation
angles at those positions sample the cos/sin curves at vastly
different frequencies. If the model were locating tokens by their
RoPE-encoded position (e.g. for in-context copying), the substitution
should hurt by a lot.

It didn't. Mass and log-mass produce statistically-equivalent PPL to
depth on both corpora. The model adapts.

### Finding 2: removing RoPE entirely costs ~5 PPL

When pos=0 for all queries (RoPE rotation becomes identity, model
loses positional signal entirely), PPL jumps from ~8.4 to ~13.4 on
Shakespeare. Highly significant (paired t ≈ 9.9 vs depth). RoPE is
contributing real work.

### Together

The model uses RoPE for **monotonic per-query differentiation** —
distinguishing "this query is in a position that should be treated
one way" from "this other query should be treated differently" — but
doesn't care about the *absolute scale* of that differentiation. Any
monotonic injection of position-dependent rotation works equally well
(depth, mass, log-mass). No rotation at all loses the differentiation
and the model collapses by 5+ PPL.

This is *consistent with* the trust-signature theory:
- RoPE here isn't carrying "absolute sequence position" because there
  isn't one — every query has its own context window with its own
  arbitrary trie path.
- What RoPE *is* carrying: a monotonic ordering signal across
  positions within a query. The model learns that
  "rotation-angle-X means deep / rare / less-trustworthy-as-statistic"
  and "rotation-angle-Y means shallow / common / more-trustworthy."
- The absolute spacing along the angle axis doesn't matter much — what
  matters is that two queries at "different depths" have *different*
  rotation angles in a monotonic way.

What *would* further test the theory: a non-monotonic substitution
(random permutation of depth → angle). If the model still learns,
RoPE is being used as essentially a per-node hash, not even an
ordering signal. If it falls apart, RoPE *does* need monotonicity,
and the role is "ordered differentiation," not "arbitrary
differentiation."

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
