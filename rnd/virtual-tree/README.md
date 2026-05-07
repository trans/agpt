# Virtual tree: per-cap multi-position composite distributions

**Status**: negative result at expansion_depth=3, alpha=0.5. Virtual-tree
regresses held-out PPL by 16% vs AGPT baseline.

**Branch**: `dual-model-fold`. Tools: `bin/agpt_build_virtual_tree`
(builder), `bin/agpt_inspect_virtual_tree` (CPU validator),
`bin/agpt_train --virtual-tree` (consumer kernel).

## Hypothesis

At d=32 Shakespeare 1M, ~98% of training events are one-hot
(cap-edge intermediates + cap endpoints) and only ~2% are KL on real
distributions (branching internal endpoints). AGPT mathematically
reduces to mini-batch SGD with structured batches plus a tiny
"distribution-target" tail. Cap edges in particular are deterministic
unary chains; their one-hot targets seemed to carry no signal beyond
what SGD on real corpus would.

Predicted: replacing the first 3 tunnel positions of each cap with
composite distributions (length-weighted mixtures of shifted-prefix
walks from root) would scale the KL-event share from ~2% to ~12%, and
the model — given richer targets where there was previously no signal
— should improve on held-out PPL.

## Result

| Variant | Final train loss | Held-out PPL @ seq=32, 8192 pos |
|---|---:|---:|
| AGPT baseline (no fold, no vtree) | 1.5048 | **4.80** |
| Single-shift cap-fold (post-cap endpoint) | 1.5043 | 4.81 |
| **Virtual-tree (+3 tunnel positions, α=0.5)** | 1.9536 | **5.57 (+16% worse)** |

Wall-time per SE: ~205 s for all three. No measurable runtime overhead
from the vtree side-table lookup.

## Why it fails

Two interacting reasons:

1. **The cap one-hots ARE load-bearing.** Counter to the hypothesis,
   removing them (replacing with softer composite distributions)
   hurts. Held-out PPL rewards sharp predictions at the actual corpus
   continuations. The composite is marginalized over many corpus
   contexts (shifted-prefix walks) and is inherently *softer* than
   the cap's deterministic one-hot. Training to KL-match the composite
   pulls the model toward broader hedging exactly where the corpus
   continuations are deterministic.

2. **Position 0 conflicts with the parent's branching distribution.**
   For a cap at depth `h`, position 0 of its edge is the cap-head char.
   This char is *already* trained via the parent branching node's
   endpoint query at depth `h−1` — the parent's counts include the
   cap-head among its children with one specific weight. The vtree
   composite at cap-edge position 0 trains a *different* distribution
   (shifted-context marginal) for the same prediction, directly
   conflicting with the parent's already-learned distribution. We're
   essentially training two different answers for the same forward-pass
   position.

Both effects combine to drag PPL up. The training loss numbers reflect
this: vtree's 1.95 is the cap-position KL on composites (entropy floor
~1.37 nats, so ~0.58 nats of fitting error) — model fits the composites
fine, it just isn't the right thing to fit for PPL.

## Implication for AGPT-vs-SGD

The user's prior framing — "at d=32, AGPT is mostly SGD over caps with
a thin AG cache" — is empirically supported, but with a load-bearing
twist: the cap one-hot SGD events are not noise, they're the bulk of
useful gradient signal. Replacing them with corpus-derived
distributions hurts more than the additional KL events help.

This doesn't kill AGPT as a framework — it preserves K/V sharing across
siblings, structured batching by 6-gram subtrees, and the small but
real ~2% KL events at branching endpoints. But it does say:
**aggressive enrichment of cap targets beyond their natural one-hot is
not free, and at +3 with α=0.5 it's net-negative.**

## Possible follow-ups (untested)

- **expansion_depth=1**: only override cap-edge position 0. Avoids
  positions 1, 2 entirely. (But position 0 is the position with the
  parent-overlap problem, so this might actually be the worst variant.)
- **Skip position 0, override positions 1, 2 only**: avoids the
  parent-overlap conflict; keeps the composite-target experiment for
  positions where there's truly no other signal source.
- **α=0.1 or hard-cutoff to shift=1 only**: makes composites much
  sharper, closer to a single-context prediction. Reduces softening.
- **Mixture target: β × one-hot + (1−β) × composite**: keep most of the
  sharp signal, add some composite info. β=0.7 or 0.8 starts close to
  baseline.

## Reproduce

```sh
# Build trie, side-tables, and trainer (one-time)
just build-agpt-train build-agpt-build-fold-table build-agpt-build-virtual-tree

bin/agpt_build_index --corpus data/input.txt --max-depth 32
bin/agpt_build_radix --leveled /tmp/agpt_input_d32

bin/agpt_build_fold_table --trie /tmp/agpt_input_d32_radix \
  --out /tmp/fold_d32_legacy.bin

bin/agpt_build_virtual_tree --trie /tmp/agpt_input_d32_radix \
  --out /tmp/agpt_vtree_d32_e3.bin \
  --expansion-depth 3 --shift-min 1 --mass-min 2 --alpha 0.5

# Train (6 SE per arm; ~21 min each)
for arm in baseline fold vtree; do
  case $arm in
    baseline) extra="" ;;
    fold)     extra="--fold-table /tmp/fold_d32_legacy.bin" ;;
    vtree)    extra="--virtual-tree /tmp/agpt_vtree_d32_e3.bin" ;;
  esac
  bin/agpt_train --model data/input.random.model \
    --trie-dir /tmp/agpt_input_d32_radix \
    --save /tmp/cap_${arm}_6se.model \
    --epochs 6 --partition-depth 6 --no-accumulate \
    --optimizer rmsprop --rmsprop-beta 0.999 --lr 3e-3 \
    --weight-decay 0.01 --entropy-lambda 0 --mass-weight off $extra
done

# PPL
for arm in baseline fold vtree; do
  bin/perplexity --model /tmp/cap_${arm}_6se.model --file data/input.txt \
    --seq-len 32 --backend openblas --max-positions 8192
done
```

## Artifacts (regenerable)

- `/tmp/agpt_vtree_d32_e3.bin` (410 MB, expansion=3 α=0.5 mass-min=2)
- `/tmp/fold_d32_legacy.bin` (56 MB, single-shift baseline)
- `/tmp/cap_{baseline,fold,vtree}_6se.model` (424 KB each)
- `/tmp/train_{baseline,fold,vtree}_6se.log`
