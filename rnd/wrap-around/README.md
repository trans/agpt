# Wrap-around corpus synthesis

Sample arbitrary-length token sequences from a depth-D radix trie by walking
root→leaf via mass-weighted child picks, then bridging back to the root via a
token sampled from the leaf's endpoint distribution and continuing. Trains a
window-mode SGD model on the synthesized corpus to test whether a depth-D
trie carries the predictive content needed for `seq_len > D` training.

Result: at d=32, the synthesized corpus is sufficient — held-out PPL on the
real corpus, multi-seed mean PPL = 7.04 (range 6.93–7.13 across 4 seeds),
matches the SGD seq=128 ceiling. See `logs/multi-seed-sweep.txt`.

## Bridge double-emit bug fix

The original synth printed each wrap's bridge token AND the first token
of the next root-walk's edge — but `pick_root_child(seed)` already finds
a root child whose first edge token equals the seed, so the bridge token
was being emitted twice. Result: every wrap glued a duplicate character
across the boundary, producing artifacts like "RRWICK" (WARWICK with
'R' doubled), "allisters", "hereen", "like  volume" (double space).

Fixed by tracking whether `pick_root_child` matched on the seed and
skipping the duplicate first-token emit when it did. Effects:

- Double-spaces in 1M synth: 5 (was ~4624 — 1000× drop).
- Lowercase double-letter rate matches real corpus: 1.68% (real: 1.70%);
  was 3.6% pre-fix.
- Avg path length per wrap: 32.0 (was 33.0 — the spurious +1 is the
  duplicated bridge token).
- PPL: 7.13 (was 7.17, ~noise floor).

## Negative results: --space-align and --space-cut

Two attempts to further "clean up" wrap boundaries, both shown to not
help once the double-emit bug was fixed.

**`--space-align`** (top-k filter on the bridge sampling): in this trie,
99.99% of leaves have exactly 1 entry in their endpoint counts (mass-1
prefixes). The "distribution" is degenerate, so forcing space when
present in top-k is identical to the natural mass-weighted pick. The
flag is a semantic no-op; any apparent PPL difference comes only from
shifted RNG trajectory.

**`--space-cut`** (back up within the leaf's edge to the last space):
98.7% of leaves have a space somewhere in their 23-char edge, so
emitting only up to the last space and wrapping there produces a
visibly cleaner synth (reads like broken Shakespeare instead of
glue-soup). But:

| config       | seeds        | PPL mean | range       |
|--------------|--------------|---------:|-------------|
| baseline     | 42,44,46,48  | **7.04** | 6.93–7.13   |
| `--space-cut`| 42,44,46     |     7.16 | 7.00–7.37   |

space-cut is 0.13 PPL **worse** on average, with wider variance.
Train loss is consistently lower (~1.91 vs 2.00) — the cleaner synth
is easier to fit, but generalizes worse to real text.

Mechanism (most likely):

1. space-cut discards the post-last-space tail of each leaf
   (~5 chars/wrap × ~360k wraps/10M tokens = ~1.7M chars of real
   corpus content lost per 10M synth).
2. Always-space-then-non-space-after-wrap teaches a synth-specific
   transition pattern that doesn't match real text.
3. Wrap-glue noise was already being absorbed by the model — gluings
   looked ugly to humans but didn't hurt real-corpus prediction.

Both flags preserved in `bin/synth_wrap_corpus` for future probes but
are off by default. See `logs/multi-seed-sweep.txt`,
`logs/synth-d32-10M-space*.log`, `logs/synth-d32-10M-FIXED-*.log`.

## Artifacts (not in git — regenerate as needed)

| File                                    | Size  | Regen                                |
|-----------------------------------------|-------|--------------------------------------|
| `data/synth_wrap_d16.txt`               | 1.1 M | step 1 below (depth=16, 1M tokens)   |
| `data/synth_wrap_d32.txt`               | 1.1 M | step 1 below (depth=32, 1M tokens)   |
| `data/synth_wrap_d32_10M.txt`           |  10 M | step 1 below (depth=32, 10M tokens)  |
| `rnd/wrap-around/synth-d32-10M-seq128-10k.model` | 425 K | step 2 below              |

## Regeneration

Prereqs: a built radix trie at the desired depth. Build one with:
```sh
bin/agpt_build_index --corpus data/input.txt --max-depth 32
bin/agpt_build_radix --leveled /tmp/agpt_input_d32
# → /tmp/agpt_input_d32_radix/
```

### Step 1: synthesize the wrap-around corpus

```sh
# 1M tokens, d=16 (the smaller experiment)
bin/synth_wrap_corpus --trie-dir /tmp/agpt_input_d16_radix \
  --vocab-text data/input.txt --total-tokens 1000000 --seed 42 \
  --output data/synth_wrap_d16.txt

# 1M tokens, d=32 (the original prefix experiment)
bin/synth_wrap_corpus --trie-dir /tmp/agpt_input_d32_radix \
  --vocab-text data/input.txt --total-tokens 1000000 --seed 42 \
  --output data/synth_wrap_d32.txt

# 10M tokens, d=32 (the headline result — PPL 7.17 at seq=128/10k steps)
bin/synth_wrap_corpus --trie-dir /tmp/agpt_input_d32_radix \
  --vocab-text data/input.txt --total-tokens 10000000 --seed 42 \
  --output data/synth_wrap_d32_10M.txt
```

### Step 2: train the headline model (PPL 7.17)

```sh
cp data/input.random.model /tmp/synth_d32_10k.model
bin/microgpt data/synth_wrap_d32_10M.txt \
  --model /tmp/synth_d32_10k.model \
  --seq-len 128 --steps 10000 --lr 3e-4 \
  --d-model 64 --n-layers 2 --backend openblas --seed 42

# Score on real corpus:
bin/perplexity --model /tmp/synth_d32_10k.model --file data/input.txt \
  --max-positions 4096 --backend openblas
# → Perplexity: 7.1737
```

`bin/microgpt` and `bin/perplexity` come from the µGPT shard — build them
with `just build-microgpt-tools` if missing.
