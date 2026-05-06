# Wormhole / topological-navigation experiments

Tests whether replacing the unary-tunnel walk with a structural
"wormhole jump" (cap → re-entry node in the prefix tree) provides
useful training signal. Two dumb-baseline variants compared head-to-head
against the existing `synth_wrap` walk-and-bridge pipeline.

See `notes/agpt/topological-navigation.md` for the framing and the
larger architectural sketch (full-attention bridge, BayesInv suffix
coupling, Cycle Consistency Loss). This experiment is the *minimal*
structural baseline — single-token routing, no learned attention.

## Variants

| Variant | Routing rule |
|---|---|
| **V1** | `cap.edge_tokens[0]` (cap head char, where prefix-tree branching dies) → depth-1 root child |
| **V2** | char at the cap-path position where **suffix-tree** branching dies → depth-1 root child |
| (synth_wrap baseline) | walk full unary tunnel, sample bridge token from leaf endpoint dist, find depth-1 root child whose first edge matches |

V1 throws away tunnel info entirely. V2 uses single-bit suffix info to
pick a "later" routing char (peak boundary depth d=26 in the d=32
trie). synth_wrap walks the full tunnel and uses a learned bridge
token sampled from the leaf's endpoint distribution.

## Result (Shakespeare 1M, d=32, seq=128, 10k steps, openblas, seed 42)

| Variant | PPL on real corpus | Δ vs synth_wrap |
|---|---:|---:|
| synth_wrap baseline | **7.17** | — |
| Wormhole V1 (cap-head routing) | **7.65** | +6.7% |
| Wormhole V2 (suffix-boundary routing) | **7.88** | +9.9% |

**Conclusion**: skipping the unary tunnel costs PPL. Both V1 and V2
underperform synth_wrap — the unary tunnel chars carry training
signal that the model uses, even though they are
information-theoretically zero-entropy under the trie distribution.

V1 < V2 (V1 is the smaller regression). Routing by cap head produces
more coherent stitches than routing by a deeper-in-tunnel boundary
char, presumably because the cap head is the structurally consistent
"natural exit" char for the prefix branch.

## Interpretation

The wormhole structure is **mathematically clean** (constant-memory
arbitrary-context navigation) but the *dumb* version (single-token
routing) discards the very signal — the corpus's actual char-level
unary suffixes — that walk-and-bridge preserves verbatim.

For wormhole to beat walk-and-bridge it would need to:

1. **Preserve tunnel signal**, e.g. by emitting cap edge_tokens AND
   then wormholing (so the model sees the unary chars, just with a
   structural transition marker).
2. **Use richer routing**, e.g. the full-attention bridge in
   `topological-navigation.md` rather than single-token depth-1
   re-entry. The dumb baseline is the floor; the bridge is what's
   supposed to recover synth_wrap's performance and then beat it via
   the suffix-key/prefix-query attention.
3. **Run at scale where d <<  the structural depth.** d=32 already
   covers most natural-language structure for this corpus; the
   wormhole's win is conjectured to appear when context > d.

## Artifacts (regenerable, not in git)

| File | Purpose |
|---|---|
| `/tmp/wormhole_d32_v1.bin` | V1 side-table (cap_id → depth-1 root child for cap head) |
| `/tmp/wormhole_d32_v2.bin` | V2 side-table (cap_id → depth-1 root child for suffix-boundary char) |
| `data/wormhole_d32_v1_10M.txt` | 10M chars stitched via V1 routing |
| `data/wormhole_d32_v2_10M.txt` | 10M chars stitched via V2 routing |
| `/tmp/wormhole_d32_v1_10k.model` | V1-trained microgpt (PPL 7.65) |
| `/tmp/wormhole_d32_v2_10k.model` | V2-trained microgpt (PPL 7.88) |

## Regeneration

```sh
# Build tools
just build-agpt-build-wormhole-table
just build-agpt-wormhole-sample

# Side-tables (V1 needs only prefix trie; V2 needs both)
bin/agpt_build_wormhole_table --trie /tmp/agpt_input_d32_radix \
  --out /tmp/wormhole_d32_v1.bin --variant v1
bin/agpt_build_wormhole_table --trie /tmp/agpt_input_d32_radix \
  --suffix-trie /tmp/agpt_input_d32_suffix_radix \
  --out /tmp/wormhole_d32_v2.bin --variant v2

# Sampling (10M chars; one giant stream, ~7-8s wall)
bin/agpt_wormhole_sample --trie /tmp/agpt_input_d32_radix \
  --wormhole-table /tmp/wormhole_d32_v1.bin --vocab-file data/input.txt \
  --out data/wormhole_d32_v1_10M.txt \
  --n-samples 1 --max-len 10000000 --n-loops 10000000 --seed 42 --text

# (V2 corpus may be missing rare chars — patch before training:)
# python3 -c "src=set(open('data/input.txt').read()); v=set(open('data/wormhole_d32_v2_10M.txt').read());\
#   import sys; sys.stdout.write(''.join(sorted(src-v)))" >> data/wormhole_d32_v2_10M.txt

# Training (same recipe as wrap-around baseline)
cp data/input.random.model /tmp/wormhole_d32_v1_10k.model
bin/microgpt data/wormhole_d32_v1_10M.txt --model /tmp/wormhole_d32_v1_10k.model \
  --seq-len 128 --steps 10000 --lr 3e-4 --d-model 64 --n-layers 2 \
  --backend openblas --seed 42

# Eval
bin/perplexity --model /tmp/wormhole_d32_v1_10k.model --file data/input.txt \
  --max-positions 4096 --backend openblas
```

## Status

**Single seed each.** The 0.23 PPL gap between V1 and V2 is within the
seed-noise band observed in `wrap-around` (PPL 6.93–7.13 across 4
seeds at the synth_wrap recipe). A multi-seed sweep would tighten the
V1 vs V2 ordering. The headline gap (wormhole > synth_wrap by ≥0.5
PPL) is well outside seed noise and is the load-bearing finding.

Single-token routing is the floor. The full attention bridge is what
would actually be a fair test of the topological-navigation claim.
That's the next mechanism to implement if this line continues.
