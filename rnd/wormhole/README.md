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

## Result (Shakespeare 1M, d=32, seq=128, 10k steps, openblas; multi-seed = 42/44/46/48 with fresh random init per seed)

| Variant | Density | Mean PPL | Range | Δ vs SGD |
|---|---|---:|---|---:|
| **SGD on real corpus** (ceiling) | — | **6.96** | 6.93–7.02 | — |
| synth_wrap | 32/tr | 7.06 | 6.92–7.18 | +1.4% |
| Wormhole V2 walk-tunnel | 32/tr | 7.08 | 6.97–7.21 | +1.7% |
| Wormhole V1 walk-tunnel | 32/tr | 7.10 | 7.06–7.19 | +2.0% |
| Wormhole V1 stream (1 seed) | 8.7/tr | 7.65 | — | +9.9% |
| Wormhole V1 aligned (1 seed) | 8.7/tr | 7.67 | — | +10.2% |
| Wormhole V2 stream (1 seed) | 8.7/tr | 7.88 | — | +13.2% |
| Wormhole V2 aligned (1 seed) | 8.7/tr | 8.29 | — | +19.1% |

Three density-matched synthetic-corpus methods (synth_wrap bridge,
V1 cap-head, V2 suffix-boundary) land in the same band ~7.06–7.10
mean, statistically indistinguishable, all ~1.4–2.0% above the SGD
ceiling. The wormhole routing rules are equivalent to synth_wrap's
bridge-token sampling as corpus-construction primitives.

Density notation: `Y/tr` = average chars per wormhole transition in the synthetic corpus.

### Density was the dominant confound

Initial result (skip-tunnel variants) showed wormhole 7-15% behind
synth_wrap. The cause was wormhole transition density: skip-tunnel
variants fire a wormhole every ~9 chars (every cap), while synth_wrap
walks the full 32-char unary tunnel between transitions. At
seq_len=128, that's ~14 wormhole-induced transitions per training
window vs ~4 for synth_wrap — the model can't recover any structure.

The `--walk-tunnel` control emits `cap.edge_tokens` (the unary tunnel
chars) BEFORE wormholing. Density goes to 32/tr exactly. Result: V1
walk-tunnel 7.19 ≈ synth_wrap 7.17 (single-seed); V2 walk-tunnel
multi-seed mean 7.08 sits in the synth_wrap range (6.93–7.13) and
just above the SGD ceiling 6.96. The wormhole routing rule is
structurally fine — what failed was skipping the tunnel.

### Sample-boundary alignment didn't matter

"Aligned" variant: emit each sample as exactly `seq_len=128` chars
with no inter-sample newline (`--no-separator`). File size =
n_samples × seq_len exactly, so microgpt's stride=seq_len windowing
puts each training window on one independent walk-with-wormholes.

Alignment didn't help (V1 aligned 7.67 ≈ V1 stream 7.65). At ~14
wormholes per 128-char sample, transition density saturates the
window regardless of where it starts. V2 aligned actually regressed
(7.88 → 8.29), suggesting restart-from-root distribution interacts
unfavorably with V2's suffix-boundary routing in some way.

**Conclusion**: skipping the unary tunnel is the single biggest cost
in the dumb-baseline experiments. The unary tunnel chars carry
training signal that the model uses, even though they are
information-theoretically zero-entropy under the trie distribution.

When density is matched (`--walk-tunnel`), the wormhole routing rule
performs comparably to synth_wrap. V2 (suffix-aware boundary char
routing) sits at multi-seed mean 7.08, within the synth_wrap range
(6.93–7.13) and just above the SGD ceiling (6.96 mean). V1
(cap-head routing) at 7.19 single-seed is ~similar.

So the structural framing is validated: routing at cap → re-entry
via depth-1 root child works as a corpus-construction primitive
when the unary tunnel is preserved. What it doesn't yet provide is
the wormhole's promised benefit — constant-memory arbitrary-context
navigation. To realize that, the model itself would need to
*learn* to skip identity tunnels at inference; the corpus-level
test here only validates that the routing rule produces trainable
data.

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
