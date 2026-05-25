# Per-Node Position Distributions — Data Structure Design

## Goal

Store, for each radix-trie node, a histogram of the long-window positions
where its prefix occurs. Enable multi-position encoding schemes (sampled,
distribution-aware-RoPE, CRT, Fourier, wavelet, ALiBi) that extend the
model's effective context beyond the trie's natural depth (d=16) without
deeper d.

The underlying data is the same across all those encoding schemes; only
the *consumer* differs. Get the data structure right once.

## Notation

- `d` = trie depth (e.g., 16) — unchanged from current setup
- `W` = position-window size (e.g., 64, 128) — the extended context the
  position-tagging operates over. Must have `W ≥ d`.
- `regime` = aligned (sub-paths at positions 1, d+1, 2d+1, ...) or
  sliding (sub-paths at every offset 1, 2, 3, ...)
- A radix node at depth `k` can be tagged with long-window positions:
  - aligned: `{k, k+d, k+2d, ..., ≤ W-d+k}` — at most `floor(W/d)` bins
  - sliding: `{k, k+1, k+2, ..., W-d+k}` — at most `W-d+1` bins

## What we store per node

For radix node `r`:

```
pos_counts[r]: histogram over the legal position bins for node r's depth.
```

That's it conceptually. The challenge is the storage layout.

## Storage options

### Option A: dense `pos_counts[radix_count][W]`

```c
int dense_pos_counts[radix_count * W];
```

Simple. But: Shakespeare 1M with radix_count=1.6M and W=128 → 800 MB.
Gutenberg 5M → 3 GB. Too big for a side-table loaded alongside training.

Rejected.

### Option B: sparse per-node (RECOMMENDED)

For each node, store only the bins with nonzero count. Cap at top-K
bins if needed (most-frequent positions only, drops long tail).

```c
// One contiguous backing store, indexed via per-node offsets.
int pos_offsets[radix_count + 1];   // start of each node's bin list
struct PosBin { uint16_t pos; uint32_t count; };  // 6 bytes per bin
PosBin pos_bins[total_bins];

// Lookup: bins of node r are pos_bins[pos_offsets[r] : pos_offsets[r+1]]
```

Empirical sparsity (sketch):
- Most radix nodes are caps with mass ≤ 5. They appear at only a few
  positions in the whole corpus. ~3-5 bins per cap node.
- Internal nodes have higher mass and more bins. Up to ~50 bins for
  the highest-mass shallow nodes.
- Average ~10 bins/node × 6 bytes = 60 bytes/node × 1.6M nodes
  ≈ 100 MB on Shakespeare. Manageable.

Compaction further possible: skip mass=1 nodes (they have trivially 1
bin, position is recoverable from edge_starts + j).

Recommended on-disk format:

```
magic:        u32 = 0x41504F53 ("APOS")
window_size:  u32 = W
regime:       u8 (0=aligned, 1=sliding)
radix_count:  u32 (must match the trie this table was built for)
total_bins:   u64

pos_offsets:  i32[radix_count + 1]  (start offsets into pos_bins)
pos_bins:     PosBin[total_bins]    (u16 pos + u32 count, packed)
```

Total size estimate (sliding regime, W=128, Shakespeare 1M):
- Header: ~24 bytes
- pos_offsets: 1.6M × 4 = 6.4 MB
- pos_bins: ~16M bins × 6 = ~100 MB

For Gutenberg 5M, ~400 MB. Acceptable.

### Option C: per-depth pooling (advanced)

All depth-`k` nodes share the same legal position set, so the bin
*positions* are redundant — only the counts differ per node. Could
store as a depth-keyed 2D layout. ~30% size win but adds indexing
complexity. Defer unless sizes get too tight.

## Build procedure

Walk the corpus once (same pass that builds the radix trie). For each
sliding-window start at corpus position `p`:

```python
sub_window = corpus[p : p+d]
long_window_pos = p % W   # position within the W-context this fire is part of

# Walk sub_window down the trie, incrementing pos_counts as we go
node = root
for char_idx in range(d):
    node = node.children[sub_window[char_idx]]
    bin_pos = (long_window_pos + char_idx) % W   # position within W-window
    node.pos_counts[bin_pos] += 1
```

Aligned regime: only fire when `p % d == 0` (sub-windows align to `d`-boundaries).
Sliding regime: fire for every `p` (matches current trie build).

After the pass, compact each node's `pos_counts` dict into the sparse
on-disk format (drop zero bins, sort by position, emit `(pos, count)`
pairs).

Build cost: one extra pass over the corpus, comparable to trie build
time. ~30s on Shakespeare 1M, ~2.5 min on Gutenberg 5M.

## Consumer interfaces

The data structure is consumed differently by each encoding scheme.
All four are supported by the same on-disk layout; pick the consumer
at training time.

### Consumer 1: per-fire sampled position

```cpp
// During training, for each fire of node r:
int sample_position(int radix_id, RNG& rng) {
    int off  = pos_offsets[radix_id];
    int n    = pos_offsets[radix_id+1] - off;
    int total = sum of pos_bins[off..off+n].count;
    int target = rng.uniform(0, total);
    int acc = 0;
    for (int i = 0; i < n; i++) {
        acc += pos_bins[off+i].count;
        if (acc > target) return pos_bins[off+i].pos;
    }
    return pos_bins[off+n-1].pos;  // tail
}
```

For efficiency: precompute alias-method tables per node (one-time
O(bins) per node, O(1) per sample). Memory: 2 extra bytes per bin.

### Consumer 2: per-fire expected position (deterministic)

```cpp
float expected_pos(int radix_id) {
    int off = pos_offsets[radix_id];
    int n   = pos_offsets[radix_id+1] - off;
    float weighted_sum = 0; uint64_t total = 0;
    for (int i = 0; i < n; i++) {
        weighted_sum += pos_bins[off+i].pos * pos_bins[off+i].count;
        total        += pos_bins[off+i].count;
    }
    return weighted_sum / total;
}
```

Precompute once per node, store as `float expected_pos[radix_count]`.
~6 MB extra side-data. Same every fire (deterministic).

### Consumer 3: distribution-aware RoPE (the interesting one)

Standard RoPE rotates Q/K by angle `θ(p, i) = p / base^(2i/HD)` for
position `p`, dim-pair `i`. With a distribution over positions, the
"effective rotation" is:

```
effective_cos(node, i) = Σ_p (count[p] / total) * cos(p / base^(2i/HD))
effective_sin(node, i) = Σ_p (count[p] / total) * sin(p / base^(2i/HD))
```

Geometrically: take the unit vectors at each (rotation_angle(p), 1) and
weighted-average them (each weighted by the position's mass). The
result is a vector inside the unit circle (length < 1 unless the
distribution is a delta function); this is the "expected rotation"
that encodes the *distribution* rather than any single position.

Precompute per (radix_id, head_dim_pair):
- `eff_cos[radix_count][HD/2]`  — float32
- `eff_sin[radix_count][HD/2]`  — float32

Storage: radix_count × HD/2 × 8 bytes (cos+sin). At HD=16, Shakespeare 1M:
1.6M × 8 × 8 = 100 MB. Comparable to the histogram itself.

Then in the model, replace the standard RoPE lookup `cos_cache[pos][i]`
with `eff_cos[radix_id][i]` (same for sin). One kernel change, same
attention math otherwise.

A key property of this encoding: nodes with tight position distributions
(low entropy) get rotations close to the unit circle (sharp positional
identity); nodes with broad distributions get rotations near the origin
(positionally ambiguous). The MODEL gets to learn how to interpret
these distinctions — a sharp rotation is "I know exactly where I am
in the long context," a soft rotation is "I appear all over the place."

This is genuinely novel positional encoding for our setting. The
information being encoded is "the distribution of positions where this
prefix lives in the corpus."

### Consumer 4: CRT / Fourier / wavelet / ALiBi variants

All take the same histogram as input and project differently:
- **Fourier**: encode the position distribution as low-frequency
  components. Like distribution-aware RoPE but explicit frequency basis.
- **Wavelet**: hierarchical scales of the distribution; coarse-to-fine
  position info.
- **CRT** (Chinese Remainder): factor `pos` into multiple coprime
  moduli, encode each modulus independently.
- **ALiBi**: bias attention scores by a function of position-distance;
  could use distribution-distance instead.

The histogram is the substrate; the four are alternative read-outs.

## CLI

```sh
# Build the position-distribution table (one-time per trie):
bin/agpt_build_position_table \
    --trie-dir /tmp/shake_baseline_d16_radix \
    --window 128 \
    --regime sliding \
    --out /tmp/shake_pos_W128_sliding.bin

# Train with the table loaded; consumer selectable via --pos-encoder:
bin/agpt_train \
    --model <init> --trie-dir <radix> \
    --pos-table /tmp/shake_pos_W128_sliding.bin \
    --pos-encoder {sampled|expected|dist-rope|fourier|wavelet|alibi} \
    [--pos-encoder-args ...] \
    --epochs 100 ...
```

## Implementation plan

1. **`bin/agpt_build_position_table`** (Crystal, half-day):
   - Reuse the corpus iteration code from the existing trie builder
   - Output the on-disk format above
2. **Loader in `agpt_train`** (a few hours):
   - Read the table; assert magic + radix_count match the loaded trie
   - Build the alias-method tables (for sampled consumer) at load time
   - Build the precomputed eff_cos / eff_sin tables (for dist-rope) at load time
3. **Consumer code paths** (varies per consumer):
   - Sampled: replace the RoPE position assignment with a per-fire RNG sample
   - Expected: replace with the precomputed `expected_pos[radix_id]`
   - Dist-rope: replace the cos/sin cache lookup with the per-node eff table
   - Others: per-consumer
4. **Memory budget validation** at startup: print `pos-table: X MB on
   disk, Y MB after preprocess` so we don't surprise ourselves.

## Effort estimate

- Build tool + loader + sampled consumer: ~1 day
- Add dist-rope consumer on top: another half-day (the per-node
  eff_cos/sin precompute is straightforward; the kernel change is
  small)
- Each of fourier/wavelet/CRT/ALiBi: ~half-day per variant, since
  they share infrastructure

## What this enables

For the first time, the trie's position info is *separable* from its
depth info. The model gets to learn how to use both:

- Sharp position-distribution nodes → strong positional anchors
  (e.g., common phrase that always appears at sentence start)
- Broad position-distribution nodes → context-invariant patterns
  (e.g., "the" appearing everywhere)
- The encoding is data-driven, not hand-tuned

And it doesn't require extending `d`, which is where the trie's
quality cliff lives.

## Open design questions

1. Aligned vs sliding regime — sliding has more info per node but
   bigger tables. Start with sliding; we can compare.
2. What `W`? 64 / 128 / 256? Bigger W = more information but bigger
   side-table and bigger encoding overhead. Start with 64.
3. Are some consumers OBVIOUSLY redundant (e.g., dist-rope and
   fourier may overlap)? Probably; pick 2-3 to compare first.
4. Per-fire sampled (consumer 1) uses sampling noise as a feature —
   does it help or hurt training stability? Worth a small-scale
   ablation before committing.
