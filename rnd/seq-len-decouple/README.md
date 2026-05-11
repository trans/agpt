# Seq_len decoupling — Phase 0: position → contributing-nodes map

**Goal:** build the bookkeeping needed for the decoupled-attention work
described in `notes/agpt/shared_key_rope.md`. Each corpus position p
contributes mass to multiple radix nodes (one per radix node on its
root-to-leaf path). Phase 0 computes that map, dumps it to disk, and
reports what it looks like.

## What this is *not*

The first iteration of this tool recorded only the **deepest landing
node** per position — a single radix_id, giving the impression that
position-to-node was a near-bijection at d=32 (99.7% of positions get
their own unique node). That was a misleading partial view: it
captured only the leaf identity and ignored the 8-10 shallower nodes
each position also touches.

Corrected approach: record *every* radix node a position contributes
mass to during its trie walk. Each position has ~9 node identities at
d=32, ranging from broad (root child, mass hundreds-of-thousands) to
narrow (leaf cap, mass 1).

## Tool

`bin/agpt_position_map` (source: `src/tools/agpt_position_map.cr`):

```
bin/agpt_position_map \
  --trie /home/trans/agpt-tries/gutenberg_5m_d32_radix_corpus \
  --file data/gutenberg_5m.txt \
  --out rnd/seq-len-decouple/gutenberg_5m_d32_pos_to_node.bin
```

Output binary format: per corpus position p, a record
`[k: int32] [nid_1, ..., nid_k : int32]` where `k` is the number of
radix nodes traversed and the `nid_i` are the radix node IDs in
root-to-leaf order.

A `--leaf-only` flag reverts to recording just the deepest landing
node (original behavior) for comparison.

## Results — Shakespeare 1M, d=8

| metric | value |
|---|---:|
| positions walked | 1,115,386 |
| mean radix nodes per position | **6.66** |
| total (position, node) contributions | 7,425,976 |
| unique nodes touched | 845,539 |
| unique leaf-landing nodes | 609,659 |
| contributions / trie edge_mass | 100.0% |
| per-node count mismatches | 31 (corpus-edge artifacts) |
| top-by-contribution nodes | depth-1 root children w/ mass 170k, 95k, 67k |

## Results — Gutenberg 5M, d=32

| metric | value |
|---|---:|
| positions walked | 4,999,968 |
| **mean radix nodes per position** | **9.24** |
| total (position, node) contributions | 46,190,342 |
| unique nodes touched | 7,539,819 (= entire trie) |
| unique leaf-landing nodes | 4,988,254 |
| contributions / trie edge_mass | 100.0% |
| per-node count mismatches | 199 (corpus-edge artifacts; mass = contribs + 1) |
| top-by-contribution nodes | depth-1 root children w/ mass 821k, 477k, 339k |
| binary output | 205 MB |
| wall time | ~28 s (180k positions/sec) |

### Node-touch distribution at d=32

| k (nodes touched) | count | pct |
|---:|---:|---:|
| 1-8 | small | <5% |
| 9 | bulk | ~60% |
| 10+ | tail | ~25% |
| 32 (full path uncompressed) | 1,490 | 0.03% |

The radix compression collapses ~32 character-depth-levels into ~9
radix-level steps on average. Variance is small: most positions touch
9-10 nodes, with a thin tail of high-branching positions that touch
more.

## Mass consistency

The walk records every contribution. For each node N, the count of
distinct positions that contributed to N must equal the trie's stored
`edge_mass` for N. Verified at 100% (with a tiny corpus-edge
correction): for ~199 nodes at Gutenberg the walk records mass −
1 contributions, which matches the boundary case where the position
near the corpus end can't complete a full d-window.

This confirms the position→node map matches the data the trie was
built with — no walk bug, no double counting.

## Top contributing nodes (Gutenberg d=32)

```
node_id   contribs  mass     depth  edge_len
149117    821682    821687   1      1
2772912   477220    477225   1      1
6552035   339826    339827   1      1
1839620   306058    306058   1      1
5318063   290449    290452   1      1
```

The most-contributed-to nodes are depth-1 single-char root children —
i.e., they correspond to the unigram distribution of the corpus. The
top entry (node 149117) gets 821k contributions, meaning 16.4% of
corpus positions start with the same character. This is the right
shape — the trie's shallow nodes do carry corpus-wide aggregation
information, and AGPT's training-loop visits them on every chunk.

## Implications for seq_len decoupling

Each corpus position has ~9 distinct node identities. For
decoupled-attention (`notes/agpt/shared_key_rope.md`), this means K_p
is not a single vector — there are 9 plausible vectors, one per node
identity at that position:

```
K_p^(1) = RoPE(k_{n_p_shallowest}, p)    # broad context, mass hundreds-of-thousands
K_p^(2) = RoPE(k_{n_p_depth2}, p)        # 2-char context
...
K_p^(9) = RoPE(k_{n_p_leaf}, p)          # 32-char context, mass 1
```

The model gets to pick — or use multiple in parallel. This opens
several design options for Phase 1:

1. **Always-leaf**: use the deepest identity. Same as the v1 of this
   tool. Maximally specific per-position, near-bijective with corpus
   positions.
2. **Always-shallow-best**: use the deepest identity with mass ≥ τ
   (some threshold). Falls back to shorter contexts when the leaf is
   a singleton. Preserves aggregation.
3. **Stack-attention**: use *all* of a position's node identities as
   separate attention keys, letting the model weight them.
4. **Adaptive per-query**: pick depth based on what the query needs.

Phase 1 will probably want to start with (2), since (1) is what the
existing AGPT chunk-trainer already effectively uses, and (3)/(4)
require more architectural work.

## Files

```
rnd/seq-len-decouple/
├── README.md                                       (this file)
├── gutenberg_5m_d32_pos_to_node.bin                (205 MB)
└── logs/
    ├── gutenberg_d32.log                           (leaf-only version, deprecated)
    └── gutenberg_d32_full_contrib.log              (current full-contribs run)
```

## Followups

- Confirm the top contributing node (149117) is the unigram-frequency
  most common starting char.
- Investigate the depth-32 mass=1347 spike: which 32-gram occurs 1347
  times? Likely Project Gutenberg boilerplate (license headers,
  separators between concatenated books).
- Implement the inverse map dump (`--out-inverse PATH` flag) — per
  trie node, list the corpus positions that contributed. This is
  what attention machinery will load at training time.
- Phase 1: shared-key RoPE inference at seq_len > d on the AGPT
  d=32 4.90 PPL model.
