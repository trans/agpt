# Seq_len decoupling — Phase 0: position → contributing-nodes map

**Goal:** build the bookkeeping needed for the decoupled-attention work
in `notes/agpt/shared_key_rope.md`. For each corpus position p, find
the radix-trie nodes whose path *terminates at* corpus position p.
Equivalently: position p has a representation at every node R where
R's path-from-root spells out a backward-suffix of length 1 to d
ending at p.

## Phase 0 framing (clarified by user 2026-05-11)

A corpus position p is represented in the trie at multiple nodes, one
per backward-suffix length:

| backward length k | node found by walking | trie node depth | unigram root |
|---:|---|---:|---|
| 1 | `corpus[p]` | 1 | `corpus[p]`'s root child |
| 2 | `corpus[p-1..p]` | 2 | `corpus[p-1]`'s root child |
| 3 | `corpus[p-2..p]` | 3 | `corpus[p-2]`'s root child |
| ... | ... | ... | ... |
| d | `corpus[p-d+1..p]` | d | `corpus[p-d+1]`'s root child |

So position p has up to **d representations spread across d different
unigram subtrees** (in the full uncompressed view). In a
radix-compressed trie, the count can be less than d because multiple
consecutive depth levels may collapse onto one radix edge.

Position 0 has exactly **1 representation** (just `corpus[0]` in its
unigram root). Position 1 has 2. Up to position d-1 the count ramps;
after that every position has up to d.

## What this is *not*

- **Not** a single landing node per position (my first iteration).
- **Not** a single root-to-leaf walk per position with all touched
  nodes attributed to the starting position (my second iteration —
  also wrong).
- The correct framing: nodes are attributed to the corpus position
  where their path *terminates*, not where the walk starts.

## Algorithm

For each starting position s ∈ [0, N-1]:
1. Walk forward up to d chars from root through the radix trie.
2. At each radix node touched, attribute that node to the corpus
   position where its path ends: `terminal_pos = s + cumulative_match_len − 1`.

Each walk emits ~9 contributions (one per radix node visited),
distributed across positions [s, s+d-1]. Aggregated over all walks,
each position p collects contributions from walks starting at s ∈
[max(0, p-d+1), p].

## Tool

`bin/agpt_position_map` (source: `src/tools/agpt_position_map.cr`):

```
bin/agpt_position_map \
  --trie /home/trans/agpt-tries/gutenberg_5m_d32_radix_corpus \
  --file data/gutenberg_5m.txt \
  --out rnd/seq-len-decouple/gutenberg_5m_d32_pos_to_node.bin
```

Output binary format: per corpus position p, a record
`[k:int32] [nid_1, ..., nid_k:int32]` where k = number of radix nodes
representing p, and the nid_i are sorted by walk (deepest first =
longest backward suffix first).

## Results — Shakespeare 1M, d=8

Corpus starts "First Citizen:..." — first 8 positions verified
manually:

| p | context_ending_at_p | k | representations (node@depth) |
|---:|---|---:|---|
| 0 | F | 1 | F-root,d1 |
| 1 | Fi | 2 | Fi@d2, i-root@d1 |
| 2 | Fir | 2 | Fir@d2 (radix-compressed), r-root@d1 |
| 3 | Firs | 3 | Firs@d3, irs@d2, s-root@d1 |
| 4 | First | 5 | First@d5, irst@d4, rst@d3, st@d2, t-root@d1 |
| 5 | First␣ | 6 | First␣@d6, ..., ␣-root@d1 |
| 6 | First␣C | 7 | First␣C@d7, ..., C-root@d1 |
| 7 | First␣Ci | 4 | First␣Ci@d8 (radix-compressed), ␣Ci@d3, Ci@d2, i-root@d1 |

Position 0: 1 representation. ✓ (matches the verification claim)

Aggregate stats:

| metric | value |
|---|---:|
| total corpus positions | 1,115,394 |
| mean radix nodes per position | 6.66 |
| total contributions | 7,426,007 |
| unique nodes touched | 845,539 |
| mass-consistency mismatches | 0 |

## Results — Gutenberg 5M, d=32

| metric | value |
|---|---:|
| total corpus positions | 5,000,000 |
| **mean radix nodes per position** | **9.24** |
| upper bound on per-position count | 32 (= d_max) |
| total contributions | 46,190,558 |
| unique nodes touched | 7,539,819 (= entire trie) |
| **mass-consistency mismatches** | **0** |
| binary output size | 205 MB |
| wall time | ~30 s |

### Contributions-per-position distribution (k = radix nodes per terminal position)

| k | count | pct |
|---:|---:|---:|
| 1 | very small | < 0.01% |
| 4-5 | rising | ~5-10% |
| 8 (peak) | 584,934 | 11.7% |
| 9 (near-peak) | 577,385 | 11.6% |
| 10 | 511,969 | 10.2% |
| 11-15 | tail | descending |
| 32 (upper bound) | 3,378 | 0.07% |

Mean 9.24, modal 8, very long tail. The k=32 tail are positions where
the walks happen to terminate at every char-depth (no radix
compression along any of the 32 backward-suffix paths).

### Top contributing nodes

```
node_id   contribs  mass     depth  edge_len
149117    821,687   821,687  1      1
2772912   477,225   477,225  1      1
6552035   339,827   339,827  1      1
1839620   306,058   306,058  1      1
5318063   290,452   290,452  1      1
```

The most-contributed-to nodes are depth-1 unigram root children: every
corpus position contributes to *some* unigram root (the one matching
its char). Top node (149117, mass 821k) is the most common starting
char in the corpus — likely space (32 in token-id terms).

### Mass consistency

For each radix node N, the count of corpus positions attributing
to N **exactly equals** the trie's stored `edge_mass` for N. Zero
mismatches across all 7.5M nodes and 46M contributions. This
verifies the walk records exactly the same contributions the trie
builder did.

## Implications for seq_len decoupling

Each corpus position has ~9 radix-node identities at d=32, ranging
from broad (depth-1 root child, mass hundreds-of-thousands) to narrow
(depth-32 leaf, mass 1). For shared-key RoPE attention at query
position q:

```
K_p^(k) = RoPE(k_{n_p^(k)}, p)    where n_p^(k) is the radix node
                                  representing p at backward-length k
```

The model has 9 plausible vectors per position. Phase 1 candidates:

1. **Always-leaf**: use the deepest backward-suffix identity (longest
   context, lowest mass).
2. **Always-shallow-best**: use the deepest with mass ≥ τ. Falls back
   to shorter contexts when leaf is a singleton.
3. **Stack-attention**: emit all of a position's identities as
   separate attention keys.
4. **Adaptive per-query**: choose backward-length based on what the
   query needs.

## Files

```
rnd/seq-len-decouple/
├── README.md                                       (this file)
├── gutenberg_5m_d32_pos_to_node.bin                (205 MB, gitignored)
└── logs/
    ├── gutenberg_d32.log                           (v1: leaf-only, wrong)
    ├── gutenberg_d32_full_contrib.log              (v2: full but mis-attributed)
    └── gutenberg_d32_terminal_attr.log             (v3: correct)
```

## Followups

- Confirm node 149117 is the unigram root for space (id 32 = ' ' in
  the dataset's char_to_id).
- Implement inverse map dump (`--out-inverse PATH`) — per radix node,
  list the corpus positions terminating there. Decoupled attention
  needs this to gather K vectors at training/inference time.
- Phase 1: shared-key RoPE inference at seq_len > d on the AGPT
  4.90 PPL model. Use the binary map to drive position-to-node lookup
  during attention.
