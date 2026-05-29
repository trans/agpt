# Worktree Brief: Fold-Map Builder for AGPT Prefix-Suffix Folding

**Status (2026-05-20):** NOT EXECUTED. Sub-agent worktree was created and
this brief drafted, but the task was never run. Branch + worktree dropped
2026-05-20 as part of branch cleanup; brief preserved here as a record of
what was scoped. Cap-folding (the mechanism this brief implements) is in
the same family as virtual-tree experiments which regressed empirically
(see `project_virtual_tree_negative.md` in user memory and the
`rnd/virtual-tree-*` dirs). The brief itself remains well-specified; if
the design ever gets revisited, the v1 algorithm here is the natural
starting point.

The other two mechanisms in `notes/prefix-suffix/prefix-suffix-fold-architecture.md`
(prefix-suffix Bayesian bridging, AGPT-aggregated training) are
independent of cap-folding and remain open follow-ups.

---

**Original brief contents follow.**

**Branch (gone):** `fold-map-builder`
**Path (gone):** `/home/trans/Projects/agpt-fold-map/`
**Status (at time of drafting):** Independent of main AGPT trainer work — safe to develop in parallel.

## Read this first

1. **`notes/prefix-suffix/prefix-suffix-fold-architecture.md`** — the full
   design synthesis for the unified prefix-suffix-fold architecture.
   This document is the canonical reference; everything else is
   subordinate.
2. **`rnd/prefix-suffix-bayes/findings.md`** — Phase 1 math validation
   (raw trie distributions agree exactly across forward/suffix tries).
3. **`rnd/granularity-redundancy/README.md`** — context on what
   AGPT's recipe is and why partition-depth=6 is the current best.
4. **`memory/project_radix_cap_dedup.md`** (in user's Claude memory) —
   the cap-dedup analysis showing 0.34% cap dedup on Shakespeare.

## Your task: build the fold-map builder

Build a CPU-side Crystal tool that scans a forward radix-trie, identifies
fold targets for each cap, and emits a binary fold-map file the trainer
will load at runtime.

### Tool location
- Source: `src/tools/fold_map_builder.cr`
- Output binary: `bin/fold_map_builder`
- Build recipe: add to `Justfile` as `build-fold-map-builder`

### CLI

```sh
bin/fold_map_builder \
  --forward-trie <DIR> \
  --out <fold_map.bin> \
  [--min-target-mass N]    # default 5; reject targets with mass < N
  [--algorithm v1]         # for now only v1 supported
```

### V1 algorithm: exact edge-text match

For each cap node C (radix node with no children, or single-child
unary chain ending without further branching):

1. Look up internal nodes (radix nodes with ≥2 children) whose
   `edge_tokens` exactly equal C's `edge_tokens`.
2. Among those candidates, pick the one with the highest `edge_mass`
   (most observations → most-trained distribution).
3. If a candidate exists with `edge_mass ≥ min-target-mass`, record
   the fold map entry: `cap_id → target_id`.
4. If no candidate exists, this cap is a "dead end" — record nothing.

This is the simplest possible algorithm. V2 (substring matching with
probability-trajectory verification) is documented in the architecture
note but is **out of scope for v1**.

### Output format

Binary file. Magic header `FMAP` + version + radix_count, then a
flat array of (cap_id: int32, target_id: int32) pairs. Pairs sorted
by cap_id ascending. End-of-file = end of pairs. Use little-endian.

Example header bytes:
```
'F' 'M' 'A' 'P'   # magic
0x01 0x00 0x00 0x00   # version 1
[radix_count : i32]   # source trie size, for sanity check at load
[n_pairs : i32]       # number of fold entries
... pairs ...
```

### Required output to stdout

```
Fold-map builder
Loaded forward trie: <DIR> (<radix_count> nodes)

Cap nodes:           <count>
Internal nodes:      <count> (with ≥2 children)
Internal nodes with mass ≥ min-target-mass:  <count>

Edge-text-keyed lookup table built: <unique_edges> unique edge sequences

Fold map results:
  Caps with fold target:    <count> (<%>)
  Caps as dead ends:        <count> (<%>)
  Mean target mass:         <num>
  Median target mass:       <num>

Fold-mass distribution by target:
  (top-10 most-targeted internal nodes by # caps folding to them)

Output written: <out_path> (<size> bytes)
```

### Validation tests

Add a test fixture in `spec/fold_map_builder_spec.cr`:
1. Build a tiny synthetic trie with known structure (say 5 internal
   nodes, 10 caps), where 3 caps should fold to known targets and
   7 should be dead ends.
2. Run the builder.
3. Verify the fold map has exactly the expected entries.

Use the existing `lib/microgpt` Crystal framework's spec runner (look
at `spec/` for examples).

### Run on Shakespeare 1M and Gutenberg 5M

After v1 is built and tested, run it on both corpora:

```sh
bin/fold_map_builder \
  --forward-trie /home/trans/agpt-tries/shakespeare_d32_radix_corpus \
  --out /tmp/shakespeare_d32_fold_map.bin

bin/fold_map_builder \
  --forward-trie /home/trans/agpt-tries/gutenberg_5m_d32_radix_corpus \
  --out /tmp/gutenberg_5m_d32_fold_map.bin
```

Document the results in `rnd/prefix-suffix-bayes/fold_map_v1_findings.md`:
- Fold rate per corpus (% of caps with fold target)
- Dead-end rate
- Distribution of target mass (caps fold to popular vs niche internal nodes)
- Comparison to the cap-dedup result (0.34% Shakespeare, 0.65% Gutenberg)
- Any surprises

### Things to NOT do (out of scope)

- Don't modify the AGPT trainer (`src/cuda/agpt_train.cu`)
- Don't modify the radix-trie reader (`src/agpt/radix_trie_reader.cr`)
  beyond adding read-only methods you need
- Don't implement substring-match v2 — wait for the architecture
  decision in `notes/prefix-suffix/prefix-suffix-fold-architecture.md` to
  evolve
- Don't train any models in this worktree

### Definition of done

- [ ] `bin/fold_map_builder` compiles and runs
- [ ] Spec tests pass
- [ ] Fold-map binary files generated for Shakespeare 1M and Gutenberg 5M
- [ ] `fold_map_v1_findings.md` documents the empirical fold rates and
      target-mass distributions
- [ ] All work committed to the `fold-map-builder` branch
- [ ] No changes to anything outside `src/tools/`, `spec/`, `bin/`,
      `Justfile`, and `rnd/prefix-suffix-bayes/`

### When you're done

Push the branch, then update this BRIEF.md with a "Status: complete"
note and a brief description of any algorithm choices or surprises
encountered. The main thread will review and merge.

## Communication with main thread

If you discover the v1 algorithm doesn't produce a useful fold rate
(e.g., <0.1% caps with targets, or all targets clustering on one node),
PAUSE and document the issue in BRIEF.md before proceeding. The main
thread will revisit the algorithm design.
