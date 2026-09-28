---
title: Depth-124 radix trie
kind: diagnostic
status: concluded
outcome: n/a
question: >-
  Can the CUDAX v2 trainer handle a full depth-124 Shakespeare radix trie (the depth at which
  raw contexts become unique), and if not, what is the blocker?
answer: >-
  It is not obviously impossible. The compact K/V cache is about 3.46 GB, only about 14% more
  slots than d16. The real costs are 129.6M query positions per epoch (about 14x d16) and
  about 1.5 GiB of full edge-char arrays in the loader. The note recommends chunk_queries=10000
  and sampling of mass-1 cap tails.
opened: 2026-05-28
updated: 2026-05-28
code: main
eval: none
tags: [trie-structure, context-length]
related: [cudax-d124-probe, window-d124-baseline]
---

# Depth-124 radix trie

> Stub README (2026-09-28): this directory predates the README convention. The
> summary below was reconstructed from its files; see them for detail.

**Question.** Can the CUDAX v2 trainer handle a full depth-124 Shakespeare radix trie (the depth at which raw contexts become unique), and if not, what is the blocker?

**Answer.** It is not obviously impossible. The compact K/V cache is about 3.46 GB, only about 14% more slots than d16. The real costs are 129.6M query positions per epoch (about 14x d16) and about 1.5 GiB of full edge-char arrays in the loader. The note recommends chunk_queries=10000 and sampling of mass-1 cap tails.

**Sources.** Directory holds only the built trie shake_d124_radix; the answer comes from notes/seq-len-extension/d124-radix-feasibility.md ('Actual Blockers' and 'Current Read'), which names this path.

**Caveats.** The dir is untracked (.gitignore excludes rnd/radix-depth*/**/*.The follow-up training runs are in rnd/cudax-d124-probe.
