---
title: Partition-scoped KV cache
kind: infrastructure
status: concluded
outcome: n/a
question: >-
  Can the CUDA trainer size its KV cache to the largest partition group rather than the whole
  trie file, so that --partition-depth reduces peak memory enough to train the full d=16 global
  trie in one super-epoch?
answer: >-
  Not answered. Phase 1 only reported the achievable savings: for the largest d=16 per-subtree
  file with bigram partitions, peak KV would fall from 1295.7 MB to 161.7 MB (8.0x). Phase
  2 (scoped allocation, index remap, ancestor mini-forward) was planned but never built. The
  branch has no commits after 2026-04-22.
opened: 2026-04-22
updated: 2026-04-22
code: {branch: agpt-partition-kv-scoping, tag: exp/partition-kv-scoping}
eval: none
family: n/a
headline:
- {label: 'd=16 largest per-subtree file (rc=2), whole-file allocation', metric: peak KV cache
    MB (Phase 1 stats), value: 1295.7}
- {label: 'same file, scoped to largest bigram partition group (projected)', metric: peak
    KV cache MB (Phase 1 stats), value: 161.7}
tags: [partitioning, trainer, scaling]
related: [partition-depth]
---

# Partition-scoped KV cache

The CUDA trainer sizes its KV cache for a whole trie file, even when
`--partition-depth` splits the file into groups that train one after another. At d=16
the global trie needs about 9.5 GB of KV and does not fit. This branch planned to
size the cache to the largest partition group instead. Each group would get a CPU-side
`global_to_local` index remap and an ancestor mini-forward, with no kernel changes.
Only Phase 1 was built: a `--partition-kv-scoped` flag that prints the achievable
savings and does not change behaviour. For the largest d=16 per-subtree file with
bigram groups it reported peak KV of 1295.7 MB unscoped vs 161.7 MB scoped (8.0x).
Phase 2 (the scoped allocation, remap, mini-forward, a parity test and the d=16
global run) is fully specified in the plan but was never implemented.

Code: branch `agpt-partition-kv-scoping`, tag `exp/partition-kv-scoping`. Key files
are `notes/agpt/partition-kv-scoping-plan.md` (the full Phase 2 plan) and
`src/cuda/agpt_train.cu` (Phase 1 stats, commit 48ee729).
