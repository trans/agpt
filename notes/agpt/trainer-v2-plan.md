# AGPT Trainer V2 Plan

## Decision

Build a new CUDA trainer core in `src/cuda/v2/` instead of continuing to layer
feature work into `src/cuda/agpt_train.cu`.

This is a trainer-engine rewrite, not an AGPT/model rewrite.

## Why

The current trainer still works as a baseline, but feature work now crosses too
many hidden boundaries at once:

- compact-cache layout
- RoPE coordinate space
- chunk planning
- subtree/fire semantics
- optimizer timing
- research-only execution flags

The descendant→ancestor scatter work exposed this directly: a mathematically
simple gradient-path fix forced simultaneous reasoning about cache indexing,
RoPE reversal, packed-K/V layout, and fire-end reductions.

That is the signal to stop patching the old execution core.

## What stays

- existing model format
- radix/trie file format
- reusable CUDA kernels where they are still correct
- current trainer as the stable baseline path

## What changes

- new standalone trainer binary: `bin/agpt_train_v2`
- new CUDA-side source tree: `src/cuda/v2/`
- explicit contracts for cache space, backward space, and training-unit scope

## First milestone

Baseline parity only:

- radix path
- `partition_depth = 1`
- no Lightning
- no virtual cycles
- no descendant→ancestor gradient flow
- no exotic experimental flags

The first job of V2 is to be simple enough to trust.

## Planned rollout

1. Skeleton binary and module layout
2. Baseline `pd=1` trainer
3. Parity test against current engine
4. Reintroduce features one family at a time:
   - descendant→ancestor scatter
   - `pd>1`
   - shuffle / mini-batch-group variants
   - Lightning / virtual cycles
   - other research flags
