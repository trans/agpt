# AGPT MVP Implementation Plan

This document turns the AGPT core note into a concrete implementation plan for
this codebase.

AGPT means:

- `aggregate gradient`, and optionally also
- `accelerate gradient`

The important architectural decision is:

> AGPT is a training/execution mode, not a replacement model family.

The existing window-based path stays intact. AGPT is added alongside it.

## Goal

Keep the current model body components where possible and introduce a parallel
corpus/loss/trainer path that operates over a weighted prefix trie instead of
sampled token windows.

Near-term objective:

1. build a trie from corpus tokens
2. iterate prefix nodes as training examples
3. compute weighted next-token loss at each prefix
4. reuse the existing model body by replaying each prefix as a token sequence

This first version is correctness-first, not yet the full factorized shared-DAG
execution from the core note.

## Current Window-Centric Assumptions

The current runtime is centered around sequence windows:

- `src/construction_kit/graph.cr`
  expects a windower and extracts `seq_len`
- `src/construction_kit/builder.cr`
  trains by `dataset.sample(seq_len, 0)`
- `src/construction_kit/executable_graph.cr`
  expects boundary inputs like `input_ids`, `target_ids`, and `stream_in`
- `src/microgpt/cooperative.cr`
  and the router stack operate over `[seq_len, ...]`

These should not be removed. AGPT should sit beside them.

## MVP Architecture

### 1. New AGPT Core Objects

Introduce a small AGPT module:

- `MicroGPT::AGPT::TrieNode`
- `MicroGPT::AGPT::TrieCorpus`
- `MicroGPT::AGPT::WeightedNextTokenLoss`
- `MicroGPT::AGPT::Trainer`

These handle trie construction, prefix-node iteration, and weighted local loss.

### 2. Replay-Prefix Execution

For the first working version, the model body can remain window/sequence-based.

Given a trie prefix `p = [t_0, t_1, ..., t_k]`:

- replay `p` through the existing model as a token sequence
- use the final hidden state / final-position logits as the node state
- apply weighted next-token loss using outgoing counts `n(p, x)`

This is slower than the true factorized formulation, but it preserves semantics
and lets the implementation converge safely.

### 3. New Training Contract

Window mode today:

- sample `(input_ids, target_ids)` from a corpus window
- compute cross-entropy against one next-token target per position

AGPT mode MVP:

- iterate `PrefixExample(prefix_tokens, next_token_counts)`
- compute logits for the final prefix position
- apply weighted next-token loss:
  - probability mass is compared to observed outgoing counts
  - local error is proportional to `pi * N_p - n(p, x)`

## What Stays the Same

For MVP, the following can be reused:

- embeddings
- linear layers
- FFN math
- output projection
- optimizer math
- most backend matrix primitives

This means AGPT should begin as a new trainer/loss/data path, not as a full set
of replacement model internals.

## What Changes First

### Must Change

- corpus representation
- batching / training-example iteration
- loss definition
- builder/trainer mode selection

### Can Wait

- full trie-level shared-prefix forward caching
- aggregated Jacobian reuse
- suffix merging
- CUDA lowering for AGPT mode
- full graph-editor component set

## Suggested Rollout

### Phase 1: CLI-Only Correctness

Implement AGPT outside the editor first.

Deliverables:

- build trie from corpus
- run replay-prefix training
- weighted next-token loss
- simple CLI entry for AGPT runs

Success criterion:

- an AGPT run executes end to end on the existing toy corpus

### Phase 2: Builder Integration

Add a mode switch in builder/runtime.

Suggested shape:

- existing `window` mode remains default
- add `agpt` mode
- choose mode from CLI and later from graph components

Success criterion:

- one codebase supports both window and AGPT training paths

### Phase 3: Graph Components

Add only the minimal new semantic components:

- `prefix_trie_corpus`
- `agpt_loss`
- optional `agpt_mode` or trainer-mode selector

Reuse model-body components initially:

- `transformer`
- `cooperative`
- `router`
- `output_head`

Success criterion:

- the editor can define an AGPT training pipeline without replacing all model
  components

### Phase 4: True Factorized Execution

Implement the core note's actual computational win:

- cache shared prefix states
- aggregate suffix gradients before applying shared prefix Jacobians
- treat the trie as a differentiable DAG rather than replayed sequences

Success criterion:

- AGPT is faster and lower-variance than replay-prefix training

## Integration Points In This Repo

### Existing files likely to change

- `src/construction_kit/builder.cr`
  Add mode selection and AGPT-specific trainer plumbing
- `src/construction_kit/graph.cr`
  Add AGPT-oriented pipeline metadata or new semantic source/loss nodes
- `src/construction_kit/cli.cr`
  Later: accept AGPT graph pipelines
- `src/microgpt/main.cr`
  Optional: add AGPT CLI mode for non-editor runs

### New files proposed

- `src/agpt.cr`
- `src/agpt/trie_node.cr`
- `src/agpt/trie_corpus.cr`
- `src/agpt/weighted_loss.cr`
- `src/agpt/trainer.cr`

## Suffix Merging

The current notes understate suffix merging.

Suffix merging should be treated as a later extension, not part of MVP, because
it changes the object from a strict trie toward a more general DAG.

That said, it has two distinct possible benefits:

- storage/computation reduction through shared suffix structure
- model-quality improvement if merging introduces useful regularization or
  better aggregation

Recommendation:

- keep suffix merging explicitly out of Phase 1
- mention it as a Phase 5 extension after basic AGPT correctness and shared
  prefix execution are in place

## MVP Decision Summary

Do now:

- AGPT as a parallel training mode
- replay-prefix execution
- weighted next-token loss
- CLI-first implementation

Do later:

- factorized trie DAG execution
- editor-first AGPT pipelines
- suffix merging
- AGPT-specific CUDA path
