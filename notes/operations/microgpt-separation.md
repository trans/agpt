# AGPT / µGPT Separation Map

*Snapshot: 2026-04-28*

This is the current coupling map for AGPT as its own repo.

The goal is **not** to eliminate all dependency on µGPT. The goal is to make
the dependency boundary explicit and narrow:

- **µGPT owns**: model/runtime primitives, math/backend code, shared CUDA
  kernels, generic eval tools.
- **AGPT owns**: trie/radix structures, AGPT trainers, AGPT research CLIs,
  AGPT notes, AGPT experiments.

That boundary matters even if more new experiments move to Python/PyTorch.
Without it, AGPT still pays a constant organizational tax every time we change
either the Crystal/CUDA stack or the research direction.

## Current coupling, classified

### Intentional dependency: µGPT as runtime library

AGPT currently and reasonably depends on µGPT for model/runtime primitives:

- `MiniGPT`
- `Mat`
- `Config`
- `RoPE`
- backend dispatch / matmul / layer-norm / softmax

This is visible immediately in [src/agpt.cr](/home/trans/Projects/agpt/src/agpt.cr:1),
which starts with:

```crystal
require "microgpt"
```

and then defines AGPT under:

```crystal
module MicroGPT
  module AGPT
```

This dependency is fine in principle. What is still undecided is whether AGPT
should keep the nested `MicroGPT::AGPT` namespace long-term, or present a
top-level `AGPT` namespace with compatibility aliases.

### Intentional dependency: shared CUDA and build stubs

The `Justfile` intentionally compiles AGPT against CUDA sources from the shard:

- `lib/microgpt/src/cuda/kernels.cu`
- `lib/microgpt/src/cuda/stubs.c`

This is a valid shared-components boundary as long as AGPT does not require
AGPT-specific code to live back inside the µGPT repo.

### Intentional dependency: reference binaries for comparison

AGPT foundational tests and some experiments use:

- `bin/microgpt`
- `bin/perplexity`

from the µGPT shard as the window-training / evaluation reference.

This is acceptable, but it should be treated as **reference tooling**, not as
evidence that AGPT itself still belongs inside the µGPT app surface.

### Accidental / historical coupling: AGPT identity still inherits µGPT shape

There are still several places where AGPT behaves like a carved-out subsection
of the old mono-repo rather than a clean dependent project:

- AGPT is namespaced as `MicroGPT::AGPT`, not clearly as its own surface.
- Notes still describe the split as unfinished and partly conceptual.
- Tests and build targets still normalize µGPT binaries as part of the default
  AGPT developer workflow.
- Some docs still frame AGPT through the old `bin/microgpt` era rather than
  through AGPT-owned entrypoints.

### Accidental / historical coupling: instrumentation APIs bleed through

AGPT directly references µGPT-specific instrumentation/backend types such as:

- `MicroGPT::PerfTrace`
- `MicroGPT::CuBLASBackend`

These show up heavily in:

- [src/agpt/batched_depth_forward.cr](/home/trans/Projects/agpt/src/agpt/batched_depth_forward.cr:1)
- [src/agpt/batched_depth_backward.cr](/home/trans/Projects/agpt/src/agpt/batched_depth_backward.cr:1)
- [src/agpt/trie_walk_trainer.cr](/home/trans/Projects/agpt/src/agpt/trie_walk_trainer.cr:1)

This is not fatal, but it means AGPT depends on more than "model + tensor
runtime". It depends on µGPT's profiling vocabulary and backend class names.

## What the split is actually blocked on

The repo is **not** blocked on removing the shard dependency.

It **is** blocked on four more specific things.

### 1. Identity boundary

AGPT should read as:

- an AGPT repo
- with AGPT-owned CLIs
- with µGPT as a dependency

not as a special mode of µGPT.

Concrete symptom:

- AGPT code still lives under `MicroGPT::AGPT`.

### 2. Runtime contract boundary

Right now AGPT reaches into µGPT at many levels:

- tensor/matrix type
- model type
- backend singleton
- backend class checks
- perf tracing
- CUDA kernels
- reference binaries

Some of those are core contract and some are convenience leakage. They need to
be separated into:

- **hard dependency contract**
- **optional tooling contract**

### 3. Comparison-tooling boundary

`bin/microgpt` and `bin/perplexity` are useful, but they should be clearly
boxed as:

- baseline reference tools
- not AGPT runtime requirements for ordinary AGPT development

That means AGPT's build/test surface should distinguish:

- AGPT-native build/test
- comparison / parity / benchmark build/test

### 4. Notes-and-experiments boundary

Because the split stayed fuzzy, research notes now mix:

- AGPT design
- µGPT implementation references
- historical split assumptions
- experiment conclusions

This makes it harder to tell which notes describe live architecture versus
which only describe the old development topology.

## Recommended order of work

### Phase 1: finish the identity split

Low risk, high leverage:

1. Fix stale naming in tests/docs/build comments when they still talk as if
   AGPT were the `microgpt` repo.
2. Make AGPT-vs-reference-tooling explicit in the `README` / `Justfile`.
3. Keep AGPT notes in AGPT terms, and refer to µGPT only as a dependency.

This does not change any code architecture but reduces ambient confusion.

### Phase 2: define the runtime contract

Write down the minimal supported µGPT surface AGPT expects:

- tensors / `Mat`
- model / `MiniGPT`
- `Config`
- RoPE
- backend ops
- shared CUDA kernels

Then separately mark:

- `PerfTrace`
- backend-class checks like `CuBLASBackend`
- reference CLIs

as optional or convenience-level coupling.

The output of this phase should be a note or small compatibility layer, not a
large refactor yet.

### Phase 3: separate AGPT-native tasks from reference tasks

In practice:

- AGPT-native builds should compile AGPT tools and trainers.
- Reference builds should compile `bin/microgpt` / `bin/perplexity`.
- AGPT-native tests should not silently imply comparison binaries unless they
  are specifically parity tests.

This is the point where the repo starts to feel truly separate even while still
using the shard.

### Phase 4: decide namespace strategy

Two viable end states:

1. Keep `MicroGPT::AGPT`.
   Pros: minimal churn.
   Cons: AGPT still reads like a sub-product.

2. Move to top-level `AGPT` with compatibility aliases where needed.
   Pros: clean repo identity.
   Cons: broader rename surface.

This should happen **after** the runtime contract is written down, not before.

## Immediate next tranche

If separation is the current priority, the next tranche should be:

1. Update `README.md` and `Justfile` language so AGPT-native vs reference
   tooling is explicit.
2. Add or refine one note describing the minimal µGPT contract AGPT expects.
3. Split tests conceptually into:
   AGPT-native checks vs reference-parity checks.
4. Only then decide whether a namespace change is worth the churn.

## Non-goal

Do not try to sever AGPT from µGPT entirely unless there is a real plan to
replace the Crystal/CUDA runtime. If future research shifts toward
Python/PyTorch, that is a reason to make the existing boundary cleaner, not a
reason to thrash the current implementation boundary immediately.
