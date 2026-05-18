# AGPT Trainer Structure and K/V Staleness Analysis

Date: 2026-05-18

This note captures a static analysis of `src/cuda/agpt_train.cu` with two goals:

1. identify cleaner interface boundaries for the trainer
2. clarify what is and is not happening with K/V cache staleness

No code changes were made as part of this analysis.

## Executive Summary

The file is hard to reason about mostly because it is carrying:

- two trainer families
- several different scheduling semantics
- cache logic
- chunk materialization
- optimizer stepping and persistence
- a growing set of experiment flags

all in one place.

On staleness, the current default path is less mysterious than it looks:

- with `--accumulate` on, the code is mostly structured to avoid post-update stale K/V reuse inside a training unit
- with `--no-accumulate`, real stale-after-update reuse comes back
- a separate issue is that ancestor K/V is not yet fully differentiated through in backward, which is not the same thing as staleness

## Main Findings

### 1. The file contains two trainers

`src/cuda/agpt_train.cu` still contains:

- the older leveled-trie path with `TrainState`, `allocate_train_state()`, and `train_epoch()`
- the newer radix trainer with `run_radix_training()`

Relevant locations:

- `TrainState`: `src/cuda/agpt_train.cu:2892`
- `allocate_train_state()`: `src/cuda/agpt_train.cu:2993`
- `train_epoch()`: `src/cuda/agpt_train.cu:3185`
- `run_radix_training()`: `src/cuda/agpt_train.cu:3948`
- dispatch in `main()`: `src/cuda/agpt_train.cu:7441`, `src/cuda/agpt_train.cu:7488`

This is the first thing making the file feel larger and more confusing than the current AGPT surface actually is.

### 2. "Subtree" means too many things

The same machinery is used for:

- root-child AGPT training units
- `partition_depth` groups
- Lightning samples
- per-subtree files in the manifest path

Most of that flows through:

- `subtree_nodes`
- `subtree_sizes`
- `subtree_n_anc`
- `n_root_children`

Relevant locations:

- root-child grouping: `src/cuda/agpt_train.cu:4533`
- single-subtree regrouping: `src/cuda/agpt_train.cu:4688`
- n-gram partitioning: `src/cuda/agpt_train.cu:4737`
- Lightning resampling: `src/cuda/agpt_train.cu:5081`
- training-unit loop: `src/cuda/agpt_train.cu:5461`

This is the biggest semantic confusion point in the file.

### 3. Experimental behavior is split between CLI flags and hidden env vars

The visible command-line surface is already large, but a second control surface exists via env vars inside `run_radix_training()`:

- `AGPT_DEPTH_ROUTE_K`
- `AGPT_DEPTH_ROUTE_PERLEAF`
- `AGPT_DECISION_ONLY`
- `AGPT_DECISION_BUFFER`
- `AGPT_JOINT_MASS`
- `AGPT_SUBTREE_DROPOUT`
- `AGPT_BRANCH_DROPOUT_DEPTH`
- `AGPT_CHAR_SUFFIX_MASS_PATH`

Relevant section:

- `src/cuda/agpt_train.cu:3976`

That makes it harder than it should be to answer “what exact training mode did this run use?”

### 4. Chunk metadata construction is mixed directly into the math loop

The host-side chunk builder is a large subsystem embedded directly in the training loop. It builds and uploads:

- radix ids
- query offsets
- kv offsets / lengths
- ancestor ids / offsets / lengths
- token ids
- RoPE positions
- query depths
- per-leaf `d_split`

Relevant area:

- `src/cuda/agpt_train.cu:5560`

It also owns several internal static device scratch caches:

- `d_anc_ids_cache`
- `d_anc_offsets_cache`
- `d_anc_lengths_cache`
- `d_own_lengths_cache`
- `d_read_pos_flat_cache`
- `d_query_depth_cache`
- `d_query_d_split_cache`

Relevant area:

- `src/cuda/agpt_train.cu:5654`
- `src/cuda/agpt_train.cu:5705`

This is a natural extraction boundary.

### 5. The default semantics are explicitly trying to avoid stale K/V

The code is written as if stale cache is the main thing to avoid.

The CLI defaults to:

- `accumulate = true`

with the explicit explanation:

- accumulate gradients across all splits and partition groups within a training unit
- fire one optimizer step at the end
- avoid K/V staleness from firing mid-subtree

Relevant locations:

- default + comment: `src/cuda/agpt_train.cu:7122`
- help text: `src/cuda/agpt_train.cu:7287`

Inside the radix loop, gradients are accumulated across chunks and the optimizer fires only at the end of the unit when accumulate is off, or even later at epoch end when accumulate is on:

- zero once at epoch top in accumulate mode: `src/cuda/agpt_train.cu:5384`
- no optimizer step inside chunk loop: `src/cuda/agpt_train.cu:6266`
- per-unit fire site: `src/cuda/agpt_train.cu:6277`
- epoch-end accumulate fire: `src/cuda/agpt_train.cu:6391`

So in the current default path, weights are mostly fixed across the forward/backward work that shares cache.

### 6. Structural freshness is enforced by ordering

The code also enforces “ancestor-before-descendant” ordering so a descendant does not read missing prefix K/V.

This shows up in:

- root-child subtree BFS/depth ordering: `src/cuda/agpt_train.cu:4564`
- Lightning ancestor prepending: `src/cuda/agpt_train.cu:5229`
- hotspot split re-sort: `src/cuda/agpt_train.cu:6664`

The comments are explicit that this ordering is required for the K/V cache ordering invariant.

That solves:

- descendant reads of zeroed cache
- descendant reads before same-weight ancestor scatter

### 7. Real stale-after-update reuse returns in `--no-accumulate`

The code’s own help text says this directly:

- `--no-accumulate` reintroduces K/V staleness

Relevant location:

- `src/cuda/agpt_train.cu:7290`

Why:

- some partition group fires the optimizer
- later groups can still read ancestor cache entries written before the step
- those cache entries exist, but they reflect old weights

The ordering invariant only guarantees presence, not current-weight freshness.

### 8. Lightning is sample-fresh, not globally fresh

Lightning forces:

- `accumulate = false`

Relevant location:

- `src/cuda/agpt_train.cu:4921`

But each sampled training unit rebuilds its own node set and explicitly prepends ancestors before descendants:

- `src/cuda/agpt_train.cu:5229`

So the freshness unit in Lightning is the sample itself:

- current sample ancestors are recomputed first
- descendants in that same sample read those fresh entries

That explains why Lightning can still behave sanely despite step-per-sample updates.

### 9. The bigger caveat may be incomplete ancestor K/V backward, not staleness

There is an important separate limitation:

- forward attention reads ancestor K/V from the compact cache
- backward does not fully propagate ancestor K/V gradients through that cache

The current radix path says this explicitly:

- ancestor-portion `dK/dV` is still dropped
- cross-chunk scatter-add into the compact cache is not implemented

Relevant location:

- `src/cuda/agpt_train.cu:6185`

This is different from stale cache. It means the cache is partly acting like a semi-detached context store rather than a fully live differentiable structure.

There is an older comment in the leveled trainer making the same broader point in rougher form:

- `Wk/Wv` grads were approximate because full multi-depth accumulation was not implemented

Relevant locations:

- `src/cuda/agpt_train.cu:3053`
- `src/cuda/agpt_train.cu:3604`

## Recommended Decomposition

The cleanest split is not “kernels vs non-kernels.” The cleanest split is by trainer role.

### A. Separate the two trainer families

Split the file into:

- `trainer_leveled.cu`
- `trainer_radix.cu`

This removes a lot of stale conceptual overlap immediately.

### B. Introduce a `TrainingUnit` abstraction

Right now `subtree_*` arrays hide too many meanings.

A `TrainingUnit` should explicitly contain:

- node list
- size
- prepended-ancestor count
- kind

Suggested kinds:

- `RootChild`
- `PartitionGroup`
- `LightningSample`
- `HotspotSplit`
- `ManifestSubtree`

This would make the scheduler logic much easier to reason about.

### C. Split config into two surfaces

Keep two structures:

- `TrainOptions`
- `ExperimentalFlags`

`TrainOptions`:

- optimizer
- lr schedule
- partition depth
- accumulate
- chunk size
- curriculum
- save cadence

`ExperimentalFlags`:

- depth-route variants
- decision-only
- joint-mass
- subtree/branch dropout
- virtual-cycle extras

This would make runs more inspectable and reduce “invisible mode” confusion.

### D. Extract a `ChunkBuilder`

The host-side chunk materialization logic around:

- `src/cuda/agpt_train.cu:5560`

should become its own subsystem.

Responsibilities:

- node-to-query expansion
- ancestor / own-edge split
- RoPE position generation
- routing metadata
- device upload of chunk descriptors

This is a good interface because it is already conceptually separate from the math.

### E. Extract a `CacheRuntime`

This module should own:

- compact cache buffers
- gather/scatter policy
- delta-RoPE gather behavior
- cache ordering invariants

Right now those assumptions are spread across comments and loop structure.

### F. Extract a `RadixStepEngine`

One step engine should be responsible for:

- forward chunk
- loss
- backward chunk
- stats accumulation

It should consume a built chunk descriptor and runtime state, not rebuild planning metadata itself.

### G. Extract an `OptimizerDriver`

This should own:

- step counters
- LR schedule
- warmup horizon
- grad clipping
- weight decay
- per-rc optimizer state
- persistence I/O

That logic is currently distributed between:

- `run_radix_training()`
- `run_per_subtree_training()`
- `main()`

## Practical Cleanup Order

If doing this incrementally, the highest-signal order looks like:

1. split leveled vs radix into separate files
2. extract `ExperimentalFlags`
3. replace `subtree_*` arrays with `TrainingUnit`
4. extract `ChunkBuilder`
5. extract `OptimizerDriver`

That would improve readability substantially without forcing a redesign of the math.

## Mental Model for Staleness

It helps to separate three different questions.

### 1. Structural freshness

Question:

- are the needed ancestor K/V entries present before a descendant reads them?

Answer:

- mostly yes

Why:

- subtree ordering and ancestor prepending are explicitly enforcing this

### 2. Weight freshness

Question:

- were those cache entries computed under the same weight snapshot as the current query?

Answer:

- yes in default accumulate semantics within a training unit
- risky in `--no-accumulate` partitioned training

### 3. Gradient completeness

Question:

- if an ancestor K/V influenced attention, does backward fully propagate through it?

Answer:

- not fully in the current compact-cache radix path

This third issue should be kept conceptually separate from staleness.

## Bottom Line

If running the current default `--accumulate` path, post-update cache staleness is probably not the main mystery anymore. The code is mostly structured to avoid it.

The more likely sources of “why does this work as well as it does?” are:

- structural freshness within a fixed-weight training unit
- reuse of ancestor K/V as a same-snapshot context store
- partial rather than full differentiation through ancestor cache entries

So the current situation is not:

- “the code is mysteriously surviving rampant stale K/V”

It is closer to:

- “the code has become complicated because it is trying to avoid stale K/V, while still accepting some approximation in the backward treatment of ancestor K/V”

## Addendum: 2026-05-18 deep-dive findings

Added by the main agent after the secondary-agent analysis was written.
Three findings from the same day's investigation that refine the
staleness picture above.

### A. Held-out evaluation: model wins 34× vs trie alone

To settle whether the model is doing real generalization work (vs being
a redundant lossy compressor of the trie), we built a 90/10 train/test
split of Gutenberg 5M and evaluated both a trained model and a
trie-only baseline on the held-out 10%.

| System | Held-out PPL |
|---|---:|
| Trained model (108K params, 100 SE on 90% train) | 5.03 |
| Trie alone (count lookup + naive backoff) | 170.4 |

Trie diagnostics: 21,279 backoff invocations across 4096 positions;
349 root-only fallbacks (8.5%); only 28 mid-edge. The trie has
essentially no useful information for unseen 16-char contexts at
d=16 on a 5M corpus.

Caveat: the trie-PPL evaluator uses naive backoff. Proper KN would
likely report trie PPL ~15-30 instead of 170; the qualitative
"model >> trie" finding is robust, the 34× magnitude is overstated
by ~5-10×. See `todo/trie-ppl-kn-backoff.md`.

Full writeup: `rnd/heldout-tree-vs-model/findings.md`.

### B. Shuffle ablation at pd=1: shuffle does not help

Tested whether `--shuffle-order` helps at pd=1 (where, by the
cache-disjointness argument below, there should be no real
staleness for shuffle to fix).

| Setup | Held-out PPL (single seed) |
|---|---:|
| pd=1 no-shuffle | 5.03 |
| pd=1 with shuffle | 5.29 |

Shuffle did not help at pd=1 (and slightly hurt, though single-seed
noise makes the +5% non-significant). Combined with earlier
cap-folding findings that shuffle DOES help at pd>1, this supports:

- Shuffle's pd>1 benefit is staleness-specific (compensating for
  cross-group cache pollution)
- Shuffle is NOT general gradient-noise regularization
- pd=1 may genuinely have no real staleness to fix

### C. Cache disjointness at pd=1 (claim, not fully verified)

The analysis above describes `--no-accumulate` as bringing back
"real stale-after-update reuse" without distinguishing pd=1 from
pd>1. A structural argument that pd=1 is special:

- At pd=1, each subtree = one root-child + all its descendants
- A query within subtree A has ancestors entirely within subtree A's
  node set (the chain from query up to root-child A; root has no K)
- Subtree A's queries only gather from cache positions that are
  trie IDs in subtree A
- These cache positions were written during subtree A's own forward
  pass under the current Wk (set at start of A's fire)
- After subtree A's optimizer step, Wk changes. The K values from
  subtree A still sit in the cache at A's positions but are never
  read by any other subtree (B's queries only gather B's positions)
- At SE boundary, cache is zeroed; next SE starts fresh

If this argument is correct, the entries at A's positions ARE stale
after A's step (they reflect old Wk), but nobody reads them, so they
don't hurt training. Subtree B reads positions in subtree B's node
set, freshly written under post-A-step Wk.

The shuffle ablation result (B) is empirical evidence consistent
with this: if pd=1 has no real staleness, shuffle has nothing to
fix, and indeed it doesn't help at pd=1.

Verification still pending: trace `ancestor_ids` at runtime for a
sample of pd=1 queries, confirm all ancestor IDs are within the
firing subtree's node-ID set. If verified, this is a meaningful
qualifier to the analysis above. If not, the conservative framing
("staleness comes back at --no-accumulate") is more accurate and
my pd=1 argument is wrong somewhere.

### D. Updated todo priorities

The descendant→ancestor scatter (this analysis's finding #9) is no
longer blocked by phantom staleness concerns — at least at pd=1.
See `todo/descendant-ancestor-scatter.md` for the implementation
plan and falsifiable hypothesis.
