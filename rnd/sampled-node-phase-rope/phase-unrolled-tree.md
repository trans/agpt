# Phase-Conditioned Prefix Trees

Status: research sketch, 2026-06-02

## Short Version

AGPT's current trie pools all occurrences of the same prefix into one node. A
prefix `P` has one global mass and one global next-token distribution.

The new idea is to lift that node into a phase fiber:

```text
P
```

becomes:

```text
P@0
P@1
...
P@(W-1)
```

where `W` is the RoPE/window modulus and `P@q` means "prefix `P` ending at
node/current-token phase `q`." The ordinary trie is then the pooled projection
of this phase-conditioned trie:

```text
mass(P) = sum_q mass(P@q)
```

This does not require physically materializing `W` copies of the trie. The
implementation should keep the base trie and store sparse phase-fiber
statistics:

```text
base node: radix_id for P
fiber coordinate: q in Z/WZ
fiber mass: mass(P, q)
```

This is not merely a uniform RoPE offset sweep. A uniform sweep over the same
tree:

```text
depths 0..15
depths 1..16
depths 2..17
```

does not add much by itself because RoPE is fundamentally relative. The useful
extra signal comes from coupling the presentation phase to phase-conditioned
mass and eventually phase-conditioned target distributions.

## Current Trie

The current prefix trie represents pooled prefix statistics:

```text
node P:
  mass(P) = count of all corpus occurrences of P
  target(P) = global next-token counts after P
```

For example:

```text
P = "the "
target(P):
  "c" -> 120
  "m" -> 97
  "s" -> 84
```

This is strong statistically, but it destroys information about where those
occurrences happened. If the continuation distribution differs by local corpus
position or RoPE phase, the normal trie pools that structure away.

## Existing Position Histogram

The current position-table artifacts already store per-prefix phase mass, but
indirectly:

```text
radix_id -> substring_id -> { position_mod_W: count }
```

Relevant artifacts:

```text
prefix_radix_to_substring.bin
prefix_position_table.bin
```

The histogram is per prefix/substring, not per token. For a prefix `P`, it
stores:

```text
mass(P, q) = count of corpus occurrences of P where start_position % W == q
```

For common prefixes this is spread across phases. For rare prefixes it may be
concentrated in only one or two phases.

The current table records prefix start phase. If a training formulation wants
the phase of a token at local depth `d`, then it must apply a consistent shift:

```text
node_position_phase = (start_phase + d) % W
start_phase = (node_position_phase - d) % W
```

The precise convention matters and should be fixed before implementation.

## Resolved Phase Convention

Use current token/node position phase, also called endpoint phase:

```text
phase(P) = absolute corpus position of the last token of P modulo W
```

For a transition by next token `t`:

```text
P@q --t--> (P + t)@(q + 1 mod W)
```

This convention keeps the prediction step natural: the target token lives at
phase `q + 1`.

The current position table is stored by prefix start position. If a prefix has
length/depth `d`, convert once when loading or building the runtime table:

```text
endpoint_phase = (start_phase + d - 1) % W
```

After that conversion, trainer code should use endpoint/node phase directly
instead of carrying start-phase arithmetic through the inner loop.

## Phase-Conditioned Tree

Conceptually, construct a larger tree whose state includes phase:

```text
root
  token_a@0
    token_b@1
      token_c@2
  token_a@1
    token_b@2
      token_c@3
```

Equivalently, for each original prefix `P`, create one node per phase that has
mass:

```text
mass(P@q) = mass(P, q)
```

The child relation advances phase:

```text
P@q --token t--> (P + t)@(q + 1 mod W)
```

This is the "unpooling" view. The original trie is recovered by summing over
phases:

```text
P = pool_q P@q
```

The phase-conditioned tree stretches the statistical structure out to `W`, not
just to the usual trie depth `D`. It exposes how mass flows through positions
rather than collapsing all positions for the same prefix into one node.

This "unrolled" picture is conceptual. In implementation, prefer the fibered
form:

```text
radix_id -> sparse { phase -> mass }
```

## Training Interpretation

A simple presentation for phase `p` is:

```text
RoPE position for local depth d = p + d
loss/gradient weight = mass(prefix_at_depth_d, p + d)
```

Under the endpoint-phase convention, the runtime phase of a prefix node is:

```text
presentation_position = presentation_phase + local_depth
node_phase = presentation_position % W
weight = mass(prefix, node_phase)
```

The modulo reduction belongs to the corpus-position histogram coordinate, not
to the RoPE presentation span. A depth-16 path in a `W=64` context may be
presented at starts `0..48`, giving contiguous spans `0..15` through `48..63`.
Starts `49..63` would wrap the RoPE positions (`49..63,0`, etc.) and should not
be used for this presentation experiment.

The model context length must remain the trie depth for this experiment. The
RoPE cache/window may be larger than the attention path, but it must not change
the checkpoint `seq_len` or fixed-token evaluation context:

```text
train.max_depth = 16
checkpoint/eval seq_len = 16
rope_cache_len = W
attention path length <= 16
```

The conversion from start-phase histogram to endpoint-phase histogram can
happen before this lookup, or the trainer can convert consistently at lookup
time.

## Target Distributions

There are two levels of sophistication.

### 1. Phase-Weighted Global Targets

Use phase-conditioned node mass, but keep the target distribution global:

```text
loss_weight(P@q) = mass(P@q)
target(P@q) = target(P)
```

This is a useful "slightly dumbed down" version. It can be implemented with the
current global `counts_tok/counts_val` tables and the per-prefix position
histogram.

It teaches the model that the same prefix appears with different weight under
different RoPE phases, but it does not teach phase-specific continuation
probabilities.

### 2. Phase-Conditioned Targets

Use both phase-conditioned mass and phase-conditioned continuation
distributions:

```text
target(P@q):
  token t -> mass((P + t)@(q + 1))
```

For interior trie nodes, this can be inferred from child prefix histograms:

```text
mass(P + t, q + 1)
```

For capped nodes at max depth `D`, the child prefix `P + t` is not emitted by
the current trie. The global next-token counts still exist on the node:

```text
counts_tok/counts_val
```

but the phase-conditioned target counts do not:

```text
node_id -> token -> phase -> count
```

To condition cap targets properly, we need either:

```text
1. direct endpoint target histograms:
   node_id -> token -> { phase: count }
```

or:

```text
2. synthetic/unmaterialized depth-(D+1) child histograms:
   P + token -> { phase: count }
```

Without one of those, cap targets can only use the phase-weighted global-target
variant.

## Experiment Matrix

Test four variants so each effect is separated:

### A. Baseline Pooled Trie

```text
weight = mass(P)
target = target(P)
RoPE = ordinary depth positions
```

This is the current baseline.

### B. Uniform Phase Sweep

```text
weight = mass(P)
target = target(P)
RoPE phase varies
```

This tests whether a RoPE offset by itself matters. Expected result: little or
no improvement, because a uniform RoPE shift preserves relative positions.

### C. Phase-Weighted Global Target

```text
weight = mass(P@q)
target = target(P)
RoPE phase varies with q
```

This is the first serious experiment. It uses phase-conditioned mass but keeps
the existing global `counts_tok/counts_val` target distribution. Cap nodes are
not special in this variant.

### D. Phase-Conditioned Target Where Available

```text
weight = mass(P@q)
target = target(P@q)
RoPE phase varies with q
```

For interior nodes, target(P@q) can be estimated from child phase masses:

```text
token t -> mass((P + t)@(q + 1))
```

For cap nodes, this requires direct endpoint target histograms or a helper
depth `D+1` structure. This should come after variant C shows whether the phase
mass signal is useful.

## Why This May Matter

The ordinary trie is high-statistics but coarse. The phase-conditioned trie is
lower-statistics but higher-resolution.

For a prefix `P`, the global tree sees:

```text
P -> {a: 10, b: 10}
```

The phase tree may reveal:

```text
P@7  -> {a: 9, b: 1}
P@22 -> {a: 1, b: 9}
```

The global model cannot distinguish those cases; the phase-conditioned view can.
This may let AGPT exploit structure that SGD over flat windows sees naturally
through position, but the pooled trie has been discarding.

## Multi-Resolution Training

The coarse and phase-conditioned trees are not mutually exclusive. A good training
schedule might use both:

```text
coarse/global passes:
  stable pooled statistics

phase-conditioned passes:
  position-conditioned refinements

deeper/synthetic cap passes:
  finer continuation distributions where useful
```

This resembles progressive training, but the axis is not only depth. It is also
resolution: pooled mass versus phase-split mass.

Possible schedules:

```text
1. pretrain on the ordinary global trie, then fine-tune on phase-conditioned mass
2. alternate global and phase-conditioned epochs
3. start with phase-weighted global targets, then add phase-conditioned targets
4. use phase-conditioned weighting only for prefixes with enough per-phase mass
```

## Open Questions

1. Should implementation store endpoint-phase histograms directly?

   The research convention is endpoint/node phase. The current artifact stores
   start phase. Runtime can translate on load, but new artifacts may be clearer
   if they store endpoint phase directly.

2. Should zero phase mass remove a node from the fire?

   If phase mass is zero, the loss/gradient contribution should be zero. The
   tree structure itself still exists; zero mass is not a signal to fall back to
   global mass.

3. How should cap targets be handled?

   Options:

   - keep global target distribution and phase-condition only the weight
   - build direct endpoint target histograms
   - materialize or synthesize depth `D+1` histograms

4. How sparse is the phase-conditioned tree?

   Common prefixes should span most phases. Rare prefixes will be highly
   localized. This may be a feature, but it affects optimizer schedule and
   epoch count.

5. Should phase resolution be fixed at RoPE window `W`?

   We may want coarser resolutions first:

   ```text
   W=8, W=16, W=32, W=64
   ```

   This would provide a progressive resolution schedule.

6. How does this interact with RoPE relativity?

   A pure uniform offset sweep is redundant. The useful signal comes from the
   phase-conditioned mass and targets attached to that offset.

## Minimal Implementation Path

1. Rename or separate the current mode in code:

   ```text
   phase-sweep
   phase-weighted
   phase-targets
   ```

   The current `sampled-unit-phase` implementation mixed these concepts and
   tried to reconstruct targets from child phase mass. The first implementation
   should separate the modes.

2. Implement variant B as a quick control:

   ```text
   weight(query) = global mass
   target(query) = existing counts_tok/counts_val
   RoPE positions = phase-shifted endpoint positions
   ```

3. Implement variant C as the first main experiment:

   ```text
   weight(query) = mass(prefix, endpoint_phase)
   target(query) = existing counts_tok/counts_val
   ```

   This avoids the capped-node target issue and tests whether the phase mass
   signal helps at all.

4. Add diagnostics:

   - mass conservation over phases
   - per-phase total training mass
   - zero-mass query counts by depth
   - cap/non-cap split

5. If promising, add endpoint target histograms:

   ```text
   node_id -> token -> phase -> count
   ```

   This enables true phase-conditioned target distributions even at depth cap.

6. Consider a multi-resolution schedule:

   ```text
   global trie -> W=8 phase tree -> W=16 -> W=64
   ```

## Current Caution

An earlier implementation attempt incorrectly treated phase mass as a hard
absolute-position filter and tried to reconstruct all endpoint targets from
child histograms. That is not the full idea.

Correct framing:

```text
phase is part of the presented/unrolled training state
mass is conditioned by phase
targets may be global first, then phase-conditioned when data exists
zero phase mass means zero contribution, not fallback
```
