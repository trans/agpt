# Lightning Training — stochastic subtree sampling

## Idea (your sketch)

Training cadence today iterates every root-child subtree in order each super-
epoch: 65 subtrees, one optimizer step (RMSProp in our recipe) each, coverage is deterministic and uniform.

**Lightning Training** replaces that with random sampling: at each step, pick
a random **subtree root** at a random **depth**, compute gradients over that
subtree, do one optimizer step (RMSProp in our recipe), move on. Instead of sampling a sequence (SGD) we
sample a **structural region of the trie**. Each step still aggregates
gradients across many corpus positions sharing that structural region, but
the region's depth/breadth varies stochastically across steps.

Key contrasts:

| | Current per-root-child | Lightning |
|---|---|---|
| sampling | deterministic (all 65) | stochastic |
| step granularity | 1 / depth-1 radix node | variable (depth-d radix node at random d) |
| cross-step coverage | uniform within SE | stochastic, biased by sampling weights |
| steps per SE | 65 | tunable (any N) |

## Why this might work

- **Breaks the 65-way fracturing at partition depth**: unlike bigram/trigram
  partitioning (which always forces a fixed narrow attention scope per
  step), Lightning can do a step over the whole 'T' subtree at d=1 **and**
  a step over just the 'Th' subtree at d=2. Attention scope varies step to
  step. Could recover the both-worlds property.
- **Lets frequency bias come from sampling weights**: sample deeper subtrees
  less often than shallower ones, matching a corpus-mass-weighted
  distribution (the pattern we'd otherwise bake into `--mass-weight`).
- **Natural curriculum**: early training could bias toward shallower
  subtrees (broader coverage, dense branching info), later toward deeper
  (finer-grained local patterns). No hand-tuned depth schedule.

## Sampling design — three variants

### L1: Uniform over all radix nodes

Each training step, sample a random radix node `r` from the trie. Subtree =
`{r} ∪ descendants(r)`. Run forward/backward over that subtree, one Adam
step.

- **Depth distribution**: the radix-profile (most nodes are at the cap, so
  this skews deep).
- **Mass distribution**: sum(edge_mass) across sampled subtree's radix
  nodes. Deep subtrees carry little mass; shallow carry lots.

### L2: Uniform over (root-child, depth)

Sample `rc ∈ {65 root-children}` and `depth d ∈ {1..D}`. Find the unique
radix node on `rc` at depth d (or the one whose edge covers d). Subtree =
its descendants. One RMSProp step.

- **Controllable depth distribution**: can weight d=1 more than d=D.
- **Root-child uniform**: each first-char gets equal sampling, independent
  of corpus mass. (Or weight by root-child mass for frequency-matching.)

### L3: Mass-weighted top-down walk

Start at root. At each level, flip a coin: stop here (emit this subtree) or
descend into a random child weighted by child mass. Emit whichever subtree
we stopped at.

- **Soft curriculum**: stopping probability `p_stop(d)` is a schedule. Early
  training: `p_stop` small (walks deep, smaller subtrees). Late: `p_stop`
  large (more root-level steps, broader coverage). Or reverse.
- **Mass-respecting**: frequent paths get more training exposure.

## Implementation scope

Relatively clean extension of existing per-subtree infra. Changes:

1. **CLI flag** `--lightning-steps N` (or `--lightning-epochs N`): total
   random-subtree samples per super-epoch. Default off (0 = current
   deterministic behavior).
2. **Flag** `--lightning-sampler {l1,l2,l3}` picking variant.
3. **Sampling-time subtree build**: given a random `r`, walk its descendants
   and build a node list (reusing the existing per-root-child grouping
   logic, but starting from `r` instead of a depth-1 root-child).
4. **RMSProp step** — reuse existing single-subtree path. Optimizer state
   persists as usual.
5. **Stats logging**: emit histogram of sampled depths per SE to catch
   accidental bias.

Minor concern: KV cache. Currently scoped per root-child subtree. If
Lightning samples an interior node `r` (depth ≥ 2), attention from `r`'s
descendants needs K/V for `r`'s ancestors (the chain from root to `r`).
Those ancestors aren't in the sampled subtree. Two options:

- **Fresh K/V each step** (safest): compute ancestor K/V inside the step
  before running the subtree forward. Cheap (chain length ≤ D).
- **Skip ancestor attention** (lightweight but not-quite-AGPT-invariant):
  start attention from depth of `r`. We lose cross-depth coherence of the
  ancestor chain for that step.

First-cut would use fresh-K/V for correctness, measure whether the cost
dominates.

## What we'd learn

A Lightning d=16 sweep could answer:

1. Does stochastic variable-depth subtree training **beat** the 65-uniform
   baseline at matched compute? (Main question.)
2. Does mass-weighted sampling (L3) emerge as naturally better than uniform
   (L1/L2) — i.e., does `--mass-weight linear`'s win generalize to the
   sampling distribution?
3. Does a depth schedule (start shallow, go deep, or vice versa) help over
   uniform-across-depths sampling?

This is the first training-schedule experiment in the project that doesn't
inherit the transformer-era "sample a window" paradigm — it's the trie's
native granularity.

## Open design questions

- **Step count per SE**: match baseline's 65? Or go higher (500, 1000) to
  give stochastic coverage time to converge? The bigram LR sweep suggested
  many-small-steps without LR retuning is bad — so probably match step
  count and tune LR comparably to baseline.
- **Variance / seed sensitivity**: stochastic sampling introduces run-to-run
  variance. Need 5+ runs per config to separate signal from noise.
- **Reproducibility**: `--seed` for the sampler.
