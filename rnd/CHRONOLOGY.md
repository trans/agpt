# `rnd/` Chronology

This is an **approximate** timeline of the research threads under `rnd/`.

It is based primarily on:

- when an experiment directory first appeared in git
- the first clearly related commit message when the directory itself was
  backfilled later

It is **not** a perfect reconstruction of thought order. Some threads began in
notes or branches before they got a clean `rnd/<name>/` home, and a few were
revisited across several days.

That is why this file exists instead of numbering the directories `01_*`,
`02_*`, and so on: the order is useful, but not precise enough to encode in
paths.

## Approximate sequence

| Order | First seen | Experiment | Notes |
|---|---|---|---|
| 1 | 2026-04-18 | [convergence](convergence/) | Early tooling + convergence / replication work. |
| 2 | 2026-04-20 | [mass-conservation](mass-conservation/) | Formalization of suffix-tree mass conservation. The directory shell itself was added on 2026-04-21. |
| 3 | 2026-04-21 | [blending](blending/) | Suffix-depth blending around radix endpoints. Directory first appeared in the initial `rnd/` setup pass. |
| 4 | 2026-04-20 | [root-loop](root-loop/) | Virtual-tree / loop-point thread. The directory shell appeared on 2026-04-21, but the related implementation-plan commits start on 2026-04-20. |
| 5 | 2026-04-21 | [sparsity-profile](sparsity-profile/) | Depth-by-depth sparsity characterization. Like `blending`, it was part of the initial `rnd/` setup wave. |
| 6 | 2026-04-21 | [sgd-sanity-check](sgd-sanity-check/) | AGPT vs SGD sanity check, including mass-weight sweeps. |
| 7 | 2026-04-22 | [radix-saturation](radix-saturation/) | Depth / saturation behavior under AGPT training. |
| 8 | 2026-04-22 | [lightning-training](lightning-training/) | Lightning L1/L2/L3 subtree-sampling training sweeps. |
| 9 | 2026-04-23 | [post-fix-baseline](post-fix-baseline/) | Baseline re-establishment after the `Wk` / `Wv` / bias-gradient fix in commit `1c858c0`. |
| 10 | 2026-04-24 | [hotspot-curriculum](hotspot-curriculum/) | Adaptive subtree-splitting curriculum based on loss concentration. |
| 11 | 2026-04-24 | [sgd-ceiling](sgd-ceiling/) | How much of AGPT's edge comes from optimizer choice vs aggregation. |
| 12 | 2026-04-25 | [agpt-optimizers](agpt-optimizers/) | Post-fix optimizer sweep showing subtree AGPT needs adaptive optimization. |
| 13 | 2026-04-25 | [wrap-around](wrap-around/) | Wrap-around corpus synthesis and long-context recipe work. |
| 14 | 2026-04-25 | [unary-pruning](unary-pruning/) | Mass-1 unary-path pruning. |
| 15 | 2026-04-26 | [gutenberg-5m](gutenberg-5m/) | Larger-corpus builder + AGPT-direct / wrap-around scaling work on Gutenberg 5M. |
| 16 | 2026-04-27 | [lightning-cap-warmup](lightning-cap-warmup/) | Mass-cap + ancestor-warmup Lightning follow-up. Closed the same day it was indexed and then pivoted onward. |
| 17 | 2026-04-27 | [p2s-attention](p2s-attention/) | Prefix-to-suffix attention and structural matching investigation. |
| 18 | 2026-04-28 | [trie-attention-framing](trie-attention-framing/) | Decision/identity decomposition framing for the trie, including `d*`, depth-routing, and decision-only operationalizations. |

## Phase view

If a coarser grouping is more useful than exact order, the work roughly falls
into five phases:

### Phase 1: initial AGPT measurement / theory cleanup

- `convergence`
- `mass-conservation`
- `blending`
- `root-loop`
- `sparsity-profile`

### Phase 2: AGPT vs SGD and Lightning sweeps

- `sgd-sanity-check`
- `radix-saturation`
- `lightning-training`

### Phase 3: post-fix re-baselining and subtree curriculum

- `post-fix-baseline`
- `hotspot-curriculum`
- `sgd-ceiling`
- `agpt-optimizers`

### Phase 4: wrap-around and scaling

- `wrap-around`
- `unary-pruning`
- `gutenberg-5m`

### Phase 5: Lightning cap follow-up and the P2S pivot

- `lightning-cap-warmup`
- `p2s-attention`

### Phase 6: trie-attention reframing

- `trie-attention-framing`

## Formerly unfiled thread: radix-cap split / decision-only loss

The trainer thread that previously looked unfiled now has a proper experiment
home at [trie-attention-framing](trie-attention-framing/).

This appears to be the line you were remembering:

- `d_split` / "real radix point" analysis
- depth-routed `Wk` / `Wv` gradient masking
- decision-only loss via `AGPT_DECISION_ONLY` and `AGPT_DECISION_BUFFER`

So the earlier ambiguity is resolved:

- it was **not** `p2s-attention`
- it is now explicitly captured as `trie-attention-framing`
- the code-level hooks in `src/cuda/agpt_train.cu` now have a matching `rnd/`
  experiment record
