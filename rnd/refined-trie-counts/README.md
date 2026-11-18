---
title: Refined trie counts
kind: experiment
status: active
outcome: n/a
question: >-
  Can AGPT keep a small corpus's trie (its nodes, so its cost) and take the trie's child counts
  from a larger corpus that contains it, as mobius refines a small corpus's count table with a
  larger corpus's counts, and how does that compare with training on the larger corpus's own
  trie or on its text?
answer: ""
opened: 2026-11-18
updated: 2026-11-18
code: main
eval: canonical
family: attention
tags: [trie-structure, count-prior, data, pg19, scaling]
related: [prior-scaling-plain-1g, count-backoff-gate, one-pass-gap, context-tree-orientation]
---

# Refined trie counts

The user, 2026-11-17: most of the useful data lives in the shallow tree, and
mobius uses a larger corpus to refine the counts of a smaller one's tree;
could AGPT use that?

## Why it fits AGPT

AGPT's loss reads the text only through the trie's counts n(c, a), and its
cost is one evaluation per trie node however often the node occurs. So the
counts can come from a larger corpus at no cost per step: the trie keeps its
nodes, and each node's targets get the larger corpus's evidence.

The shallow trie is where this pays. Counted on the first 32M characters of
mobius's PG-19 text (`imm65-200m.txt`), distinct contexts grow much more
slowly than the corpus up to depth 8, and deep contexts mostly occur once
(sharing = positions × depth / nodes, the events each node evaluation covers):

| chars | nodes, depth ≤ 8 | sharing | nodes, depth ≤ 16 | sharing |
|---:|---:|---:|---:|---:|
| 1M | 1.19M | 6.7× | 7.84M | 2.0× |
| 4M | 3.60M | 8.9× | 28.5M | 2.2× |
| 16M | 8.73M | 14.7× | 97.1M | 2.6× |
| 32M | 13.4M | 19.2× | 176.9M | 2.9× |

Up to depth 8 the nodes grow about as corpus^0.7, and the sharing is still
rising at 32M. Depth 16 stays near Shakespeare's 1.9×.

## Setup

- **Data:** PG-19 as mobius normalizes it (96 symbols), linked under
  `data/pg19/` (see `data/pg19/README.md`). A is the first 1,000,000
  characters (40 pieces of 25,000) of the 4.75M split's training text. The
  4.75M, 20M and 100M training texts each contain the one before. Held out:
  the 50 chunks of 5,000 characters every PG-19 split shares.
- **Model:** the d64/L2 attention model (h4, d_ff 256) from one random init
  (seed 42), contexts of up to 8 characters, Adam at a constant 0.003.
- **AGPT arms** replay v2's pd=3 + crown plan through
  `src/tools/agpt_minibatch_train.py` (`trie_plan pd`): 19,999 steps a pass
  on A, a new order each pass. With `trie_targets_corpus` each event's
  target is its context's next-token distribution over the larger text. A
  plan step holds every event of each of its contexts, so with the corpus's
  own counts the soft-target steps are the hard-target steps (checked: the
  first steps agree to 1e-6, and the trajectories then drift apart at the
  rate two hard-target runs on 1 and 2 CPU threads do).
- **Trie sizes up to depth 8:** A has 1,539,197 nodes (8.0M events, 5.2×
  sharing; 1,032,007 contexts seen once). Every one of A's contexts occurs in
  the larger texts; the median count per context is 2 in the 4.75M text and
  7 in the 20M. The 4.75M text's own trie has 4,538,285 nodes (38.0M
  events, 8.4×), 2.95× A's.

| run | trie / text | counts | schedule | passes |
|---|---|---|---|---:|
| agpt-a1m (+ s2) | A | A | AGPT pd=3 | 8 |
| agpt-a1m-ref5m | A | 4.75M | AGPT pd=3 | 8 |
| agpt-a1m-ref20m | A | 20M | AGPT pd=3 | 8 |
| agpt-a1m-ref100m (+ s2) | A | 100M | AGPT pd=3 | 8 |
| agpt-5m | 4.75M | 4.75M | AGPT pd=3 | 3 |
| mb-a1m | A | A | minibatch, 128 windows | 8 |
| mb-a1m-ref100m | A | 100M | minibatch, 128 windows | 8 |
| mb-5m | 4.75M | 4.75M | minibatch, 128 windows | 2 |

Configs: `make_configs.py`; runs: `run.sh`.

## Results

Pending (GPU slot).
