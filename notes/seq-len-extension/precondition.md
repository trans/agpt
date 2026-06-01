# Precondition Strand

## Status

Plumbing landed. Encoder + injection mechanism in progress.

## The idea

Decompose effective context into two pieces:

1. **Precondition strand**: a `d_pre`-char prefix processed by a sequential
   encoder (GRU initially; Mamba upgrade path for longer prefixes) → produces
   a single hidden state per query position.
2. **Choice tensor tree**: the regular AGPT trie, but conditioned on the
   precondition state instead of starting from an unconditioned root. The
   trie sees the next `d` chars and predicts as usual.

Effective context: `d_pre + d` chars. The precondition is content-bearing
(it's a learned summary of the prefix) rather than positional (a different
RoPE position over the same nodes). That's the structural difference vs.
"just stack two trees' worth of K slots."

## Per-K aggregation discipline (joint training, S=1)

For each radix node K, the precondition state aggregates over all corpus
instances where K's prefix occurs. We fit AGPT's per-node aggregation
structure by:

- Per fire, sampling **one instance** per K (S=1)
- Running the GRU over that instance's preceding `d_pre` chars
- Using the resulting state as K's precondition for this fire

Across epochs, the encoder sees many different instances of each K; the
stochastic gradient produces the average representation AGPT would want.
This trades per-fire determinism for joint-training efficiency.

True S=m_K aggregation (run encoder on every instance, mean states) is
exponentially more expensive at high-mass K's. It's a second-pass
optimization if S=1 plateaus.

## Injection point

Residual add at layer 0 LN1 input:

```
query_residual_stream[K] += W_pre @ precondition_state[K]
```

Where:
- `W_pre` is a learned [D, D] projection
- Initialized to **zero**, so `precondition_d_pre=0` → bit-exact baseline
- Gradient flows: `dL/d(W_pre) += precondition_state^T · dL/d(LN1_input)`
  and `dL/d(precondition_state) = W_pre^T · dL/d(LN1_input)` propagates
  back through the GRU.

## Constraints (enforced at startup)

- `experimental.precondition.d_pre > 0` requires `train.partition_depth == 0`
  (per-K instance sampling is reproducible only within a single fire per
  epoch — same as slot-selection's pd=0 constraint, but distinct in
  mechanism).

## Path beyond v1

The injection mechanism is designed source-agnostic. Once the GRU-over-raw-
chars encoder is validated, the same residual injection can take state from
a different source — specifically, the previous segment's end-cap hidden
state in a tree-to-tree state-passing setup. That's the natural transition
to the full tree-to-tree RNN architecture without rebuilding the injection
plumbing.

## Commits (planned, in order)

1. **YAML plumbing** (current). Field is parsed; hard-error if d_pre > 0
   until subsequent commits land. Baseline parity preserved when d_pre = 0.
2. **Instance index** — host-side data structure mapping K → list of corpus
   positions. Built once at trie load.
3. **Precondition gather kernel** — per-fire S=1 sampler producing
   `[N_chunk, d_pre]` raw-char tensor per chunk.
4. **GRU encoder forward + backward** — kernel taking the raw-char tensor
   and producing `[N_chunk, D]` state. Initialized so zero state if d_pre=0.
5. **`W_pre` parameter + residual injection** — added to weight buffer,
   zero-initialized, residual-added at layer 0 LN1 input. Standard backward
   for the projection.
6. **B=0 parity check** — runs at `precondition_d_pre=0` must match
   pre-precondition main bit-exactly (within v1's known nondet floor).
7. **Smoke + paired runs** — first PPL signal vs baseline.
