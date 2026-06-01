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

## Sidecar format

The per-K instance d_pre tokens are pre-extracted offline into a sidecar
binary (`bin/agpt_build_precondition_sidecar`):

```
magic        u32 = 'PREC' (0x43455250)
version      u32 = 1
n_radix      u32
d_pre        u32
n_instances  u64
skipped      u64                    diagnostic (instances with start_pos < d_pre)
offsets      u32[n_radix + 1]       offsets[k]..offsets[k+1] = K's instance slice
inst_tokens  i32[n_instances * d_pre]   per-instance d_pre tokens, forward order
```

Pivot from earlier dual-tree plan: storing the d_pre tokens directly
(not as suffix-tree-node references) sidesteps mid-edge handling, doesn't
require loading a suffix trie in the trainer, and is structurally simpler.
The "dual-tree elegance" framing is preserved as a future optimization
if storage becomes a concern.

Shakespeare d=16/d_pre=16 measured: 8.1M instances, 502 MB sidecar,
100% node coverage (1,527,313 / 1,527,328 radix nodes have ≥1 instance),
max 161,751 instances per node (high-mass K's), build time ~5 sec.

The instance count is higher than corpus_size because the walker emits
at every trie level visited per corpus position (each radix node at every
depth has its own real precondition contexts) — this is correct: a
depth-3 node like "the" gets ALL the preceding-16-char contexts wherever
"the" appears, not just one.

## Commits (in order)

1. ✓ **YAML plumbing**. Field is parsed; hard-error if d_pre > 0 until
   subsequent commits land. Baseline parity preserved when d_pre = 0.
2. ✓ **Sidecar tool** (`bin/agpt_build_precondition_sidecar`). Walks
   forward corpus via `CorpusTrieWalker`, emits the format above.
3. **v1 trainer: load sidecar** — `PreconditionSidecar` struct + load()
   in C++ side; allocate at trie load; no usage yet. Verify load is OK
   and B=0 baseline parity holds.
4. **Per-fire instance sampling** — host-side seed-based sampler; per
   chunk produces `d_precondition_input_tokens[N_chunk, d_pre]` on
   device.
5. **GRU encoder forward + backward** — kernel taking `[N_chunk, d_pre]`
   int tokens, producing `[N_chunk, D]` state via GRU over d_pre time
   steps. New parameters in the weight buffer.
6. **`W_pre` parameter + residual injection** — added to weight buffer,
   zero-initialized for parity, residual-added at layer 0 LN1 input.
7. **B=0 parity check** — runs at `precondition_d_pre=0` must match
   baseline bit-exactly (within v1's known nondet floor).
8. **Smoke + paired runs** — first PPL signal vs baseline.
