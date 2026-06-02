# Precondition Strand

## Status

**CLOSED 2026-06-02 — null result + structural misalignment.**

The mechanism was built and tested end-to-end. Results below.
The deeper reason it doesn't fit AGPT is structural and explained
in the "Why this line is closed" section at the bottom — the design
violates AGPT's SGD-equivalence property, which is what AGPT IS.

## Verdict — paired runs at Codex's best recipe (d_model=128, L=6, h=8, d_ff=512, pd=1, 128 ep)

| Metric                 | Baseline (d_pre=0) | Precondition (d_pre=16, GRU) | Δ |
| ---                    | ---                | ---                          | --- |
| **byte_PPL (held-out)**| **6.366**          | **6.474**                    | **+0.108 (+1.70%) — slight regression** |
| fixed_token_PPL        | 6.049              | 6.139                        | +0.090 (+1.49%) |
| BPB                    | 2.6704             | 2.6946                       | +0.0242 |
| Train loss (ep128, nats)| 2.6224            | 2.6236                       | +0.0012 (tied within noise) |
| Wall                   | 3950 sec           | 28341 sec                    | **7.2× slower** |

Source runs:
- `rnd/precondition-d128L6/20260602T050004-baseline-d128l6-seed1/`
- `rnd/precondition-d128L6/20260602T060555-precondition-d128l6-seed1/`

Reading: train losses identical to within 0.0012 nats. The GRU adds
capacity but doesn't fit training data better — so the encoder isn't
extracting useful information that the trie doesn't already have.
The slight held-out regression is consistent with the encoder finding
spurious training-distribution structure that doesn't generalize.

## Earlier d=64/L=2 paired matrix (Step 1, informative but PPL-saturated)

12 runs, 3 seeds × 4 conditions, on the unconverged d=64/L=2 regime
(~22.69 byte_PPL — well above the 16-token KN floor of ~4.x; this is
the informatively-empty regime where models haven't learned much yet):

- Baseline (b=0):                       22.687 ± 0.002
- Precondition mean-pool (post-LN1):    22.941 ± 0.002 (+0.254 regression)
- Precondition GRU (post-LN1):          22.687 ± 0.001 (null — tied to baseline)
- Precondition GRU v1 (post-LN1, fresh): 22.686 ± 0.000 (null — tied to baseline)

Same conclusion: mean-pool overfits the training distribution at the
LN1 input; GRU produces no detectable held-out signal. Replicates at
the converged d=128/L=6 regime (above) where the GRU instead nulls
on training loss but slightly regresses on held-out.

Source runs: `rnd/precondition-step1/`.

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

## Commits (in order — all landed)

1. ✓ **YAML plumbing**. Field is parsed; hard-error if d_pre > 0 until
   subsequent commits land. Baseline parity preserved when d_pre = 0.
2. ✓ **Sidecar tool** (`bin/agpt_build_precondition_sidecar`).
3. ✓ **v1 trainer: load sidecar** — `PreconditionSidecar` struct + load().
4. ✓ **Per-fire instance sampling** — host-side seed-based sampler.
5. ✓ **GRU encoder forward + backward** — kernel taking `[N_chunk, d_pre]`
   int tokens, producing `[N_chunk, D]` state via GRU over d_pre time
   steps.
6. ✓ **`W_pre` parameter + residual injection** — added to weight buffer,
   zero-initialized for parity, residual-added at layer 0 LN1 input.
   Variant 1 (post-LN1 injection) also tested — no difference at scale.
7. ✓ **B=0 parity check** — passed.
8. ✓ **Smoke + paired runs** — null result at d=64/L=2 (3 seeds × 4 conditions)
   and at d=128/L=6 (Codex's best recipe). See verdict above.

## Why this line is closed — the structural argument

The precondition strand was an attempt to extend AGPT's effective
context by bolting a foreign encoder (GRU) onto a foreign input
(d_pre preceding chars) and injecting the result into AGPT's
residual stream. Even where the plumbing works correctly, the
design violates the property that makes AGPT what it is.

**AGPT is a formal SGD-equivalence.** Each radix node K's gradient
is provably equal to the sum of per-instance gradient contributions
across all corpus positions where K's prefix occurs. The aggregation
is in a specific algebraic family — sums of gradients of a shared
loss over partitions of the corpus.

The GRU's output state is not in that family. The W_pre injection
adds a term to K's residual that comes from outside AGPT's aggregation
graph, computed by a sequence model with its own backprop path through
its own params and (worse) sharing the token embedding with the trie.
The combined object is no longer "AGPT", it's "AGPT + a GRU residual
adapter". Whatever helps or doesn't help in such a system tells us
about the combined object, not about AGPT.

Empirically the combined object didn't help anyway. But that's not
the deep reason to close: the deep reason is that even a working
version wouldn't extend AGPT — it would be a different architecture
that happens to contain AGPT.

**Efficiency point**: even ignoring the algebraic argument, GRUs cap
around a few hundred steps before vanishing-gradient hits. The 7.2×
wall cost is structural (heterogeneous architectures don't share
parameters or backprop paths). The right way to extend AGPT's
effective context is through extensions to AGPT's aggregation
structure itself — partitioning K's instances by structured
properties of their context (cf. the suffix-fold / dual-tree
threads in earlier notes) — so the aggregation graph grows but
stays algebraically clean and SGD-equivalent.

The plumbing (sidecar format, per-K instance sampling, fire-end
parameter accumulation) remains in the branch and is reusable by
any future variant that needs per-K context that respects AGPT's
aggregation discipline. The W_pre / GRU encoder paths are
abandoned. The branch will be archived rather than merged.
