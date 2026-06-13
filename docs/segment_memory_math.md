# Segment Memory Formulation

This note describes the current segment-memory experiments in
`scripts/run_segment_memory_model.py`. The goal is to separate:

- a local high-resolution recurrent model, currently a GRU;
- a long-range low-resolution memory reader, currently attention over previous
  segment records;
- a final prediction rule.

The strongest current result is the simple `late` formulation:

```text
RNN local state + long-memory attention -> concat head
```

On a 200k-character Tiny Shakespeare slice:

```text
200k GRU late, 10 epochs:
9.245 -> 8.299 -> 7.845 -> 7.590 -> 7.408 -> 7.305 -> 7.254 -> 7.256 -> 7.270 -> 7.315
```

Best validation PPL was about `7.25`.

## Data And Segments

Given a token sequence:

```text
x_0, x_1, ..., x_{N-1}
```

we build a prefix-count trie up to max depth `D`. The current experiments use
`D = 16`.

The corpus is then partitioned into variable-length segments:

```text
sigma_j = x_{a_j : b_j}
```

where a segment ends when either:

- max depth is reached; or
- the current trie prefix becomes unique enough according to
  `unique_threshold`.

For Tiny Shakespeare at 200k chars:

```text
train_segments = 24706
mean_segment_len = 8.10
max_segment_len = 16
```

The model processes segments in corpus order.

## Local RNN

The local model is recurrent. For token `x_t` in a segment:

```text
e_t = Emb(x_t)
h_t = RNN(e_t, h_{t-1})
```

The best current core is GRU:

```text
h_t = GRU(e_t, h_{t-1})
```

We also tested a plain tanh RNN:

```text
h_t = tanh(W_ih e_t + b_ih + W_hh h_{t-1} + b_hh)
```

At 200k, tanh lagged GRU:

```text
200k GRU late, 3 epochs:   9.245 -> 8.299 -> 7.845
200k tanh late, 3 epochs: 10.177 -> 9.094 -> 8.808
```

## Segment Carry

The RNN hidden state is carried between segments:

```text
h_init(sigma_{j+1}) = h_{b_j - 1}
```

That is, only the RNN terminal state is fed back into the RNN.

The attention context is not fed back into the RNN in the best current setup.

## Memory Records

At the end of each segment, the terminal RNN state is converted into a memory
record:

```text
r_j = Writer(h_{b_j - 1})
```

The current writer is:

```text
Writer(h) = LayerNorm(Linear(GELU(Linear(LayerNorm(h)))))
```

The memory bank before segment `j` is:

```text
M_j = [r_{j-K}, ..., r_{j-1}]
```

where `K = max_memory`, usually `32`.

Each memory record also has a corpus position used for RoPE in the older
single-head attention path.

## Long-Memory Attention

For each token state `h_t`, the long-memory attention reads previous segment
records:

```text
q_t = W_q h_t
k_i = W_k r_i
v_i = W_v r_i
```

With RoPE over character positions:

```text
q_t <- RoPE(q_t, position=t)
k_i <- RoPE(k_i, position=end_position(r_i))
```

Then:

```text
a_{t,i} = softmax_i(q_t k_i^T / sqrt(d))
c_t = sum_i a_{t,i} v_i
```

This is the long-memory attention context.

Important limitation: this is still a bare attention read. It is not a full
transformer block:

- no multi-head attention in the `late` path;
- no attention output projection block;
- no residual attention stack;
- no MLP after attention.

## Best Current Prediction Rule: `late`

The best current formulation is:

```text
logits_t = W_out [h_t ; c_t] + b_out
loss_t = CE(logits_t, x_{t+1})
```

Full objective:

```text
L = sum_t CE(W_out [h_t ; c_t] + b_out, x_{t+1})
```

This gives the raw RNN state a direct path into the final logits.

That is both a strength and a weakness:

- strength: optimization is easy and performance is currently best;
- weakness: the RNN can dominate the decision, making long-memory attention a
  side feature rather than the owner of the final prediction.

Diagnostics support this:

```text
200k GRU late no-memory diagnostic:
11.637 -> 8.216

200k GRU late context-only diagnostic:
about 27-30
```

The memory context is useful in combination with the RNN state, but poor as an
independent predictor.

## Interface Normalization Tests

We tested several normalizations between RNN and attention:

```text
none       current baseline
attention  LayerNorm before Q and K/V projections only
context    LayerNorm Q/K/V plus LayerNorm context before final head
layer      LayerNorm Q/K/V plus LayerNorm both h_t and context before head
```

Results:

```text
200k GRU late baseline:
9.245 -> 8.299 -> 7.845

200k GRU late + layer norm:
9.431 -> 8.393 -> 8.003

200k GRU late + context norm:
9.617 -> 8.393 -> 7.942
```

Normalization improved context-only PPL substantially, but did not improve the
combined model. This suggests normalization made the memory branch cleaner while
removing or distorting information the late head uses from the raw RNN state.

## Attention-Owned Decision Variants

A concern with `late` is that:

```text
logits_t = head([h_t ; c_t])
```

lets the RNN vote directly. We tested variants where the final decision is owned
by the attention/composition side.

### Gated Attention Decision

The first version used:

```text
s_t = Project(h_t)
g_t = sigmoid(Gate([s_t ; c_t]))
z_t = LayerNorm(c_t + g_t * s_t)
z_t = LayerNorm(z_t + MLP(z_t))
logits_t = Head(z_t)
```

This was slower and worse:

```text
200k attention-decision:
10.012 -> 9.138 -> 8.431
```

The gate was initialized with bias `-2`, so the RNN signal initially entered at:

```text
sigmoid(-2) ~= 0.12
```

This may have starved the decision state.

### LMA Token Formulation

The cleaner formulation treats the current RNN state as just another token in
the long-memory attention module:

```text
S_0 = h_t
S_i = r_i for previous segment records
```

Then:

```text
Z = TransformerEncoder([S_0, S_1, ..., S_K])
logits_t = Head(Z_0)
```

This gives the long-memory attention module final say, while still letting the
RNN provide the current high-resolution local state.

In code this is `--mixing lma-token`.

Results:

```text
10k LMA-token:
23.20 -> 16.69 -> 15.59

50k LMA-token:
12.49 -> 10.38 -> 9.99

50k simple late:
12.46 -> 10.44 -> 9.46
```

The LMA-token formulation is conceptually clean, but currently slower and worse
than the simple `late` head.

## Current Open Problem

The main unresolved representational issue:

```text
RNN state h_t:
  optimized strongly for local recurrence and next-token prediction.

Attention context c_t:
  weighted sum of memory value vectors.
  optimized indirectly through final loss.
```

In `late`, the RNN has an easy direct prediction path:

```text
loss -> head -> h_t -> RNN
```

So it may learn to be a standalone local predictor, while long-memory attention
only becomes an auxiliary feature.

In `lma-token`, the final decision belongs to the attention/composition module,
but the current implementation does not yet beat the simpler concatenation
head.

## Controlled Harness Check

The old comparison between the segment harness and standalone GRU was not
matched. A controlled check used the same 200k training slice, first 20k
validation chars, `embedding_size=64`, `hidden_size=64`, `lr=0.001`, and about
3,860 optimizer updates.

Random-window sequence GRU:

```text
batch_size=32, block_size=16, steps=3860
val PPL: 7.83
```

Segment-stream GRU:

```text
mixing=none, carry_hidden=true, max_depth=16, segments_per_update=64
val PPL by epoch:
10.587 -> 9.107 -> 8.577 -> 8.349 -> 8.192
 8.072 -> 7.966 -> 7.879 -> 7.808 -> 7.757
```

So the segment harness is not carrying a multi-PPL local-model handicap under
matched settings. It is slower, and its ordered stream differs from random
window batches, but the final local GRU floor is comparable.

## Gated Cross-Attention Test

The current principled long-memory block is a gated cross-attention block:

```text
a_t = MHA(LN(h_t), LN(M), LN(M))
u_t = h_t + tanh(alpha) a_t
z_t = u_t + tanh(beta) MLP(LN(u_t))
logits_t = W_out z_t
```

Both gates initialize to zero. In code this is `--mixing gated-xattn`, and it
uses the same `gru_head` as `--mixing none`, so its initial function is exactly
the carried-GRU path.

On the same 200k/d64 setup:

```text
gated-xattn: 10.390 -> 9.599 -> 9.575
no-memory:   10.587 -> 9.107 -> 8.577
```

This suggests the block shape is no longer the immediate failure point. The
remaining problem is likely the memory record objective: terminal/written RNN
states are not yet reliable retrieval records for improving next-character
prediction.

## Terminal Record Auxiliary

The first direct record objective supervises the exact stored memory record:

```text
m_j = write_memory(cap_state_j)
y_j = x_{end_j}
L_record = CE(record_head(m_j), y_j)
L_total = L_next_token + alpha L_record
```

where `x_{end_j}` is the token immediately after the segment. This is distinct
from the older per-token record auxiliary, which supervised every hidden state
before writing memory.

Results on 200k/d64 gated-xattn:

```text
no terminal aux: 10.390 -> 9.599 -> 9.575
alpha=0.10:     10.379 -> 9.575 -> 9.458
alpha=0.25:     10.453 -> 9.641 -> 9.483
no-memory GRU:  10.587 -> 9.107 -> 8.577
```

The auxiliary is directionally useful at low weight, but too weak to make the
memory branch competitive with the local GRU. Higher weight hurts, which
suggests immediate-next-token prediction is not the right complete semantics
for a segment record.

## Full-Corpus Route Scale

The 200k slice may not contain enough segment routes for the long-memory branch
to learn useful retrieval behavior. On the full training split:

```text
train_chars = 1,003,854
segments    =   108,041
mean_len    =      9.29
eval_chars  =    20,000
```

One epoch of local GRU only:

```text
mixing=none, carry_hidden=true
val_ppl = 6.988
```

One epoch of gated cross-attention:

```text
mixing=gated-xattn
memory_record=written
terminal_record_aux_weight=0.1
max_memory=16
val_ppl = 6.503
mean_gate = 0.148
```

This is the first matched setting where long-memory attention improves over the
local GRU floor. It suggests that route count/data scale was a major factor in
the smaller-slice failures.

The paired full-corpus curve is stronger than the one-epoch result alone:

```text
no-memory GRU:
6.988 -> 6.303 -> 5.942

gated-xattn + terminal aux 0.1, max_memory=16:
6.503 -> 5.867
```

The memory model's second epoch is already below the local model's third epoch.
This still does not establish the final asymptote, but it means the memory
branch is not just a one-epoch acceleration artifact.

The checkpointed 5-epoch full-corpus curves are:

```text
no-memory GRU:
6.988 -> 6.303 -> 5.942 -> 5.744 -> 5.627

gated-xattn + terminal aux 0.1, max_memory=16:
6.503 -> 5.867 -> 5.546 -> 5.344 -> 5.199
```

The memory branch remains ahead at matched epoch 5 and has not yet flattened.
This supports the interpretation that full-corpus route-memory attention is
adding useful capacity, though the result is still above the desired `~4` PPL
range.

## Count-Gate Prior

The old count-gate baseline was reproduced. It uses a recursive probability
mixture, not a product-of-experts logit sum:

```text
q_d = w_d p_mle_d + (1 - w_d) q_backoff
```

With prefix statistics, entropy-delta, and suffix-side statistics, the count-only
model reaches:

```text
depth 8 carved heldout fixed-skip PPL: 3.860
```

This indicates that the prefix/suffix count tables already contain enough
information to reach the target PPL range when composed correctly.

The count prior was integrated as a frozen prior under the neural model:

```text
logits_t = log p_count_gate(x_{<=t}) + residual_logits_t
```

Prediction heads are zero-initialized in this mode, so training starts exactly
at the count prior. On the current full 90/10 segment split:

```text
count prior fit-tail PPL: 4.648

count prior + carried GRU residual:
4.995 -> 4.876 -> 4.781 -> 4.686 -> 4.594
```

The count prior can also be combined with the gated cross-attention residual:

```text
count prior + gated-xattn residual:
4.869 -> 4.675 -> 4.538 -> 4.407 -> 4.286
```

This is the best current segment-harness result and is still improving at epoch
5. It suggests that the main missing piece was not only long-memory attention,
but correct heldout-safe count/backoff composition with suffix-side features.
Once that prior is in place, the gated memory branch provides an additional
residual improvement.

Operationally, count-prior precomputation is now cacheable. The cold full
depth-8 precompute took about `194s`; a matching cached resume loaded in about
`0.1s`. The remaining bottleneck is the gated recurrent attention backward pass.

Continuation to 10 total epochs produced:

```text
4.869 -> 4.675 -> 4.538 -> 4.407 -> 4.286
      -> 4.205 -> 4.153 -> 4.121 -> 4.116 -> 4.136
```

Best point: epoch 9 at `4.116` heldout PPL. The slight epoch-10 regression is a
good sign for protocol sanity: the curve crossed the KN-parity region and then
flattened, rather than falling indefinitely.

## Carved Split Audit

The old carved split must be treated as a separate protocol. Evaluating a model
trained on the current contiguous 90/10 split against the old carved heldout is
invalid because most old heldout chunks lie inside the current train slice.

The fair carved protocol uses:

```text
train_input = old train_corpus.txt
eval_input  = old heldout_corpus.txt
vocab_input = full data/input.txt
```

Under that protocol, the frozen recursive count-gate prior alone reaches:

```text
count prior only: 3.859 PPL
```

That reproduces the old count-gate baseline. Adding the current gated-xattn
residual for one epoch worsened heldout:

```text
epoch 0 prior:     3.859
epoch 1 residual:  3.958
```

So the residual-memory branch is useful on the current tail-10% split, but the
current objective is not automatically beneficial when the count prior is already
near the carved-split floor. This is the main caveat for the present math:

```text
logits_t = log p_count_gate(x_{<=t}) + residual_logits_t
```

is structurally clean, but the residual needs a trust/regularization rule so it
does not damage a strong prior.

## Questions For Review

Useful second-opinion questions:

1. Is `head([h_t ; c_t])` fundamentally the wrong objective if we want LMA to own
   final prediction?
2. Should `h_t` enter LMA as token `S_0`, or only as a query into memory records?
3. How should the memory records `r_i` be trained so they are good retrieval
   objects rather than merely transformed RNN terminal states?
4. Should the long-memory path have an auxiliary next-token or contrastive loss?
5. Is there a better way to combine local RNN state and long-memory attention
   without giving raw RNN logits a bypass?
6. Should the LMA block use causal/self attention over `[S_0, S_1..S_K]`, or
   cross-attention with `S_0` as the only query?
7. Is the memory bank too small or too low-resolution for Tiny Shakespeare?
8. Is RoPE over absolute character positions the right positional signal for
   segment memory, or should relative distance be used instead?
