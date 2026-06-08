# Tanh Recurrent AGPT

Status: closed branch `tanh-recurrence`.

This line tests AGPT as a framework rather than as attention specifically:

```text
h_child = tanh(W_h h_parent + W_x emb[token] + b)
logits = W_o h_child + c_o
```

The trainer is `bin/agpt_train_recur`, implemented in
`src/tools/agpt_train_recur.cr`. Held-out evaluation uses
`bin/agpt_recur_perplexity`.

## Close-Out

This R&D line answered its main question: `f_theta` is free in the AGPT
framework. Attention is not structurally required. Replacing attention with a
simple recurrent tanh transition trains cleanly over the same prefix/mass
framework:

```text
h_child = tanh(W_h h_parent + W_x emb[token] + b)
```

The best clean tanh result in this branch is `6.3099` heldout PPL at depth 8,
`d_model=64`, `partition-depth=1`, epoch 500. This is not competitive with the
best attention AGPT runs, but it is strong enough to establish that AGPT is a
framework over trainable prefix transitions, not an attention-only
implementation.

The next research line is stride/dilated trees: use the same freedom in
`f_theta` to test whether longer-range structure can be exposed through
skip-step recurrence or auxiliary stride states.

## Clean Methodology

All reportable recurrent results in this file use a trie and any position tables
built from the train split only:

```text
train:   /home/trans/Projects/agpt/data/.splits/4fa9aec1db6b3aea/train_corpus.txt
heldout: /home/trans/Projects/agpt/data/.splits/4fa9aec1db6b3aea/heldout_corpus.txt
vocab:   data/input.txt
```

Train-only artifacts:

```text
prefix trie:   /tmp/agpt_tanh_clean_d8_radix
suffix trie:   /tmp/agpt_tanh_clean_d8_suffix_radix
W=16 position: /tmp/agpt_tanh_clean_d8_w16_position_data
```

The clean d8 prefix trie has `814,759` records and `7,038,366` loss events.

Deferred rerun queue:

- Phase-embedded/phase-weighted W=16 pd=1 lr=0.001 to 512.
- Phase-weighted W=32 pd=1 lr=0.001 to 512.
- Plain tanh pd=0 full-batch lr=0.001 to 500.
- RMSNorm clean rerun, if comparison against Claude's variants needs it.

## Clean Results

All runs below use depth 8, `d_model=64`, Adam, constant `lr=0.001`, seed 1,
`partition-depth=1`, and held-out rolling eval with `seq-len=8`, unless noted.

| variant | epoch | train PPL | clean heldout PPL | bpc | status |
|---------|------:|----------:|------------------:|----:|--------|
| plain tanh | 100 | 8.4876 | 7.2456 | 2.8571 | complete |
| plain tanh | 500 | 7.7584 | 6.3099 | 2.6576 | complete |
| phase-weighted W=16 | 128 | 8.4604 | 7.2145 | 2.8509 | complete |
| phase-weighted W=16 | 256 | 8.0396 | 6.7060 | 2.7454 | complete |
| phase-weighted W=16 | 512 | 7.8321 | 6.4035 | 2.6789 | complete |
| phase-embedded/weighted W=16 | 100 | 8.5876 | 7.3624 | 2.8802 | pending longer rerun |
| singleton-backoff stopgrad | 100 | 8.8663 | 7.8265 | 2.9684 | complete negative probe |

Current read:

- Plain tanh pd=1 is the clean baseline to beat: `6.3099` heldout PPL at epoch
  500.
- Phase-weighted W=16 is viable and improves steadily, but does not beat clean
  plain tanh by epoch 512.
- Phase-embedded/weighted W=16 is slightly ahead of plain early at epoch 20, but
  behind by epoch 100; it still needs a clean 512-epoch run before closing that
  variant.
- Stop-gradient singleton backoff is worse than plain tanh at matched epoch
  100, so that exact implementation is not promising.

### Plain Tanh pd=1

Command:

```text
OPENBLAS_NUM_THREADS=1 nice -n 10 bin/agpt_train_recur \
  --trie /tmp/agpt_tanh_clean_d8_radix \
  --d-model 64 \
  --epochs 480 \
  --lr 0.001 \
  --seed 1 \
  --load /tmp/agpt_tanh_clean_d8_d64_pd1_lr001_20.recur \
  --save /tmp/agpt_tanh_clean_d8_d64_pd1_lr001_500.recur \
  --partition-depth 1 \
  --checkpoint-every 50
```

Clean held-out sweep:

| epoch | train PPL | clean heldout PPL | bpc |
|------:|----------:|------------------:|----:|
| 10 | 12.6756 | 11.7094 | 3.5496 |
| 20 | 10.9332 | 10.0680 | 3.3317 |
| 50 | 9.3819 | 8.2995 | 3.0530 |
| 100 | 8.4876 | 7.2456 | 2.8571 |
| 150 | 8.1717 | 6.8447 | 2.7750 |
| 200 | 8.0084 | 6.6547 | 2.7344 |
| 250 | 7.9143 | 6.5466 | 2.7107 |
| 300 | 7.8553 | 6.4460 | 2.6884 |
| 350 | 7.8186 | 6.3972 | 2.6774 |
| 400 | 7.7851 | 6.3457 | 2.6658 |
| 450 | 7.7631 | 6.3385 | 2.6641 |
| 500 | 7.7584 | 6.3099 | 2.6576 |

`partition-depth=1` splits by root-child subtree, giving 65 Adam updates per
epoch on the Tiny Shakespeare character vocabulary. This is the main reason the
pd=1 model improves much faster than full-batch pd=0.

### Phase-Weighted W=16

Phase weighting is a training-measure change, not an attention/RoPE-specific
mechanism. This variant keeps the global target distribution at each prefix and
replaces the node's loss mass with its mass at the current corpus-position
phase:

```text
q_global(token | prefix) = global_count(prefix, token) / global_mass(prefix)
loss_weight(prefix, phase) = position_mass(prefix, phase)
```

No phase embedding or architecture change is used.

Command:

```text
OPENBLAS_NUM_THREADS=1 nice -n 10 bin/agpt_train_recur \
  --trie /tmp/agpt_tanh_clean_d8_radix \
  --d-model 64 \
  --epochs 512 \
  --lr 0.001 \
  --seed 1 \
  --save /tmp/agpt_tanh_clean_phaseweighted_w16_d8_d64_pd1_lr001_512.recur \
  --partition-depth 1 \
  --phase-weighted-position-data /tmp/agpt_tanh_clean_d8_w16_position_data \
  --checkpoint-every 32
```

Clean held-out sweep:

| epoch | phase-weighted train PPL | clean heldout PPL | bpc |
|------:|-------------------------:|------------------:|----:|
| 32 | 10.1772 | 9.1968 | 3.2011 |
| 64 | 9.1654 | 8.0369 | 3.0066 |
| 96 | 8.7110 | 7.5077 | 2.9084 |
| 128 | 8.4604 | 7.2145 | 2.8509 |
| 160 | 8.3018 | 7.0206 | 2.8116 |
| 192 | 8.1899 | 6.8789 | 2.7822 |
| 224 | 8.1029 | 6.7782 | 2.7609 |
| 256 | 8.0396 | 6.7060 | 2.7454 |
| 288 | 7.9292 | 6.6415 | 2.7315 |
| 320 | 7.9044 | 6.5901 | 2.7203 |
| 352 | 7.8807 | 6.5550 | 2.7126 |
| 384 | 7.8758 | 6.5112 | 2.7029 |
| 416 | 7.8766 | 6.4815 | 2.6963 |
| 448 | 7.8611 | 6.4515 | 2.6896 |
| 480 | 7.8440 | 6.4233 | 2.6833 |
| 512 | 7.8321 | 6.4035 | 2.6789 |

Conclusion: phase-weighted W=16 is healthy and close, but not a clean win over
plain tanh at this setting.

### Phase-Embedded / Weighted W=16

`--phase-conditioned-position-data DIR` is currently a phase-embedded,
phase-weighted variant. The name is unfortunate: it does not train
phase-conditioned target distributions. It keeps global targets and adds a
learned phase vector to the recurrent transition:

```text
h_child = tanh(W_h h_parent + W_x emb[token] + phase_emb[(start_phase + depth) mod W] + b)
```

For compressed radix edges, the phase advances token by token. For held-out
rolling evaluation, `bin/agpt_recur_perplexity` reads the phase-embedded
checkpoint and uses the actual held-out corpus position `q mod W` for each
consumed context token.

Clean 100-epoch probe:

```text
OPENBLAS_NUM_THREADS=1 nice -n 10 bin/agpt_train_recur \
  --trie /tmp/agpt_tanh_clean_d8_radix \
  --d-model 64 \
  --epochs 80 \
  --lr 0.001 \
  --seed 1 \
  --load /tmp/agpt_tanh_clean_phasecond_w16_d8_d64_pd1_lr001_20.recur \
  --save /tmp/agpt_tanh_clean_phaseembed_w16_d8_d64_pd1_lr001_100.recur \
  --partition-depth 1 \
  --phase-conditioned-position-data /tmp/agpt_tanh_clean_d8_w16_position_data \
  --checkpoint-every 20
```

| epoch | phase-embedded train PPL | clean heldout PPL | bpc |
|------:|-------------------------:|------------------:|----:|
| 10 | 12.3170 | 11.5589 | 3.5309 |
| 20 | 10.8034 | 9.9197 | 3.3103 |
| 40 | 9.7798 | 8.7086 | 3.1224 |
| 60 | 9.1405 | 8.0517 | 3.0093 |
| 80 | 8.8219 | 7.6205 | 2.9299 |
| 100 | 8.5876 | 7.3624 | 2.8802 |

Clean read: explicit phase embeddings help slightly early, but are behind plain
tanh by epoch 100. The matched 512-epoch rerun is still pending.

## Singleton Cap Backoff Analysis

Question: if a radix end cap becomes singleton deterministic (`mass=1`, `H=0`),
can we jump to a suffix/backoff position and continue from a reusable context
instead of training through the unique cap?

Clean d8 trie singleton structure:

```text
total records:              814,759
singleton deterministic:    448,685 records (55.07%)
singleton loss events:      448,685 / 7,038,366 (6.37%)
singleton edge chars:       981,177
mean singleton edge length: 2.187 chars
```

Singleton edge-length histogram:

| edge len | records |
|---------:|--------:|
| 1 | 153,498 |
| 2 | 136,420 |
| 3 | 97,347 |
| 4 | 46,189 |
| 5 | 13,424 |
| 6 | 1,728 |
| 7 | 78 |
| 8 | 1 |

For each character position inside a singleton cap, we recursively dropped the
oldest context token until the suffix context had mass greater than one.
Example:

```text
A B C D [E F G]  -> at E, try B C D E, then C D E, then D E...
                    stop at the first suffix context with mass > 1
```

Results over `981,176` singleton-cap positions:

```text
resolved to mass > 1 suffix:       979,778 (99.86%)
unresolved before empty context:     1,398 (0.14%)
mean token drops needed:              2.427
mean target context length:           4.689
mean target mass:                   207.032
target entropy > 0:                979,778 (100.00%)
```

Drop-count histogram:

| drops | positions |
|------:|----------:|
| 1 | 301,028 |
| 2 | 267,698 |
| 3 | 204,049 |
| 4 | 126,731 |
| 5 | 60,998 |
| 6 | 17,194 |
| 7 | 2,080 |

Target context-length histogram:

| target len | positions |
|-----------:|----------:|
| 1 | 5,209 |
| 2 | 44,327 |
| 3 | 145,280 |
| 4 | 236,716 |
| 5 | 264,107 |
| 6 | 191,697 |
| 7 | 92,442 |

Target mass histogram:

| mass bucket | positions |
|-------------|----------:|
| 2-4 | 247,465 |
| 5-16 | 294,574 |
| 17-64 | 222,361 |
| >64 | 215,378 |

Interpretation: recursive suffix backoff almost always escapes singleton cap
tunnels into a reusable, branching context. This is much stronger than simple
one-token backoff. It makes suffix backoff a plausible structural variant, not
just an optimization, but the first training probe below did not help.

### Hard Backoff Prototype

`bin/agpt_train_recur --singleton-backoff` implements the first hard-routing
prototype:

```text
when a singleton cap token is consumed:
  replace the local state with the precomputed recursive suffix target state
  continue through the remaining edge from that reusable suffix state
```

Backward pass semantics:

```text
gradient after the route is credited to the backoff target endpoint
gradient does not flow through the discarded singleton tunnel
```

The fully routed version requires `--partition-depth 0`. With
`partition-depth=1`, a backoff target can live under a different root-child
partition, so the target state and reverse pass would not necessarily be
present in the same optimizer batch.

Dry-run on the clean d8 trie:

```text
singleton_backoff: on (routes=979778, unresolved=1398)
```

Full-trie one-epoch smoke, `d_model=16`, pd=0:

```text
epoch 1  nll 4.175799  ppl 65.091810  events 7038366  updates 1
```

`--singleton-backoff-stopgrad` adds the pd=1-compatible probe variant:

```text
route hidden states through the recursive suffix target
stop gradient at the route boundary
preserve partition-depth=1 update cadence
```

A 100-epoch clean probe:

```text
OPENBLAS_NUM_THREADS=1 nice -n 10 bin/agpt_train_recur \
  --trie /tmp/agpt_tanh_clean_d8_radix \
  --d-model 64 \
  --epochs 100 \
  --lr 0.001 \
  --seed 1 \
  --save /tmp/agpt_tanh_clean_backoff_stopgrad_d8_d64_pd1_lr001_100.recur \
  --partition-depth 1 \
  --singleton-backoff-stopgrad \
  --checkpoint-every 20
```

| epoch | stopgrad train PPL | clean heldout PPL | bpc |
|------:|-------------------:|------------------:|----:|
| 20 | 11.0025 | 10.2141 | 3.3525 |
| 40 | 9.8831 | 9.0039 | 3.1706 |
| 60 | 9.3691 | 8.3767 | 3.0664 |
| 80 | 9.0642 | 8.0535 | 3.0096 |
| 100 | 8.8663 | 7.8265 | 2.9684 |

This is a negative result versus clean plain tanh at matched epoch 100:

```text
clean plain tanh epoch 100:        7.2456
backoff stopgrad epoch 100:        7.8265
phase-embedded W=16 epoch 100:     7.3624
```

Stop-gradient routing preserves the pd=1 update cadence, but the missing
gradient through the reusable suffix targets appears to matter. The fully
correct hard-backoff route still requires either pd=0 or a new partitioning
scheme that carries cross-partition backoff dependencies.

## Open Questions

- Compare clean tanh recurrence against Claude's clean linear, linear+RMSNorm,
  and GRU variants with identical split, depth, `d_model`, optimizer, and
  held-out evaluator.
- Complete the clean phase-embedded W=16 run to 512 before closing phase inputs.
- Decide whether W=32 is still worth a clean rerun now that W=16 did not beat
  plain tanh.
- Try depth 16 only after the depth-8 clean result table is stable.
- Add K-FAC or a related curvature-aware optimizer. This model is small enough
  that a structured second-order method may be practical.
- Explore stride/dilated trees as a grounded way to expose longer-range
  structure to recurrent `f_theta`.
