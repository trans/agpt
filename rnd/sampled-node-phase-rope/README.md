# sampled-node-phase-rope

Status: active

Trainer note: post-fix

## Question

Can sampled corpus-position / phase information improve the depth-16 CUDAX
AGPT recipe by presenting trie paths at RoPE positions that match where their
prefixes occur in the corpus?

The motivating concern is that the ordinary prefix trie pools all occurrences
of the same prefix into one node:

```text
mass(P) = sum_q mass(P, q)
target(P) = global next-token counts after P
```

If continuation statistics depend on local corpus phase, that pooling discards
information that RoPE could use. This experiment tests several ways to expose
phase information without physically materializing a full phase-conditioned
trie.

See `phase-unrolled-tree.md` for the design sketch and phase convention.

## Code Context

Branch: `sampled-node-phase-rope`

The run metadata records base commit `0e118a213e000e7d6fedc002470ef75549f67c84`
with a dirty worktree. The relevant implementation changes live in the CUDAX v2
trainer path and position-data tooling:

- `src/cudax/*_v2.cuh`
- `src/cudax/agpt_train_v2.cu`
- `src/tools/build_position_table.cr`
- phase inspection / validation tools in `src/tools/`

The experiment artifacts are in `rnd/sampled-node-phase-rope/`.

## Protocol

Shared settings unless noted:

- Corpus: `data/input.txt`, sampled heldout carve `5% x 10` chunks, seed 42
- Trainer: `bin/agpt_train_v2`
- Model: `data/seeds/shake-d64L2-h4-dff256-s128-seed42.model`
- Architecture: `d_model=64`, `n_layers=2`, `n_heads=4`, `d_ff=256`
- Effective AGPT depth/window: 16
- Optimizer: Adam `lr=0.0015`, `beta1=0.9`, `beta2=0.999`
- Training: `pd=1`, `anc_grad=true`, `chunk_queries=50000`
- Eval: multi-chunk heldout, CPU, `batch_size=1`
- Position data: `/tmp/agpt_snp_25ep_position_data`

The main tested `experimental.rope_position_mode` values were:

- `sampled-unit-phase`: sample RoPE phase from prefix occurrence statistics.
- `phase-sweep`: control that changes presentation phase without changing
  target weighting.
- `phase-weighted`: weight prefix losses by phase mass while keeping global
  next-token targets.
- `phase-conditioned`: use phase-conditioned target distributions.

Later runs moved from wrapped / non-wrapped presentations to the current
`unwrapped` interpretation: RoPE positions are presented without wrapping the
local attention path, while phase mass is still read modulo the phase window.

## Results

Rows report final heldout metrics for each run. Most are from `result.json`;
the RunPod d128/L6 phase-conditioned rows are from locally generated
`eval_raw_epoch_*.json` files, because the pod image lacked Python for the
post-training eval step. Lower is better.

| Run ID | Mode / delta | fixed_token_ppl | rolling_byte_ppl | bits/byte |
|--------|--------------|----------------:|-----------------:|----------:|
| `20260602T042851-d64l2-depth16-pd1-adam-lr0015-25ep-snp-wrap` | sampled-unit-phase, 25ep, wrapped | 10.0677 | 8.9265 | 3.1581 |
| `20260602T043825-d64l2-depth16-pd1-adam-lr0015-100ep-snp-wrap` | sampled-unit-phase, 100ep, wrapped | 8.2229 | 7.0580 | 2.8193 |
| `20260602T144633-d64l2-depth16-pd1-adam-lr0015-25ep-snp-wrap` | sampled-unit-phase rerun, 25ep, wrapped | 37.4570 | 14.9928 | 3.9062 |
| `20260602T155141-d64l2-depth16-pd1-adam-lr0015-4ep-phaseflow-wrap` | sampled phase-flow, 4ep, wrapped | 15.8868 | 14.1828 | 3.8261 |
| `20260602T203605-d64l2-depth16-pd1-adam-lr0015-4ep-phase-sweep-wrap` | phase-sweep, 4ep, wrapped | 16.6378 | 14.4838 | 3.8564 |
| `20260602T203831-d64l2-depth16-pd1-adam-lr0015-4ep-phase-weighted-wrap` | phase-weighted, 4ep, wrapped | 15.8214 | 14.0734 | 3.8149 |
| `20260602T204315-d64l2-depth16-pd1-adam-lr0015-25ep-phase-sweep-wrap` | phase-sweep, 25ep, wrapped | 31.5454 | 14.1796 | 3.8257 |
| `20260602T204942-d64l2-depth16-pd1-adam-lr0015-25ep-phase-weighted-wrap` | phase-weighted, 25ep, wrapped | 17.3149 | 10.6788 | 3.4167 |
| `20260602T205743-d64l2-depth16-pd1-adam-lr0015-100ep-phase-weighted-wrap` | phase-weighted, 100ep, wrapped | 37.2467 | 12.5751 | 3.6525 |
| `20260602T220556-d64l2-depth16-pd1-adam-lr0015-100ep-phase-weighted-nonwrap` | phase-weighted, 100ep, nonwrap | 34.5860 | 12.4310 | 3.6359 |
| `20260602T223837-d64l2-depth16-pd1-adam-lr0015-4ep-phase-sweep-nonwrap-decoup` | phase-sweep, 4ep, nonwrap decoupled | 12.7436 | 12.7244 | 3.6695 |
| `20260602T224525-d64l2-depth16-pd1-adam-lr0015-4ep-phase-fixed1-decoupled` | fixed offset control, 4ep | 12.7432 | 12.7277 | 3.6699 |
| `20260602T225143-d64l2-depth16-pd1-adam-lr0015-32ep-phase-sweep-nonwrap-decou` | phase-sweep, 32ep, nonwrap decoupled | 6.1646 | 6.5766 | 2.7173 |
| `20260602T230402-d64l2-depth16-pd1-adam-lr0015-32ep-phase-weighted-nonwrap-de` | phase-weighted, 32ep, nonwrap decoupled | 6.9947 | 7.3706 | 2.8818 |
| `20260602T231436-d64l2-depth16-pd1-adam-lr0015-64ep-phase-weighted-nonwrap-de` | phase-weighted, 64ep, nonwrap decoupled | 6.0060 | 6.4669 | 2.6931 |
| `20260603T044622-d64l2-depth16-pd1-adam-lr0015-16ep-phase-weighted-unwrapped` | phase-weighted, 16ep, unwrapped | 8.4195 | 8.6135 | 3.1066 |
| `20260603T045235-d64l2-depth16-pd1-adam-lr0015-128ep-phase-weighted-unwrapped` | phase-weighted, 128ep, unwrapped | 5.2452 | 5.7713 | 2.5289 |
| `20260603T051439-d64l2-depth16-pd1-adam-lr0015-256ep-phase-weighted-unwrapped` | phase-weighted, 256ep, unwrapped | 4.8021 | 5.3398 | 2.4168 |
| `20260603T055517-d64l2-depth16-pd1-adam-lr0015-32ep-phase-conditioned-unwrapp` | phase-conditioned, 32ep, unwrapped | 7.5194 | 7.8220 | 2.9675 |
| `20260603T084147-d64l2-depth16-pd1-adam-lr0015-4ep-phase-conditioned-diag` | phase-conditioned diagnostic, 4ep | 13.0819 | 12.9235 | 3.6919 |
| `20260603T102636-d64l2-depth16-pd1-adam-lr0015-4ep-phase-conditioned-direct` | phase-conditioned direct, 4ep | 12.7731 | 12.6773 | 3.6642 |
| `20260603T103341-d64l2-depth16-pd1-adam-lr0015-32ep-phase-conditioned-direct` | phase-conditioned direct, 32ep | 7.1793 | 7.5365 | 2.9139 |
| `20260603T104709-d64l2-depth16-pd1-adam-lr0015-256ep-phase-conditioned-direct` | phase-conditioned direct, 256ep | 4.8388 | 5.3765 | 2.4267 |
| `20260603T112551-d64l2-depth16-pd1-adam-lr0015-512ep-phase-weighted-unwrapped` | phase-weighted, 512ep, unwrapped | 4.4887 | 5.0428 | 2.3342 |
| `20260603T174700-d64l2-depth16-pd1-adam-lr0015-512ep-phase-conditioned-direct` | phase-conditioned direct, 512ep | 4.5670 | 5.1197 | 2.3561 |
| `20260604T031818-d128l6-depth16-pd1-adam-lr0010-256ep-cq25k-phase-weighted-un` | d128/L6, phase-weighted, 256ep, cq25k, unwrapped | 4.0317 | 4.6199 | 2.2079 |
| `20260604T102205-d128l6-depth16-pd1-adam-lr0010-512ep-cq25k-phase-weighted-un` | d128/L6, phase-weighted, 512ep, cq25k, unwrapped | 3.9452 | 4.5201 | 2.1764 |
| `20260604T190600-d128l6-depth16-pd1-adam-lr0010-512ep-cq50k-phase-conditioned` | d128/L6, phase-conditioned direct, 512ep, cq50k | 3.9396 | 4.5106 | 2.1733 |
| `20260604T224307-d128l6-depth16-pd1-adam-lr0010-768ep-cq50k-phase-weighted-un` | d128/L6, phase-weighted, 768ep, cq50k, unwrapped | 4.0457 | 4.5713 | 2.1926 |
| `20260605T025029-d128l6-depth16-pd1-adam-lr0010-768ep-cq50k-phase-conditioned` | d128/L6, phase-conditioned direct, 768ep, cq50k | 3.9491 | 4.5030 | 2.1709 |
| `20260605T132031-d128l6-depth16-pd1-adam-lr0010-512ep-cq50k-phase-conditioned` | d128/L6, phase-conditioned direct, 512ep, cq50k, shuffled phase order | 3.9157 | 4.4974 | 2.1691 |
| `20260605T181308-d128l6-depth16-pd1-adam-lr0010-512ep-cq50k-phase-conditioned` | d128/L6, phase-conditioned direct, 512ep, cq50k, shuffled phase order seed123 | 3.9274 | 4.5137 | 2.1743 |

## Current Best

Best fixed-token result in this experiment:

| Run ID | checkpoint | fixed_token_ppl | rolling_byte_ppl | bits/byte |
|--------|-----------:|----------------:|-----------------:|----------:|
| `20260605T132031-d128l6-depth16-pd1-adam-lr0010-512ep-cq50k-phase-conditioned` | 448 | 3.9070 | 4.4939 | 2.1680 |

Best standard chunked rolling byte result in this experiment:

| Run ID | checkpoint | fixed_token_ppl | rolling_byte_ppl | bits/byte |
|--------|-----------:|----------------:|-----------------:|----------:|
| `20260605T132031-d128l6-depth16-pd1-adam-lr0010-512ep-cq50k-phase-conditioned` | 448 | 3.9070 | 4.4939 | 2.1680 |

The first major finding in this experiment was the d128/L6 `phase-weighted`
unwrapped run. It was the first result in this line to beat the previous
larger-model static best (`fixed_token_ppl=4.1705`, byte PPL `4.7086`), and it
showed that phase-aware training could keep improving past the point where the
static d128/L6 run had already peaked. That made `phase-weighted` the first
clear evidence that the phase signal was useful, not just a presentation
artifact.

The later d128/L6 `phase-conditioned direct` runs are now the numerical best,
with shuffled phase order improving both fixed-token and standard chunked
rolling byte PPL. This should be read as a follow-up on the phase-weighted
finding, not as a replacement for it: `phase-weighted` established the result
and remains the simpler/scalability-friendlier baseline, while
`phase-conditioned direct` shows that phase-specific targets can squeeze out a
little more on this setup. The shuffled phase-order result suggests the exact
phase presentation curriculum matters too.

Unlike the static d128/L6 runs, which peaked near epoch 128 and then degraded,
the phase-aware heldout curves improved through epoch 512. The first d128/L6
phase-weighted run improved as follows:

| Epoch | train (s) | fixed_token_ppl | rolling_byte_ppl | bits/byte |
|------:|----------:|----------------:|-----------------:|----------:|
| 1 | 31.9 | 15.3782 | 15.3675 | 3.9418 |
| 2 | 63.9 | 11.8660 | 11.8624 | 3.5683 |
| 4 | 128.3 | 9.9021 | 10.1150 | 3.3384 |
| 8 | 256.2 | 8.1138 | 8.4367 | 3.0767 |
| 16 | 507.5 | 6.8425 | 7.2852 | 2.8650 |
| 32 | 1017.9 | 5.8362 | 6.3354 | 2.6634 |
| 64 | 2020.8 | 5.0401 | 5.5677 | 2.4771 |
| 96 | 3026.5 | 4.6546 | 5.2409 | 2.3898 |
| 112 | 3527.3 | 4.4940 | 5.0764 | 2.3438 |
| 128 | 4044.3 | 4.4232 | 5.0062 | 2.3237 |
| 160 | 5071.3 | 4.2656 | 4.8521 | 2.2786 |
| 192 | 6101.1 | 4.1221 | 4.7071 | 2.2348 |
| 224 | 7106.3 | 4.0573 | 4.6433 | 2.2151 |
| 256 | 8096.4 | 4.0317 | 4.6199 | 2.2079 |

The 512-epoch run used the same setup but a longer cosine schedule, so
same-epoch checkpoints are not a pure continuation of the 256-epoch schedule.
It still improved beyond the 256 run by the later checkpoints:

| Epoch | train (s) | fixed_token_ppl | rolling_byte_ppl | bits/byte |
|------:|----------:|----------------:|-----------------:|----------:|
| 256 | 8286.1 | 4.1256 | 4.7296 | 2.2417 |
| 288 | 9277.5 | 4.0568 | 4.6542 | 2.2185 |
| 320 | 10264.3 | 4.0198 | 4.6077 | 2.2040 |
| 384 | 12240.9 | 3.9790 | 4.5623 | 2.1897 |
| 448 | 14226.3 | 3.9591 | 4.5405 | 2.1828 |
| 512 | 16231.8 | 3.9452 | 4.5201 | 2.1764 |

The 512-epoch phase-weighted unwrapped run was still improving at the recorded
checkpoints:

| Epoch | fixed_token_ppl | rolling_byte_ppl | bits/byte |
|------:|----------------:|-----------------:|----------:|
| 256 | 4.8238 | 5.3728 | 2.4257 |
| 384 | 4.5729 | 5.1277 | 2.3583 |
| 512 | 4.4887 | 5.0428 | 2.3342 |

The d64/L2 512-epoch phase-conditioned direct run also improved monotonically
over the recorded late checkpoints, but stayed behind d64/L2 phase-weighted:

| Epoch | fixed_token_ppl | rolling_byte_ppl | bits/byte |
|------:|----------------:|-----------------:|----------:|
| 256 | 4.9198 | 5.4825 | 2.4548 |
| 384 | 4.6663 | 5.2232 | 2.3849 |
| 512 | 4.5670 | 5.1197 | 2.3561 |

The d128/L6 phase-conditioned direct run was launched on an A100 via RunPod
with `chunk_queries=50000` after 8-epoch smoke checks at 25k, 50k, and 100k.
The smoke runs had identical trained-query counts and effectively identical
heldout metrics; 50k was marginally best at epoch 8. The 512-epoch run took
`6385.1s` of trainer wall time and continued improving through the final
checkpoint:

| Epoch | train (s) | fixed_token_ppl | rolling_byte_ppl | bits/byte |
|------:|----------:|----------------:|-----------------:|----------:|
| 384 | 4788.0 | 4.0336 | 4.6040 | 2.2029 |
| 448 | 5587.8 | 3.9654 | 4.5389 | 2.1823 |
| 512 | 6385.1 | 3.9396 | 4.5106 | 2.1733 |

A follow-up d128/L6 `phase-weighted` run matched `chunk_queries=50000` and used
a longer 768-epoch cosine schedule to test whether the simpler phase-weighted
method would catch up under the same chunking and more epochs. It did not catch
the phase-conditioned result. Its best late fixed-token checkpoint was epoch
512, while rolling byte PPL was lowest at epoch 768:

| Epoch | train (s) | fixed_token_ppl | rolling_byte_ppl | bits/byte |
|------:|----------:|----------------:|-----------------:|----------:|
| 512 | 6757.7 | 4.0047 | 4.5749 | 2.1938 |
| 576 | 7602.1 | 4.0316 | 4.5823 | 2.1961 |
| 640 | 8449.6 | 4.0621 | 4.5909 | 2.1988 |
| 704 | 9292.7 | 4.0659 | 4.5875 | 2.1977 |
| 768 | 10135.4 | 4.0457 | 4.5713 | 2.1926 |

A matched d128/L6 `phase-conditioned direct` 768-epoch run used the same
`chunk_queries=50000` and checkpoint schedule. Unlike the phase-weighted 768
run, the late fixed-token curve kept improving through epoch 768. It did not
beat the earlier 512-epoch phase-conditioned run on fixed-token PPL, but it did
produce the best standard chunked rolling byte PPL recorded in this experiment:

| Epoch | train (s) | fixed_token_ppl | rolling_byte_ppl | bits/byte |
|------:|----------:|----------------:|-----------------:|----------:|
| 512 | 6795.3 | 4.0249 | 4.5968 | 2.2006 |
| 576 | 7644.1 | 4.0124 | 4.5710 | 2.1925 |
| 640 | 8488.2 | 3.9984 | 4.5528 | 2.1868 |
| 704 | 9334.6 | 3.9749 | 4.5295 | 2.1794 |
| 768 | 10178.3 | 3.9491 | 4.5030 | 2.1709 |

A matched d128/L6 `phase-conditioned direct` 512-epoch run then changed only the
phase presentation order from the regular sequential cycle to a deterministic
shuffle (`experimental.phase_order: shuffle`, seed 42). It covered each phase
once per 64-epoch cycle, but in permuted order. This was the first run to move
the best fixed-token PPL below 3.92 and the best standard chunked rolling byte
PPL below 4.50. The best checkpoint was epoch 448; epoch 512 regressed slightly
but still beat the earlier sequential 512 run:

| Epoch | train (s) | fixed_token_ppl | rolling_byte_ppl | bits/byte |
|------:|----------:|----------------:|-----------------:|----------:|
| 256 | 3276.6 | 4.1301 | 4.7236 | 2.2399 |
| 288 | 3697.0 | 4.0411 | 4.6435 | 2.2152 |
| 320 | 4099.4 | 4.0105 | 4.6058 | 2.2034 |
| 384 | 4908.6 | 3.9322 | 4.5156 | 2.1749 |
| 448 | 5703.9 | 3.9070 | 4.4939 | 2.1680 |
| 512 | 6504.5 | 3.9157 | 4.4974 | 2.1691 |

A second shuffled phase-order seed (`phase_order_seed: 123`) was run as a
variance check. It improved over the earlier sequential 512-epoch run on
fixed-token PPL at its best checkpoint, but did not reproduce the seed-42 best:

| Epoch | train (s) | fixed_token_ppl | rolling_byte_ppl | bits/byte |
|------:|----------:|----------------:|-----------------:|----------:|
| 384 | - | 3.9343 | 4.5262 | 2.1783 |
| 448 | - | 3.9257 | 4.5122 | 2.1738 |
| 512 | 6397.3 | 3.9274 | 4.5137 | 2.1743 |

For reference, the best recorded larger-model static baseline from
`rnd/baseline-calibration-v2-static-sampled` was:

| Run ID | Checkpoint | Architecture | fixed_token_ppl | rolling_byte_ppl | bits/byte |
|--------|-----------:|--------------|----------------:|-----------------:|----------:|
| `20260601T064609-d128l6-depth16-pd1-adam-lr0010-512ep-wrap` | 128 | d128/L6 | 4.1705 | 4.7086 | 2.2353 |

## External Comparisons

The strongest current result is now in the range of published Tiny Shakespeare
character-model baselines, but the comparisons are not all strict
apples-to-apples.

| System | Context | Split / scorer | Loss nats | bits/char or byte | PPL |
|--------|--------:|----------------|----------:|------------------:|----:|
| AGPT d128/L6 phase-conditioned direct, shuffled phase order | 16 chars | current sampled heldout, fixed-token, chunked | 1.3628 | 1.9661 char | 3.9070 |
| AGPT d128/L6 phase-weighted | 16 chars | current sampled heldout, fixed-token, chunked | 1.3725 | 1.9801 char | 3.9452 |
| AGPT d128/L6 phase-weighted | 16 chars | current sampled heldout, fixed-token, concatenated | 1.3747 | 1.9833 char | 3.9540 |
| AGPT d128/L6 phase-conditioned direct, shuffled phase order | 16 chars | current sampled heldout, lm-eval rolling byte, chunked | 1.5027 | 2.1680 byte | 4.4939 |
| AGPT d128/L6 phase-weighted | 16 chars | current sampled heldout, lm-eval rolling byte, chunked | 1.5085 | 2.1764 byte | 4.5201 |
| AGPT d128/L6 phase-weighted | 16 chars | current sampled heldout, lm-eval rolling byte, concatenated | 1.5032 | 2.1687 byte | 4.4961 |
| KenLM modified Kneser-Ney | 7-gram | current sampled heldout, KenLM `query` | 1.4155 | 2.0421 char | 4.1183 |
| nanoGPT char Transformer | 256 chars | nanoGPT 90/10 contiguous split, reported val loss | 1.4697 | 2.1203 char | 4.3479 |

nanoGPT's reported Tiny Shakespeare character run uses the canonical
`data/shakespeare_char` corpus (`1,115,394` chars, 65-char vocab), but a
different validation protocol from this experiment: first 90% train / last 10%
validation, `block_size=256`, `n_layer=6`, `n_head=6`, `n_embd=384`, and
reported best validation loss `1.4697`. Since PyTorch cross entropy is in
natural-log units, the corresponding perplexity is `exp(1.4697)=4.3479`; it
does not become `2^1.4697` unless the loss was already measured in bits.

The clean claim is therefore not that AGPT has beaten nanoGPT under an
identical protocol. The defensible claim is that the best phase-aware AGPT run
has reached nanoGPT-class Tiny Shakespeare character perplexity while using a
16-character explicit window, versus nanoGPT's 256-character block context.
The strict same-split baseline in hand is KenLM: AGPT's fixed-token PPL
`3.9070` is below the current-split KenLM modified-KN best of `4.1183`.

The phase-weighted concatenated-heldout sanity eval uses the same heldout bytes
as the chunked run, but treats them as one document. It is close to the chunked
result:
fixed-token PPL moves from `3.9452` to `3.9540`, while rolling byte PPL moves
from `4.5201` to `4.4961`. This suggests the headline result is not an artifact
of the heldout chunk boundary convention.

## Scaling Notes

The phase variants differ materially in how they should scale:

- `phase-weighted` is the practical path. It needs prefix/node phase-mass
  statistics and changes how sampled losses are weighted, but it does not
  multiply the target table by every phase.
- `phase-conditioned` is much more expensive if materialized directly, because
  targets become prefix-by-phase distributions. The storage and sampling
  pressure grow with the number of active `(prefix, phase)` pairs, not just the
  number of prefixes.
- As corpus size grows, rare prefixes spread out and many phase bins will have
  little or no mass. That sparsity is helpful if we keep the representation
  sparse, but wasteful if we allocate dense phase state.
- The likely scalable recipe is to keep a global prefix target, store compact
  phase summaries only where there is enough mass, and sample phase-aware
  presentations from those summaries. Bigger corpora should give better phase
  estimates for common prefixes, but deeper/rarer prefixes will still need
  smoothing or fallback to global targets.

Open scaling question: whether the useful phase signal is local modulo-position
structure that saturates quickly, or whether larger corpora expose enough
stable phase-conditioned continuation structure to justify richer sparse
phase-conditioned targets.

## Interpretation

- Phase-aware presentation did eventually improve the d64/L2 line, but the
  useful variant was not the first sampled-unit-phase or wrapped presentation.
- `phase-weighted` with unwrapped RoPE positions was the first strong result in
  this experiment. It established that phase-aware training could beat the
  static d128/L6 baseline and keep improving through 512 epochs.
- The d64/L2 `phase-conditioned direct` run improved through 512 epochs but did
  not beat phase-weighted. The d128/L6 phase-conditioned direct run did beat
  the d128/L6 phase-weighted result by a small margin at epoch 512, making it
  the current numerical best.
- Matching `chunk_queries=50000` and extending d128/L6 phase-weighted to 768
  epochs did not close that gap. This supports, but does not yet fully prove,
  the hypothesis that phase-conditioned targets add useful information beyond
  phase-weighted global targets.
- Combining phase-aware training with d128/L6 beat the previous static d128/L6
  project best and did not show the static run's post-128 heldout degradation
  by epoch 512.
- The d128/L6 phase-weighted curve crossed the `4.0` fixed-token threshold and
  was still improving in the 512-epoch `cq25k` run, though the gains from 384
  to 512 were tapering. Under the 768-epoch `cq50k` schedule, its fixed-token
  PPL was already worse by epoch 576, while byte PPL only recovered slightly by
  epoch 768.

## Reproduction

The configs are checked into this directory. Example:

```bash
XDG_CACHE_HOME=/tmp/agpt-xdg-cache \
HF_DATASETS_CACHE=/tmp/agpt-hf-datasets \
HF_HOME=/tmp/agpt-hf-home \
bin/agpt_experiment \
  --config rnd/sampled-node-phase-rope/d64L2-depth16-pd1-adam-lr0015-512ep-phase-weighted-unwrapped.yml \
  --trainer v2
```

Prerequisites:

- Build `bin/agpt_train_v2` and `bin/agpt_experiment`.
- Ensure the trie path in the config exists or is rebuilt.
- Ensure `/tmp/agpt_snp_25ep_position_data` exists. Use the local
  `build_position_table` / phase validation tools from this branch to rebuild
  the position data if needed.

## Artifact Notes

The crash on 2026-06-03 interrupted post-training evaluation for
`20260603T112551-d64l2-depth16-pd1-adam-lr0015-512ep-phase-weighted-unwrapped`.
The checkpoint files were already present through epoch 512. The missing HF
conversion/eval artifacts, `result.json`, and `runs.json` were reconstructed
from the saved checkpoints using the same conversion/eval commands as
`src/tools/agpt_experiment.cr`.

The durable records are:

- Per-run `result.json`
- Per-run `meta.json`
- `rnd/sampled-node-phase-rope/runs.json`

Large model checkpoints and HF conversion directories are output artifacts, not
the core experiment record.
