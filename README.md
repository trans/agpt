# AGPT — Aggregated-Gradient Pretraining

Research project on aggregated-gradient pretraining for autoregressive
language models. Trains a transformer on a prefix-trie representation of
the corpus, factoring the gradient over branching subtrees rather than
sliding context windows. Built on top of the
[µGPT](https://github.com/trans/microgpt) Crystal/CUDA components kit.

## What's here

- **[Paper](notes/agpt/paper.md)** — the gradient-factorization theorem,
  memory-scalable implementation, and empirical results on Shakespeare.
- **CUDA training engine** (`src/cuda/agpt_train.cu`, `bin/agpt_train`) —
  the GPU trainer. Radix-compressed trie input, per-subtree KV-cache
  scoping, bigram partitioning, auto-LR scaling, frequency-based
  pruning, and several sampler modes (L1 uniform, L2 root-child uniform,
  L3 mass-weighted, L4 path).
- **Trie builders** — `bin/agpt_build_index` produces a leveled
  per-depth trie from a corpus; `bin/agpt_build_radix` compresses unary
  chains into multi-character edges.
- **Wrap-around corpus synthesis** (`bin/synth_wrap_corpus`) — sample
  arbitrary-length token sequences from a depth-D trie via leaf→root
  wrapping with bridge-token sampling.
- **Diagnostic tools** — `bin/radix-verify`, `bin/trie-profile`,
  `bin/bayesian-posterior`, `bin/convergence`, `bin/check_weights`.

## Building

```sh
shards install         # resolves the µGPT shard dependency
just build-all         # compiles every AGPT binary
just build-microgpt-tools  # also build bin/microgpt + bin/perplexity from the shard
```

CUDA kernels are sourced from the µGPT shard at `lib/microgpt/`. `nvcc`
on `PATH` (or at `/opt/cuda/bin/nvcc`) is required for the GPU trainer.

## Quick start (Shakespeare, depth 32)

```sh
# 1. Build a depth-32 leveled trie from the corpus.
bin/agpt_build_index --corpus data/input.txt --max-depth 32

# 2. Compress it to a radix trie.
bin/agpt_build_radix --leveled /tmp/agpt_input_d32

# 3. Train.
cp data/input.random.model /tmp/run.model
bin/agpt_train \
  --model /tmp/run.model --trie-dir /tmp/agpt_input_d32_radix \
  --save /tmp/run.model --epochs 3 --lr 3e-3 \
  --optimizer rmsprop --rmsprop-beta 0.999 \
  --lr-schedule warmup-cosine --warmup-epochs 1 \
  --entropy-lambda 1.0 --mass-weight linear --no-accumulate

# 4. Evaluate held-out perplexity.
bin/perplexity --model /tmp/run.model --file data/input.txt \
  --max-positions 4096 --backend openblas
```

## Tests

```sh
just test          # Crystal specs + foundational CUDA-trainer tests
just test-crystal  # Crystal specs only
just test-agpt     # foundational tests (gradient flow, build, NaN, PPL)
```

Foundational tests require `bin/microgpt` and `bin/perplexity` from the
µGPT shard — `just build-microgpt-tools` builds them.

## Layout

```
src/agpt/        Crystal: trie, radix, samplers, KV store, walkers
src/cuda/        agpt_train.cu (GPU trainer; kernels.cu lives in µGPT)
src/tools/       Crystal CLIs (builders, synthesis, diagnostics)
spec/            Crystal specs for AGPT-only modules
tests/           Foundational shell tests
notes/agpt/      Design notes, paper drafts, status
notes/grants/    Grant pitch
rnd/             Research logs (per-experiment subdirectories)
```

## License

Released under the [PolyForm Noncommercial License
1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/) — see
[`LICENSE`](LICENSE). Academic and research use is permitted and
encouraged. Commercial licensing available — see
[`COMMERCIAL_LICENSE.md`](COMMERCIAL_LICENSE.md) or contact
**`transfire@gmail.com`**.
