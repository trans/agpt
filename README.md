# MicroGPT

A minimal Transformer language model in Crystal. Character-level, from scratch,
with heterogeneous attention head support and pluggable backends (Crystal, OpenBLAS, cuBLAS).

## Design Decisions

### Data Chunking

Sequential walk through the text with configurable stride. Default stride equals
`seq_len`, giving non-overlapping chunks with full coverage per epoch.

- ~1.1M tokens in tinyshakespeare, vocab size 65 (character-level).
- At seq_len=32, one epoch = ~34k steps.
- No random sampling — every token is seen exactly once per epoch.
- Stride is configurable: set below seq_len for overlapping context at chunk
  boundaries if desired.

### Embedding / Unembedding Weights

W_e (token embeddings) and W_unembed (output projection) are **independent**.
`Embedding.token_emb` is (vocab × d_model), `OutputHead.proj` is a separate
`Linear` with its own (d_model × vocab) weight matrix. No weight tying.

### W_o and Heterogeneous Heads

`MultiHeadAttention` supports heads of different dimensions. Head outputs are
concatenated column-wise into a (seq_len × d_model) matrix — head dims must sum
to d_model. W_o is a plain (d_model × d_model) linear projection over the full
concatenated vector. It has no awareness of head boundaries; it treats the
concatenation uniformly.

### Feed-Forward Dimension

`d_ff = d_model` (1:1 ratio). The FF block is two linear layers:
d_model → d_ff → d_model. At d_model=64 this keeps the parameter count low
(~61k total) and training fast (~116 steps/sec on OpenBLAS).

### Memory Protection

- `Mat` tracks global allocated bytes with a configurable cap (default 3 GiB).
  Raises with a detailed message before exceeding the limit.
- `GC.collect` runs every 10 training steps to reclaim intermediate matrices.
- `just run` wraps execution with `ulimit -v` (default 8 GiB) as an OS-level
  safety net.

## Usage

The CLI uses a JSON schema for arguments. All flags use `--flag value` syntax.

### Window Training (standard)

```sh
# Quick training run
./bin/microgpt data/input.txt --steps 2000 --d-model 64 --n-layers 2 --seq-len 32

# With held-out validation
./bin/microgpt data/input.txt --steps 5000 --d-model 64 --n-layers 2 --seq-len 32 \
  --val-tokens 5000 --val-interval 30

# GPU accelerated
./bin/microgpt data/input.txt --steps 5000 --d-model 64 --n-layers 2 --seq-len 32 \
  --backend cublas

# Other options
--seed 42              # reproducible init
--lr 0.0003            # learning rate (default 0.0003)
--no-save              # don't save checkpoint
--model path.model     # save/load checkpoint path
```

### AGPT Training (trie-walk)

Trains on a prefix trie of the corpus. Shared prefixes are computed once.

```sh
# Basic AGPT run
./bin/microgpt data/input.txt --steps 10 --d-model 32 --n-layers 1 --seq-len 16 \
  --agpt --agpt-max-starts 2000

# With held-out validation (for comparison with window training)
./bin/microgpt data/input.txt --steps 20 --d-model 32 --n-layers 1 --seq-len 16 \
  --agpt --agpt-max-starts 2000 --val-tokens 5000 --val-interval 10

# AGPT-specific options
--agpt                     # enable AGPT trie-walk mode
--agpt-max-starts 2000     # number of corpus start positions (0 = all)
--agpt-start-offset 0      # deterministic offset for start positions
--agpt-progress 1000       # print trie build progress every N starts
--agpt-save-index path.idx # save built trie index to disk
--agpt-load-index path.idx # load previously saved trie index
--build-only               # build/report trie without training
```

**AGPT terminology:**
- **steps** = number of epochs (full trie traversals)
- **epoch** = one forward pass over all trie nodes + backward + weight updates
- **starts** = number of corpus positions used to build the trie (more starts = more prefix sharing but slower)

### Comparison Runs

To compare AGPT vs window training fairly, use the same model config, seed, and held-out split:

```sh
# Window baseline
./bin/microgpt data/input.txt --steps 5000 --d-model 32 --n-layers 1 --seq-len 16 \
  --val-tokens 5000 --val-interval 10 --seed 42 --no-save > window.log 2>&1

# AGPT comparison
./bin/microgpt data/input.txt --steps 20 --d-model 32 --n-layers 1 --seq-len 16 \
  --val-tokens 5000 --val-interval 10 --seed 42 --no-save \
  --agpt --agpt-max-starts 2000 > agpt.log 2>&1

# Extract held-out CE curves (plot wall-clock vs held_out_ce)
grep '\[val\]' window.log
grep '\[val\]' agpt.log
```

### Backends

- `crystal` — pure Crystal CPU (default, no dependencies)
- `openblas` — CPU with OpenBLAS acceleration
- `cublas` — GPU via cuBLAS + custom CUDA kernels (requires NVIDIA GPU + CUDA toolkit)

## Building

```sh
just build           # debug build (CPU only)
just build-release   # release build (CPU only)
just build-cuda      # release build with GPU support
```

## Contributors

- [Thomas Sawyer](https://github.com/trans) — creator and maintainer

## License

This repository is released under the [PolyForm Noncommercial License
1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/) —
see [`LICENSE`](LICENSE).

Academic research, personal study, and use by educational or
research institutions are permitted and encouraged.

### Commercial licensing available

If you want to use this code in a commercial product, service, or
for-profit internal workflow, a separate commercial license is
required. Terms are flexible and sized to the deployment —
startup-friendly arrangements are welcome.

See [`COMMERCIAL_LICENSE.md`](COMMERCIAL_LICENSE.md) for details, or
contact **`transfire@gmail.com`** directly.
