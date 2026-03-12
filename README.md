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

- `Mat` tracks global allocated bytes with a configurable cap (default 2 GiB).
  Raises with a detailed message before exceeding the limit.
- `GC.collect` runs every 10 training steps to reclaim intermediate matrices.
- `just run` wraps execution with `ulimit -v` (default 8 GiB) as an OS-level
  safety net.

## Usage

```
microgpt <text_file> [steps] [head_type] [backend] [seq_len]
```

- **text_file** — input text (e.g. `data/input.txt`)
- **steps** — training steps (default 1000)
- **head_type** — `uniform`, `exponential`, `prime`, `pyramid`, `ones`
- **backend** — `crystal`, `openblas`, `cublas`
- **seq_len** — sequence length (default 128)

```sh
# Quick test
just build-release
just run data/input.txt 1000 uniform openblas 32

# Benchmarks
just bench 100
```

## Building

```sh
just build           # debug build (CPU only)
just build-release   # release build (CPU only)
just build-cuda      # release build with GPU support
```

## Contributors

- [Thomas Sawyer](https://github.com/your-github-user) - creator and maintainer
