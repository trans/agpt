# Successor Prefix Attention

This line tests whether a depth-16 AGPT path can expose a real observed
continuation path directly, without collapsing identity into token/phase
buckets.

For a depth cap occurrence:

- `end` anchor: `A = corpus[pos..pos+d-1]`, successor starts at `pos+d`.
- `head` anchor: `A` is the same cap, successor starts at the first token of
  the cap's compressed radix edge: `pos + first_char_depth - 1`.

The first diagnostic is implemented in `src/tools/successor_prefix_map.cr`.
It walks the corpus through the prefix radix trie, records cap occurrences, and
aggregates observed `A -> B` successor counts while preserving radix node ids.
It also writes trainer-readable deterministic tables:

- `successors_end.bin`
- `successors_head.bin`

The table format is `ASUC` v1 plus one `Int32` successor radix id per source
radix id, with `-1` for no deterministic retained successor.

## 2026-06-05 Diagnostic

Trie:

```text
/tmp/agpt_snp_25ep_prefix_radix
```

Corpus:

```text
/home/trans/Projects/agpt/data/.splits/4fa9aec1db6b3aea/train_corpus.txt
```

Command shape:

```text
bin/agpt_successor_prefix_map \
  --trie /tmp/agpt_snp_25ep_prefix_radix \
  --corpus /home/trans/Projects/agpt/data/.splits/4fa9aec1db6b3aea/train_corpus.txt \
  --out rnd/successor-prefix-attention/successor-prefix-map
```

Strict mass-one variant:

```text
bin/agpt_successor_prefix_map \
  --trie /tmp/agpt_snp_25ep_prefix_radix \
  --corpus /home/trans/Projects/agpt/data/.splits/4fa9aec1db6b3aea/train_corpus.txt \
  --out rnd/successor-prefix-attention/successor-prefix-map-mass1 \
  --mass-one-only
```

## Results

All depth-16 cap occurrences:

| metric | value |
|---|---:|
| corpus tokens | 1,059,634 |
| cap occurrences | 1,059,634 |
| distinct cap nodes | 1,027,793 |
| cap nodes with `edge_mass=1` | 1,010,616 (98.33%) |
| cap nodes with compressed edge len > 1 | 1,009,093 (98.18%) |
| end-anchor source nodes | 1,027,793 |
| end-anchor distinct edges | 1,058,682 |
| end-anchor single-successor nodes | 1,011,261 (98.39%) |
| end-anchor fanout p50 / p90 / p99 / max | 1 / 1 / 2 / 224 |
| end-anchor top-1 / top-4 / top-8 occurrence coverage | 97.08% / 99.16% / 99.41% |
| head-anchor source nodes | 1,027,793 |
| head-anchor distinct edges | 1,056,095 |
| head-anchor single-successor nodes | 1,012,377 (98.50%) |
| head-anchor fanout p50 / p90 / p99 / max | 1 / 1 / 2 / 156 |
| head-anchor top-1 / top-4 / top-8 occurrence coverage | 97.27% / 99.23% / 99.47% |

Mass-one caps only:

| metric | value |
|---|---:|
| cap occurrences | 1,010,616 |
| distinct cap nodes | 1,010,616 |
| compressed edge len > 1 | 994,333 (98.39%) |
| end-anchor skipped because successor was not mass-one | 43,659 |
| end-anchor source nodes / edges / occurrences | 966,957 / 966,957 / 966,957 |
| end-anchor single-successor nodes | 966,957 (100.00%) |
| head-anchor skipped because successor was not mass-one | 43,025 |
| head-anchor source nodes / edges / occurrences | 967,591 / 967,591 / 967,591 |
| head-anchor single-successor nodes | 967,591 (100.00%) |

## Interpretation

The basic shape is tractable. For all caps, nearly every source node has a
single observed successor and the remaining fanout is very concentrated. For
strict mass-one caps, the map is deterministic for every retained source node;
the only loss comes from cases where the continuation cap is not also mass-one.

The head anchor is slightly more concentrated than the end anchor in the all-cap
view. That makes it worth keeping both variants through the first trainer
prototype rather than choosing prematurely.

## Trainer Prototype

The first CUDAX prototype uses the strict deterministic mass-one table. It keeps
the representation sparse:

- if `successor[A] == -1`, the node is unchanged and has no empty successor
  positions;
- if `successor[A] = B`, the full B path is appended as zero-loss continuation
  rows assigned to A's attention group;
- appended rows use `query_weight=0`, so they do not add direct loss;
- appended rows use `char_pos=-1`, so they do not write into the global compact
  K/V cache;
- end-anchor rows use RoPE positions `16..31` for depth-16 tries.

The initial config is:

```text
rnd/successor-prefix-attention/d128L6-depth16-pd1-adam-lr0010-8ep-cq50k-successor-end.yml
```

Validation:

```text
bin/agpt_train_v2 \
  --config rnd/successor-prefix-attention/d128L6-depth16-pd1-adam-lr0010-8ep-cq50k-successor-end.yml \
  --validate-only
```

Result:

```text
mode=train-epoch, trie_depth=16, context_seq_len=16, rope_seq_len=32, successor_prefix=true
```

Forward smoke:

```text
bin/agpt_train_v2 \
  --config rnd/successor-prefix-attention/d128L6-depth16-pd1-adam-lr0010-8ep-cq50k-successor-end.yml \
  --run-forward-prefix
```

Result:

```text
mode=forward
successor_prefix=end deterministic=966957 skipped_fanout=0
runtime max_kv_len=32
forward full-depth scored chunk executed
```

Cap-heavy smoke:

```text
bin/agpt_train_v2 \
  --config rnd/successor-prefix-attention/d128L6-depth16-pd1-adam-lr0010-8ep-cq50k-successor-end.yml \
  --mode train-small \
  --steps 72
```

Result: all 72 chunks of the largest unit completed and a single Adam step was
applied. This exercises later cap-heavy chunks, not only the shallow first
chunk.
