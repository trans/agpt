# D124 Radix Feasibility Notes

Status: preliminary feasibility pass on Shakespeare, using the full radix built
at `rnd/radix-depth124/shake_d124_radix`.

## Why D124

On the Shakespeare corpus, raw context uniqueness first reaches zero repeated
contexts at depth 124. This makes `d=124` a useful probe for the idea that a
larger AGPT tree depth can expose the corpus's unary identity paths rather than
collapsing them into shallow radix end caps.

The practical question is whether the full d124 radix is too large to train
with CUDAX, and if so which part is the actual blocker.

## Radix Size

Full d124 radix:

- radix records: `1,668,175`
- total edge chars: `129,635,306`
- compact slots / mass>1 edge chars: `1,125,836`
- max endpoint depth: `124`
- on-disk size: about `302 MiB`

The important split is that most edge chars are mass-1 cap tails:

| Category | Edge chars |
|---|---:|
| total edge chars | `129,635,306` |
| mass > 1 edge chars | `1,125,836` |
| mass = 1 edge chars | `128,509,470` |
| mass > 1 interior edge chars | `1,125,718` |
| mass > 1 cap edge chars | `118` |

So the K/V cache does not scale with all 129.6M edge chars. It scales with the
1.126M compact slots.

## Current CUDAX Loader Memory

Current `src/cudax/io.cuh` materializes several arrays over all edge chars.
For d124, the rough steady host allocation is about `1.6 GiB`.

Approximate current host allocations:

| Structure | Size |
|---|---:|
| parent/start/len/depth/mass arrays | `31.8 MiB` |
| `edge_tokens_flat` int32 | `494.5 MiB` |
| `real_pos_of_char` int32 | `494.5 MiB` |
| `compact_slot` int32 | `494.5 MiB` |
| counts arrays | `23.4 MiB` |
| ancestor offsets + ids | `57.8 MiB` |
| steady total | `~1.6 GiB` |

The ancestor list is not the main memory problem here. It is only about
`51.4 MiB` for `ancestor_char_ids`, because radix compression keeps ancestor
paths bounded by endpoint depth.

The three full edge-char int32 arrays are the obvious host-memory waste.

## CUDAX Plan Estimates

Plan-only run:

```text
bin/agpt_train_v2 --mode plan \
  --model rnd/cudax-section2-progressive/seeds/shake-d128L6-h8-dff512-d16-seed42.model \
  --trie-dir rnd/radix-depth124/shake_d124_radix \
  --chunk-queries 50000 \
  --anc-grad
```

The trainer reconciles the model header `seq_len=16` to trie max depth 124 for
planning:

```text
seq_len reconcile: model header says 16, trie max_depth=124 -> effective 124
```

For `d_model=128`, `L=6`, `chunk_queries=50000`:

| Item | Estimate |
|---|---:|
| query positions / epoch | `129,635,306` |
| chunks / epoch | `~2,630` |
| K/V cache | `3,458.6 MB` |
| chunk buffers | `5,229.1 MB` |
| params + grads | `9.7 MB` |
| optimizer state | `9.7 MB` |
| combined estimate | `8,706.9 MB` |

At lower chunk sizes:

| `chunk_queries` | chunks / epoch | K/V cache | chunk buffers | combined estimate |
|---:|---:|---:|---:|---:|
| `50000` | `~2,630` | `3,458.6 MB` | `5,229.1 MB` | `8,706.9 MB` |
| `20000` | `~6,533` | `3,458.6 MB` | `2,096.6 MB` | `5,574.4 MB` |
| `10000` | `~13,067` | `3,458.6 MB` | `1,059.5 MB` | `4,537.4 MB` |

The laptop-safe setting is likely `chunk_queries=10000` or lower. The tradeoff
is more chunks per epoch.

## Actual Blockers

The K/V cache is not the surprising blocker. With compact slots, d124 only
needs about 14% more K/V slots than the d16 Shakespeare plan.

The actual blockers are:

- Full edge-char metadata: current loader keeps about `1.5 GiB` just in
  `edge_tokens_flat`, `real_pos_of_char`, and `compact_slot`.
- Chunk activation memory: because max K/V length can reach 124, chunk buffers
  dominate at `chunk_queries=50000`.
- Work per epoch: current training treats every edge char as a query position,
  so d124 means `129.6M` query positions per epoch, about 14x the d16 radix.

The query-position count is probably the largest practical problem for full
d124 training.

## Mitigations

Low-risk mitigations:

- Run d124 probes with `chunk_queries=10000` to keep GPU memory reasonable.
- Keep `d_model=128`, `L=6` for the first d124 test; avoid `L=8` until the
  memory and runtime path is known.

Loader/memory mitigations:

- Store `edge_tokens_flat` as `uint16_t`/`int16_t` in memory; vocab is 65.
- Remove the full `real_pos_of_char` array where positions can be derived from
  `edge_first_char_depth + local_offset - 1`.
- Replace the full `compact_slot[total_edge_chars]` array with per-edge
  interval data or chunk-local compact slot metadata.
- Avoid copying global `compact_slot` to the GPU; pass chunk-local compact
  slots where possible.

Runtime/work mitigations:

- Add a d124 sampling mode for mass-1 cap tails instead of training every
  mass-1 cap character every epoch.
- Consider preserving long unary paths as attention context while limiting
  which mass-1 cap positions contribute loss in each epoch.
- Use document/corpus wrapping when building deep tries to avoid corpus-tail
  artifacts at high depths.

## Current Read

D124 is not obviously impossible. The compact K/V cache is manageable. A
straight full d124 exhaustive epoch is probably too slow on the laptop because
it expands to 129.6M query positions, not because the K/V cache explodes.

The first serious probe should be a small-run d124 plan with:

- `d_model=128`
- `L=6`
- `chunk_queries=10000`
- depth 124 trie
- very small epoch count or limited units first

If that runs, the next engineering target should be reducing full edge-char
metadata and adding a principled cap-tail sampling policy.
