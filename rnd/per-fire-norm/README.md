# per-fire-norm — fixing chunk count out of the gradient math

> **Status (2026-05-20):** per-fire (1/N) is now the default normalizer on main as of commit **609e7ab**. The per-chunk (1/T_q_chunk) behavior is gone. This README documents how we got there.

## TL;DR

The trainer used to average weight gradients **per-chunk** (`grad_scale = 1/T_q_chunk` applied to each chunk's contribution). That's mathematically incorrect: chunks are a memory-layout artifact, and with unlimited memory we'd run one giant chunk where `T_q_chunk == N` and grad_scale would naturally be `1/N`. The per-chunk normalizer happens to up-weight events in small partial chunks, which BFS-orders into the deepest queries of each subtree — an accidental partial implementation of depth-weighted loss. It helped on Shakespeare (where 13% of chunks are partial) and hurt slightly on Gutenberg (where only 4% are partial). The fix: accumulate raw per-chunk, scale once by `1/fire_events` at each Adam fire (`cublasSscal`). Anc-grad delta is preserved.

## Three regimes tested

| | per-chunk (was default) | per-fire (now default) | raw (no normalizer) |
|---|---|---|---|
| `grad_scale` per chunk | `1/T_q_chunk` | `1.0` | `1.0` |
| fire-end scaling | none | `cublasSscal(d_grads, 1/fire_events)` | none |
| event weight in dW | `1/T_q_chunk` (over-weights events in small chunks) | `1/N` (uniform) | raw sum (uniform, no normalization) |
| matches "unlimited memory" reference? | no | yes | yes in direction, not magnitude |

3-seed mean training-set PPL (10 SE, RMSprop, warmup-cosine, --no-accumulate, partition-depth 1):

| corpus | per-chunk OFF | per-fire OFF | raw OFF | per-chunk ON | per-fire ON | raw ON |
|---|---|---|---|---|---|---|
| Shakespeare 1M | **7.79** | 8.34 | 8.06 | **7.29** | 7.81 | 7.77 |
| Gutenberg 5M | 8.25 | **7.65** | 8.44 | 7.95 | **7.36** | 7.96 |

(OFF = `--anc-grad` not set, ON = set.)

Anc-grad delta is robust across regimes:

| corpus | per-chunk Δ | per-fire Δ | raw Δ |
|---|---|---|---|
| Shakespeare | -6.4% | -6.4% | -3.6% |
| Gutenberg | -3.6% | -3.8% | -5.7% |

(Notes: anc-grad held-out PPL was measured only under the per-chunk regime — see `rnd/anc-grad/`. The held-out delta from that work doesn't change conceptually with the normalizer switch, but the absolute held-out numbers will shift if re-measured under per-fire.)

## The mechanism — what per-chunk averaging was actually doing

Chunks are built BFS-ordered through each subtree's mass>1 nodes. The first chunks of a subtree saturate at `CHUNK_QUERIES = 50000` and contain shallow-to-middle queries. The **last** chunk holds the deepest queries and is partial (smaller than 50k).

Under per-chunk averaging:
- Each event in a saturated chunk contributes `1/50000` to `dW`
- Each event in a 25k partial chunk contributes `1/25000` ≈ 2× more

So events in partial chunks (= deepest queries) are silently up-weighted by ~2×. The training signal tilts toward deep queries.

Chunk-size distribution by corpus (one-epoch instrument):

| | Shakespeare | Gutenberg |
|---|---|---|
| total chunks/epoch | 225 | 730 |
| saturated chunks at ~50k | 72% | 91% |
| partial chunks | **30 (13.3%)** | **32 (4.4%)** |
| partial chunk mean T_q | 24,721 | 21,726 |
| coefficient of variation | **0.378** | 0.193 |

Shakespeare has ~3× more partial chunks per total (13.3% vs 4.4%) and 2× the CV. That explains why the per-chunk regime helps Shakespeare more than Gutenberg — Shakespeare gets more of the accidental deep-node up-weighting.

## Trans's reframing — mass vs relevance

> A unigram-root node like "A" has high mass but no prefix grounding — it represents `P(next | "A" anywhere in corpus)`, essentially a bigram statistic. The training signal asks the model to handle a low-context prediction we will never make at inference. Deep nodes are low-mass but high-relevance: `P(next | long meaningful prefix)` specifies the predictions we actually care about. So shallow ≠ trustworthy in the sense of "useful for what the model will be asked to do".

This frame explains why the accidental deep-node up-weighting **helped** rather than hurt: it's tilting the gradient toward queries that are more contextually informative, which is the right direction. But:

1. It's accidental. Not designed.
2. It only fires on partial chunks — uneven application.
3. It's tangled with a math-incorrect chunk-count normalization.

The principled redesign would be an **explicit depth-weighted loss** `w(d)` applied per event, with clean per-event averaging. Connects to `project_predictive_certainty_weighting.md` (U-shape weighting prefix↔suffix depth) and `project_blending_experiment.md` (count-aware `log(1+count)` helped d=16, hurt d=32). Not implemented in this thread — the current commit just removes the accidental implementation.

## The footgun caught during this work

`src/cuda/agpt_train.cu:6407` (pre-fix) defaulted `save_path = model_path` when `--save` was omitted. A one-epoch chunk-distribution diagnostic without `--save` silently appended optimizer state to `/tmp/seed1.model` (adam_t=130). The raw-no-scale sweep then loaded that contaminated state, trained 650 more, ended at adam_t=780, and the last ~130 steps ran at literal `lr=0` (cosine schedule clamped past `total_opt_steps_estimate`). The "raw plateaus dead" finding I initially reported was 100% this dead-LR tail, not a real RMSprop saturation.

Caught by codex-agpt — the adam_t=130 in the log header was the smoking gun.

Fix (commit **ce48b0e**): if `--save` is omitted, print a stderr warning that weights + optimizer state will be discarded, and write nothing. Preserves the "don't silently discard" principle without the silent-mutation-of-input failure mode. Seeds 2 and 3 of the raw sweep were unaffected (clean cold starts); seed 1 was rerun cold.

## Open question — raw vs per-fire under RMSprop

On clean data, raw and per-fire give different PPL (Gutenberg: raw 8.44 vs per-fire 7.65) despite computing gradients with the same **direction** (uniform per-event weighting) — they differ only by a uniform N× magnitude factor. RMSprop's variance norm should absorb that uniform scaling in steady state.

Conjecture: with β2=0.999 the time constant is ~1000 steps; our 650-step training horizon plus warmup-cosine never reaches steady-state v, so uniform-scaling invariance doesn't engage. Not confirmed. The raw regime is preserved as a documented data point but not pursued further in this thread.

## Files

- `shakespeare/`, `gutenberg/` — full per-fire sweep, 3 seeds × {off, on}
- `exp1-ln-emb-rescaled/` — codex's diagnostic (per-chunk W kept, LN/emb scaled to match per-chunk semantics) — isolates class asymmetry contribution
- `raw-no-scale/` — `grad_scale = 1.0` everywhere; seed1 reran cold after contamination fix
- Each subdir contains `results.txt` and per-seed `train.log` files

## Two-commit story for this thread

- `ce48b0e` — Close the auto-save footgun. `--save` is now explicit; omitting it prints a warning and discards.
- `609e7ab` — Switch default gradient normalizer from per-chunk `1/T_q_chunk` to per-fire `1/N` via fire-end `cublasSscal`.

Both pushed to origin/main.
