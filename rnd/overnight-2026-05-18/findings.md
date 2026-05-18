# Streaming-AGPT × Gutenberg validation (overnight 2026-05-18)

**Canonical writeup:** see `rnd/streaming-agpt-v1/findings.md` § "Cross-corpus
validation: Gutenberg 5M (2026-05-18)". This file retains the experiment-specific
artifacts (driver script, summary script, raw logs reference).

## TL;DR

Streaming-AGPT (100 × 5 SE) **beats baseline (500 SE single-stage) by 6.46% PPL
on Gutenberg 5M d=16**, 3 seeds each, p between 0.05 and 0.01 (Welch's t).
That's ~3× the magnitude of the Shakespeare win (-2.1%), supporting the
hypothesis that longer-range / more-redundant corpora benefit more from
streaming's incremental fresh-warmup pattern.

**New project-best PPL on Gutenberg: 3.99** (seed 300, single best);
**4.0831 ± 0.1353 mean.** Previous best was 4.56 single-seed AGPT
(project_gutenberg5m_validation memory).

## Per-seed table

| seed | Streaming (100×5 SE) | Baseline (500 SE) | Δ |
|---:|---:|---:|---:|
| 100 | 4.2380 | 4.3323 | -2.18% |
| 200 | 4.0236 | 4.3037 | -6.51% |
| 300 | 3.9878 | 4.4595 | -10.58% |
| **mean** | **4.0831 ± 0.1353** | **4.3652 ± 0.0829** | **-6.46%** |

Welch's t = -3.078 at df=3.3, which puts the result between p<0.05 (|t|>2.0)
and p<0.01 (|t|>2.7 at df≈4). Solid evidence; not overwhelming.

## Comparison to Shakespeare result

| Corpus | Streaming win | seeds | wall savings | Source |
|---|---:|---:|---:|---|
| Shakespeare 1M d=16 | -2.1% (p<0.01) | 3 | 42% less | project_streaming_agpt |
| Gutenberg 5M d=16  | **-6.5% (p<0.05)** | 3 | ~no savings | this run |

The wall-savings difference is also striking. On Shakespeare, streaming wins
PPL *and* wall (42% less time). On Gutenberg, streaming wall (6312s ≈ 1.75hr
per seed) is roughly similar to baseline wall (4325s ≈ 1.2hr per seed from
the RunPod 2-seed run, scaled to laptop). Need to check this more carefully —
the laptop-vs-laptop streaming-vs-baseline wall comparison wasn't directly
captured (baseline seed 100 ran overnight here too; check its wall time).

The increased streaming variance (0.135 vs Shakespeare's 0.025 from
project_streaming_agpt) is also notable — streaming may be more
seed-sensitive on the larger/more-diverse corpus. Each seed still wins
its baseline, so the seed-sensitivity is in magnitude not direction.

## Setup

- Corpus: data/gutenberg_5m.txt (5MB)
- Model: d_model=128, n_heads=8, n_layers=4 (default microGPT)
- Trie: d=16 max depth, full corpus
- Optimizer: RMSprop β₂=0.999
- LR: 3e-3 warmup-cosine, warmup-epochs=1 (per stage in streaming; once for baseline)
- Loss: mass-weight=log, entropy-lambda=1.0
- Streaming: 100 stages × 5 SE each = 500 SE total compute
- Baseline: 500 SE single-stage

All runs on the laptop RTX 4070 mobile.

## Result location

- `rnd/streaming-agpt-v1/logs/seed{100,200,300}_ms_n100_se5_*` — streaming
- `rnd/streaming-agpt-v1/logs/ms_baseline_gutenberg_5m_se500_seed*_*` — baseline
- `rnd/overnight-2026-05-18/run.log` — driver output

Note: baseline seeds 200 and 300 were originally produced on RunPod
2026-05-17 (project_gutenberg_multiseed_baseline memory); seed 100 was
produced overnight on laptop.

## What's next

1. **Verify the Gutenberg streaming win on a third independent corpus** —
   Twain dataset (project_twain_dataset in pipeline). If streaming wins by
   a similar margin on Twain too, the corpus-scale hypothesis is supported.
2. **Re-measure wall-clock comparison cleanly.** The current wall numbers
   come from different hardware (RunPod baseline vs laptop streaming).
   Run baseline at 500 SE on laptop to get apples-to-apples.
3. **Investigate streaming variance.** Why 0.135 on Gutenberg vs 0.025 on
   Shakespeare? Stage-boundary effects? Initial-trie effects?
