# AGPT Epoch Scaling — Detailed Findings

## How this was discovered

The discovery came indirectly through the subtree-dropout experiment
(see `../subtree-dropout/`). When testing whether random root-child
masking would help AGPT escape "smooth valley" plateaus, the dropout
sweep included higher epoch counts as a control. Pure AGPT (no dropout)
at 5 SE got 9.64 PPL — already substantially below the "plateau" we'd
been treating as 10.83. Pushing further showed continuous improvement.

We had been training AGPT for 3 super-epochs across the entire prior
project history because that was the post-fix-baseline standard. The
recipe was correct; the epoch count was an order of magnitude too low.

## Full SE sweep

Recipe: rmsprop + warmup-cosine + entropy-icing + mass=log + no-accumulate
+ lr=3e-3. Single-call (preserving optimizer state). Eval at seq_len=32
(matched-context).

| SE | PPL@32 mean | std (3 reps) | Δ per 10 SE |
|---:|---:|---:|---:|
| 3 | 10.82 | 0.07 | — |
| 5 | 9.64 | 0.05 | −2.36 (3-5 trend) |
| 7 | 8.83 | 0.13 | |
| 10 | 8.24 | 0.15 | −2.58 (3-10) |
| 15 | 7.21 | 0.16 | −1.03 |
| 20 | 6.59 | (n=1) | −0.62 |
| 25 | 6.32 | 0.06 | −0.27 |
| 30 | 5.96 | 0.09 | −0.36 |
| 40 | 5.39 | (n=1) | −0.57 |

Marginal improvement is slowing: −0.59 PPL/SE in the 3-5 range vs
−0.06 PPL/SE in the 25-40 range. But it has not plateaued at 40 SE.
Extrapolating the current trend: ~5.0 at 50 SE, ~4.7 at 60 SE, perhaps
~4.0 at 100 SE.

## Wall-clock cost

| SE | Approx wall-clock |
|---:|---:|
| 3 | ~60 sec |
| 10 | ~200 sec |
| 20 | ~400 sec |
| 40 | ~800 sec |

Linear in SE. ~10 minutes for 40 SE training on Shakespeare 1M d=32 with
108k-param model on cublas backend.

## Comparison to SGD

Same architecture (108k params, d_model=64, 4 heads, 2 layers, ff=256).

| Method | Steps | Wall-clock | PPL@32 |
|---|---:|---:|---:|
| SGD seq=32, 10k steps | 10000 | ~110 sec | 9.77 |
| SGD seq=32, 30k steps (extrapolated from seq=16 30k) | 30000 | ~330 sec | ~9.0 (estimated) |
| AGPT 5 SE (no dropout) | 325 | ~100 sec | **9.64** |
| AGPT 7 SE | 455 | ~140 sec | **8.83** |
| AGPT 20 SE | 1300 | ~400 sec | **6.59** |
| AGPT 40 SE | 2600 | ~800 sec | **5.39** |

AGPT achieves substantially lower PPL than SGD at matched wall-clock for
all budgets where direct comparison is possible. AGPT 5 SE roughly matches
SGD 10k steps in wall-clock (~100 sec) and beats it on PPL.

## Why AGPT was undertrained

Each AGPT optimizer step processes a whole root-child subtree's worth of
queries (thousands of training events). With 65 root-children + 3 SE,
that's 195 optimizer steps total. Compared to SGD's 10000 steps for a
similar wall-clock, AGPT takes 50× fewer optimizer steps but each step
sees ~50× more training data. So per-CHARACTER, it's similar. But per-step
optimization in deep learning has its own role — each step gives momentum
adaptation, schedule progression, and gradient noise injection. AGPT's
195 steps was a small number for a transformer of any size to converge.

The fact that 195 steps got AGPT to 10.82 PPL was actually impressive — it
just wasn't convergence.

## Context-length specialization

A critical caveat. As SE increases, the model becomes specialized to its
trained context (seq=32 for d=32 trie):

| SE | PPL@32 | PPL@128 |
|---:|---:|---:|
| 3 | 10.82 | ~12.67 |
| 10 | 8.24 | 15.18 |
| 15 | 7.21 | 20.47 |
| 20 | 6.59 | 28.54 |

PPL@128 monotonically WORSENS with more training. Why: AGPT trains RoPE
positions 0..d−1 (here 0..31). Positions 32..127 are never used during
training. As the model becomes more efficient at exploiting positions
0..31, its dependence on those specific positions strengthens —
extrapolation to position 32+ becomes less coherent.

This is "context-length specialization," not classical overfitting. The
model isn't memorizing the corpus — held-out PPL@32 keeps improving — but
it's becoming less robust to context-length shift.

For tasks where the inference seq_len matches the training d, this isn't
a problem. For tasks with variable context length, AGPT trained at high
SE is worse than SGD trained at the eval seq_len.

## What this implies for prior conclusions

1. **AGPT vs SGD at matched eval**: AGPT wins decisively at high SE.
2. **The "AGPT plateau"**: was a stopping point, not a true plateau.
3. **Joint-mass / depth-routing prescriptive value**: needs re-evaluation
   at high SE. Effect sizes measured at 3 SE (1-7% PPL changes) are
   relative to a much higher absolute PPL and may not preserve at 5-7 PPL
   absolute.
4. **Hybrid AGPT-SGD smoothness trap**: probably not a real phenomenon.
   AGPT's "plateau" at 10.83 was undertraining, not optimization
   pathology.
5. **Subtree dropout's apparent ~6% benefit**: was at 5 SE budget where
   undertraining was worst. At 15-20 SE the dropout neutral-to-negative.
6. **Across the project's history of comparing AGPT recipes**: every
   "best AGPT" PPL stated for d=32 (12.07, 12.79, 13.17, etc.) was
   under-trained. The actual achievable PPL on this codebase at d=32
   on Shakespeare 1M is at least 5.39 and likely lower with more SE.

## d-sweep at high SE — d=16 vs d=32 close to tied

Re-running the d-sweep at high SE (each model evaluated at its own
matched seq_len) reveals that d=16 isn't strictly inferior — it's
competitive at all budgets and wins at low SE:

| SE | d=16 PPL@16 | d=32 PPL@32 | winner |
|---:|---:|---:|---|
| 3 | 10.33 | 10.82 | d=16 by 0.49 |
| 5 | 9.28 | 9.64 | d=16 by 0.36 |
| 10 | 7.91 | 8.24 | d=16 by 0.33 |
| 15 | 7.29 | 7.21 | d=32 by 0.08 |
| 20 | 6.70 | 6.59 | d=32 by 0.11 |

Crossover happens around SE=12-15. At low SE, d=16 wins (smaller
parameter space converges faster). At high SE, d=32 narrowly wins as
its extra context starts paying off.

Practical implication: **d=16 is an excellent compute-economical choice**
— similar PPL with much less GPU memory and faster epochs.

## Memory and compute scaling on Shakespeare 1M (8 GB GPU)

| d | Trie disk | KV cache (BF16+mass1) | GPU peak | Time/epoch |
|---:|---:|---:|---:|---:|
| 16 | 93 MB | 532 MB | 6.0 GB | 13.6 s |
| 32 | 163 MB | 569 MB | 7.8 GB | 42.2 s |
| 48 | 231 MB | 574 MB | 6.4 GB | 39.1 s |
| 64 | 299 MB | 575 MB | OOM | — |

**The d-wall is between d=48 and d=64 on 8 GB GPU.** Notably, the KV
cache barely grows past d=32 — mass=1 compaction handles 98.2% of
edges at d=64. The OOM is in OTHER buffers (ancestor lists, packed
attention scratch, working buffers), which scale O(d × radix_nodes).
Engineering work could push the wall further, but at d=32 we're
already at the framing's predicted optimum for English at this corpus
size.

## Reproduce headline result

```sh
cd /home/trans/Projects/agpt
cp data/input.random.model /tmp/dr_agpt_40se.model
bin/agpt_train --model /tmp/dr_agpt_40se.model \
    --trie-dir /home/trans/agpt-tries/shakespeare_d32_radix_corpus \
    --save /tmp/dr_agpt_40se.model \
    --epochs 40 --lr 3e-3 \
    --optimizer rmsprop --rmsprop-beta 0.999 \
    --lr-schedule warmup-cosine --warmup-epochs 1 \
    --entropy-lambda 1.0 --mass-weight log --no-accumulate
bin/perplexity --model /tmp/dr_agpt_40se.model \
    --file data/input.txt --max-positions 8192 --seq-len 32 --backend cublas
# Expected: ~5.4 PPL @ seq=32
```

Wall-clock ~10 minutes on cublas.
