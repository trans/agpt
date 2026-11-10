# Measurements worth keeping

*Updated 2026-11-10*

## Common setup and how to reproduce

Unless a row says otherwise:

- **Data:** Tiny Shakespeare on the canonical carved split `4fa9aec1db6b3aea`
  (10 chunks, 5%, seed 42).
- **Model and trie:** d64 L2 h4 ff256, depth-16 trie, seq_len 16, pd=1,
  Adam at lr 0.0015 with warmup-cosine.
- **Metrics:** "rolling" is canonical rolling byte PPL. "Fixed" is
  fixed-window PPL.

To reproduce a run:

- **Build:** `just build-agpt-train-v2 build-agpt-experiment`
- **Run:** `bin/agpt_experiment --config rnd/gradient-population/configs/<name>.yml`.
  This writes a new run directory.
- **On a fresh clone:** `data/.splits/` and `data/.tries/` are gitignored.
  Re-carve the split with `bin/agpt_carve` (recipe in
  `rnd/kn-shakespeare-baseline`), and drop `trie.path` so that the
  orchestrator builds the trie.

## Canonical runs

| What | Rolling | Fixed | Config or record | Notes |
|---|---:|---:|---|---|
| Adam pd=1, exact gradient, 625 ep | **4.764** | **4.208** | `adam-pd1-exact-625ep` | best d64/L2 on record |
| Adam pd=1, exact, 100 ep | 5.225 | 4.684 | `adam-pd1-exact-100ep` | 654 s train |
| Adam pd=1, truncated, 100 ep | 5.348 | 4.810 | `cadence-pd1-100ep` | 570 s train |
| Adam pd=0, 100 ep, exact / truncated | 12.084 / 12.054 | 12.105 / 12.110 | `adam-pd0-exact-100ep`, `cadence-pd0-100ep` | the fix doesn't matter at pd=0 |
| L-BFGS on the trie, exact, 25 Adam ep + 600 passes | 5.019 | 4.481 | `lbfgs-trie-exact-pd1-ep25-600` | 3 backtracks; still descending |
| Same, truncated gradient | 5.407 | 4.888 | `lbfgs-trie-pd1-ep25-600` | 264 of 600 evaluations backtracked |
| d128/L6, phase-conditioned RoPE, 512 ep | 4.497 | 3.92 | `rnd/sampled-node-phase-rope` | 2026-06-05: before the race fix, truncated gradient; the RoPE shift is a no-op, so this is strided-minibatch training (2026-11-10) |
| Superposed tail, d64/L2 no trie, 20k steps: none / fourier / wavelet / decay (seed 42) | diag. NLL 1.3583 / 1.3598 / 1.3609 / 1.3468 | — | `rnd/superposed-tail-rope` `20261110T182838-sweep20k` | diagnostic full-window NLL, not lm-eval; seed 43: 1.3521 / 1.3526 / 1.3559 / 1.3545 |
| Same baseline, softmax window 16 / 32 / 64, t ≥ 256 (seeds 42, 43) | diag. NLL 1.3578, 1.3523 / 1.3467, 1.3511 / 1.3574, 1.3504 | — | `20261110T190338-window-control20k` | undertrained (~20M targets): NOT evidence about context length; the control failed |
| d64/L2 phase-weighted 8 ep, phase + depth / depth-only RoPE | 10.170 / 10.175 | 10.132 / 10.137 | `rnd/sampled-node-phase-rope` `20261110T1146*` / `T1148*` | 2026-11-10 identity control; post-fix trainer, `anc_grad_exact: false` |

At equal traversals (125 / 225 / 325 / 425 / 625), Adam-625 leads L-BFGS by
about 5–6% at every point (Exp 8 table).

## Baselines on the same split

These need care (see [open-lines.md](open-lines.md) §5):

- **KenLM KN, order 7:**
  - 4.114 in `rnd/kn-shakespeare-baseline` (2026-05-28);
  - 4.195 in `rnd/kn-4fa9aec/kn_sweep.txt` on `exp/linear-recurrence-final`
    (2026-06-10).

  Both records describe the same 10-chunk, seed-42 split. **Unresolved.**
- **Vanilla attention, no trie, same architecture:** 4.45 at seq 16, 4.17 at
  seq 32. Both come from session memory and have no `result.json`.

## Trainer cost

Measured with `bin/agpt_train_v2` at pd=1, d64/L2, depth 16, on the
development GPU:

- about 5.6 s per epoch;
- the exact second pass adds about 1.0 s (+18%). It covers the 983,047
  internal-node query positions.

The profile breakdown is in [open-lines.md](open-lines.md) §3.

## Gradient checks (Exp 8, `agpt_grad_check.py`)

These were run at the L-BFGS-600 checkpoint with the fp32 cache and TF32 off.

- **FD ratio at ε 0.001:** 1.0003 along the exact gradient, and 1.0000 along
  (exact − truncated).
- **The old truncated gradient there:** 35° off the true gradient and 21% too
  short.

## Diagnostics (not canonical)

- **Coherence of the frozen pd=2 fan-out (Exp 1).** Event-weighted global ρ by
  checkpoint:

  | Seed | ep10 | ep50 | ep100 |
  |---:|---:|---:|---:|
  | 0.719 | 0.590 | 0.394 | 0.130 |

  Per-root, it goes from 0.851 to 0.351.
  - To reproduce: `AGPT_GRAD_DUMP_DIR=<dir> bin/agpt_train_v2 --config
    rnd/gradient-population/configs/pd2-ep100.yml`, then
    `python3 src/tools/agpt_grad_population.py <dir>`.
  - Each dump takes about 15 s and is about 600 MB; dumps are gitignored.
- **PyTorch L-BFGS on a fixed 65k-window objective (Exp 6).** Measured as
  held-out NLL in nats on 8192 fixed-window targets:
  - from the ep25 checkpoint, it reaches 1.588 in 240 passes; plain gradient
    descent needs 10,000 (42×);
  - after 600 passes it is at 1.5228;
  - tool: `src/tools/agpt_lbfgs_test.py`.
- **Context-tree orientation** (GRU d64, depth 8, diagnostic held-out PPL):

  | Run | Prefix trie | Suffix tree | Suffix + residual readout |
  |---|---:|---:|---:|
  | pd=0, 2000 full-batch steps | 4.750 | 5.094 | 4.956 |
  | pd=1, 500 epochs | 5.093 | 6.286 | 9.222 |

  - Run with `python3 src/tools/agpt_context_tree_coherence.py --out DIR
    --depth 8 --steps 2000`. For the pd=1 run, add `--pd 1 --steps 500`
    (`--steps` counts epochs).
  - Each model takes about 8 minutes.
  - The run directories under `rnd/context-tree-orientation/` hold the
    summaries.
- **Unit grouping without the tree** (2026-11-10, same GRU and table, 65
  steps per epoch, diagnostic held-out NLL):

  | Grouping | Target skew (KL, nats) | Constant 3e-3, 500 epochs | Cosine, 100 epochs |
  |---|---:|---:|---:|
  | prefix-trie units (oldest char) | 0.090 | 1.628 / 1.632 (2 seeds) | 1.704 |
  | suffix-tree units (newest char) | 0.861 | 1.759 | 1.880 |
  | shuffled minibatches | 0.0002 | unstable after epoch 100 (1.694 → 1.738 at 300) | 1.714 |
  | newest-first GRU, prefix units | | 1.681 | |
  | newest-first GRU, suffix units (= October pd=1) | | 1.838 | |

  - Tool: `src/tools/agpt_unit_grouping_control.py --out DIR --cells
    fwd/first,fwd/last,fwd/window --epochs 500` (add `--schedule cosine
    --warmup-epochs 2 --epochs 100` for the cosine column).
  - Speed: about 2–5 s per epoch for node groupings and 8–14 s for window
    groupings, on a GPU shared with other jobs.
