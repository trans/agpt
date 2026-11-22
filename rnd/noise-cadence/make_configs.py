#!/usr/bin/env python3
"""Configs for rnd/noise-cadence: trie growth with window starts in random
order, and the split-half noise cadence (Tiny Shakespeare, trie-growth's and
one-pass-gap's setup: d64/L2, depth 16, the same seed weights and split).

Arms (slug: what):
  r1-full-e1  random-order growth, pd=1, one stage (the full trie): the same trie
              as static s1-e1 (trie-growth 15.72), node numbering aside
  r1-x2-eE    random-order growth, pd=1, doubling from 1,000 starts, E passes a stage
              (trie-growth corpus order: 9.51 / 8.15 at E = 1 / 2)
  r3-x2-eE    the same at pd=3 + crown, shuffled (corpus order: 5.96 / 5.51)
  r0-xR       random-order growth, pd=0, one update a stage, the trie x R a stage
              from 1,000 starts (corpus order: 12.74 at x1.1, 7.02 at x1.01)
  n0-tT       the noise cadence (pd=0, random order): step when the split-half
              noise is at most T x the gradient's signal, else add windows
              (gradient of the new windows only, at the same weights); from 32
              windows; the LR schedule runs over a budget of 54.2M trie positions,
              the doubling E=2 runs' compute (3.2 minibatch epochs' worth); the sample
              is cumulative (windows stay in the tries for the rest of the run)
  f0-tT       the same with a fresh sample after every step (growth_noise_resample:
              the size the test last passed with, the next windows in the shuffled
              order, reshuffled each pass through the text); the user (2026-11-22)
              asked for it after the cumulative sample overfit its ~10k windows
  f0-tT-bM    the same at a budget of minibatch's 4 / 8 / 16 epochs (67.8M / 135.6M / 271.2M positions)
  f0-...-s43  the same with train seed 43 (a different window order; the same init)
  fc-tT-bM    the fresh-sample cadence at a constant LR: the noise test alone anneals
              (it grows the sample as the signal shrinks; Smith et al. 2018)
  m0-wW-eE    v2 at a fixed sample: W fresh windows every step (growth_start = growth_noise_max_windows
              = W), E minibatch epochs' positions: plain minibatch training through tries, the control
              for the trainer (v2 against PyTorch's mb-eE)
  f0-...-capW the fresh-sample cadence with the sample capped at W windows (the late samples grew to
              30k-250k)
  mb-eE       ordinary minibatch Adam, 128 windows a step (one-pass-gap's recipe), for
              the 16-epoch bar; run with src/tools/agpt_minibatch_train.py

Usage: make_configs.py  (writes configs/*.yml)
"""
from pathlib import Path

D = Path(__file__).resolve().parent
SPLIT = "data/.splits/4fa9aec1db6b3aea"
SEED = "rnd/cudax-d16-linear-mass-rerun/seeds/shake-d64L2-h4-dff256-d16-seed42.model"
TRIE = "data/.tries/64e9fe109211b366"
BUDGET = 54_200_000


def config(slug, desc, pd, passes, lr, exp_extra, crown=False, schedule="warmup-cosine", seed=42):
    exp = ["  anc_grad_exact: true"]
    if crown:
        exp += ["  train_crown: true", "  unit_order_seed: 1"]
    exp += exp_extra
    return f"""---
description: "noise cadence: {desc}, Adam lr {lr} {schedule}"
experiment: noise-cadence
run_slug: {slug}-lr{lr}
corpus:
  path: {SPLIT}/train_corpus.txt
  heldout: {SPLIT}/heldout_corpus.txt
  vocab_source: data/input.txt
trie:
  prune_min_mass: 1
  prune_min_depth: 0
  path: {TRIE}
  virtual_tree: false
model:
  init_file: {SEED}
train:
  budget:
    unit: epochs
    value: {passes}
  seed: {seed}
  quiet: true
  optimizer:
    name: adam
    lr: {lr}
    beta: 0.999
    momentum_beta: 0.9
  lr_schedule:
    name: {schedule}
    warmup_epochs: 0
  max_depth: 16
  partition_depth: {pd}
  chunk_queries: 50000
  anc_grad: true
  seq_len: 16
  growth:
    divisions: 0
    min_epochs: 1
    epoch_ramp: fixed
eval:
  split: tail-heldout
  train_sanity: false
  batch_size: 1
  device: cpu
experimental:
""" + "\n".join(exp) + "\n"


def growth(start, ratio):
    return [f"  growth_start: {start}", f"  growth_ratio: {ratio}", "  growth_exact_schedule: true",
            "  growth_random_order: true"]


def noise(theta, resample=False, budget=BUDGET, start=32, cap=0):
    out = [f"  growth_start: {start}", f"  growth_noise_theta: {theta}",
           f"  growth_noise_budget_positions: {budget}", "  growth_noise_max_ratio: 2",
           "  growth_random_order: true"]
    if resample:
        out.append("  growth_noise_resample: true")
    if cap:
        out.append(f"  growth_noise_max_windows: {cap}")
    return out


def minibatch(epochs, lr, seed=42):
    return f"""---
description: "noise cadence bar: ordinary minibatch Adam (128 windows of 16 per step), {epochs} pass(es), lr {lr} warmup-cosine per step (one-pass-gap's recipe)"
experiment: noise-cadence
run_slug: mb-e{epochs}{"" if seed == 42 else f"-s{seed}"}-lr{lr}
corpus:
  path: {SPLIT}/train_corpus.txt
  heldout: {SPLIT}/heldout_corpus.txt
  vocab_source: data/input.txt
model:
  init_file: {SEED}
train:
  budget:
    unit: epochs
    value: {epochs}
  seed: {seed}
  quiet: true
  optimizer:
    name: adam
    lr: {lr}
    beta: 0.999
    momentum_beta: 0.9
  lr_schedule:
    name: warmup-cosine
    warmup_epochs: 0
  seq_len: 16
eval:
  split: tail-heldout
  train_sanity: false
  batch_size: 1
  device: cpu
experimental:
  minibatch_size: 128
  device: cuda
"""


def main():
    out = D / "configs"
    out.mkdir(exist_ok=True)
    cfgs = {}

    def add(slug, desc, pd, passes, lr, extra, crown=False, **kw):
        cfgs[f"{slug}-lr{lr}"] = config(slug, desc, pd, passes, lr, extra, crown, **kw)

    add("r1-full-e1", "random-order growth pd=1, one stage (the full trie), 1 pass", 1, 1, 0.012,
        growth(2_000_000, 2))
    for theta, lrs in ((1, (0.003, 0.006, 0.012, 0.024)), (2, (0.003, 0.006, 0.012, 0.024)),
                       (4, (0.003, 0.006, 0.012, 0.024)), (8, (0.003, 0.006, 0.012, 0.024)), (16, (0.006,))):
        for lr in lrs:
            add(f"n0-t{theta}", f"noise cadence pd=0, theta {theta}, from 32 windows, budget {BUDGET:,} positions",
                0, 1, lr, noise(theta))
    for theta, lrs in ((1, (0.006, 0.012, 0.024)), (2, (0.003, 0.006, 0.012, 0.024)),
                       (4, (0.003, 0.006, 0.012, 0.024)), (8, (0.006, 0.012, 0.024))):
        for lr in lrs:
            add(f"f0-t{theta}", f"noise cadence pd=0, a fresh sample each step, theta {theta}, from 32 windows, "
                f"budget {BUDGET:,} positions", 0, 1, lr, noise(theta, resample=True))
    # minibatch's 4 and 8 epochs' compute (one-pass-gap: 4.58 / 4.53)
    for budget, tag in ((67_800_000, "b68"), (135_600_000, "b136")):
        for theta, lrs in ((2, (0.006, 0.012)), (4, (0.006, 0.012))):
            for lr in lrs:
                add(f"f0-t{theta}-{tag}", f"noise cadence pd=0, a fresh sample each step, theta {theta}, from 32 "
                    f"windows, budget {budget:,} positions", 0, 1, lr, noise(theta, resample=True, budget=budget))
    # follow-up (2026-11-22): does the gap to minibatch cross zero?
    fresh = lambda t, b: f"noise cadence pd=0, a fresh sample each step, theta {t}, from 32 windows, budget {b:,} positions"
    for theta in (4, 8):
        add(f"f0-t{theta}-b271", fresh(theta, 271_200_000), 0, 1, 0.006,
            noise(theta, resample=True, budget=271_200_000))
    add("f0-t8-b136", fresh(8, 135_600_000), 0, 1, 0.006, noise(8, resample=True, budget=135_600_000))
    add("f0-t4-b136-s43", fresh(4, 135_600_000) + ", seed 43", 0, 1, 0.006,
        noise(4, resample=True, budget=135_600_000), seed=43)
    for budget, tag in ((54_200_000, "b54"), (135_600_000, "b136")):
        for lr in (0.003, 0.006):
            add(f"fc-t4-{tag}", fresh(4, budget) + ", constant LR", 0, 1, lr,
                noise(4, resample=True, budget=budget), schedule="constant")
    for lr in (0.003, 0.006):
        cfgs[f"mb-e16-lr{lr}"] = minibatch(16, lr)
    # free rein (2026-11-22): the trainer control and a cap on the late samples
    MB_EPOCH = 16_954_128   # 1,059,633 windows x 16
    for e in (1, 2, 4, 8):
        add(f"m0-w128-e{e}", f"v2 at a fixed sample of 128 fresh windows a step (minibatch through tries), "
            f"budget {e * MB_EPOCH:,} positions ({e} minibatch epochs)", 0, 1, 0.006,
            noise(1, resample=True, budget=e * MB_EPOCH, start=128, cap=128))
    # seeds: is v2's 1-epoch gap to PyTorch (4.870 vs 4.826) the trainer or the window order?
    for seed in (43, 44):
        add(f"m0-w128-e1-s{seed}", f"v2 at a fixed sample of 128 fresh windows a step (minibatch through tries), "
            f"budget {MB_EPOCH:,} positions (1 minibatch epoch), seed {seed}", 0, 1, 0.006,
            noise(1, resample=True, budget=MB_EPOCH, start=128, cap=128), seed=seed)
        cfgs[f"mb-e1-s{seed}-lr0.006"] = minibatch(1, 0.006, seed=seed)
    # seeds at 8 and 16 epochs: minibatch's seed spread at 1 epoch is 0.08 (seed 42 its best)
    for seed in (43, 44):
        cfgs[f"mb-e8-s{seed}-lr0.006"] = minibatch(8, 0.006, seed=seed)
    cfgs["mb-e16-s43-lr0.006"] = minibatch(16, 0.006, seed=43)
    add("f0-t4-b271-s43", fresh(4, 271_200_000) + ", seed 43", 0, 1, 0.006,
        noise(4, resample=True, budget=271_200_000), seed=43)
    for cap in (2048, 8192):
        add(f"f0-t4-b136-cap{cap}", fresh(4, 135_600_000) + f", sample capped at {cap} windows", 0, 1, 0.006,
            noise(4, resample=True, budget=135_600_000, cap=cap))
    for e, lrs in ((1, (0.006, 0.012)), (2, (0.006,))):
        for lr in lrs:
            add(f"r1-x2-e{e}", f"random-order growth pd=1, doubling from 1,000 starts, {e} pass(es) a stage",
                1, e, lr, growth(1000, 2))
    for e, lrs in ((1, (0.0015, 0.003)), (2, (0.0015,))):
        for lr in lrs:
            add(f"r3-x2-e{e}", f"random-order growth pd=3 + crown, shuffled, doubling from 1,000 starts, "
                f"{e} pass(es) a stage", 3, e, lr, growth(1000, 2), crown=True)
    for lr in (0.006, 0.012):
        add("r0-x1.1", "random-order growth pd=0, one update a stage, trie x1.1 a stage from 1,000 starts",
            0, 1, lr, growth(1000, 1.1))
    add("r0-x1.01", "random-order growth pd=0, one update a stage, trie x1.01 a stage from 1,000 starts",
        0, 1, 0.012, growth(1000, 1.01))
    for name, text in cfgs.items():
        (out / f"{name}.yml").write_text(text)
    print(f"{len(cfgs)} configs")


if __name__ == "__main__":
    main()
