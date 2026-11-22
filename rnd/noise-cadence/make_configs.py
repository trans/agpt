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

Usage: make_configs.py  (writes configs/*.yml)
"""
from pathlib import Path

D = Path(__file__).resolve().parent
SPLIT = "data/.splits/4fa9aec1db6b3aea"
SEED = "rnd/cudax-d16-linear-mass-rerun/seeds/shake-d64L2-h4-dff256-d16-seed42.model"
TRIE = "data/.tries/64e9fe109211b366"
BUDGET = 54_200_000


def config(slug, desc, pd, passes, lr, exp_extra, crown=False):
    exp = ["  anc_grad_exact: true"]
    if crown:
        exp += ["  train_crown: true", "  unit_order_seed: 1"]
    exp += exp_extra
    return f"""---
description: "noise cadence: {desc}, Adam lr {lr} warmup-cosine"
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
  seed: 42
  quiet: true
  optimizer:
    name: adam
    lr: {lr}
    beta: 0.999
    momentum_beta: 0.9
  lr_schedule:
    name: warmup-cosine
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


def noise(theta, resample=False):
    out = ["  growth_start: 32", f"  growth_noise_theta: {theta}",
           f"  growth_noise_budget_positions: {BUDGET}", "  growth_noise_max_ratio: 2",
           "  growth_random_order: true"]
    if resample:
        out.append("  growth_noise_resample: true")
    return out


def main():
    out = D / "configs"
    out.mkdir(exist_ok=True)
    cfgs = {}

    def add(slug, desc, pd, passes, lr, extra, crown=False):
        cfgs[f"{slug}-lr{lr}"] = config(slug, desc, pd, passes, lr, extra, crown)

    add("r1-full-e1", "random-order growth pd=1, one stage (the full trie), 1 pass", 1, 1, 0.012,
        growth(2_000_000, 2))
    for theta, lrs in ((1, (0.003, 0.006, 0.012, 0.024)), (2, (0.003, 0.006, 0.012, 0.024)),
                       (4, (0.003, 0.006, 0.012, 0.024)), (8, (0.003, 0.006, 0.012, 0.024)), (16, (0.006,))):
        for lr in lrs:
            add(f"n0-t{theta}", f"noise cadence pd=0, theta {theta}, from 32 windows, budget {BUDGET:,} positions",
                0, 1, lr, noise(theta))
    for theta in (1, 2, 4, 8):
        for lr in (0.012, 0.024):
            add(f"f0-t{theta}", f"noise cadence pd=0, a fresh sample each step, theta {theta}, from 32 windows, "
                f"budget {BUDGET:,} positions", 0, 1, lr, noise(theta, resample=True))
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
