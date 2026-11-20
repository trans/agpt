#!/usr/bin/env python3
"""Configs for rnd/trie-growth: AGPT trained while the trie grows, against
full-trie AGPT and minibatch training at equal compute (Tiny Shakespeare,
one-pass-gap's setup: d64/L2, depth 16, the same seed weights and split).

Arms (slug: what):
  s1-eP     static full trie, pd=1, P passes (fp32 build; one-pass-gap's were bf16)
  s3-eP     static full trie, pd=3 + crown units, shuffled each pass (one-pass-gap's pd3cs)
  g1-x2-eE  growth, pd=1, the trie doubled each stage from 1,000 window starts, E passes a stage
  g3-x2-eE  growth, pd=3 + crown, shuffled, doubling, E passes a stage
  g0-xR     growth, pd=0 (one update on the whole trie a stage), the trie grown by
            a factor R each stage from 1,000 starts: the user's "N changes, then update"
  g1-full, g3-full   growth with one stage (the full trie): must reproduce s1-e1 / s3-e1

Doubling from 1,000 starts gives 11 stages and about 2E full-trie passes of
compute, so E = 1, 2, 4 sit near 1, 2, 4 minibatch epochs (16.95M positions an
epoch, 8.90M a full-trie pass). Growth runs count their optimizer steps before
training (experimental.growth_exact_schedule) so warmup-cosine ends on the last
step. Usage: make_configs.py  (writes configs/*.yml)
"""
from pathlib import Path

D = Path(__file__).resolve().parent
SPLIT = "data/.splits/4fa9aec1db6b3aea"
SEED = "rnd/cudax-d16-linear-mass-rerun/seeds/shake-d64L2-h4-dff256-d16-seed42.model"
TRIE = "data/.tries/64e9fe109211b366"


def config(slug, desc, pd, passes, lr, growth=None, crown=False):
    exp = ["  anc_grad_exact: true"]
    if crown:
        exp += ["  train_crown: true", "  unit_order_seed: 1"]
    g = ""
    if growth:
        start, ratio = growth
        g = "  growth:\n    divisions: 0\n    min_epochs: 1\n    epoch_ramp: fixed\n"
        exp += [f"  growth_start: {start}", f"  growth_ratio: {ratio}", "  growth_exact_schedule: true"]
    return f"""---
description: "trie growth: {desc}, Adam lr {lr} warmup-cosine per step"
experiment: trie-growth
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
{g}eval:
  split: tail-heldout
  train_sanity: false
  batch_size: 1
  device: cpu
experimental:
""" + "\n".join(exp) + "\n"


def main():
    out = D / "configs"
    out.mkdir(exist_ok=True)
    cfgs = {}

    def add(slug, *a, **k):
        cfgs[f"{slug}-lr{a[3]}"] = config(slug, *a, **k)

    # equivalence: one growth stage = the full trie
    add("s1-e1", "static pd=1, 1 pass", 1, 1, 0.012)
    add("g1-full-e1", "growth pd=1, one stage (the full trie), 1 pass", 1, 1, 0.012, growth=(2_000_000, 2))
    add("s3-e1", "static pd=3 + crown, shuffled, 1 pass", 3, 1, 0.0015, crown=True)
    add("g3-full-e1", "growth pd=3 + crown, shuffled, one stage (the full trie), 1 pass", 3, 1, 0.0015,
        growth=(2_000_000, 2), crown=True)
    for e in (1, 2, 4):
        for lr in (0.006, 0.012, 0.024):
            add(f"g1-x2-e{e}", f"growth pd=1, doubling from 1,000 starts, {e} pass(es) a stage", 1, e, lr,
                growth=(1000, 2))
    for p in (2, 4, 8):
        for lr in (0.006, 0.012):
            add(f"s1-e{p}", f"static pd=1, {p} passes", 1, p, lr)
    for lr in (0.012, 0.024, 0.048):
        add("g0-x1.1", "growth pd=0, one update a stage, trie x1.1 a stage from 1,000 starts", 0, 1, lr,
            growth=(1000, 1.1))
    for e in (1, 2):
        for lr in (0.00075, 0.0015, 0.003):
            add(f"g3-x2-e{e}", f"growth pd=3 + crown, shuffled, doubling from 1,000 starts, {e} pass(es) a stage",
                3, e, lr, growth=(1000, 2), crown=True)
    for p in (2, 4):
        for lr in (0.00075, 0.0015):
            add(f"s3-e{p}", f"static pd=3 + crown, shuffled, {p} passes", 3, p, lr, crown=True)
    for lr in (0.006, 0.012):
        add("g0-x1.01", "growth pd=0, one update a stage, trie x1.01 a stage from 1,000 starts", 0, 1, lr,
            growth=(1000, 1.01))
    order = []
    for name, text in cfgs.items():
        (out / f"{name}.yml").write_text(text)
        order.append(name)
    (D / "order.txt").write_text("\n".join(order) + "\n")
    print(f"{len(cfgs)} configs -> {out}")


if __name__ == "__main__":
    main()
