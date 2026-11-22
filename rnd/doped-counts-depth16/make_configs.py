#!/usr/bin/env python3
"""Configs for rnd/doped-counts-depth16: refined-trie-counts' minibatch arms at depth 16.

The 1M PG-19 slice (A), d64/L2 from one random init, contexts of up to 16
characters, Adam 0.003 constant, 128 windows a step, 8 epochs (62,504 steps):
refined-trie-counts' recipe with seq_len 16 in place of 8. Each arm at train
seeds 42 and 43 (the window order; the init is init_seed 42 throughout).

  mb16-a1m           hard targets (the next character)
  mb16-a1m-tt        A's own counts as soft targets
  mb16-a1m-ref100m   each context's next-character distribution over the 100M text
  mb16-a1m-ref1g     ... over the 1G text (the first 1e9 characters of the 10G text)
  mb16-a1m-ref10g    ... over the 10G text (data/pg19/imm-split-10g, the 100M text's
                     continuation: tools/build_pg19_big.py)
  mb16-100m          minibatch on the 100M text itself, hard targets, windows every 100
                     characters from a random offset each epoch (1M windows an epoch), 25
                     epochs, checkpoint at 8 (A's steps) and 16

The ref arms read tables src/tools/doped_counts.py built (tables/a1m-s16-<mark>.npz,
not committed); a table arm's config is written only once its table exists.

Usage: make_configs.py
"""
import hashlib
from pathlib import Path

D = Path(__file__).resolve().parent
ROOT = D.parents[1]
A = "data/pg19/a1m/train_corpus.txt"
T100M = "data/pg19/imm-split-100m/train_corpus.txt"
HELD = "data/pg19/imm-split/heldout_corpus.txt"
VOCAB = "data/pg19/imm-split/vocab.txt"
TABLES = {"ref100m": 100_000_000, "ref1g": 1_000_000_000, "ref10g": 10_000_000_000}


def sha(path):
    h = hashlib.sha256()
    with open(ROOT / path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def config(slug, desc, seed, corpus=A, epochs=8, checkpoints=(1, 2, 4), exp=()):
    s = "" if seed == 42 else f"-s{seed}"
    return f"""---
description: "doped counts at depth 16: {desc}; minibatch (128 windows a step), d64/L2, seq_len 16, Adam 0.003 constant, train seed {seed}"
experiment: doped-counts-depth16
run_slug: {slug}{s}
corpus:
  path: {corpus}
  heldout: {HELD}
  vocab_source: {VOCAB}
model:
  d_model: 64
  n_layers: 2
  n_heads: 4
  d_ff: 256
  init_seed: 42
train:
  budget:
    unit: epochs
    value: {epochs}
  seed: {seed}
  quiet: true
  optimizer:
    name: adam
    lr: 0.003
    beta: 0.999
    momentum_beta: 0.9
  lr_schedule:
    name: constant
  seq_len: 16
  checkpoint_epochs: [{", ".join(str(c) for c in checkpoints)}]
eval:
  split: tail-heldout
  train_sanity: false
  batch_size: 1
  device: cpu
experimental:
  device: cuda
""" + "".join(f"  {line}\n" for line in exp)


def main():
    out = D / "configs"
    out.mkdir(exist_ok=True)
    cfgs = {}
    for seed in (42, 43):
        s = "" if seed == 42 else f"-s{seed}"
        cfgs[f"mb16-a1m{s}"] = config("mb16-a1m", "the 1M slice, hard targets", seed)
        cfgs[f"mb16-a1m-tt{s}"] = config("mb16-a1m-tt", "the 1M slice, its own counts as soft targets", seed,
                                         exp=["trie_targets: true"])
        for tag, mark in TABLES.items():
            table = f"rnd/doped-counts-depth16/tables/a1m-s16-{mark}.npz"
            if not (ROOT / table).exists():
                continue
            cfgs[f"mb16-a1m-{tag}{s}"] = config(
                f"mb16-a1m-{tag}", f"the 1M slice, each context's next-character distribution over the first "
                f"{mark:,} characters of the 10G text", seed,
                exp=["trie_targets: true", f"trie_targets_table: {table}", f"trie_targets_table_sha256: {sha(table)}"])
        cfgs[f"mb16-100m{s}"] = config("mb16-100m", "minibatch on the 100M text itself, hard targets, windows every "
                                       "100 characters (1M an epoch), 25 epochs", seed, corpus=T100M, epochs=25,
                                       checkpoints=(8, 16), exp=["window_stride: 100"])
    for name, text in cfgs.items():
        (out / f"{name}.yml").write_text(text)
    print(f"{len(cfgs)} configs")


if __name__ == "__main__":
    main()
