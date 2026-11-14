#!/usr/bin/env python3
"""Write the one-pass-gap grid's orchestrator configs into configs/.

Two trainers on the same d64/L2 model, the same seed weights, the same
Tiny Shakespeare split and the same Adam and per-step warmup-cosine schedule:

- agpt:  bin/agpt_train_v2, depth-16 trie, pd=1 (65 updates per pass),
         exact ancestor gradient; the setup of the record's best d64/L2 run
         (gradient-population adam-pd1-exact-625ep), only budget and lr change.
- mb:    src/tools/agpt_minibatch_train.py, every window start once per pass,
         128 windows of 16 per step (8,279 updates per pass).
- mbL:   the same trainer with 16,302 windows per step, so 65 updates per pass,
         as many as AGPT pd=1 takes: the update-count control.

Each budget is its own run, so every run anneals its own schedule.
Usage: python3 rnd/one-pass-gap/make_configs.py
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "configs"
BUDGETS = [1, 2, 4, 8]
LRS = {"agpt": [0.003, 0.006, 0.012, 0.024], "mb": [0.0015, 0.003, 0.006],
       "mbL": [0.003, 0.006, 0.012, 0.024]}
MINIBATCH = {"mb": 128, "mbL": 16302}
MICRO = {"mb": "", "mbL": "  micro_batch: 2048\n"}

COMMON = """corpus:
  path: data/.splits/4fa9aec1db6b3aea/train_corpus.txt
  heldout: data/.splits/4fa9aec1db6b3aea/heldout_corpus.txt
  vocab_source: data/input.txt
{trie}model:
  init_file: rnd/cudax-d16-linear-mass-rerun/seeds/shake-d64L2-h4-dff256-d16-seed42.model
train:
  budget:
    unit: epochs
    value: {epochs}
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
{train_extra}  seq_len: 16
eval:
  split: tail-heldout
  train_sanity: false
  batch_size: 1
  device: cpu
experimental:
{experimental}"""

TRIE = """trie:
  prune_min_mass: 1
  prune_min_depth: 0
  path: data/.tries/64e9fe109211b366
  virtual_tree: false
"""
AGPT_TRAIN = """  max_depth: 16
  partition_depth: 1
  chunk_queries: 50000
  anc_grad: true
"""


def main():
    OUT.mkdir(exist_ok=True)
    n = 0
    for method, lrs in LRS.items():
        for epochs in BUDGETS:
            for lr in lrs:
                slug = f"{method}-e{epochs}-lr{lr:g}"
                if method == "agpt":
                    desc = (f"one-pass gap: AGPT pd=1 (depth-16 trie, exact ancestor gradient), "
                            f"{epochs} pass(es), Adam lr {lr:g} warmup-cosine per step")
                    body = COMMON.format(trie=TRIE, epochs=epochs, lr=lr, train_extra=AGPT_TRAIN,
                                         experimental="  anc_grad_exact: true\n")
                else:
                    mbs = MINIBATCH[method]
                    desc = (f"one-pass gap: ordinary minibatch Adam ({mbs} windows of 16 per step), "
                            f"{epochs} pass(es), lr {lr:g} warmup-cosine per step")
                    body = COMMON.format(trie="", epochs=epochs, lr=lr, train_extra="",
                                         experimental=f"  minibatch_size: {mbs}\n{MICRO[method]}  device: cuda\n")
                text = (f'---\ndescription: "{desc}"\nexperiment: one-pass-gap\n'
                        f"run_slug: {slug}\n{body}")
                (OUT / f"{slug}.yml").write_text(text)
                n += 1
    print(f"wrote {n} configs to {OUT}")


if __name__ == "__main__":
    main()
