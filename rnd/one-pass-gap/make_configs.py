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

Follow-up (AGPT at minibatch cadence): v2 at pd=2, 3 and 4 with
experimental.train_crown, so the targets above the partition depth are
trained too (each crown node is a unit of its own), and minibatch runs with
the same number of updates per pass as pd=2 and pd=3:

- pd2c:  pd=2 + crown, 1,463 units per pass, radix-id order.
- pd3c:  pd=3 + crown, 12,675 units per pass, radix-id order.
- pd3cs: the same, shuffled each pass (experimental.unit_order_seed 1).
  (2- and 4-pass budgets are added once the one-pass rates are known.)
- pd4c:  pd=4 + crown, 58,721 units per pass.
- mb724: 724 windows per step, 1,464 updates per pass (matches pd2c).
- mb84:  84 windows per step, 12,615 updates per pass (matches pd3c).
- pd1s, pd3css: pd=1 and shuffled pd=3 + crown with train.fire_norm: sum,
  every unit's summed gradient divided by one constant, so that a pass
  weights every event equally (the default event mean weights every unit
  equally, whatever its size).

Each budget is its own run, so every run anneals its own schedule.
Usage: python3 rnd/one-pass-gap/make_configs.py
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "configs"
BUDGETS = [1, 2, 4, 8]
LRS = {"agpt": [0.003, 0.006, 0.012, 0.024], "mb": [0.0015, 0.003, 0.006, 0.012],
       "mbL": [0.003, 0.006, 0.012, 0.024]}
MINIBATCH = {"mb": 128, "mbL": 16302}
MICRO = {"mb": "", "mbL": "  micro_batch: 2048\n"}

# Follow-up arms: ({budget: lrs}, partition depth or minibatch size, shuffled order).
# pd=3's best one-pass rate was 0.0015, the bottom of the first grid, so its
# grid was extended downwards (and pd=4's started lower) before the rest ran.
LO, HI = [0.0015, 0.003, 0.006, 0.012], [0.003, 0.006, 0.012, 0.024]
PD3_E1 = [0.000375, 0.00075] + LO
CADENCE = {"pd2c": ({1: [0.0015] + HI}, 2, False), "pd3c": ({1: PD3_E1}, 3, False),
           "pd3cs": ({1: PD3_E1}, 3, True), "pd4c": ({1: [0.000375, 0.00075, 0.0015]}, 4, False),
           "mb724": ({1: HI}, 724, False), "mb84": ({1: LO}, 84, False),
           # train.fire_norm: sum (every unit's summed gradient over one constant, so a
           # pass weights every event equally instead of every unit equally)
           "pd1s": ({1: HI, 8: [0.003, 0.006, 0.012]}, 1, False),
           "pd3css": ({1: [0.00075, 0.0015, 0.003, 0.006]}, 3, True)}

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
    for arm, (grid, size, shuffled) in CADENCE.items():
        for epochs, lrs in grid.items():
            for lr in lrs:
                slug = f"{arm}-e{epochs}-lr{lr:g}"
                if arm.startswith("pd"):
                    order = "shuffled each pass" if shuffled else "radix-id order"
                    desc = (f"one-pass gap, cadence: AGPT pd={size} + crown units (depth-16 trie, exact "
                            f"ancestor gradient, {order}), {epochs} pass(es), Adam lr {lr:g} warmup-cosine per step")
                    fire_sum = arm in ("pd1s", "pd3css")
                    crown = size >= 2
                    if fire_sum:
                        desc = desc.replace(" + crown units (", " + crown units, fire_norm sum (") if crown else (
                            f"one-pass gap, cadence: AGPT pd=1 (depth-16 trie, exact ancestor gradient), fire_norm sum, "
                            f"{epochs} pass(es), Adam lr {lr:g} warmup-cosine per step")
                    exp = "  anc_grad_exact: true\n" + ("  train_crown: true\n" if crown else "")
                    if shuffled:
                        exp += "  unit_order_seed: 1\n"
                    extra = AGPT_TRAIN.replace("partition_depth: 1", f"partition_depth: {size}")
                    if fire_sum:
                        extra += "  fire_norm: sum\n"
                    body = COMMON.format(trie=TRIE, epochs=epochs, lr=lr, train_extra=extra,
                                         experimental=exp)
                else:
                    desc = (f"one-pass gap, cadence: ordinary minibatch Adam ({size} windows of 16 per step), "
                            f"{epochs} pass(es), lr {lr:g} warmup-cosine per step")
                    body = COMMON.format(trie="", epochs=epochs, lr=lr, train_extra="",
                                         experimental=f"  minibatch_size: {size}\n  device: cuda\n")
                text = (f'---\ndescription: "{desc}"\nexperiment: one-pass-gap\n'
                        f"run_slug: {slug}\n{body}")
                (OUT / f"{slug}.yml").write_text(text)
                n += 1
    print(f"wrote {n} configs to {OUT}")


if __name__ == "__main__":
    main()
