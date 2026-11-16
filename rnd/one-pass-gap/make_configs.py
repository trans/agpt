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
- mbu3:  random windows in batches with pd=3's 12,675 unit sizes
  (pd3_unit_events.txt: each unit's trained events, from the pd3cs lr 0.0015
  run's log; a step takes max(1, round(events / 16)) windows), in a new
  random order each pass. Against pd3cs it isolates the trie's grouping;
  against mb84 (same steps, equal sizes) it isolates the unequal sizes.
- pd1s, pd3css: pd=1 and shuffled pd=3 + crown with train.fire_norm: sum,
  every unit's summed gradient divided by one constant, so that a pass
  weights every event equally (the default event mean weights every unit
  equally, whatever its size).

Second follow-up (2026-11-14), both through the minibatch trainer:

- cv:    conventional epochs (experimental.window_stride 16: non-overlapping
         windows from a random offset each epoch, each target once per epoch),
         128 windows per step, 518 steps per epoch. A pass above is 16 such
         epochs of events; an AGPT depth-16 pass computes as many positions as
         ~8.4 of them.
- tp3:   replay of v2's pd=3 + crown plan (experimental.trie_plan pd, depth 3):
         exactly v2's 12,675 units, one per step, shuffled each pass. The
         control: it should reproduce pd3cs (7.81) if the replay is faithful.
- tqP:   packed plans (experimental.trie_plan packed): pieces (the shallowest
         subtrees with <= P events) and chunks of heavier nodes' own events,
         packed at random into 12,675 equal steps. P = 1338, 335, 84 keeps 97%,
         88%, 65% of the trie's sharing (trie positions per pass 9.18M, 9.90M,
         11.71M, against 8.90M for one trie step and 16.95M with none).

Fourth follow-up (2026-11-15), longer windows: conventional epochs with
one-hot and with trie targets at seq_len T = 32, 64, 128, with 2048 targets
per step (2048 / T windows), so every window length takes 518 steps per
epoch, as T = 16 does with 128 windows:

- cv{T}:  conventional epochs, one-hot targets.
- cvt{T}: the same with trie targets.

Fifth follow-up (2026-11-15), exact emulation, through the minibatch trainer:

- tp1x: v2's pd=1 plan (`agpt`), its 65 units in v2's order (the units.tsv
        of a frozen v2 gradient dump, v2_units/pd1.tsv), over the trie's
        windows (one per character, the tail wrapped to the front).
- tp3x: the same for pd=3 + crown shuffled (`pd3cs`), v2's epoch-1 order
        (v2_units/pd3s.tsv).

Sixth follow-up (2026-11-15), equal-mass steps at pd=1's cadence (65 steps):

- pkP:  packed plans with 65 equal steps of ~260,833 events, P = 260833 (one
        step), 65208, 16302, 1338 and 84: ~1 to ~3,100 pieces per step, keeping
        100% to 84% of the sharing (trie positions per pass 8.90M to 10.22M).
- mbu1: random windows in batches with pd=1's 65 unit sizes
        (pd1_unit_events.txt, from v2_units/pd1.tsv), shuffled each pass.

Seventh follow-up (2026-11-16), stochastic average gradient (SAG) on AGPT's units,
the smallest plans first (the table holds one gradient per unit: 28 MB at pd=1,
0.6 GB at pd=2, 5.5 GB at pd=3), through the minibatch trainer's replay:

- sg1:  v2's pd=1 plan in v2's order over v2's events, as tp1x, with
        experimental.grad_average sag: each step recomputes its unit and Adam gets
        the mass-weighted mean of every unit's latest gradient. Adam's own momentum
        is off (momentum_beta 0; on a 20k-character calibration it did better than
        0.9). Baselines: agpt (v2, same plan and order; tp1x matched it to 0.02) and
        mbL, at the same budgets.
pd=2 (SAG and its shuffled replay without SAG) waits on pd=1's outcome.

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
           "pd3cs": ({1: PD3_E1, 2: [0.00075, 0.0015], 4: [0.00075, 0.0015]}, 3, True), "pd4c": ({1: [0.000375, 0.00075, 0.0015]}, 4, False),
           "mb724": ({1: HI}, 724, False), "mb84": ({1: LO}, 84, False),
           # train.fire_norm: sum (every unit's summed gradient over one constant, so a
           # pass weights every event equally instead of every unit equally)
           "pd1s": ({1: HI, 8: [0.003, 0.006, 0.012]}, 1, False),
           "pd3css": ({1: [0.00075, 0.0015, 0.003, 0.006]}, 3, True),
           # random batches with pd=3's unit sizes (pd3_unit_events.txt), shuffled each pass
           "mbu3": ({1: [0.00075, 0.0015, 0.003, 0.006]}, 0, True)}

# Second follow-up arms: {budget: lrs} and the experimental keys.
NEXT = {
    "cv": ({1: [0.003, 0.006, 0.012, 0.024], 2: [0.003, 0.006, 0.012, 0.024], 4: [0.003, 0.006, 0.012],
            8: [0.003, 0.006, 0.012], 16: [0.003, 0.006, 0.012]},
           "  minibatch_size: 128\n  window_stride: 16\n"),
    "cvt": ({1: [0.003, 0.006, 0.012, 0.024], 2: [0.003, 0.006, 0.012], 4: [0.003, 0.006, 0.012],
             8: [0.003, 0.006, 0.012], 16: [0.003, 0.006, 0.012]},
            "  minibatch_size: 128\n  window_stride: 16\n  trie_targets: true\n"),
    "mbt": ({1: [0.003, 0.006, 0.012]}, "  minibatch_size: 128\n  trie_targets: true\n"),
    # Follow-up 5: exact replays, v2's units in v2's order over v2's (tail-wrapped) events
    "tp1x": ({1: [0.003, 0.006, 0.012, 0.024]},
             "  trie_plan: pd\n  trie_plan_depth: 1\n  trie_plan_order_file: rnd/one-pass-gap/v2_units/pd1.tsv\n"
             "  wrap_tail: true\n  log_step_loss: true\n"),
    "tp3x": ({1: [0.00075, 0.0015, 0.003]},
             "  trie_plan: pd\n  trie_plan_depth: 3\n  trie_plan_order_file: rnd/one-pass-gap/v2_units/pd3s.tsv\n"
             "  wrap_tail: true\n  log_step_loss: true\n"),
    "tp3": ({1: [0.00075, 0.0015, 0.003]}, "  trie_plan: pd\n  trie_plan_depth: 3\n"),
    "tq1338": ({1: [0.0015, 0.003, 0.006]},
               "  trie_plan: packed\n  trie_plan_piece_max: 1338\n  trie_plan_steps: 12675\n"),
    "tq335": ({1: [0.0015, 0.003, 0.006, 0.012]},
              "  trie_plan: packed\n  trie_plan_piece_max: 335\n  trie_plan_steps: 12675\n"),
    "tq84": ({1: [0.0015, 0.003, 0.006, 0.012]},
             "  trie_plan: packed\n  trie_plan_piece_max: 84\n  trie_plan_steps: 12675\n"),
    # Follow-up 6: packed plans at pd=1's cadence (65 equal steps of ~260,833 events), from
    # pieces as big as a step (P = 260833) down to P = 84, so ~1 to ~3,100 pieces per step
    "pk260833": ({1: [0.003, 0.006, 0.012, 0.024]},
              "  trie_plan: packed\n  trie_plan_piece_max: 260833\n  trie_plan_steps: 65\n"),
    "pk65208": ({1: [0.003, 0.006, 0.012, 0.024]},
              "  trie_plan: packed\n  trie_plan_piece_max: 65208\n  trie_plan_steps: 65\n"),
    "pk16302": ({1: [0.003, 0.006, 0.012, 0.024]},
              "  trie_plan: packed\n  trie_plan_piece_max: 16302\n  trie_plan_steps: 65\n"),
    "pk1338": ({1: [0.003, 0.006, 0.012, 0.024]},
              "  trie_plan: packed\n  trie_plan_piece_max: 1338\n  trie_plan_steps: 65\n"),
    "pk84": ({1: [0.003, 0.006, 0.012, 0.024]},
              "  trie_plan: packed\n  trie_plan_piece_max: 84\n  trie_plan_steps: 65\n"),
    "mbu1": ({1: [0.003, 0.006, 0.012, 0.024]},
             "  step_events_file: rnd/one-pass-gap/pd1_unit_events.txt\n  micro_batch: 2048\n"),
    # Follow-up 7: SAG on v2's pd=1 units (Adam momentum off, see make_body)
    "sg1": ({1: [0.0015, 0.003, 0.006], 8: [0.0015, 0.003, 0.006]},
            "  trie_plan: pd\n  trie_plan_depth: 1\n  trie_plan_order_file: rnd/one-pass-gap/v2_units/pd1.tsv\n"
            "  wrap_tail: true\n  grad_average: sag\n"),
}
NO_MOMENTUM = {"sg1"}  # arms run with Adam momentum_beta 0
NEXT_DESC = {
    "cv": "conventional epochs (non-overlapping windows from a random offset each epoch, 128 per step)",
    "tp3": "replay of AGPT's pd=3 + crown plan (12,675 units, one per step, shuffled)",
    "tq1338": "packed AGPT plan: pieces <= 1338 events in 12,675 equal steps",
    "tq335": "packed AGPT plan: pieces <= 335 events in 12,675 equal steps",
    "tq84": "packed AGPT plan: pieces <= 84 events in 12,675 equal steps",
    "pk260833": "packed AGPT plan at pd=1's cadence: pieces <= 260833 events in 65 equal steps",
    "pk65208": "packed AGPT plan at pd=1's cadence: pieces <= 65208 events in 65 equal steps",
    "pk16302": "packed AGPT plan at pd=1's cadence: pieces <= 16302 events in 65 equal steps",
    "pk1338": "packed AGPT plan at pd=1's cadence: pieces <= 1338 events in 65 equal steps",
    "pk84": "packed AGPT plan at pd=1's cadence: pieces <= 84 events in 65 equal steps",
    "mbu1": "minibatch Adam over random windows in batches with AGPT pd=1's 65 unit sizes (shuffled each pass)",
    "sg1": "SAG on AGPT pd=1's units (v2's 65 units in v2's order over the trie's tail-wrapped windows; "
           "Adam gets the mass-weighted mean of every unit's latest gradient, Adam momentum off)",
    "cvt": "conventional epochs with trie targets (each position's target is the trie's next-character "
           "distribution for its context)",
    "mbt": "minibatch 128 over every window start with trie targets",
    "tp1x": "exact replay of AGPT pd=1 (v2's 65 units in v2's order, over the trie's tail-wrapped windows)",
    "tp3x": "exact replay of AGPT pd=3 + crown, shuffled (v2's 12,675 units in v2's epoch-1 order, "
            "over the trie's tail-wrapped windows)",
}

# Fourth follow-up: window length T -> ({budget: lrs}), both arms.
LONG = {T: {1: [0.003, 0.006, 0.012], 4: [0.003, 0.006, 0.012]} for T in (32, 64, 128)}
LONG_TARGETS = 2048

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
                elif arm == "mbu3":
                    desc = (f"one-pass gap, cadence: minibatch Adam over random windows in batches with AGPT pd=3's "
                            f"12,675 unit sizes (shuffled each pass), {epochs} pass(es), lr {lr:g} warmup-cosine per step")
                    body = COMMON.format(trie="", epochs=epochs, lr=lr, train_extra="",
                                         experimental="  step_events_file: rnd/one-pass-gap/pd3_unit_events.txt\n"
                                                      "  micro_batch: 2048\n  device: cuda\n")
                else:
                    desc = (f"one-pass gap, cadence: ordinary minibatch Adam ({size} windows of 16 per step), "
                            f"{epochs} pass(es), lr {lr:g} warmup-cosine per step")
                    body = COMMON.format(trie="", epochs=epochs, lr=lr, train_extra="",
                                         experimental=f"  minibatch_size: {size}\n  device: cuda\n")
                text = (f'---\ndescription: "{desc}"\nexperiment: one-pass-gap\n'
                        f"run_slug: {slug}\n{body}")
                (OUT / f"{slug}.yml").write_text(text)
                n += 1
    for arm, (grid, exp) in NEXT.items():
        for epochs, lrs in grid.items():
            for lr in lrs:
                slug = f"{arm}-e{epochs}-lr{lr:g}"
                unit = "epoch(s)" if arm in ("cv", "cvt") else "pass(es)"
                desc = (f"one-pass gap, next: {NEXT_DESC[arm]}, {epochs} {unit}, "
                        f"Adam lr {lr:g} warmup-cosine per step")
                body = COMMON.format(trie="", epochs=epochs, lr=lr, train_extra="",
                                     experimental=exp + "  device: cuda\n")
                if arm in NO_MOMENTUM:
                    body = body.replace("    momentum_beta: 0.9\n", "    momentum_beta: 0.0\n")
                text = (f'---\ndescription: "{desc}"\nexperiment: one-pass-gap\n'
                        f"run_slug: {slug}\n{body}")
                (OUT / f"{slug}.yml").write_text(text)
                n += 1
    for T, grid in LONG.items():
        windows = LONG_TARGETS // T
        for arm in (f"cv{T}", f"cvt{T}"):
            tt = arm.startswith("cvt")
            what = "with trie targets" if tt else "with one-hot targets"
            exp = f"  minibatch_size: {windows}\n  window_stride: {T}\n" + ("  trie_targets: true\n" if tt else "")
            for epochs, lrs in grid.items():
                for lr in lrs:
                    slug = f"{arm}-e{epochs}-lr{lr:g}"
                    desc = (f"one-pass gap, longer windows: conventional epochs at seq_len {T} {what} "
                            f"({windows} windows of {T} per step), {epochs} epoch(s), Adam lr {lr:g} warmup-cosine per step")
                    body = COMMON.format(trie="", epochs=epochs, lr=lr, train_extra="",
                                         experimental=exp + "  device: cuda\n")
                    body = body.replace("  seq_len: 16\n", f"  seq_len: {T}\n")
                    text = (f'---\ndescription: "{desc}"\nexperiment: one-pass-gap\n'
                            f"run_slug: {slug}\n{body}")
                    (OUT / f"{slug}.yml").write_text(text)
                    n += 1
    print(f"wrote {n} configs to {OUT}")


if __name__ == "__main__":
    main()
