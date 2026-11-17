#!/usr/bin/env python3
"""Write the attention grouping control's orchestrator configs into configs/.

Proviso check 2 of the context-tree-orientation README: the no-trie grouping
control (src/tools/agpt_unit_grouping_control.py, a GRU) with an attention
f_theta instead, at depth 16. The d64/L2 attention model and seed weights of
rnd/one-pass-gap, trained by src/tools/agpt_minibatch_train.py with
experimental.trie_plan group over the trie's tail-wrapped windows: every
event (window start i, context length L <= 16) once per pass, 65 steps per
pass (one per character), the steps in a new random order each pass, an
event-mean gradient per step, Adam at a constant rate (as the GRU control).

- first:  each step holds the events whose context starts with one character,
          i.e. AGPT's prefix-trie pd=1 units (the same partition as pd depth 1).
- last:   each step holds the events whose context ends with one character,
          i.e. the suffix tree's pd=1 units: target-sorted batches.
- window: whole windows assigned at random each pass to groups of first's
          sizes: the size-matched shuffled control.

With tail wrapping all three take steps of exactly the same sizes. One run
per (arm, rate) is evaluated at passes 1, 2, 4, 8 and 16 (and 32 for first and
window, which cost a seventh as much per pass as last). A second order seed
(trie_plan_seed 2) at 0.003 for every arm.

Cosine check (cfirst, cwindow): first and window for 16 passes under
warmup-cosine (one pass of warmup, then cosine to 0), peak rates 0.003 and
0.006, evaluated at passes 4, 8 and 16. Under a constant rate the window
arm's train loss is not monotone (slots 66 and 68), as the GRU control's
shuffled-window cell's was at a constant 3e-3, so the late comparison needs a
decaying schedule.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "configs"

RATES = [0.0015, 0.003, 0.006]
PASSES = {"first": 32, "last": 16, "window": 32}
SEED2_RATE = 0.003
DESC = {
    "first": "steps group events by their context's oldest character (AGPT's prefix-trie pd=1 units)",
    "last": "steps group events by their context's newest character (the suffix tree's pd=1 units)",
    "window": "whole windows assigned at random each pass to groups of the prefix units' sizes",
}
COSINE = {"first": [0.003, 0.006], "window": [0.003, 0.006]}
COSINE_PASSES = 16

COMMON = """corpus:
  path: data/.splits/4fa9aec1db6b3aea/train_corpus.txt
  heldout: data/.splits/4fa9aec1db6b3aea/heldout_corpus.txt
  vocab_source: data/input.txt
model:
  init_file: rnd/cudax-d16-linear-mass-rerun/seeds/shake-d64L2-h4-dff256-d16-seed42.model
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
{schedule}
  seq_len: 16
  checkpoint_epochs: [{checkpoints}]
eval:
  split: tail-heldout
  train_sanity: false
  batch_size: 1
  device: cpu
experimental:
  trie_plan: group
  trie_plan_group: {arm}
  trie_plan_seed: {plan_seed}
  wrap_tail: true
  device: cuda
"""


def main():
    OUT.mkdir(exist_ok=True)
    n = 0
    for arm, passes in PASSES.items():
        checkpoints = ", ".join(str(p) for p in (1, 2, 4, 8, 16, 32) if p < passes)
        runs = [(lr, 1) for lr in RATES] + [(SEED2_RATE, 2)]
        for lr, s in runs:
            slug = f"att-{arm}-lr{lr:g}" + ("-s2" if s == 2 else "")
            desc = (f"context-tree orientation, attention grouping control: d64/L2 attention, depth 16, "
                    f"{DESC[arm]}; 65 steps per pass in a new random order, {passes} passes, "
                    f"Adam lr {lr:g} constant, order seed {s}")
            body = COMMON.format(passes=passes, seed=42 if s == 1 else 43, lr=lr, checkpoints=checkpoints,
                                 arm=arm, plan_seed=s, schedule="    name: constant")
            (OUT / f"{slug}.yml").write_text(f'---\ndescription: "{desc}"\nexperiment: context-tree-orientation\n'
                                             f"run_slug: {slug}\n{body}")
            n += 1
    for arm, rates in COSINE.items():
        for lr in rates:
            slug = f"att-c{arm}-lr{lr:g}"
            desc = (f"context-tree orientation, attention grouping control, cosine check: d64/L2 attention, depth 16, "
                    f"{DESC[arm]}; 65 steps per pass in a new random order, {COSINE_PASSES} passes, "
                    f"Adam lr {lr:g} warmup-cosine (1 pass of warmup), order seed 1")
            body = COMMON.format(passes=COSINE_PASSES, seed=42, lr=lr, checkpoints="4, 8", arm=arm, plan_seed=1,
                                 schedule="    name: warmup-cosine\n    warmup_epochs: 1")
            (OUT / f"{slug}.yml").write_text(f'---\ndescription: "{desc}"\nexperiment: context-tree-orientation\n'
                                             f"run_slug: {slug}\n{body}")
            n += 1
    print(f"wrote {n} configs to {OUT}")


if __name__ == "__main__":
    main()
