#!/usr/bin/env python3
"""Write the forward-backward coupling configs into configs/.

The question (the user, 2026-11-19, after the record's dual-model-fold
measured only the mechanism): does pulling a forward model toward a backward
model's predictions, the same character predicted from the text that follows
it, make the forward model better at predicting the next character?

All runs: the refined-trie-counts minibatch recipe (d64/L2, h4, d_ff 256,
random init seed 42, contexts of up to 8 characters, the 1M PG-19 slice,
128 windows a step, every window start once an epoch, 8 epochs, Adam 0.003
constant), canonical eval on the 50 held-out chunks. Trained on the CPU (2
threads, under 1 GB; the GPU queue was hours long on 2026-11-19), so all six
runs share one device and mb-a1m-base, not the GPU mb-a1m, is the baseline.
- mb-a1m-rev:  the backward model B: the same recipe on the reversed slice,
               scored on the reversed held-out chunks
- mb-a1m-base: the plain forward model (refined-trie-counts' mb-a1m again,
               under this code)
- fb-b{0.1,1}: forward, each target's loss (CE + beta CE(B, model)) / (1 + beta),
               B's distribution for that character given the 8 that follow it
               (teacher_preds.py --direction backward, B fixed)
- ff-b{0.1,1}: the control: the same with a forward teacher, mb-a1m (the plain
               forward model of the same recipe) given the 8 characters before,
               so plain distillation of the same strength. Backward helping
               more than this is the other side's information; the same is
               distillation.
Teacher files are 192 MB each and live outside the repo (TEACH below), checked
by SHA-256; teacher_preds.py rebuilds them from the checkpoints.
"""
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "configs"
TEACH = Path("/home/trans/agpt-tries/coupling")
P = "data/pg19"

BODY = """---
description: "{desc}"
experiment: forward-backward-coupling
run_slug: {slug}
corpus:
  path: {corpus}
  heldout: {heldout}
  vocab_source: {P}/imm-split/vocab.txt
model:
  d_model: 64
  n_layers: 2
  n_heads: 4
  d_ff: 256
  init_seed: 42
train:
  budget:
    unit: epochs
    value: 8
  seed: 42
  quiet: true
  optimizer:
    name: adam
    lr: 0.003
    beta: 0.999
    momentum_beta: 0.9
  lr_schedule:
    name: constant
  seq_len: 8
  checkpoint_epochs: [1, 2, 4]
eval:
  split: tail-heldout
  train_sanity: false
  batch_size: 1
  device: cpu
experimental:
  device: cpu
{exp}"""

RECIPE = "minibatch (128 windows a step), d64/L2, depth 8, the 1M PG-19 slice; 8 epochs, Adam 0.003 constant"


def write(slug, desc, corpus=f"{P}/a1m/train_corpus.txt", heldout=f"{P}/imm-split/heldout_corpus.txt", exp=()):
    OUT.mkdir(exist_ok=True)
    (OUT / f"{slug}.yml").write_text(BODY.format(desc=desc, slug=slug, corpus=corpus, heldout=heldout, P=P,
                                                 exp="".join(f"  {e}\n" for e in exp)))
    print(f"wrote configs/{slug}.yml")


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


write("mb-a1m-rev", f"forward-backward coupling: the backward model, {RECIPE}, on the reversed slice, scored on "
                    "the reversed held-out chunks", corpus=f"{P}/a1m-rev/train_corpus.txt",
      heldout=f"{P}/imm-split-rev/heldout_corpus.txt")
write("mb-a1m-base", f"forward-backward coupling: the plain forward model, {RECIPE}, hard targets")
for kind, f, who in (("fb", "backward_mb-a1m-rev.npy", "the backward model mb-a1m-rev (the 8 characters after)"),
                     ("ff", "forward_mb-a1m.npy", "the forward model mb-a1m (the 8 characters before; control)")):
    path = TEACH / f
    if not path.exists():
        print(f"skipped {kind}-*: {path} not built yet")
        continue
    h = sha(path)
    for b in ("0.1", "1"):
        write(f"{kind}-b{b}", f"forward-backward coupling: forward, {RECIPE}; each target's loss "
                              f"(CE + {b} CE(teacher, model)) / (1 + {b}), teacher {who}",
              exp=[f"teacher_targets_file: {path}", f"teacher_targets_sha256: {h}", f"teacher_beta: {b}"])
