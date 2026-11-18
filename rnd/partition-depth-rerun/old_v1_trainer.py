#!/usr/bin/env python3
"""The partition-depth sweep's trainer (2026-04-30) as a custom trainer for
bin/agpt_experiment.

bin/agpt_experiment runs  old_v1_trainer.py --config RESOLVED.yml [--seed N].
The old agpt_train (a8ddb15, built by build_old.sh) predates --config, so this
maps the config onto its command line, the flags rnd/partition-depth/README.md
gives for its Adam result:

  agpt_train --model M --trie-dir T --save M --epochs E --lr LR --optimizer adam
             --lr-schedule warmup-cosine --warmup-epochs W --entropy-lambda L
             --mass-weight log --no-accumulate --partition-depth P

- The trie is built by the old trie builder from corpus.path at train.max_depth
  (the old trainer reads only the old trie format), cached under
  /home/trans/agpt-tries/old-a8ddb15/, or taken from experimental.old_trie_dir.
- The old trainer has no seed: repeats differ only by GPU nondeterminism, as
  the sweep's three repeats did. A --seed from the runner is reported, unused.
- The old trainer keeps the init model's header seq_len (128 for
  data/input.random.model) though it trains contexts of at most max_depth. The
  model uses RoPE, so seq_len only sets the evaluation window: this sets it to
  max_depth, the window the sweep scored at (bin/perplexity --seq-len 32).
  The weights are untouched.
"""
import argparse
import hashlib
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

import yaml

OLD = Path("/home/trans/agpt-tries/old-a8ddb15")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run(cmd):
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    t0 = time.time()
    subprocess.run([str(c) for c in cmd], check=True)
    print(f"  ({time.time() - t0:.0f} s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    tr, mc, exp = cfg["train"], cfg["model"], cfg.get("experimental") or {}
    if tr["budget"]["unit"] != "epochs":
        sys.exit("old_v1_trainer: budget.unit must be epochs")
    trainer = Path(exp.get("old_trainer", OLD / "agpt_train"))
    builder = Path(exp.get("old_trie_builder", OLD / "agpt_build_radix_corpus"))
    depth = int(tr["max_depth"])
    corpus = cfg["corpus"]["path"]
    print(f"old_v1_trainer: trainer {trainer} sha256 {sha256(trainer)}", flush=True)
    if args.seed is not None:
        print(f"old_v1_trainer: --seed {args.seed} ignored (the old trainer has no seed)", flush=True)

    trie = exp.get("old_trie_dir")
    if trie is None:
        print(f"old_v1_trainer: builder {builder} sha256 {sha256(builder)}", flush=True)
        trie = OLD / f"trie-{sha256(corpus)[:16]}-d{depth}"
        if not (trie / "meta.bin").is_file():
            tmp = trie.with_name(trie.name + ".partial")
            shutil.rmtree(tmp, ignore_errors=True)
            run([builder, "--corpus", corpus, "--max-depth", depth, "--out", tmp])
            tmp.rename(trie)
    print(f"old_v1_trainer: trie {trie}", flush=True)

    save = Path(mc["save_file"])
    shutil.copyfile(mc["init_file"], save)
    opt, sched = tr["optimizer"], tr.get("lr_schedule") or {}
    cmd = [trainer, "--model", save, "--trie-dir", trie, "--save", save,
           "--epochs", tr["budget"]["value"], "--lr", opt["lr"], "--optimizer", opt["name"]]
    if opt["name"] == "rmsprop" and opt.get("beta") is not None:
        cmd += ["--rmsprop-beta", opt["beta"]]
    cmd += ["--lr-schedule", sched.get("name", "constant"), "--warmup-epochs", sched.get("warmup_epochs", 0),
            "--entropy-lambda", tr.get("entropy_lambda") or 0.0, "--mass-weight", tr.get("mass_weight") or "off",
            "--no-accumulate", "--partition-depth", tr["partition_depth"]]
    run(cmd)

    with open(save, "r+b") as f:
        f.seek(24)
        old_sl = struct.unpack("<i", f.read(4))[0]
        f.seek(24)
        f.write(struct.pack("<i", depth))
    print(f"old_v1_trainer: header seq_len {old_sl} -> {depth} (the trained context; RoPE, weights unchanged)",
          flush=True)


if __name__ == "__main__":
    main()
