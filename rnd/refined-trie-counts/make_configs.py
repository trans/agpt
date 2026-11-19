#!/usr/bin/env python3
"""Write the refined-trie-counts configs into configs/.

The question (the user, 2026-11-17): mobius refines a small corpus's count
table with a larger corpus's counts; can AGPT use that? AGPT's loss reads the
text only through its trie's counts n(c, a), and its cost is one evaluation
per trie node however often the node occurs. So the trie of a small corpus A
can keep its nodes and take its child counts from a larger corpus B that
contains A: same nodes, same steps, the larger corpus's evidence.

All runs: the d64/L2 attention model (h4, d_ff 256) from one random init
(seed 42), contexts of up to 8 characters (seq_len 8, the depth where the
trie's sharing grows with corpus size), PG-19 as mobius normalizes it (96
symbols), Adam at a constant 0.003, canonical eval on the 50 held-out chunks
of 5,000 characters shared by every PG-19 split. A = the first 1,000,000
characters (40 pieces) of the 4.75M split's training text; the 4.75M, 20M
and 100M training texts each contain the one before.

AGPT arms replay v2's pd=3 + crown plan (trie_plan pd, depth 3: 19,999 steps
a pass on A, a new random order each pass):
- agpt-a1m:            A's trie, A's counts (hard targets; the same steps)
- agpt-a1m-ref{5,20,100}m: A's trie, counts from the 4.75M / 20M / 100M text
- agpt-5m:             the 4.75M text's own trie (2.95x A's nodes), 3 passes
Minibatch arms (128 windows a step, every window start once an epoch):
- mb-a1m, mb-a1m-ref100m, mb-5m (2 epochs = 9.5 of A's in windows)
Second order seeds (train.seed 43, trie_plan_seed 2) for agpt-a1m and
agpt-a1m-ref100m.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "configs"
P = "data/pg19"
REFS = {
    "5m": (f"{P}/imm-split/train_corpus.txt", "4b7c1b1069f3d50deb37a04b216e4475c61ba1db8c5b624d1ac7453708618ddc"),
    "20m": (f"{P}/imm-split-20m/train_corpus.txt", "754b3660b01e6d172c20c19d7c0ec59ffe43067a66f56eb31bb3e10b75cef6ba"),
    "100m": (f"{P}/imm-split-100m/train_corpus.txt", "fc55205ead4c13554ab2992b5891263aad822956618d801720dab840af124c54"),
}
CORPUS = {"a1m": f"{P}/a1m/train_corpus.txt", "5m": REFS["5m"][0], "100m": REFS["100m"][0]}
LR = 0.003

BODY = """corpus:
  path: {corpus}
  heldout: {P}/imm-split/heldout_corpus.txt
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
    value: {passes}
  seed: {seed}
  quiet: true
  optimizer:
    name: adam
    lr: {lr}
    beta: 0.999
    momentum_beta: 0.9
  lr_schedule:
    name: constant
  seq_len: 8
  checkpoint_epochs: [{checkpoints}]
eval:
  split: tail-heldout
  train_sanity: false
  batch_size: 1
  device: cpu
experimental:
  device: cuda
{exp}"""


def config(slug, desc, corpus, passes, checkpoints, seed, agpt, ref, pd=3, lr=LR, extra=()):
    exp = []
    if agpt:
        exp += ["trie_plan: pd", f"trie_plan_depth: {pd}", f"trie_plan_seed: {1 if seed == 42 else 2}"]
    if ref == "own":
        exp += ["trie_targets: true"]
    elif ref:
        path, sha = REFS[ref]
        exp += ["trie_targets: true", f"trie_targets_corpus: {path}", f"trie_targets_corpus_sha256: {sha}"]
    exp += list(extra)
    body = BODY.format(corpus=corpus, P=P, passes=passes, seed=seed, lr=lr,
                       checkpoints=", ".join(str(c) for c in checkpoints),
                       exp="".join(f"  {e}\n" for e in exp))
    (OUT / f"{slug}.yml").write_text(f'---\ndescription: "{desc}"\nexperiment: refined-trie-counts\n'
                                     f"run_slug: {slug}\n{body}")


def main():
    OUT.mkdir(exist_ok=True)
    runs = []
    for seed in (42, 43):
        s2 = "" if seed == 42 else "-s2"
        for ref in ([None, "5m", "20m", "100m"] if seed == 42 else [None, "100m"]):
            slug = "agpt-a1m" + (f"-ref{ref}" if ref else "") + s2
            counts = f"its child counts from the {ref.upper()} PG-19 text containing it" if ref else "its own counts"
            runs.append((slug, f"refined trie counts: AGPT pd=3 replay, d64/L2, depth 8, the 1M PG-19 slice's trie "
                               f"with {counts}; 8 passes, Adam 0.003 constant, order seed {1 if seed == 42 else 2}",
                         CORPUS["a1m"], 8, [1, 2, 4], seed, True, ref))
    runs.append(("agpt-5m", "refined trie counts: AGPT pd=3 replay, d64/L2, depth 8, the 4.75M PG-19 text's own trie; "
                            "3 passes, Adam 0.003 constant", CORPUS["5m"], 3, [1, 2], 42, True, None))
    runs.append(("mb-a1m", "refined trie counts: minibatch (128 windows a step), d64/L2, depth 8, the 1M PG-19 slice, "
                           "hard targets; 8 epochs, Adam 0.003 constant", CORPUS["a1m"], 8, [1, 2, 4], 42, False, None))
    runs.append(("mb-a1m-ref100m", "refined trie counts: minibatch (128 windows a step), d64/L2, depth 8, the 1M PG-19 "
                                   "slice with each context's next-token distribution from the 100M text containing it; "
                                   "8 epochs, Adam 0.003 constant", CORPUS["a1m"], 8, [1, 2, 4], 42, False, "100m"))
    # Added after the grid (slot 86): the minibatch pair mixes soft-vs-one-hot targets with the 100M
    # counts; this splits them (in the AGPT arms own-count soft targets are the hard-target steps).
    runs.append(("mb-a1m-tt", "refined trie counts: minibatch (128 windows a step), d64/L2, depth 8, the 1M PG-19 "
                              "slice with each context's next-token distribution from the slice itself (trie targets, "
                              "own counts); 8 epochs, Adam 0.003 constant", CORPUS["a1m"], 8, [1, 2, 4], 42, False,
                 "own"))
    runs.append(("mb-5m","refined trie counts: minibatch (128 windows a step), d64/L2, depth 8, the 4.75M PG-19 text, "
                          "hard targets; 2 epochs, Adam 0.003 constant", CORPUS["5m"], 2, [1], 42, False, None))
    # Follow-up (the user, 2026-11-18: pd=1 and pd=0 too; pd=6 is the deepest plan on record and
    # its best score): the same pair, own counts against the 100M text's, at other plan depths.
    # pd=0 is one full-batch step a pass (no order, so one seed), pd=1 89 steps a pass, pd=6
    # 379,312 (at the record's pd=6 rate, 0.001).
    PD = {0: (512, [32, 128, 256], [42], LR), 1: (128, [8, 32, 64], [42, 43], LR), 6: (2, [1], [42], 0.001)}
    for pd, (passes, cps, seeds, lr) in PD.items():
        for seed in seeds:
            s2 = "" if seed == 42 else "-s2"
            for ref in (None, "100m"):
                slug = f"agpt-a1m-pd{pd}" + (f"-ref{ref}" if ref else "") + s2
                counts = "its child counts from the 100M PG-19 text containing it" if ref else "its own counts"
                order = "one full-batch step a pass" if pd == 0 else f"order seed {1 if seed == 42 else 2}"
                runs.append((slug, f"refined trie counts, plan depths: AGPT pd={pd} replay, d64/L2, depth 8, the 1M "
                                   f"PG-19 slice's trie with {counts}; {passes} passes, Adam {lr:g} constant, {order}",
                             CORPUS["a1m"], passes, cps, seed, True, ref, pd, lr))
    # Weighted (the user, 2026-11-18: loading the small trie with the larger text's statistics
    # should raise the sharing; go-ahead 2026-11-19): each event weighted by its context's
    # count in the 100M text over its count in A (trie_weights_corpus), so A's 1.54M nodes
    # stand for the 100M text's loss on the contexts A has (666.9M of its 800M depth-8
    # events, 433 a node). Against the unweighted 100M-count runs above at the same steps,
    # and against ordinary minibatch on the 100M text itself, the control the epoch-only
    # budget left out: window_stride 100 makes an epoch 1M windows (7,813 steps, as an
    # epoch of A) from a random offset, so its epochs line up with A's in steps and
    # positions. 25 epochs = 200M positions, the trie positions of 128 pd=1 passes
    # (128 x 1,539,197 = 197M); 8 pd=3 passes (8 x 1,576,699 = 12.6M) fall between its
    # epochs 1 and 2.
    W = ["trie_weights_corpus: true"]
    wdesc = ("its child counts from the 100M PG-19 text containing it, and each event weighted by its "
             "context's count there over its count in the slice")
    runs.append(("agpt-a1m-w100m", f"refined trie counts, weighted: AGPT pd=3 replay, d64/L2, depth 8, the 1M "
                                   f"PG-19 slice's trie with {wdesc}; 8 passes, Adam 0.003 constant, order seed 1",
                 CORPUS["a1m"], 8, [1, 2, 4], 42, True, "100m", 3, LR, W))
    runs.append(("agpt-a1m-pd1-w100m", f"refined trie counts, weighted: AGPT pd=1 replay, d64/L2, depth 8, the 1M "
                                       f"PG-19 slice's trie with {wdesc}; 128 passes, Adam 0.003 constant, "
                                       f"order seed 1", CORPUS["a1m"], 128, [8, 32, 64], 42, True, "100m", 1, LR, W))
    runs.append(("mb-a1m-w100m", f"refined trie counts, weighted: minibatch (128 windows a step), d64/L2, depth 8, "
                                 f"the 1M PG-19 slice with {wdesc}; 8 epochs, Adam 0.003 constant",
                 CORPUS["a1m"], 8, [1, 2, 4], 42, False, "100m", 3, LR, W))
    runs.append(("mb-100m-st100", "refined trie counts: minibatch (128 windows a step), d64/L2, depth 8, the 100M "
                                  "PG-19 text itself, hard targets, windows every 100 characters from a random offset "
                                  "each epoch (1M windows, 7,813 steps an epoch); 25 epochs, Adam 0.003 constant",
                 CORPUS["100m"], 25, [1, 2, 4, 8, 16], 42, False, None, 3, LR, ["window_stride: 100"]))
    for r in runs:
        config(*r)
    print(f"wrote {len(runs)} configs to {OUT}")


if __name__ == "__main__":
    main()
