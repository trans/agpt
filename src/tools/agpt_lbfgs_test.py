#!/usr/bin/env python3
"""Curvature test (rnd/gradient-population, Experiment 6): passes-to-parity for L-BFGS.

Plain full-batch GD (lr 0.1, 65k-window batches) needed ~2500 / 5000 /
10000 gradient passes from the epoch-25 checkpoint to reach held-out NLL
1.698 / 1.639 / 1.588 (Experiment 5b, variant A). How many passes does
L-BFGS need on the same model, same start, same batch size?

The objective is a FIXED random subset of windows (deterministic, as a
quasi-Newton method requires). Every closure call = one gradient pass.

Usage:
  python3 src/tools/agpt_lbfgs_test.py --init CKPT.model --out DIR [--windows 65536]
      [--max-passes 600] [--history 50] [--eval-every 20]
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agpt_ppl  # noqa: E402


def load_tokens(path, char_to_id):
    return [char_to_id[c] for c in open(path, encoding="utf-8").read() if c in char_to_id]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", default="data/.splits/4fa9aec1db6b3aea/train_corpus.txt")
    ap.add_argument("--heldout", default="data/.splits/4fa9aec1db6b3aea/heldout_corpus.txt")
    ap.add_argument("--vocab", default="data/input.txt")
    ap.add_argument("--windows", type=int, default=65536)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--max-passes", type=int, default=600)
    ap.add_argument("--history", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--positions", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--optimizer", choices=["lbfgs", "gd"], default="lbfgs")
    ap.add_argument("--lr", type=float, default=1.0, help="lbfgs: initial step (line search scales it); gd: step size")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    cfg, sd = agpt_ppl.load_model(args.init)
    W = cfg["seq_len"]
    char_to_id, _ = agpt_ppl.build_vocab(args.vocab)
    tok = torch.tensor(load_tokens(args.train, char_to_id), dtype=torch.long, device=dev)
    tok_h = torch.tensor(load_tokens(args.heldout, char_to_id), dtype=torch.long, device=dev)
    stride = max(1, (tok_h.numel() - W) // args.positions)
    targets_idx = torch.arange(W, tok_h.numel(), stride, device=dev)[:args.positions]
    tgt = tok_h[targets_idx]
    offsets = torch.arange(W + 1, device=dev).view(1, -1)
    starts = torch.tensor(np.sort(rng.integers(0, tok.numel() - W - 1, size=args.windows)), device=dev)

    model = agpt_ppl.AGPTModel(cfg, sd, device=dev).to(dev).train()
    params = list(model.parameters())

    @torch.no_grad()
    def heldout_nll():
        model.eval()
        tot = 0.0
        off = torch.arange(W, device=dev).view(1, -1)
        for s in range(0, len(targets_idx), 1024):
            i_range = targets_idx[s:s + 1024]
            ctx = tok_h[(i_range.view(-1, 1) - W) + off]
            lp = F.log_softmax(model(ctx)[:, -1, :].double(), dim=-1)
            tot += float(-lp.gather(1, tgt[s:s + 1024].unsqueeze(1)).sum())
        model.train()
        return tot / len(targets_idx)

    passes = 0
    log = {"args": vars(args), "records": []}
    t0 = time.time()
    last_loss = [float("nan")]

    def closure():
        nonlocal passes
        for p in params:
            p.grad = None
        total = 0.0
        for s in range(0, args.windows, args.batch):
            win = tok[starts[s:s + args.batch].view(-1, 1) + offsets]
            x, y = win[:, :-1], win[:, 1:]
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)) * (x.size(0) / args.windows)
            loss.backward()
            total += float(loss)
        passes += 1
        last_loss[0] = total
        if passes % args.eval_every == 0 or passes == 1:
            rec = {"passes": passes, "train_loss": total, "heldout_nll": heldout_nll(), "wall_s": time.time() - t0}
            log["records"].append(rec)
            print(f"pass {passes:5d}  train {total:.4f}  heldout {rec['heldout_nll']:.4f}  {rec['wall_s']:.0f}s", flush=True)
            json.dump(log, open(os.path.join(args.out, "log.json"), "w"), indent=1)
        return torch.tensor(total, device=dev)

    rec0 = {"passes": 0, "train_loss": None, "heldout_nll": heldout_nll(), "wall_s": 0.0}
    log["records"].append(rec0)
    print(f"start: heldout {rec0['heldout_nll']:.4f}  optimizer={args.optimizer}  windows={args.windows}  max_passes={args.max_passes}")

    if args.optimizer == "lbfgs":
        opt = torch.optim.LBFGS(params, lr=args.lr, max_iter=args.max_passes, max_eval=args.max_passes,
                                history_size=args.history, tolerance_grad=1e-9, tolerance_change=1e-12,
                                line_search_fn="strong_wolfe")
        opt.step(closure)
    else:
        while passes < args.max_passes:
            closure()
            with torch.no_grad():
                for p in params:
                    p.add_(p.grad, alpha=-args.lr)

    final = {"passes": passes, "train_loss": last_loss[0], "heldout_nll": heldout_nll(), "wall_s": time.time() - t0}
    log["records"].append(final)
    print(f"final: passes {passes}  train {final['train_loss']:.4f}  heldout {final['heldout_nll']:.4f}  {final['wall_s']:.0f}s")
    # passes-to-target table against the GD reference marks
    marks = [(1.698, 2500), (1.639, 5000), (1.588, 10000)]
    for target, gd_passes in marks:
        hit = next((r["passes"] for r in log["records"] if r["heldout_nll"] is not None and r["heldout_nll"] <= target), None)
        print(f"  heldout <= {target}: {args.optimizer} {hit if hit is not None else 'not reached'} passes  (GD: {gd_passes})")
    json.dump(log, open(os.path.join(args.out, "log.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
