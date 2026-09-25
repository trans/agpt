#!/usr/bin/env python3
"""Implicit-gradient-regularizer test (rnd/gradient-population, Experiment 5).

Backward-error analysis of SGD (Smith, Dherin, Barrett, De 2021): the mean
of random-order SGD with step eta follows gradient descent on

    L~(theta) = L(theta) + (eta / 4) * mean_i ||g_i(theta)||^2

where g_i is the per-example (per-window == chain of nested trie nodes)
gradient. The extra term's gradient is (eta/2) * mean_i H_i g_i, a
Hessian-vector product per example.

This script runs, from the same checkpoint theta_0 that a pool was run
from:

  A. full-batch GD on L            (aggregation as AGPT does it today)
  B. full-batch GD on L~           (aggregation + explicit regulariser)

both for the same total learning-rate mass as one pool member
(steps_member * eta_sgd == steps_gd * lr_gd), and compares the endpoints
to the pool members and the pool mean: weight-space distances and
held-out function-space KL / NLL.

Usage:
  python3 src/tools/agpt_igr_test.py --init CKPT.model --pool-dir rnd/gradient-population/pool-sgd/lr0.002 \
      --eta-sgd 0.002 --member-steps 100000 --lr-gd 0.5 --out DIR [--sub 2048] [--eps 1e-3]
"""
import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agpt_ppl  # noqa: E402


def load_tokens(path, char_to_id):
    return [char_to_id[c] for c in open(path, encoding="utf-8").read() if c in char_to_id]


def flat(params):
    return torch.cat([p.detach().reshape(-1) for p in params.values()]).double()


def window_loss(model, params, x, y):
    logits = functional_call(model, params, (x,))
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))


@torch.no_grad()
def heldout_logprobs(model, params, tok, targets_idx, W, batch=1024):
    out = []
    offsets = torch.arange(W, device=tok.device).view(1, -1)
    for s in range(0, len(targets_idx), batch):
        i_range = targets_idx[s:s + batch]
        ctx = tok[(i_range.view(-1, 1) - W) + offsets]
        logits = functional_call(model, params, (ctx,))[:, -1, :]
        out.append(F.log_softmax(logits.double(), dim=-1))
    return torch.cat(out, 0)


def full_gradient(model, params, tok, W, batch, device, n_sample=0, rng=None):
    """Mean over ALL windows (stride 1) of the mean causal CE: the aggregated gradient.
    With n_sample > 0, a random subset of n_sample windows per call instead."""
    N = tok.numel()
    n_win = N - W
    all_starts = None
    if n_sample > 0:
        all_starts = torch.tensor(rng.integers(0, n_win, size=n_sample), device=device)
        n_win = n_sample
    g = {k: torch.zeros_like(v) for k, v in params.items()}
    total = 0.0
    offsets = torch.arange(W + 1, device=device).view(1, -1)
    p = {k: v.detach().requires_grad_(True) for k, v in params.items()}
    for s in range(0, n_win, batch):
        starts = torch.arange(s, min(s + batch, n_win), device=device) if all_starts is None else all_starts[s:s + batch]
        win = tok[starts.view(-1, 1) + offsets]
        x, y = win[:, :-1], win[:, 1:]
        loss = window_loss(model, p, x, y) * len(starts)
        gs = torch.autograd.grad(loss, list(p.values()))
        for (k, _), gk in zip(p.items(), gs):
            g[k] += gk
        total += float(loss)
    for k in g:
        g[k] /= n_win
    return g, total / n_win


def penalty_gradient(model, params, tok, W, n_sub, eps, rng, device, chunk=256):
    """(1/2) * grad of mean_i ||g_i||^2  ==  mean_i H_i g_i, by finite-difference HVP
    along each window's own gradient. Also returns mean_i ||g_i||^2."""
    N = tok.numel()
    starts = torch.tensor(rng.integers(0, N - W - 1, size=n_sub), device=device)
    offsets = torch.arange(W + 1, device=device).view(1, -1)
    keys = list(params.keys())
    base = {k: v.detach() for k, v in params.items()}

    def loss_fn(p, x, y):
        logits = functional_call(model, p, (x.unsqueeze(0),))
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

    per_sample_grad = vmap(grad(loss_fn), in_dims=(None, 0, 0))
    per_sample_grad_pp = vmap(grad(loss_fn), in_dims=(0, 0, 0))  # per-sample params
    hvp_sum = {k: torch.zeros_like(v) for k, v in base.items()}
    sqnorm_sum = 0.0
    for s in range(0, n_sub, chunk):
        win = tok[starts[s:s + chunk].view(-1, 1) + offsets]
        x, y = win[:, :-1], win[:, 1:]
        gi = per_sample_grad(base, x, y)  # dict of (B, ...)
        B = x.size(0)
        sq = sum(gi[k].reshape(B, -1).pow(2).sum(1) for k in keys)  # (B,)
        sqnorm_sum += float(sq.sum())
        # perturb each sample's params along its own gradient, unit-length step of size eps
        nrm = sq.sqrt().clamp_min(1e-12)
        pp = {k: base[k].unsqueeze(0) + eps * gi[k] / nrm.view(-1, *([1] * gi[k].dim())[1:]) for k in keys}
        gi2 = per_sample_grad_pp(pp, x, y)
        for k in keys:
            # H_i (g_i/|g_i|) * |g_i|  = H_i g_i
            hv = (gi2[k] - gi[k]) / eps * nrm.view(-1, *([1] * gi[k].dim())[1:])
            hvp_sum[k] += hv.sum(0)
    return {k: v / n_sub for k, v in hvp_sum.items()}, sqnorm_sum / n_sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--pool-dir", required=True, help="dir with member*.pt from agpt_pool_sgd.py --save-members")
    ap.add_argument("--eta-sgd", type=float, required=True, help="the pool's SGD learning rate (sets the penalty coefficient)")
    ap.add_argument("--member-steps", type=int, required=True, help="SGD steps one pool member took (total steps / members)")
    ap.add_argument("--lr-gd", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", default="data/.splits/4fa9aec1db6b3aea/train_corpus.txt")
    ap.add_argument("--heldout", default="data/.splits/4fa9aec1db6b3aea/heldout_corpus.txt")
    ap.add_argument("--vocab", default="data/input.txt")
    ap.add_argument("--sub", type=int, default=2048, help="windows sampled per step for the penalty term")
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--full-windows", type=int, default=0, help="0 = every window (true full batch); N = N random windows per step (large-batch approximation)")
    ap.add_argument("--positions", type=int, default=8192)
    ap.add_argument("--record-every", type=int, default=0)
    ap.add_argument("--only", choices=["A", "B", "both"], default="both")
    ap.add_argument("--seed", type=int, default=11)
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

    model = agpt_ppl.AGPTModel(cfg, sd, device=dev).to(dev).eval()
    theta0 = {k: v.detach().clone() for k, v in model.named_parameters()}
    keys = list(theta0.keys())

    # pool members + mean
    members = []
    for f in sorted(os.listdir(args.pool_dir)):
        if f.startswith("member") and f.endswith(".pt"):
            sdm = torch.load(os.path.join(args.pool_dir, f), map_location=dev)
            members.append({k: sdm[k].detach() for k in keys})
    if not members:
        sys.exit("no member*.pt in pool dir")
    pool_mean = {k: torch.stack([m[k] for m in members]).mean(0) for k in keys}
    v0 = flat(theta0)
    vmean = flat(pool_mean)
    vmembers = [flat(m) for m in members]
    spread = float(torch.stack([(v - vmean).norm() for v in vmembers]).pow(2).mean().sqrt())
    lp_mean = heldout_logprobs(model, pool_mean, tok_h, targets_idx, W)
    lp_members = [heldout_logprobs(model, m, tok_h, targets_idx, W) for m in members]

    def nll(lp):
        return float(-lp.gather(1, tgt.unsqueeze(1)).mean())

    def kl(lpa, lpb):
        return float((lpa.exp() * (lpa - lpb)).sum(-1).mean())

    steps_gd = int(round(args.member_steps * args.eta_sgd / args.lr_gd))
    print(f"theta0 NLL {nll(heldout_logprobs(model, theta0, tok_h, targets_idx, W)):.4f} | pool: {len(members)} members, "
          f"mean disp {float((vmean - v0).norm()):.4f}, spread {spread:.4f}, mean-model NLL {nll(lp_mean):.4f}, "
          f"members NLL {min(nll(l) for l in lp_members):.4f}..{max(nll(l) for l in lp_members):.4f}")
    print(f"GD: lr {args.lr_gd} x {steps_gd} steps == lr-mass {args.member_steps * args.eta_sgd:.1f} (one member: {args.member_steps} x {args.eta_sgd})")
    print(f"penalty coefficient eta/4 = {args.eta_sgd / 4:.2e}; penalty gradient = (eta/2) * mean_i H_i g_i from {args.sub} windows/step\n")

    results = {"args": vars(args), "steps_gd": steps_gd, "pool": {"spread": spread, "mean_nll": nll(lp_mean),
               "member_nll": [nll(l) for l in lp_members], "mean_disp": float((vmean - v0).norm())}, "runs": {}}
    variants = {"A_plain": False, "B_penalised": True}
    for name, use_pen in variants.items():
        if args.only != "both" and not name.startswith(args.only):
            continue
        params = {k: v.clone() for k, v in theta0.items()}
        traj = []
        t0 = time.time()
        diverged = False
        for s in range(1, steps_gd + 1):
            g, train_loss = full_gradient(model, params, tok, W, args.batch, dev, args.full_windows, rng)
            pen_sq = None
            if use_pen:
                hv, pen_sq = penalty_gradient(model, params, tok, W, args.sub, args.eps, rng, dev)
                for k in keys:
                    g[k] = g[k] + (args.eta_sgd / 2.0) * hv[k]
            with torch.no_grad():
                for k in keys:
                    params[k] -= args.lr_gd * g[k]
            if not math.isfinite(train_loss) or train_loss > 10:
                print(f"  {name}: diverged at step {s} (train loss {train_loss})"); diverged = True; break
            if (args.record_every and s % args.record_every == 0) or s == steps_gd:
                v = flat(params)
                lp = heldout_logprobs(model, params, tok_h, targets_idx, W)
                rec = {"step": s, "lr_mass": s * args.lr_gd, "train_loss": train_loss, "heldout_nll": nll(lp),
                       "disp": float((v - v0).norm()), "dist_to_pool_mean": float((v - vmean).norm()),
                       "dist_to_members": [float((v - vm).norm()) for vm in vmembers],
                       "kl_to_pool_mean": kl(lp, lp_mean), "kl_pool_mean_to_this": kl(lp_mean, lp),
                       "kl_to_members_mean": float(np.mean([kl(lp, l) for l in lp_members])),
                       "mean_sq_grad_norm": pen_sq, "wall_s": time.time() - t0}
                traj.append(rec)
                print(f"  {name} step {s:5d}/{steps_gd} lr-mass {rec['lr_mass']:7.1f}  train {train_loss:.4f}  heldout {rec['heldout_nll']:.4f}  "
                      f"disp {rec['disp']:.3f}  |to pool mean| {rec['dist_to_pool_mean']:.3f}  |to members| {min(rec['dist_to_members']):.3f}..{max(rec['dist_to_members']):.3f}  "
                      f"KL(this->mean) {rec['kl_to_pool_mean']:.4f}  KL(this->members) {rec['kl_to_members_mean']:.4f}"
                      + (f"  mean|g_i|^2 {pen_sq:.3f}" if pen_sq is not None else "") + f"  {rec['wall_s']:.0f}s", flush=True)
        results["runs"][name] = {"diverged": diverged, "trajectory": traj}
        json.dump(results, open(os.path.join(args.out, "results.json"), "w"), indent=1)
        torch.save({k: v.cpu() for k, v in params.items()}, os.path.join(args.out, f"{name}.pt"))
    print("\nwrote", os.path.join(args.out, "results.json"))


if __name__ == "__main__":
    main()
