#!/usr/bin/env python3
"""Population / pool SGD on the AGPT attention model (rnd/gradient-population).

Per-node updates == plain SGD on (prefix, next-char) examples. A causal
window of W chars is W nested nodes (depths 1..W) in one step. No trie, no
partition depth, no pre-aggregation.

Pool dynamics: keep M models started from the same checkpoint. Each step
draws a member uniformly and a batch of random windows from the training
corpus, and takes one SGD step on that member only. Periodically snapshot:

  * weight space: mean displacement from theta_0, RMS spread of members
    around the pool mean, pairwise member distances, per-tensor share of
    the spread
  * function space (held-out fixed windows): NLL per member, NLL of the
    pool-mean model, mean pairwise KL between members, KL(mean -> members)

Usage:
  python3 src/tools/agpt_pool_sgd.py --init CKPT.model --out DIR \
      --members 4 --steps 200000 --lr 0.05 [--batch 1] [--window 16] \
      [--snap-every 20000] [--seed 1] [--positions 8192]
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agpt_ppl  # noqa: E402


def load_tokens(path, char_to_id):
    return [char_to_id[c] for c in open(path, encoding="utf-8").read() if c in char_to_id]


def params_vector(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()]).double()


def param_names(model):
    return [n for n, _ in model.named_parameters()]


def set_params_from_vector(model, vec):
    off = 0
    with torch.no_grad():
        for p in model.parameters():
            k = p.numel()
            p.copy_(vec[off:off + k].view_as(p).to(p.dtype))
            off += k


@torch.no_grad()
def heldout_logprobs(model, tok, targets_idx, W, batch=1024):
    model.eval()
    out = []
    offsets = torch.arange(W, device=tok.device).view(1, -1)
    for s in range(0, len(targets_idx), batch):
        i_range = targets_idx[s:s + batch]
        ctx = tok[(i_range.view(-1, 1) - W) + offsets]
        logits = model(ctx)[:, -1, :]
        out.append(F.log_softmax(logits.double(), dim=-1))
    model.train()
    return torch.cat(out, 0)


def snapshot(step, members, theta0, mean_model, tok_h, targets_idx, tgt, W, names, shapes):
    T = torch.stack([params_vector(m) for m in members])
    mean = T.mean(0)
    disp = float((mean - theta0).norm())
    to_mean = (T - mean).norm(dim=1)
    spread = float(to_mean.pow(2).mean().sqrt())
    M = len(members)
    pair = [float((T[a] - T[b]).norm()) for a in range(M) for b in range(a + 1, M)]
    # per-tensor share of spread
    shares, off = [], 0
    tot = float(((T - mean) ** 2).sum())
    for n, k in zip(names, shapes):
        shares.append((n, float(((T[:, off:off + k] - mean[off:off + k]) ** 2).sum() / tot)))
        off += k
    shares.sort(key=lambda x: -x[1])
    # function space
    lps = [heldout_logprobs(m, tok_h, targets_idx, W) for m in members]
    nlls = [float(-lp.gather(1, tgt.unsqueeze(1)).mean()) for lp in lps]
    kls = [float((lps[a].exp() * (lps[a] - lps[b])).sum(-1).mean()) for a in range(M) for b in range(M) if a != b]
    set_params_from_vector(mean_model, mean)
    lp_m = heldout_logprobs(mean_model, tok_h, targets_idx, W)
    nll_mean = float(-lp_m.gather(1, tgt.unsqueeze(1)).mean())
    kl_mean = [float((lp_m.exp() * (lp_m - lp)).sum(-1).mean()) for lp in lps]
    return {"step": step, "mean_displacement": disp, "spread_rms": spread,
            "spread_over_disp": spread / disp if disp > 0 else None,
            "pair_dist_min": min(pair) if pair else None, "pair_dist_max": max(pair) if pair else None,
            "member_nll": nlls, "mean_model_nll": nll_mean,
            "pair_kl_mean": float(np.mean(kls)) if kls else None,
            "pair_kl_min": min(kls) if kls else None, "pair_kl_max": max(kls) if kls else None,
            "kl_meanmodel_to_members_mean": float(np.mean(kl_mean)),
            "spread_sections_top5": shares[:5]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", default="data/.splits/4fa9aec1db6b3aea/train_corpus.txt")
    ap.add_argument("--heldout", default="data/.splits/4fa9aec1db6b3aea/heldout_corpus.txt")
    ap.add_argument("--vocab", default="data/input.txt")
    ap.add_argument("--members", type=int, default=4)
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--batch", type=int, default=1, help="windows per SGD step (1 = one chain of nested nodes)")
    ap.add_argument("--window", type=int, default=None, help="default: checkpoint seq_len")
    ap.add_argument("--snap-every", type=int, default=20000)
    ap.add_argument("--positions", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save-members", action="store_true")
    ap.add_argument("--recenter-every", type=int, default=0, help="every K steps set all members to the pool mean (0 = never)")
    ap.add_argument("--polyak-from", type=int, default=0, help="members=1: from this step keep a running iterate average; reported as the mean-model")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = args.device

    cfg, sd = agpt_ppl.load_model(args.init)
    W = args.window or cfg["seq_len"]
    char_to_id, _ = agpt_ppl.build_vocab(args.vocab)
    tok_tr = torch.tensor(load_tokens(args.train, char_to_id), dtype=torch.long, device=dev)
    tok_h = torch.tensor(load_tokens(args.heldout, char_to_id), dtype=torch.long, device=dev)
    N = tok_tr.numel()
    stride = max(1, (tok_h.numel() - W) // args.positions)
    targets_idx = torch.arange(W, tok_h.numel(), stride, device=dev)[:args.positions]
    tgt = tok_h[targets_idx]

    base = agpt_ppl.AGPTModel(cfg, sd, device=dev).to(dev)
    members = [copy.deepcopy(base).train() for _ in range(args.members)]
    mean_model = copy.deepcopy(base)
    theta0 = params_vector(base)
    names = param_names(base)
    shapes = [p.numel() for p in base.parameters()]
    print(f"pool-sgd: init={args.init} params={theta0.numel()} members={args.members} steps={args.steps} "
          f"lr={args.lr} batch={args.batch} window={W} device={dev} train_tokens={N}")

    log = {"args": vars(args), "snapshots": []}
    snap = snapshot(0, members, theta0, mean_model, tok_h, targets_idx, tgt, W, names, shapes)
    log["snapshots"].append(snap)
    print(f"step {0:8d}  disp {snap['mean_displacement']:8.4f}  spread {snap['spread_rms']:8.4f}  "
          f"member NLL {min(snap['member_nll']):.4f}..{max(snap['member_nll']):.4f}  mean-model {snap['mean_model_nll']:.4f}")

    offsets = torch.arange(W + 1, device=dev).view(1, -1)
    t0 = time.time()
    run_loss = 0.0
    polyak_sum, polyak_n = None, 0
    for step in range(1, args.steps + 1):
        m = int(rng.integers(args.members))
        starts = torch.tensor(rng.integers(0, N - W - 1, size=args.batch), device=dev)
        win = tok_tr[starts.view(-1, 1) + offsets]  # (B, W+1)
        x, y = win[:, :-1], win[:, 1:]
        model = members[m]
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        model.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                p.add_(p.grad, alpha=-args.lr)
        run_loss += float(loss)
        if args.polyak_from and step >= args.polyak_from:
            v = params_vector(members[0])
            polyak_sum = v.clone() if polyak_sum is None else polyak_sum + v
            polyak_n += 1
        if step % args.snap_every == 0 or step == args.steps:
            snap = snapshot(step, members, theta0, mean_model, tok_h, targets_idx, tgt, W, names, shapes)
            snap["train_loss_running"] = run_loss / args.snap_every
            snap["wall_s"] = time.time() - t0
            if polyak_n > 0:
                set_params_from_vector(mean_model, polyak_sum / polyak_n)
                lp_p = heldout_logprobs(mean_model, tok_h, targets_idx, W)
                snap["mean_model_nll"] = float(-lp_p.gather(1, tgt.unsqueeze(1)).mean())
                snap["polyak_n"] = polyak_n
            log["snapshots"].append(snap)
            run_loss = 0.0
            top = ", ".join(f"{n}:{s:.2f}" for n, s in snap["spread_sections_top5"][:3])
            fmt = lambda v: "   -  " if v is None else f"{v:.4f}"
            print(f"step {step:8d}  disp {snap['mean_displacement']:8.4f}  spread {snap['spread_rms']:8.4f}  "
                  f"s/d {fmt(snap['spread_over_disp'])}  member NLL {min(snap['member_nll']):.4f}..{max(snap['member_nll']):.4f}  "
                  f"mean-model {snap['mean_model_nll']:.4f}  pairKL {fmt(snap['pair_kl_mean'])}  "
                  f"KL(mean->m) {snap['kl_meanmodel_to_members_mean']:.4f}  train {snap['train_loss_running']:.4f}  "
                  f"[{top}]  {snap['wall_s']:.0f}s", flush=True)
            json.dump(log, open(os.path.join(args.out, "log.json"), "w"), indent=1)
        # re-centre AFTER the snapshot so the logged spread is the pre-reset cloud
        if args.recenter_every > 0 and step % args.recenter_every == 0 and step < args.steps:
            with torch.no_grad():
                Tm = torch.stack([params_vector(mm) for mm in members]).mean(0)
                for mm in members:
                    set_params_from_vector(mm, Tm)
    if args.save_members:
        for i, mdl in enumerate(members):
            torch.save(mdl.state_dict(), os.path.join(args.out, f"member{i}.pt"))
    sp = np.array([s["spread_rms"] for s in log["snapshots"][1:]])
    st = np.array([s["step"] for s in log["snapshots"][1:]], dtype=float)
    if len(sp) >= 3 and (sp > 0).all():
        alpha = float(np.polyfit(np.log(st), np.log(sp), 1)[0])
        log["spread_growth_exponent"] = alpha
        print(f"spread ~ step^{alpha:.2f}  (0.5 diffusive, 1 ballistic, ~0 saturated)")
    json.dump(log, open(os.path.join(args.out, "log.json"), "w"), indent=1)
    print("wrote", os.path.join(args.out, "log.json"))


if __name__ == "__main__":
    main()
