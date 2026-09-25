#!/usr/bin/env python3
"""Pool-convergence analysis (rnd/gradient-population/pool).

Members are independent SGD chains from one checkpoint, each with its own
random unit order. For every checkpoint epoch this reports:

  * weight-space: pairwise distance between members, distance of each
    member to the pool mean, displacement of the mean from theta_0, and
    the ratio spread / displacement (spread = RMS distance to mean)
  * per-section share of the spread
  * function-space: held-out fixed-window NLL per member and mean pairwise
    KL(p_a || p_b) over the same held-out targets (do members that differ in
    weight space differ as functions?)

Usage:
  python3 src/tools/agpt_pool_analysis.py rnd/gradient-population/pool --lr 0.02 \
      --heldout data/.splits/4fa9aec1db6b3aea/heldout_corpus.txt --vocab data/input.txt \
      [--positions 8192] [--out summary.json]
"""
import argparse
import glob
import json
import math
import os
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agpt_ppl  # noqa: E402


def flatten(sd):
    """Flatten the state dict in a fixed order; returns (vector, [(name, off, len)])."""
    names = sorted(sd.keys())
    parts, secs, off = [], [], 0
    for n in names:
        v = sd[n].detach().cpu().numpy().astype(np.float64).ravel()
        parts.append(v)
        secs.append((n, off, v.size))
        off += v.size
    return np.concatenate(parts), secs


def unflatten(vec, sd_like):
    """Inverse of flatten(): a new state dict with vec's values in sd_like's shapes."""
    out, off = {}, 0
    for n in sorted(sd_like.keys()):
        t = sd_like[n]
        k = t.numel()
        out[n] = torch.tensor(vec[off:off + k], dtype=t.dtype).view(t.shape)
        off += k
    return out


def member_dirs(root, lr):
    ds = sorted(glob.glob(os.path.join(root, f"lr{lr}-m*")))
    return [d for d in ds if os.path.isdir(d)]


def checkpoint_epochs(d):
    eps = []
    for f in glob.glob(os.path.join(d, "checkpoint.epoch_*.model")):
        m = re.search(r"epoch_(\d+)\.model$", f)
        if m:
            eps.append(int(m.group(1)))
    return sorted(eps)


def heldout_logprobs(model, tokens, d_window, n_positions, device, batch=512):
    """Log-probs (n_positions, V) at the last position of fixed windows, plus targets."""
    N = len(tokens)
    stride = max(1, (N - d_window) // n_positions)
    targets_idx = np.arange(d_window, N, stride)[:n_positions]
    tok = torch.tensor(tokens, dtype=torch.long, device=device)
    out = []
    with torch.no_grad():
        for s in range(0, len(targets_idx), batch):
            i_range = torch.tensor(targets_idx[s:s + batch], device=device)
            offsets = torch.arange(d_window, device=device).view(1, -1)
            ctx = tok[(i_range.view(-1, 1) - d_window) + offsets]
            logits = model(ctx)[:, -1, :]
            out.append(F.log_softmax(logits.double(), dim=-1))
    return torch.cat(out, 0), tok[torch.tensor(targets_idx, device=device)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--lr", required=True)
    ap.add_argument("--heldout", default="data/.splits/4fa9aec1db6b3aea/heldout_corpus.txt")
    ap.add_argument("--vocab", default="data/input.txt")
    ap.add_argument("--positions", type=int, default=8192)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-function-space", action="store_true")
    args = ap.parse_args()

    dirs = member_dirs(args.root, args.lr)
    if len(dirs) < 2:
        sys.exit(f"need >=2 members under {args.root} for lr {args.lr}")
    init_path = None
    for line in open(os.path.join(dirs[0], "config.yml")):
        if line.strip().startswith("init_file:"):
            init_path = line.split(":", 1)[1].strip()
    cfg0, sd0 = agpt_ppl.load_model(init_path)
    theta0, secs = flatten(sd0)
    epochs = sorted(set.intersection(*[set(checkpoint_epochs(d)) for d in dirs]))
    print(f"pool lr={args.lr}: {len(dirs)} members, common checkpoint epochs {epochs}")
    print(f"theta0 = {init_path}  ({theta0.size} params)")

    char_to_id, _ = agpt_ppl.build_vocab(args.vocab)
    tokens = [char_to_id[c] for c in open(args.heldout, encoding="utf-8").read() if c in char_to_id]
    device = "cpu"

    summary = {"lr": args.lr, "members": [os.path.basename(d) for d in dirs], "epochs": []}
    print("\nepoch  mean_disp   spread(RMS to mean)  spread/disp   pair_dist(min..max)   member NLL(min..max)   pair KL mean(min..max)   spread top sections")
    for ep in epochs:
        thetas, sds = [], []
        for d in dirs:
            cfg, sd = agpt_ppl.load_model(os.path.join(d, f"checkpoint.epoch_{ep:06d}.model"))
            v, _ = flatten(sd)
            thetas.append(v)
            sds.append((cfg, sd))
        T = np.stack(thetas)
        mean = T.mean(0)
        disp = np.linalg.norm(mean - theta0)
        to_mean = np.linalg.norm(T - mean, axis=1)
        spread = float(np.sqrt((to_mean ** 2).mean()))
        M = len(dirs)
        pair = [np.linalg.norm(T[a] - T[b]) for a in range(M) for b in range(a + 1, M)]
        # per-section share of the spread
        sec_share = []
        for name, off, ln in secs:
            s2 = ((T[:, off:off + ln] - mean[off:off + ln]) ** 2).sum()
            sec_share.append((name, float(s2 / ((T - mean) ** 2).sum())))
        sec_share.sort(key=lambda x: -x[1])
        entry = {"epoch": ep, "mean_displacement": float(disp), "spread_rms": spread,
                 "spread_over_disp": float(spread / disp) if disp > 0 else None,
                 "pair_dist_min": float(min(pair)), "pair_dist_max": float(max(pair)),
                 "member_to_mean": [float(x) for x in to_mean],
                 "spread_sections_top5": sec_share[:5]}
        nll_str, kl_str = "-", "-"
        if not args.no_function_space:
            lps, tgt = [], None
            for cfg, sd in sds:
                model = agpt_ppl.AGPTModel(cfg, sd, device=device).to(device).eval()
                lp, tgt = heldout_logprobs(model, tokens, cfg["seq_len"], args.positions, device)
                lps.append(lp)
            nlls = [float(-lp.gather(1, tgt.unsqueeze(1)).mean()) for lp in lps]
            kls = []
            for a in range(M):
                for b in range(M):
                    if a == b:
                        continue
                    kls.append(float((lps[a].exp() * (lps[a] - lps[b])).sum(-1).mean()))
            # the pool's weight-space mean as a model: is the center of the cloud a good model?
            cfg_m, sd_m = sds[0][0], unflatten(mean, sds[0][1])
            model_m = agpt_ppl.AGPTModel(cfg_m, sd_m, device=device).to(device).eval()
            lp_m, _ = heldout_logprobs(model_m, tokens, cfg_m["seq_len"], args.positions, device)
            nll_mean_model = float(-lp_m.gather(1, tgt.unsqueeze(1)).mean())
            kl_mean_to_members = [float((lp_m.exp() * (lp_m - lp)).sum(-1).mean()) for lp in lps]
            entry.update({"member_nll": nlls, "member_ppl": [math.exp(x) for x in nlls],
                          "pair_kl_mean": float(np.mean(kls)), "pair_kl_min": float(min(kls)),
                          "pair_kl_max": float(max(kls)),
                          "mean_model_nll": nll_mean_model,
                          "kl_meanmodel_to_members_mean": float(np.mean(kl_mean_to_members))})
            nll_str = f"{min(nlls):.4f}..{max(nlls):.4f} | mean-model {nll_mean_model:.4f}"
            kl_str = f"{np.mean(kls):.5f}({min(kls):.5f}..{max(kls):.5f}) | mean->members {np.mean(kl_mean_to_members):.5f}"
        summary["epochs"].append(entry)
        top = ", ".join(f"{n}:{s:.2f}" for n, s in sec_share[:3])
        print(f"{ep:5d}  {disp:9.4f}   {spread:9.4f}          {entry['spread_over_disp'] if disp > 0 else float('nan'):7.3f}   "
              f"{min(pair):.4f}..{max(pair):.4f}   {nll_str}   {kl_str}   {top}")

    # growth law: fit spread ~ epoch^alpha over the recorded epochs
    eps = np.array([e["epoch"] for e in summary["epochs"]], dtype=float)
    sp = np.array([e["spread_rms"] for e in summary["epochs"]])
    if len(eps) >= 3 and (sp > 0).all():
        alpha = np.polyfit(np.log(eps), np.log(sp), 1)[0]
        summary["spread_growth_exponent"] = float(alpha)
        print(f"\nspread ~ epoch^{alpha:.2f}  (0.5 = diffusive random walk, 1 = ballistic divergence, ~0 = saturated cloud)")
    out = args.out or os.path.join(args.root, f"summary-lr{args.lr}.json")
    json.dump(summary, open(out, "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
