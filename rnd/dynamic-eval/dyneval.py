#!/usr/bin/env python3
"""Dynamic evaluation: let the text already scored adapt the weights.

The user (2026-11-19): "the missing link is the deep past having some
influence in the present, like a LoRA on the trie." A window model sees only
its last seq_len characters. Dynamic evaluation (score a block, then take a
gradient step on the block just scored, so the weights carry everything
earlier in the document) is the plainest way for the deep past to act, and a
low-rank delta is the LoRA form of it.

Protocol: the fixed-window protocol of agpt_lm_eval's fixed-token PPL
(target i predicted from tokens[i-d:i], d = the model's seq_len, every chunk
of the held-out carve scored on its own, NLL pooled). The static setting is
exactly that number, so it must reproduce result.json's
agpt_fixed_token_perplexity. An adaptive setting scores each chunk in order
in blocks of `--block` targets: score the block with the current weights,
then take one optimizer step on the block's mean NLL (the same windows, the
same last-position targets: text already scored). Weights and optimizer state
are reset to the checkpoint at every chunk, so a chunk's past is only its own
earlier text.

Settings (--settings, comma-separated):
  static
  full-sgd-LR, full-adam-LR    every parameter adapts
  loraR-sgd-LR, loraR-adam-LR  a rank-R delta B @ A on every linear map (Q, K,
                               V, O, both FFN maps, the output head); B starts
                               at zero, A at N(0, 1/in) from a fixed seed
  loraRb-...                   the same, and every bias adapts too (the output
                               bias is a unigram cache)

The hyperparameters should be chosen on one half of the chunks (--tune, the
first half by default) and read on the other; the JSON keeps both halves and
the NLL by position in the chunk (1,000-character bins), the gain's growth
with the amount of past.

Control (--past other): the same steps, but chunk i's steps train on chunk
i+1's text at the same positions, so the weights learn held-out-like text
without this document's past. Own-past gain minus this gain is what the
document's own earlier text is worth.

Usage: dyneval.py --model CKPT --vocab VOCAB --chunks DIR --out OUT.json
                  [--settings static,full-adam-1e-3,...] [--block 64] [--past own|other]
Diagnostic only: not the canonical lm-eval protocol.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "tools"))
from agpt_ppl import AGPTModel, build_vocab, load_model  # noqa: E402

BIN = 1000


class LowRank(nn.Module):
    def __init__(self, out_f, in_f, r, gen):
        super().__init__()
        self.A = nn.Parameter(torch.randn(r, in_f, generator=gen) / math.sqrt(in_f))
        self.B = nn.Parameter(torch.zeros(out_f, r))

    def forward(self, W):
        return W + self.B @ self.A


def linears(model):
    out = [model.out_head]
    for layer in model.layers:
        out += [layer.wq, layer.wk, layer.wv, layer.wo, layer.l1, layer.l2]
    return out


def parse(s):
    if s == "static":
        return {"name": s, "kind": "static"}
    kind, opt, lr = s.split("-", 2)
    bias = kind.startswith("lora") and kind.endswith("b")
    r = int(kind[4:len(kind) - bias]) if kind.startswith("lora") else None
    if kind != "full" and r is None:
        raise ValueError(f"unknown setting {s!r}")
    if opt not in ("sgd", "adam"):
        raise ValueError(f"unknown optimizer in {s!r}")
    return {"name": s, "kind": "full" if r is None else "lora", "rank": r, "bias": bias, "opt": opt,
            "lr": float(lr)}


def score_chunk(cfg, sd, tokens, st, d, block, device, adapt_tokens=None):
    """Returns per-target NLL (float64 tensor) for one chunk under setting st.
    adapt_tokens: the text the steps train on (default the chunk itself); the
    control passes another chunk, stepped through in lockstep by position."""
    model = AGPTModel(cfg, sd, device="cpu").to(device)
    params = []
    if st["kind"] == "full":
        params = list(model.parameters())
    elif st["kind"] == "lora":
        for p in model.parameters():
            p.requires_grad_(False)
        gen = torch.Generator().manual_seed(0)
        for lin in linears(model):
            o, i = lin.weight.shape
            P.register_parametrization(lin, "weight", LowRank(o, i, st["rank"], gen).to(device))
            params += [lin.parametrizations.weight[0].A, lin.parametrizations.weight[0].B]
        if st["bias"]:
            for name, p in model.named_parameters():
                if name.endswith("bias"):
                    p.requires_grad_(True)
                    params.append(p)
    else:
        for p in model.parameters():
            p.requires_grad_(False)
    opt = None
    if params:
        opt = (torch.optim.SGD(params, lr=st["lr"]) if st["opt"] == "sgd"
               else torch.optim.Adam(params, lr=st["lr"]))
    t = torch.tensor(tokens, dtype=torch.long, device=device)
    a = t if adapt_tokens is None else torch.tensor(adapt_tokens, dtype=torch.long, device=device)
    N = t.numel()
    offs = torch.arange(d, device=device).view(1, -1)
    nll = torch.empty(N - d, dtype=torch.float64)
    model.eval()  # no dropout anywhere; eval() only for clarity
    for s in range(d, N, block):
        e = min(s + block, N)
        i = torch.arange(s, e, device=device)
        ctx, tgt = t[(i.view(-1, 1) - d) + offs], t[i]
        if opt is None:
            with torch.no_grad():
                lp = F.log_softmax(model(ctx)[:, -1, :], dim=-1)
                nll[s - d:e - d] = (-lp.gather(1, tgt.view(-1, 1)).squeeze(1)).double().cpu()
            continue
        lp = F.log_softmax(model(ctx)[:, -1, :], dim=-1)
        ev = -lp.gather(1, tgt.view(-1, 1)).squeeze(1)
        nll[s - d:e - d] = ev.detach().double().cpu()  # scored before the step below
        if adapt_tokens is not None:  # control: step on another chunk's block at the same positions
            ea = min(e, a.numel())
            if ea <= s:
                continue
            ia = torch.arange(s, ea, device=device)
            lpa = F.log_softmax(model(a[(ia.view(-1, 1) - d) + offs])[:, -1, :], dim=-1)
            ev = -lpa.gather(1, a[ia].view(-1, 1)).squeeze(1)
        opt.zero_grad(set_to_none=True)
        ev.mean().backward()
        opt.step()
    return nll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--settings", default="static")
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--tune", default="first-half", help="first-half: chunks [0, n/2) tune, the rest report")
    ap.add_argument("--max-chunks", type=int, default=0, help="smoke tests only")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--past", choices=["own", "other"], default="own",
                    help="other (control): chunk i's steps train on chunk i+1's text (mod n), not its own past")
    ap.add_argument("--bins", default="",
                    help="comma-separated bin edges in characters (e.g. 0,1000,10000,100000); default 1,000-wide")
    args = ap.parse_args()
    edges = [int(x) for x in args.bins.split(",")] if args.bins else None

    cfg, sd = load_model(args.model)
    d = cfg["seq_len"]
    c2i, V = build_vocab(args.vocab)
    files = sorted(Path(args.chunks).glob("chunk_*.txt"))
    if args.max_chunks:
        files = files[:args.max_chunks]
    chunks = []
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        chunks.append([c2i[c] for c in text if c in c2i])
    n_tune = len(chunks) // 2
    device = torch.device(args.device)
    res = {"model": args.model, "seq_len": d, "block": args.block, "past": args.past, "chunks": len(chunks),
           "tune_chunks": [0, n_tune], "report_chunks": [n_tune, len(chunks)], "settings": {}}
    for name in args.settings.split(","):
        st = parse(name)
        t0 = time.perf_counter()
        tot = {"tune": [0.0, 0], "report": [0.0, 0]}
        bins = {}
        for ci, toks in enumerate(chunks):
            other = chunks[(ci + 1) % len(chunks)] if args.past == "other" else None
            nll = score_chunk(cfg, sd, toks, st, d, args.block, device, other)
            half = "tune" if ci < n_tune else "report"
            tot[half][0] += nll.sum().item()
            tot[half][1] += nll.numel()
            pos = torch.arange(d, d + nll.numel())
            pos = pos // BIN if edges is None else torch.bucketize(pos, torch.tensor(edges[1:]), right=True)
            for b in pos.unique().tolist():
                m = pos == b
                acc = bins.setdefault(b, [0.0, 0])
                acc[0] += nll[m].sum().item()
                acc[1] += int(m.sum())
        all_nll = tot["tune"][0] + tot["report"][0]
        all_n = tot["tune"][1] + tot["report"][1]
        r = {"ppl_all": math.exp(all_nll / all_n), "nll_all": all_nll / all_n, "targets": all_n,
             "nll_tune": tot["tune"][0] / tot["tune"][1], "nll_report": tot["report"][0] / tot["report"][1],
             "nll_by_bin": {(f"{b * BIN}-{(b + 1) * BIN}" if edges is None else
                             f"{edges[b]}-{edges[b + 1] if b + 1 < len(edges) else 'end'}"): v[0] / v[1]
                            for b, v in sorted(bins.items())},
             "seconds": time.perf_counter() - t0}
        res["settings"][name] = r
        print(f"{name:20} ppl {r['ppl_all']:.4f}  nll tune {r['nll_tune']:.4f} report {r['nll_report']:.4f}  "
              f"bins " + " ".join(f"{v:.3f}" for v in r["nll_by_bin"].values()) + f"  ({r['seconds']:.0f}s)",
              flush=True)
        Path(args.out).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
