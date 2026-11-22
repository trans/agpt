#!/usr/bin/env python3
"""Diagnostic: how much of the gradient noise sits at shallow depths of the trie?

Idea (depth-stratified AGPT): the full trie truncated at depth k is small
(distinct prefixes of length <= k), so its gradient - the exact full-corpus
gradient of every window's first k targets - is cheap. Sample windows only for
targets k+1..T. That estimator is unbiased and its noise is the deep part's
alone: B_noise(k) = tr(S_deep(k)) / |G|^2, against tr(S_total) / |G|^2 now.

Per window, the gradient is split into depth bands (one autograd.grad per band);
any cut k sums the bands above it. Also counts distinct prefixes of length <= k
over all window starts (the shallow trie's positions, its per-step cost).
Training text, exact PyTorch gradients. Not a reportable number.
Usage: depth_noise.py OUT.json VOCAB CORPUS CKPT...
"""
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/trans/Projects/air/agpt/src/tools")
from agpt_ppl import AGPTModel, build_vocab, load_model  # noqa: E402

torch.set_num_threads(2)
out, vocab, corpus, *ckpts = sys.argv[1:]
c2i, V = build_vocab(vocab)
text = Path(corpus).read_text(encoding="utf-8")
a = torch.tensor([c2i[c] for c in text], dtype=torch.long)
BANDS = ((1, 1), (2, 2), (3, 3), (4, 4), (5, 8), (9, 16))   # target depths, 1-based, inclusive
CUTS = (0, 1, 2, 3, 4, 8)
BATCHES = ((1024, 16), (8192, 6))
CHUNK = 4096
T = 16
N = a.numel() - T

res = {"corpus": corpus, "bands": BANDS, "batches": BATCHES, "ckpts": {}}
if Path(out).exists():
    res = json.loads(Path(out).read_text())
if "prefix_positions" not in res:
    pp, tot = {}, 0
    for d in range(1, 9):
        tot += len({text[s:s + d] for s in range(N)})
        pp[d] = tot
    res["prefix_positions"] = pp
    res["flat_positions_per_window"] = T
    Path(out).write_text(json.dumps(res, indent=1) + "\n")
print("distinct prefixes <= k (shallow trie positions):", res["prefix_positions"], flush=True)

offs = torch.arange(T + 1).view(1, -1)


def band_grads(model, params, starts):
    """Per-band sums over the given starts of the per-window gradient, flat float64: [len(BANDS), P]."""
    acc = None
    for s in range(0, starts.numel(), CHUNK):
        idx = starts[s:s + CHUNK]
        w = a[idx.view(-1, 1) + offs]
        logits = model(w[:, :-1])
        ce = F.cross_entropy(logits.reshape(-1, V), w[:, 1:].reshape(-1), reduction="none").view(-1, T)
        rows = []
        for bi, (lo, hi) in enumerate(BANDS):
            gs = torch.autograd.grad(ce[:, lo - 1:hi].sum(), params, retain_graph=bi + 1 < len(BANDS))
            rows.append(torch.cat([g.reshape(-1) for g in gs]).double())
        m = torch.stack(rows)
        acc = m if acc is None else acc + m
    return acc


for ckpt in ckpts:
    if ckpt in res["ckpts"]:
        continue
    t0 = time.time()
    cfg, sd = load_model(ckpt)
    assert cfg["seq_len"] == T
    model = AGPTModel(cfg, sd, device="cpu")
    params = list(model.parameters())
    Gb = band_grads(model, params, torch.arange(N)) / N          # [bands, P]
    G = Gb.sum(0)
    G2 = (G @ G).item()
    deep = lambda m, k: m[[i for i, (lo, _) in enumerate(BANDS) if lo > k]].sum(0)
    r = {"grad_norm": G2 ** 0.5,
         "band_grad_norm": [Gb[i].norm().item() for i in range(len(BANDS))], "by_batch": {}}
    gen = torch.Generator().manual_seed(1234)
    for B, K in BATCHES:
        acc = {k: 0.0 for k in CUTS}
        for _ in range(K):
            st = torch.randperm(N, generator=gen)[:B]
            gb = band_grads(model, params, st) / B
            for k in CUTS:
                d = deep(gb, k) - deep(Gb, k)
                acc[k] += (d @ d).item()
        fpc = (N - 1) / (N - B)
        r["by_batch"][B] = {str(k): acc[k] / K * B * fpc / G2 for k in CUTS}   # B_noise with exact depths <= k
    r["seconds"] = time.time() - t0
    res["ckpts"][ckpt] = r
    Path(out).write_text(json.dumps(res, indent=1) + "\n")
    bb = r["by_batch"]
    print(ckpt, f"|G| {G2 ** 0.5:.4g} bands |G_b| " + " ".join(f"{x:.3g}" for x in r["band_grad_norm"]), flush=True)
    for B, _ in BATCHES:
        b0 = bb[B]["0"]
        print(f"  B={B}: B_noise exact<=k: " + "  ".join(f"k{k} {bb[B][str(k)]:.0f} ({bb[B][str(k)] / b0:.0%})" for k in CUTS),
              flush=True)
    print(f"  ({r['seconds']:.0f} s)", flush=True)
