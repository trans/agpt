#!/usr/bin/env python3
"""Check v2's noise-cadence gradient against PyTorch's on the same windows.

v2 (AGPT_NOISE_DUMP=PREFIX) writes PREFIX.grad.model (the first sample's mean
gradient per target, in the .model layout) and PREFIX.starts.txt (its window
starts). Here: the mean cross-entropy gradient of those windows (seq_len
targets each) at the same weights, in float64, compared per parameter.
Usage: grad_check.py PREFIX MODEL VOCAB CORPUS
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/trans/Projects/air/agpt/src/tools")
from agpt_ppl import AGPTModel, build_vocab, load_model  # noqa: E402

torch.set_num_threads(2)
prefix, model_path, vocab, corpus = sys.argv[1:]
c2i, V = build_vocab(vocab)
a = torch.tensor([c2i[c] for c in Path(corpus).read_text(encoding="utf-8")], dtype=torch.long)
lines = Path(prefix + ".starts.txt").read_text().splitlines()
head = lines[0].split()
events, loss_sum = float(head[2]), float(head[4])
starts = torch.tensor([int(x) for x in lines[1:]])

cfg, sd = load_model(model_path)
T = cfg["seq_len"]
model = AGPTModel(cfg, sd, device="cpu").double()
w = a[starts.view(-1, 1) + torch.arange(T + 1).view(1, -1)]
logits = model(w[:, :-1])
ce = F.cross_entropy(logits.reshape(-1, V), w[:, 1:].reshape(-1), reduction="sum")
(ce / (starts.numel() * T)).backward()
print(f"windows {starts.numel()}  events v2 {events:.0f} vs {starts.numel() * T}  "
      f"mean loss v2 {loss_sum / events:.8f} vs torch {ce.item() / (starts.numel() * T):.8f}")

gcfg, gsd = load_model(prefix + ".grad.model")
gmodel = AGPTModel(gcfg, gsd, device="cpu").double()
num = den = dot = nv2 = 0.0
rows = []
for (name, p), (_, q) in zip(model.named_parameters(), gmodel.named_parameters()):
    gt, gv = p.grad.reshape(-1), q.detach().reshape(-1)
    d = (gv - gt).norm().item()
    rows.append((name, gt.norm().item(), gv.norm().item(), d / max(gt.norm().item(), 1e-30),
                 (gv @ gt).item() / max(gv.norm().item() * gt.norm().item(), 1e-30)))
    num += d * d
    den += gt.norm().item() ** 2
    nv2 += gv.norm().item() ** 2
    dot += (gv @ gt).item()
print(f"overall: |g_torch| {den ** 0.5:.6g} |g_v2| {nv2 ** 0.5:.6g} rel_diff {(num / den) ** 0.5:.4g} "
      f"cos {dot / (den * nv2) ** 0.5:.6f}")
for name, nt, nv, rd, cs in rows:
    flag = "" if rd < 1e-3 else "   <--"
    print(f"  {name:28s} |torch| {nt:10.4g} |v2| {nv:10.4g} rel_diff {rd:9.3g} cos {cs:9.6f}{flag}")
