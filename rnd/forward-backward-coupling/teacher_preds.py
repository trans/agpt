#!/usr/bin/env python3
"""A teacher model's next-character distribution at every position of a text.

Usage: teacher_preds.py --model CKPT --vocab VOCAB --corpus A.txt --direction forward|backward --out P.npy

forward:  P[p] = the model's distribution for A[p] given A[p-d:p] (d = its seq_len;
          fewer characters near the start).
backward: the model was trained on reversed text; P[p] = its distribution for A[p]
          given the d characters that FOLLOW it, A[p+1:p+1+d], read in reverse
          (fewer near the end).
P[p] for the one position with no context at all (p = 0 forward, p = n-1 backward)
is one-hot on A[p], so a coupling term there adds nothing. Stored float16, shape
(n, V); rows renormalized after the cast. Prints the file's SHA-256 and the
teacher's mean NLL of A under P.
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "tools"))
from agpt_ppl import AGPTModel, build_vocab, load_model  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--vocab", required=True)
ap.add_argument("--corpus", required=True)
ap.add_argument("--direction", choices=["forward", "backward"], required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--batch", type=int, default=8192)
ap.add_argument("--device", default="cpu")
a = ap.parse_args()
cfg, sd = load_model(a.model)
d = cfg["seq_len"]
c2i, V = build_vocab(a.vocab)
text = Path(a.corpus).read_text(encoding="utf-8")
if any(c not in c2i for c in set(text)):
    sys.exit("corpus has characters outside the vocab")
A = np.array([c2i[c] for c in text], dtype=np.int64)
x = A if a.direction == "forward" else A[::-1].copy()
n = x.size
dev = torch.device(a.device)
model = AGPTModel(cfg, sd, device="cpu").to(dev).eval()
xt = torch.from_numpy(x).to(dev)
P = np.zeros((n, V), dtype=np.float16)
P[0, x[0]] = 1.0
with torch.no_grad():
    for q in range(1, min(d, n)):  # short contexts at the start
        P[q] = F.softmax(model(xt[None, :q])[0, -1].float(), -1).cpu().numpy()
    offs = torch.arange(d, device=dev).view(1, -1)
    for s in range(d, n, a.batch):
        e = min(s + a.batch, n)
        q = torch.arange(s, e, device=dev)
        P[s:e] = F.softmax(model(xt[(q.view(-1, 1) - d) + offs])[:, -1].float(), -1).cpu().numpy()
P = P.astype(np.float32)
P /= P.sum(1, keepdims=True)
P = P.astype(np.float16)
if a.direction == "backward":
    P = P[::-1].copy()
np.save(a.out, P)
nll = -np.log(np.maximum(P[np.arange(n), A].astype(np.float64), 1e-12)).mean()
print(f"{hashlib.sha256(Path(a.out).read_bytes()).hexdigest()}  {a.out}  ({n} x {V}, teacher NLL of the text {nll:.4f})")
