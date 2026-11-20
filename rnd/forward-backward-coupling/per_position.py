#!/usr/bin/env python3
"""Diagnostic: where the forward and backward models find the text hard.

Both teacher files hold, for every position p of the 1M training slice, a
distribution for A[p]: forward from the 8 characters before, backward from the
8 after. The chain rule says the two totals are the same (one text, one
probability, split two ways); this shows how differently the two split it.
Train text, not held-out: a diagnostic, not a reportable number.
Usage: per_position.py FORWARD.npy BACKWARD.npy CORPUS VOCAB
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "tools"))
from agpt_ppl import build_vocab  # noqa: E402

fwd, bwd = np.load(sys.argv[1], mmap_mode="r"), np.load(sys.argv[2], mmap_mode="r")
text = Path(sys.argv[3]).read_text(encoding="utf-8")
c2i, V = build_vocab(sys.argv[4])
a = np.array([c2i[c] for c in text], dtype=np.int64)
n = len(a)
f, b = np.empty(n), np.empty(n)
for s in range(0, n, 100_000):
    e = min(s + 100_000, n)
    r = np.arange(s, e)
    f[s:e] = -np.log(np.maximum(fwd[s:e][r - s, a[s:e]].astype(np.float64), 1e-12))
    b[s:e] = -np.log(np.maximum(bwd[s:e][r - s, a[s:e]].astype(np.float64), 1e-12))
keep = np.arange(1, n - 1)  # drop the two one-hot ends
f, b, t = f[keep], b[keep], np.array(list(text))[keep]
prev, nxt = np.array(list(text))[keep - 1], np.array(list(text))[keep + 1]
print(f"positions {len(keep)}  forward {f.mean():.4f}  backward {b.mean():.4f}  nats/char")
print(f"mean |forward - backward| per position {np.abs(f - b).mean():.4f}   corr {np.corrcoef(f, b)[0, 1]:.3f}")
print(f"positions where they differ by > 1 nat: {np.mean(np.abs(f - b) > 1):.1%}")
sp = np.isin(t, [" ", "\n"])
alpha = np.char.isalpha(t)
groups = {
    "space or newline": sp,
    "first letter of a word": alpha & np.isin(prev, [" ", "\n"]),
    "last letter of a word": alpha & ~np.char.isalpha(nxt) & ~np.isin(prev, [" ", "\n"]),
    "other letters (mid-word)": alpha & ~np.isin(prev, [" ", "\n"]) & np.char.isalpha(nxt),
    "punctuation, digits, etc.": ~sp & ~alpha,
}
tot_f, tot_b = f.sum(), b.sum()
print(f"\n{'target':28} {'share':>6} {'fwd':>6} {'bwd':>6}  {'fwd share':>9} {'bwd share':>9}")
for k, m in groups.items():
    print(f"{k:28} {m.mean():6.1%} {f[m].mean():6.3f} {b[m].mean():6.3f}  {f[m].sum() / tot_f:9.1%} {b[m].sum() / tot_b:9.1%}")
