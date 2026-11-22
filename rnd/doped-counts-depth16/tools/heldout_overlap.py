#!/usr/bin/env python3
"""Diagnostic: how much of each held-out chunk occurs verbatim in a training text?

For each of the held-out chunks (heldout_chunks/chunk_*.txt), the fraction of
its positions whose next 32 and 64 characters occur somewhere in the text's
first `mark` characters, for each mark. Ordinary English repeats 32-character
strings now and then (quotations, set phrases); a chunk with a large share of
64-character matches is a book (or an edition of one) the text contains. Hashes
are 64-bit (prefix_keys' hash, built by doubling: h_2n[i] = h_n[i] * B^n + h_n[i+n]).
Not a reportable number.

Usage: heldout_overlap.py CHUNK_DIR VOCAB TEXT OUT.json --marks M,M,...
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src" / "tools"))
from agpt_ppl import build_vocab  # noqa: E402

B = np.uint64(0x9E3779B97F4A7C15)


def hashes(t, n):
    """{32: h32, 64: h64} for every start with n chars after it in t (t uint64 tokens + 1)."""
    h, length, out = t.copy(), 1, {}
    while length < 64:
        p = B ** np.uint64(length)
        h = h[:-length] * p + h[length:]
        length *= 2
        if length in (32, 64):
            out[length] = h
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chunks")
    ap.add_argument("vocab")
    ap.add_argument("text")
    ap.add_argument("out")
    ap.add_argument("--marks", required=True)
    ap.add_argument("--chunk", type=int, default=1 << 24)
    a = ap.parse_args()
    c2i, V = build_vocab(a.vocab)
    lut = np.full(256, 255, dtype=np.uint8)
    for ch, i in c2i.items():
        b = ch.encode()
        if len(b) == 1:
            lut[b[0]] = i
    files = sorted(Path(a.chunks).glob("chunk_*.txt"))
    keys, owner = {32: [], 64: []}, {32: [], 64: []}
    sizes = []
    for ci, f in enumerate(files):
        t = lut[np.frombuffer(f.read_bytes(), dtype=np.uint8)].astype(np.uint64) + np.uint64(1)
        hs = hashes(t, 64)
        for n in (32, 64):
            keys[n].append(hs[n])
            owner[n].append(np.full(hs[n].size, ci, dtype=np.int32))
        sizes.append({n: hs[n].size for n in (32, 64)})
    found = {}
    for n in (32, 64):
        k = np.concatenate(keys[n])
        o = np.argsort(k)
        keys[n], owner[n] = k[o], np.concatenate(owner[n])[o]
        found[n] = np.zeros(k.size, dtype=bool)
    ref = np.memmap(a.text, dtype=np.uint8, mode="r")
    marks = sorted(int(m) for m in a.marks.split(","))
    res = {"text": a.text, "chunks": [f.name for f in files], "marks": {}}
    t0, pos = time.time(), 0
    for mark in marks:
        while pos < mark - 64:
            e = min(pos + a.chunk, mark - 64)
            t = lut[np.asarray(ref[pos:e + 64])].astype(np.uint64) + np.uint64(1)
            hs = hashes(t, 64)
            for n in (32, 64):
                h = hs[n][:e - pos]
                u = keys[n]
                j = np.minimum(np.searchsorted(u, h), u.size - 1)
                found[n][j[u[j] == h]] = True   # one copy of an equal key; found_frac spreads it
            pos = e
        per = []
        for ci in range(len(files)):
            row = {}
            for n in (32, 64):
                row[f"frac{n}"] = float(found_frac(found[n], keys[n], owner[n], ci))
            per.append(row)
        res["marks"][str(mark)] = per
        worst = sorted(range(len(files)), key=lambda c: -per[c]["frac64"])[:5]
        print(f"  {mark:,} chars ({time.time() - t0:.0f}s): mean frac32 {np.mean([r['frac32'] for r in per]):.4f} "
              f"frac64 {np.mean([r['frac64'] for r in per]):.4f}; worst frac64 "
              + ", ".join(f"{files[c].name} {per[c]['frac64']:.3f}" for c in worst), flush=True)
        Path(a.out).write_text(json.dumps(res, indent=1) + "\n")


def found_frac(found, keys, owner, ci):
    """Fraction of chunk ci's positions whose key was seen (all copies of an equal key count)."""
    # propagate a hit to every position sharing the key
    run = np.r_[True, keys[1:] != keys[:-1]]
    gid = np.cumsum(run) - 1
    g_hit = np.zeros(gid[-1] + 1, dtype=bool)
    np.logical_or.at(g_hit, gid, found)
    m = owner == ci
    return g_hit[gid[m]].mean()


if __name__ == "__main__":
    main()
