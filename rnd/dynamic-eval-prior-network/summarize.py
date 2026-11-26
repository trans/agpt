#!/usr/bin/env python3
"""Tables of rnd/dynamic-eval-prior-network: every run in out/ paired per book against the static run of its
checkpoint on the same books (mobius's pairing, pair_books.py: the pooled difference by characters, the per-book
mean ± its standard error, and the books improved), by position within the book, and own past against the
other-book control. Also the static 10.8M against mobius's own static losses (loss_stream.npy). Writes
summary.json. Bits per character throughout.

    python3 -I summarize.py
"""
import glob
import json
import math
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("DYN_OUT", HERE / "out"))
SPLIT = Path("/home/trans/Projects/air/mobius/data/pg19/imm-split-20m-books")
MOBIUS_STATIC = Path("/home/trans/.cache/tmp/claude/claude-1000/-home-trans-Projects-air-mobius/"
                     "977a5e13-797d-4dd7-8bf0-a6c47bdf75b9/scratchpad/rescore/loss_stream.npy")
EDGES = [0, 1000, 5000, 25000, 100000, 300000, 10**9]
L2 = math.log(2)

sizes = np.array([r["size"] for r in json.loads((SPLIT / "manifest.json").read_text())["chunk_records"]])
bounds = np.concatenate([[0], np.cumsum(sizes)])
n, nb = int(bounds[-1]), len(sizes)
book = np.repeat(np.arange(nb), sizes)
posin = np.arange(n) - bounds[book]


def load(name):
    return np.load(OUT / f"{name}.npy", mmap_mode="r")


def paired(a, b, books):
    """b against a (bits) on the positions both scored in `books`."""
    sel = np.isin(book, books) & ~np.isnan(a) & ~np.isnan(b)
    da, db = np.asarray(a[sel], dtype=np.float64) / L2, np.asarray(b[sel], dtype=np.float64) / L2
    bk = book[sel]
    cnt = np.bincount(bk, None, nb)[books]
    d = (np.bincount(bk, db, nb)[books] - np.bincount(bk, da, nb)[books]) / cnt
    bands = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = (posin[sel] >= lo) & (posin[sel] < hi)
        if m.any():
            bands.append({"band": [lo, hi], "share": float(m.mean()), "a": float(da[m].mean()), "b": float(db[m].mean()),
                          "diff": float(db[m].mean() - da[m].mean())})
    return {"a": float(da.mean()), "b": float(db.mean()), "by_chars": float(db.mean() - da.mean()),
            "per_book": float(d.mean()), "se": float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else None,
            "better": int((d < 0).sum()), "books": len(d), "chars": int(sel.sum()), "bands": bands,
            "book_diff": {int(i): float(x) for i, x in zip(books, d)}}


def fmt(p):
    se = f" ± {p['se']:.4f}" if p["se"] is not None else ""
    return (f"{p['b']:.4f} vs {p['a']:.4f}: {p['by_chars']:+.4f} by chars; per book {p['per_book']:+.4f}{se}, "
            f"better in {p['better']}/{p['books']}")


def main():
    runs = {}
    for f in sorted(glob.glob(str(OUT / "*.json"))):
        r = json.loads(Path(f).read_text())
        runs[Path(f).stem] = r
    res = {"runs": {}, "checks": {}}
    # each adaptive run pairs with the static run of the same checkpoint
    statics = {r["checkpoint_sha256"]: name for name, r in runs.items() if r["steps"] == 0 and not r.get("limit")}
    if "static10" in runs and MOBIUS_STATIC.exists():
        p = paired(np.load(MOBIUS_STATIC, mmap_mode="r"), load("static10"), list(range(nb)))
        res["checks"]["static10_vs_mobius"] = {k: p[k] for k in ("a", "b", "by_chars", "chars")}
        print(f"static 10.8M (agpt, batch 1) vs mobius's score_books.py: {p['b']:.4f} vs {p['a']:.4f} "
              f"({p['by_chars']:+.5f}) over {p['chars']:,} characters")
    for name, r in runs.items():
        if r["steps"] == 0 or r.get("limit"):
            if r["steps"] == 0:
                print(f"{name}: static, {r['bits']:.4f} bits/char over {len(r['books'])} books "
                      f"({r['seconds']:.0f} s, peak GPU {r.get('peak_gpu_mb')} MB)")
            continue
        static = statics.get(r["checkpoint_sha256"])
        if static is None:
            continue
        p = paired(load(static), load(name), r["books"])
        res["runs"][name] = {"setting": r["setting"]["name"], "past": r["past"], "static": static, "seconds": r["seconds"],
                             "steps": r["steps"], "peak_gpu_mb": r.get("peak_gpu_mb"), **p}
        print(f"{name} ({r['setting']['name']}, past {r['past']}, {r['seconds']:.0f} s): {fmt(p)}")
    label = {"10": "10.8M", "25": "25.8M", "10h": "10.8M on 100M", "25h": "25.8M on 100M"}
    # every report pair: rNET-own[-ARM] against rNET-other[-ARM]
    for own in [k for k in runs if k.startswith("r") and "-own" in k]:
        net, other = own[1:].split("-own")[0], own.replace("-own", "-other")
        if own in runs and other in runs:
            p = paired(load(other), load(own), runs[own]["books"])
            res["runs"][f"{own}-minus-other"] = p
            print(f"\n{label[net]} ({runs[own]['setting']['name']}), report books: own past {res['runs'][own]['by_chars']:+.4f}, "
                  f"other-book control {res['runs'][other]['by_chars']:+.4f}; own vs control: {fmt(p)}")
            print("  by position in the book: band, share, static -> own (gain), control gain, own minus control")
            ps, po = res["runs"][own]["bands"], res["runs"][other]["bands"]
            for bo, bc, bd in zip(ps, po, p["bands"]):
                print(f"    [{bo['band'][0]:>7d},{bo['band'][1]:>10d}) {bo['share']:.3f} {bo['a']:.4f} -> {bo['b']:.4f} "
                      f"({bo['diff']:+.4f}), control {bc['diff']:+.4f}, own - control {bd['diff']:+.4f}")
    (HERE / "summary.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
