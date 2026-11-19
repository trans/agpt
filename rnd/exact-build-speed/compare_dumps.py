#!/usr/bin/env python3
"""Compare frozen-weight gradient dumps (AGPT_GRAD_DUMP_DIR) between builds.

Usage: compare_dumps.py WORKDIR OUT.json A:B [A:B ...]
Each name is a dump directory under WORKDIR (grads.f32, units.tsv,
layout.json). For each pair it reports the whole-pass gradient (the
event-weighted mean of the unit gradients) relative error |a - b| / |b| and
cosine, and the per-unit relative errors (median, 90th percentile, max), with
b as the reference.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np


def load(d):
    lay = json.loads((d / "layout.json").read_text())
    rows = list(csv.DictReader(open(d / "units.tsv"), delimiter="\t"))
    X = np.fromfile(d / "grads.f32", dtype=np.float32).reshape(len(rows), lay["total_floats"]).astype(np.float64)
    ev = np.array([float(r["trained_events"]) for r in rows])
    key = [(r["anchor_id"], r["anchor_endpoint_depth"]) for r in rows]
    return X, ev, key


W, out = Path(sys.argv[1]), sys.argv[2]
res = {}
for pair in sys.argv[3:]:
    a, b = pair.split(":")
    Xa, ea, ka = load(W / a)
    Xb, eb, kb = load(W / b)
    assert ka == kb and np.array_equal(ea, eb), f"{a} and {b} dumped different units"
    ga, gb = (Xa * ea[:, None]).sum(0) / ea.sum(), (Xb * eb[:, None]).sum(0) / eb.sum()
    whole = float(np.linalg.norm(ga - gb) / np.linalg.norm(gb))
    cos = float(ga @ gb / np.linalg.norm(ga) / np.linalg.norm(gb))
    nb = np.linalg.norm(Xb, axis=1)
    per = np.linalg.norm(Xa - Xb, axis=1) / np.where(nb > 0, nb, 1)
    r = {"units": len(ka), "whole_rel_err": whole, "whole_cos": cos,
         "unit_rel_err_median": float(np.median(per)), "unit_rel_err_p90": float(np.percentile(per, 90)),
         "unit_rel_err_max": float(per.max())}
    res[pair] = r
    print(f"{pair:28} whole {whole:.2e} (cos {cos:.6f}); per unit median {r['unit_rel_err_median']:.2e}, "
          f"p90 {r['unit_rel_err_p90']:.2e}, max {r['unit_rel_err_max']:.2e} ({len(ka)} units)", flush=True)
Path(out).write_text(json.dumps(res, indent=1) + "\n")
