#!/usr/bin/env python3
"""Table of the forward-backward coupling runs: rolling (canonical) and fixed-token
PPL at epochs 1, 2, 4, 8, and the final gap to mb-a1m-base in nats.
Usage: summarize.py [--json OUT]"""
import glob
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORDER = ["mb-a1m-base", "fb-b0-1", "fb-b1", "ff-b0-1", "ff-b1", "mb-a1m-rev"]
runs = {}
for d in sorted(glob.glob(str(HERE / "2026*"))):
    slug = Path(d).name.split("-", 1)[1]
    r = json.load(open(f"{d}/result.json"))
    curve = {int(c["epoch"]): (c["metrics"]["lm_eval_rolling_byte_perplexity"], c["metrics"]["agpt_fixed_token_perplexity"])
             for c in r.get("checkpoint_results") or []}
    m = r["metrics"]
    curve[8] = (m["lm_eval_rolling_byte_perplexity"], m["agpt_fixed_token_perplexity"])
    runs[slug] = {"run": Path(d).name, "curve": {e: curve[e] for e in sorted(curve)}}
base = runs["mb-a1m-base"]["curve"]
print(f"{'run':14} " + " ".join(f"{'ep' + str(e) + ' roll/fixed':>18}" for e in base) + "   gap@8 roll / fixed (nats)")
for s in ORDER:
    c = runs[s]["curve"]
    g = (math.log(c[8][0] / base[8][0]), math.log(c[8][1] / base[8][1]))
    runs[s]["gap8_nats"] = g
    print(f"{s:14} " + " ".join(f"{c[e][0]:8.4f}/{c[e][1]:.4f}".rjust(18) for e in c)
          + (f"   {g[0]:+.4f} / {g[1]:+.4f}" if s != "mb-a1m-rev" else "   (backward; reversed held-out)"))
if "--json" in sys.argv:
    Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps(runs, indent=1) + "\n")
