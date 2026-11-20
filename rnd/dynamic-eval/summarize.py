#!/usr/bin/env python3
"""Summarize the dynamic-eval grid: for each checkpoint and past (own, other),
the setting with the lowest NLL on the tune half, read on the report half,
and its gain over static by 1,000-character bin (all 50 chunks).
Usage: summarize.py [--json OUT]"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CK = ["mb-a1m", "mb-a1m-ref100m", "mb-100m-st100", "agpt-pd1-ref100m"]
out = {}
print(f"{'checkpoint':18} {'static':>7} | {'own: best on tune':20} {'gain':>6} | {'other: best on tune':20} {'gain':>6} | own-other")
for ck in CK:
    row = {}
    for past in ("own", "other"):
        r = json.loads((HERE / f"results/{ck}-{past}.json").read_text())["settings"]
        st = r["static"]
        best = min((k for k in r if k != "static"), key=lambda k: r[k]["nll_tune"])
        b = r[best]
        row[past] = {"setting": best, "static_report": st["nll_report"], "report": b["nll_report"],
                     "gain_report": st["nll_report"] - b["nll_report"],
                     "gain_by_bin": {k: st["nll_by_bin"][k] - v for k, v in b["nll_by_bin"].items()}}
    o, x = row["own"], row["other"]
    row["own_minus_other"] = o["gain_report"] - x["gain_report"]
    out[ck] = row
    print(f"{ck:18} {o['static_report']:7.4f} | {o['setting']:20} {o['gain_report']:6.4f} | {x['setting']:20} "
          f"{x['gain_report']:6.4f} | {row['own_minus_other']:.4f}")
print("\ngain over static by position in the chunk (nats/char, all chunks; own / other / own-other)")
bins = list(out[CK[0]]["own"]["gain_by_bin"])
print(f"{'checkpoint':18} " + " ".join(f"{b:>17}" for b in bins))
for ck in CK:
    o, x = out[ck]["own"]["gain_by_bin"], out[ck]["other"]["gain_by_bin"]
    print(f"{ck:18} " + " ".join(f"{o[b]:.3f}/{x[b]:.3f}/{o[b] - x[b]:.3f}".rjust(17) for b in bins))
print("\nlow-rank (own past, report half): best LoRA setting on tune vs best full")
for ck in CK:
    r = json.loads((HERE / f"results/{ck}-own.json").read_text())["settings"]
    st = r["static"]["nll_report"]
    lo = min((k for k in r if k.startswith("lora")), key=lambda k: r[k]["nll_tune"])
    fu = min((k for k in r if k.startswith("full")), key=lambda k: r[k]["nll_tune"])
    print(f"{ck:18} {lo:18} {st - r[lo]['nll_report']:.4f}   {fu:16} {st - r[fu]['nll_report']:.4f}")
    out[ck]["lora_vs_full"] = {"lora": lo, "lora_gain": st - r[lo]["nll_report"], "full": fu,
                               "full_gain": st - r[fu]["nll_report"]}
if "--json" in sys.argv:
    Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps(out, indent=1) + "\n")
