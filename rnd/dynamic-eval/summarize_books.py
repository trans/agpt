#!/usr/bin/env python3
"""Summarize the whole-book runs (run_books.sh) -> books_summary.json and tables.

For each checkpoint: the own-past setting and the other-book setting with the
lowest NLL on the tune books (0-9), their gains over static on the report books
(10-19), and the gains by position (all 20 books, log-spaced bins).
Diagnostic fixed-window protocol, not canonical.
"""
import json
from pathlib import Path

D = Path(__file__).resolve().parent
out = {}
for slug in ("mb-a1m", "mb-100m-st100"):
    own = json.loads((D / f"results/books-{slug}-own.json").read_text())["settings"]
    oth = json.loads((D / f"results/books-{slug}-other.json").read_text())["settings"]
    st = own["static"]
    adapt = {k: v for k, v in own.items() if k != "static"}
    bo = min(adapt, key=lambda k: adapt[k]["nll_tune"])
    bc = min(oth, key=lambda k: oth[k]["nll_tune"])
    lora = {k: v for k, v in adapt.items() if k.startswith("lora")}
    bl = min(lora, key=lambda k: lora[k]["nll_tune"])
    bins = list(st["nll_by_bin"])
    r = {"static_report": st["nll_report"],
         "own": {"setting": bo, "report": adapt[bo]["nll_report"], "gain": st["nll_report"] - adapt[bo]["nll_report"]},
         "other": {"setting": bc, "report": oth[bc]["nll_report"], "gain": st["nll_report"] - oth[bc]["nll_report"]},
         "lora": {"setting": bl, "report": lora[bl]["nll_report"], "gain": st["nll_report"] - lora[bl]["nll_report"]},
         "by_bin": {b: {"static": st["nll_by_bin"][b],
                        "own_gain": st["nll_by_bin"][b] - adapt[bo]["nll_by_bin"][b],
                        "other_gain": st["nll_by_bin"][b] - oth[bc]["nll_by_bin"][b]} for b in bins}}
    r["own_minus_other"] = r["own"]["gain"] - r["other"]["gain"]
    r["lora_share"] = r["lora"]["gain"] / r["own"]["gain"]
    out[slug] = r
(D / "books_summary.json").write_text(json.dumps(out, indent=1) + "\n")

print("| checkpoint | static | own past: setting | gain | other book: setting | gain | own - other | best LoRA | gain | share |")
for s, r in out.items():
    print(f"| {s} | {r['static_report']:.4f} | {r['own']['setting']} | {r['own']['gain']:.4f} | {r['other']['setting']} | "
          f"{r['other']['gain']:.4f} | {r['own_minus_other']:.4f} | {r['lora']['setting']} | {r['lora']['gain']:.4f} | "
          f"{r['lora_share']:.0%} |")
print()
bins = list(next(iter(out.values()))["by_bin"])
print("| checkpoint | " + " | ".join(bins) + " |")
for s, r in out.items():
    print(f"| {s} | " + " | ".join(f"{v['own_gain']:.3f} / {v['other_gain']:.3f} / {v['own_gain'] - v['other_gain']:.3f}"
                                    for v in r["by_bin"].values()) + " |")
