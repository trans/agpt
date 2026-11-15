#!/usr/bin/env python3
"""Tabulate the one-pass-gap runs from their result.json files.

Prints every run, then the best learning rate per arm and budget (by rolling
byte PPL), flagging a best value at the edge of its grid.
Usage: python3 rnd/one-pass-gap/summarize.py [--markdown]
"""
import json
import re
import sys

import yaml
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARMS = {"agpt": "AGPT pd=1 (65 steps/pass)", "mb": "minibatch 128 (8,279 steps/pass)",
        "mbL": "minibatch 16,302 (65 steps/pass)",
        "pd2c": "AGPT pd=2 + crown (1,463 steps/pass)", "pd3c": "AGPT pd=3 + crown (12,675 steps/pass)",
        "pd3cs": "AGPT pd=3 + crown, shuffled (12,675 steps/pass)",
        "pd4c": "AGPT pd=4 + crown (58,721 steps/pass)",
        "mb724": "minibatch 724 (1,464 steps/pass)", "mb84": "minibatch 84 (12,615 steps/pass)",
        "pd1s": "AGPT pd=1, fire_norm sum (65 steps/pass)",
        "pd3css": "AGPT pd=3 + crown, shuffled, fire_norm sum (12,675 steps/pass)",
        "mbu3": "random batches with pd=3's unit sizes (12,675 steps/pass)",
        "cv": "conventional epochs, 128 windows (518 steps/epoch; 'passes' = epochs)",
        "tp3": "replay of AGPT pd=3 + crown plan, shuffled (12,675 steps/pass)",
        "tq1338": "packed AGPT plan, pieces <= 1338 (12,675 equal steps, 97% sharing)",
        "tq335": "packed AGPT plan, pieces <= 335 (12,675 equal steps, 88% sharing)",
        "tq84": "packed AGPT plan, pieces <= 84 (12,675 equal steps, 65% sharing)",
        "cvt": "conventional epochs with trie targets (518 steps/epoch; 'passes' = epochs)",
        "mbt": "minibatch 128 with trie targets (8,279 steps/pass)"}
for _T in (32, 64, 128):
    ARMS[f"cv{_T}"] = f"conventional epochs at seq_len {_T}, {2048 // _T} windows (518 steps/epoch)"
    ARMS[f"cvt{_T}"] = f"conventional epochs at seq_len {_T} with trie targets, {2048 // _T} windows (518 steps/epoch)"


def load():
    rows = []
    for rj in sorted(HERE.glob("*/result.json")):
        r = json.loads(rj.read_text())
        # run ids are slugified (lower case), so the arm comes from config.yml's run_slug
        m = re.match(r"\d{8}T\d{6}-([a-z0-9]+)-e(\d+)-", r["run_id"])
        if not m or not r.get("metrics"):
            continue
        cfg = yaml.safe_load((rj.parent / "config.yml").read_text())
        rows.append(dict(arm=cfg["run_slug"].split("-")[0], epochs=int(m[2]), lr=float(cfg["train"]["optimizer"]["lr"]),
                         run=r["run_id"],
                         rolling=r["metrics"]["lm_eval_rolling_byte_perplexity"],
                         fixed=r["metrics"].get("agpt_fixed_token_perplexity"),
                         train_s=r["train_wall_seconds"], split=r["eval"]["split"]))
    return rows


def main():
    md = "--markdown" in sys.argv
    rows = load()
    grid = defaultdict(list)
    for r in rows:
        grid[(r["arm"], r["epochs"])].append(r)
    print(f"{len(rows)} runs; splits: {sorted({r['split'] for r in rows})}\n")
    if md:
        print("| Arm | Passes | Best lr | Rolling byte PPL | Fixed PPL | Train s | Other lrs (rolling) |")
        print("|---|---:|---:|---:|---:|---:|---|")
    for arm in ARMS:
        for e in sorted({k[1] for k in grid if k[0] == arm}):
            rs = sorted(grid[(arm, e)], key=lambda r: r["lr"])
            best = min(rs, key=lambda r: r["rolling"])
            edge = " (edge)" if best is rs[0] or best is rs[-1] else ""
            others = ", ".join(f"{r['lr']:g}: {r['rolling']:.3f}" for r in rs if r is not best)
            if md:
                print(f"| {ARMS[arm]} | {e} | {best['lr']:g}{edge} | {best['rolling']:.3f} | "
                      f"{best['fixed']:.3f} | {best['train_s']:.0f} | {others} |")
            else:
                print(f"{arm:5} e{e}  best lr {best['lr']:<7g}{edge:7} rolling {best['rolling']:.4f}  "
                      f"fixed {best['fixed']:.4f}  train {best['train_s']:7.1f}s   [{others}]")


if __name__ == "__main__":
    main()
