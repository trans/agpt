#!/usr/bin/env python3
"""Tables of the doped-counts-depth16 runs: canonical rolling byte PPL at every
evaluated epoch from each run's result.json (fixed-window PPL in brackets),
seed pairs side by side with their mean in nats per character against the
hard-target arm. Writes runs.json.

Usage: python3 rnd/doped-counts-depth16/summarize.py
"""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARMS = ["mb16-a1m", "mb16-a1m-tt", "mb16-a1m-ref100m", "mb16-a1m-ref1g", "mb16-a1m-ref10g", "mb16-100m"]


def runs():
    out = {}
    for d in sorted(HERE.glob("2026*-*")):
        r = d / "result.json"
        if not r.is_file():
            continue
        res = json.loads(r.read_text())
        slug = res["run_id"].split("-", 1)[1]
        epochs = int((d / "config.yml").read_text().split("value:")[1].split()[0])
        curve = {}
        for c in res.get("checkpoint_results") or []:
            m = c.get("metrics") or {}
            curve[int(c["epoch"])] = [m.get("lm_eval_rolling_byte_perplexity"), m.get("agpt_fixed_token_perplexity")]
        m = res["metrics"]
        curve[epochs] = [m["lm_eval_rolling_byte_perplexity"], m.get("agpt_fixed_token_perplexity")]
        out[slug] = {"run": d.name, "split": (res.get("eval") or {}).get("split"), "curve": curve,
                     "train_wall": res.get("train_wall_seconds")}
    return out


def main():
    R = runs()
    (HERE / "runs.json").write_text(json.dumps(R, indent=1) + "\n")
    eps = sorted({e for v in R.values() for e in v["curve"]})
    print("Rolling byte PPL by epoch, seeds 42 / 43:\n")
    print("| arm | " + " | ".join(str(e) for e in eps) + " |")
    print("|---|" + "---:|" * len(eps))
    means = {}
    for arm in ARMS:
        cells = []
        for e in eps:
            v = [R[s]["curve"].get(e, [None])[0] for s in (arm, arm + "-s43") if s in R]
            v = [x for x in v if x]
            if v:
                means[(arm, e)] = sum(math.log(x) for x in v) / len(v)
            cells.append(" / ".join(f"{x:.3f}" for x in v))
        if any(cells):
            print(f"| {arm} | " + " | ".join(cells) + " |")
    print("\nMean log PPL (nats per char) against mb16-a1m at the same epoch:\n")
    for arm in ARMS[1:]:
        row = [f"{e}: {means[(arm, e)] - means[('mb16-a1m', e)]:+.4f}" for e in eps
               if (arm, e) in means and ("mb16-a1m", e) in means]
        if row:
            print(f"  {arm:18} " + "  ".join(row))


if __name__ == "__main__":
    main()
