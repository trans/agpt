#!/usr/bin/env python3
"""Tables of the attention grouping control: canonical rolling byte PPL (and
the fixed-window AGPT PPL) at every evaluated pass, per arm, rate and order
seed, from the run directories' result.json (checkpoint_results + final).

Usage: python3 rnd/context-tree-orientation/attention/summarize.py [--json OUT]
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = ["first", "window", "last"]


def runs():
    out = {}
    for d in sorted(ROOT.glob("2026*-att-*")):
        r = d / "result.json"
        if not r.is_file():
            continue
        res = json.loads(r.read_text())
        slug = res["run_id"].split("-", 1)[1]  # att-first-lr0-003[-s2]
        _, arm, rest = slug.split("-", 2)
        seed = 2 if rest.endswith("-s2") else 1
        lr = float(rest.replace("-s2", "")[2:].replace("-", "."))
        cfg = (d / "config.yml").read_text()
        passes = int(cfg.split("value:")[1].split()[0])
        curve = {}
        for c in res.get("checkpoint_results") or []:
            m = c.get("metrics") or {}
            curve[int(c["epoch"])] = (m.get("lm_eval_rolling_byte_perplexity"), m.get("agpt_fixed_token_perplexity"))
        m = res["metrics"]
        curve[passes] = (m["lm_eval_rolling_byte_perplexity"], m.get("agpt_fixed_token_perplexity"))
        out[(arm, lr, seed)] = dict(run=d.name, curve=curve, train_wall=res.get("train_wall_seconds"))
    return out


def main():
    R = runs()
    passes = sorted({p for v in R.values() for p in v["curve"]})
    print("Rolling byte PPL (fixed-window in brackets) by pass:\n")
    print("| arm | lr | seed | " + " | ".join(str(p) for p in passes) + " |")
    print("|---|---:|---:|" + "---:|" * len(passes))
    for arm in ARMS:
        for (a, lr, s), v in sorted(R.items(), key=lambda kv: (ARMS.index(kv[0][0]), kv[0][1], kv[0][2])):
            if a != arm:
                continue
            cells = []
            for p in passes:
                x = v["curve"].get(p)
                cells.append(f"{x[0]:.3f} ({x[1]:.3f})" if x and x[1] else (f"{x[0]:.3f}" if x else ""))
            print(f"| {arm} | {lr:g} | {s} | " + " | ".join(cells) + " |")
    print("\nBest rate per arm and pass (seed 1), and the gap to `first` in nats of rolling byte PPL:\n")
    print("| pass | " + " | ".join(ARMS) + " | last - first | first - window |")
    print("|---:|" + "---|" * len(ARMS) + "---:|---:|")
    for p in passes:
        best = {}
        for arm in ARMS:
            c = [(v["curve"][p][0], lr) for (a, lr, s), v in R.items() if a == arm and s == 1 and p in v["curve"]]
            if c:
                best[arm] = min(c)
        cells = [f"{best[a][0]:.3f} @{best[a][1]:g}" if a in best else "" for a in ARMS]
        g1 = f"{math.log(best['last'][0] / best['first'][0]):+.3f}" if "last" in best and "first" in best else ""
        g2 = f"{math.log(best['first'][0] / best['window'][0]):+.3f}" if "first" in best and "window" in best else ""
        print(f"| {p} | " + " | ".join(cells) + f" | {g1} | {g2} |")
    cos = {k: v for k, v in R.items() if k[0] in ("cfirst", "cwindow")}
    if cos:
        cp = sorted({p for v in cos.values() for p in v["curve"]})
        print("\nCosine check (warmup-cosine over 16 passes, peak rate), rolling byte PPL (fixed-window):\n")
        print("| arm | peak lr | " + " | ".join(str(p) for p in cp) + " |")
        print("|---|---:|" + "---:|" * len(cp))
        for (a, lr, s), v in sorted(cos.items()):
            cells = [f"{v['curve'][p][0]:.3f} ({v['curve'][p][1]:.3f})" if p in v["curve"] else "" for p in cp]
            print(f"| {a[1:]} | {lr:g} | " + " | ".join(cells) + " |")
    if "--json" in sys.argv:
        out = {f"{a}|{lr:g}|{s}": v for (a, lr, s), v in R.items()}
        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
