#!/usr/bin/env python3
"""Tables of the refined-trie-counts runs: canonical rolling byte PPL (and the
fixed-window PPL) at every evaluated pass, from each run directory's
result.json, with the train loss, steps and trie positions a pass from its
train.log when that is present (train.log is not committed; --json keeps them).

Usage: python3 rnd/refined-trie-counts/summarize.py [--json OUT]
"""
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORDER = ["agpt-a1m", "agpt-a1m-s2", "agpt-a1m-ref5m", "agpt-a1m-ref20m", "agpt-a1m-ref100m",
         "agpt-a1m-ref100m-s2", "agpt-5m", "mb-a1m", "mb-a1m-tt", "mb-a1m-ref100m", "mb-5m",
         "agpt-a1m-pd0", "agpt-a1m-pd0-ref100m", "agpt-a1m-pd1", "agpt-a1m-pd1-ref100m", "agpt-a1m-pd1-s2",
         "agpt-a1m-pd1-ref100m-s2", "agpt-a1m-pd6", "agpt-a1m-pd6-ref100m"]


def runs():
    out = {}
    for d in sorted(HERE.glob("2026*-*")):
        r = d / "result.json"
        if not r.is_file():
            continue
        res = json.loads(r.read_text())
        slug = res["run_id"].split("-", 1)[1]
        passes = int((d / "config.yml").read_text().split("value:")[1].split()[0])
        curve = {}
        for c in res.get("checkpoint_results") or []:
            m = c.get("metrics") or {}
            curve[int(c["epoch"])] = (m.get("lm_eval_rolling_byte_perplexity"), m.get("agpt_fixed_token_perplexity"))
        m = res["metrics"]
        curve[passes] = (m["lm_eval_rolling_byte_perplexity"], m.get("agpt_fixed_token_perplexity"))
        info = dict(run=d.name, split=(res.get("eval") or {}).get("split"), curve=curve,
                    train_wall=res.get("train_wall_seconds"), train={})
        log = d / "train.log"
        if log.is_file():
            t = log.read_text()
            g = re.search(r"steps_per_epoch=(\d+)", t)
            info["steps_per_pass"] = int(g.group(1)) if g else None
            g = re.search(r"trie positions per pass (\d+)", t)
            info["trie_positions_per_pass"] = int(g.group(1)) if g else None
            g = re.search(r"targets_per_epoch=(\d+)", t)
            info["events_per_pass"] = int(g.group(1)) if g else None
            for e, nll, soft in re.findall(r"epoch (\d+) train_nll=([\d.]+)(?:.*?soft_ce=([\d.]+))?", t):
                info["train"][int(e)] = (float(nll), float(soft) if soft else None)
            info["targets_line"] = next((ln.strip() for ln in t.splitlines() if "trie_targets" in ln), None)
        out[slug] = info
    return out


def main():
    R = runs()
    passes = sorted({p for v in R.values() for p in v["curve"]})
    print("Rolling byte PPL (fixed-window in brackets) by pass:\n")
    print("| run | " + " | ".join(str(p) for p in passes) + " | train wall s |")
    print("|---|" + "---:|" * len(passes) + "---:|")
    for s in ORDER + sorted(set(R) - set(ORDER)):
        if s not in R:
            continue
        v = R[s]
        cells = [f"{v['curve'][p][0]:.3f} ({v['curve'][p][1]:.3f})" if p in v["curve"] else "" for p in passes]
        print(f"| {s} | " + " | ".join(cells) + f" | {v['train_wall'] or 0:.0f} |")

    print("\nAgainst A's own counts, same order seed (nats of rolling byte PPL; negative = better;\n"
          "mb-a1m-tt: own counts as soft targets):\n")
    pairs = [(f"agpt-a1m-ref{r}{s}", f"agpt-a1m{s}") for r in ("5m", "20m", "100m") for s in ("", "-s2")]
    pairs += [("mb-a1m-tt", "mb-a1m"), ("mb-a1m-ref100m", "mb-a1m"), ("mb-a1m-ref100m", "mb-a1m-tt")]
    pairs += [(f"agpt-a1m-pd{k}-ref100m{s}", f"agpt-a1m-pd{k}{s}") for k in (0, 1, 6) for s in ("", "-s2")]
    print("| run | vs | " + " | ".join(str(p) for p in passes) + " |")
    print("|---|---|" + "---:|" * len(passes))
    for a, b in pairs:
        if a in R and b in R:
            cells = [f"{math.log(R[a]['curve'][p][0] / R[b]['curve'][p][0]):+.3f}"
                     if p in R[a]["curve"] and p in R[b]["curve"] else "" for p in passes]
            print(f"| {a} | {b} | " + " | ".join(cells) + " |")

    print("\nTrain loss (nats, on the run's own targets; soft cross-entropy in brackets) by pass:\n")
    tp = sorted({p for v in R.values() for p in v["train"]} & set(passes))  # the evaluated passes
    print("| run | steps/pass | trie positions/pass | " + " | ".join(str(p) for p in tp) + " |")
    print("|---|---:|---:|" + "---:|" * len(tp))
    for s in ORDER + sorted(set(R) - set(ORDER)):
        if s not in R or not R[s]["train"]:
            continue
        v = R[s]
        cells = [(f"{v['train'][p][0]:.3f}" + (f" ({v['train'][p][1]:.3f})" if v["train"][p][1] else ""))
                 if p in v["train"] else "" for p in tp]
        print(f"| {s} | {v.get('steps_per_pass') or ''} | {v.get('trie_positions_per_pass') or ''} | "
              + " | ".join(cells) + " |")
    if "--json" in sys.argv:
        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps(R, indent=1, default=str))


if __name__ == "__main__":
    main()
