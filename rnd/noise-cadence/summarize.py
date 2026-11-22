#!/usr/bin/env python3
"""Summarize rnd/noise-cadence runs -> runs.json and a table on stdout.

Compute is trie positions (edge characters of every gradient pass):
- fixed-cadence growth runs: the sum over stages of the stage's edge chars
  times its passes, from train.log;
- noise-cadence runs: the trainer's own count (both tries' whole recomputes
  after every step, plus each growth's new-window tries), from its last line.
A minibatch epoch on this split is 16.95M positions (one-pass-gap); minibatch
runs (mb-*) are counted that way, epochs x 16.95M. Steps:
optimizer updates. For noise runs also: the final sample (windows) and the
number of growths.
"""
import json
import re
from pathlib import Path

D = Path(__file__).resolve().parent
MB_EPOCH = 16.95e6
rows = []
for rd in sorted(D.glob("2026*-*")):
    res, log = rd / "result.json", rd / "train.log"
    if not res.exists():
        continue
    r = json.loads(res.read_text())
    cfg = (rd / "config.yml").read_text()
    slug = re.search(r"run_slug: (\S+)", cfg).group(1)
    m = r["metrics"]
    row = {"run": rd.name, "slug": slug, "rolling": m["lm_eval_rolling_byte_perplexity"],
           "fixed": m["agpt_fixed_token_perplexity"], "train_s": r["train_wall_seconds"]}
    text = log.read_text() if log.exists() else ""
    noise = re.search(r"completed noise-cadence optimizer_steps=(\d+) windows=(\d+) .*?positions=(\d+) "
                      r"full_recomputes=(\d+) growths=(\d+)", text)
    mb = re.search(r"agpt_minibatch_train: done steps=(\d+)", text)
    if mb:
        epochs = int(re.search(r"budget:\s+unit: epochs\s+value: (\d+)", cfg).group(1))
        row.update(steps=int(mb.group(1)), positions=int(epochs * MB_EPOCH))
    elif noise:
        row.update(steps=int(noise.group(1)), windows=int(noise.group(2)), positions=int(noise.group(3)),
                   growths=int(noise.group(5)))
    else:
        stages = re.findall(r"growth-stage (\d+)/(\d+): .*?edge_chars=(\d+)", text)
        epochs = re.findall(r"train-growth-stage: epochs=(\d+) units=(\d+)", text)
        row["stages"] = len(stages)
        row["positions"] = sum(int(e) * int(s[2]) for s, (e, _) in zip(stages, epochs))
        row["final_edge_chars"] = int(stages[-1][2]) if stages else None
        done = re.search(r"train-growth: completed stages=\d+ optimizer_steps=(\d+)", text)
        row["steps"] = int(done.group(1)) if done else None
    if row.get("positions"):
        row["mb_epochs"] = row["positions"] / MB_EPOCH
    rows.append(row)
(D / "runs.json").write_text(json.dumps({"runs": rows}, indent=1) + "\n")
print(f"{'slug':22} {'steps':>7} {'positions':>10} {'mb ep':>6} {'windows':>8} {'rolling':>8} {'fixed':>8} {'train s':>7}")
for row in sorted(rows, key=lambda x: x["slug"]):
    pos = row.get("positions")
    print(f"{row['slug']:22} {row.get('steps') or 0:7} {(pos or 0) / 1e6:9.2f}M {row.get('mb_epochs', 0):6.2f} "
          f"{row.get('windows', 0):8} {row['rolling']:8.4f} {row['fixed']:8.4f} {row['train_s']:7.0f}")
