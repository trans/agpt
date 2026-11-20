#!/usr/bin/env python3
"""Summarize rnd/trie-growth runs -> runs.json and a table on stdout.

Compute is trie positions: for growth runs, the sum over stages of the stage's
edge characters (the radix trie's positions) times its passes, from train.log;
for static runs, passes times the full trie's edge characters, taken from the
one-stage growth run's log (the same trie). A minibatch epoch on this split is
16.95M positions (one-pass-gap). Steps: optimizer updates.
"""
import json
import re
from pathlib import Path

D = Path(__file__).resolve().parent
MB_EPOCH = 16.95e6
rows, full_edge = [], None
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
    stages = re.findall(r"growth-stage (\d+)/(\d+): .*?edge_chars=(\d+)", text)
    epochs = re.findall(r"train-growth-stage: epochs=(\d+) units=(\d+)", text)
    if stages:
        row["stages"] = len(stages)
        row["positions"] = sum(int(e) * int(s[2]) for s, (e, _) in zip(stages, epochs))
        done = re.search(r"train-growth: completed stages=\d+ optimizer_steps=(\d+)", text)
        row["steps"] = int(done.group(1)) if done else None
        exact = re.search(r"growth-exact-schedule: total_unit_steps=(\d+)", text)
        row["scheduled_steps"] = int(exact.group(1)) if exact else None
        if len(stages) == 1:
            full_edge = int(stages[0][2])
    else:
        st = re.search(r"train-epoch: epochs=(\d+) units=(\d+)", text)
        if st:
            row["passes"], row["steps"] = int(st.group(1)), int(st.group(1)) * int(st.group(2))
    rows.append(row)
for row in rows:
    if "passes" in row and full_edge:
        row["positions"] = row["passes"] * full_edge
    if row.get("positions"):
        row["mb_epochs"] = row["positions"] / MB_EPOCH
(D / "runs.json").write_text(json.dumps({"full_trie_positions": full_edge, "runs": rows}, indent=1) + "\n")
print(f"full trie: {full_edge} positions a pass")
print(f"{'slug':28} {'steps':>7} {'positions':>10} {'mb ep':>6} {'rolling':>8} {'fixed':>8} {'train s':>7}")
for row in sorted(rows, key=lambda x: x["slug"]):
    pos = row.get("positions")
    print(f"{row['slug']:28} {row.get('steps') or 0:7} {(pos or 0) / 1e6:9.2f}M {row.get('mb_epochs', 0):6.2f} "
          f"{row['rolling']:8.4f} {row['fixed']:8.4f} {row['train_s']:7.0f}")
