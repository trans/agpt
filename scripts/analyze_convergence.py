#!/usr/bin/env python3
"""Convergence-dynamics analysis from AGPT/SGD training logs.

Static metrics like final loss and held-out PPL leave out the most
important question: how much compute did it take to get there?

For each AGPT log this reports:
  - Total wall, total optimizer steps, final loss
  - Time-to-loss-X for X in a configurable list (default: 5.0/3.0/2.0/1.7/1.5/1.3)
  - Loss reduction per wall-second (mloss/sec — milli-loss per second)
  - Loss reduction per optimizer step (μloss/step — micro-loss per step)
  - Saturation indicator: |loss[N] - loss[N-1]| in the final epoch
For SGD logs it reports loss-target hits at sampled step lines.

Usage:
    python3 scripts/analyze_convergence.py LOG [LOG ...]
    python3 scripts/analyze_convergence.py --targets 4,3,2,1.7  LOG

Pass any agpt_train log files (per-epoch lines) or microgpt SGD logs
(per-step "loss = X avg = Y" lines). Auto-detects format by content.
"""
import argparse
import re
from pathlib import Path

EPOCH_RE = re.compile(r"^Epoch (\d+): loss=([\d.]+)\s+\((\d+\.\d+) sec, (\d+) subtrees,")
STEP_RE  = re.compile(r"^Step (\d+)/\d+ \(epoch \d+\): loss = [\d.]+ avg = ([\d.]+)")


def parse_agpt(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            m = EPOCH_RE.match(line)
            if m:
                ep, loss, sec, sub = (
                    int(m.group(1)), float(m.group(2)),
                    float(m.group(3)), int(m.group(4))
                )
                rows.append((ep, loss, sec, sub))
    return rows


def parse_sgd(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            m = STEP_RE.match(line)
            if m:
                rows.append((int(m.group(1)), float(m.group(2))))
    return rows


def first_le(rows, key, target):
    """Return first row where row[key] <= target, with cumulative wall, or (None, None)."""
    cum = 0.0
    for r in rows:
        cum += r[2] if len(r) > 2 else 0.0
        if r[key] <= target:
            return r, cum
    return None, None


def report_agpt(name, rows, targets):
    if not rows:
        print(f"  {name}: empty"); return
    total_sec = sum(r[2] for r in rows)
    n_subtrees = rows[0][3]
    n_epochs = rows[-1][0]
    total_steps = n_subtrees * n_epochs
    final_loss = rows[-1][1]
    initial_loss = rows[0][1]
    sat = abs(rows[-1][1] - rows[-2][1]) if len(rows) >= 2 else 0.0
    print(f"  {name}:")
    print(f"    final_loss={final_loss:.3f}  wall={total_sec:.0f}s  epochs={n_epochs}  steps/SE={n_subtrees}  total_steps={total_steps}")
    print(f"    saturation Δ(last 2 epochs)={sat:.4f}")
    for t in targets:
        cum = 0.0
        hit = None
        for ep, loss, sec, _ in rows:
            cum += sec
            if loss <= t:
                hit = (ep, cum)
                break
        if hit:
            print(f"    loss<={t:>4.1f}:  epoch={hit[0]:3d}  wall={hit[1]:6.0f}s")
        else:
            print(f"    loss<={t:>4.1f}:  never (best={final_loss:.3f})")
    if total_sec > 0 and total_steps > 0:
        dl_sec = (initial_loss - final_loss) / total_sec
        dl_step = (initial_loss - final_loss) / total_steps
        print(f"    Δloss rate: {dl_sec*1000:.3f} mloss/sec, {dl_step*1e6:.2f} μloss/step")


def report_sgd(name, rows, targets):
    if not rows:
        print(f"  {name}: empty"); return
    final_step, final_avg = rows[-1]
    print(f"  {name}: final_step={final_step}, final_avg_loss={final_avg:.3f}")
    for t in targets:
        first = next((s for s, l in rows if l <= t), None)
        if first is not None:
            print(f"    avg<={t:>4.1f}:  step={first}")
        else:
            print(f"    avg<={t:>4.1f}:  never (best={final_avg:.3f})")


def detect_format(path):
    with open(path) as fh:
        for line in fh:
            if EPOCH_RE.match(line):
                return "agpt"
            if STEP_RE.match(line):
                return "sgd"
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="Training log files")
    ap.add_argument("--targets", default="5.0,3.0,2.0,1.7,1.5,1.3",
                    help="Comma-separated loss targets (default 5,3,2,1.7,1.5,1.3)")
    args = ap.parse_args()
    targets = [float(t) for t in args.targets.split(",")]
    print(f"Targets: {targets}")
    print("=" * 78)
    for path in args.logs:
        p = Path(path)
        if not p.exists():
            print(f"  {path}: missing"); continue
        fmt = detect_format(path)
        name = p.stem
        if fmt == "agpt":
            report_agpt(name, parse_agpt(path), targets)
        elif fmt == "sgd":
            report_sgd(name, parse_sgd(path), targets)
        else:
            print(f"  {name}: unrecognized log format")


if __name__ == "__main__":
    main()
