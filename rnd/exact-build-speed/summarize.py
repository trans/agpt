#!/usr/bin/env python3
"""Tables for rnd/exact-build-speed from the runs' result.json."""
import json
from pathlib import Path

D = Path(__file__).resolve().parent
R = {}
for j in sorted(D.glob("2026*/result.json")):
    r = json.loads(j.read_text())
    slug = r["run_id"].split("-", 1)[1]            # e.g. pd1-fp32-fp32-r2
    case, cache, mm, rep = slug.split("-")
    R[(case, f"{cache}-{mm}", rep)] = r
S = ["bf16-tf32", "bf16-fp32", "fp32-tf32", "fp32-fp32"]
for case in ("pd1", "pd3"):
    print(f"\n{case}: training seconds (r1, r2; mean) | rolling byte PPL (r1, r2) | fixed-window PPL (r1, r2)")
    base = sum(R[(case, S[0], r)]["train_wall_seconds"] for r in ("r1", "r2")) / 2
    for s in S:
        t = [R[(case, s, r)]["train_wall_seconds"] for r in ("r1", "r2")]
        m = [R[(case, s, r)]["metrics"] for r in ("r1", "r2")]
        print(f"| {s} | {t[0]:.1f}, {t[1]:.1f}; {sum(t) / 2:.1f} ({sum(t) / 2 / base - 1:+.1%}) | "
              f"{m[0]['lm_eval_rolling_byte_perplexity']:.3f}, {m[1]['lm_eval_rolling_byte_perplexity']:.3f} | "
              f"{m[0]['agpt_fixed_token_perplexity']:.3f}, {m[1]['agpt_fixed_token_perplexity']:.3f} |")
