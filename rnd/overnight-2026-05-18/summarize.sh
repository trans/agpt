#!/usr/bin/env bash
# Run this when you wake up. Pulls per-seed PPLs from the logs and computes
# means, std, and Welch's t-test for streaming vs baseline on Gutenberg.

set -e
cd "$(dirname "$0")/../.."

python3 - <<'PYEOF'
import os, glob, re, statistics, math
LOGS = "rnd/streaming-agpt-v1/logs"

def grep_ppl(path):
    if not os.path.exists(path):
        return None
    for line in open(path):
        m = re.match(r"^Perplexity:\s*([\d.]+)", line)
        if m:
            return float(m.group(1))
    return None

streaming = []
baseline = []
for seed in (100, 200, 300):
    sp = grep_ppl(f"{LOGS}/seed{seed}_ms_n100_se5_ppl.log")
    if sp is not None:
        streaming.append((seed, sp))
    # Baseline file names from run_multiseed_baseline.sh:
    bp = grep_ppl(f"{LOGS}/ms_baseline_gutenberg_5m_se500_seed{seed}_ppl.log")
    if bp is not None:
        baseline.append((seed, bp))

def stats(xs):
    vals = [v for _, v in xs]
    if len(vals) < 2:
        return None
    return statistics.mean(vals), statistics.stdev(vals), len(vals)

def welch(a, b):
    ma, sa, na = a; mb, sb, nb = b
    se = math.sqrt(sa*sa/na + sb*sb/nb)
    t = (ma - mb) / se
    df_num = (sa*sa/na + sb*sb/nb)**2
    df_den = (sa*sa/na)**2/(na-1) + (sb*sb/nb)**2/(nb-1)
    df = df_num / df_den
    return t, df

print("=" * 60)
print("Streaming-AGPT vs Baseline @ Gutenberg 5M d=16, 500 SE-equiv")
print("=" * 60)
print()
print("Streaming (100×5 SE):")
for s, p in streaming:
    print(f"  seed {s}: PPL {p:.4f}")
print()
print("Baseline (500 SE single-stage):")
for s, p in baseline:
    print(f"  seed {s}: PPL {p:.4f}")
print()

ss, bs = stats(streaming), stats(baseline)
if ss and bs:
    print(f"Streaming: PPL {ss[0]:.4f} ± {ss[1]:.4f} (n={ss[2]})")
    print(f"Baseline:  PPL {bs[0]:.4f} ± {bs[1]:.4f} (n={bs[2]})")
    print(f"Δ:         {ss[0] - bs[0]:+.4f} ({100*(ss[0]-bs[0])/bs[0]:+.2f}%)")
    t, df = welch(ss, bs)
    print(f"Welch's t: {t:.3f}, df={df:.1f}")
    print(f"  (|t| > 2.0 ≈ p<0.05; |t| > 2.7 ≈ p<0.01 at df≈4)")
else:
    print("Incomplete results — wait for runs to finish.")
PYEOF
