#!/usr/bin/env python3
"""Compute and inspect the eff_cos/eff_sin magnitudes that dist-rope produces
from a per-substring position table. Reports magnitude distribution by
substring (correlated with mass / distribution sharpness) and by dim-pair.

Usage:
  python3 src/tools/inspect_eff_rope.py \\
      --position-table /tmp/shake_position_data/prefix_position_table.bin \\
      --head-dim 16 --base 10000
"""
import argparse, struct, math, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--position-table', required=True)
ap.add_argument('--head-dim', type=int, default=16)
ap.add_argument('--base', type=float, default=10000.0)
ap.add_argument('--sample', type=int, default=20000, help='sample N substrings for stats')
args = ap.parse_args()

# Parse APOS file (matches src/agpt/position_table.cr layout)
with open(args.position_table, 'rb') as f:
    magic = f.read(4)
    assert magic == b'APOS', f"bad magic {magic}"
    regime = f.read(1)[0]
    window_size = struct.unpack('<H', f.read(2))[0]
    f.read(1)  # reserved
    substring_count = struct.unpack('<I', f.read(4))[0]
    total_bins = struct.unpack('<Q', f.read(8))[0]
    pos_offsets = struct.unpack(f'<{substring_count + 1}i', f.read(4 * (substring_count + 1)))
    # pos_bins are u16 pos + u32 count, packed (6 bytes each)
    raw = f.read(6 * total_bins)

print(f'window_size={window_size} regime={regime} substring_count={substring_count} total_bins={total_bins}', file=sys.stderr)

# Compute inv_freq
hd = args.head_dim
inv_freq = [1.0 / (args.base ** (2 * i / hd)) for i in range(hd // 2)]

# Sample
import random
random.seed(42)
sample_ids = random.sample(range(substring_count), min(args.sample, substring_count))

# For each sampled substring, compute eff vector magnitude per dim-pair
mag_by_dim = [[] for _ in range(hd // 2)]
mag_by_sid_mass = []  # (total_mass, n_bins, mean_magnitude_across_dims)

for sid in sample_ids:
    start, end = pos_offsets[sid], pos_offsets[sid + 1]
    n = end - start
    if n == 0:
        continue
    bins = []
    total = 0
    for b in range(n):
        off = (start + b) * 6
        pos = struct.unpack('<H', raw[off:off+2])[0]
        cnt = struct.unpack('<I', raw[off+2:off+6])[0]
        bins.append((pos, cnt))
        total += cnt
    # Per dim-pair
    sub_mags = []
    for di, freq in enumerate(inv_freq):
        ec = 0.0
        es = 0.0
        for pos, cnt in bins:
            w = cnt / total
            ang = pos * freq
            ec += w * math.cos(ang)
            es += w * math.sin(ang)
        m = math.sqrt(ec*ec + es*es)
        mag_by_dim[di].append(m)
        sub_mags.append(m)
    mag_by_sid_mass.append((total, n, sum(sub_mags) / len(sub_mags)))

# Stats
def stats(xs):
    xs = sorted(xs)
    n = len(xs)
    return (n, xs[0], xs[n//4], xs[n//2], xs[3*n//4], xs[-1], sum(xs)/n)

print(f"\n## Eff-rope magnitude distribution by dim-pair (sample of {len(mag_by_dim[0])} substrings)")
print(f"  dim_pair  freq                  count  min     p25     median  p75     max     mean")
for di in range(hd // 2):
    n, mn, p25, med, p75, mx, mean = stats(mag_by_dim[di])
    period = 2 * math.pi / inv_freq[di] if inv_freq[di] > 0 else float('inf')
    print(f"  pair {di:2d}     period~{period:7.1f}        {n}   {mn:.3f}   {p25:.3f}   {med:.3f}   {p75:.3f}   {mx:.3f}   {mean:.3f}")

# Bucket by mass
print(f"\n## Mean eff-rope magnitude (averaged across all dim-pairs) by substring mass")
buckets = {'mass=1': [], 'mass=2-9': [], 'mass=10-99': [], 'mass=100-999': [], 'mass>=1000': []}
for total, _, mean_mag in mag_by_sid_mass:
    if total == 1:
        buckets['mass=1'].append(mean_mag)
    elif total < 10:
        buckets['mass=2-9'].append(mean_mag)
    elif total < 100:
        buckets['mass=10-99'].append(mean_mag)
    elif total < 1000:
        buckets['mass=100-999'].append(mean_mag)
    else:
        buckets['mass>=1000'].append(mean_mag)
print(f"  bucket            count   median magnitude   mean magnitude")
for k, xs in buckets.items():
    if xs:
        n, _, _, med, _, _, mean = stats(xs)
        print(f"  {k:18s} {n:6d}   {med:.4f}             {mean:.4f}")
    else:
        print(f"  {k:18s}      0   (empty)")
