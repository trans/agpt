#!/usr/bin/env python3
"""Stratified harmonic-filter diagnostic: bucket on-path/off-path pairs by
K's mass and by individual dim-pair, to find where the chord signal lives.

Hypothesis: chord correlation has a positive |z_Q|² bias for ancestor-
descendant pairs, but it's washed out for high-mass K (where K's other
contexts dominate). Low-mass K should show clearer separation.

Per-dim-pair: signal should live in pair 2 (period 62.8 ≈ W=64), be
absent at high-freq pairs (z magnitude small) and saturated at low-freq
pairs (all chords point same direction).

Usage:
  python3 src/tools/harmonic_filter_stratified.py \\
      --position-data /tmp/shake_position_data \\
      --corpus data/input.txt \\
      --n-pairs 30000
"""

import argparse
import math
import random
import struct
import sys
from pathlib import Path


def load_catalog(path):
    with open(path, 'rb') as f:
        assert f.read(4) == b'ASUB'
        count = struct.unpack('<I', f.read(4))[0]
        cat = []
        for _ in range(count):
            length = f.read(1)[0]
            cat.append(tuple(struct.unpack(f'<{length}B', f.read(length))))
    return cat


def load_position_table(path):
    with open(path, 'rb') as f:
        assert f.read(4) == b'APOS'
        regime = f.read(1)[0]
        W = struct.unpack('<H', f.read(2))[0]
        f.read(1)
        n_sub = struct.unpack('<I', f.read(4))[0]
        total_bins = struct.unpack('<Q', f.read(8))[0]
        pos_offsets = struct.unpack(f'<{n_sub + 1}i', f.read(4 * (n_sub + 1)))
        raw = f.read(6 * total_bins)
    return W, n_sub, pos_offsets, raw


def compute_chord(bins, freqs, weight='count'):
    """Build the per-substring Fourier chord.
    weight: 'count' (raw mass), 'logcount' (log(1+count)), 'one' (uniform)
    Returns (z_re, z_im, total) where `total` is the sum of effective weights.
    """
    n_freq = len(freqs)
    z_re = [0.0] * n_freq
    z_im = [0.0] * n_freq
    total = 0.0
    for (p, c) in bins:
        if weight == 'logcount':
            w = math.log(1.0 + c)
        elif weight == 'one':
            w = 1.0
        else:
            w = float(c)
        total += w
        for j, omega in enumerate(freqs):
            ang = p * omega
            z_re[j] += w * math.cos(ang)
            z_im[j] += w * math.sin(ang)
    return z_re, z_im, total


def parse_bins(raw, offset, count):
    out = []
    for i in range(count):
        off = (offset + i) * 6
        out.append((struct.unpack('<H', raw[off:off+2])[0],
                    struct.unpack('<I', raw[off+2:off+6])[0]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--position-data', required=True)
    ap.add_argument('--corpus', required=True)
    ap.add_argument('--d', type=int, default=16)
    ap.add_argument('--head-dim', type=int, default=16)
    ap.add_argument('--base', type=float, default=10000.0)
    ap.add_argument('--n-pairs', type=int, default=30000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--weight', choices=['count', 'logcount', 'one'],
                    default='count')
    ap.add_argument('--freq-mode', choices=['rope', 'dft'], default='rope',
                    help='rope: geometric base^(-2j/HD); '
                         'dft: ω_j = 2π·(j+1)/W (skip DC)')
    ap.add_argument('--window', type=int, default=64,
                    help='only used for --freq-mode dft')
    args = ap.parse_args()
    random.seed(args.seed)
    print(f'Weighting positions by: {args.weight}', file=sys.stderr)

    catalog = load_catalog(f'{args.position_data}/substrings.bin')
    chars_to_id = {c: i for i, c in enumerate(catalog)}
    n_sub = len(catalog)

    W, _, pos_offsets, pos_bins_raw = load_position_table(
        f'{args.position_data}/prefix_position_table.bin')

    hd = args.head_dim
    n_pairs = hd // 2
    if args.freq_mode == 'dft':
        # Skip DC (j=0); use first n_pairs non-DC harmonics: 2π·(j+1)/W
        freqs = [2.0 * math.pi * (j + 1) / args.window
                 for j in range(n_pairs)]
        print(f'Using DFT frequencies, W={args.window}, j=1..{n_pairs}',
              file=sys.stderr)
    else:
        freqs = [1.0 / (args.base ** (2 * j / hd)) for j in range(n_pairs)]
        print(f'Using RoPE frequencies, base={args.base}, hd={hd}',
              file=sys.stderr)

    # Corpus + vocab
    corpus_text = Path(args.corpus).read_text()
    chars = sorted(set(corpus_text))
    c2i = {c: i for i, c in enumerate(chars)}
    corpus_tokens = [c2i[c] for c in corpus_text]
    n_corpus = len(corpus_tokens)

    # Chord cache + raw mass cache (for bucketing — independent of --weight)
    chord_cache = {}
    raw_mass = {}

    def get_chord(sid):
        if sid in chord_cache:
            return chord_cache[sid]
        start, end = pos_offsets[sid], pos_offsets[sid+1]
        if end == start:
            r = ([0.0]*n_pairs, [0.0]*n_pairs, 0.0)
            raw_mass[sid] = 0
        else:
            bins = parse_bins(pos_bins_raw, start, end-start)
            r = compute_chord(bins, freqs, weight=args.weight)
            raw_mass[sid] = sum(c for _, c in bins)
        chord_cache[sid] = r
        return r

    # Sample pairs as tuples (sid_q, sid_k, k_mass)
    print(f'Sampling {args.n_pairs} on-path + {args.n_pairs} off-path pairs...',
          file=sys.stderr)
    on_pairs = []
    off_pairs = []

    while len(on_pairs) < args.n_pairs:
        p = random.randint(0, n_corpus - args.d)
        window = tuple(corpus_tokens[p:p + args.d])
        k_k = random.randint(1, args.d - 1)
        k_q = random.randint(k_k + 1, args.d)
        sid_k = chars_to_id.get(window[:k_k])
        sid_q = chars_to_id.get(window[:k_q])
        if sid_k is None or sid_q is None:
            continue
        _, _, ck = get_chord(sid_k)
        _, _, cq = get_chord(sid_q)
        if ck == 0 or cq == 0:
            continue
        # p_Q = corpus position of Q (same as K's start on this path), mod W
        on_pairs.append((sid_q, sid_k, raw_mass[sid_k], p % W))

    while len(off_pairs) < args.n_pairs:
        p1 = random.randint(0, n_corpus - args.d)
        p2 = random.randint(0, n_corpus - args.d)
        if abs(p1 - p2) < args.d:
            continue
        k_k = random.randint(1, args.d - 1)
        k_q = random.randint(k_k + 1, args.d)
        sid_k = chars_to_id.get(tuple(corpus_tokens[p2:p2 + k_k]))
        sid_q = chars_to_id.get(tuple(corpus_tokens[p1:p1 + k_q]))
        if sid_k is None or sid_q is None:
            continue
        _, _, ck = get_chord(sid_k)
        _, _, cq = get_chord(sid_q)
        if ck == 0 or cq == 0:
            continue
        # Q's position is p1 (random); K's distribution is asked at p1 mod W
        off_pairs.append((sid_q, sid_k, raw_mass[sid_k], p1 % W))

    print(f'  collected {len(on_pairs)} on-path, {len(off_pairs)} off-path',
          file=sys.stderr)

    # Compute per-dim-pair contribution to the correlation (E3, raw with r gate)
    # AND per-dim-pair contribution to normalized correlation (E3-norm)
    def per_pair_e3(zq_re, zq_im, cq, zk_re, zk_im, ck):
        out = []
        for j in range(n_pairs):
            out.append((zq_re[j]*zk_re[j] + zq_im[j]*zk_im[j]) / (cq*ck))
        return out

    def per_pair_e3_norm(zq_re, zq_im, zk_re, zk_im):
        out = []
        for j in range(n_pairs):
            mq = math.sqrt(zq_re[j]**2 + zq_im[j]**2)
            mk = math.sqrt(zk_re[j]**2 + zk_im[j]**2)
            if mq < 1e-9 or mk < 1e-9:
                out.append(0.0)
            else:
                out.append((zq_re[j]*zk_re[j] + zq_im[j]*zk_im[j]) / (mq*mk))
        return out

    def per_pair_asym(zk_re, zk_im, ck, p_q):
        """Asymmetric response: 'does K's distribution have power at p_Q?'
        For each dim-pair j, contributes:
            (1/ck) * (zk_re[j] * cos(p_q * ω_j) + zk_im[j] * sin(p_q * ω_j))
        i.e. the inverse-DFT value of K's distribution at position p_q for that
        single frequency component, normalized by total mass.
        """
        out = []
        for j, omega in enumerate(freqs):
            ang = p_q * omega
            comp = (zk_re[j] * math.cos(ang) +
                    zk_im[j] * math.sin(ang)) / ck
            out.append(comp)
        return out

    # Compute for each pair, output per-dim-pair lists indexed by mass bucket
    mass_buckets = [
        ('mass=1', lambda m: m == 1),
        ('mass=2-9', lambda m: 2 <= m <= 9),
        ('mass=10-99', lambda m: 10 <= m <= 99),
        ('mass=100-999', lambda m: 100 <= m <= 999),
        ('mass≥1000', lambda m: m >= 1000),
    ]

    # results[(bucket, on/off, formulation, dim_pair_j)] = list of scores
    on_e3 = {b[0]: [[] for _ in range(n_pairs)] for b in mass_buckets}
    on_e3n = {b[0]: [[] for _ in range(n_pairs)] for b in mass_buckets}
    on_asym = {b[0]: [[] for _ in range(n_pairs)] for b in mass_buckets}
    off_e3 = {b[0]: [[] for _ in range(n_pairs)] for b in mass_buckets}
    off_e3n = {b[0]: [[] for _ in range(n_pairs)] for b in mass_buckets}
    off_asym = {b[0]: [[] for _ in range(n_pairs)] for b in mass_buckets}
    bucket_n = {b[0]: 0 for b in mass_buckets}

    def assign(pairs, e3_dict, e3n_dict, asym_dict, count_dict):
        for sid_q, sid_k, k_mass_raw, p_q in pairs:
            zq_re, zq_im, cq = get_chord(sid_q)
            zk_re, zk_im, ck = get_chord(sid_k)
            pp3 = per_pair_e3(zq_re, zq_im, cq, zk_re, zk_im, ck)
            ppn = per_pair_e3_norm(zq_re, zq_im, zk_re, zk_im)
            ppa = per_pair_asym(zk_re, zk_im, ck, p_q)
            for name, fn in mass_buckets:
                if fn(k_mass_raw):
                    for j in range(n_pairs):
                        e3_dict[name][j].append(pp3[j])
                        e3n_dict[name][j].append(ppn[j])
                        asym_dict[name][j].append(ppa[j])
                    if count_dict is bucket_n:
                        count_dict[name] += 1
                    break

    assign(on_pairs, on_e3, on_e3n, on_asym, bucket_n)
    assign(off_pairs, off_e3, off_e3n, off_asym,
           {b[0]: 0 for b in mass_buckets})

    # Compute separation per (bucket, dim_pair, formulation)
    def median(xs):
        if not xs:
            return 0.0
        xs = sorted(xs)
        return xs[len(xs)//2]

    def iqr(xs):
        if len(xs) < 4:
            return 0.0
        xs = sorted(xs)
        return xs[3*len(xs)//4] - xs[len(xs)//4]

    print("\n## Stratified separation per (K mass bucket, dim-pair)\n")
    print("E3 (raw chord correlation, with r gate):")
    print(f"  {'bucket':<14} n_on  " + ' '.join(f' pair{j:1d} ' for j in range(n_pairs)))
    for name, _ in mass_buckets:
        n_on = bucket_n[name]
        if n_on < 50:
            print(f"  {name:<14} {n_on:5d}  (too few pairs to summarize)")
            continue
        seps = []
        for j in range(n_pairs):
            on_med = median(on_e3[name][j])
            off_med = median(off_e3[name][j])
            on_iqr = iqr(on_e3[name][j])
            off_iqr = iqr(off_e3[name][j])
            avg_iqr = (on_iqr + off_iqr) / 2
            if avg_iqr > 0:
                seps.append((on_med - off_med) / avg_iqr)
            else:
                seps.append(0.0)
        print(f"  {name:<14} {n_on:5d}  " + ' '.join(f'{s:+5.2f}' for s in seps))

    print("\nE3-norm (unit chords per dim-pair, no r gate):")
    print(f"  {'bucket':<14} n_on  " + ' '.join(f' pair{j:1d} ' for j in range(n_pairs)))
    for name, _ in mass_buckets:
        n_on = bucket_n[name]
        if n_on < 50:
            print(f"  {name:<14} {n_on:5d}  (too few pairs)")
            continue
        seps = []
        for j in range(n_pairs):
            on_med = median(on_e3n[name][j])
            off_med = median(off_e3n[name][j])
            on_iqr = iqr(on_e3n[name][j])
            off_iqr = iqr(off_e3n[name][j])
            avg_iqr = (on_iqr + off_iqr) / 2
            if avg_iqr > 0:
                seps.append((on_med - off_med) / avg_iqr)
            else:
                seps.append(0.0)
        print(f"  {name:<14} {n_on:5d}  " + ' '.join(f'{s:+5.2f}' for s in seps))

    print("\nASYM (K's distribution evaluated at p_Q — original concept):")
    print(f"  {'bucket':<14} n_on  " + ' '.join(f' pair{j:1d} ' for j in range(n_pairs)))
    for name, _ in mass_buckets:
        n_on = bucket_n[name]
        if n_on < 50:
            print(f"  {name:<14} {n_on:5d}  (too few pairs)")
            continue
        seps = []
        for j in range(n_pairs):
            on_med = median(on_asym[name][j])
            off_med = median(off_asym[name][j])
            on_iqr = iqr(on_asym[name][j])
            off_iqr = iqr(off_asym[name][j])
            avg_iqr = (on_iqr + off_iqr) / 2
            if avg_iqr > 0:
                seps.append((on_med - off_med) / avg_iqr)
            else:
                seps.append(0.0)
        print(f"  {name:<14} {n_on:5d}  " + ' '.join(f'{s:+5.2f}' for s in seps))

    # ASYM all-pairs aggregate score (sum across dim-pairs, before per-pair stats)
    print("\nASYM aggregate score (sum over all 8 dim-pairs):")
    print(f"  {'bucket':<14} n_on   on_med  off_med  separation")
    for name, _ in mass_buckets:
        n_on = bucket_n[name]
        if n_on < 50:
            print(f"  {name:<14} {n_on:5d}  (too few pairs)")
            continue
        on_scores = [sum(on_asym[name][j][i] for j in range(n_pairs))
                     for i in range(n_on)]
        n_off = len(off_asym[name][0])
        off_scores = [sum(off_asym[name][j][i] for j in range(n_pairs))
                      for i in range(n_off)]
        on_med = median(on_scores)
        off_med = median(off_scores)
        avg_iqr = (iqr(on_scores) + iqr(off_scores)) / 2
        sep = (on_med - off_med) / avg_iqr if avg_iqr > 0 else 0.0
        print(f"  {name:<14} {n_on:5d}  {on_med:+6.3f}  {off_med:+6.3f}  {sep:+5.2f}")


if __name__ == '__main__':
    main()
