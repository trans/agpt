#!/usr/bin/env python3
"""Offline diagnostic for the harmonic-filter / chord-correlation RoPE design.

Computes on-path vs off-path phase_score histograms from the existing
position-table precompute. If on-path and off-path histograms don't
separate, the chord-mod-W formulation isn't carrying path structure,
and no CUDA kernel will save it. See notes/seq-len-extension/
harmonic-filter-brief.md for design context.

Phase score variants computed:
  E3 (no shift):  phase_score(Q, K)    = Σ_j Re(conj(z_Q/C_Q) · z_K/C_K)
  E4 (depth shift): phase_score(Q, K, Δ) = Σ_j Re(conj(z_Q/C_Q) · z_K/C_K · e^{i Δ ω_j})
  E4-norm (unit chords): same as E4 but with z/|z| instead of z/C
                          (eliminates the r_Q · r_K gate)

For each on-path (Q, K) pair (ancestor-descendant on some real path),
compute all three phase_scores. For each off-path pair (random Q, K
not in ancestor-descendant relationship), same.

Usage:
  python3 src/tools/harmonic_filter_diagnostic.py \
      --position-data /tmp/shake_position_data \
      --corpus data/input.txt \
      --d 16 --head-dim 16 --base 10000 \
      --n-pairs 50000 \
      --out rnd/harmonic-filter-diagnostic/
"""

import argparse
import math
import random
import struct
import sys
from pathlib import Path
from collections import defaultdict


def load_substring_catalog(path):
    """Load substrings.bin → list of (token tuples) indexed by substring_id."""
    with open(path, 'rb') as f:
        magic = f.read(4)
        assert magic == b'ASUB', f"bad magic {magic}"
        count = struct.unpack('<I', f.read(4))[0]
        catalog = []
        for _ in range(count):
            length = f.read(1)[0]
            tokens = struct.unpack(f'<{length}B', f.read(length))
            catalog.append(tuple(tokens))
    return catalog


def load_position_table(path):
    """Load *_position_table.bin → (window_size, substring_count, pos_offsets, pos_bins)."""
    with open(path, 'rb') as f:
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
    return window_size, substring_count, list(pos_offsets), raw


def parse_pos_bins(raw, offset, count):
    """Return list of (pos, count) tuples for a substring's slice of pos_bins."""
    bins = []
    for i in range(count):
        off = (offset + i) * 6
        p = struct.unpack('<H', raw[off:off+2])[0]
        c = struct.unpack('<I', raw[off+2:off+6])[0]
        bins.append((p, c))
    return bins


def compute_chord(bins, freqs):
    """Compute z = Σ count(p) e^{i p ω_j} and total_count for each frequency ω_j.

    Returns (z_real, z_imag, total_count) where z_real and z_imag are lists
    of length len(freqs).
    """
    n_freq = len(freqs)
    z_real = [0.0] * n_freq
    z_imag = [0.0] * n_freq
    total = 0
    for (p, c) in bins:
        total += c
        for i, omega in enumerate(freqs):
            angle = p * omega
            z_real[i] += c * math.cos(angle)
            z_imag[i] += c * math.sin(angle)
    return z_real, z_imag, total


def phase_score_e3(zq_re, zq_im, cq, zk_re, zk_im, ck, freqs, pair_subset=None):
    """E3: no depth shift. Returns scalar = Σ_j Re(conj(z_Q/C_Q) · z_K/C_K).

    pair_subset: list of dim-pair indices to use (None = all).
    """
    score = 0.0
    indices = pair_subset if pair_subset is not None else range(len(freqs))
    for j in indices:
        # conj(z_Q/C_Q) · z_K/C_K = (zq_re - i·zq_im)/cq · (zk_re + i·zk_im)/ck
        # Real part: (zq_re · zk_re + zq_im · zk_im) / (cq · ck)
        score += (zq_re[j] * zk_re[j] + zq_im[j] * zk_im[j]) / (cq * ck)
    return score


def phase_score_e4(zq_re, zq_im, cq, zk_re, zk_im, ck, freqs, delta, pair_subset=None):
    """E4: depth shift Δ. Returns scalar = Σ_j Re(conj(z_Q/C_Q) · z_K/C_K · e^{i Δ ω_j})."""
    score = 0.0
    indices = pair_subset if pair_subset is not None else range(len(freqs))
    for j in indices:
        omega = freqs[j]
        # Apply e^{i Δ ω} = cos(Δω) + i sin(Δω) to z_K/C_K
        cos_d = math.cos(delta * omega)
        sin_d = math.sin(delta * omega)
        # z_K_shifted = (zk_re + i·zk_im)·(cos_d + i·sin_d)
        #             = (zk_re·cos_d - zk_im·sin_d) + i·(zk_re·sin_d + zk_im·cos_d)
        zk_re_shifted = zk_re[j] * cos_d - zk_im[j] * sin_d
        zk_im_shifted = zk_re[j] * sin_d + zk_im[j] * cos_d
        # conj(z_Q/C_Q) · z_K_shifted/C_K, real part:
        # (zq_re - i·zq_im)·(zk_re_s + i·zk_im_s) → Re = zq_re·zk_re_s + zq_im·zk_im_s
        score += (zq_re[j] * zk_re_shifted + zq_im[j] * zk_im_shifted) / (cq * ck)
    return score


def phase_score_e4_norm(zq_re, zq_im, zk_re, zk_im, freqs, delta, pair_subset=None):
    """E4-norm: depth shift Δ with unit-magnitude chords (no r gate).

    Returns Σ_j Re(conj(z_Q/|z_Q|) · z_K/|z_K| · e^{i Δ ω_j}).
    """
    score = 0.0
    indices = pair_subset if pair_subset is not None else range(len(freqs))
    for j in indices:
        omega = freqs[j]
        # Normalize z_Q and z_K to unit magnitude per dim-pair
        mag_q = math.sqrt(zq_re[j]**2 + zq_im[j]**2)
        mag_k = math.sqrt(zk_re[j]**2 + zk_im[j]**2)
        if mag_q < 1e-9 or mag_k < 1e-9:
            # Skip if either is degenerate
            continue
        nq_re = zq_re[j] / mag_q
        nq_im = zq_im[j] / mag_q
        nk_re = zk_re[j] / mag_k
        nk_im = zk_im[j] / mag_k
        cos_d = math.cos(delta * omega)
        sin_d = math.sin(delta * omega)
        nk_re_shifted = nk_re * cos_d - nk_im * sin_d
        nk_im_shifted = nk_re * sin_d + nk_im * cos_d
        score += nq_re * nk_re_shifted + nq_im * nk_im_shifted
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--position-data', required=True, help='Position data directory')
    ap.add_argument('--corpus', required=True, help='Corpus file (for sampling windows)')
    ap.add_argument('--d', type=int, default=16, help='Trie depth')
    ap.add_argument('--head-dim', type=int, default=16, help='Per-head RoPE dimension')
    ap.add_argument('--base', type=float, default=10000.0, help='RoPE base')
    ap.add_argument('--n-pairs', type=int, default=10000, help='Number of pairs to sample (on-path AND off-path each)')
    ap.add_argument('--out', required=True, help='Output directory for histograms + summary')
    ap.add_argument('--seed', type=int, default=42, help='RNG seed')
    ap.add_argument('--chord-type', choices=['position', 'gap'], default='position',
                    help='position: chord over corpus-position-mod-W (original). '
                         'gap: chord over gap-mod-W (distance from preceding occurrence).')
    ap.add_argument('--gap-mod', type=int, default=64,
                    help='Modulus for gap chord (default 64 = same as W for clean swap)')
    ap.add_argument('--shrinkage-n0', type=int, default=0,
                    help='Count shrinkage: r_shrunk = r · n/(n+n0). 0 = no shrinkage.')
    args = ap.parse_args()

    random.seed(args.seed)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    # Phase-score subsets to compute for each pair, identified by name + indices.
    # "all" uses all dim-pairs; "useful" restricts to pairs with periods comparable to W;
    # "noise" uses the very-low-frequency pairs to confirm they don't carry signal.
    SUBSETS = {
        'all': None,
        'useful (pairs 0-2)': [0, 1, 2],
        'noise (pairs 4-7)': [4, 5, 6, 7],
    }

    # Load substring catalog
    print(f'Loading substring catalog from {args.position_data}/substrings.bin', file=sys.stderr)
    catalog = load_substring_catalog(f'{args.position_data}/substrings.bin')
    n_substrings = len(catalog)
    print(f'  {n_substrings} substrings', file=sys.stderr)

    # Build chars-tuple → substring_id lookup
    chars_to_id = {chars: idx for idx, chars in enumerate(catalog)}

    # Load position table (prefix only — we're sampling forward paths)
    print(f'Loading position table from {args.position_data}/prefix_position_table.bin', file=sys.stderr)
    window_size, substring_count, pos_offsets, pos_bins_raw = load_position_table(
        f'{args.position_data}/prefix_position_table.bin')
    assert substring_count == n_substrings
    print(f'  W={window_size}, {substring_count} substrings, {len(pos_bins_raw)//6} bins', file=sys.stderr)

    # Compute frequencies (= angular frequencies for each dim-pair)
    n_pairs = args.head_dim // 2
    freqs = [1.0 / (args.base ** (2 * i / args.head_dim)) for i in range(n_pairs)]
    print(f'\nFrequencies (ω) per dim-pair (head_dim={args.head_dim}, base={args.base}):', file=sys.stderr)
    for i, f in enumerate(freqs):
        period = 2 * math.pi / f
        print(f'  pair {i}: ω={f:.5f}  period={period:.1f}  cycles_in_W={window_size/period:.2f}', file=sys.stderr)

    # Lazy chord computation with cache. For 8M substrings, precomputing
    # everything eats >2 GB of Python list overhead and OOMs. Instead,
    # cache only the chords of substrings actually sampled.
    chord_cache = {}  # sid → (z_real_list, z_imag_list, total_count)

    if args.chord_type == 'position':
        # Standard mod-W position chord (current implementation)
        def get_chord(sid):
            if sid in chord_cache:
                return chord_cache[sid]
            start = pos_offsets[sid]
            end = pos_offsets[sid + 1]
            nbins = end - start
            if nbins == 0:
                result = ([0.0] * n_pairs, [0.0] * n_pairs, 0)
            else:
                bins = parse_pos_bins(pos_bins_raw, start, nbins)
                zr, zi, total = compute_chord(bins, freqs)
                result = (zr, zi, total)
            chord_cache[sid] = result
            return result
        gap_data_provider = None
    else:
        # Gap-distribution chord. Defer the corpus walk until AFTER we know
        # which substrings we need (post-sampling). For now, just expose a
        # placeholder; the real walk happens once we have the needed SIDs.
        # get_chord will populate from a gap_counts dict built lazily.
        gap_counts = {}  # sid → dict of {gap_mod: count}
        n0 = args.shrinkage_n0

        def get_chord(sid):
            if sid in chord_cache:
                return chord_cache[sid]
            if sid not in gap_counts:
                result = ([0.0] * n_pairs, [0.0] * n_pairs, 0)
                chord_cache[sid] = result
                return result
            bins_dict = gap_counts[sid]
            bins = [(g, c) for g, c in bins_dict.items()]
            zr, zi, total = compute_chord(bins, freqs)
            if n0 > 0 and total > 0:
                shrinkage = total / (total + n0)
                zr = [v * shrinkage for v in zr]
                zi = [v * shrinkage for v in zi]
            result = (zr, zi, total)
            chord_cache[sid] = result
            return result
        gap_data_provider = gap_counts

    # Load corpus for sampling on-path pairs
    print(f'\nLoading corpus from {args.corpus}', file=sys.stderr)
    corpus_text = Path(args.corpus).read_text()
    # Build char vocab consistent with how the trie was built
    chars = sorted(set(corpus_text))
    char_to_id = {c: i for i, c in enumerate(chars)}
    corpus_tokens = [char_to_id[c] for c in corpus_text]
    n_corpus = len(corpus_tokens)
    print(f'  {n_corpus} tokens, vocab={len(chars)}', file=sys.stderr)

    # Phase 1: sample pair (sid_q, sid_k, delta) tuples without computing chord.
    # For gap mode, we use these to know which substrings need their gaps
    # accumulated. For position mode, we can compute chord on-the-fly via the
    # existing position table.
    print(f'\nPhase 1: sampling {args.n_pairs} on-path + {args.n_pairs} off-path pair tuples...', file=sys.stderr)
    on_path_tuples = []  # list of (sid_q, sid_k, delta)
    off_path_tuples = []
    n_skipped_on = 0
    n_skipped_off = 0

    attempts = 0
    while len(on_path_tuples) < args.n_pairs and attempts < args.n_pairs * 5:
        attempts += 1
        p = random.randint(0, n_corpus - args.d)
        window = tuple(corpus_tokens[p:p + args.d])
        k_k = random.randint(1, args.d - 1)
        k_q = random.randint(k_k + 1, args.d)
        sid_k = chars_to_id.get(window[:k_k])
        sid_q = chars_to_id.get(window[:k_q])
        if sid_k is None or sid_q is None:
            n_skipped_on += 1
            continue
        on_path_tuples.append((sid_q, sid_k, k_q - k_k))
    attempts = 0
    while len(off_path_tuples) < args.n_pairs and attempts < args.n_pairs * 5:
        attempts += 1
        p1 = random.randint(0, n_corpus - args.d)
        p2 = random.randint(0, n_corpus - args.d)
        if abs(p1 - p2) < args.d:
            continue
        k_k = random.randint(1, args.d - 1)
        k_q = random.randint(k_k + 1, args.d)
        sid_k = chars_to_id.get(tuple(corpus_tokens[p2:p2 + k_k]))
        sid_q = chars_to_id.get(tuple(corpus_tokens[p1:p1 + k_q]))
        if sid_k is None or sid_q is None:
            n_skipped_off += 1
            continue
        off_path_tuples.append((sid_q, sid_k, k_q - k_k))
    print(f'  on-path: {len(on_path_tuples)} tuples ({n_skipped_on} skipped)', file=sys.stderr)
    print(f'  off-path: {len(off_path_tuples)} tuples ({n_skipped_off} skipped)', file=sys.stderr)

    # Phase 2 (gap mode only): walk corpus, accumulate gap histograms for
    # only the substrings that appear in our pair tuples.
    if args.chord_type == 'gap':
        needed_sids = set()
        for sid_q, sid_k, _ in on_path_tuples:
            needed_sids.add(sid_q); needed_sids.add(sid_k)
        for sid_q, sid_k, _ in off_path_tuples:
            needed_sids.add(sid_q); needed_sids.add(sid_k)
        print(f'\nPhase 2 (gap mode): walking corpus for {len(needed_sids)} needed substrings...',
              file=sys.stderr)
        import time
        t0 = time.time()
        last_seen = {}  # sid → last absolute position
        for p in range(n_corpus - args.d + 1):
            for k in range(1, args.d + 1):
                if p + k > n_corpus:
                    break
                sub = tuple(corpus_tokens[p:p + k])
                sid = chars_to_id.get(sub)
                if sid is None or sid not in needed_sids:
                    continue
                if sid in last_seen:
                    gap = p - last_seen[sid]
                    bin_idx = gap % args.gap_mod
                    if sid not in gap_data_provider:
                        gap_data_provider[sid] = {}
                    gap_data_provider[sid][bin_idx] = gap_data_provider[sid].get(bin_idx, 0) + 1
                last_seen[sid] = p
            if (p + 1) % 200000 == 0:
                print(f'  {p+1}/{n_corpus} ({time.time()-t0:.1f}s)', file=sys.stderr)
        print(f'  walk done: {len(gap_data_provider)} substrings have gaps ({time.time()-t0:.1f}s)',
              file=sys.stderr)

    # Phase 3: compute phase scores for each tuple
    print(f'\nPhase 3: computing phase scores...', file=sys.stderr)
    on_path_results = {(v, s): [] for v in ('e3', 'e4', 'e4_norm') for s in SUBSETS}
    off_path_results = {(v, s): [] for v in ('e3', 'e4', 'e4_norm') for s in SUBSETS}
    on_path_delta = []

    def compute_scores_for_tuples(tuples, results, record_delta=False):
        n_skipped = 0
        for sid_q, sid_k, delta in tuples:
            zq_re, zq_im, cq = get_chord(sid_q)
            zk_re, zk_im, ck = get_chord(sid_k)
            if cq == 0 or ck == 0:
                n_skipped += 1
                continue
            for subset_name, subset in SUBSETS.items():
                results[('e3', subset_name)].append(
                    phase_score_e3(zq_re, zq_im, cq, zk_re, zk_im, ck, freqs, subset))
                results[('e4', subset_name)].append(
                    phase_score_e4(zq_re, zq_im, cq, zk_re, zk_im, ck, freqs, delta, subset))
                results[('e4_norm', subset_name)].append(
                    phase_score_e4_norm(zq_re, zq_im, zk_re, zk_im, freqs, delta, subset))
            if record_delta:
                on_path_delta.append(delta)
        return n_skipped

    n_skipped_on_score = compute_scores_for_tuples(on_path_tuples, on_path_results, record_delta=True)
    n_skipped_off_score = compute_scores_for_tuples(off_path_tuples, off_path_results)
    n_collected_on = len(on_path_results[('e3', 'all')])
    n_collected_off = len(off_path_results[('e3', 'all')])
    print(f'  on-path scored: {n_collected_on} (skipped {n_skipped_on_score} due to zero count)', file=sys.stderr)
    print(f'  off-path scored: {n_collected_off} (skipped {n_skipped_off_score} due to zero count)', file=sys.stderr)
    print(f'  chord cache size: {len(chord_cache)}', file=sys.stderr)

    # Compute summary stats
    def stats(xs):
        xs = sorted(xs)
        n = len(xs)
        return {
            'n': n,
            'min': xs[0],
            'p10': xs[n//10],
            'p25': xs[n//4],
            'median': xs[n//2],
            'p75': xs[3*n//4],
            'p90': xs[9*n//10],
            'max': xs[-1],
            'mean': sum(xs)/n,
        }

    summary_lines = []
    summary_lines.append("# Harmonic-filter diagnostic results")
    summary_lines.append("")
    summary_lines.append(f"Data: {args.position_data}")
    summary_lines.append(f"Corpus: {args.corpus}")
    summary_lines.append(f"d={args.d}  head_dim={args.head_dim}  base={args.base}")
    summary_lines.append(f"W (chord window): {window_size}")
    summary_lines.append(f"n_pairs sampled: {args.n_pairs} on-path, {args.n_pairs} off-path")
    summary_lines.append("")
    summary_lines.append(f"Frequencies per dim-pair:")
    for i, f in enumerate(freqs):
        period = 2 * math.pi / f
        summary_lines.append(f"  pair {i}: ω={f:.5f}  period={period:.1f}  cycles_in_W={window_size/period:.2f}")
    summary_lines.append("")

    for variant_name, variant_key in [
        ('E3 (no shift)', 'e3'),
        ('E4 (with depth shift)', 'e4'),
        ('E4-norm (unit chords + shift)', 'e4_norm'),
    ]:
        summary_lines.append(f"## {variant_name}")
        summary_lines.append("")
        for subset_name in SUBSETS:
            on_data = on_path_results[(variant_key, subset_name)]
            off_data = off_path_results[(variant_key, subset_name)]
            on_stats = stats(on_data)
            off_stats = stats(off_data)
            gap = on_stats['median'] - off_stats['median']
            iqr_on = on_stats['p75'] - on_stats['p25']
            iqr_off = off_stats['p75'] - off_stats['p25']
            avg_iqr = (iqr_on + iqr_off) / 2
            separation = gap / avg_iqr if avg_iqr > 0 else float('inf')
            summary_lines.append(f"### dim-pairs: {subset_name}")
            summary_lines.append(f"```")
            summary_lines.append(f"            min       p25     median   p75     mean")
            summary_lines.append(f"on-path  {on_stats['min']:+.4f}  {on_stats['p25']:+.4f}  {on_stats['median']:+.4f}  {on_stats['p75']:+.4f}  {on_stats['mean']:+.4f}")
            summary_lines.append(f"off-path {off_stats['min']:+.4f}  {off_stats['p25']:+.4f}  {off_stats['median']:+.4f}  {off_stats['p75']:+.4f}  {off_stats['mean']:+.4f}")
            summary_lines.append(f"gap (on - off): {gap:+.4f}    IQR avg: {avg_iqr:.4f}    separation: {separation:+.3f}")
            summary_lines.append(f"```")
            summary_lines.append("")

    summary_path = Path(args.out) / 'summary.md'
    summary_path.write_text('\n'.join(summary_lines))
    print(f'\nSummary written to {summary_path}', file=sys.stderr)

    # Dump raw data for plotting later
    import json
    raw_data = {
        'config': {
            'd': args.d, 'head_dim': args.head_dim, 'base': args.base,
            'n_pairs_target': args.n_pairs, 'seed': args.seed,
            'window_size': window_size, 'frequencies': freqs,
            'subsets': {name: list(idx) if idx else 'all' for name, idx in SUBSETS.items()},
        },
        'on_path': {f'{v}_{s}': on_path_results[(v, s)] for v in ('e3', 'e4', 'e4_norm') for s in SUBSETS},
        'off_path': {f'{v}_{s}': off_path_results[(v, s)] for v in ('e3', 'e4', 'e4_norm') for s in SUBSETS},
        'delta': on_path_delta,
    }
    raw_path = Path(args.out) / 'raw.json'
    raw_path.write_text(json.dumps(raw_data))
    print(f'Raw data written to {raw_path}', file=sys.stderr)

    # Print summary to stdout too
    print('\n' + '\n'.join(summary_lines))


if __name__ == '__main__':
    main()
