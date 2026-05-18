#!/usr/bin/env python3
"""
Analyze a per-rc v dump produced by `agpt_train --dump-per-rc-v PATH`.

File format (little-endian):
  magic[4] = 'PRVD'
  int32 n_rc
  int32 total_floats
  int32[n_rc] per-bucket step counts (adam_t per bucket)
  float32[n_rc * total_floats] flat per-rc v buffer

What we look at to decide whether the localization hypothesis is alive:
  - Per-bucket ||v||_2 (gradient-scale magnitude)
  - Per-bucket mean(v), max(v)
  - Pairwise cosine similarity between high-mass buckets' v arrays
"""
import struct
import sys
import numpy as np

def load_dump(path):
    with open(path, 'rb') as fp:
        magic = fp.read(4)
        if magic != b'PRVD':
            raise ValueError(f"bad magic: {magic!r}")
        n_rc = struct.unpack('<i', fp.read(4))[0]
        tf   = struct.unpack('<i', fp.read(4))[0]
        steps = np.frombuffer(fp.read(4 * n_rc), dtype='<i4')
        v = np.frombuffer(fp.read(4 * n_rc * tf), dtype='<f4').reshape(n_rc, tf).copy()
    return v, steps

def main():
    if len(sys.argv) < 2:
        print("usage: analyze_dump.py PATH [TOP_K]")
        sys.exit(1)
    path = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    v, steps = load_dump(path)
    n_rc, tf = v.shape
    print(f"n_rc = {n_rc}, total_floats = {tf}")

    # Per-bucket scalar stats
    l2 = np.linalg.norm(v, axis=1)
    mean = v.mean(axis=1)
    mx = v.max(axis=1)
    print(f"\nPer-bucket stats (sorted by step count desc):")
    print(f"{'rc':>4} {'steps':>8} {'||v||':>12} {'mean':>12} {'max':>12}")
    order = np.argsort(-steps)
    for rc in order[:top_k]:
        print(f"{rc:>4} {steps[rc]:>8} {l2[rc]:>12.4e} {mean[rc]:>12.4e} {mx[rc]:>12.4e}")

    # Cosine similarity between top buckets
    print(f"\nCosine similarity between top-{min(top_k, 8)} buckets (by step count):")
    top = order[:min(top_k, 8)]
    norm_v = v[top] / (np.linalg.norm(v[top], axis=1, keepdims=True) + 1e-12)
    sim = norm_v @ norm_v.T
    header = "       " + " ".join(f"{rc:>7}" for rc in top)
    print(header)
    for i, rc in enumerate(top):
        row = f"  {rc:>4}: " + " ".join(f"{sim[i,j]:>7.4f}" for j in range(len(top)))
        print(row)

    # Quick verdict
    print(f"\nInterpretation:")
    if sim[~np.eye(len(top), dtype=bool)].min() > 0.95:
        print("  All bucket-v's are highly correlated (>0.95). Topological localization")
        print("  signal is weak — most variance is shared. Hypothesis looks weak.")
    elif sim[~np.eye(len(top), dtype=bool)].max() < 0.5:
        print("  Bucket-v's are quite different from each other (<0.5 max similarity).")
        print("  Topological localization signal is strong. Hypothesis looks alive;")
        print("  regression is likely cold-start / sample-size artifact.")
    else:
        print("  Mixed: buckets differ but share structure. Read individual rows above.")

if __name__ == '__main__':
    main()
