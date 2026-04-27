"""For each prefix-tree leaf, compute d* = depth at which its prefix
becomes unique (mass first drops to 1) walking root→leaf. Histogram d*
across all mass-1 leaves.

Expected: distribution peaked around 10-12 (the corpus's natural branching
depth that we estimated from log2(N)/per-char-entropy ≈ 11).
"""
import os, struct, time
from collections import Counter

PREFIX_DIR = "/home/trans/agpt-tries/gutenberg_5m_d32_radix_corpus"
RDXA_MAGIC = 0x52445841

# Custom loader that stores first_char_depth + edge_len + mass per node
def load_full(dirpath):
    meta_path = os.path.join(dirpath, "meta.bin")
    with open(meta_path, "rb") as f:
        magic, = struct.unpack("<I", f.read(4))
        assert magic == RDXA_MAGIC
        version, = struct.unpack("<i", f.read(4))
        assert version == 2
        radix_count, = struct.unpack("<i", f.read(4))
        depth_file_count, = struct.unpack("<i", f.read(4))
        f.read(8)  # total_edge_chars (i64)
        f.read(4)  # corpus_token_count
        f.read(4)  # vocab_size
        f.read(8)  # corpus_hash
        tlen, = struct.unpack("<i", f.read(4))
        f.read(tlen)

    parent_arr = [0] * (radix_count + 1)
    fcd_arr    = [0] * (radix_count + 1)
    edge_len_arr = [0] * (radix_count + 1)
    mass_arr   = [0] * (radix_count + 1)

    print(f"[load] {dirpath}: {radix_count} nodes")
    t0 = time.time()
    for d in range(depth_file_count):
        path = os.path.join(dirpath, f"radix_depth_{d:03d}.bin")
        if not os.path.exists(path):
            continue
        with open(path, "rb") as f:
            buf = f.read()
        pos = 0
        magic, = struct.unpack_from("<I", buf, pos); pos += 4
        assert magic == RDXA_MAGIC
        stored_d, = struct.unpack_from("<i", buf, pos); pos += 4
        n, = struct.unpack_from("<i", buf, pos); pos += 4
        for _ in range(n):
            rid, parent, fcd, edge_len = struct.unpack_from("<iiii", buf, pos); pos += 16
            pos += 4 * edge_len  # skip edge tokens
            edge_mass, = struct.unpack_from("<i", buf, pos); pos += 4
            ec, = struct.unpack_from("<i", buf, pos); pos += 4
            pos += 8 * ec
            parent_arr[rid] = parent
            fcd_arr[rid] = fcd
            edge_len_arr[rid] = edge_len
            mass_arr[rid] = edge_mass
    print(f"[load] loaded in {time.time() - t0:.1f}s")
    return parent_arr, fcd_arr, edge_len_arr, mass_arr, radix_count

print("Loading prefix tree...")
parent_arr, fcd_arr, elen_arr, mass_arr, n_records = load_full(PREFIX_DIR)

print("Identifying leaves (records that aren't anyone's parent)...")
is_parent = [False] * (n_records + 1)
for cid in range(1, n_records + 1):
    p = parent_arr[cid]
    if 0 < p <= n_records:
        is_parent[p] = True

leaves = [cid for cid in range(1, n_records + 1) if not is_parent[cid]]
print(f"Found {len(leaves)} leaves")

print("Computing d* (depth at which prefix becomes unique) per leaf...")
t0 = time.time()
d_star_hist = Counter()
mass1_leaf_count = 0
multi_mass_leaf_count = 0
no_unique_count = 0

# For mass-1 leaves: walk parent chain, find earliest ancestor with mass=1
# That ancestor's first_char_depth is the d* (depth at which path becomes unique)
for leaf_id in leaves:
    if mass_arr[leaf_id] != 1:
        multi_mass_leaf_count += 1
        continue
    mass1_leaf_count += 1

    # Walk parent chain to root, collect (fcd, mass) pairs in order
    chain = []
    cur = leaf_id
    while cur > 0:
        chain.append((fcd_arr[cur], mass_arr[cur]))
        cur = parent_arr[cur]
    chain.reverse()  # now in root→leaf order

    # Find first (shallowest) node in chain with mass=1
    d_star = None
    for fcd, m in chain:
        if m == 1:
            d_star = fcd
            break
    if d_star is None:
        no_unique_count += 1
    else:
        d_star_hist[d_star] += 1

print(f"[done] {time.time() - t0:.1f}s")
print(f"  mass-1 leaves analyzed:     {mass1_leaf_count}")
print(f"  multi-mass leaves (skipped): {multi_mass_leaf_count}")
print(f"  no-unique (shouldn't occur): {no_unique_count}")

print("\n=== d* histogram (depth at which mass-1 leaves first become unique) ===")
total = sum(d_star_hist.values())
ks = sorted(d_star_hist.keys())
for k in ks:
    c = d_star_hist[k]
    pct = 100.0 * c / total
    bar = '█' * min(60, int(pct * 1.5))
    print(f"  d*={k:2d}: {c:8d} ({pct:5.2f}%)  {bar}")

# Also stats
mean_d_star = sum(k * c for k, c in d_star_hist.items()) / total
median_idx = total // 2
running = 0
median_d_star = None
for k in ks:
    running += d_star_hist[k]
    if running >= median_idx and median_d_star is None:
        median_d_star = k
print(f"\n  mean d*:   {mean_d_star:.2f}")
print(f"  median d*: {median_d_star}")
