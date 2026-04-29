"""Compute per-character-offset joint-mass lookup table for AGPT training.

For each character c in the prefix tree's edge_tokens_flat (the flattened
representation of all radix-tree edges), records the AVERAGE suffix-tree
mass at the complementary depth (D_max + 1 - depth_of_c), averaged across
all corpus positions whose path traverses char c.

The output table is indexed by char offset and used at training time as
a per-query suffix factor: joint = edge_mass[r] * char_suffix_mass[c],
where c is the query's char offset.

This replaces the AGPT_JOINT_MASS aggregate-mean proxy (which used
trie-wide per-depth means as a global suffix factor) with a per-position
lookup that captures actual suffix-side variation.

Usage:
    python3 compute_char_suffix_mass.py \\
        --prefix-trie /home/trans/agpt-tries/shakespeare_d32_radix_corpus \\
        --suffix-trie /home/trans/agpt-tries/shakespeare_d32_suffix_radix \\
        --corpus /home/trans/Projects/agpt/data/input.txt \\
        --out /tmp/shakespeare_d32_char_suffix_mass.bin
"""
import os, struct, sys, time
import argparse

RDXA_MAGIC = 0x52445841


def load_trie(dirpath):
    """Load a radix trie. Returns dict with parents, fcd, edge_lens, edge_tokens_flat,
    edge_mass, edge_starts, vocab_size, depth_file_count, radix_count."""
    meta_path = os.path.join(dirpath, "meta.bin")
    with open(meta_path, "rb") as f:
        magic, = struct.unpack("<I", f.read(4))
        assert magic == RDXA_MAGIC, f"bad magic in {meta_path}"
        version, = struct.unpack("<i", f.read(4))
        assert version == 2
        radix_count, = struct.unpack("<i", f.read(4))
        depth_file_count, = struct.unpack("<i", f.read(4))
        total_edge_chars, = struct.unpack("<q", f.read(8))
        f.read(4)  # corpus_token_count
        vocab_size, = struct.unpack("<i", f.read(4))
        f.read(8)  # corpus_hash
        tlen, = struct.unpack("<i", f.read(4))
        f.read(tlen)

    parents      = [0] * (radix_count + 1)
    fcd_arr      = [0] * (radix_count + 1)
    edge_len_arr = [0] * (radix_count + 1)
    edge_starts  = [0] * (radix_count + 1)
    edge_mass    = [0] * (radix_count + 1)
    edge_tokens_flat = [0] * total_edge_chars

    edge_fill = 0
    for d in range(depth_file_count):
        path = os.path.join(dirpath, f"radix_depth_{d:03d}.bin")
        if not os.path.exists(path):
            continue
        with open(path, "rb") as f:
            buf = f.read()
        pos = 0
        struct.unpack_from("<I", buf, pos); pos += 4
        struct.unpack_from("<i", buf, pos); pos += 4
        n, = struct.unpack_from("<i", buf, pos); pos += 4
        for _ in range(n):
            rid, parent, fcd, edge_len = struct.unpack_from("<iiii", buf, pos); pos += 16
            parents[rid] = parent
            fcd_arr[rid] = fcd
            edge_len_arr[rid] = edge_len
            edge_starts[rid] = edge_fill
            for j in range(edge_len):
                tok, = struct.unpack_from("<i", buf, pos); pos += 4
                edge_tokens_flat[edge_fill + j] = tok
            edge_fill += edge_len
            mass, = struct.unpack_from("<i", buf, pos); pos += 4
            edge_mass[rid] = mass
            ec, = struct.unpack_from("<i", buf, pos); pos += 4
            pos += 8 * ec  # skip counts entries

    # Build child lookup: parent_id, first_token → child_id
    # For trie walking, need to find children by first token of their edge.
    children = [{} for _ in range(radix_count + 1)]
    for rid in range(1, radix_count):
        p = parents[rid]
        if p < 0 or p >= radix_count + 1:
            continue
        first_tok = edge_tokens_flat[edge_starts[rid]] if edge_len_arr[rid] > 0 else -1
        if first_tok >= 0:
            children[p][first_tok] = rid

    return {
        "parents": parents,
        "fcd": fcd_arr,
        "edge_len": edge_len_arr,
        "edge_starts": edge_starts,
        "edge_mass": edge_mass,
        "edge_tokens_flat": edge_tokens_flat,
        "children": children,
        "vocab_size": vocab_size,
        "depth_file_count": depth_file_count,
        "radix_count": radix_count,
        "total_edge_chars": total_edge_chars,
    }


def walk_trie(trie, tokens):
    """Walk the trie matching the token sequence. Returns list of
    (depth, radix_node, char_offset) at each step (depth 1, 2, ...)."""
    edge_tokens_flat = trie["edge_tokens_flat"]
    edge_starts = trie["edge_starts"]
    edge_len_arr = trie["edge_len"]
    edge_mass = trie["edge_mass"]
    children = trie["children"]

    cur = 0  # root
    cur_depth = 0
    edge_pos = -1  # offset into current node's edge if we're mid-edge
    visits = []  # (depth, node, char_offset, mass)

    i = 0
    while i < len(tokens):
        if edge_pos >= 0 and edge_pos < edge_len_arr[cur]:
            # Continue down current node's edge
            expected = edge_tokens_flat[edge_starts[cur] + edge_pos]
            if expected != tokens[i]:
                break
            cur_depth += 1
            char_off = edge_starts[cur] + edge_pos
            visits.append((cur_depth, cur, char_off, edge_mass[cur]))
            edge_pos += 1
            i += 1
            continue
        # Need to descend into a child
        ch = children[cur].get(tokens[i])
        if ch is None or ch == 0:
            break
        cur = ch
        edge_pos = 0
        # Now consume this character at the start of the new edge
        cur_depth += 1
        char_off = edge_starts[cur] + edge_pos
        visits.append((cur_depth, cur, char_off, edge_mass[cur]))
        edge_pos += 1
        i += 1
    return visits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix-trie", required=True)
    ap.add_argument("--suffix-trie", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-depth", type=int, default=32)
    args = ap.parse_args()

    print(f"Loading prefix trie from {args.prefix_trie}...", flush=True)
    t0 = time.time()
    pfx = load_trie(args.prefix_trie)
    print(f"  {pfx['radix_count']} nodes, {pfx['total_edge_chars']} edge chars, {time.time()-t0:.1f}s")

    print(f"Loading suffix trie from {args.suffix_trie}...", flush=True)
    t0 = time.time()
    sfx = load_trie(args.suffix_trie)
    print(f"  {sfx['radix_count']} nodes, {time.time()-t0:.1f}s")

    print(f"Loading corpus from {args.corpus}...", flush=True)
    with open(args.corpus, "r") as f:
        corpus = f.read()
    print(f"  {len(corpus)} chars")

    # Build char→token mapping. Same vocab as trie expects.
    chars_sorted = sorted(set(corpus))
    char_to_tok = {c: i for i, c in enumerate(chars_sorted)}
    if len(chars_sorted) != pfx["vocab_size"]:
        print(f"  warning: corpus vocab {len(chars_sorted)} differs from trie vocab {pfx['vocab_size']}")
    corpus_toks = [char_to_tok[c] for c in corpus]

    D = args.max_depth
    N = len(corpus_toks)

    # Accumulators
    sums = [0.0] * pfx["total_edge_chars"]
    counts = [0] * pfx["total_edge_chars"]

    # For each corpus position p (1-indexed semantics), walk both trees.
    # Prefix walk: forward chars ENDING at p, i.e., corpus[p-D..p-1] (D chars before p).
    #   In our 0-indexed implementation: prefix window = corpus_toks[p-D : p].
    # Suffix walk (--reverse builder): prefix tree on REVERSED corpus.
    #   Walking the suffix tree from root using chars X1, X2, ... finds positions
    #   in REVERSED corpus where the chars start with X1 X2. In original corpus,
    #   this means reading BACKWARD: position q in reversed = N-1-q in original.
    #   So suffix-walk-depth k from "position p" = chars going BACKWARD from p
    #   in original = corpus[p-1], corpus[p-2], corpus[p-3], ...
    # For complementary alignment: at prefix-tree depth d_p, the analogous
    # suffix-tree depth is D_max + 1 - d_p (since they together span d_p + d_s = D_max+1
    # if we treat the boundary as shared).
    # Actually, the user's "ABCD = DCBA" example had a 4-char window with d_p+d_s=4
    # at the boundaries (no +1). But that was for a window WHOSE LENGTH = 4. Here our
    # max trie depth is D, and a "window" passing through p has D chars total, so
    # d_p + d_s = D+1 at the boundary if positions are shared, or d_p + d_s = D if
    # they are not.
    #
    # We use d_p + d_s = D + 1 (complementary depth, sharing one position): for
    # d_p ∈ [1, D], d_s = D + 1 - d_p ∈ [D, 1]. Same convention as the C++
    # trainer's mean_edge_mass[D_max - d_q] (where D_max = depth_file_count = D+1).

    print("\nWalking corpus...", flush=True)
    t0 = time.time()
    n_walked = 0
    for p in range(D, N):
        # Prefix window: chars at corpus positions p-D..p-1 (D chars before p)
        prefix_window = corpus_toks[p - D : p]
        # Walk prefix tree: visits[k-1] = (k, node, char_off, mass) at depth k
        pfx_visits = walk_trie(pfx, prefix_window)
        # Suffix walk: chars going backward from p, i.e., corpus[p-1], corpus[p-2], ...
        # For depth d_s, we use chars corpus[p-1], corpus[p-2], ..., corpus[p-d_s].
        suffix_window = list(reversed(prefix_window))
        sfx_visits = walk_trie(sfx, suffix_window)

        # Index suffix visits by depth for fast lookup
        sfx_mass_at_depth = {v[0]: v[3] for v in sfx_visits}

        # For each prefix visit at depth d_p, get suffix mass at d_s = D + 1 - d_p
        for d_p, _r, char_off, _m in pfx_visits:
            d_s = D + 1 - d_p
            sm = sfx_mass_at_depth.get(d_s, 1)  # default to 1 if walk didn't reach
            sums[char_off] += sm
            counts[char_off] += 1

        n_walked += 1
        if n_walked % 100000 == 0:
            print(f"  {n_walked}/{N} positions ({n_walked*100/N:.1f}%, {time.time()-t0:.1f}s)", flush=True)

    print(f"  done: {n_walked} positions in {time.time()-t0:.1f}s")

    # Normalize: char_suffix_mass[c] = sums[c] / counts[c], or 1.0 if count==0
    print("\nNormalizing...", flush=True)
    out = [1.0] * pfx["total_edge_chars"]
    for c in range(pfx["total_edge_chars"]):
        if counts[c] > 0:
            out[c] = sums[c] / counts[c]

    # Also report some diagnostics
    sample_depths = [1, 5, 11, 16, 21, 27, 32]
    # Group char offsets by their trie depth
    depth_of_char = [-1] * pfx["total_edge_chars"]
    for r in range(1, pfx["radix_count"] + 1):
        if r >= len(pfx["fcd"]): continue
        st = pfx["edge_starts"][r]
        L = pfx["edge_len"][r]
        for j in range(L):
            depth_of_char[st + j] = pfx["fcd"][r] + j

    print("\n=== char_suffix_mass distribution by trie depth ===")
    print(f"{'d_p':>4} {'comp_d':>7} {'n_chars':>10} {'mean(csm)':>10} {'min':>8} {'max':>10}")
    for d in sample_depths:
        chars_at_d = [c for c in range(pfx["total_edge_chars"]) if depth_of_char[c] == d]
        if not chars_at_d:
            print(f"{d:>4} {D+1-d:>7} {0:>10} (no chars)")
            continue
        csms = [out[c] for c in chars_at_d]
        mean_csm = sum(csms) / len(csms)
        print(f"{d:>4} {D+1-d:>7} {len(chars_at_d):>10} {mean_csm:>10.2f} {min(csms):>8.2f} {max(csms):>10.2f}")

    # Save: int64 total_edge_chars + array of float64 length total_edge_chars
    print(f"\nSaving to {args.out}...", flush=True)
    with open(args.out, "wb") as f:
        f.write(struct.pack("<q", pfx["total_edge_chars"]))
        for v in out:
            f.write(struct.pack("<d", v))
    print(f"  done. {pfx['total_edge_chars'] * 8 + 8} bytes")


if __name__ == "__main__":
    main()
