#!/usr/bin/env python3
"""Doped trie counts: a training corpus's contexts with next-token counts from a larger text.

The table agpt_minibatch_train.py's trie_targets_corpus builds in memory
(trie_plan.context_targets_from), built once by streaming the larger text, so
the text may be far larger than memory (10G characters), and written to disk
for experimental.trie_targets_table. The contexts are the training corpus's:
every tokens[i:i+L], 0 <= i < n_starts = chars - seq_len - 1 (the trainer's
window starts), 1 <= L <= seq_len. Their next-token counts n_ref(c, a) are
counted over the reference text: every position where c occurs and a follows.

One pass over the reference writes a table at every mark: the counts over
its first `mark` characters (the events whose next token lies before the
mark), so nested texts that are prefixes of one file (100M, 1G, 10G) come
from one pass. Each reference position is followed leafward one character at
a time and dropped at the first context the corpus lacks: the corpus's
contexts are closed under prefixes, so no longer context can match either.
The walk runs in torch (--device cuda for a 10G text); counts are kept dense,
[contexts, vocab] int32, one band of depths per pass over the text when the
whole table exceeds --max-dense-gib.

Each table (PREFIX-<mark>.npz) holds ctx [seq_len, n_starts] int32 (event
(i, L)'s context id among the contexts found in the reference, -1 if not
found), the CSR indptr int64 / child uint8 / count int64 over those ids,
n_all, n_single (contexts found once), and meta (JSON: corpus and reference
SHA-256s, seq_len, vocab, n_starts, mark, and per-depth coverage). The same
arrays as context_targets_from, which `--check` compares against.

Usage: doped_counts.py --corpus C --vocab V --seq-len T --ref R --marks M,M,... --out PREFIX
       [--device cuda] [--max-dense-gib G] [--max-corpus-chars N] [--check]
       (--check: also run context_targets_from at every mark and require equality;
       small corpora only)
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trie_plan as tp  # noqa: E402
from agpt_ppl import build_vocab  # noqa: E402


def sha256_file(path, limit=None):
    h = hashlib.sha256()
    left = limit
    with open(path, "rb") as f:
        while True:
            n = 1 << 24 if left is None else min(1 << 24, left)
            if n == 0:
                break
            b = f.read(n)
            if not b:
                break
            h.update(b)
            if left is not None:
                left -= len(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--seq-len", type=int, required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--marks", required=True, help="comma-separated character counts; 'end' for the whole file")
    ap.add_argument("--out", required=True, help="output prefix: PREFIX-<mark>.npz")
    ap.add_argument("--max-corpus-chars", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=1 << 24)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-dense-gib", type=float, default=1.75)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    T = a.seq_len
    char_to_id, V = build_vocab(a.vocab)
    if V > 256:
        sys.exit("doped_counts: vocab does not fit uint8")
    text = Path(a.corpus).read_text(encoding="utf-8", errors="replace")
    if a.max_corpus_chars:
        text = text[: a.max_corpus_chars]
    tok = np.array([char_to_id.get(c, 0) for c in text], dtype=np.uint8)
    n_starts = tok.size - T - 1
    ref = np.memmap(a.ref, dtype=np.uint8, mode="r")
    nr = ref.size
    marks = sorted({nr if m == "end" else min(int(m), nr) for m in a.marks.split(",")})
    lut = np.full(256, 255, dtype=np.uint8)
    for ch, i in char_to_id.items():
        b = ch.encode("utf-8")
        if len(b) == 1:
            lut[b[0]] = i

    t0 = time.time()
    tk = tok.astype(np.uint64) + np.uint64(1)
    h = np.zeros(n_starts, dtype=np.uint64)
    ctx = np.empty((T, n_starts), dtype=np.int32)
    us, bases = [], []
    base = 0
    for L in range(1, T + 1):
        h = h * tp._HB + tk[L - 1:L - 1 + n_starts]
        u, inv = np.unique(h, return_inverse=True)
        ctx[L - 1] = base + inv
        us.append(u)
        bases.append(base)
        base += u.size
    n_all = base
    del h
    # depth bands, each counted in its own pass with a dense [contexts, V] int32 table of at most
    # --max-dense-gib (every pass walks from depth 1: the walk prunes)
    bands, cur, size = [], [], 0
    for L in range(1, T + 1):
        b = us[L - 1].size * V * 4
        if cur and size + b > a.max_dense_gib * 2**30:
            bands.append(cur)
            cur, size = [], 0
        cur.append(L)
        size += b
    bands.append(cur)
    print(f"doped_counts: corpus {a.corpus} ({tok.size} chars, {n_starts} starts), seq_len {T}, V {V}: "
          f"{n_all} contexts ({', '.join(str(u.size) for u in us)} by depth); dense counts "
          f"{n_all * V * 4 / 2**30:.2f} GiB in {len(bands)} band(s) {[(b[0], b[-1]) for b in bands]}; "
          f"ref {a.ref} ({nr} chars), marks {marks}; device {a.device}; contexts built in "
          f"{time.time() - t0:.1f}s", flush=True)
    corpus_sha = sha256_file(a.corpus) if not a.max_corpus_chars else hashlib.sha256(text.encode()).hexdigest()
    vocab_sha = sha256_file(a.vocab)

    import torch
    dev = torch.device(a.device)
    hb = int(tp._HB) - (1 << 64)                 # the same bits as int64: torch's int64 products wrap
    u_s, u_o = [], []
    for u in us:
        ui = u.view(np.int64)
        o = np.argsort(ui, kind="stable")
        u_s.append(torch.from_numpy(ui[o]).to(dev))
        u_o.append(torch.from_numpy(o.astype(np.int64)).to(dev))
    lut_t = torch.from_numpy(lut.astype(np.int64)).to(dev)
    hits_by_depth = np.zeros(T, dtype=np.int64)
    parts = {m: [] for m in marks}               # per mark: (keys, counts) of each band, in depth order
    t1 = time.time()
    for band in bands:
        L0, L1 = band[0], band[-1]
        off = bases[L0 - 1] * V
        span = (bases[L1 - 1] + us[L1 - 1].size) * V - off
        dense = torch.zeros(span, dtype=torch.int32, device=dev)
        lo = 0
        for mark in marks:
            hi = mark
            for s in range(max(0, lo - T), hi, a.chunk):
                e = min(s + a.chunk, hi)
                seg = lut_t[torch.from_numpy(np.asarray(ref[s:min(e + T, hi)]).astype(np.int64)).to(dev)]
                if bool((seg == 255).any()):
                    sys.exit(f"doped_counts: ref bytes outside the vocab near {s}")
                n = seg.numel()
                alive = torch.arange(e - s, device=dev, dtype=torch.int64)
                hh = torch.zeros(alive.numel(), device=dev, dtype=torch.int64)
                for L in range(1, L1 + 1):
                    keep = alive + L < n                  # the next token exists before the mark
                    alive, hh = alive[keep], hh[keep]
                    if alive.numel() == 0:
                        break
                    hh = hh * hb + (seg[alive + L - 1] + 1)
                    u = u_s[L - 1]
                    j = torch.searchsorted(u, hh).clamp_(max=u.numel() - 1)
                    hit = u[j] == hh
                    alive, hh, j = alive[hit], hh[hit], j[hit]
                    if alive.numel() == 0:
                        break
                    if L >= L0:
                        c = (alive + (s + L)) >= lo        # events whose next token is in [lo, hi)
                        k = (u_o[L - 1][j[c]] + bases[L - 1]) * V + seg[alive[c] + L] - off
                        hits_by_depth[L - 1] += k.numel()
                        uk, uc = torch.unique(k, return_counts=True)
                        dense.index_add_(0, uk, uc.to(torch.int32))
                if (s // a.chunk) % 64 == 0:
                    el = time.time() - t1
                    print(f"  band {L0}-{L1}: {e:,} / {nr:,} chars, {el:.0f}s", flush=True)
            lo = hi
            if int(dense.max()) >= 2**31 - 1:
                sys.exit("doped_counts: a count reached the int32 limit")
            nz = torch.nonzero(dense).squeeze(1)
            parts[mark].append(((nz + off).cpu().numpy(), dense[nz].long().cpu().numpy()))
            print(f"  band {L0}-{L1} at {mark:,}: {nz.numel()} pairs ({time.time() - t1:.0f}s)", flush=True)
        del dense
    for mark in marks:
        nz = np.concatenate([p[0] for p in parts[mark]])
        cnt = np.concatenate([p[1] for p in parts[mark]])
        write_table(a, nz, cnt, ctx, n_all, V, T, n_starts, mark, corpus_sha, vocab_sha, nr, hits_by_depth)
        parts[mark] = None
        if a.check:
            check(a, tok, n_starts, T, V, ref, mark)
    print(f"doped_counts: done in {time.time() - t0:.0f}s", flush=True)


def write_table(a, nz, cnt, ctx, n_all, V, T, n_starts, mark, corpus_sha, vocab_sha, nr, hits):
    t0 = time.time()
    cid = nz // V
    per = np.bincount(cid, minlength=n_all)
    found = per > 0
    new_id = np.full(n_all, -1, dtype=np.int64)
    new_id[found] = np.arange(int(found.sum()))
    ctx_out = new_id[ctx].astype(np.int32)
    tot = np.bincount(cid, weights=cnt, minlength=n_all)
    indptr = np.r_[0, np.cumsum(per[found])].astype(np.int64)
    child = (nz % V).astype(np.uint8)
    n_single = int((tot[found] == 1).sum())
    cover = []
    for L in range(T):
        n_ref = tot[ctx[L]]
        q = np.percentile(n_ref, [50, 90])
        cover.append({"depth": L + 1, "median_n_ref": float(q[0]), "p90_n_ref": float(q[1]),
                      "frac_n_ref_ge2": float((n_ref >= 2).mean()), "frac_n_ref_ge10": float((n_ref >= 10).mean()),
                      "frac_n_ref_ge100": float((n_ref >= 100).mean()), "missing": int((n_ref == 0).sum())})
    meta = {"corpus": a.corpus, "corpus_sha256": corpus_sha, "max_corpus_chars": a.max_corpus_chars,
            "vocab": a.vocab, "vocab_sha256": vocab_sha, "seq_len": T, "V": V, "n_starts": n_starts,
            "ref": a.ref, "ref_chars": int(nr), "mark": int(mark),
            "ref_prefix_sha256": sha256_file(a.ref, mark), "n_all": int(n_all), "n_found": int(found.sum()),
            "n_single": n_single, "n_pairs": int(nz.size), "hits_by_depth": hits.tolist(), "coverage": cover}
    out = f"{a.out}-{mark}.npz"
    np.savez(out, ctx=ctx_out, indptr=indptr, child=child, count=cnt, n_all=np.int64(n_all),
             n_single=np.int64(n_single), meta=np.array(json.dumps(meta)))
    print(f"  table at {mark:,}: {out}, {int(found.sum())} of {n_all} contexts found, {nz.size} pairs, "
          f"{n_single} found once ({time.time() - t0:.0f}s)", flush=True)
    for c in cover:
        print(f"    depth {c['depth']:2}: median n_ref {c['median_n_ref']:.0f}, p90 {c['p90_n_ref']:.0f}, "
              f">=2 {c['frac_n_ref_ge2']:.3f}, >=10 {c['frac_n_ref_ge10']:.3f}, >=100 {c['frac_n_ref_ge100']:.3f}",
              flush=True)


def check(a, tok, n_starts, T, V, ref, mark):
    t0 = time.time()
    want = tp.context_targets_from(tok, n_starts, T, V, _ref_tokens(a, ref, mark))
    got = np.load(f"{a.out}-{mark}.npz")
    names = ("ctx", "indptr", "child", "count")
    ok = all(np.array_equal(got[k], w) for k, w in zip(names, want[:4]))
    ok = ok and int(got["n_all"]) == want[4] and int(got["n_single"]) == want[5]
    print(f"  check against trie_plan.context_targets_from at {mark:,}: {'EQUAL' if ok else 'DIFFERENT'} "
          f"({time.time() - t0:.0f}s)", flush=True)
    if not ok:
        sys.exit(1)


def _ref_tokens(a, ref, mark):
    char_to_id, _ = build_vocab(a.vocab)
    lut = np.full(256, 255, dtype=np.uint8)
    for ch, i in char_to_id.items():
        b = ch.encode("utf-8")
        if len(b) == 1:
            lut[b[0]] = i
    return lut[np.asarray(ref[:mark])]


if __name__ == "__main__":
    main()
