#!/usr/bin/env python3
"""Build a PG-19 training text of up to ~10G characters nested on the 20M split.

The same text mobius's scripts/pg19-more.cr builds (the 20M split's training
text, then whole train books in the seeded order of train-shuffled.names that
are not among the pieces' books, normalized as pg19-normal.cr does, the last
book cut at the mark), written as a stream: pg19-more reads its whole output
back into memory to hash it, which a 10G text can't do here. Its 100M split
(data/pg19/imm-split-100m) is a prefix of this text, so the build checks the
SHA-256 of its first 100,000,000 characters against that split's manifest and
stops if they differ: that is the check that this normalization is mobius's.

Writes DEST/train_corpus.txt and DEST/manifest.json (the appended books with
their offsets and SHA-256, the SHA-256 of the prefixes at each mark in
--marks, and of the whole text), and links the 20M split's held-out text,
chunks and vocabulary into DEST.

Usage: build_pg19_big.py PG19_ROOT SPLIT20M SPLIT100M DEST CHARS [--marks N,N,...]
"""
import argparse
import hashlib
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

TABLE = {"“": '"', "”": '"', "‘": "'", "’": "'", "—": "-", "–": "-",
         "…": "...", " ": " ", "\t": " ",
         "\u0091": "'", "\u0092": "'", "\u0093": '"', "\u0094": '"', "\u0096": "-", "\u0097": "-"}
_cache = {}
PURE = re.compile(rb"[\x20-\x7e\n]*")
OTHER = re.compile("[^\x20-\x7e\n]")


def replacement(c):
    """pg19-normal.cr Normal.replacement: the table, else the NFKD form without marks if it is ASCII letters."""
    r = _cache.get(c)
    if r is None:
        r = TABLE.get(c)
        if r is None:
            base = "".join(ch for ch in unicodedata.normalize("NFKD", c) if not unicodedata.category(ch).startswith("M"))
            r = base if base and all(("a" <= ch <= "z") or ("A" <= ch <= "Z") for ch in base) else ""
        _cache[c] = r
    return r


def normal(raw: bytes) -> bytes:
    """pg19-normal.cr Normal.call: "\\r\\n" and a lone "\\r" become "\\n"; printable ASCII stays; other
    characters are replaced, folded or dropped."""
    if PURE.fullmatch(raw):
        return raw
    s = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n").decode("utf-8", errors="replace")
    return OTHER.sub(lambda m: replacement(m.group()), s).encode("ascii")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("split20m")
    ap.add_argument("split100m")
    ap.add_argument("dest")
    ap.add_argument("chars", type=int)
    ap.add_argument("--marks", default="")
    a = ap.parse_args()
    marks = sorted({int(x) for x in a.marks.split(",") if x} | {100_000_000})
    man = json.loads(Path(a.split20m, "manifest.json").read_text())
    base = Path(a.split20m, "train_corpus.txt").read_bytes()
    assert len(base) == man["train_size"] and hashlib.sha256(base).hexdigest() == man["train_sha256"]
    want100 = json.loads(Path(a.split100m, "manifest.json").read_text())["train_sha256"]
    piece_books = {r["book"].removeprefix("train/") for r in man["train_records"]}
    names = Path(a.root, "train-shuffled.names").read_text().split()
    dest = Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    prefix_sha = {}
    size = 0
    appended = []

    def write(io, data):
        """Write data, hashing it and noting the SHA-256 of the text at every mark crossed."""
        nonlocal size
        while data:
            nxt = next((m for m in marks if m > size), None)
            take = len(data) if nxt is None else min(len(data), nxt - size)
            h.update(data[:take])
            io.write(data[:take])
            size += take
            data = data[take:]
            if nxt is not None and size == nxt:
                prefix_sha[str(nxt)] = h.copy().hexdigest()
                if nxt == 100_000_000:
                    ok = prefix_sha[str(nxt)] == want100
                    print(f"  first 100,000,000 chars sha256 {prefix_sha[str(nxt)]} "
                          f"{'matches' if ok else 'DIFFERS FROM'} imm-split-100m's", flush=True)
                    if not ok:
                        sys.exit("build_pg19_big: the normalization is not mobius's; stopped")

    with open(dest / "train_corpus.txt", "wb") as io:
        write(io, base)
        for name in names:
            if size >= a.chars:
                break
            if name in piece_books:
                continue
            p = Path(a.root, "train", name)
            if not p.exists():
                continue
            text = normal(p.read_bytes())
            if not text:
                continue
            take = min(len(text), a.chars - size)
            part = text[:take]
            appended.append({"start": size, "end": size + take, "size": take,
                             "sha256": hashlib.sha256(part).hexdigest(), "book": f"train/{name}",
                             "whole_book": take == len(text)})
            write(io, part)
            if len(appended) % 1000 == 0:
                print(f"  {len(appended)} books, {size:,} chars", flush=True)
    m = {"description": f"PG-19, {size} characters of training text: the 20M split's {len(man['train_records'])} "
                        f"pieces, then {len(appended)} whole train books (the last cut at the mark) from the seeded "
                        f"order, none a piece's book, normalized as mobius's pg19-normal.cr (Python port, "
                        f"rnd/doped-counts-depth16/tools/build_pg19_big.py); the 100M split is its prefix",
         "derived_from": a.split20m, "train_size": size, "train_sha256": h.hexdigest(),
         "prefix_sha256": prefix_sha, "appended_books": len(appended), "train_records_appended": appended}
    (dest / "manifest.json").write_text(json.dumps(m, indent=1) + "\n")
    for f in ("heldout_corpus.txt", "vocab.txt"):
        if not (dest / f).exists():
            os.link(Path(a.split20m, f), dest / f)
    print(f"{dest}: {size:,} chars, {len(appended)} books appended, sha256 {h.hexdigest()}", flush=True)


if __name__ == "__main__":
    main()
