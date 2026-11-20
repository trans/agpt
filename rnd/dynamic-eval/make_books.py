#!/usr/bin/env python3
"""Whole PG-19 test books for dyneval.py, normalized as the held-out pieces are.

Normalization is a port of mobius's scripts/pg19-normal.cr (imm's text): line
ends to \\n; curly quotes straight, em and en dashes "-", the ellipsis "...",
the no-break space and the tab a space (and the same characters in
Windows-1252's C1 codes); a letter with accents loses them when its NFKD form
without marks is ASCII letters; any other character outside printable ASCII
and \\n is dropped.

Books: every PG-19 test book whose normalized text has at least SKIP + LENGTH
characters, sorted by file name, the first N of them; each contributes
characters [SKIP, SKIP + LENGTH), so every book has the same length and the
--past other control (book i steps on book i+1) stays aligned. SKIP leaves out
the front matter, as the held-out pieces start at least 10,000 characters in.

Usage: make_books.py --src DIR --out DIR [--n 20] [--length 300000] [--skip 10000]
Writes OUT/chunk_NN.txt and OUT/manifest.json (book, characters, SHA-256).
"""
import argparse
import hashlib
import json
import unicodedata
from pathlib import Path

TABLE = {"“": '"', "”": '"', "‘": "'", "’": "'", "—": "-", "–": "-",
         "…": "...", " ": " ", "\t": " ",
         "\u0091": "'", "\u0092": "'", "\u0093": '"', "\u0094": '"', "\u0096": "-", "\u0097": "-"}


def replacement(c, cache={}):
    if c not in cache:
        if c in TABLE:
            cache[c] = TABLE[c]
        else:
            base = "".join(x for x in unicodedata.normalize("NFKD", c) if not unicodedata.combining(x))
            cache[c] = base if base and all(x.isascii() and x.isalpha() for x in base) else ""
    return cache[c]


def normalize(raw: bytes) -> str:
    text = raw.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    return "".join(c if (" " <= c <= "~" or c == "\n") else replacement(c) for c in text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--length", type=int, default=300_000)
    ap.add_argument("--skip", type=int, default=10_000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    books = []
    for f in sorted(Path(a.src).glob("*.txt")):
        if f.stat().st_size < a.skip + a.length:
            continue
        t = normalize(f.read_bytes())
        if len(t) < a.skip + a.length:
            continue
        piece = t[a.skip:a.skip + a.length]
        i = len(books)
        (out / f"chunk_{i:02d}.txt").write_text(piece, encoding="utf-8")
        books.append({"chunk": i, "book": f.name, "book_chars": len(t), "start": a.skip, "length": a.length,
                      "sha256": hashlib.sha256(piece.encode()).hexdigest()})
        if len(books) == a.n:
            break
    (out / "manifest.json").write_text(json.dumps({"src": str(a.src), "skip": a.skip, "length": a.length,
                                                   "books": books}, indent=1) + "\n")
    print(f"{len(books)} books of {a.length} characters -> {out}")


if __name__ == "__main__":
    main()
