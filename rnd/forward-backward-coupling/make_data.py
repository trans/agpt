#!/usr/bin/env python3
"""Write the reversed texts the backward model trains and is scored on.

data/pg19/a1m-rev/train_corpus.txt     the 1M PG-19 slice, reversed
data/pg19/imm-split-rev/heldout_corpus.txt, heldout_chunks/chunk_NN.txt
                                       the held-out carve, each chunk reversed
                                       on its own (the chunks stay separate books)
data/pg19/imm-split-rev/vocab.txt      the vocab file, unchanged
Reversal is by character; the texts are ASCII. Prints each file's SHA-256.
"""
import hashlib
from pathlib import Path

P = Path("data/pg19")


def put(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path}")


put(P / "a1m-rev/train_corpus.txt", (P / "a1m/train_corpus.txt").read_text(encoding="utf-8")[::-1])
src = P / "imm-split"
dst = P / "imm-split-rev"
chunks = sorted((src / "heldout_chunks").glob("chunk_*.txt"))
rev = [c.read_text(encoding="utf-8")[::-1] for c in chunks]
for c, r in zip(chunks, rev):
    put(dst / "heldout_chunks" / c.name, r)
put(dst / "heldout_corpus.txt", "".join(rev))
put(dst / "vocab.txt", (src / "vocab.txt").read_text(encoding="utf-8"))
