#!/usr/bin/env python3
"""Build a multi-chunk held-out split.

Replaces the tail-only convention with K disjoint randomly-placed
chunks totaling holdout_frac of the corpus. More representative of
the corpus distribution than tail-only, while keeping chunks large
enough (≫ LM context window) for boundary effects to be negligible.

Usage:
    python3 build_multi_chunk_split.py \\
        --corpus data/input.txt \\
        --outdir rnd/kn-shakespeare-baseline \\
        --k 10 --holdout-frac 0.05 --seed 42

Outputs (in outdir/):
    train_corpus.txt       — corpus with K chunks removed, concatenated
    heldout_corpus.txt     — the K chunks concatenated (for eval-as-one-text)
    heldout_chunks/        — one file per chunk (for eval-per-chunk)
    manifest.json          — chunk positions, sizes, seed, sha256s

Provenance: anyone can re-run with the same args + corpus → bit-identical
outputs (modulo OS-level write nondeterminism).
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, sys
from pathlib import Path


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="path to corpus text file")
    ap.add_argument("--outdir", required=True, help="output directory")
    ap.add_argument("--k", type=int, default=10, help="number of held-out chunks")
    ap.add_argument("--holdout-frac", type=float, default=0.05, help="total held-out fraction")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = ap.parse_args()

    with open(args.corpus, "rb") as f:
        corpus = f.read()
    N = len(corpus)
    corpus_sha = sha256(corpus)
    holdout_total = int(N * args.holdout_frac)
    chunk_size = holdout_total // args.k
    if chunk_size < 64:
        sys.exit(f"chunk_size={chunk_size} is too small; reduce --k or raise --holdout-frac")

    # Seeded-random non-overlapping placement via rejection sampling.
    rng = random.Random(args.seed)
    positions: list[int] = []
    max_attempts = 100 * args.k
    attempts = 0
    while len(positions) < args.k:
        attempts += 1
        if attempts > max_attempts:
            sys.exit(f"could not place {args.k} non-overlapping chunks of {chunk_size} after {max_attempts} attempts")
        pos = rng.randint(0, N - chunk_size)
        if all(abs(pos - p) >= chunk_size for p in positions):
            positions.append(pos)
    positions.sort()

    # Build heldout chunks + manifest.
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    chunks_dir = outdir / "heldout_chunks"
    chunks_dir.mkdir(exist_ok=True)
    chunks: list[bytes] = []
    chunk_records = []
    for i, p in enumerate(positions):
        chunk = corpus[p : p + chunk_size]
        chunks.append(chunk)
        chunk_path = chunks_dir / f"chunk_{i:02d}.txt"
        chunk_path.write_bytes(chunk)
        chunk_records.append({
            "index": i,
            "start": p,
            "end": p + chunk_size,
            "size": chunk_size,
            "sha256": sha256(chunk),
        })

    # Build train_corpus.txt: concatenate the non-held-out spans in order.
    train_pieces: list[bytes] = []
    last_end = 0
    for p in positions:
        train_pieces.append(corpus[last_end:p])
        last_end = p + chunk_size
    train_pieces.append(corpus[last_end:])
    train = b"".join(train_pieces)
    (outdir / "train_corpus.txt").write_bytes(train)

    # Heldout as one concatenated text (for single-pass eval).
    heldout = b"".join(chunks)
    (outdir / "heldout_corpus.txt").write_bytes(heldout)

    # Manifest.
    manifest = {
        "source_corpus": os.path.abspath(args.corpus),
        "source_corpus_sha256": corpus_sha,
        "source_corpus_size": N,
        "k": args.k,
        "holdout_frac": args.holdout_frac,
        "chunk_size": chunk_size,
        "seed": args.seed,
        "chunks": chunk_records,
        "train_size": len(train),
        "train_sha256": sha256(train),
        "heldout_size": len(heldout),
        "heldout_sha256": sha256(heldout),
        "seams_in_train": args.k,
        "seam_char_neighborhood_for_d16": args.k * 32,
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"corpus:    {N:>10,} chars  sha256={corpus_sha[:12]}")
    print(f"k:         {args.k}")
    print(f"chunk size: {chunk_size:>10,} chars  (positions: {positions})")
    print(f"train:     {len(train):>10,} chars  ({100 * len(train) / N:.2f}%)")
    print(f"heldout:   {len(heldout):>10,} chars  ({100 * len(heldout) / N:.2f}%)")
    print(f"  written to {outdir}/")


if __name__ == "__main__":
    main()
