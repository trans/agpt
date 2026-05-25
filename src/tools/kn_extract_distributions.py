#!/usr/bin/env python3
"""
Extract KN-smoothed P(w | context) distributions from a KenLM ARPA file.

Reads contexts from --contexts-file (one per line, each a Python repr-style
sequence of chars joined by spaces — same tokenization as our KenLM training)
and for each context outputs a distribution over the corpus vocab as a
float32 binary blob.

Output format (binary):
    magic:    u32 = 0x4B4E4453 ("KNDS")
    n_ctx:    u32  number of contexts (rows)
    vocab:    u32  size of vocab
    chars:    u8[vocab]  the char vocabulary (sorted)
    probs:    float32[n_ctx * vocab]  P(w | context) row-major

The probs are linear-domain (not log). Sum per row ≈ 1.0 (KN is properly normalized).

Usage:
    src/tools/kn_extract_distributions.py \\
        --arpa /tmp/kn_o6.arpa \\
        --contexts-file /tmp/trie_contexts.txt \\
        --vocab data/input.txt \\
        --out /tmp/kn_distributions.bin
"""

import argparse
import struct
import sys
import time
from pathlib import Path


def parse_arpa(arpa_path):
    """
    Parse an ARPA file. Returns:
        ngrams: dict {tuple_of_chars: log10_prob}
        backoffs: dict {tuple_of_chars: log10_backoff}
        max_order: int
    """
    ngrams = {}
    backoffs = {}
    current_order = None
    max_order = 0

    # First pass: find max_order from \data\ header
    with open(arpa_path) as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('ngram '):
                # e.g. "ngram 6=1000616"
                order = int(line.split(' ')[1].split('=')[0])
                max_order = max(max_order, order)
            elif line.startswith('\\1-grams:'):
                break

    # Second pass: load n-grams + backoffs
    with open(arpa_path) as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            if line.startswith('\\') and line.endswith('-grams:'):
                current_order = int(line[1:line.index('-')])
                continue
            if line == '\\data\\' or line == '\\end\\':
                current_order = None
                continue
            if current_order is None:
                continue

            parts = line.split('\t')
            if len(parts) < 2:
                continue
            log_prob = float(parts[0])
            words = tuple(parts[1].split(' '))
            ngrams[words] = log_prob
            # Backoff only stored for orders < max_order (highest order has no backoff field)
            if current_order < max_order and len(parts) >= 3:
                log_backoff = float(parts[2])
                if log_backoff != 0.0:
                    backoffs[words] = log_backoff

    return ngrams, backoffs, max_order


def score_kn(ngrams, backoffs, max_order, context, word):
    """
    Score log10 P(word | context) using KN backoff.

    context: tuple of chars (oldest first, most recent last). Truncated to max_order-1.
    word: single char string.

    Standard KN backoff: if (context, word) is seen, return its prob; else use
    the backoff weight associated with `context` (if any) and recurse with the
    oldest context token dropped.
    """
    if len(context) > max_order - 1:
        context = context[-(max_order - 1):]

    ngram = context + (word,)
    if ngram in ngrams:
        return ngrams[ngram]

    if not context:
        return ngrams.get(('<unk>',), -10.0)

    bo = backoffs.get(context, 0.0)  # 0.0 in log10 = 1.0 linear (no penalty if missing)
    return bo + score_kn(ngrams, backoffs, max_order, context[1:], word)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arpa', required=True, help='KenLM .arpa file')
    ap.add_argument('--contexts-file', required=True,
                    help='Contexts to score (one per line, chars space-separated)')
    ap.add_argument('--vocab', required=True,
                    help='Path to training corpus; vocab inferred from unique chars')
    ap.add_argument('--out', required=True, help='Output binary file')
    ap.add_argument('--limit', type=int, default=0,
                    help='Limit number of contexts processed (0=all)')
    args = ap.parse_args()

    # Build vocab from corpus (sorted, no newlines).
    # KenLM uses '_' as the space sentinel (because whitespace is its delimiter);
    # the extracted distributions therefore expose ' ' under the key '_'.
    text = Path(args.vocab).read_text(encoding='utf-8', errors='replace')
    vocab = sorted(set(c for c in text if c != '\n'))
    # Substitute ' ' -> '_' to match the KenLM training tokenization
    vocab = sorted(set('_' if c == ' ' else c for c in vocab))
    n_vocab = len(vocab)
    print(f'Vocab: {n_vocab} chars: {"".join(vocab)[:80]}', file=sys.stderr)

    print(f'Parsing ARPA from {args.arpa}...', file=sys.stderr)
    t0 = time.time()
    ngrams, backoffs, max_order = parse_arpa(args.arpa)
    print(f'  {len(ngrams)} n-grams, {len(backoffs)} backoffs, max_order={max_order} '
          f'({time.time() - t0:.1f}s)', file=sys.stderr)

    # Read contexts. Each line: a context string (chars verbatim, no spaces between).
    # Empty line = empty context (unigram). Spaces in the input are mapped to '_'
    # to match the KenLM tokenization.
    contexts = []
    vocab_set = set(vocab)
    with open(args.contexts_file) as f:
        for line in f:
            line = line.rstrip('\n')
            ctx_chars = ['_' if c == ' ' else c for c in line]
            ctx = tuple(c for c in ctx_chars if c in vocab_set)
            contexts.append(ctx)

    if args.limit > 0:
        contexts = contexts[:args.limit]
    n_ctx = len(contexts)
    print(f'Contexts to score: {n_ctx}', file=sys.stderr)

    # Score every (context, word) pair
    print(f'Scoring {n_ctx * n_vocab} (context, word) pairs...', file=sys.stderr)
    t0 = time.time()
    probs = bytearray(4 * n_ctx * n_vocab)  # float32
    vocab_set = set(vocab)

    log_total = [0.0] * n_ctx  # for normalization sanity
    for i, ctx in enumerate(contexts):
        row_offset = i * n_vocab * 4
        row = []
        for w in vocab:
            log_p = score_kn(ngrams, backoffs, max_order, ctx, w)
            # log10 → linear
            p = 10.0 ** log_p
            row.append(p)
        # Renormalize (KN may sum to slightly != 1 due to OOV mass)
        s = sum(row)
        if s > 0:
            row = [p / s for p in row]
        struct.pack_into(f'<{n_vocab}f', probs, row_offset, *row)
        if (i + 1) % 10000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_ctx - i - 1) / rate
            print(f'  {i+1}/{n_ctx} ({rate:.0f} ctx/s, ETA {eta:.0f}s)', file=sys.stderr)

    print(f'Scoring done ({time.time() - t0:.1f}s)', file=sys.stderr)

    # Write output
    print(f'Writing {args.out}...', file=sys.stderr)
    with open(args.out, 'wb') as f:
        f.write(struct.pack('<I', 0x4B4E4453))   # magic "KNDS"
        f.write(struct.pack('<I', n_ctx))
        f.write(struct.pack('<I', n_vocab))
        # chars as bytes
        f.write(bytes(c.encode('ascii')[0] if len(c.encode('ascii')) == 1 else 0 for c in vocab))
        f.write(bytes(probs))
    size_mb = (8 + n_vocab + 4 * n_ctx * n_vocab) / (1024 * 1024)
    print(f'  {size_mb:.1f} MB', file=sys.stderr)


if __name__ == '__main__':
    main()
