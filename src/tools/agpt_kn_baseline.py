#!/usr/bin/env python3
"""Kneser-Ney smoothed n-gram PPL baseline.

Builds char-level KN models at a configurable order from a training
corpus, computes PPL on a held-out file. Gives us a strong classical
baseline to compare AGPT's trained-model PPL against.

KN was the SOTA for word-level n-gram LMs for ~15 years. Char-level
KN on Shakespeare-class corpora typically achieves PPL 2-3 at order
5-6; higher orders give diminishing returns.

Usage:
    python src/tools/agpt_kn_baseline.py \\
        --train data/input.txt --heldout /tmp/shake_holdout.txt \\
        --orders 3,4,6,8,12 \\
        [--max-positions N]

Output (matches existing PPL tool convention):
    Perplexity:    X.XXXX   (per order)
"""

import argparse
import math
import sys
import time
from pathlib import Path

from nltk.lm import KneserNeyInterpolated
from nltk.lm.preprocessing import padded_everygram_pipeline


def build_and_score(train_text, heldout_text, order, max_positions=0):
    """Train KN model at given order, score heldout PPL."""
    # Treat each character as a "word"; nltk expects iterables of tokens
    train_tokens = [list(train_text)]
    train_data, padded_vocab = padded_everygram_pipeline(order, train_tokens)
    model = KneserNeyInterpolated(order)

    t0 = time.time()
    model.fit(train_data, padded_vocab)
    fit_t = time.time() - t0

    # Score held-out. For each position i (with at least order-1 prior chars),
    # compute log P(char_i | char_{i-order+1..i-1}). Use natural log conventions
    # consistent with AGPT eval (NLL = -log p).
    n = len(heldout_text)
    if max_positions > 0 and n > max_positions + order:
        # Take the first max_positions targets
        n = max_positions + order

    t0 = time.time()
    total_nll = 0.0
    count = 0
    for i in range(order - 1, n):
        context = tuple(heldout_text[i - (order - 1):i])
        target = heldout_text[i]
        # nltk's logscore uses base 2 by default; convert to nats
        log2_p = model.logscore(target, context)
        nll = -log2_p * math.log(2)
        total_nll += nll
        count += 1
        if max_positions > 0 and count >= max_positions:
            break
    score_t = time.time() - t0

    ppl = math.exp(total_nll / count) if count > 0 else float('inf')
    return ppl, count, fit_t, score_t


def main():
    ap = argparse.ArgumentParser(description="KN n-gram PPL baseline for AGPT comparison")
    ap.add_argument('--train', required=True, help='Training corpus text file')
    ap.add_argument('--heldout', required=True, help='Held-out text file')
    ap.add_argument('--orders', default='3,4,6,8',
                    help='Comma-separated KN orders to try (default 3,4,6,8)')
    ap.add_argument('--max-positions', type=int, default=10000,
                    help='Cap heldout positions evaluated (default 10000, 0=all)')
    args = ap.parse_args()

    train_text = Path(args.train).read_text(encoding='utf-8', errors='replace')
    heldout_text = Path(args.heldout).read_text(encoding='utf-8', errors='replace')

    # Filter heldout to chars that appear in train (OOV would crash KN)
    train_chars = set(train_text)
    heldout_text = ''.join(c for c in heldout_text if c in train_chars)
    print(f"Train: {len(train_text)} chars, {len(train_chars)} vocab", file=sys.stderr)
    print(f"Heldout: {len(heldout_text)} chars (after OOV filter)", file=sys.stderr)

    orders = [int(x) for x in args.orders.split(',')]
    print(f"Orders to fit: {orders}", file=sys.stderr)
    print(f"", file=sys.stderr)

    for order in orders:
        print(f"--- order {order} ---", file=sys.stderr)
        ppl, count, fit_t, score_t = build_and_score(
            train_text, heldout_text, order, args.max_positions
        )
        print(f"  Fit: {fit_t:.1f}s  Score: {score_t:.1f}s  evaluated={count}", file=sys.stderr)
        print(f"Order {order}: Perplexity:    {ppl:.4f}")


if __name__ == '__main__':
    main()
