#!/usr/bin/env python3
"""Sliding-window byte perplexity matching agpt_recur_perplexity.cr's
default protocol: for each position p in [seq_len, n_tokens), feed
context tokens[p-seq_len:p] (no state carry), softmax over the model's
last-position logits, sum -log p(tokens[p]).

Used to put .model checkpoints (microgpt-trained or AGPT-trained) on
the same scale as the cheap-f_θ Crystal-eval table.

Usage:
    python3 src/tools/sliding_window_ppl.py \\
        --checkpoint PATH/in.model --file HELDOUT --vocab-file PATH \\
        [--seq-len 8] [--batch-size 1024] [--device cuda|cpu]
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from agpt_ppl import AGPTModel, build_vocab, load_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--file', required=True, help='Held-out corpus')
    p.add_argument('--vocab-file', help='Vocab source (defaults to --file)')
    p.add_argument('--seq-len', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    if args.vocab_file is None:
        args.vocab_file = args.file

    cfg, sd = load_model(args.checkpoint)
    char_to_id, V = build_vocab(args.vocab_file)
    assert V == cfg['vocab_size'], f"vocab mismatch: file={V}, checkpoint={cfg['vocab_size']}"

    # We can train at one seq_len and eval at another (RoPE cache limits the max).
    if args.seq_len > cfg['seq_len']:
        raise ValueError(f"--seq-len {args.seq_len} exceeds checkpoint's RoPE cache {cfg['seq_len']}")

    text = Path(args.file).read_text(encoding='utf-8', errors='replace')
    tokens = torch.tensor([char_to_id.get(c, 0) for c in text], dtype=torch.long)
    n_tokens = tokens.numel()
    if n_tokens <= args.seq_len:
        raise ValueError(f"file too short: {n_tokens} tokens for seq_len={args.seq_len}")

    device = torch.device(args.device)
    model = AGPTModel(cfg, sd, device='cpu').to(device).eval()
    tokens_dev = tokens.to(device)

    # Positions scored: p in [seq_len, n_tokens). Context = tokens[p-seq_len:p],
    # target = tokens[p].
    n_positions = n_tokens - args.seq_len
    starts = torch.arange(n_positions, dtype=torch.long, device=device)

    T = args.seq_len
    total_nll = 0.0
    n_scored = 0

    t0 = time.time()
    with torch.no_grad():
        for i in range(0, n_positions, args.batch_size):
            batch_starts = starts[i:i+args.batch_size]
            B = batch_starts.numel()
            # context: tokens[start : start+T], target: tokens[start+T]
            idx = batch_starts.view(B, 1) + torch.arange(T, device=device).view(1, T)
            inputs = tokens_dev[idx]
            targets = tokens_dev[batch_starts + T]
            logits = model(inputs)
            last_logits = logits[:, -1, :]
            log_probs = F.log_softmax(last_logits, dim=-1)
            total_nll += -log_probs.gather(1, targets.view(-1, 1)).sum().item()
            n_scored += B

    wall = time.time() - t0
    mean_nll = total_nll / n_scored
    ppl = math.exp(mean_nll)
    bpc = mean_nll / math.log(2)

    print(f"Mode:               sliding-window (seq-len {T})")
    print(f"Positions scored:   {n_scored}")
    print(f"Mean per-token NLL: {mean_nll:.6f} nats")
    print(f"Perplexity:         {ppl:.4f}")
    print(f"Bits per character: {bpc:.4f} bpc")
    print(f"Elapsed:            {wall:.2f}s ({n_scored/wall:.1f} pos/sec)")


if __name__ == '__main__':
    main()
