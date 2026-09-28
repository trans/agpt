#!/usr/bin/env python3
"""Vanilla mini-batch SGD trainer for the AGPT attention architecture.

Same model as the canonical AGPT trainer (RoPE + MHA + LN + FFN), but
trained without the trie's gradient aggregation. Lets us isolate
whether the trie provides structural value beyond what plain SGD on
the same architecture achieves.

Saves checkpoints in the MGPT .model format so they can be evaluated
with the canonical lm-eval-harness pipeline (agpt_lm_eval.py).

Usage:
    python3 src/tools/agpt_vanilla_attn_train.py \\
        --corpus PATH --vocab-source PATH \\
        --d-model 64 --n-layers 2 --n-heads 4 --d-ff 256 --seq-len 8 \\
        --batch-size 128 --epochs 50 --lr 1.5e-3 \\
        --save PATH/out.model

    # Or initialize from seed model:
    python3 src/tools/agpt_vanilla_attn_train.py \\
        --load-seed data/seeds/shake-d64L2-h4-dff256-s128-seed42.model \\
        --corpus PATH --vocab-source PATH \\
        --seq-len 16 --batch-size 128 --epochs 512 --lr 1.5e-3 \\
        --save PATH/out.model
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from agpt_ppl import AGPTModel, MGPT_MAGIC, build_rope_cache, build_vocab, load_model


def random_state_dict(d_model, n_heads, n_layers, d_ff, vocab_size, seq_len, seed=42):
    """Create a randomly-initialized state dict matching the load_model format.

    Uses standard transformer init: ~N(0, 0.02) for weights, zero biases.
    LayerNorm gains = 1, biases = 0.
    """
    g = torch.Generator().manual_seed(seed)
    D, F_dim, V, L = d_model, d_ff, vocab_size, n_layers

    def randn(*shape):
        return torch.randn(*shape, generator=g) * 0.02

    def zeros(*shape):
        return torch.zeros(*shape)

    sd = {}
    sd['token_emb'] = randn(V, D)
    for l in range(L):
        sd[f'l{l}.wq_w'] = randn(D, D)
        sd[f'l{l}.wq_b'] = zeros(1, D)
        sd[f'l{l}.wk_w'] = randn(D, D)
        sd[f'l{l}.wk_b'] = zeros(1, D)
        sd[f'l{l}.wv_w'] = randn(D, D)
        sd[f'l{l}.wv_b'] = zeros(1, D)
        sd[f'l{l}.wo_w'] = randn(D, D)
        sd[f'l{l}.wo_b'] = zeros(1, D)
        sd[f'l{l}.ln1_g'] = torch.ones(1, D)
        sd[f'l{l}.ln1_b'] = zeros(1, D)
        sd[f'l{l}.l1_w'] = randn(D, F_dim)
        sd[f'l{l}.l1_b'] = zeros(1, F_dim)
        sd[f'l{l}.l2_w'] = randn(F_dim, D)
        sd[f'l{l}.l2_b'] = zeros(1, D)
        sd[f'l{l}.ln2_g'] = torch.ones(1, D)
        sd[f'l{l}.ln2_b'] = zeros(1, D)
    sd['final_g'] = torch.ones(1, D)
    sd['final_b'] = zeros(1, D)
    sd['out_w'] = randn(D, V)
    sd['out_b'] = zeros(1, V)
    return sd


def extract_state_dict(model, cfg):
    """Read tensors out of the trained AGPTModel back into the load_model
    format so we can write them to a .model file."""
    sd = {}
    sd['token_emb'] = model.tok_emb.weight.data.cpu().clone()
    for l, layer in enumerate(model.layers):
        # .model format stores weights as (in_dim, out_dim) row-major; nn.Linear
        # stores them as (out_features, in_features). Transpose ALL Linear
        # weights on save — even square ones (D x D Q/K/V/O), since shape
        # equality hides the semantic transpose.
        sd[f'l{l}.wq_w'] = layer.wq.weight.data.cpu().clone().T.contiguous()
        sd[f'l{l}.wq_b'] = layer.wq.bias.data.cpu().clone().view(1, -1)
        sd[f'l{l}.wk_w'] = layer.wk.weight.data.cpu().clone().T.contiguous()
        sd[f'l{l}.wk_b'] = layer.wk.bias.data.cpu().clone().view(1, -1)
        sd[f'l{l}.wv_w'] = layer.wv.weight.data.cpu().clone().T.contiguous()
        sd[f'l{l}.wv_b'] = layer.wv.bias.data.cpu().clone().view(1, -1)
        sd[f'l{l}.wo_w'] = layer.wo.weight.data.cpu().clone().T.contiguous()
        sd[f'l{l}.wo_b'] = layer.wo.bias.data.cpu().clone().view(1, -1)
        sd[f'l{l}.ln1_g'] = layer.ln1.weight.data.cpu().clone().view(1, -1)
        sd[f'l{l}.ln1_b'] = layer.ln1.bias.data.cpu().clone().view(1, -1)
        sd[f'l{l}.l1_w'] = layer.l1.weight.data.cpu().clone().T.contiguous()
        sd[f'l{l}.l1_b'] = layer.l1.bias.data.cpu().clone().view(1, -1)
        sd[f'l{l}.l2_w'] = layer.l2.weight.data.cpu().clone().T.contiguous()
        sd[f'l{l}.l2_b'] = layer.l2.bias.data.cpu().clone().view(1, -1)
        sd[f'l{l}.ln2_g'] = layer.ln2.weight.data.cpu().clone().view(1, -1)
        sd[f'l{l}.ln2_b'] = layer.ln2.bias.data.cpu().clone().view(1, -1)
    sd['final_g'] = model.final_ln.weight.data.cpu().clone().view(1, -1)
    sd['final_b'] = model.final_ln.bias.data.cpu().clone().view(1, -1)
    sd['out_w'] = model.out_head.weight.data.cpu().clone().T.contiguous()
    sd['out_b'] = model.out_head.bias.data.cpu().clone().view(1, -1)
    return sd


def save_model(path, cfg, sd):
    """Write .model file matching the MGPT format that load_model reads."""
    D = cfg['d_model']
    F_dim = cfg['d_ff']
    V = cfg['vocab_size']
    L = cfg['n_layers']
    H = cfg['n_heads']
    S = cfg['seq_len']

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        f.write(struct.pack('<I', MGPT_MAGIC))
        f.write(struct.pack('<6i', D, H, L, F_dim, V, S))

        def write_mat(tensor, rows, cols):
            assert tensor.shape == (rows, cols), f"shape mismatch {tensor.shape} vs ({rows},{cols})"
            f.write(struct.pack('<2i', rows, cols))
            flat = tensor.contiguous().view(-1).tolist()
            f.write(struct.pack(f'<{len(flat)}f', *flat))

        write_mat(sd['token_emb'], V, D)
        for l in range(L):
            write_mat(sd[f'l{l}.wq_w'], D, D)
            write_mat(sd[f'l{l}.wq_b'], 1, D)
            write_mat(sd[f'l{l}.wk_w'], D, D)
            write_mat(sd[f'l{l}.wk_b'], 1, D)
            write_mat(sd[f'l{l}.wv_w'], D, D)
            write_mat(sd[f'l{l}.wv_b'], 1, D)
            write_mat(sd[f'l{l}.wo_w'], D, D)
            write_mat(sd[f'l{l}.wo_b'], 1, D)
            write_mat(sd[f'l{l}.ln1_g'], 1, D)
            write_mat(sd[f'l{l}.ln1_b'], 1, D)
            write_mat(sd[f'l{l}.l1_w'], D, F_dim)
            write_mat(sd[f'l{l}.l1_b'], 1, F_dim)
            write_mat(sd[f'l{l}.l2_w'], F_dim, D)
            write_mat(sd[f'l{l}.l2_b'], 1, D)
            write_mat(sd[f'l{l}.ln2_g'], 1, D)
            write_mat(sd[f'l{l}.ln2_b'], 1, D)
        write_mat(sd['final_g'], 1, D)
        write_mat(sd['final_b'], 1, D)
        write_mat(sd['out_w'], D, V)
        write_mat(sd['out_b'], 1, V)


def tokenize_corpus(corpus_path, vocab_source):
    char_to_id, V = build_vocab(vocab_source)
    text = Path(corpus_path).read_text(encoding='utf-8', errors='replace')
    ids = torch.tensor([char_to_id.get(c, 0) for c in text], dtype=torch.long)
    return ids, V


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--corpus', required=True)
    p.add_argument('--vocab-source')
    p.add_argument('--load-seed', help='Init from .model file (overrides arch flags if given)')
    p.add_argument('--d-model', type=int, default=64)
    p.add_argument('--n-layers', type=int, default=2)
    p.add_argument('--n-heads', type=int, default=4)
    p.add_argument('--d-ff', type=int, default=256)
    p.add_argument('--seq-len', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=1.5e-3)
    p.add_argument('--beta1', type=float, default=0.9)
    p.add_argument('--beta2', type=float, default=0.999)
    p.add_argument('--eps', type=float, default=1e-8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--save', required=True)
    p.add_argument('--save-every', type=int, default=0,
                   help='Write epoch checkpoints every N epochs (in addition to final).')
    args = p.parse_args()

    if args.vocab_source is None:
        args.vocab_source = args.corpus

    tokens, vocab_size = tokenize_corpus(args.corpus, args.vocab_source)
    n_tokens = tokens.numel()
    n_starts = n_tokens - args.seq_len - 1
    assert n_starts > 0, f"corpus too short: {n_tokens} tokens for seq_len={args.seq_len}"

    if args.load_seed:
        cfg, sd = load_model(args.load_seed)
        assert cfg['vocab_size'] == vocab_size, \
            f"vocab mismatch: seed={cfg['vocab_size']}, corpus={vocab_size}"
        seed_seq_len = cfg['seq_len']
        # Override to training seq_len so the saved checkpoint declares the
        # actual evaluation context. Saving the seed's wider seq_len causes
        # lm-eval to feed positions the post-fine-tune model never trained on.
        cfg['seq_len'] = max(args.seq_len, 16)
        print(f"Loaded seed: d={cfg['d_model']} L={cfg['n_layers']} h={cfg['n_heads']} "
              f"dff={cfg['d_ff']} V={cfg['vocab_size']} seed_seq_len={seed_seq_len} "
              f"-> training_seq_len={cfg['seq_len']}")
    else:
        cfg = dict(
            d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
            d_ff=args.d_ff, vocab_size=vocab_size, seq_len=max(args.seq_len, 16),
            head_dim=args.d_model // args.n_heads,
        )
        sd = random_state_dict(args.d_model, args.n_heads, args.n_layers,
                               args.d_ff, vocab_size, cfg['seq_len'], seed=args.seed)
        print(f"Random init: d={cfg['d_model']} L={cfg['n_layers']} h={cfg['n_heads']} "
              f"dff={cfg['d_ff']} V={cfg['vocab_size']} seq_len={cfg['seq_len']}")

    device = torch.device(args.device)
    model = AGPTModel(cfg, sd, device='cpu').to(device)

    print(f"  corpus: {args.corpus} ({n_tokens} tokens, vocab_size={vocab_size})")
    print(f"  starts_per_epoch: {n_starts}")
    print(f"  train_seq_len: {args.seq_len}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  updates_per_epoch: {(n_starts + args.batch_size - 1) // args.batch_size}")
    print(f"  device: {device}")
    print(f"  params: {sum(p.numel() for p in model.parameters())}")

    optim = torch.optim.Adam(
        model.parameters(),
        lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps,
    )

    tokens_dev = tokens.to(device)
    rng = torch.Generator(device='cpu').manual_seed(args.seed)

    T = args.seq_len
    starts_buffer = torch.arange(n_starts, dtype=torch.long)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        perm = torch.randperm(n_starts, generator=rng)
        loss_total = 0.0
        n_events = 0
        n_updates = 0

        model.train()
        for i in range(0, n_starts, args.batch_size):
            batch_starts = perm[i:i+args.batch_size].to(device)
            B = batch_starts.numel()

            # Inputs: tokens[start : start+T], targets: tokens[start+1 : start+T+1]
            idx = batch_starts.view(B, 1) + torch.arange(T + 1, device=device).view(1, T + 1)
            chunks = tokens_dev[idx]
            inputs = chunks[:, :T]
            targets = chunks[:, 1:T+1]

            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(B * T, vocab_size),
                                   targets.reshape(B * T))

            optim.zero_grad()
            loss.backward()
            optim.step()

            loss_total += loss.item() * B * T
            n_events += B * T
            n_updates += 1

        wall = time.time() - t0
        mean_nll = loss_total / n_events
        ppl = math.exp(mean_nll)
        ckpt_msg = ""
        if args.save_every > 0 and epoch % args.save_every == 0:
            ext = Path(args.save).suffix
            base = str(Path(args.save).with_suffix(''))
            ckpt_path = f"{base}.epoch_{epoch:06d}{ext}"
            sd_out = extract_state_dict(model, cfg)
            save_model(ckpt_path, cfg, sd_out)
            ckpt_msg = f" checkpoint={ckpt_path}"
        print(f"epoch {epoch:6d}  nll {mean_nll:.6f}  ppl {ppl:.6f}  "
              f"events {n_events}  updates {n_updates}  wall {wall:.2f}s{ckpt_msg}",
              flush=True)

    sd_out = extract_state_dict(model, cfg)
    save_model(args.save, cfg, sd_out)
    print(f"saved {args.save}")


if __name__ == '__main__':
    main()
