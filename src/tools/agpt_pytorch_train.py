#!/usr/bin/env python3
"""Standalone PyTorch trainer for AGPTForCausalLM, used as a prototype
ground for architecture changes (e.g. the asym-DFT harmonic-bias work)
before committing to a CUDA-kernel implementation.

This is NOT the canonical trainer — that's bin/agpt_train_v2 — and it
does NOT replicate AGPT's trie/subtree-fire schedule. It does standard
batched sliding-window char-LM training. The point is to A/B
architectural changes on a small Shakespeare model in minutes and decide
whether to invest in the CUDA-side rewrite.

Saves to the same .model binary format the Crystal trainer uses, so the
output is directly consumable by bin/agpt_experiment, agpt_hf.py convert,
and agpt_lm_eval.py — no special-case handling needed downstream.

CLI matches the orchestrator's flag set as much as possible; unsupported
flags are accepted with a warning so configs don't need a separate
`tool: python3 ...` schema variant.

Usage (matches a subset of bin/agpt_train_v2):
    python3 src/tools/agpt_pytorch_train.py \\
        --model INIT.model --corpus CORPUS.txt --save OUT.model \\
        --epochs 10 --lr 3e-3 --optimizer rmsprop --growth-max-depth 16 \\
        [--harmonic-bias]   # enables the asym-DFT bias prototype
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
from agpt_ppl import load_model, build_vocab  # noqa: E402
from agpt_hf import AGPTConfig, AGPTForCausalLM  # noqa: E402


MGPT_MAGIC = 0x4D475054


# ---------------------------------------------------------------------------
# .model save (round-trip of agpt_ppl.load_model)
# ---------------------------------------------------------------------------

def save_model(path: str, model: AGPTForCausalLM) -> None:
    cfg = model.config
    backbone = model.backbone

    def emit_mat(out, rows: int, cols: int, tensor: torch.Tensor) -> None:
        flat = tensor.detach().cpu().contiguous().view(-1).float().numpy()
        assert flat.size == rows * cols, f"shape mismatch: expect {rows}*{cols}, got {flat.size}"
        out.write(struct.pack('<2i', rows, cols))
        out.write(flat.tobytes())

    with open(path, 'wb') as out:
        out.write(struct.pack('<I', MGPT_MAGIC))
        out.write(struct.pack('<6i',
            cfg.d_model, cfg.n_heads, cfg.n_layers,
            cfg.d_ff, cfg.vocab_size, cfg.seq_len))

        D, F_dim, V, L = cfg.d_model, cfg.d_ff, cfg.vocab_size, cfg.n_layers

        emit_mat(out, V, D, backbone.tok_emb.weight.data)
        for layer_idx, layer in enumerate(backbone.layers):
            # Trainer .model stores W as [in_dim, out_dim]; nn.Linear stores
            # as [out_features, in_features]. So transpose on save.
            emit_mat(out, D, D, layer.wq.weight.data.T)
            emit_mat(out, 1, D, layer.wq.bias.data.unsqueeze(0))
            emit_mat(out, D, D, layer.wk.weight.data.T)
            emit_mat(out, 1, D, layer.wk.bias.data.unsqueeze(0))
            emit_mat(out, D, D, layer.wv.weight.data.T)
            emit_mat(out, 1, D, layer.wv.bias.data.unsqueeze(0))
            emit_mat(out, D, D, layer.wo.weight.data.T)
            emit_mat(out, 1, D, layer.wo.bias.data.unsqueeze(0))
            emit_mat(out, 1, D, layer.ln1.weight.data.unsqueeze(0))
            emit_mat(out, 1, D, layer.ln1.bias.data.unsqueeze(0))
            emit_mat(out, D, F_dim, layer.l1.weight.data.T)
            emit_mat(out, 1, F_dim, layer.l1.bias.data.unsqueeze(0))
            emit_mat(out, F_dim, D, layer.l2.weight.data.T)
            emit_mat(out, 1, D, layer.l2.bias.data.unsqueeze(0))
            emit_mat(out, 1, D, layer.ln2.weight.data.unsqueeze(0))
            emit_mat(out, 1, D, layer.ln2.bias.data.unsqueeze(0))
        emit_mat(out, 1, D, backbone.final_ln.weight.data.unsqueeze(0))
        emit_mat(out, 1, D, backbone.final_ln.bias.data.unsqueeze(0))
        emit_mat(out, D, V, backbone.out_head.weight.data.T)
        emit_mat(out, 1, V, backbone.out_head.bias.data.unsqueeze(0))


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------

def tokenize_corpus(corpus_path: str, vocab_path: str | None = None
                    ) -> tuple[torch.Tensor, dict[str, int]]:
    char_to_id, _ = build_vocab(vocab_path or corpus_path)
    text = Path(corpus_path).read_text(encoding='utf-8', errors='replace')
    ids = torch.tensor(
        [char_to_id.get(c, 0) for c in text], dtype=torch.long
    )
    return ids, char_to_id


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------

def cosine_warmup_lr(step: int, total: int, base_lr: float,
                     warmup_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    print(f"[{time.strftime('%H:%M:%S')}] loading init model {args.model}", file=sys.stderr)
    model = AGPTForCausalLM.from_agpt_checkpoint(args.model).to(device)
    cfg = model.config
    d_window = args.growth_max_depth or cfg.seq_len

    print(f"[{time.strftime('%H:%M:%S')}] tokenizing {args.corpus}", file=sys.stderr)
    ids, _ = tokenize_corpus(args.corpus, vocab_path=args.vocab_file)
    N = ids.numel()
    print(f"  corpus tokens: {N}", file=sys.stderr)
    if N < d_window + 2:
        print(f"corpus too short for d={d_window}", file=sys.stderr)
        sys.exit(2)

    if args.optimizer == 'rmsprop':
        opt = torch.optim.RMSprop(model.parameters(), lr=args.lr,
                                  alpha=args.rmsprop_beta)
    elif args.optimizer == 'adam':
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=args.lr)

    batch_size = args.batch_size
    # One "epoch" = full pass through positions [d_window, N). Drop tail.
    positions = torch.arange(d_window, N, dtype=torch.long, device=device)
    positions_per_epoch = positions.numel()
    batches_per_epoch = positions_per_epoch // batch_size
    total_batches = batches_per_epoch * args.epochs
    warmup_steps = max(1, int(args.warmup_epochs * batches_per_epoch))
    print(f"  d_window={d_window} batch_size={batch_size} "
          f"batches_per_epoch={batches_per_epoch} total={total_batches}",
          file=sys.stderr)

    model.train()
    ids_dev = ids.to(device)
    step = 0
    epoch_start = time.time()
    for epoch in range(args.epochs):
        perm = positions[torch.randperm(positions.numel(), device=device)]
        epoch_loss = 0.0
        epoch_count = 0
        for b in range(batches_per_epoch):
            batch_pos = perm[b * batch_size:(b + 1) * batch_size]
            # Build context [d_window] for each target position; target is ids[pos].
            offs = torch.arange(-d_window, 0, device=device)
            ctx_idx = batch_pos.unsqueeze(1) + offs.unsqueeze(0)
            ctx = ids_dev[ctx_idx]
            target = ids_dev[batch_pos]

            opt.zero_grad(set_to_none=True)
            out = model(input_ids=ctx)
            logits_at_last = out.logits[:, -1, :]
            loss = F.cross_entropy(logits_at_last, target)
            loss.backward()
            lr_now = cosine_warmup_lr(step, total_batches, args.lr, warmup_steps)
            for g in opt.param_groups:
                g['lr'] = lr_now
            opt.step()

            step += 1
            epoch_loss += loss.item() * batch_pos.numel()
            epoch_count += batch_pos.numel()

            if not args.quiet and step % args.log_every == 0:
                cur_lr = opt.param_groups[0]['lr']
                print(f"  step {step}/{total_batches} loss={loss.item():.4f} lr={cur_lr:.2e}",
                      file=sys.stderr)

        mean_loss = epoch_loss / max(1, epoch_count)
        elapsed = time.time() - epoch_start
        print(f"[{time.strftime('%H:%M:%S')}] epoch {epoch+1}/{args.epochs} "
              f"mean_loss={mean_loss:.4f} ppl_train={math.exp(mean_loss):.3f} "
              f"elapsed={elapsed:.1f}s",
              file=sys.stderr)

    print(f"[{time.strftime('%H:%M:%S')}] saving {args.save}", file=sys.stderr)
    save_model(args.save, model)
    print(f"  wrote {args.save}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--model', required=True, help='Init .model checkpoint')
    p.add_argument('--corpus', required=True, help='Training corpus text')
    p.add_argument('--vocab-file', default=None,
                   help='Explicit vocab corpus (defaults to --corpus)')
    p.add_argument('--save', required=True, help='Output .model path')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=3e-3)
    p.add_argument('--rmsprop-beta', type=float, default=0.999)
    p.add_argument('--optimizer', default='rmsprop',
                   choices=['rmsprop', 'adam', 'sgd'])
    p.add_argument('--lr-schedule', default='warmup-cosine')
    p.add_argument('--warmup-epochs', type=float, default=0.0)
    p.add_argument('--growth-max-depth', type=int, default=None,
                   help='Sliding-window context size (defaults to model.seq_len)')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--quiet', action='store_true')
    p.add_argument('--log-every', type=int, default=200)

    # Orchestrator flags this trainer doesn't use yet — accept silently.
    for ignored in [
        '--mode', '--growth-frontiers', '--growth-divisions',
        '--growth-min-epochs', '--growth-epoch-ramp',
        '--partition-depth', '--chunk-queries', '--anc-grad',
        '--ablate-anc-grad', '--accumulate', '--no-accumulate',
        '--momentum-beta',
    ]:
        p.add_argument(ignored, default=None,
                       help=argparse.SUPPRESS)

    # Prototype-only flag: enable the asym-DFT harmonic bias (when wired).
    p.add_argument('--harmonic-bias', action='store_true',
                   help='Enable prototype asym-DFT bias in attention (NYI in this build)')

    args = p.parse_args()
    train(args)


if __name__ == '__main__':
    main()
