#!/usr/bin/env python3
"""Independent PyTorch reference perplexity for AGPT models.

Loads agpt .model files (MGPT magic + cfg + matrices) directly into a
plain PyTorch transformer (nn.Embedding + nn.LayerNorm + nn.Linear +
RoPE + standard MHA + softmax + cross-entropy). Zero shared code with
the agpt trainer's CUDA kernels — used as an independent judge for
model quality (cross-check against bin/agpt_sliding_window_perplexity).

Architecture must match what agpt trains:
  - char-level tokenization, vocab from sorted unique chars
  - token embedding (V × D)
  - per-layer: LN → QKV linear → RoPE on Q/K → causal MHA → Wo + residual
               → LN → Linear(D→F) → ReLU → Linear(F→D) + residual
  - final LN → Linear(D → V) (no tying)
  - RoPE base 10000, applied to head-dim pairs (2i, 2i+1)

Output line matches bin/perplexity / bin/agpt_sliding_window_perplexity:
    Perplexity:    X.XXXX

Usage:
    python tools/agpt_ppl.py --model PATH --file PATH \\
        [--vocab-file PATH] [--d N] [--max-positions N] [--device cpu|cuda]
"""

import argparse
import math
import struct
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


MGPT_MAGIC = 0x4D475054  # matches src/cuda/agpt_train.cu's #define MGPT_MAGIC


def load_model(path):
    """Parse a .model file. Returns (cfg dict, dict of name → torch.Tensor).

    Format (see save_model_weights in src/cuda/agpt_train.cu):
        u32 magic
        i32 d_model, n_heads, n_layers, d_ff, vocab_size, seq_len
        for each matrix in trainer order:
            i32 rows, i32 cols, float32[rows*cols] (row-major)
    Optional optimizer-state footer is ignored.
    """
    data = Path(path).read_bytes()
    off = 0

    def take(n):
        nonlocal off
        b = data[off:off + n]
        off += n
        return b

    magic = struct.unpack('<I', take(4))[0]
    if magic != MGPT_MAGIC:
        raise ValueError(
            f"bad magic 0x{magic:08X} in {path}; expected 0x{MGPT_MAGIC:08X}"
        )
    d_model, n_heads, n_layers, d_ff, vocab_size, seq_len = struct.unpack(
        '<6i', take(24)
    )
    cfg = dict(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        vocab_size=vocab_size,
        seq_len=seq_len,
        head_dim=d_model // n_heads,
    )

    def read_mat(expect_rows, expect_cols):
        rows, cols = struct.unpack('<2i', take(8))
        if rows != expect_rows or cols != expect_cols:
            raise ValueError(
                f"expected matrix {expect_rows}x{expect_cols}, got {rows}x{cols}"
            )
        n_floats = rows * cols
        flat = struct.unpack(f'<{n_floats}f', take(n_floats * 4))
        return torch.tensor(flat, dtype=torch.float32).view(rows, cols)

    D, F_dim, V, L = d_model, d_ff, vocab_size, n_layers
    sd = {}
    sd['token_emb'] = read_mat(V, D)
    for l in range(L):
        sd[f'l{l}.wq_w'] = read_mat(D, D)
        sd[f'l{l}.wq_b'] = read_mat(1, D)
        sd[f'l{l}.wk_w'] = read_mat(D, D)
        sd[f'l{l}.wk_b'] = read_mat(1, D)
        sd[f'l{l}.wv_w'] = read_mat(D, D)
        sd[f'l{l}.wv_b'] = read_mat(1, D)
        sd[f'l{l}.wo_w'] = read_mat(D, D)
        sd[f'l{l}.wo_b'] = read_mat(1, D)
        sd[f'l{l}.ln1_g'] = read_mat(1, D)
        sd[f'l{l}.ln1_b'] = read_mat(1, D)
        sd[f'l{l}.l1_w'] = read_mat(D, F_dim)
        sd[f'l{l}.l1_b'] = read_mat(1, F_dim)
        sd[f'l{l}.l2_w'] = read_mat(F_dim, D)
        sd[f'l{l}.l2_b'] = read_mat(1, D)
        sd[f'l{l}.ln2_g'] = read_mat(1, D)
        sd[f'l{l}.ln2_b'] = read_mat(1, D)
    sd['final_g'] = read_mat(1, D)
    sd['final_b'] = read_mat(1, D)
    sd['out_w'] = read_mat(D, V)
    sd['out_b'] = read_mat(1, V)
    return cfg, sd


def build_rope_cache(seq_len, head_dim, base=10000.0, device='cpu'):
    """Match trainer's build_rope_cache (src/cuda/agpt_train.cu:3285).
    cos/sin shape (seq_len, head_dim). For each pair index i, both
    positions 2i and 2i+1 store the same cos/sin value (cos θ_i, sin θ_i)
    where θ_i = pos / base^(2i/head_dim).
    """
    half = head_dim // 2
    pos = torch.arange(seq_len, dtype=torch.float32, device=device).view(-1, 1)
    i = torch.arange(half, dtype=torch.float32, device=device).view(1, -1)
    theta = pos / (base ** (2.0 * i / head_dim))  # (seq_len, half)
    c = torch.cos(theta)
    s = torch.sin(theta)
    cos = torch.zeros(seq_len, head_dim, device=device)
    sin = torch.zeros(seq_len, head_dim, device=device)
    cos[:, 0::2] = c
    cos[:, 1::2] = c
    sin[:, 0::2] = s
    sin[:, 1::2] = s
    return cos, sin


def rope_apply(x, cos, sin):
    """Rotate pairs (x[..., 2i], x[..., 2i+1]) by θ_i.
    x: (..., seq, head_dim).  cos, sin: broadcastable to x.
    """
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    c = cos[..., 0::2]
    s = sin[..., 0::2]
    out_even = x_even * c - x_odd * s
    out_odd = x_even * s + x_odd * c
    out = torch.empty_like(x)
    out[..., 0::2] = out_even
    out[..., 1::2] = out_odd
    return out


class AGPTLayer(nn.Module):
    """One transformer block matching agpt's per-layer forward path."""

    def __init__(self, cfg, sd, layer):
        super().__init__()
        D = cfg['d_model']
        F_dim = cfg['d_ff']
        # Weight matrices in .model files are stored row-major as
        # [in_dim, out_dim]; nn.Linear stores [out_features, in_features].
        # So we transpose on load.
        self.wq = nn.Linear(D, D, bias=True)
        self.wk = nn.Linear(D, D, bias=True)
        self.wv = nn.Linear(D, D, bias=True)
        self.wo = nn.Linear(D, D, bias=True)
        self.l1 = nn.Linear(D, F_dim, bias=True)
        self.l2 = nn.Linear(F_dim, D, bias=True)
        self.ln1 = nn.LayerNorm(D, eps=1e-5)
        self.ln2 = nn.LayerNorm(D, eps=1e-5)

        self.wq.weight.data = sd[f'l{layer}.wq_w'].T.contiguous()
        self.wq.bias.data = sd[f'l{layer}.wq_b'].view(-1)
        self.wk.weight.data = sd[f'l{layer}.wk_w'].T.contiguous()
        self.wk.bias.data = sd[f'l{layer}.wk_b'].view(-1)
        self.wv.weight.data = sd[f'l{layer}.wv_w'].T.contiguous()
        self.wv.bias.data = sd[f'l{layer}.wv_b'].view(-1)
        self.wo.weight.data = sd[f'l{layer}.wo_w'].T.contiguous()
        self.wo.bias.data = sd[f'l{layer}.wo_b'].view(-1)
        self.l1.weight.data = sd[f'l{layer}.l1_w'].T.contiguous()
        self.l1.bias.data = sd[f'l{layer}.l1_b'].view(-1)
        self.l2.weight.data = sd[f'l{layer}.l2_w'].T.contiguous()
        self.l2.bias.data = sd[f'l{layer}.l2_b'].view(-1)
        self.ln1.weight.data = sd[f'l{layer}.ln1_g'].view(-1)
        self.ln1.bias.data = sd[f'l{layer}.ln1_b'].view(-1)
        self.ln2.weight.data = sd[f'l{layer}.ln2_g'].view(-1)
        self.ln2.bias.data = sd[f'l{layer}.ln2_b'].view(-1)

        self.n_heads = cfg['n_heads']
        self.head_dim = cfg['head_dim']

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        H, HD = self.n_heads, self.head_dim

        # First sub-block: LN1 → QKV → RoPE → causal MHA → Wo → residual
        residual = x
        h = self.ln1(x)
        q = self.wq(h).view(B, T, H, HD).transpose(1, 2)  # (B, H, T, HD)
        k = self.wk(h).view(B, T, H, HD).transpose(1, 2)
        v = self.wv(h).view(B, T, H, HD).transpose(1, 2)

        cos_b = cos.view(1, 1, T, HD)
        sin_b = sin.view(1, 1, T, HD)
        q = rope_apply(q, cos_b, sin_b)
        k = rope_apply(k, cos_b, sin_b)

        scale = 1.0 / math.sqrt(HD)
        scores = (q @ k.transpose(-2, -1)) * scale  # (B, H, T, T)
        causal = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(causal, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        attn = (weights @ v).transpose(1, 2).contiguous().view(B, T, D)
        x = residual + self.wo(attn)

        # Second sub-block: LN2 → L1 → ReLU → L2 → residual
        residual = x
        h = self.ln2(x)
        h = F.relu(self.l1(h))
        x = residual + self.l2(h)
        return x


class AGPTModel(nn.Module):
    def __init__(self, cfg, sd, device='cpu'):
        super().__init__()
        self.cfg = cfg
        D = cfg['d_model']
        V = cfg['vocab_size']

        self.tok_emb = nn.Embedding(V, D)
        self.tok_emb.weight.data = sd['token_emb']

        self.layers = nn.ModuleList(
            [AGPTLayer(cfg, sd, l) for l in range(cfg['n_layers'])]
        )

        self.final_ln = nn.LayerNorm(D, eps=1e-5)
        self.final_ln.weight.data = sd['final_g'].view(-1)
        self.final_ln.bias.data = sd['final_b'].view(-1)

        self.out_head = nn.Linear(D, V, bias=True)
        self.out_head.weight.data = sd['out_w'].T.contiguous()
        self.out_head.bias.data = sd['out_b'].view(-1)

        cos, sin = build_rope_cache(
            max(cfg['seq_len'], 1), cfg['head_dim'], device=device
        )
        self.register_buffer('rope_cos', cos)
        self.register_buffer('rope_sin', sin)

    def forward(self, ids):
        """ids: (B, T) int64. Returns logits (B, T, V)."""
        B, T = ids.shape
        if T > self.rope_cos.size(0):
            raise ValueError(
                f"context length T={T} exceeds RoPE cache seq_len={self.rope_cos.size(0)}"
            )
        x = self.tok_emb(ids)
        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.final_ln(x)
        return self.out_head(x)


def build_vocab(path):
    """Sorted-unique-chars vocab matching microgpt's TextDataset convention."""
    text = Path(path).read_text(encoding='utf-8', errors='replace')
    chars = sorted(set(text))
    return {c: i for i, c in enumerate(chars)}, len(chars)


def resolve_target_range(
    token_count,
    d_window,
    max_positions,
    eval_start,
    eval_end,
    eval_tail_frac,
    needs_full_future_window=False,
):
    min_start = d_window
    max_stop = token_count
    if needs_full_future_window:
        max_stop = token_count - d_window + 1
    if max_stop <= min_start:
        raise ValueError(
            f"need enough tokens for d_window={d_window}; got {token_count}"
        )

    start = min_start
    if eval_tail_frac is not None:
        if not (0.0 < eval_tail_frac <= 1.0):
            raise ValueError("--eval-tail-frac must be in (0, 1]")
        start = max(start, int(token_count * (1.0 - eval_tail_frac)))
    if eval_start is not None:
        start = max(start, eval_start)

    stop = max_stop
    if eval_end is not None:
        stop = min(stop, eval_end)
    if max_positions > 0:
        stop = min(stop, start + max_positions)
    if start >= stop:
        raise ValueError(
            f"empty eval target range [{start}, {stop}); token_count={token_count}"
        )
    return start, stop


def fixed_window_ppl(
    model,
    tokens,
    d_window,
    max_positions,
    device,
    batch_size=256,
    eval_start=None,
    eval_end=None,
    eval_tail_frac=None,
):
    """Standard fixed-window PPL: target i predicted from tokens[i-d:i],
    cross-entropy at the last-position logits against tokens[i]. Matches
    agpt_sliding_window_perplexity --pool deep_only."""
    model.eval()
    N = len(tokens)
    if N < d_window + 1:
        raise ValueError(f"need ≥ d_window+1 = {d_window + 1} tokens, got {N}")
    target_start, target_stop = resolve_target_range(
        N, d_window, max_positions, eval_start, eval_end, eval_tail_frac
    )
    n_targets = target_stop - target_start

    total_nll = 0.0
    tokens_t = torch.tensor(tokens, dtype=torch.long, device=device)
    with torch.no_grad():
        for start in range(target_start, target_stop, batch_size):
            stop = min(start + batch_size, target_stop)
            i_range = torch.arange(start, stop, device=device)
            offsets = torch.arange(d_window, device=device).view(1, -1)
            idx = (i_range.view(-1, 1) - d_window) + offsets  # (B, d_window)
            ctx = tokens_t[idx]
            tgt = tokens_t[i_range]
            logits = model(ctx)[:, -1, :]
            log_probs = F.log_softmax(logits, dim=-1)
            nll = -log_probs.gather(1, tgt.unsqueeze(1)).squeeze(1)
            total_nll += nll.sum().item()

    return math.exp(total_nll / n_targets), n_targets, target_start, target_stop


def sliding_window_ppl(
    model,
    tokens,
    d_window,
    max_positions,
    device,
    pool='uniform',
    batch_size=256,
    eval_start=None,
    eval_end=None,
    eval_tail_frac=None,
):
    """Logit-pooled sliding-window PPL (AGPT eval protocol).

    For each target i ∈ [d_window, N): the j-th window (j=0..d-1) starts at
    w = i - 1 - j and the within-window position j produces logits predicting
    tokens[i]. Pool log-probs across j with weights, then NLL = -pooled[target].

    pool='uniform' → weights = 1/d (matches Crystal --pool uniform).
    pool='depth_w' → weights ∝ (j+1), normalized (Crystal --pool depth_w).
    pool='deep_only' is equivalent to fixed_window_ppl; raise instead.
    """
    if pool == 'deep_only':
        raise ValueError("for deep_only use fixed_window mode")
    model.eval()
    N = len(tokens)
    if N < d_window + 1:
        raise ValueError(f"need ≥ d_window+1 = {d_window + 1} tokens, got {N}")
    target_start, target_stop = resolve_target_range(
        N,
        d_window,
        max_positions,
        eval_start,
        eval_end,
        eval_tail_frac,
        needs_full_future_window=True,
    )
    n_targets = target_stop - target_start

    if pool == 'uniform':
        weights = torch.full((d_window,), 1.0 / d_window, device=device)
    elif pool == 'depth_w':
        ramp = torch.arange(1, d_window + 1, dtype=torch.float32, device=device)
        weights = ramp / ramp.sum()
    else:
        raise ValueError(f"unknown pool mode: {pool}")

    tokens_t = torch.tensor(tokens, dtype=torch.long, device=device)
    total_nll = 0.0
    with torch.no_grad():
        # Two-level loop: outer over targets (batched), inner over the d
        # window-shifts. Each (target, j) pair feeds one forward pass on a
        # d_window-length context, reading logits at within-window position j.
        for start in range(target_start, target_stop, batch_size):
            stop = min(start + batch_size, target_stop)
            B = stop - start
            i_range = torch.arange(start, stop, device=device)
            tgt = tokens_t[i_range]  # (B,)
            pooled_lp = torch.zeros(B, model.cfg['vocab_size'], device=device)
            for j in range(d_window):
                w_start = i_range - 1 - j  # (B,), window starts here
                if (w_start < 0).any():
                    # Skip windows that fall off the front. The Crystal tool's
                    # `next if w < 0` does the same — but since i_range starts
                    # at d_window, w_start = i - 1 - j ≥ d_window - 1 - (d-1) = 0,
                    # so this is defensive: should never trigger at start=0.
                    continue
                offsets = torch.arange(d_window, device=device).view(1, -1)
                idx = w_start.view(-1, 1) + offsets
                ctx = tokens_t[idx]
                logits_j = model(ctx)[:, j, :]  # (B, V) — within-window position j
                lp_j = F.log_softmax(logits_j, dim=-1)
                pooled_lp += weights[j] * lp_j
            nll = -pooled_lp.gather(1, tgt.unsqueeze(1)).squeeze(1)
            total_nll += nll.sum().item()

    return math.exp(total_nll / n_targets), n_targets, target_start, target_stop


def main():
    ap = argparse.ArgumentParser(
        description="Independent PyTorch reference PPL for AGPT models"
    )
    ap.add_argument('--model', required=True, help='Path to .model file')
    ap.add_argument('--file', required=True, help='Evaluation text file')
    ap.add_argument(
        '--vocab-file',
        default=None,
        help='Vocab source (default: --file). Must match the corpus the model was trained on.',
    )
    ap.add_argument('--d', type=int, default=16, help='Context window length (default 16)')
    ap.add_argument(
        '--max-positions',
        type=int,
        default=0,
        help='Cap evaluated targets (0 = all). Matches the existing tool flag.',
    )
    ap.add_argument(
        '--eval-start',
        type=int,
        default=None,
        help='Absolute token index of first evaluated target. Default starts at d.',
    )
    ap.add_argument(
        '--eval-end',
        type=int,
        default=None,
        help='Absolute token index one past the last evaluated target.',
    )
    ap.add_argument(
        '--eval-tail-frac',
        type=float,
        default=None,
        help=(
            'Evaluate only the final fraction of --file by target index, e.g. 0.05. '
            'This is held-out only if training excluded that tail or --file is separate.'
        ),
    )
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument(
        '--mode',
        default='fixed',
        choices=['fixed', 'uniform', 'depth_w', 'both'],
        help=(
            "PPL protocol:\n"
            "  fixed   — standard PPL: predict from last position only (matches Crystal --pool deep_only)\n"
            "  uniform — logit-pooled sliding window, uniform weights (matches Crystal --pool uniform)\n"
            "  depth_w — logit-pooled sliding window, depth-weighted (matches Crystal --pool depth_w)\n"
            "  both    — print fixed and uniform PPLs"
        ),
    )
    args = ap.parse_args()

    cfg, sd = load_model(args.model)
    print(
        f"Model: d={cfg['d_model']} h={cfg['n_heads']} L={cfg['n_layers']} "
        f"ff={cfg['d_ff']} vocab={cfg['vocab_size']} seq_len={cfg['seq_len']}",
        file=sys.stderr,
    )

    vocab_path = args.vocab_file or args.file
    char_to_id, vocab_size = build_vocab(vocab_path)
    print(f"Vocab: {vocab_size} unique chars from {vocab_path}", file=sys.stderr)
    if vocab_size != cfg['vocab_size']:
        print(
            f"WARNING: vocab_size mismatch (vocab file={vocab_size}, model={cfg['vocab_size']})",
            file=sys.stderr,
        )

    text = Path(args.file).read_text(encoding='utf-8', errors='replace')
    tokens = [char_to_id[c] for c in text if c in char_to_id]
    print(f"Tokens: {len(tokens)} (after OOV filtering)", file=sys.stderr)

    model = AGPTModel(cfg, sd, device=args.device).to(args.device)

    if args.mode == 'fixed':
        ppl, n, start, stop = fixed_window_ppl(
            model, tokens, args.d, args.max_positions, args.device, args.batch_size,
            args.eval_start, args.eval_end, args.eval_tail_frac,
        )
        print(f"Eval target range: [{start}, {stop})", file=sys.stderr)
        print(f"Tokens evaluated: {n}", file=sys.stderr)
        print(f"Perplexity:    {ppl:.4f}")
    elif args.mode in ('uniform', 'depth_w'):
        ppl, n, start, stop = sliding_window_ppl(
            model, tokens, args.d, args.max_positions, args.device,
            pool=args.mode, batch_size=args.batch_size,
            eval_start=args.eval_start, eval_end=args.eval_end,
            eval_tail_frac=args.eval_tail_frac,
        )
        print(f"Eval target range: [{start}, {stop})", file=sys.stderr)
        print(f"Tokens evaluated: {n}", file=sys.stderr)
        print(f"Perplexity:    {ppl:.4f}")
    else:  # both
        ppl_f, n_f, start_f, stop_f = fixed_window_ppl(
            model, tokens, args.d, args.max_positions, args.device, args.batch_size,
            args.eval_start, args.eval_end, args.eval_tail_frac,
        )
        ppl_u, n_u, start_u, stop_u = sliding_window_ppl(
            model, tokens, args.d, args.max_positions, args.device,
            pool='uniform', batch_size=args.batch_size,
            eval_start=args.eval_start, eval_end=args.eval_end,
            eval_tail_frac=args.eval_tail_frac,
        )
        print(f"Fixed eval target range: [{start_f}, {stop_f})", file=sys.stderr)
        print(f"Uniform eval target range: [{start_u}, {stop_u})", file=sys.stderr)
        print(f"Tokens evaluated fixed={n_f} uniform={n_u}", file=sys.stderr)
        print(f"Perplexity (fixed):   {ppl_f:.4f}")
        print(f"Perplexity (uniform): {ppl_u:.4f}")


if __name__ == '__main__':
    main()
