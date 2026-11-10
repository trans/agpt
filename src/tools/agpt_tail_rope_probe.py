#!/usr/bin/env python3
"""Superposed-tail RoPE probe.

Question: does attention to history older than the 16-character window help,
when every character's older occurrences are superposed into one RoPE key?

No trie. A plain minibatch-Adam transformer with the AGPT d64/L2 layout
(pre-LN, RoPE with interleaved pairs, ReLU FFN, untied output head) sees a
16-character softmax window. In the tail arms, layer 1's heads also attend,
linearly, to every older position of the same document through a
per-character state

    S_s[c, j] = (1 - r_j) * sum_{p < s, x_p = c} lambda_j^(s - p),
    lambda_j  = r_j * exp(i * w_j),

where s is the window start. Layer-1 keys and values depend only on the
character, so the summed RoPE score from the query at t = s + i to all older
occurrences of c factors exactly:

    w[t, c] = Re sum_j  zq_j(t) * lambda_j^i * conj(zk_j(c)) * S_s[c, j] / sqrt(hd)

(zq, zk: the unrotated query and key as complex pairs). The head adds
gate_h * sum_c w[t, c] * v(c) to its softmax output; gate_h starts at 0, so
every arm starts as the baseline. The arms differ only in the 8 atoms
(w_j, r_j = exp(-1 / tau_j)):

  none     no tail (baseline)
  fourier  w_j = the head's RoPE frequencies, one shared tau (--fourier-tau)
  wavelet  w_j = RoPE frequencies, tau_j = cycles * 2 pi / w_j (constant Q)
  decay    w_j = 0, tau_j = the wavelet ladder (timescales only, no phase)

Training scores only the target after a full 16-character window (one per
window, --loss-positions last), so training sees what evaluation scores.

Evaluation is DIAGNOSTIC, not the canonical lm-eval byte PPL: each held-out
chunk is its own document (tail history starts at the chunk start), and each
target is predicted from up to 16 preceding characters plus the tail.
Training cuts the training split into documents of --doc-len characters, so
the tail sees about as much history in training as in evaluation. The
held-out NLL is also reported by position in the chunk, which shows where
any gain comes from.

Usage:
    python3 src/tools/agpt_tail_rope_probe.py --arm wavelet --steps 40000 \\
        --out rnd/superposed-tail-rope/RUN_ID
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

SPLIT = Path('data/.splits/4fa9aec1db6b3aea')
K = 64  # state checkpoint stride, in characters
BUCKETS = [(1, 16), (16, 256), (256, 1024), (1024, 10 ** 9)]


# ----------------------------------------------------------------- data

def build_vocab(path):
    text = Path(path).read_text(encoding='utf-8', errors='replace')
    return {c: i for i, c in enumerate(sorted(set(text)))}


def encode(path, vocab):
    text = Path(path).read_text(encoding='utf-8', errors='replace')
    return torch.tensor([vocab[c] for c in text], dtype=torch.long)


# ---------------------------------------------------------------- atoms

def make_atoms(arm, head_dim, base, fourier_tau, cycles):
    J = head_dim // 2
    rope_w = base ** (-2.0 * torch.arange(J, dtype=torch.float64) / head_dim)
    tau_ladder = cycles * 2.0 * math.pi / rope_w
    if arm == 'fourier':
        omega, tau = rope_w, torch.full_like(rope_w, float(fourier_tau))
    elif arm == 'wavelet':
        omega, tau = rope_w, tau_ladder
    elif arm == 'decay':
        omega, tau = torch.zeros_like(rope_w), tau_ladder
    else:
        raise ValueError(arm)
    r = torch.exp(-1.0 / tau)
    lam = torch.polar(r, omega)  # complex128
    return lam, r, omega, tau


@torch.no_grad()
def checkpoints(x, lam, r, V, doc_blocks=None):
    """C[g] = state at position g*K, offsets measured from g*K.
    The state resets to 0 at every doc start (every doc_blocks blocks)."""
    nb = x.numel() // K
    J = lam.numel()
    m = torch.arange(K, dtype=torch.float64)
    contrib_w = (1.0 - r)[None, :] * lam[None, :] ** (K - m)[:, None]  # (K, J)
    xb = x[:nb * K].view(nb, K)
    rows = (torch.arange(nb)[:, None] * V + xb).reshape(-1)
    re = torch.zeros(nb * V, J, dtype=torch.float64)
    im = torch.zeros(nb * V, J, dtype=torch.float64)
    re.index_add_(0, rows, contrib_w.real.repeat(nb, 1))
    im.index_add_(0, rows, contrib_w.imag.repeat(nb, 1))
    contrib = torch.complex(re, im).view(nb, V, J)
    C = torch.zeros(nb + 1, V, J, dtype=torch.complex128)
    lamK = lam ** K
    for g in range(nb):
        if doc_blocks is not None and (g + 1) % doc_blocks == 0:
            continue  # position (g+1)*K starts a new document: state 0
        C[g + 1] = lamK * C[g] + contrib[g]
    return C.to(torch.complex64)


class TailState:
    """Per-document-set tail states: base checkpoints + exact advance."""

    def __init__(self, x, lam, r, V, device, doc_blocks=None):
        self.V = V
        self.x = x.to(device)
        self.C = checkpoints(x, lam, r, V, doc_blocks).to(device)
        ks = torch.arange(K + 1, dtype=torch.float64)
        self.P = (lam[None, :] ** ks[:, None]).to(torch.complex64).to(device)  # lam^k
        self.one_minus_r = (1.0 - r).to(torch.float32).to(device)

    @torch.no_grad()
    def at(self, starts):
        """S_s for a batch of window starts: (B, V, J) complex64."""
        dev = starts.device
        g = starts // K
        s0 = g * K
        d = starts - s0
        m = torch.arange(K, device=dev)
        pos = (s0[:, None] + m[None, :]).clamp(max=self.x.numel() - 1)
        valid = m[None, :] < d[:, None]
        off = (d[:, None] - m[None, :]).clamp(min=0)
        w = self.P[off] * (valid[..., None] * self.one_minus_r)  # (B, K, J)
        onehot = F.one_hot(self.x[pos], self.V).to(torch.complex64)  # (B, K, V)
        S = self.C[g] * self.P[d][:, None, :]
        return S + torch.einsum('bkv,bkj->bvj', onehot, w)


# ---------------------------------------------------------------- model

def rope_cache(T, head_dim, base, device):
    half = head_dim // 2
    pos = torch.arange(T, dtype=torch.float32).view(-1, 1)
    i = torch.arange(half, dtype=torch.float32).view(1, -1)
    theta = pos / (base ** (2.0 * i / head_dim))
    cos = torch.repeat_interleave(torch.cos(theta), 2, dim=1)
    sin = torch.repeat_interleave(torch.sin(theta), 2, dim=1)
    return cos.to(device), sin.to(device)


def rope_apply(x, cos, sin):
    xe, xo = x[..., 0::2], x[..., 1::2]
    c, s = cos[..., 0::2], sin[..., 0::2]
    out = torch.empty_like(x)
    out[..., 0::2] = xe * c - xo * s
    out[..., 1::2] = xe * s + xo * c
    return out


class Block(nn.Module):
    def __init__(self, D, H, Fd, tail):
        super().__init__()
        self.H = H
        self.ln1 = nn.LayerNorm(D)
        self.wq, self.wk = nn.Linear(D, D), nn.Linear(D, D)
        self.wv, self.wo = nn.Linear(D, D), nn.Linear(D, D)
        self.ln2 = nn.LayerNorm(D)
        self.l1, self.l2 = nn.Linear(D, Fd), nn.Linear(Fd, D)
        self.tail = tail
        if tail:
            self.gate = nn.Parameter(torch.zeros(H))

    def forward(self, x, cos, sin, tail_ctx=None):
        B, T, D = x.shape
        H = self.H
        HD = D // H
        h = self.ln1(x)
        q = self.wq(h).view(B, T, H, HD).transpose(1, 2)
        k = self.wk(h).view(B, T, H, HD).transpose(1, 2)
        v = self.wv(h).view(B, T, H, HD).transpose(1, 2)
        att = F.scaled_dot_product_attention(
            rope_apply(q, cos, sin), rope_apply(k, cos, sin), v, is_causal=True)
        if self.tail and tail_ctx is not None:
            S, lam_i, emb = tail_ctx  # (B,V,J) c64, (T,J) c64, (V,D)
            V = emb.size(0)
            he = self.ln1(emb)
            kc = self.wk(he).view(V, H, HD)
            vc = self.wv(he).view(V, H, HD)
            zq = torch.complex(q[..., 0::2].contiguous(), q[..., 1::2].contiguous())
            zq = zq * lam_i[None, None]                                # (B,H,T,J)
            zk = torch.complex(kc[..., 0::2].contiguous(), kc[..., 1::2].contiguous())
            Y = zk.conj().permute(1, 0, 2)[None] * S[:, None]           # (B,H,V,J)
            w = torch.einsum('bhtj,bhvj->bhtv', zq, Y).real / math.sqrt(HD)
            att = att + self.gate.view(1, H, 1, 1) * torch.einsum('bhtv,vhd->bhtd', w, vc)
        x = x + self.wo(att.transpose(1, 2).reshape(B, T, D))
        return x + self.l2(F.relu(self.l1(self.ln2(x))))


class TailRopeLM(nn.Module):
    def __init__(self, V, D, H, L, Fd, T, base, tail, device):
        super().__init__()
        self.tok = nn.Embedding(V, D)
        self.blocks = nn.ModuleList([Block(D, H, Fd, tail and l == 0) for l in range(L)])
        self.ln_f = nn.LayerNorm(D)
        self.head = nn.Linear(D, V)
        cos, sin = rope_cache(T, D // H, base, device)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

    def init_weights(self, seed):
        """N(0, 0.02) weights, zero biases, unit LayerNorm; tail gates stay 0."""
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for m in self.modules():  # deterministic order
                if isinstance(m, (nn.Linear, nn.Embedding)):
                    m.weight.copy_(torch.randn(m.weight.shape, generator=g) * 0.02)
                    if getattr(m, 'bias', None) is not None:
                        m.bias.zero_()
                elif isinstance(m, nn.LayerNorm):
                    m.weight.fill_(1.0)
                    m.bias.zero_()

    def forward(self, ids, tail_ctx=None):
        T = ids.size(1)
        x = self.tok(ids)
        cos, sin = self.cos[:T], self.sin[:T]
        for b in self.blocks:
            x = b(x, cos, sin, tail_ctx)
        return self.head(self.ln_f(x))


# ----------------------------------------------------------------- eval

@torch.no_grad()
def evaluate(model, chunks, states, lam_T, T, batch):
    model.eval()
    nll = torch.zeros(len(BUCKETS), dtype=torch.float64)
    cnt = torch.zeros(len(BUCKETS), dtype=torch.float64)

    def add(t_pos, losses):
        for bi, (lo, hi) in enumerate(BUCKETS):
            sel = (t_pos >= lo) & (t_pos < hi)
            nll[bi] += losses[sel].double().sum().cpu()
            cnt[bi] += sel.sum().cpu()

    for ci, x in enumerate(chunks):
        n = x.numel()
        dev = x.device
        starts_all = torch.arange(0, n - T, device=dev)  # target t = s + T at pos T-1
        for i in range(0, starts_all.numel(), batch):
            s = starts_all[i:i + batch]
            ids = x[s[:, None] + torch.arange(T, device=dev)[None, :]]
            ctx = None
            if states is not None:
                ctx = (states[ci].at(s), lam_T, model.tok.weight)
            logits = model(ids, ctx)
            if i == 0:  # the s = 0 window also scores targets 1..T-1
                l0 = F.cross_entropy(logits[0, :T - 1], x[1:T], reduction='none')
                add(torch.arange(1, T, device=dev), l0)
            lt = F.cross_entropy(logits[:, T - 1], x[s + T], reduction='none')
            add(s + T, lt)
    model.train()
    total = float(nll.sum() / cnt.sum())
    by = {f'{lo}-{hi if hi < 10 ** 9 else "end"}': float(nll[b] / cnt[b])
          for b, (lo, hi) in enumerate(BUCKETS)}
    return total, by, int(cnt.sum())


# ----------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--arm', choices=['none', 'fourier', 'wavelet', 'decay'], required=True)
    ap.add_argument('--steps', type=int, default=40000)
    ap.add_argument('--batch-size', type=int, default=1024)
    ap.add_argument('--loss-positions', choices=['last', 'all'], default='last',
                    help='last: score only the full-window target, as in evaluation. '
                         'all: score every window position; then the tail learns to stand '
                         'in for the missing in-window context of early positions, which '
                         'helps training loss but not held-out (smoke test, 2026-11-10)')
    ap.add_argument('--lr', type=float, default=1.5e-3)
    ap.add_argument('--warmup', type=int, default=500)
    ap.add_argument('--min-lr-frac', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--window', type=int, default=16)
    ap.add_argument('--doc-len', type=int, default=5568, help='multiple of 64')
    ap.add_argument('--fourier-tau', type=float, default=512.0)
    ap.add_argument('--cycles', type=float, default=1.0)
    ap.add_argument('--rope-base', type=float, default=10000.0)
    ap.add_argument('--d-model', type=int, default=64)
    ap.add_argument('--n-heads', type=int, default=4)
    ap.add_argument('--n-layers', type=int, default=2)
    ap.add_argument('--d-ff', type=int, default=256)
    ap.add_argument('--eval-every', type=int, default=10000)
    ap.add_argument('--eval-batch', type=int, default=512)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    assert args.doc_len % K == 0

    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_f = open(out / 'train.log', 'w')

    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + '\n')
        log_f.flush()

    vocab = build_vocab('data/input.txt')
    V = len(vocab)
    x_train = encode(SPLIT / 'train_corpus.txt', vocab)
    chunk_paths = sorted((SPLIT / 'heldout_chunks').glob('chunk_*.txt'))
    chunks = [encode(p, vocab) for p in chunk_paths]
    T = args.window
    N = x_train.numel()

    tail = args.arm != 'none'
    states_tr = states_ev = lam_T = None
    atoms = None
    if tail:
        lam, r, omega, tau = make_atoms(args.arm, args.d_model // args.n_heads,
                                        args.rope_base, args.fourier_tau, args.cycles)
        atoms = {'omega': omega.tolist(), 'tau': tau.tolist()}
        t0 = time.time()
        states_tr = TailState(x_train, lam, r, V, dev, doc_blocks=args.doc_len // K)
        states_ev = [TailState(c, lam, r, V, dev) for c in chunks]
        lam_T = (lam[None, :] ** torch.arange(T, dtype=torch.float64)[:, None]
                 ).to(torch.complex64).to(dev)
        log(f'tail states built in {time.time() - t0:.1f}s: '
            f'train ckpts {tuple(states_tr.C.shape)}; tau {[round(t, 1) for t in tau.tolist()]}; '
            f'omega {[round(w, 4) for w in omega.tolist()]}')

    # training starts: the window and its targets stay inside one document
    s_all = torch.arange(0, N - T - 1)
    starts = s_all[(s_all % args.doc_len) + T + 1 <= args.doc_len].to(dev)

    model = TailRopeLM(V, args.d_model, args.n_heads, args.n_layers, args.d_ff, T,
                       args.rope_base, tail, dev)
    model.init_weights(args.seed)
    model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    x_dev = x_train.to(dev)
    chunks_dev = [c.to(dev) for c in chunks]
    rng = torch.Generator(device='cpu').manual_seed(args.seed + 1)
    ar = torch.arange(T + 1, device=dev)

    try:
        sha = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(['git', 'status', '--porcelain', '--', 'src/tools'],
                                    capture_output=True, text=True).stdout.strip())
    except Exception:
        sha, dirty = None, None
    (out / 'args.json').write_text(json.dumps({**vars(args), 'git_sha': sha, 'tools_dirty': dirty,
                                               'atoms': atoms, 'train_chars': N,
                                               'train_starts': int(starts.numel()),
                                               'params': sum(p.numel() for p in model.parameters())},
                                              indent=2) + '\n')
    log(f'arm={args.arm} seed={args.seed} steps={args.steps} batch={args.batch_size} lr={args.lr} '
        f'loss_positions={args.loss_positions} '
        f'params={sum(p.numel() for p in model.parameters())} starts={starts.numel()} device={dev}')

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * prog)))

    curve = []
    ema = None
    t_start = time.time()
    for step in range(args.steps):
        for gp in opt.param_groups:
            gp['lr'] = lr_at(step)
        idx = torch.randint(0, starts.numel(), (args.batch_size,), generator=rng).to(dev)
        s = starts[idx]
        xb = x_dev[s[:, None] + ar[None, :]]
        ctx = (states_tr.at(s), lam_T, model.tok.weight) if tail else None
        logits = model(xb[:, :T], ctx)
        if args.loss_positions == 'last':
            loss = F.cross_entropy(logits[:, -1], xb[:, T])
        else:
            loss = F.cross_entropy(logits.reshape(-1, V), xb[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        lv = loss.item()
        ema = lv if ema is None else 0.99 * ema + 0.01 * lv
        if (step + 1) % 1000 == 0:
            gate = ''
            if tail:
                gate = ' gate=' + ','.join(f'{g:+.3f}' for g in model.blocks[0].gate.tolist())
            log(f'step {step + 1} loss_ema={ema:.4f} lr={lr_at(step):.2e} '
                f'{time.time() - t_start:.0f}s{gate}')
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            total, by, n = evaluate(model, chunks_dev, states_ev, lam_T, T, args.eval_batch)
            curve.append({'step': step + 1, 'heldout_nll': total, 'heldout_ppl': math.exp(total),
                          'by_position': by, 'train_loss_ema': ema})
            log(f'EVAL step {step + 1} heldout_nll={total:.4f} ppl={math.exp(total):.4f} '
                f'targets={n} by_position=' + ' '.join(f'{k}:{v:.4f}' for k, v in by.items()))

    final = curve[-1]
    metrics = {
        'arm': args.arm, 'seed': args.seed, 'steps': args.steps,
        'eval': 'diagnostic: multi-chunk held-out (10 x 5576), each chunk its own document, '
                'targets from <=16 preceding chars plus tail; NOT lm-eval byte PPL',
        'heldout_nll': final['heldout_nll'], 'heldout_ppl': final['heldout_ppl'],
        'by_position': final['by_position'], 'curve': curve,
        'gate': model.blocks[0].gate.tolist() if tail else None,
        'train_seconds': time.time() - t_start,
    }
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    log(f'done: heldout_nll={final["heldout_nll"]:.4f} ppl={final["heldout_ppl"]:.4f}')


if __name__ == '__main__':
    main()
