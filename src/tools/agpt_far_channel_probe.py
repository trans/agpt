#!/usr/bin/env python3
"""Far-channel probe: an exact 16-character softmax window plus a far channel
with linear content and a smooth, ordinal positional kernel.

Question: can the text older than AGPT's 16-character window be read through a
channel that is blurry in position but exact and recurrent, and how much of the
gain of plain wider attention does it recover?

No trie. A plain minibatch-Adam transformer with the AGPT d64/L2 layout (pre-LN,
RoPE with interleaved pairs, ReLU FFN, untied output head). Layer 1 always
attends over offsets 0-15. The arms differ only in what layer 2 sees:

  w16      softmax over offsets 0-15 (the baseline; receptive field 30)
  exact    softmax over offsets 0-127 (the far channel's exact counterpart and
           upper bound; receptive field 142)
  decay    softmax over 0-15, plus a far channel over offsets 16-127
  cq       the same far channel with a different positional kernel

Spans compound across layers, so the receptive fields are 30 (w16) and 142
(the other three arms).

The far channel, per head, for the query at t and a key at p, Δ = t - p:

    a[t, p] = phi(qf_t) . phi(kf_p) * w_t(Δ),     16 <= Δ < 128
    far_t   = sum_p a[t, p] v_p / sum_p a[t, p]
    out_t   = window softmax output + gate_h * far_t

phi(z) = elu(z) + 1 (positive features), qf and kf are the far channel's own
projections, and v is the layer's value. The content term is linear, so it has
no position; position enters only through the kernel w_t, which the query steers:

  decay  w_t(Δ) = sum_a pi_{t,a} exp(-(Δ - 16) / tau_a), pi_t = softmax over 8
         atoms, tau = 2, 4, ..., 256. Completely monotone: recency only.
  cq     w_t(Δ) = |sum_j b_{t,j} lambda_j^(Δ - 16)|^2, b_t complex, 8 constant-Q
         atoms lambda_j = exp(-1 / P_j) exp(2 pi i / P_j), P = 4, 8, ..., 512.
         Nonnegative, and it can put a bump at a chosen distance, with a width
         that grows with the distance.

Both kernels are finite sums of exponentials in Δ, so the channel is exactly
recurrent: state per atom sum_p mu^(t-p) phi(kf_p) v_p^T and sum_p mu^(t-p)
phi(kf_p), with 8 atoms (decay) or 64 products lambda_j conj(lambda_l) (cq).
Training uses the parallel form, which is the same sum. A self-check at start-up
compares the two.

Training: blocks of 512 characters from the training split, every position
scored. Evaluation is DIAGNOSTIC, not lm-eval byte PPL: each target of the 10
held-out chunks is scored once, from a window that gives it at least 144
preceding characters when the chunk has them (receptive fields cap what each arm
uses). NLL is also reported by the target's position in its chunk: 1-30 (short
context for every arm), 31-142 (full for w16 only) and 143 to the end (full for
every arm; the main comparison).

Usage:
    python3 src/tools/agpt_far_channel_probe.py --arm cq --steps 20000 \\
        --out rnd/far-channel-probe/RUN_ID
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

SPLIT = Path('data/.splits/4fa9aec1db6b3aea')
BUCKETS = [(1, 31), (31, 143), (143, 10 ** 9)]
WINDOW, FAR_HI = 16, 128
RF = (WINDOW - 1) + (FAR_HI - 1)   # 142, receptive field of exact / decay / cq
CTX_MIN = RF + 2                    # eval: scored targets have >= 144 preceding chars


# ----------------------------------------------------------------- data

def build_vocab(path):
    text = Path(path).read_text(encoding='utf-8', errors='replace')
    return {c: i for i, c in enumerate(sorted(set(text)))}


def encode(path, vocab):
    text = Path(path).read_text(encoding='utf-8', errors='replace')
    return torch.tensor([vocab[c] for c in text], dtype=torch.long)


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


def span_mask(T, lo, hi, device):
    """(T, T) bool: key p visible to query t iff lo <= t - p < hi."""
    d = torch.arange(T, device=device)[:, None] - torch.arange(T, device=device)[None, :]
    return (d >= lo) & (d < hi)


class FarChannel(nn.Module):
    def __init__(self, D, H, kind, n_atoms=8):
        super().__init__()
        self.H, self.kind, self.A = H, kind, n_atoms
        self.qf, self.kf = nn.Linear(D, D), nn.Linear(D, D)
        n = FAR_HI - WINDOW
        d = torch.arange(n, dtype=torch.float64)
        if kind == 'decay':
            tau = 2.0 ** torch.arange(1, n_atoms + 1, dtype=torch.float64)       # 2 .. 256
            self.mix = nn.Linear(D, H * n_atoms)
            basis = torch.exp(-d[None, :] / tau[:, None])                        # (A, n)
            self.register_buffer('basis', basis.float(), persistent=False)
            self.atoms = {'tau': tau.tolist()}
        elif kind == 'cq':
            P = 2.0 ** torch.arange(2, n_atoms + 2, dtype=torch.float64)         # 4 .. 512
            lam = torch.polar(torch.exp(-1.0 / P), 2 * math.pi / P)
            self.coef = nn.Linear(D, H * n_atoms * 2)
            basis = lam[:, None] ** d[None, :]                                    # (J, n)
            self.register_buffer('basis', basis.to(torch.complex64), persistent=False)
            self.register_buffer('lam', lam, persistent=False)
            self.atoms = {'period': P.tolist(), 'decay_time': P.tolist()}
        else:
            raise ValueError(kind)
        self.gate = nn.Parameter(torch.ones(H))

    def profile(self, h):
        """Kernel over d = Δ - 16 for each query: (B, H, T, n)."""
        B, T, _ = h.shape
        if self.kind == 'decay':
            pi = torch.softmax(self.mix(h).view(B, T, self.H, self.A), -1).transpose(1, 2)
            return pi @ self.basis
        c = self.coef(h).view(B, T, self.H, self.A, 2).transpose(1, 2)
        b = torch.complex(c[..., 0].contiguous(), c[..., 1].contiguous())
        amp = b @ self.basis
        return amp.real ** 2 + amp.imag ** 2

    def weights(self, h, valid, didx):
        """a[b, h, t, p] before normalization, plus the content and kernel factors."""
        B, T, D = h.shape
        H, HD = self.H, D // self.H
        q = F.elu(self.qf(h).view(B, T, H, HD).transpose(1, 2)) + 1
        k = F.elu(self.kf(h).view(B, T, H, HD).transpose(1, 2)) + 1
        prof = self.profile(h)                                               # (B, H, T, n)
        w = torch.gather(prof, 3, didx[None, None].expand(B, H, T, T)) * valid
        return (q @ k.transpose(-1, -2)) * w, q, k, prof

    def forward(self, h, v, valid, didx):
        a = self.weights(h, valid, didx)[0]
        far = (a @ v) / (a.sum(-1, keepdim=True) + 1e-6)
        return self.gate.view(1, -1, 1, 1) * far


class Block(nn.Module):
    def __init__(self, D, H, Fd, far_kind):
        super().__init__()
        self.H = H
        self.ln1 = nn.LayerNorm(D)
        self.wq, self.wk = nn.Linear(D, D), nn.Linear(D, D)
        self.wv, self.wo = nn.Linear(D, D), nn.Linear(D, D)
        self.ln2 = nn.LayerNorm(D)
        self.l1, self.l2 = nn.Linear(D, Fd), nn.Linear(Fd, D)
        self.far = FarChannel(D, H, far_kind) if far_kind else None

    def forward(self, x, cos, sin, attn_mask, far_ctx):
        B, T, D = x.shape
        H, HD = self.H, D // self.H
        h = self.ln1(x)
        q = self.wq(h).view(B, T, H, HD).transpose(1, 2)
        k = self.wk(h).view(B, T, H, HD).transpose(1, 2)
        v = self.wv(h).view(B, T, H, HD).transpose(1, 2)
        att = F.scaled_dot_product_attention(
            rope_apply(q, cos, sin), rope_apply(k, cos, sin), v, attn_mask=attn_mask)
        if self.far is not None:
            att = att + self.far(h, v, *far_ctx)
        x = x + self.wo(att.transpose(1, 2).reshape(B, T, D))
        return x + self.l2(F.relu(self.l1(self.ln2(x))))


class FarLM(nn.Module):
    def __init__(self, V, D, H, L, Fd, T, base, arm, device):
        super().__init__()
        far_kind = arm if arm in ('decay', 'cq') else None
        self.spans = [WINDOW] * (L - 1) + [FAR_HI if arm == 'exact' else WINDOW]
        self.tok = nn.Embedding(V, D)
        self.blocks = nn.ModuleList([Block(D, H, Fd, far_kind if l == L - 1 else None)
                                     for l in range(L)])
        self.ln_f = nn.LayerNorm(D)
        self.head = nn.Linear(D, V)
        cos, sin = rope_cache(T, D // H, base, device)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

    def init_weights(self, seed):
        """N(0, 0.02) weights, zero biases, unit LayerNorm; far gates stay 1."""
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

    def forward(self, ids):
        T = ids.size(1)
        dev = ids.device
        x = self.tok(ids)
        cos, sin = self.cos[:T], self.sin[:T]
        valid = span_mask(T, WINDOW, FAR_HI, dev)
        d = torch.arange(T, device=dev)[:, None] - torch.arange(T, device=dev)[None, :]
        didx = (d - WINDOW).clamp(0, FAR_HI - WINDOW - 1)
        for b, span in zip(self.blocks, self.spans):
            x = b(x, cos, sin, span_mask(T, 0, span, dev), (valid, didx))
        return self.head(self.ln_f(x))


# --------------------------------------------------------------- checks

@torch.no_grad()
def check_recurrent(model, V, dev, T=200, seed=0):
    """Far channel, parallel form vs an explicit recurrence over atom states.
    Returns the max relative error of far_t (head 0, batch 0)."""
    far = model.blocks[-1].far
    if far is None:
        return None
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, V, (1, T), generator=g).to(dev)
    blk = model.blocks[-1]
    x = model.tok(ids)
    cos, sin = model.cos[:T], model.sin[:T]
    for b, span in zip(model.blocks[:-1], model.spans[:-1]):
        x = b(x, cos, sin, span_mask(T, 0, span, dev), (None, None))
    h = blk.ln1(x)
    D = h.size(-1)
    H, HD = blk.H, D // blk.H
    v = blk.wv(h).view(1, T, H, HD).transpose(1, 2)
    valid = span_mask(T, WINDOW, FAR_HI, dev)
    d = torch.arange(T, device=dev)[:, None] - torch.arange(T, device=dev)[None, :]
    didx = (d - WINDOW).clamp(0, FAR_HI - WINDOW - 1)
    a, q, k, _ = far.weights(h, valid, didx)
    par = ((a @ v) / (a.sum(-1, keepdim=True) + 1e-6))[0, 0].double()       # (T, HD)
    # recurrence, head 0, float64: atoms mu and per-query coefficients c_t(mu)
    q0, k0, v0 = q[0, 0].double(), k[0, 0].double(), v[0, 0].double()
    if far.kind == 'decay':
        mu = torch.exp(-1.0 / torch.tensor(far.atoms['tau'], dtype=torch.float64, device=dev)).to(torch.complex128)
        pi = torch.softmax(far.mix(h).view(1, T, H, far.A), -1)[0, :, 0].double()
        coef = pi.to(torch.complex128)                                          # (T, A)
    else:
        lam = far.lam.to(dev)
        mu = (lam[:, None] * lam.conj()[None, :]).reshape(-1)                  # (J*J,)
        c = far.coef(h).view(1, T, H, far.A, 2)[0, :, 0].double()
        b = torch.complex(c[..., 0], c[..., 1])                                # (T, J)
        coef = (b[:, :, None] * b.conj()[:, None, :]).reshape(T, -1)           # (T, J*J)
    n_hi = FAR_HI - WINDOW
    S = torch.zeros(mu.numel(), HD, HD, dtype=torch.complex128, device=dev)   # sum mu^d k v^T
    Z = torch.zeros(mu.numel(), HD, dtype=torch.complex128, device=dev)       # sum mu^d k
    worst = 0.0
    kc, vc = k0.to(torch.complex128), v0.to(torch.complex128)
    for t in range(T):
        if t >= WINDOW:                       # key p = t - 16 enters at d = 0
            p = t - WINDOW
            S += kc[p][None, :, None] * vc[p][None, None, :]
            Z += kc[p][None, :]
        if t >= FAR_HI:                       # key p = t - 128 leaves (it was at d = n_hi)
            p = t - FAR_HI
            S -= (mu ** n_hi)[:, None, None] * kc[p][None, :, None] * vc[p][None, None, :]
            Z -= (mu ** n_hi)[:, None] * kc[p][None, :]
        if t >= WINDOW:
            qc = q0[t].to(torch.complex128)
            num = torch.einsum('a,d,ade->e', coef[t], qc, S).real
            den = torch.einsum('a,d,ad->', coef[t], qc, Z).real
            rec = num / (den + 1e-6)
            worst = max(worst, float((rec - par[t]).abs().max() / par[t].abs().max().clamp_min(1e-12)))
        S *= mu[:, None, None]
        Z *= mu[:, None]
    return worst


# ----------------------------------------------------------------- eval

@torch.no_grad()
def evaluate(model, chunks, T, batch):
    """Each target x[i] scored once. Windows of T inputs start every T - CTX_MIN
    characters; a window scores its targets at positions >= CTX_MIN (the first
    window scores all of them), so every scored target has >= CTX_MIN preceding
    characters unless it is within CTX_MIN of its chunk start. A last window
    ending at the chunk end scores whatever the full windows left."""
    model.eval()
    nll = torch.zeros(len(BUCKETS), dtype=torch.float64)
    cnt = torch.zeros(len(BUCKETS), dtype=torch.float64)
    stride = T - CTX_MIN
    for x in chunks:
        n = x.numel()
        dev = x.device
        wins = [(s0, 0 if s0 == 0 else CTX_MIN) for s0 in range(0, n - T, stride)]
        last = wins[-1][0] + T                       # last target index scored so far
        if last < n - 1:
            s_f = n - 1 - T
            wins.append((s_f, last - s_f))
        for i in range(0, len(wins), batch):
            s = torch.tensor([w[0] for w in wins[i:i + batch]], device=dev)
            j0 = torch.tensor([w[1] for w in wins[i:i + batch]], device=dev)
            win = x[s[:, None] + torch.arange(T + 1, device=dev)[None, :]]
            lp = F.cross_entropy(model(win[:, :T]).transpose(1, 2), win[:, 1:], reduction='none')
            keep = torch.arange(T, device=dev)[None, :] >= j0[:, None]
            tpos = s[:, None] + 1 + torch.arange(T, device=dev)[None, :]
            tp, ls = tpos[keep], lp[keep]
            for bi, (lo, hi) in enumerate(BUCKETS):
                sel = (tp >= lo) & (tp < hi)
                nll[bi] += ls[sel].double().sum().cpu()
                cnt[bi] += sel.sum().cpu()
    model.train()
    total = float(nll.sum() / cnt.sum())
    by = {f'{lo}-{hi if hi < 10 ** 9 else "end"}': float(nll[b] / cnt[b]) for b, (lo, hi) in enumerate(BUCKETS)}
    return total, by, int(cnt.sum())


# ----------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--arm', choices=['w16', 'exact', 'decay', 'cq'], required=True)
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--batch-size', type=int, default=16, help='blocks of --block characters')
    ap.add_argument('--block', type=int, default=512)
    ap.add_argument('--lr', type=float, default=1.5e-3)
    ap.add_argument('--warmup', type=int, default=500)
    ap.add_argument('--min-lr-frac', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--rope-base', type=float, default=10000.0)
    ap.add_argument('--d-model', type=int, default=64)
    ap.add_argument('--n-heads', type=int, default=4)
    ap.add_argument('--n-layers', type=int, default=2)
    ap.add_argument('--d-ff', type=int, default=256)
    ap.add_argument('--eval-every', type=int, default=2000)
    ap.add_argument('--eval-batch', type=int, default=32)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
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
    chunks = [encode(p, vocab) for p in sorted((SPLIT / 'heldout_chunks').glob('chunk_*.txt'))]
    T = args.block
    N = x_train.numel()

    model = FarLM(V, args.d_model, args.n_heads, args.n_layers, args.d_ff, T,
                  args.rope_base, args.arm, dev)
    model.init_weights(args.seed)
    model.to(dev)
    n_params = sum(p.numel() for p in model.parameters())

    # checks before training: receptive fields, and the far channel's recurrent form
    with torch.no_grad():
        g = torch.Generator().manual_seed(1)
        ids = torch.randint(0, V, (2, T), generator=g).to(dev)
        t = T - 1
        base_out = model(ids)[:, t]

        def changed(lo, hi):  # change the inputs at offsets lo .. hi from t
            ids2 = ids.clone()
            ids2[:, t - hi:t - lo + 1] = (ids2[:, t - hi:t - lo + 1] + 1) % V
            return float((model(ids2)[:, t] - base_out).abs().max())
        span_beyond = changed(RF + 1, T - 1)          # must be 0 for every arm
        span_far = changed(2 * (WINDOW - 1) + 1, RF)  # must be 0 for w16 only
    rec_err = check_recurrent(model, V, dev)
    checks = {f'change_at_offsets_{RF + 1}_up_max_abs': span_beyond,
              f'change_at_offsets_{2 * (WINDOW - 1) + 1}_{RF}_max_abs': span_far,
              'far_recurrent_vs_parallel_max_rel_err': rec_err}
    log(f'checks: {checks}')

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
    far = model.blocks[-1].far
    (out / 'args.json').write_text(json.dumps({**vars(args), 'git_sha': sha, 'tools_dirty': dirty,
                                               'atoms': far.atoms if far is not None else None,
                                               'train_chars': N, 'params': n_params,
                                               'checks': checks}, indent=2) + '\n')
    log(f'arm={args.arm} seed={args.seed} steps={args.steps} batch={args.batch_size}x{T} lr={args.lr} '
        f'params={n_params} device={dev}')

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
        s = torch.randint(0, N - T - 1, (args.batch_size,), generator=rng).to(dev)
        xb = x_dev[s[:, None] + ar[None, :]]
        logits = model(xb[:, :T])
        loss = F.cross_entropy(logits.reshape(-1, V), xb[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        lv = loss.item()
        ema = lv if ema is None else 0.99 * ema + 0.01 * lv
        if (step + 1) % 1000 == 0:
            gate = '' if far is None else ' gate=' + ','.join(f'{g:+.3f}' for g in far.gate.tolist())
            log(f'step {step + 1} loss_ema={ema:.4f} lr={lr_at(step):.2e} {time.time() - t_start:.0f}s{gate}')
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            total, by, n = evaluate(model, chunks_dev, T, args.eval_batch)
            curve.append({'step': step + 1, 'heldout_nll': total, 'heldout_ppl': math.exp(total),
                          'by_context': by, 'train_loss_ema': ema})
            log(f'EVAL step {step + 1} heldout_nll={total:.4f} ppl={math.exp(total):.4f} targets={n} '
                f'by_context=' + ' '.join(f'{k}:{v:.4f}' for k, v in by.items()))
    final = curve[-1]
    metrics = {
        'arm': args.arm, 'seed': args.seed, 'steps': args.steps,
        'eval': 'diagnostic: multi-chunk held-out (10 x 5576), each target once, with >= 144 preceding chars '
                'where the chunk has them (receptive fields 30 / 142 cap use); by_context buckets are the target '
                'position in its chunk; NOT lm-eval byte PPL',
        'heldout_nll': final['heldout_nll'], 'heldout_ppl': final['heldout_ppl'],
        'by_context': final['by_context'], 'curve': curve,
        'gate': far.gate.tolist() if far is not None else None,
        'params': n_params, 'checks': checks,
        'train_seconds': time.time() - t_start,
    }
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    torch.save(model.state_dict(), out / 'model.pt')
    log(f'done: heldout_nll={final["heldout_nll"]:.4f} ppl={final["heldout_ppl"]:.4f}')


if __name__ == '__main__':
    main()
